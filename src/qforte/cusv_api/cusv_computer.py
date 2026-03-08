from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np
import math

from qforte import Tensor
from qforte import local_timer

import io
from contextlib import redirect_stdout

from qforte.helper.printing import matprint, tensor_str
from qforte import FCIGraph 

# ---------------- Optional OpenFermion import (soft dependency) ----------------
try:
    from openfermion import FermionOperator  # type: ignore
    from openfermion.transforms import normal_ordered as nod
    from openfermion.transforms import jordan_wigner
        
except Exception:
    FermionOperator = Any  # permissive fallback for type hints / annotations

# ---------------- Optional cuQuantum / cuStateVec imports (soft dependency) ----
# Mirrors the FQE soft-dep pattern in fqe_computer.py. :contentReference[oaicite:1]{index=1}
_CUSV_AVAILABLE = True
try:
    import cuquantum  # type: ignore
    # cuStateVec bindings live under cuquantum.custatevec in cuQuantum Python
    from cuquantum import custatevec  # type: ignore

    # Optional: most cusv workflows will likely want CuPy for device arrays.
    # Keep soft as well.
    try:
        import cupy as cp  # type: ignore
        _CUPY_AVAILABLE = True
    except Exception:
        cp = Any  # type: ignore
        _CUPY_AVAILABLE = False

except Exception:
    _CUSV_AVAILABLE = False

    def _missing_cusv_error() -> None:
        raise ImportError(
            "cuQuantum / cuStateVec is not installed but is required for this operation.\n"
            "Install with (conda-forge):\n"
            "  conda install -c conda-forge cuquantum-python custatevec cuda-version=13.0\n"
            "Note: QForte itself imports fine without cuQuantum; this error only appears when "
            "a cuStateVec-backed method is invoked."
        )

    class _CUSVPlaceholder:
        def __getattr__(self, _name: str) -> Any:
            _missing_cusv_error()

    cuquantum = _CUSVPlaceholder()  # type: ignore
    custatevec = _CUSVPlaceholder()  # type: ignore

    # CuPy placeholder too (so imports never crash)
    cp = _CUSVPlaceholder()  # type: ignore
    _CUPY_AVAILABLE = False

# -------------------- Static type checking-only imports (optional) ------------
if TYPE_CHECKING:
    import cuquantum as _cuquantum_t  # noqa: F401
    from cuquantum import custatevec as _custatevec_t  # noqa: F401
    try:
        import cupy as _cp_t  # noqa: F401
    except Exception:
        _cp_t = Any  # type: ignore

# ----------------------------- Convenience decorator --------------------------
def require_cusv(fn):
    """Guard cuStateVec-dependent methods so they fail gracefully if cuQuantum is missing."""
    def _wrapped(*args, **kwargs):
        if not _CUSV_AVAILABLE:
            raise ImportError(
                "This method requires cuQuantum/cuStateVec. Install with:\n"
                "  conda install -c conda-forge cuquantum-python custatevec cuda-version=13.0\n"
                "QForte runs without cuQuantum; only cuStateVec-backed features need it."
            )
        return fn(*args, **kwargs)
    return _wrapped


class CUSVComputer:
    """
    Thin Python adapter that exposes an FCIComputer-like API backed by cuStateVec.

    Key difference vs FQEComputer:
      - cuStateVec is NOT symmetry preserving: it stores the full 2^n complex statevector
        over n qubits. For electronic structure, n is typically 2*norb (spin-orbitals).
      - (nel, sz) are still accepted so this class can “look like” your other backends,
        but they will NOT reduce the stored dimension (you'll likely enforce symmetry
        constraints at the operator/application layer, or via your own indexing/masking).

    This file is a *template*: methods are stubs where QForte types (Tensor, SQOperator,
    DFHamiltonian, etc.) must be converted to cuStateVec-friendly representations.
    """

    # ---------- construction & basic state management ----------

    def __init__(
        self,
        nel: int,
        sz: int,
        norb: int,
        on_gpu: bool = True,
        device_id: int = 0,
        dtype: Any = np.complex128,
    ) -> None:
        """
        Persistent host+device storage with a single authoritative location flag.

        Buffers:
        - self._state_cpu : NumPy array (always allocated)
        - self._state_gpu : CuPy array (allocated if CuPy + CUDA are available)

        Authority flags:
        - self._on_cpu : CPU copy is authoritative (GPU stale)
        - self._on_gpu : GPU copy is authoritative (CPU stale)

        Transfers:
        - to_cpu() / to_gpu() copy INTO the preallocated buffer on the destination.
        - No reallocation, and the "other" location becomes stale (flag flips).
        """
        self.nel = int(nel)
        self.sz = int(sz)
        self.norb = int(norb)
        self.device_id = int(device_id)
        self.dtype = np.dtype(dtype)

        self.n_qubits = 2 * self.norb
        self.dim = 1 << self.n_qubits

        self.nalfa_el = int((nel + sz) / 2)
        self.nbeta_el = int(nel - self.nalfa_el)

        self.fci_graph = FCIGraph(self.nalfa_el, self.nbeta_el, self.norb)

        if self.dtype not in (
            np.dtype(np.float32), np.dtype(np.float64),
            np.dtype(np.complex64), np.dtype(np.complex128),
        ):
            raise TypeError("CUSVComputer supports float32/float64/complex64/complex128 only.")
        self.is_complex = np.issubdtype(self.dtype, np.complexfloating)

        self._hf_idx = self.get_hf_sv_index()

        # Always allocate host state
        self._state_cpu = np.zeros((self.dim,), dtype=self.dtype)

        # Try to allocate device state (optional)
        self._cp = None
        self._state_gpu = None
        self._sv_ptr = None
        try:
            import cupy as cp
            self._cp = cp
            cp.cuda.Device(self.device_id).use()
            self._state_gpu = cp.zeros((self.dim,), dtype=self.dtype)
            self._sv_ptr = int(self._state_gpu.data.ptr)
        except Exception:
            self._cp = None
            self._state_gpu = None
            self._sv_ptr = None

        # cuStateVec handle/workspace placeholders (you can create handle lazily later)
        self._handle = None
        self._workspace = None
        self._workspace_bytes = 0

        # Choose a single authoritative location.
        # Constructor zeros CPU and (if present) GPU, but we pick GPU as authoritative by default.
        if(on_gpu):
            self._on_cpu = False
            self._on_gpu = True
        else:
            self._on_cpu = True
            self._on_gpu = False

        self._timings: List[Tuple[str, float]] = []

        self._timer = local_timer()


    # ---------- location checks ----------

    def has_gpu(self) -> bool:
        return self._state_gpu is not None

    def on_cpu(self) -> bool:
        return bool(self._on_cpu)

    def on_gpu(self) -> bool:
        return bool(self._on_gpu)


    # ---------- "mark modified" helpers (call these after in-place writes) ----------

    def _mark_cpu(self) -> None:
        self._on_cpu = True
        self._on_gpu = False

    def _mark_gpu(self) -> None:
        self._on_gpu = True
        self._on_cpu = False


    # ---------- shuttling (no reallocation, just copy into persistent buffers) ----------

    def to_cpu(self) -> np.ndarray:
        """
        Ensure CPU buffer holds the authoritative data and return it.
        If GPU is authoritative, copy GPU -> CPU into the existing NumPy buffer.
        """
        if self._on_cpu:
            return self._state_cpu

        if not self.has_gpu():
            # If we *think* GPU is authoritative but there is no GPU buffer,
            # something went wrong.
            raise RuntimeError("State marked as on_gpu, but no GPU buffer exists.")

        # GPU -> CPU copy into preallocated host buffer
        self._state_gpu.get(out=self._state_cpu)
        self._mark_cpu()
        return self._state_cpu


    def to_gpu(self):
        """
        Ensure GPU buffer holds the authoritative data and return it.
        If CPU is authoritative, copy CPU -> GPU into the existing CuPy buffer.
        """
        if not self.has_gpu():
            raise RuntimeError("No GPU buffer available (CuPy/CUDA not available).")

        if self._on_gpu:
            # refresh pointer in case device context changed elsewhere
            self._sv_ptr = int(self._state_gpu.data.ptr)
            return self._state_gpu

        cp = self._cp
        cp.cuda.Device(self.device_id).use()

        # CPU -> GPU copy into preallocated device buffer
        cp.copyto(self._state_gpu, self._state_cpu)
        self._sv_ptr = int(self._state_gpu.data.ptr)
        self._mark_gpu()
        return self._state_gpu


    def device_ptr(self) -> int:
        """
        Return device pointer for cuStateVec calls.
        Ensures authoritative data is on GPU first.
        """
        self.to_gpu()
        return int(self._sv_ptr)


    def _dtype_to_cuda_data_type(self, np_dt: np.dtype) -> int:
        """Map NumPy dtype -> cuQuantum cudaDataType enum value."""
        dt = np.dtype(np_dt)
        cudaDataType = self._cudaDataType

        if dt == np.dtype(np.float32):
            return int(cudaDataType.CUDA_R_32F)
        if dt == np.dtype(np.float64):
            return int(cudaDataType.CUDA_R_64F)
        if dt == np.dtype(np.complex64):
            return int(cudaDataType.CUDA_C_32F)
        if dt == np.dtype(np.complex128):
            return int(cudaDataType.CUDA_C_64F)

        raise TypeError(f"Unsupported dtype for cuStateVec: {dt}")

        # ---------- small helpers / reporting ----------

    def str(self, print_data: bool = True, print_complex: bool = True):
        # For now, just mirror your FQEComputer approach; actual state retrieval
        # will depend on whether you store on device or host.
        return tensor_str("CUSVComputer State", self.get_state(), print_complex=print_complex)

    def __str__(self):
        return tensor_str("CUSVComputer State", self.get_state())

    # Keep for interface similarity; does NOT imply symmetry-preserving storage here.
    def sector_key(self):
        return (self.nel, self.sz)
    
    def on_gpu(self) -> bool:
        """True if the current statevector lives on the GPU (CuPy ndarray)."""
        try:
            import cupy as cp  # local import keeps module optional for CPU-only installs
        except Exception:
            return False
        return self._on_gpu


    def on_cpu(self) -> bool:
        """True if the current statevector lives on the CPU (NumPy ndarray)."""
        return self._on_cpu


    # YOU ARE HERE NEED TO FIX H-D data transfer...

    def to_cpu(self) -> None:
        """
        Ensure CPU buffer holds the authoritative state.

        - If already authoritative on CPU: no-op.
        - If authoritative on GPU: copy GPU -> CPU (into existing _state_cpu), then flip flags.

        Persistent buffers are never reallocated.
        """
        if self._on_cpu:
            return

        if not self.has_gpu():
            raise RuntimeError("State marked as on_gpu, but no GPU buffer exists.")

        # GPU -> CPU into preallocated host buffer
        self._state_gpu.get(out=self._state_cpu)
        self._mark_cpu()


    def to_gpu(self, device_id: Optional[int] = None) -> None:
        if device_id is not None:
            self.device_id = int(device_id)

        if not self.has_gpu():
            raise RuntimeError("No GPU buffer available (CuPy/CUDA not available).")

        cp = self._cp
        if cp is None:
            raise RuntimeError("GPU buffer exists but CuPy is not available (unexpected).")

        cp.cuda.Device(self.device_id).use()

        if self._on_gpu:
            self._sv_ptr = int(self._state_gpu.data.ptr)
            return

        # CPU -> GPU into preallocated device buffer (NO TEMP ALLOCATION)
        # Works with NumPy source
        self._state_gpu.set(self._state_cpu)

        self._sv_ptr = int(self._state_gpu.data.ptr)
        self._mark_gpu()


    def ensure_on_gpu(self) -> None:
        """
        Convenience guard for cusv methods.
        Call this at the top of any method that invokes cuStateVec.
        """
        if not self.on_gpu():
            raise RuntimeError(
                "CUSVComputer statevector is not on GPU. Call `to_gpu()` before using cuStateVec methods."
            )
        if getattr(self, "_sv_ptr", None) is None:
            # Shouldn't happen if on_gpu() is True, but keep it safe.
            self._sv_ptr = int(self._state.data.ptr)

    # ---------- state access ----------

    def get_state(self):
        """
        Return the authoritative dense coefficient array (no deep copy).

        - If self._on_cpu: returns the persistent NumPy host buffer.
        - If self._on_gpu: returns the persistent CuPy device buffer.

        Note: This does NOT synchronize. Use to_cpu()/to_gpu() if you want to
        force residency and copy the authoritative data to the other side.
        """
        if self._on_cpu:
            return self._state_cpu

        if self._on_gpu:
            if self._state_gpu is None:
                raise RuntimeError("State marked as on_gpu, but no GPU buffer exists.")
            return self._state_gpu

        # Should never happen if flags are maintained correctly
        raise RuntimeError("Invalid state location flags: neither _on_cpu nor _on_gpu is True.")


    def set_state(self, other) -> None:
        """
        Overwrite the authoritative state from `other`, without reallocating.

        Rules:
        - If self._on_cpu is True: `other` must be a NumPy array.
        - If self._on_gpu is True: `other` must be a CuPy array.
        - Shapes must match (dim,). Dtype must match exactly.

        This does NOT do any automatic host<->device transfers.
        Use to_cpu()/to_gpu() explicitly if you want to move residency.
        """
        # ---- basic shape/dtype checks (work for both numpy/cupy arrays) ----
        if getattr(other, "shape", None) != (self.dim,):
            raise ValueError(f"Expected state shape {(self.dim,)}, got {getattr(other, 'shape', None)}.")
        if np.dtype(getattr(other, "dtype", None)) != self.dtype:
            raise TypeError(f"Expected dtype {self.dtype}, got {getattr(other, 'dtype', None)}.")

        # ---- CPU authoritative path ----
        if self._on_cpu:
            if not isinstance(other, np.ndarray):
                raise TypeError(
                    "State is authoritative on CPU, but `other` is not a NumPy array. "
                    "Either pass a NumPy array or switch authority to GPU and pass a CuPy array."
                )
            np.copyto(self._state_cpu, other)
            self._mark_cpu()
            return

        # ---- GPU authoritative path ----
        if self._on_gpu:
            if self._state_gpu is None:
                raise RuntimeError("State marked as on_gpu, but no GPU buffer exists.")

            # Soft import Cupy for isinstance check
            if self._cp is None:
                raise RuntimeError("State marked as on_gpu, but CuPy is not available.")

            cp = self._cp
            if not isinstance(other, cp.ndarray):
                raise TypeError(
                    "State is authoritative on GPU, but `other` is not a CuPy array. "
                    "Either pass a CuPy array or switch authority to CPU and pass a NumPy array."
                )

            cp.cuda.Device(self.device_id).use()
            cp.copyto(self._state_gpu, other)
            self._sv_ptr = int(self._state_gpu.data.ptr)
            self._mark_gpu()
            return

        raise RuntimeError("Invalid state location flags: neither _on_cpu nor _on_gpu is True.")


    def get_state_deep(self):
        """
        Return a deep copy of the authoritative state *without changing residency*.

        - If self._on_cpu: returns a new NumPy array copy.
        - If self._on_gpu: returns a new CuPy array copy (still on GPU).

        Note: This does NOT synchronize CPU/GPU, and does NOT flip _on_cpu/_on_gpu.
        """
        if self._on_cpu:
            return np.array(self._state_cpu, copy=True)

        if self._on_gpu:
            if self._state_gpu is None:
                raise RuntimeError("State marked as on_gpu, but no GPU buffer exists.")
            return self._state_gpu.copy()

        raise RuntimeError("Invalid state location flags: neither _on_cpu nor _on_gpu is True.")


    # ---------- initialization ----------

    def zero(self) -> None:
        """Zero-out the full statevector on the currently-authoritative side."""
        if self._on_cpu:
            self._state_cpu.fill(0)
            self._mark_cpu()
            return

        if self._on_gpu:
            if self._state_gpu is None:
                raise RuntimeError("State marked as on_gpu, but no GPU buffer exists.")
            self._state_gpu.fill(0)
            self._sv_ptr = int(self._state_gpu.data.ptr)
            self._mark_gpu()
            return

        raise RuntimeError("Invalid state location flags: neither _on_cpu nor _on_gpu is True.")


    def zero_state(self) -> None:
        self.zero()

    def hartree_fock(self) -> None:
        """
        Set the HF Slater determinant amplitude to 1.0 and all others to 0.

        Behavior:
        - If the authoritative copy is on CPU: write into self._state_cpu.
        - If the authoritative copy is on GPU: write into self._state_gpu.
        """
        # hf_idx = self.get_hf_sv_index()
        hf_idx = self._hf_idx
        self.zero()

        if self._on_cpu:
            # zero + set HF on host
            self._state_cpu[hf_idx] = self.dtype.type(1.0)
            self._mark_cpu()
            return

        if self._on_gpu:
            if self._state_gpu is None:
                raise RuntimeError("State marked as on_gpu, but no GPU buffer exists.")
            # zero + set HF on device
            self._state_gpu[hf_idx] = self._state_gpu.dtype.type(1.0)
            self._sv_ptr = int(self._state_gpu.data.ptr)
            self._mark_gpu()
            return

        raise RuntimeError("Invalid state location flags: neither _on_cpu nor _on_gpu is True.")

    
    def get_hf_sv_index(self) -> int:
        """
        Gets the statevector *flat index* corresponding to the Hartree–Fock determinant
        in **abab...** (interleaved spin-orbital) ordering.

        Assumptions:
        - You use n_qubits = 2*norb with qubit layout:
                qubit 2*i     = orbital i, alpha  (a_i)
                qubit 2*i + 1 = orbital i, beta   (b_i)
            i = 0 is the lowest spatial orbital.
        - cuStateVec / computational basis indexing is little-endian:
            qubit 0 is the least-significant bit of the flat index.
        - The HF reference is the "lowest-energy" filling:
                occupy alpha for i=0..(n_alpha-1)
                occupy beta  for i=0..(n_beta-1)

        Returns:
        int: flat basis index in [0, 2**(2*norb)) whose bitstring matches HF.
        """
        # Infer (n_alpha, n_beta) from (nel, sz) assuming sz = n_alpha - n_beta.
        # This matches the common convention used in many QC codes.
        nel = int(self.nel)
        sz = int(self.sz)

        if (nel + sz) % 2 != 0:
            raise ValueError(f"Inconsistent (nel, sz)=({nel}, {sz}): (nel+sz) must be even.")
        n_alpha = (nel + sz) // 2
        n_beta = (nel - sz) // 2

        if n_alpha < 0 or n_beta < 0:
            raise ValueError(f"Inconsistent (nel, sz)=({nel}, {sz}): negative spin counts.")
        if n_alpha > self.norb or n_beta > self.norb:
            raise ValueError(
                f"HF fill exceeds norb: (n_alpha, n_beta)=({n_alpha},{n_beta}) with norb={self.norb}."
            )

        idx = 0

        # Set alpha occupations: qubit = 2*i
        for i in range(n_alpha):
            idx |= (1 << (2 * i))

        # Set beta occupations: qubit = 2*i + 1
        for i in range(n_beta):
            idx |= (1 << (2 * i + 1))

        return idx



    # ---------- element access ----------
    def _sv_index_from_IaIb(self, IaIb) -> int:
        """Internal: map (Ia,Ib) -> flat statevector index (abab... ordering)."""
        if IaIb is None or len(IaIb) != 2:
            raise ValueError("IaIb must be a length-2 container like [Ia, Ib].")
        
        # the probelm starts here, presently this class is not aware of the
        # FCIGraph data structure, the most obvious issues is that I think 
        # we are currently assuming the translation from Ia, Ib indicies (in the a-block/b-block)
        # matrix rep of the state-vector to the I single index in the vector rep of the 
        # 2^n state vector 

        Ia = int(IaIb[0])
        Ib = int(IaIb[1])

        # nel = int(self.nel)
        # sz = int(self.sz)
        # if (nel + sz) % 2 != 0:
        #     raise ValueError(f"Inconsistent (nel, sz)=({nel},{sz}): (nel+sz) must be even.")
        # n_alpha = (nel + sz) // 2
        # n_beta  = (nel - sz) // 2

        # nIa = math.comb(self.norb, n_alpha)
        # nIb = math.comb(self.norb, n_beta)
        # if Ia < 0 or Ia >= nIa:
        #     raise IndexError(f"Ia out of range: {Ia} (valid [0,{nIa}))")
        # if Ib < 0 or Ib >= nIb:
        #     raise IndexError(f"Ib out of range: {Ib} (valid [0,{nIb}))")

        # alpha_mask = self._unrank_comb_lex(self.norb, n_alpha, Ia)
        # beta_mask  = self._unrank_comb_lex(self.norb, n_beta,  Ib)

        alpha_mask = self.fci_graph.get_astr_at_idx(Ia)
        beta_mask = self.fci_graph.get_bstr_at_idx(Ib)

        # Try using FCI Graph class here to get masks...


        sv_idx = 0
        for i in range(self.norb):
            if (alpha_mask >> i) & 1:
                sv_idx |= (1 << (2 * i))
            if (beta_mask >> i) & 1:
                sv_idx |= (1 << (2 * i + 1))
        return sv_idx


    # --- helper: unrank combinations in lex order of occupied indices ----------------
    @staticmethod
    def _unrank_comb_lex(n: int, k: int, r: int) -> int:
        """
        Return a bitmask (over n bits) for the r-th k-combination in lex order
        over tuples of occupied indices (0-based).

        Lex order here means combinations are ordered by their occupied-index tuples,
        e.g. for n=5,k=3:
        (0,1,2),(0,1,3),(0,1,4),(0,2,3),(0,2,4),(0,3,4),(1,2,3),(1,2,4),(1,3,4),(2,3,4)
        """
        if k < 0 or k > n:
            raise ValueError(f"Invalid k={k} for n={n}")
        total = math.comb(n, k)
        if r < 0 or r >= total:
            raise IndexError(f"Combination rank r={r} out of range [0,{total}) for (n,k)=({n},{k})")

        mask = 0
        start = 0
        remaining = k
        rr = r

        while remaining > 0:
            # choose next occupied orbital 'a' from [start, n-remaining]
            for a in range(start, n - remaining + 1):
                count = math.comb(n - a - 1, remaining - 1)  # ways to choose rest if we pick 'a' now
                if rr < count:
                    mask |= (1 << a)
                    start = a + 1
                    remaining -= 1
                    break
                rr -= count

        return mask
    
    def set_element(self, idx, val) -> None:
        """
        Set a single amplitude by *flat* statevector index idx in [0, 2**n_qubits).

        Writes to:
        - self._state_cpu if self._on_cpu
        - self._state_gpu if self._on_gpu
        """
        idx = int(idx)
        if idx < 0 or idx >= self.dim:
            raise IndexError(f"idx out of range: {idx} (dim={self.dim})")

        if self._on_cpu:
            self._state_cpu[idx] = self.dtype.type(val)
            self._mark_cpu()
            return

        if self._on_gpu:
            if self._state_gpu is None:
                raise RuntimeError("State marked as on_gpu, but no GPU buffer exists.")
            self._state_gpu[idx] = self._state_gpu.dtype.type(val)
            self._sv_ptr = int(self._state_gpu.data.ptr)
            self._mark_gpu()
            return

        raise RuntimeError("Invalid state location flags: neither _on_cpu nor _on_gpu is True.")


    def set_element_from_IaIb(self, IaIb, val) -> None:
        """
        Set a single amplitude using FCIComputer-style indexing (Ia, Ib).
        """
        self.set_element(self._sv_index_from_IaIb(IaIb), val)


    def get_element(self, idx):
        """Get a single amplitude by flat statevector index."""
        idx = int(idx)
        if idx < 0 or idx >= self.dim:
            raise IndexError(f"idx out of range: {idx} (dim={self.dim})")

        if self._on_cpu:
            return self._state_cpu[idx]

        if self._on_gpu:
            if self._state_gpu is None:
                raise RuntimeError("State marked as on_gpu, but no GPU buffer exists.")
            # cupy scalar -> python scalar via .item()
            return self._state_gpu[idx].item()

        raise RuntimeError("Invalid state location flags: neither _on_cpu nor _on_gpu is True.")


    def get_element_from_IaIb(self, IaIb):
        """Get a single amplitude using FCIComputer-style indexing (Ia, Ib)."""
        sv_idx = self._sv_index_from_IaIb(IaIb)
        return self.get_element(sv_idx)


    def add_to_element(self, idx, val) -> None:
        """Add to a single amplitude by flat statevector index."""
        idx = int(idx)
        if idx < 0 or idx >= self.dim:
            raise IndexError(f"idx out of range: {idx} (dim={self.dim})")

        if self._on_cpu:
            self._state_cpu[idx] += self.dtype.type(val)
            self._mark_cpu()
            return

        if self._on_gpu:
            if self._state_gpu is None:
                raise RuntimeError("State marked as on_gpu, but no GPU buffer exists.")
            self._state_gpu[idx] += self._state_gpu.dtype.type(val)
            self._sv_ptr = int(self._state_gpu.data.ptr)
            self._mark_gpu()
            return

        raise RuntimeError("Invalid state location flags: neither _on_cpu nor _on_gpu is True.")


    def add_to_element_from_IaIb(self, IaIb, val) -> None:
        """Add to a single amplitude using FCIComputer-style indexing (Ia, Ib)."""
        sv_idx = self._sv_index_from_IaIb(IaIb)
        self.add_to_element(sv_idx, val)

    def get_fci_tensor_diff(self, T: Tensor, do_phase_compare: bool = True) -> float:
        """
        Same as before, but if do_phase_compare=True it accounts for the fermionic parity phase
        between ab-block determinant convention (FCI tensor) and abab convention (CUSV statevector),
        plus an automatically inferred *global* phase calibrated on the HF determinant.
        """
        if not self._on_cpu:
            raise RuntimeError("get_fci_tensor_diff requires the authoritative state to be on CPU.")

        nel = int(self.nel)
        sz = int(self.sz)
        if (nel + sz) % 2 != 0:
            raise ValueError(f"Inconsistent (nel, sz)=({nel},{sz}): (nel+sz) must be even.")
        n_alpha = (nel + sz) // 2
        n_beta  = (nel - sz) // 2
        if n_alpha < 0 or n_beta < 0 or n_alpha > self.norb or n_beta > self.norb:
            raise ValueError(
                f"Invalid spin counts from (nel,sz)=({nel},{sz}) with norb={self.norb}: "
                f"(n_alpha,n_beta)=({n_alpha},{n_beta})"
            )

        nIa = math.comb(self.norb, n_alpha)
        nIb = math.comb(self.norb, n_beta)

        if T.ndim() != 2:
            raise ValueError(f"Expected T.ndim()==2 for FCI tensor, got {T.ndim()}.")
        tshape = tuple(T.shape())
        if tshape != (nIa, nIb):
            raise ValueError(
                f"T has shape {tshape}, expected ({nIa},{nIb}) for (norb,nel,sz)=({self.norb},{nel},{sz})."
            )

        C = self._state_cpu
        full_abs2 = float(np.vdot(C, C).real)

        def phase_abblock_to_abab(alpha_mask: int, beta_mask: int) -> int:
            """
            Return ±1 such that:
                |A,B>_(ab-block) = phase * |A,B>_(abab)
            where A/B are occupied spatial orbitals for alpha/beta.
            """
            parity = 0
            bm = int(beta_mask)
            am = int(alpha_mask)
            while bm:
                lsb = bm & -bm
                b = lsb.bit_length() - 1
                parity ^= (int((am >> (b + 1)).bit_count()) & 1)  # #alpha > b  (mod 2)
                bm ^= lsb
            return -1 if parity else 1  # (-1)^parity

        def masks_to_sv_idx(alpha_mask: int, beta_mask: int) -> int:
            sv_idx = 0
            for i in range(self.norb):
                if (alpha_mask >> i) & 1:
                    sv_idx |= (1 << (2 * i))
                if (beta_mask >> i) & 1:
                    sv_idx |= (1 << (2 * i + 1))
            return sv_idx

        # --- optional: calibrate a single global phase using HF (Ia=0, Ib=0 in lex unranking) ---
        global_phase = 1.0 + 0.0j
        if do_phase_compare:
            hf_alpha_mask = self._unrank_comb_lex(self.norb, n_alpha, 0)
            hf_beta_mask  = self._unrank_comb_lex(self.norb, n_beta,  0)
            hf_sv_idx = masks_to_sv_idx(hf_alpha_mask, hf_beta_mask)

            c_hf = C[hf_sv_idx]
            t_hf = T.get([0, 0])
            det_phase_hf = phase_abblock_to_abab(hf_alpha_mask, hf_beta_mask)

            # If both are nonzero, solve c_hf ≈ global_phase * det_phase_hf * t_hf
            if abs(t_hf) > 0 and abs(c_hf) > 0:
                ratio = c_hf / (det_phase_hf * t_hf)
                # normalize to unit-modulus (handles ±1, ±i, etc.)
                global_phase = ratio / abs(ratio)

        sector_c_abs2 = 0.0
        sector_diff_abs2 = 0.0

        for Ia in range(nIa):
            alpha_mask = self._unrank_comb_lex(self.norb, n_alpha, Ia)
            for Ib in range(nIb):
                beta_mask = self._unrank_comb_lex(self.norb, n_beta, Ib)

                sv_idx = masks_to_sv_idx(alpha_mask, beta_mask)
                cval = C[sv_idx]
                tval = T.get([Ia, Ib])

                sector_c_abs2 += float((cval * np.conj(cval)).real)

                if do_phase_compare:
                    tval = global_phase * phase_abblock_to_abab(alpha_mask, beta_mask) * tval

                # ---- inline debug checks: opposite sign / conjugation ----
                # (paste right here, immediately after cval and tval are defined)

                atol = 1e-12
                rtol = 1e-10

                # skip near-zero comparisons to avoid noise
                # if (abs(cval) > atol) or (abs(tval) > atol):
                #     if not np.isclose(cval, tval, rtol=rtol, atol=atol):
                #         if np.isclose(cval, -tval, rtol=rtol, atol=atol):
                #             print(f"[opposite sign] Ia={Ia} Ib={Ib} sv_idx={sv_idx} cval={cval} tval={tval}")
                #         elif np.isclose(cval, np.conj(tval), rtol=rtol, atol=atol):
                #             print(f"[conjugate] Ia={Ia} Ib={Ib} sv_idx={sv_idx} cval={cval} tval={tval}")
                #         elif np.isclose(cval, -np.conj(tval), rtol=rtol, atol=atol):
                #             print(f"[-conjugate] Ia={Ia} Ib={Ib} sv_idx={sv_idx} cval={cval} tval={tval}")

                diff = cval - tval

                sector_diff_abs2 += np.real((diff * np.conj(diff)).real)

                # if(abs(np.real(diff)) > 1.0e-16 or abs(np.imag(diff)) > 1.0e-16):
                    
                #     print(f"    Ia: {Ia}  Ib: {Ib}")
                #     print(f"    CIaIb: {cval}")
                #     print(f"    tval:  {tval}")
                #     print(self._state_gpu.dtype)
                #     print(self._state_cpu.dtype)
                #     print(f"type(cval): {type(cval)})")
                #     print(f"type(tval): {type(tval)})")
                #     print(f"type(diff): {type(diff)})")
                #     print(f"type(norm): {type(sector_diff_abs2)})")

        outside_abs2 = full_abs2 - sector_c_abs2

        if np.abs(outside_abs2) > 1e-10:
            print(f"outside_abs2: {outside_abs2}")
            raise RuntimeError(f"Computed non trival state vectour outsize symmetry sector!")

        if outside_abs2 < 0.0 and outside_abs2 > -1e-10:
            outside_abs2 = 0.0
        if outside_abs2 < 0.0:
            raise RuntimeError(f"Computed negative outside-sector norm^2: {outside_abs2}")

        # print(f"outside_abs2:     {outside_abs2}")
        # print(f"sector_diff_abs2: {sector_diff_abs2}")

        # return np.sqrt(sector_diff_abs2 + outside_abs2)
        return np.sqrt(sector_diff_abs2)

    def get_fci_comp_state_diff(self, fci_comp, do_phase_compare=True) -> float:
        """
        Same as get_fci_tensor_diff, but accepts either an FCIComputer (CPU) or FCIComputerGPU.

        Behavior:
        - Ensures *this* CUSVComputer authoritative state is on CPU (required by get_fci_tensor_diff).
        - Extracts an Ia/Ib-indexed FCI Tensor on the host from fci_comp, regardless of backend.
        - Returns ||C - T|| with the same semantics as get_fci_tensor_diff().

        Expected interop:
        - FCIComputer:      T = fci_comp.get_state_deep()   (Tensor)
        - FCIComputerGPU:   fci_comp.to_cpu(); T = Tensor([Na,Nb]); fci_comp.copy_to_tensor_cpu(T)
        """
        # This routine relies on CPU-resident qubit statevector comparison.
        if not self._on_cpu:
            raise RuntimeError("get_fci_comp_state_diff requires authoritative CUSVComputer state on CPU.")

        # Determine (Na, Nb) from this CUSVComputer's sector definition.
        nel = int(self.nel)
        sz = int(self.sz)
        if (nel + sz) % 2 != 0:
            raise ValueError(f"Inconsistent (nel, sz)=({nel},{sz}): (nel+sz) must be even.")
        n_alpha = (nel + sz) // 2
        n_beta  = (nel - sz) // 2
        Na = math.comb(self.norb, n_alpha)
        Nb = math.comb(self.norb, n_beta)

        # --- Try CPU FCIComputer path first: get_state_deep() returns a Tensor ---
        T = None
        if hasattr(fci_comp, "get_state_deep"):
            T_try = fci_comp.get_state_deep()
            if isinstance(T_try, Tensor):
                T = T_try

        # --- Otherwise try GPU path: to_cpu() + copy_to_tensor_cpu(T) ---
        if T is None:
            if hasattr(fci_comp, "to_cpu") and hasattr(fci_comp, "copy_to_tensor_cpu"):
                # ensure fci_comp is on host
                fci_comp.cpu_error()
                T = Tensor([Na, Nb], "T")
                fci_comp.copy_to_tensor_cpu(T)

        if T is None:
            raise TypeError(
                "Unsupported fci_comp type: expected FCIComputer with get_state_deep(), "
                "or FCIComputerGPU with to_cpu() and copy_to_tensor_cpu(T)."
            )

        # Optional: sanity-check the tensor shape matches what we expect
        tshape = tuple(T.shape())
        if tshape != (Na, Nb):
            raise ValueError(f"FCI tensor shape mismatch: got {tshape}, expected ({Na},{Nb}).")

        return self.get_fci_tensor_diff(T, do_phase_compare=do_phase_compare)


    # ---------- scaling & simple ops ----------

    def scale(self, a: complex) -> None:
        """Multiply the wavefunction by a scalar (on the authoritative side)."""
        if self._on_cpu:
            self._state_cpu *= self.dtype.type(a)
            self._mark_cpu()
            return

        if self._on_gpu:
            if self._state_gpu is None:
                raise RuntimeError("State marked as on_gpu, but no GPU buffer exists.")
            self._state_gpu *= self._state_gpu.dtype.type(a)
            self._sv_ptr = int(self._state_gpu.data.ptr)
            self._mark_gpu()
            return

        raise RuntimeError("Invalid state location flags: neither _on_cpu nor _on_gpu is True.")


    # ---------- expectation values ----------

    def get_exp_val(self, sqop) -> complex:
        """
        Expectation value ⟨ψ|O|ψ⟩ for a QForte SQOperator (or convertible).
        """
        raise NotImplementedError("CUSVComputer.get_exp_val(): implement expectation evaluation.")

    def get_hf_dot(self) -> complex:
        """Return ⟨HF|ψ⟩ (requires same HF indexing as hartree_fock())."""
        raise NotImplementedError("CUSVComputer.get_hf_dot(): implement HF overlap.")

    # ---------- tensor / Hamiltonian application & time evolution ----------

    def apply_tensor_spat_1bdy(self, h1e: np.ndarray, norb: int) -> None:
        raise NotImplementedError("CUSVComputer.apply_tensor_spat_1bdy(): stub.")

    def apply_tensor_spin_1bdy(self, h1e: np.ndarray, norb: int) -> None:
        raise NotImplementedError("CUSVComputer.apply_tensor_spin_1bdy(): stub.")

    def apply_tensor_spat_12bdy(
        self,
        h1e: np.ndarray,
        h2e: np.ndarray,
        h2e_einsum: Optional[np.ndarray],
        norb: int,
    ) -> None:
        raise NotImplementedError("CUSVComputer.apply_tensor_spat_12bdy(): stub.")

    def apply_tensor_spat_012bdy(self, h0e: complex, h1e: np.ndarray, h2e: np.ndarray) -> None:
        raise NotImplementedError("CUSVComputer.apply_tensor_spat_012bdy(): stub.")

    def apply_tensor_spin_12bdy(self, h1e: np.ndarray, h2e: np.ndarray, norb: int) -> None:
        raise NotImplementedError("CUSVComputer.apply_tensor_spin_12bdy(): stub.")

    def apply_tensor_spin_012bdy(self, h0e: complex, h1e: np.ndarray, h2e: np.ndarray, norb: int) -> None:
        raise NotImplementedError("CUSVComputer.apply_tensor_spin_012bdy(): stub.")

    # ---------- SQOperator application ----------

    def apply_sqop(self, sqop, antiherm: bool = False) -> None:
        """
        Apply a QForte SQOperator to the state via:
        SQOperator -> OpenFermion FermionOperator -> OpenFermion QubitOperator (JW)
        -> (sum of Pauli strings) applied using cuStateVec.

        Semantics implemented here:
        If Q = sum_j c_j P_j  (P_j are Pauli strings),
        this computes |psi'> = Q |psi>.

        Notes:
        - This is correct but not optimized.
        - Multi-term sums allocate 2 extra full statevectors (temp + out).
        - antiherm is currently not used; sqop is assumed to already encode the desired operator.
        """
        # Ensure we are on GPU since we're calling cuStateVec
        if not self._on_gpu:
            raise RuntimeError("CUSVComputer must be on GPU for apply_sqop.")

        if self._state_gpu is None:
            raise RuntimeError("No GPU state buffer available.")
        if self._cp is None:
            raise RuntimeError("CuPy not available (unexpected if GPU buffer exists).")

        cp = self._cp
        cp.cuda.Device(self.device_id).use()

        # cuStateVec handle + dtype
        self._ensure_cusv_handle()
        cusv = self._cusv
        sv_dtype = self._cusv_sv_dtype()

        n_index_bits = np.uint32(self.n_qubits)
        theta = float(np.pi / 2.0)  # so exp(i theta P) = i P

        # 1) SQOperator -> FermionOperator
        # self._timer.acc_begin("apply_sqop: form of op from sqop")
        fop = self.convert_sqop_to_openfermion(sqop)
        # self._timer.acc_end("apply_sqop: form of op from sqop")

        # print(fop)

        # # 2) FermionOperator -> QubitOperator (Jordan-Wigner)
        # try:
        #     from openfermion.transforms import jordan_wigner
        # except Exception as e:
        #     raise RuntimeError("OpenFermion is required for apply_sqop() JW transform.") from e

        # self._timer.acc_begin("apply_sqop: jw transfrom")
        qop = jordan_wigner(fop)
        # self._timer.acc_end("apply_sqop: jw transform")

        # print(qop)

        # Normalize representation: dict of {term_tuple: coeff}
        terms_items = list(qop.terms.items())
        if len(terms_items) == 0:
            # operator is zero
            self._state_gpu.fill(0)
            self._sv_ptr = int(self._state_gpu.data.ptr)
            self._mark_gpu()
            return

        # ---- fast path: single term (in-place) ----
        if len(terms_items) == 1:
            term, coeff = terms_items[0]

            # coeff = complex(coeff)
            if np.abs(np.imag(coeff)) < 1.0e-14:
                coeff = np.real(coeff)
            else:
                coeff = complex(coeff)

            if len(term) == 0:
                # identity
                self._state_gpu *= self._state_gpu.dtype.type(coeff)
                self._sv_ptr = int(self._state_gpu.data.ptr)
                self._mark_gpu()
                return

            paulis, targets = self._of_qubitop_to_cusv_paulis(term)

            # print(f"\n cusv paulis {paulis}\n")
            # print(f"\n cusv targets {targets}\n")

            # self._timer.acc_begin("apply_sqop: apply pauli rotation")

            cusv.apply_pauli_rotation(
                self._handle,
                int(self._state_gpu.data.ptr),
                sv_dtype,
                n_index_bits,
                theta,
                paulis,
                targets,
                np.uint32(len(targets)),
                [],  # controls
                [],  # control_bit_values
                np.uint32(0),
            )

            # self._timer.acc_end("apply_sqop: apply pauli rotation")

            # exp(i*pi/2 P) = iP, so multiply by (-i) to get P
            self._state_gpu *= self._state_gpu.dtype.type(coeff * (-1j))

            self._sv_ptr = int(self._state_gpu.data.ptr)
            self._mark_gpu()
            return

        # ---- general path: accumulate out = sum c_j P_j |psi> ----
        psi = self._state_gpu
        tmp = cp.empty_like(psi)
        out = cp.zeros_like(psi)

        for term, coeff in terms_items:
            # coeff = complex(coeff)
            if np.abs(np.imag(coeff)) < 1.0e-14:
                coeff = np.real(coeff)
            else:
                coeff = complex(coeff)

            if len(term) == 0:
                # identity term
                out += psi * psi.dtype.type(coeff)
                continue

            # tmp = psi
            cp.copyto(tmp, psi)

            paulis, targets = self._of_qubitop_to_cusv_paulis(term)

            # print(f"\n cusv paulis {paulis}\n")
            # print(f"\n cusv targets {targets}\n")

            # self._timer.acc_begin("apply_sqop: apply pauli rotation")

            cusv.apply_pauli_rotation(
                self._handle,
                int(tmp.data.ptr),
                sv_dtype,
                n_index_bits,
                theta,
                paulis,
                targets,
                np.uint32(len(targets)),
                [],
                [],
                np.uint32(0),
            )

            # self._timer.acc_end("apply_sqop: apply pauli rotation")

            # self._timer.acc_begin("apply_sqop: axpby")
            # tmp now holds i P|psi>, so scale by coeff*(-i) to add coeff*P|psi>
            out += tmp * tmp.dtype.type(coeff * (-1j))
            # self._timer.acc_end("apply_sqop: axpby")

        # write result back into persistent GPU buffer
        cp.copyto(self._state_gpu, out)
        self._sv_ptr = int(self._state_gpu.data.ptr)
        self._mark_gpu()

    def apply_sqop_v2(self, sqop, antiherm: bool = False) -> None:
        """
        Apply a QForte SQOperator by:
        SQOperator -> OpenFermion FermionOperator -> OpenFermion QubitOperator (JW)
        then computing:
            |psi_out> = sum_k coeff_k * P_k |psi_in>

        Notes:
        - This does NOT apply exp(P) rotations. It applies the *linear combination*.
        - No 2^n x 2^n matrices are formed. Each Pauli string is applied as a
        sequence of 1-qubit 2x2 gate applications.
        - Requires GPU (cuStateVec acts on device pointers).
        - Memory: needs 2 extra full statevectors on GPU (tmp + out).
        """
        if not self.on_gpu():
            raise RuntimeError("apply_sqop_v2 requires the authoritative state on GPU. Call to_gpu() first.")

        # --- imports (kept local so the module can import on CPU-only machines) ---
        import numpy as np
        import cupy as cp
        from cuquantum import cudaDataType, ComputeType
        from cuquantum.bindings import custatevec as cusv
        from openfermion.transforms import jordan_wigner

        # self._ensure_cusv_handle2()
        self._ensure_cusv_handle()
        cusv = self._cusv

        # 1) SQOperator -> FermionOperator (you already have this helper)
        ferm_op = self.convert_sqop_to_openfermion(sqop)

        # 2) FermionOperator -> QubitOperator (sum of Pauli strings)
        qubit_op = jordan_wigner(ferm_op)

        # If SQOperator is "antiherm" in your sense, you may want an extra factor.
        # Leaving as-is for now (since your coefficients already encode your intent).
        _ = antiherm  # placeholder to avoid lint warnings

        psi = self._state_gpu
        if psi is None:
            raise RuntimeError("GPU state not allocated.")

        # dtype checks / cudaDataType mapping
        sv_dtype = self._cuda_data_type_from_dtype(psi.dtype, cudaDataType=cudaDataType)
        # For compute type, default is fine; matches cuQuantum docs :contentReference[oaicite:1]{index=1}
        compute_type = ComputeType.COMPUTE_DEFAULT

        # Allocate output + tmp buffers once (avoid per-term allocations)
        out = cp.zeros_like(psi)
        tmp = cp.empty_like(psi)

        # Main loop: out += coeff * (P |psi>)
        # OpenFermion: qubit_op.terms is a dict: {term_tuple: coeff}
        # term_tuple looks like ((q0,'X'), (q5,'Z'), ...)
        for term, coeff in qubit_op.terms.items():
            # c = complex(coeff)

            if np.abs(np.imag(coeff)) < 1.0e-14:
                c = np.real(coeff)
            else:
                c = complex(coeff)

            if c == 0.0:
                continue

            # tmp = psi
            cp.copyto(tmp, psi)

            # Apply each single-qubit Pauli in the string
            # Identity term is empty tuple () -> no ops
            for (q, pchar) in term:
                self._apply_single_qubit_pauli_inplace_cusv(
                    tmp, int(q), str(pchar),
                    cusv=cusv, cudaDataType=cudaDataType, ComputeType=ComputeType,
                    sv_dtype=sv_dtype, compute_type=compute_type,
                )

            # out += c * tmp
            # (cupy will handle complex scalars correctly)
            out += c * tmp

        # Commit result back into the authoritative GPU buffer
        cp.copyto(self._state_gpu, out)

        # Mark CPU stale, GPU fresh/authoritative 
        self._on_gpu = True
        self._on_cpu = False

    def get_exp_val(self, sqop) -> complex:
        """
        Return <psi| sqop |psi> for the current state.

        Strategy:
        - Default (works for any sqop): make a temp copy of |psi>, compute |phi> = sqop|psi>
            using apply_sqop_v2(), then return <psi|phi>.
        - This does NOT try to do cuStateVec "expectations_on_pauli_basis" yet, because:
            (i) sqop comes in as a fermionic operator,
            (ii) we'd need to JW->QubitOperator and then feed a batch of Pauli strings with
                correct coefficient handling. That's a good later optimization.

        Notes:
        - Requires authoritative state on GPU (we use cuStateVec path).
        - Allocates two full statevectors on GPU: tmp_in (copy of psi) and tmp_out (phi).
        """
        if not self.on_gpu():
            raise RuntimeError("get_sqop_exp_val requires authoritative state on GPU. Call to_gpu() first.")

        if self._state_gpu is None or self._cp is None:
            raise RuntimeError("GPU state not allocated.")

        cp = self._cp

        # Save |psi> into tmp_in
        psi = self._state_gpu
        tmp_in = cp.empty_like(psi)
        cp.copyto(tmp_in, psi)

        # Compute |phi> = sqop |psi> into persistent buffer via apply_sqop_v2,
        # but we don't want to destroy the original state, so:
        #   1) leave psi in self._state_gpu
        #   2) overwrite self._state_gpu with tmp_in, apply, then restore
        #
        # To avoid reworking apply_sqop_v2 to accept in/out buffers, we do a simple swap.

        # allocate tmp_out by reusing the persistent buffer after apply, then restore
        try:
            # overwrite state with tmp_in and apply in-place producing sum(P)|tmp_in>
            cp.copyto(self._state_gpu, tmp_in)
            self._on_gpu = True
            self._on_cpu = False

            self.apply_sqop(sqop)

            # now self._state_gpu holds |phi>
            tmp_out = cp.empty_like(psi)
            cp.copyto(tmp_out, self._state_gpu)

        finally:
            # restore original |psi> into persistent state buffer
            cp.copyto(self._state_gpu, psi if psi is not None else tmp_in)
            self._on_gpu = True
            self._on_cpu = False

        # <psi|phi> = vdot(psi, phi)
        # Note: tmp_in currently holds original |psi>
        expval = cp.vdot(tmp_in, tmp_out)

        # Return as python complex
        return complex(expval.get())
    
    def get_exp_val_opt(self, sqop) -> complex:
        """
        Optimized <psi|sqop|psi> using cuStateVec compute_expectations_on_pauli_basis.

        Pipeline:
        SQOperator -> OpenFermion FermionOperator -> (JW) QubitOperator (sum of Pauli strings)
        -> batch compute <psi|P_k|psi> for all Pauli strings via cuStateVec
        -> return sum_k coeff_k * <P_k>

        Notes:
        - Requires authoritative GPU state.
        - cuStateVec returns expectation values for Pauli strings as float64 on the host;
            we combine them with possibly-complex coefficients to form the final complex scalar.
        """
        if not self.on_gpu():
            raise RuntimeError("get_exp_val_opt requires authoritative state on GPU. Call to_gpu() first.")
        if self._state_gpu is None:
            raise RuntimeError("GPU state not allocated.")

        # import numpy as np
        from cuquantum import cudaDataType
        # from cuquantum.bindings import custatevec as cusv
        # from openfermion.transforms import jordan_wigner

        self._ensure_cusv_handle()
        cusv = self._cusv

        # SQOperator -> FermionOperator (assumed already in abab mode labeling)
        fop = self.convert_sqop_to_openfermion(sqop)

        # FermionOperator -> QubitOperator (sum of Pauli strings)
        qop = jordan_wigner(fop)


        items = list(qop.terms.items())
        if len(items) == 0:
            return 0.0 + 0.0j

        # Separate identity term (empty Pauli string)
        expval = 0.0 + 0.0j
        pauli_terms = []
        pauli_coeffs = []
        for term, coeff in items:
            c = complex(coeff)
            if len(term) == 0:
                expval += c  # <I> = 1
            else:
                pauli_terms.append(term)
                pauli_coeffs.append(c)

        n = len(pauli_terms)
        if n == 0:
            return complex(expval)

        # Build nested arrays for cuStateVec:
        #   pauli_operators_array: list[list[_Pauli]]
        #   basis_bits_array:      list[list[int32]]
        #   n_basis_bits_array:    list[uint32]
        pauli_operators_array = []
        basis_bits_array = []
        n_basis_bits_array = []

        for term in pauli_terms:
            # term: tuple((q, 'X'/'Y'/'Z'), ...), may be unsorted
            term_sorted = sorted(term, key=lambda x: x[0])

            bits = []
            paulis = []
            for (q, pchar) in term_sorted:
                q = int(q)
                bits.append(np.int32(q))
                if pchar == "X":
                    paulis.append(cusv.Pauli.X)
                elif pchar == "Y":
                    paulis.append(cusv.Pauli.Y)
                elif pchar == "Z":
                    paulis.append(cusv.Pauli.Z)
                else:
                    raise ValueError(f"Unexpected Pauli label {pchar!r}")

            basis_bits_array.append(bits)
            pauli_operators_array.append(paulis)
            n_basis_bits_array.append(np.uint32(len(bits)))

        # Host output buffer (cuStateVec stores expectations as float64 on host)
        expectations = np.empty(n, dtype=np.float64)

        # Call optimized batched expectation routine
        sv_dtype = self._cuda_data_type_from_dtype(self._state_gpu.dtype, cudaDataType=cudaDataType)

        cusv.compute_expectations_on_pauli_basis(
            self._handle,
            int(self._state_gpu.data.ptr),
            sv_dtype,
            np.uint32(self.n_qubits),
            int(expectations.ctypes.data),   # host pointer
            pauli_operators_array,           # nested sequence of _Pauli
            np.uint32(n),
            basis_bits_array,                # nested sequence of int32
            n_basis_bits_array,              # sequence of uint32
        )

        # Combine coeffs (complex) with expectations (real)
        # (Pauli expectation values should be real for Hermitian strings; we treat them as real floats.)
        for c, e in zip(pauli_coeffs, expectations):
            expval += c * float(e)

        return complex(expval)

    def get_exp_val_tensor(self, h0e: complex, h1e: np.ndarray, h2e: np.ndarray) -> complex:
        raise NotImplementedError("CUSVComputer.get_exp_val_tensor(): stub.")

    # ---------- time evolution ----------

    def evolve_tensor_taylor(
        self,
        h0e: complex,
        h1e: np.ndarray,
        h2e: np.ndarray,
        evolution_time: float,
        convergence_thresh: float,
        max_taylor_iter: int,
        real_evolution: bool,
    ) -> None:
        raise NotImplementedError("CUSVComputer.evolve_tensor_taylor(): stub.")

    def apply_sqop_evolution(
        self,
        time: complex,
        sqop: Any,
        antiherm: bool = False,
        adjoint: bool = False,
    ) -> None:
        # TODO: ask Nick about this
        self.evolve_individual_sqop_term(time, 1.0, sqop, antiherm, adjoint)

    def evolve_pool_trotter_basic(self, pool: Any, antiherm: bool = False, adjoint: bool = False) -> None:
        self.evolve_pool_trotter(pool, 1.0, 1, 1, antiherm, adjoint)

    def evolve_pool_trotter(
        self,
        pool,
        evolution_time: float,
        trotter_steps: int,
        trotter_order: int,
        antiherm: bool = True,
        adjoint: bool = False,
    ) -> None:
        
        dt = float(evolution_time) / max(1, int(trotter_steps))

        if trotter_order == 1:
            terms = list(pool.terms())
            base = list(reversed(terms)) if adjoint else terms
            for _ in range(int(trotter_steps)):
                for coeff0, sq_term in base:
                    self.evolve_individual_sqop_term(dt, coeff0, sq_term, antiherm, adjoint)

        elif trotter_order == 2:
            terms = list(pool.terms())
            base = list(reversed(terms)) if adjoint else terms

            for _ in range(int(trotter_steps)):
                # first half-step in base order
                for coeff0, sq_term in base:
                    self.evolve_individual_sqop_term(0.5 * dt, coeff0, sq_term, antiherm, adjoint)
                # second half-step in reverse order (Strang symmetry)
                for coeff0, sq_term in reversed(base):
                    self.evolve_individual_sqop_term(0.5 * dt, coeff0, sq_term, antiherm, adjoint)
        else:
            raise NotImplementedError("Only first- and second-order Trotter are supported.")
    

    def evolve_individual_sqop_term(self, dt, coeff0, sq_term, antiherm, adjoint):
        """
        Debuggable version of evolve_individual_sqop_term.

        Toggles (set True/False as needed):
        test_1_angle_debug:
            Print/check rotation-angle mapping details (factor/sign/imag part).
        test_2_check_commutation:
            Check whether all Pauli strings in this Kqb commute; print any violating pairs.
        test_3_order_shuffle:
            Apply commuting-product in a deterministic *sorted* order (or reverse) to test
            whether ordering affects results (reveals hidden non-commutation).
        test_4_theta_convention_debug:
            Print the exact theta/theta_eff conventions used (antiherm/adjoint effects).
        test_5_identity_phase_debug:
            Print identity-term phases and their magnitudes.
        """
        # ---------------- toggles ----------------
        test_1_angle_debug = False # lgtm
        test_2_check_commutation = False # lgtm
        test_3_order_shuffle = False        # if True, apply in reversed order (stress test)
        test_4_theta_convention_debug = False
        test_5_identity_phase_debug = False

        # ---------------- structural checks ----------------
        if len(sq_term.terms()) != 2:
            raise ValueError("Each SQ term must have exactly 2 subterms (cre/ann pairs).")

        # pure-number special case
        if (len(sq_term.terms()[0][1]) == 0 and len(sq_term.terms()[1][1]) == 0):
            if adjoint:
                gphase = np.exp(+2j * coeff0 * sq_term.terms()[0][0] * dt)
            else:
                gphase = np.exp(-2j * coeff0 * sq_term.terms()[0][0] * dt)
            self.scale(gphase)
            return

        # ---------------- build JW Pauli sum ----------------
        K = self.convert_sqop_to_openfermion(sq_term)
        K = nod(K)
        Kqb = jordan_wigner(K)

        # ---------------- ensure cusv availability + GPU residency ----------------
        self.to_gpu()
        if self._state_gpu is None:
            raise RuntimeError("GPU state not allocated.")
        if self._cp is None:
            raise RuntimeError("CuPy not available (unexpected if GPU buffer exists).")
        if not self.is_complex:
            raise RuntimeError(
                "evolve_individual_sqop_term requires complex dtype (complex64/complex128)."
            )

        self._ensure_cusv_handle()
        cusv = self._cusv
        sv_dtype = self._cusv_sv_dtype()
        n_index_bits = np.uint32(self.n_qubits)

        # ---------------- theta convention ----------------
        theta = complex(coeff0) * float(dt)
        if adjoint:
            theta = -theta

        # If antiherm=False, interpret as Hermitian evolution exp(-i * theta * H)
        theta_eff = theta if antiherm else (-1j * theta)

        if test_4_theta_convention_debug:
            print("\n[evolve_individual_sqop_term] theta convention debug")
            print(f"  dt={dt} coeff0={coeff0} antiherm={antiherm} adjoint={adjoint}")
            print(f"  theta     = {theta}")
            print(f"  theta_eff = {theta_eff}   (used as exp(theta_eff * sum c_l P_l))")

        # ---------------- helpers ----------------
        def _pauli_commute(termA, termB) -> bool:
            """Return True if Pauli strings termA and termB commute."""
            A = {int(q): str(p) for (q, p) in termA}
            B = {int(q): str(p) for (q, p) in termB}
            anti = 0
            for q in set(A) & set(B):
                pa, pb = A[q], B[q]
                if pa != pb:
                    anti ^= 1
            return (anti == 0)

        # ---------------- optional commutation check ----------------
        if test_2_check_commutation:
            terms_only = [t for t in Kqb.terms.keys() if len(t) > 0]
            bad = []
            for i in range(len(terms_only)):
                for j in range(i + 1, len(terms_only)):
                    if not _pauli_commute(terms_only[i], terms_only[j]):
                        bad.append((terms_only[i], terms_only[j]))
                        if len(bad) >= 5:
                            break
                if len(bad) >= 5:
                    break

            if bad:
                print("\n[commutation-check] NONCOMMUTING Pauli pairs detected inside K_mu:")
                for a, b in bad:
                    print("  A:", a)
                    print("  B:", b)
                print("  NOTE: If noncommuting, Π exp(theta_l P_l) is NOT equal to exp(sum theta_l P_l).")

        # ---------------- apply exact product (assumes commutation) ----------------
        items = list(Kqb.terms.items())

        # deterministic ordering to remove dict-order nondeterminism:
        # sort by (len(term), tuple(targets), tuple(paulis))
        def _term_sort_key(kv):
            term, coeff = kv
            if len(term) == 0:
                return (-1, (), ())
            term_sorted = tuple(sorted(term, key=lambda x: x[0]))
            targets = tuple(int(q) for (q, _) in term_sorted)
            paulis = tuple(str(p) for (_, p) in term_sorted)
            return (len(term), targets, paulis)

        items.sort(key=_term_sort_key)
        if test_3_order_shuffle:
            items = list(reversed(items))

        tol = 1.0e-10

        for term, c_l in items:
            c_l = complex(c_l)

            # identity term -> scalar phase
            if len(term) == 0:
                phase = np.exp(theta_eff * c_l)
                if test_5_identity_phase_debug:
                    print(f"[identity] c_l={c_l}  phase={phase}  |phase|={abs(phase)}")
                self.scale(phase)
                continue

            paulis, targets = self._of_qubitop_to_cusv_paulis(term)

            # map exp(theta_eff*c_l*P) to cusv exp(i*angle*P)
            angle_c = (-1j) * theta_eff * c_l

            if test_1_angle_debug:
                # Print the mapping; also show whether angle is nearly real

                if(np.abs(angle_c.imag) > 1.0e-14):
                    print("\n[angle-debug]")
                    print("  term:", term)
                    print("  c_l:", c_l)
                    print("  theta_eff:", theta_eff)
                    print("  angle_c:", angle_c, " (should be ~real)")
                    print("  angle_real:", angle_c.real, " angle_imag:", angle_c.imag)

            if abs(angle_c.imag) > tol:
                raise ValueError(
                    f"Non-real Pauli rotation angle encountered: angle={angle_c} "
                    f"(theta_eff={theta_eff}, coeff={c_l}, term={term})."
                )

            angle = float(angle_c.real)

            cusv.apply_pauli_rotation(
                self._handle,
                int(self._state_gpu.data.ptr),
                sv_dtype,
                n_index_bits,
                angle,
                paulis,
                targets,
                np.uint32(len(targets)),
                [],
                [],
                np.uint32(0),
            )

        # authoritative on GPU
        self._sv_ptr = int(self._state_gpu.data.ptr)
        self._mark_gpu()


    # ---------- diagonal & simple transforms ----------

    def evolve_diagonal_from_mat(self, V: np.ndarray, evolution_time: float) -> None:
        raise NotImplementedError("CUSVComputer.evolve_diagonal_from_mat(): stub.")

    def apply_diagonal_from_mat(self, V: np.ndarray) -> None:
        raise NotImplementedError("CUSVComputer.apply_diagonal_from_mat(): stub.")

    def evolve_givens(self, U: np.ndarray, is_alfa: bool) -> None:
        raise NotImplementedError("CUSVComputer.evolve_givens(): stub.")

    # ---------- timings ----------

    def get_timings(self) -> List[Tuple[str, float]]:
        raise NotImplementedError("CUSVComputer.get_timings(): stub (match your other backends).")

    def clear_timings(self) -> None:
        self._timings.clear()

    def print_timer(self) -> None:
        print(self._timer.acc_str_table())
        # print(self._timer.get_timings())
        # print(self._timer.get_acc_timings())

    # ---------- internal adapters (keep consistent with your FQE file) ----------

    def convert_sqop_to_openfermion(self, sqop) -> FermionOperator:
        """
        Convert a QForte SQOperator to OpenFermion FermionOperator.

        Keeping this identical to your FQE backend is handy because it lets you
        share downstream logic that starts from OF operators. :contentReference[oaicite:4]{index=4}
        """
        op = FermionOperator()
        for coeff, cre_ops, ann_ops in sqop.terms():
            term = tuple([(i, 1) for i in cre_ops] + [(j, 0) for j in ann_ops])
            op += FermionOperator(term, complex(coeff))

        nodop = nod(op)

        return nodop

    # ---------- future: handle management hooks ----------

    def _ensure_cusv_handle2(self) -> None:
        """Create a cuStateVec handle if needed."""
        if getattr(self, "_handle", None) is not None:
            return
        from cuquantum.bindings import custatevec as cusv
        self._handle = cusv.create()  # handle management API (bindings)


    def _cuda_data_type_from_dtype(self, dt, cudaDataType):
        """Map numpy/cupy dtype -> cuquantum.cudaDataType enum."""
        import numpy as np
        dt = np.dtype(dt)

        if dt == np.dtype(np.complex128):
            return int(cudaDataType.CUDA_C_64F)
        if dt == np.dtype(np.complex64):
            return int(cudaDataType.CUDA_C_32F)
        if dt == np.dtype(np.float64):
            return int(cudaDataType.CUDA_R_64F)
        if dt == np.dtype(np.float32):
            return int(cudaDataType.CUDA_R_32F)

        raise TypeError(f"Unsupported state dtype: {dt}. Use float32/float64/complex64/complex128.")

    def _ensure_cusv_handle(self):
        """Create a cuStateVec handle if we don't have one yet."""
        if self._handle is not None:
            return
        try:
            import cuquantum.custatevec as cusv
        except Exception:
            from cuquantum.bindings import custatevec as cusv  # fallback
        self._cusv = cusv
        self._handle = cusv.create()


    def _cusv_sv_dtype(self):
        """
        Map self.dtype -> cuQuantum cudaDataType (works for common cuQuantum Python builds).
        """
        # Try to use cuquantum.cudaDataType enum if present
        try:
            from cuquantum import cudaDataType
            dt = np.dtype(self.dtype)
            if dt == np.dtype(np.complex64):
                return cudaDataType.CUDA_C_32F
            if dt == np.dtype(np.complex128):
                return cudaDataType.CUDA_C_64F
            if dt == np.dtype(np.float32):
                return cudaDataType.CUDA_R_32F
            if dt == np.dtype(np.float64):
                return cudaDataType.CUDA_R_64F
        except Exception:
            pass

        # If enum import fails, let the caller error loudly (better than silent wrong type)
        raise RuntimeError("Could not map dtype to cuQuantum cudaDataType; check cuQuantum installation.")


    def _of_qubitop_to_cusv_paulis(self, term):
        """
        term: OpenFermion QubitOperator term key: tuple((q, 'X'/'Y'/'Z'), ...)
        returns: (paulis_list, targets_list) sorted by target index
        """
        cusv = self._cusv
        if len(term) == 0:
            return [], []

        items = sorted(term, key=lambda x: x[0])
        targets = [int(q) for (q, _) in items]
        paulis = []
        for (_, p) in items:
            if p == 'X':
                paulis.append(cusv.Pauli.X)
            elif p == 'Y':
                paulis.append(cusv.Pauli.Y)
            elif p == 'Z':
                paulis.append(cusv.Pauli.Z)
            else:
                raise ValueError(f"Unknown Pauli label: {p}")
        return paulis, targets
    
    def _apply_single_qubit_pauli_inplace_cusv(
        self,
        sv_dev,
        q: int,
        pchar: str,
        *,
        cusv,
        cudaDataType,
        ComputeType,
        sv_dtype: int,
        compute_type: int,
    ) -> None:
        """
        Apply a single-qubit Pauli (X/Y/Z) in-place on a device statevector using cusv.apply_matrix.

        We use cusv.apply_matrix (general gate) rather than apply_pauli_rotation since we are NOT
        exponentiating. :contentReference[oaicite:2]{index=2}
        """
        import numpy as np
        import cupy as cp

        if pchar not in ("X", "Y", "Z"):
            raise ValueError(f"Unsupported Pauli '{pchar}' (expected X/Y/Z).")

        # If the state is real, Y is not representable (introduces imaginary parts)
        if sv_dev.dtype in (cp.float32, cp.float64) and pchar == "Y":
            raise ValueError("Cannot apply Pauli-Y to a real-valued statevector (would become complex).")

        # Build (or reuse) a tiny 2x2 matrix on device
        key = (str(sv_dev.dtype), pchar)
        mats = getattr(self, "_pauli_mats_gpu", None)
        if mats is None:
            mats = {}
            self._pauli_mats_gpu = mats

        if key not in mats:
            # Create in host numpy first, then transfer to device
            if pchar == "X":
                m = np.array([[0, 1], [1, 0]], dtype=sv_dev.dtype)
            elif pchar == "Z":
                m = np.array([[1, 0], [0, -1]], dtype=sv_dev.dtype)
            else:  # "Y"
                m = np.array([[0, -1j], [1j, 0]], dtype=np.complex128 if sv_dev.dtype == cp.complex128 else np.complex64)
                m = m.astype(sv_dev.dtype, copy=False)

            mats[key] = cp.asarray(m)

        mat_dev = mats[key]

        # Workspace query (safe even if it returns 0)
        # apply_matrix_get_workspace_size signature :contentReference[oaicite:3]{index=3}
        ws_bytes = cusv.apply_matrix_get_workspace_size(
            self._handle,
            sv_dtype,
            int(self.n_qubits),
            int(mat_dev.data.ptr),
            int(sv_dtype),  # same dtype as state/matrix
            int(cusv.MatrixLayout.ROW),  # ROW/COL are defined :contentReference[oaicite:4]{index=4}
            0,  # adjoint
            1,  # n_targets
            0,  # n_controls
            int(compute_type),
        )

        ws_ptr = 0
        ws = None
        if ws_bytes > 0:
            ws = cp.empty((ws_bytes,), dtype=cp.uint8)
            ws_ptr = int(ws.data.ptr)

        # Apply the 2x2 matrix on target qubit q
        cusv.apply_matrix(
            self._handle,
            int(sv_dev.data.ptr),
            sv_dtype,
            int(self.n_qubits),
            int(mat_dev.data.ptr),
            int(sv_dtype),
            cusv.MatrixLayout.ROW,
            0,          # adjoint
            [q],        # targets (Python sequence allowed)
            1,          # n_targets
            [],         # controls
            [],         # control_bit_values
            0,          # n_controls
            compute_type,
            ws_ptr,
            int(ws_bytes),
        )

    
