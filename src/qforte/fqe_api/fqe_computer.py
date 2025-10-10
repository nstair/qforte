from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import numpy as np

from qforte import Tensor

import io
from contextlib import redirect_stdout

from qforte.helper.printing import matprint, tensor_str

# ---------------- Optional OpenFermion import (soft dependency) ----------------
try:
    from openfermion import FermionOperator  # type: ignore
except Exception:
    FermionOperator = Any  # permissive fallback for type hints / annotations

# --------------------- Optional FQE imports (soft dependency) ------------------
_FQE_AVAILABLE = True
try:
    # import os
    # os.environ["OMP_NUM_THREADS"] = "2"
    import fqe  # type: ignore
    from fqe import get_wavefunction  # type: ignore
    from fqe.wavefunction import Wavefunction  # type: ignore
    from fqe.openfermion_utils import integrals_to_fqe_restricted  # type: ignore
    from fqe.hamiltonians.sparse_hamiltonian import SparseHamiltonian
    from openfermion.transforms import normal_ordered as nod
    from openfermion.utils import hermitian_conjugated
except Exception:
    _FQE_AVAILABLE = False

    def _missing_fqe_error() -> None:
        raise ImportError(
            "FQE is not installed but is required for this operation.\n"
            "Install it with:  pip install fqe  (and optionally: pip install openfermion)\n"
            "Note: QForte itself imports fine without FQE; this error only appears when "
            "an FQE-backed method is invoked."
        )

    class _FQEPlaceholder:
        def __getattr__(self, _name: str) -> Any:
            _missing_fqe_error()

    # Placeholders so module import never crashes; using them raises only on use.
    fqe = _FQEPlaceholder()  # type: ignore

    def get_wavefunction(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore
        _missing_fqe_error()

    class Wavefunction:  # type: ignore
        def __init__(self, *_: Any, **__: Any) -> None:
            _missing_fqe_error()

    def integrals_to_fqe_restricted(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore
        _missing_fqe_error()

# -------------------- Static type checking-only imports (optional) ------------
if TYPE_CHECKING:
    import fqe as _fqe_t
    from fqe.wavefunction import Wavefunction as _Wavefunction_t
    from fqe.openfermion_utils import (
        integrals_to_fqe_restricted as _integrals_to_fqe_restricted_t,
    )

# ----------------------------- Convenience decorator --------------------------
def require_fqe(fn):
    """Guard FQE-dependent methods so they fail gracefully if FQE is missing."""
    def _wrapped(*args, **kwargs):
        if not _FQE_AVAILABLE:
            raise ImportError(
                "This method requires FQE. Install with: pip install fqe\n"
                "QForte runs without FQE; only FQE-backed features need it."
            )
        return fn(*args, **kwargs)
    return _wrapped


class FQEComputer:
    """
    Thin Python adapter that exposes an FCIComputer-like API backed by FQE’s
    particle-/spin-resolved Wavefunction simulator.

    Notes on conventions:
      - QForte FCIComputer ctor: (nel, sz, norb)
        FQE: fqe.get_wavefunction(nele, m_s, norb)
        Here we assume `sz` you pass is the integer m_s (S_z), i.e. (n_alpha - n_beta)/2.
      - Coeff storage: FQE partitions by sectors (n_alpha, n_beta).
        You can access dense sector arrays via wfn.get_coeff((na, nb)).
      - For 1e/2e Hamiltonians: prefer building FQE ops/hamiltonians from (h0, h1, h2)
        using fqe.openfermion_utils helpers, then use Wavefunction.apply / time_evolve.

    This file is a *template*: several methods are left as TODOs where QForte types
    (Tensor, SQOperator, TensorOperator, DFHamiltonian, etc.) must be converted to
    FQE/OpenFermion types.
    """

    # ---------- construction & basic state management ----------

    def __init__(self, nel: int, sz: int, norb: int) -> None:
        """
        Create an FQE-backed emulator for a given (nele, S_z, norb).
        """
        self.nel = int(nel)
        self.sz = int(sz)           # m_s (S_z), integer projection
        self.norb = int(norb)

        # Deduce (n_alpha, n_beta) consistent with (nel, m_s)
        # nel = na + nb, m_s = (na - nb)/2  => na = (nel + 2*m_s)/2
        na = (self.nel + 2 * self.sz) // 2
        nb = self.nel - na
        if na < 0 or nb < 0 or (na - nb) != 2 * self.sz:
            raise ValueError("Inconsistent (nel, sz) combination for α/β electrons.")

        # Build a number- and spin-conserving Wavefunction
        # FQE allocates all symmetry-consistent sectors you request here.
        # For now we request just the single sector (na, nb).
        self._wfn: Wavefunction = get_wavefunction(nele=self.nel, m_s=self.sz, norb=self.norb)

        # Initialize to Hartree–Fock (single determinant in sector)
        # FQE has utilities to get HF bitstrings; easiest path is zero + set basis coeff later.
        # As a sensible default, set all-zero then set HF amplitude=1.0 (see hartree_fock()).
        self.zero()
        # self.hartree_fock()

        # Timing log placeholder (to mirror FCIComputer.get_timings)
        self._timings: List[Tuple[str, float]] = []

    # Convenience: expose Wavefunction for advanced users
    @property
    def wfn(self) -> Wavefunction:
        return self._wfn


    def sector_key(self):
        """Default single sector defined at instantiation."""
        return (self.nel, self.sz)

    def str(self, print_data: bool = True, print_complex: bool = True):
        state_str = tensor_str(
            "FQEComputer State",
            self.get_state(),
            print_complex=print_complex
            )

        return state_str


    def __str__(self, print_data: bool = True, print_complex: bool = True):

        state_str = tensor_str(
            "FQEComputer State",
            self.get_state()
            )

        return state_str


    # def get_state(self) -> Dict[Tuple[int, int], np.ndarray]:
    def get_state(self) -> np.ndarray:
        """
        Return a dense coefficient array (copy).
        """
        return np.array(self._wfn.get_coeff(self.sector_key()), copy=False)

    def set_state(self, other):
        """
        Set this state to another np arrays data (works like move)
        """

        #pointer to this arrays data
        arr = self.get_state()

        # If 'other' is a NumPy array with the same shape and dtype, assign by reference:
        if arr.base is not other and arr.shape == other.shape and arr.dtype == other.dtype:
            arr[...] = other  # in-place assignment, minimal overhead
        else:
            raise ValueError("Input array must match shape and dtype of current state.")

       

    def get_state_deep(self) -> Dict[Tuple[int, int], np.ndarray]:
        # Same as get_state() here (but returns copies).
        return np.array(self._wfn.get_coeff(self.sector_key()), copy=True)

    # def set_state(self, sector_arrays: Dict[Tuple[int, int], np.ndarray]) -> None:
    # def set_state(self, sector_array: np.ndarray) -> None:
    #     """
    #     Overwrite the wavefunction coefficients for specified sectors.

    #     Parameters
    #     ----------
    #     sector_arrays : dict
    #         Keys are (n_alpha, n_beta), values are complex numpy arrays with shapes
    #         that match FQE’s internal sector layout for the given (norb, sector).
    #     """
    #     # FQE lets you set all sector arrays at once via set_wfn(raw_data=...).
    #     # It replaces only provided sectors; others stay unchanged.
    #     self._wfn.set_wfn(strategy="zeros", raw_data=sector_arrays)  # strategy ignored if raw_data supplied
    #     # optional: renormalize
    #     self._wfn.normalize()

    def zero(self) -> None:
        """Zero-out the sole sector (self.nel, self.sz)."""
        key = (self.nel, self.sz)
        arr = self._wfn.get_coeff(key)   # assumes your API exposes a writable NumPy array
        arr[...] = 0.0 + 0.0j            # in-place, keeps complex dtype

    def zero_state(self) -> None:
        self.zero()

    def hartree_fock(self) -> None:
        """
        Set the HF Slater determinant amplitude to 1.0 in the (na, nb) sector (others 0).
        """
        self.zero()
        
        key = (self.nel, self.sz)
        arr = self._wfn.get_coeff(key)  
        arr[0][0] = 1.0 + 0.0j

    # ---------- element access (optional; relies on sector indexing) ----------

    def set_element(self, idx, val) -> None:
        """
        Set a single CI coefficient in a sector by flat index.  (Adapter for
        FCIComputer.set_element which addressed a Tensor index list.)

        NOTE: If your FCIComputer addressed (alpha_state_idx, beta_state_idx) separately,
        map that to a flat index here, or replace this method to take a 2D index and reshape.
        """
        key = (self.nel, self.sz)
        arr = self._wfn.get_coeff(key)  
        arr[idx[0]][idx[1]] = val 

    def get_element(self, idx) -> complex:
        """
        Get a single CI coefficient in a sector by flat index.  (Adapter for
        FCIComputer.get_element which addressed a Tensor index list.)

        NOTE: If your FCIComputer addressed (alpha_state_idx, beta_state_idx) separately,
        map that to a flat index here, or replace this method to take a 2D index and reshape.
        """
        key = (self.nel, self.sz)
        arr = self._wfn.get_coeff(key)  
        return arr[idx[0]][idx[1]]

    def add_to_element(self, idx, val) -> None:
        key = (self.nel, self.sz)
        arr = self._wfn.get_coeff(key)
        arr.flat[idx] += val

    def get_tensor_diff(self, T):
        key = (self.nel, self.sz)
        arr = np.array(self._wfn.get_coeff(key), copy=True)

        C = Tensor(arr.shape, "Cfqe")
        C.fill_from_nparray(arr.ravel(), arr.shape)
        C.subtract(T)

        return C.norm()

        

    # ---------- scaling & simple ops ----------

    def scale(self, a: complex) -> None:
        """
        Multiply the wavefunction by a scalar.
        """
        self._wfn.scale(a)

    # ---------- expectation values ----------

    def get_exp_val(self, sqop) -> complex:
        """
        Expectation value ⟨ψ|O|ψ⟩ for an FQE operator or Hamiltonian.
        """

        key = (self.nel, self.sz)
        bra_arr = np.array(self._wfn.get_coeff(key), copy=True)

        self.apply_sqop(sqop)

        ket_arr = np.array(self._wfn.get_coeff(key), copy=False)

        val = np.vdot(bra_arr, ket_arr)

        self.set_state(bra_arr)

        return val

    def get_hf_dot(self) -> complex:
        """
        Get the value of <HF|psi>
        """
        key = (self.nel, self.sz)
        arr = np.array(self._wfn.get_coeff(key), copy=False)
        return arr[0][0]

    # ---------- tensor / Hamiltonian application & time evolution ----------

    # These map your Tensor/TensorOperator to FQE objects. Fill the TODOs with your conversions.

    def apply_tensor_spat_1bdy(self, h1e: np.ndarray, norb: int) -> None:
        """
        Apply a spatial 1-body operator to |ψ⟩.
        Expected h1e shape: (norb, norb) in physics (chemist) ordering.

        Implementation: build a restricted FQE operator from integrals and call wfn.apply().
        """
        raise NotImplementedError("1-body tensor application not yet implemented for FQEComputer.")
        fqe_op = integrals_to_fqe_restricted(h1e=h1e, h2e=None, constant=0.0)
        self._wfn = self._wfn.apply(fqe_op)  # FQE accepts Hamiltonian or FqeOperator

    def apply_tensor_spin_1bdy(self, h1e: np.ndarray, norb: int) -> None:
        """
        Same as apply_tensor_spat_1bdy but for explicit spin-orbital 1-body matrices
        if you choose to pass them; otherwise just call the spatial method.
        """
        raise NotImplementedError("1-body tensor application not yet implemented for FQEComputer.")
        self.apply_tensor_spat_1bdy(h1e, norb)

    def apply_tensor_spat_12bdy(
        self,
        h1e: np.ndarray,
        h2e: np.ndarray,
        h2e_einsum: Optional[np.ndarray],
        norb: int,
    ) -> None:
        """
        Apply (h1 + h2) to |ψ⟩. `h2e_einsum` is not needed for FQE; kept for API match.
        """
        raise NotImplementedError("1+2 body tensor application not yet implemented for FQEComputer.") 
        fqe_op = integrals_to_fqe_restricted(h1e=h1e, h2e=h2e, constant=0.0)
        self._wfn = self._wfn.apply(fqe_op)

    def apply_tensor_spat_012bdy(self, h0e: complex, h1e: np.ndarray, h2e: np.ndarray) -> None:

        # fqe.settings.use_accelerated_code = True

        # print(f"fqe accel code? : {fqe.settings.use_accelerated_code}")

        # build RH from integrals
        rh = integrals_to_fqe_restricted(h1e=h1e, h2e=h2e)  # no constant arg
        # rewrap with scalar part e_0
        rh = fqe.get_restricted_hamiltonian(rh.tensors(), e_0=h0e)
        # apply
        self._wfn = self._wfn.apply(rh)

    # Spin-resolved 1+2 body “apply” variants can simply call the spatial versions for now.
    def apply_tensor_spin_12bdy(self, h1e: np.ndarray, h2e: np.ndarray, norb: int) -> None:
        self.apply_tensor_spat_12bdy(h1e, h2e, None, norb)

    def apply_tensor_spin_012bdy(
        self, h0e: complex, h1e: np.ndarray, h2e: np.ndarray, norb: int
    ) -> None:
        raise NotImplementedError("spin orbital tensor appliction not yet implemented")
        self.apply_tensor_spat_012bdy(h0e, h1e, h2e, None, norb)

    # ---------- sqop appliction ---------

    def apply_sqop(self, sqop, antiherm=False) -> None:
        """
        Apply a single SQOperator to the wavefunction.
        TODO: Convert your SQOperator → OpenFermion FermionOperator → FQE operator/Hamiltonian.
        """
        of_op = self.convert_sqop_to_openfermion(sqop)

        if antiherm:
            of_op = 1.0j * of_op

        # note need to use fqe_apply to avoid hermitian check for general sqop
        # This function will also take any sqop thats a linear combination of more than
        # two sqops and make it a tensor and apply it more efficiently.
        # self._wfn = fqe_apply(of_op, self._wfn) 
        self._wfn = self._wfn.apply(of_op)

        if antiherm:
            self.scale(-1.0j)


    # ---------- measurement helpers ----------

    def direct_expectation_value(self, fqe_op_or_hamil: Any) -> List[float]:
        """
        Return [Re, Im] or other shape as list; adapter to FCIComputer’s real-valued list.
        """
        raise NotImplementedError("Expectation value not yet implemented for FQEComputer.")
        val = self._wfn.expectationValue(fqe_op_or_hamil)
        if np.ndim(val) == 0:
            return [float(np.real(val))]
        return np.array(val).ravel().real.tolist()
    

    def get_exp_val_tensor(self, 
        h0e: complex, 
        h1e: np.ndarray, 
        h2e: np.ndarray) -> None:

        key = (self.nel, self.sz)
        bra_arr = np.array(self._wfn.get_coeff(key), copy=True)

        self.apply_tensor_spat_012bdy(h0e, h1e, h2e)

        ket_arr = np.array(self._wfn.get_coeff(key), copy=False)

        val = np.vdot(bra_arr, ket_arr)

        self.set_state(bra_arr)

        return val

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
        """
        Use FQE polynomial expansion to approximate exp(-i t H) |ψ⟩.
        """
        # raise NotImplementedError("Evolve Tensor Taylor not yet implemented for FQEComputer.")
        # Build op/H and use apply_generated_unitary (choose algo="taylor" or "chebyshev").

        if(real_evolution):
            # Victor named this "real" evolution because the wfn remains real, confusingly it means
            # applying the imaginary time evolution operator, where False applies the real time 
            # evolution operator
            raise NotImplementedError("Real evolution e^beta H not yet implemented for FQEComputer.")

        # build RH from integrals
        rh = integrals_to_fqe_restricted(h1e=h1e, h2e=h2e)  # no constant arg
        # rewrap with scalar part e_0
        rh = fqe.get_restricted_hamiltonian(rh.tensors(), e_0=h0e)


        algo = "taylor"
        # algo = "chebyshev"  # alternative, need to specify spectral limits tho..
        acc = float(convergence_thresh) if convergence_thresh is not None else 1.0e-15
     

        self._wfn = self._wfn.apply_generated_unitary(
            float(evolution_time),
            algo,
            rh,     # FqeOperator is accepted
            acc,
            int(max_taylor_iter),
            None,
        )

    def apply_sqop_evolution(
        self,
        time: complex,
        sqop: Any,  # expected: your SQOperator or an OpenFermion FermionOperator
        antiherm: bool = False,
        adjoint: bool = False,
    ) -> None:
        """
        Evolve by exp(time * (sqop +- sqop^\dagger)) depending on flags.
        TODO: Convert your SQOperator → OpenFermion FermionOperator → FQE operator/Hamiltonian.
        """

        self.evolve_individual_sqop_term(
                time, 
                1.0, 
                sqop, 
                antiherm, 
                adjoint)

    # Pool/Trotter driver hooks (skeletons)
    def evolve_pool_trotter_basic(self, pool: Any, antiherm: bool = False, adjoint: bool = False) -> None:
        """
        Apply first-order Trotterized product of exp(θ_k O_k) with θ_k=1.0 by default.
        TODO: iterate terms, convert to FQE ops, call time_evolve for each.
        """

        self.evolve_pool_trotter(
            pool,
            1.0,
            1,
            1,
            antiherm,
            adjoint
        )


    def evolve_pool_trotter(
        self,
        pool,
        evolution_time: float,
        trotter_steps: int,
        trotter_order: int,
        antiherm: bool = True,   # default to anti-Hermitian evolution
        adjoint: bool = False,   # if True, use i(g + g†); else use (g - g†)
        ) -> None:
            
            dt = float(evolution_time) / max(1, int(trotter_steps))

            if trotter_order == 1:
                it = reversed(pool.terms()) if adjoint else pool.terms()
                for _ in range(int(trotter_steps)):
                    for coeff0, sq_term in it:
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
        if(len(sq_term.terms()) != 2):
            raise ValueError("Each SQ term must have exactly 2 subterms (cre/ann pairs).")

        if(len(sq_term.terms()[0][1]) == 0 and len(sq_term.terms()[1][1]) == 0):
            if adjoint:
                gphase = np.exp(+2j * coeff0 * sq_term.terms()[0][0]  * dt)
            else:
                gphase = np.exp(-2j * coeff0 * sq_term.terms()[0][0]  * dt)

            self._wfn.scale(gphase)
        
        else:

            K = self.convert_sqop_to_openfermion(sq_term)  # already g±g†
            K = nod(K)

            if antiherm:
                if adjoint:
                    H = (-1j * K) # case 4
                else:
                    H = (1j * K)  # case 3
            else:
                if adjoint:
                    H = -K         # case 2
                else:
                    H = K          # case 1

            theta = coeff0 * dt

             # --- sparse path for 2-term operators ---
            # Use FQE's SparseHamiltonian explicitly (avoids dense build).
            # time_evolve accepts any fqe Hamiltonian object.
            Hs = SparseHamiltonian(H, conserve_spin=True)
            self._wfn = self._wfn.time_evolve(theta, Hs, True)


    # ---------- diagonal & simple transforms ----------

    def evolve_diagonal_from_mat(self, V: np.ndarray, evolution_time: float) -> None:
        """
        Apply exp(-i t * diag(V)) assuming V encodes a diagonal number operator in your basis.
        TODO: map V to an FQE diagonal Hamiltonian and time_evolve.
        """
        raise NotImplementedError("Map your diagonal operator to FQE diagonal Hamiltonian, then time_evolve.")

    def apply_diagonal_from_mat(self, V: np.ndarray) -> None:
        """
        Apply a diagonal operator once: |ψ⟩ ← V |ψ⟩ (non-unitary apply).
        TODO: wrap as an FqeOperator and use Wavefunction.apply().
        """
        raise NotImplementedError

    def evolve_givens(self, U: np.ndarray, is_alfa: bool) -> None:
        """
        Apply an orbital rotation to α or β subspace.
        In FQE, Wavefunction.transform(rotation) applies a unitary orbital rotation.
        """
        # Single-spin rotations would require block-embedding into spinor space if needed.
        # For now, apply a full rotation U to all spin-orbitals (TODO: spin-selective path).
        raise NotImplementedError("Givens evolution not yet supported for FQEComputer.")
        _P, _L, _U, new_wfn = self._wfn.transform(U)
        self._wfn = new_wfn


    # ---------- timings ----------

    def get_timings(self) -> List[Tuple[str, float]]:
        raise NotImplementedError("internal timings not yet supported for FQEComputer")
        return list(self._timings)

    def clear_timings(self) -> None:
        self._timings.clear()

    # ---------- internal adapters (fill these out in your repo) ----------

    def convert_sqop_to_openfermion(self, sqop) -> FermionOperator:
        """
        Convert a QForte SQOperator (terms of (coeff, [cre], [ann])) to
        an OpenFermion FermionOperator. Creation operators are encoded
        as (i, 1) and annihilation as (i, 0).
        """
        op = FermionOperator()
        for coeff, cre_ops, ann_ops in sqop.terms():
            term = tuple([(i, 1) for i in cre_ops] + [(j, 0) for j in ann_ops])
            op += FermionOperator(term, complex(coeff))
        
        return op


