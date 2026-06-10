"""Shared helpers for informal Hamiltonian-application checks."""

from __future__ import annotations

import contextlib
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_LIBRARY", "serial")
os.environ.setdefault("KMP_AFFINITY", "disabled")

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
PYSCF_DUMP_TEST_DIR = REPO_ROOT / "informal_tests" / "pyscf_fci_dump_api"
REFERENCE_DIR = THIS_DIR / "reference_states"
LOG_DIR = THIS_DIR / "logs"
SYSTEMS = ("beh2", "benzene")
OPTIONAL_RUNTIME_SKIP_BACKENDS = {"fqe", "fci_gpu", "cusv"}

if str(PYSCF_DUMP_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(PYSCF_DUMP_TEST_DIR))

import pyscf_dump_informal_common as dump_common


@dataclass(frozen=True)
class HamCase:
    label: str
    backend: str
    route: str
    tol: float
    optional: bool = False


CASE_MATRIX = [
    HamCase("fci.tensor", "fci", "tensor", 1.0e-10, optional=False),
    HamCase("fci.sqop", "fci", "sqop", 1.0e-10, optional=False),
    HamCase("fci_gpu.tensor", "fci_gpu", "tensor", 1.0e-8, optional=True),
    HamCase("fci_gpu.sqop", "fci_gpu", "sqop", 1.0e-8, optional=True),
    HamCase("fock.qubit_operator", "fock", "qubit_operator", 1.0e-10, optional=False),
    HamCase("fqe.tensor", "fqe", "tensor", 1.0e-8, optional=True),
    HamCase("fqe.sqop", "fqe", "sqop", 1.0e-8, optional=True),
    HamCase("cusv.sqop", "cusv", "sqop", 1.0e-8, optional=True),
]


def get_qforte():
    return dump_common.get_qforte()


def system_label(system: str) -> str:
    if system == "beh2":
        return "bent BeH2"
    if system == "benzene":
        return "benzene C 2pz AVAS"
    raise ValueError(f"Unknown system {system!r}.")


def system_geometry(system: str):
    if system == "beh2":
        return dump_common.beh2_geometry()
    if system == "benzene":
        return dump_common.benzene_geometry()
    raise ValueError(f"Unknown system {system!r}.")


def reference_path(system: str) -> Path:
    return REFERENCE_DIR / f"{system}_fci_tensor_sigma_reference.npz"


def tensor_to_numpy(tensor) -> np.ndarray:
    return np.asarray(tensor.read_data(), dtype=np.complex128).reshape(tuple(tensor.shape()))


def numpy_to_tensor(array: np.ndarray, name: str = "tensor"):
    qf = get_qforte()
    array = np.asarray(array, dtype=np.complex128)
    tensor = qf.Tensor(list(array.shape), name)
    tensor.fill_from_nparray(array.ravel(), list(array.shape))
    return tensor


def run_with_log(log_path: Path, func, *args, **kwargs):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                return func(*args, **kwargs)
            except Exception:
                traceback.print_exc(file=log)
                raise


def build_dump_molecule(system: str, log_root: Path = LOG_DIR, run_fci: bool = False):
    mol, log_path, error = dump_common.maybe_build_molecule(
        system,
        "pyscf_dump",
        log_root=log_root,
        run_fci=run_fci,
    )
    if error:
        raise RuntimeError(f"Could not build {system} pyscf_dump molecule: {error}")
    return mol, log_path


def occupancy_counts(mol) -> dict[str, int]:
    reference = list(getattr(mol, "hf_reference", []))
    if not reference:
        raise ValueError("Molecule is missing hf_reference; cannot build backend states.")
    if len(reference) % 2 != 0:
        raise ValueError("Expected an even-length interleaved hf_reference.")

    norb = len(reference) // 2
    nalpha = int(sum(reference[0::2]))
    nbeta = int(sum(reference[1::2]))
    nel = nalpha + nbeta
    sz = nalpha - nbeta
    return {
        "nel": nel,
        "sz": sz,
        "norb": norb,
        "nalpha": nalpha,
        "nbeta": nbeta,
        "nqubit": 2 * norb,
    }


def zero_body_energy(mol) -> float:
    return dump_common.scalar_energy(mol)


def set_fock_hartree_fock(comp, mol) -> int:
    reference = list(getattr(mol, "hf_reference", []))
    hf_index = 0
    for qubit, occ in enumerate(reference):
        if occ:
            hf_index |= 1 << qubit

    coeffs = [0.0 + 0.0j] * (1 << len(reference))
    coeffs[hf_index] = 1.0 + 0.0j
    comp.set_coeff_vec(coeffs)
    return hf_index


def make_reference_fci_computer(mol, reference_state: np.ndarray):
    qf = get_qforte()
    counts = occupancy_counts(mol)
    comp = qf.FCIComputer(counts["nel"], counts["sz"], counts["norb"])
    comp.set_state(numpy_to_tensor(reference_state, name="reference_sigma"))
    return comp


def save_reference_state(system: str, mol, state: np.ndarray) -> Path:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    counts = occupancy_counts(mol)
    path = reference_path(system)
    np.savez_compressed(
        path,
        state=np.asarray(state, dtype=np.complex128),
        hf_reference=np.asarray(list(mol.hf_reference), dtype=np.int64),
        nel=np.int64(counts["nel"]),
        sz=np.int64(counts["sz"]),
        norb=np.int64(counts["norb"]),
        zero_body_energy=np.float64(zero_body_energy(mol)),
    )
    return path


def load_reference_state(system: str) -> np.ndarray:
    path = reference_path(system)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing saved Hamiltonian-application reference: {path}. "
            "Run run_ham_application.py --write-reference after reviewing the FCI tensor output."
        )
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["state"], dtype=np.complex128)


def backend_available(backend: str) -> tuple[bool, str | None]:
    qf = get_qforte()

    if backend == "fci":
        return hasattr(qf, "FCIComputer"), None if hasattr(qf, "FCIComputer") else "qforte.FCIComputer is not available"

    if backend == "fock":
        return hasattr(qf, "Computer"), None if hasattr(qf, "Computer") else "qforte.Computer is not available"

    if backend == "fqe":
        try:
            from qforte.fqe_api.fqe_computer import _FQE_AVAILABLE
        except Exception as exc:
            return False, f"could not inspect FQE availability: {exc}"
        if not _FQE_AVAILABLE:
            return False, "FQE/OpenFermion dependency is not installed"
        if not hasattr(qf, "FQEComputer"):
            return False, "qforte.FQEComputer is not available"
        return True, None

    if backend == "fci_gpu":
        if not hasattr(qf, "FCIComputerGPU"):
            return False, "qforte was not compiled with FCIComputerGPU support"
        if not hasattr(qf, "TensorGPU"):
            return False, "qforte was not compiled with CUDA TensorGPU support"
        return True, None

    if backend == "cusv":
        try:
            from qforte.cusv_api.cusv_computer import _CUPY_AVAILABLE, _CUSV_AVAILABLE
        except Exception as exc:
            return False, f"could not inspect CUSV availability: {exc}"
        if not _CUSV_AVAILABLE:
            return False, "cuQuantum/cuStateVec dependency is not installed"
        if not _CUPY_AVAILABLE:
            return False, "CuPy dependency is not installed"
        if not hasattr(qf, "CUSVComputer"):
            return False, "qforte.CUSVComputer is not available"
        return True, None

    raise ValueError(f"Unknown backend {backend!r}.")


def should_skip_runtime_error(backend: str, exc: Exception) -> bool:
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return True
    if backend not in OPTIONAL_RUNTIME_SKIP_BACKENDS:
        return False
    if isinstance(exc, NotImplementedError):
        return True

    message = str(exc).lower()
    skip_tokens = [
        "cuda",
        "gpu",
        "cuquantum",
        "custatevec",
        "cupy",
        "not implemented",
        "not yet implemented",
        "compiled",
    ]
    return any(token in message for token in skip_tokens)


def selected_systems(values: list[str] | None):
    if not values:
        return list(SYSTEMS)
    unknown = sorted(set(values) - set(SYSTEMS))
    if unknown:
        raise ValueError(f"Unknown system(s): {unknown}. Expected one of {SYSTEMS}.")
    return values


def selected_cases(labels: list[str] | None, backends: list[str] | None):
    cases = CASE_MATRIX
    if backends:
        wanted_backends = set(backends)
        known_backends = {case.backend for case in CASE_MATRIX}
        unknown = sorted(wanted_backends - known_backends)
        if unknown:
            raise ValueError(f"Unknown backend(s): {unknown}. Expected one of {sorted(known_backends)}.")
        cases = [case for case in cases if case.backend in wanted_backends]

    if labels:
        wanted = set(labels)
        cases = [case for case in cases if case.label in wanted]
        found = {case.label for case in cases}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"Unknown case label(s): {missing}")

    return cases


def print_system_header(system: str, mol, build_log: Path, reference_file: Path):
    print(f"\n==> system={system} <==")
    print(f"  label:          {system_label(system)}")
    print(f"  build log:      {build_log}")
    print(f"  reference file: {reference_file}")
    counts = occupancy_counts(mol)
    print(f"  active:         {counts['nel']} electrons in {counts['norb']} orbitals")
    print(f"  zero-body E:    {zero_body_energy(mol):+18.12f}")
    print("  geometry:")
    print(f"{'atom':>6s} {'x':>14s} {'y':>14s} {'z':>14s}")
    print("  " + "-" * 52)
    for atom, xyz in system_geometry(system):
        print(f"{atom:>6s} {xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f}")


def print_case_table(records: list[dict[str, Any]]):
    header = (
        f"{'case':22s} {'status':>8s} {'time (s)':>12s} "
        f"{'||dC||':>12s} {'tol':>12s} {'log':s}"
    )
    print(header)
    print("-" * len(header))
    for record in records:
        diff = "-" if record["diff_norm"] is None else f"{record['diff_norm']:.4e}"
        elapsed = "-" if record["elapsed_s"] is None else f"{record['elapsed_s']:.6f}"
        tol = "-" if record["tol"] is None else f"{record['tol']:.1e}"
        print(
            f"{record['case']:22s} {record['status']:>8s} {elapsed:>12s} "
            f"{diff:>12s} {tol:>12s} {record['log_path']}"
        )
        if record.get("reason"):
            print(f"  reason: {record['reason']}")
