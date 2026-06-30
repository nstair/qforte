"""Shared helpers for informal PySCF dump API checks."""

from __future__ import annotations

import contextlib
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
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
SANDBOX_DUMP_DIR = REPO_ROOT / "sandbox" / "pyscf_dump_qforte"
DUMP_DIR = THIS_DIR / "dumps"
LOG_DIR = THIS_DIR / "logs"

if str(SANDBOX_DUMP_DIR) not in sys.path:
    sys.path.insert(0, str(SANDBOX_DUMP_DIR))

from pyscf_dump_utils import (  # noqa: E402
    active_electron_count,
    active_spin_counts,
    closed_shell_hf_reference,
    geometry_to_pyscf_atom,
    write_pyscf_dump,
)


BASIS = "sto-3g"
CHARGE = 0
MULTIPLICITY = 1
SPIN = MULTIPLICITY - 1
SYMMETRY = "c1"

BEH2_DUMP_PATH = DUMP_DIR / "beh2_bent_sto3g_pyscf_dump.npz"
BENZENE_AVAS_DUMP_PATH = DUMP_DIR / "benzene_avas_c2pz_6e6o_sto3g_pyscf_dump.npz"

BENZENE_AVAS_LABELS = ["C 2pz"]
BENZENE_AVAS_THRESHOLD = 0.2
BENZENE_EXPECT_ACTIVE = (6, 6)

MOL_TOLERANCES = {
    "tensor": 1.0e-8,
    "abs_tensor": 1.0e-8,
    "energy": 1.0e-7,
    "vector": 1.0e-8,
    "integer": 0.0,
}

ALGORITHM_TOLERANCES = {
    "energy": 2.0e-7,
    "matrix": 1.0e-7,
    "time": 1.0e-10,
}


def get_qforte():
    try:
        import qforte as qf
    except ImportError as exc:
        raise ImportError(
            "qforte is required for molecule and algorithm parity checks. "
            "Run those scripts in a qforte-enabled environment."
        ) from exc
    return qf


def import_pyscf_for_generation(*names):
    try:
        import pyscf  # noqa: F401
        modules = []
        for name in names:
            if name == "avas":
                from pyscf.mcscf import avas

                modules.append(avas)
            else:
                modules.append(__import__("pyscf", fromlist=[name]).__dict__[name])
        return modules[0] if len(modules) == 1 else tuple(modules)
    except ImportError as exc:
        raise ImportError(
            "PySCF is required to generate dump files and to build direct PySCF "
            "molecules. The legacy sandbox used, for example, "
            "`conda run -n forte_pyscf_env python ...` for dump generation."
        ) from exc


def beh2_geometry():
    """A small bent C1 BeH2 geometry for direct tensor parity checks."""
    return [
        ("Be", (0.00000000, 0.00000000, 0.00000000)),
        ("H", (0.00000000, 0.00000000, 1.32640000)),
        ("H", (0.00000000, 1.15000000, -0.66320000)),
    ]


def benzene_geometry():
    """Planar benzene with C-C/C-H distances near standard STO-3G examples."""
    carbon_radius = 1.397
    hydrogen_radius = 2.479
    geom = []
    for i in range(6):
        theta = math.pi / 3.0 * i
        geom.append(("C", (carbon_radius * math.cos(theta), carbon_radius * math.sin(theta), 0.0)))
    for i in range(6):
        theta = math.pi / 3.0 * i
        geom.append(("H", (hydrogen_radius * math.cos(theta), hydrogen_radius * math.sin(theta), 0.0)))
    return geom


def print_section(title):
    print(f"\n==> {title} <==")


def print_geometry(label, geom, basis=BASIS):
    print_section("Molecule")
    print(f"  system: {label}")
    print(f"  basis:  {basis}")
    print("  geometry:")
    print(f"{'atom':>6s} {'x':>14s} {'y':>14s} {'z':>14s}")
    print("-" * 52)
    for atom, xyz in geom:
        print(f"{atom:>6s} {xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f}")


def run_with_log(log_path, func, *args, **kwargs):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                return func(*args, **kwargs)
            except Exception:
                traceback.print_exc(file=log)
                raise


def _build_pyscf_rhf_mol(geom, basis=BASIS, verbose=0):
    gto, scf = import_pyscf_for_generation("gto", "scf")

    mol = gto.Mole()
    mol.build(
        atom=geometry_to_pyscf_atom(geom),
        basis=basis,
        charge=CHARGE,
        spin=SPIN,
        unit="Angstrom",
        symmetry=False,
        verbose=verbose,
    )

    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-10
    mf.conv_tol_grad = 1.0e-8
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF RHF did not converge.")
    return mol, mf


def generate_beh2_dump(path=BEH2_DUMP_PATH, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        return {
            "system": "BeH2 bent STO-3G",
            "path": path,
            "status": "reused",
        }

    ao2mo, fci = import_pyscf_for_generation("ao2mo", "fci")

    geom = beh2_geometry()
    mol, mf = _build_pyscf_rhf_mol(geom, basis=BASIS)

    C = mf.mo_coeff
    hcore = mf.get_hcore()
    nmo = int(C.shape[1])
    mo_oeis = C.T @ hcore @ C
    mo_teis = ao2mo.kernel(mol, C, compact=False).reshape(nmo, nmo, nmo, nmo)

    cisolver = fci.FCI(mf)
    fci_energy, _ = cisolver.kernel()

    nalpha, nbeta = mol.nelec
    nelec = int(nalpha + nbeta)
    scalar_energy = float(mol.energy_nuc())
    metadata = {
        "system_label": "beh2_bent_sto3g",
        "source": "pyscf",
        "mol_geometry": geom,
        "basis": BASIS,
        "charge": CHARGE,
        "multiplicity": MULTIPLICITY,
        "num_active_orbitals": nmo,
        "num_active_electrons": nelec,
        "num_alpha": int(nalpha),
        "num_beta": int(nbeta),
        "hf_reference": closed_shell_hf_reference(nelec, nmo),
        "scalar_energy": scalar_energy,
        "nuclear_repulsion_energy": scalar_energy,
        "frozen_core_energy": 0.0,
        "hf_energy": float(mf.e_tot),
        "fci_energy_list": [float(np.real(fci_energy))],
        "hf_orbital_energies": [float(eps) for eps in mf.mo_energy],
        "point_group": "c1",
        "irreps": ["A"],
        "orb_irreps": ["A"] * nmo,
        "orb_irreps_to_int": [0] * nmo,
        "notes": "Full BeH2/STO-3G RHF molecular-orbital dump from a bent C1 geometry.",
    }

    write_pyscf_dump(path, mo_oeis, mo_teis, metadata)
    return {
        "system": "BeH2 bent STO-3G",
        "path": path,
        "status": "wrote",
        "hf_energy": float(mf.e_tot),
        "fci_energy": float(np.real(fci_energy)),
        "active": (nelec, nmo),
    }


def generate_benzene_avas_dump(path=BENZENE_AVAS_DUMP_PATH, overwrite=False, run_casci=True):
    path = Path(path)
    if path.exists() and not overwrite:
        return {
            "system": "benzene C 2pz AVAS STO-3G",
            "path": path,
            "status": "reused",
        }

    ao2mo, gto, mcscf, scf, avas = import_pyscf_for_generation(
        "ao2mo",
        "gto",
        "mcscf",
        "scf",
        "avas",
    )

    geom = benzene_geometry()
    mol = gto.Mole()
    mol.build(
        atom=geometry_to_pyscf_atom(geom),
        basis=BASIS,
        charge=CHARGE,
        spin=SPIN,
        unit="Angstrom",
        symmetry=False,
        verbose=0,
        max_memory=4000,
    )

    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-10
    mf.conv_tol_grad = 1.0e-8
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Benzene RHF did not converge.")

    avas_obj = avas.AVAS(mf, BENZENE_AVAS_LABELS, BENZENE_AVAS_THRESHOLD)
    avas_obj.kernel()
    ncas = int(avas_obj.ncas)
    nelecas = avas_obj.nelecas
    mo_coeff = avas_obj.mo_coeff
    nelec_active = active_electron_count(nelecas)
    nalpha, nbeta = active_spin_counts(nelecas)

    if (nelec_active, ncas) != BENZENE_EXPECT_ACTIVE:
        raise RuntimeError(
            f"Unexpected benzene AVAS active space: got ({nelec_active}e, {ncas}o), "
            f"expected ({BENZENE_EXPECT_ACTIVE[0]}e, {BENZENE_EXPECT_ACTIVE[1]}o)."
        )

    casci = mcscf.CASCI(mf, ncas, nelecas)
    casci.mo_coeff = mo_coeff
    h1eff, ecore = casci.get_h1eff()
    ncore = int(casci.ncore)
    C_cas = casci.mo_coeff[:, ncore:ncore + ncas]
    eri_cas = ao2mo.kernel(mol, C_cas, compact=False).reshape(ncas, ncas, ncas, ncas)

    fci_energy_list = []
    if run_casci:
        result = casci.kernel()
        fci_energy_list = [float(np.real(result[0]))]

    F = mf.get_fock()
    F_avas = mo_coeff.T @ F @ mo_coeff
    # PySCF AVAS returns orbitals ordered as [frozen | inactive/core | active | virtual].
    # The active block starts at the CASCI ncore boundary, not at the tail of mo_coeff.
    F_active = F_avas[ncore:ncore + ncas, ncore:ncore + ncas]
    mo_energy_active = np.linalg.eigvalsh(0.5 * (F_active + F_active.T))

    nuclear_repulsion = float(mol.energy_nuc())
    scalar_energy = float(ecore)
    metadata = {
        "system_label": "benzene_avas_c2pz_6e6o_sto3g",
        "source": "pyscf_avas",
        "mol_geometry": geom,
        "basis": BASIS,
        "charge": CHARGE,
        "multiplicity": MULTIPLICITY,
        "avas_labels": BENZENE_AVAS_LABELS,
        "avas_threshold": BENZENE_AVAS_THRESHOLD,
        "avas_minao": None,
        "avas_with_iao": None,
        "avas_canonicalize": None,
        "num_active_orbitals": ncas,
        "num_active_electrons": nelec_active,
        "num_alpha": nalpha,
        "num_beta": nbeta,
        "hf_reference": closed_shell_hf_reference(nelec_active, ncas),
        "scalar_energy": scalar_energy,
        "nuclear_repulsion_energy": nuclear_repulsion,
        "frozen_core_energy": float(ecore - nuclear_repulsion),
        "hf_energy": float(mf.e_tot),
        "fci_energy_list": fci_energy_list,
        "hf_orbital_energies": [float(eps) for eps in mo_energy_active],
        "point_group": "c1",
        "irreps": ["A"],
        "orb_irreps": ["A"] * ncas,
        "orb_irreps_to_int": [0] * ncas,
        "notes": "Unrelaxed PySCF AVAS active-space dump targeting benzene C 2pz.",
    }

    write_pyscf_dump(path, h1eff, eri_cas, metadata)
    return {
        "system": "benzene C 2pz AVAS STO-3G",
        "path": path,
        "status": "wrote",
        "hf_energy": float(mf.e_tot),
        "fci_energy": fci_energy_list[0] if fci_energy_list else None,
        "active": (nelec_active, ncas),
    }


def ensure_dumps(overwrite=False, run_benzene_casci=True):
    return [
        generate_beh2_dump(overwrite=overwrite),
        generate_benzene_avas_dump(overwrite=overwrite, run_casci=run_benzene_casci),
    ]


def build_molecule(system, build_type, run_fci=True):
    qf = get_qforte()
    common_kwargs = {
        "system_type": "molecule",
        "basis": BASIS,
        "symmetry": SYMMETRY,
        "multiplicity": MULTIPLICITY,
        "charge": CHARGE,
        "run_fci": run_fci,
        "run_ccsd": False,
        "store_mo_ints": True,
        "store_mo_ints_np": True,
        "build_df_ham": False,
    }

    if system == "beh2":
        if build_type == "pyscf_dump":
            generate_beh2_dump()
            return qf.system_factory(
                build_type="pyscf_dump",
                dump_file=str(BEH2_DUMP_PATH),
                **common_kwargs,
            )
        return qf.system_factory(
            build_type=build_type,
            mol_geometry=beh2_geometry(),
            **common_kwargs,
        )

    if system == "benzene":
        if build_type == "psi4":
            raise ValueError("The benzene AVAS dump test does not use the psi4 backend.")
        if build_type == "pyscf_dump":
            generate_benzene_avas_dump()
            return qf.system_factory(
                build_type="pyscf_dump",
                dump_file=str(BENZENE_AVAS_DUMP_PATH),
                **common_kwargs,
            )
        return qf.system_factory(
            build_type="pyscf",
            mol_geometry=benzene_geometry(),
            use_avas=True,
            avas_atoms_or_orbitals=BENZENE_AVAS_LABELS,
            avas_threshold=BENZENE_AVAS_THRESHOLD,
            **common_kwargs,
        )

    raise ValueError(f"Unknown system {system!r}.")


def maybe_build_molecule(system, build_type, log_root=LOG_DIR, run_fci=True):
    log_path = log_root / f"build_{system}_{build_type}.log"
    if str(build_type).startswith("pyscf"):
        # PySCF binds StreamObject.stdout to sys.stdout at import time.  If
        # the first import happens under run_with_log(), later direct builds
        # can inherit a closed redirected stream after the log context exits.
        import_pyscf_for_generation("gto", "scf", "ao2mo", "fci", "mcscf", "avas")
    try:
        mol = run_with_log(log_path, build_molecule, system, build_type, run_fci=run_fci)
    except Exception as exc:
        return None, log_path, f"{type(exc).__name__}: {exc}"
    return mol, log_path, None


def tensor_to_array(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    if hasattr(value, "read_data"):
        return np.asarray(value.read_data())
    if hasattr(value, "data"):
        return np.asarray(value.data)
    return np.asarray(value)


def scalar_energy(mol):
    return float(getattr(mol, "nuclear_repulsion_energy", 0.0)) + float(
        getattr(mol, "frozen_core_energy", 0.0)
    )


def fci_energy(mol):
    values = getattr(mol, "fci_energy_list", [])
    if values:
        return float(values[0])
    if hasattr(mol, "fci_energy"):
        return float(mol.fci_energy)
    return None


def max_abs_diff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return None, f"shape {a.shape} != {b.shape}"
    if a.size == 0:
        return 0.0, None
    return float(np.max(np.abs(a - b))), None


def compare_array(name, observed, expected, tol, *, abs_values=False):
    obs = tensor_to_array(observed)
    exp = tensor_to_array(expected)
    if obs is None or exp is None:
        return None, "attribute is unavailable", tol
    if abs_values:
        obs = np.abs(obs)
        exp = np.abs(exp)
    diff, reason = max_abs_diff(obs, exp)
    if reason is not None:
        return False, reason, tol
    return diff <= tol, diff, tol


def compare_array_norm(name, observed, expected, tol):
    obs = tensor_to_array(observed)
    exp = tensor_to_array(expected)
    if obs is None or exp is None:
        return None, "attribute is unavailable", tol
    if obs.shape != exp.shape:
        return False, f"shape {obs.shape} != {exp.shape}", tol
    diff = abs(float(np.linalg.norm(obs.ravel())) - float(np.linalg.norm(exp.ravel())))
    return diff <= tol, diff, tol


def compare_scalar(name, observed, expected, tol):
    if observed is None or expected is None:
        return None, "attribute is unavailable", tol
    diff = abs(float(observed) - float(expected))
    return diff <= tol, diff, tol


def compare_sequence(name, observed, expected, tol):
    if observed is None or expected is None:
        return None, "attribute is unavailable", tol
    observed = list(observed)
    expected = list(expected)
    if len(observed) != len(expected):
        return False, f"length {len(observed)} != {len(expected)}", tol
    if not observed:
        return True, 0.0, tol
    diffs = [abs(float(a) - float(b)) for a, b in zip(observed, expected)]
    diff = max(diffs)
    return diff <= tol, diff, tol


def compare_exact_sequence(name, observed, expected):
    if observed is None or expected is None:
        return None, "attribute is unavailable", 0
    observed = list(observed)
    expected = list(expected)
    if observed != expected:
        return False, f"{observed!r} != {expected!r}", 0
    return True, 0.0, 0


def print_check_table(title, rows, *, fail_on_required=True):
    print_section(title)
    header = f"{'check':38s} {'status':>8s} {'max |diff| / reason':>36s} {'tol':>12s} {'req':>5s}"
    print(header)
    print("-" * len(header))
    failures = []
    for label, status, diff, tol, required in rows:
        if status is None:
            status_text = "SKIP"
        else:
            status_text = "PASS" if status else ("FAIL" if required else "WARN")
        if isinstance(diff, str):
            diff_text = diff
        else:
            diff_text = f"{float(diff):14.6e}"
        tol_text = f"{float(tol):12.2e}" if isinstance(tol, float) else str(tol)
        req_text = "yes" if required else "no"
        print(f"{label:38s} {status_text:>8s} {diff_text:>36s} {tol_text:>12s} {req_text:>5s}")
        if status is False and required:
            failures.append((label, diff, tol))
    if failures and fail_on_required:
        details = "\n".join(f"  - {label}: diff/reason={diff}, tol={tol}" for label, diff, tol in failures)
        raise AssertionError(f"{title} failed:\n{details}")


def molecule_comparison_rows(
    reference,
    candidate,
    *,
    include_phase_relaxed=False,
    psi4_candidate=False,
    rotation_relaxed_two_body=False,
):
    rows = []
    tensor_tol = MOL_TOLERANCES["tensor"]
    abs_tol = MOL_TOLERANCES["abs_tensor"]
    vector_tol = MOL_TOLERANCES["vector"]
    energy_tol = MOL_TOLERANCES["energy"]

    for attr, label in [
        ("mo_oeis", "qforte Tensor one electron ints"),
        ("mo_teis", "qforte Tensor two electron ints"),
        ("mo_teis_einsum", "qforte Tensor two electron einsum"),
        ("mo_oeis_np", "NumPy spatial one electron ints"),
        ("mo_teis_np", "NumPy spatial two electron ints"),
    ]:
        is_two_body = attr in {"mo_teis", "mo_teis_einsum", "mo_teis_np"}
        required = not psi4_candidate and not (rotation_relaxed_two_body and is_two_body)
        rows.append(
            (
                label,
                *compare_array(label, getattr(candidate, attr, None), getattr(reference, attr, None), tensor_tol),
                required,
            )
        )
        if rotation_relaxed_two_body and is_two_body:
            rows.append(
                (
                    f"{label} Frobenius norm",
                    *compare_array_norm(
                        f"{label} Frobenius norm",
                        getattr(candidate, attr, None),
                        getattr(reference, attr, None),
                        vector_tol,
                    ),
                    True,
                )
            )
        if include_phase_relaxed:
            rows.append(
                (
                    f"{label} |abs|",
                    *compare_array(
                        f"{label} |abs|",
                        getattr(candidate, attr, None),
                        getattr(reference, attr, None),
                        abs_tol,
                        abs_values=True,
                    ),
                    True,
                )
            )

    rows.extend(
        [
            (
                "orbital energies",
                *compare_sequence(
                    "orbital energies",
                    getattr(candidate, "hf_orbital_energies", None),
                    getattr(reference, "hf_orbital_energies", None),
                    vector_tol,
                ),
                not psi4_candidate,
            ),
            (
                "zero/scalar body energy",
                *compare_scalar("zero/scalar body energy", scalar_energy(candidate), scalar_energy(reference), energy_tol),
                True,
            ),
            (
                "nuclear repulsion energy",
                *compare_scalar(
                    "nuclear repulsion energy",
                    getattr(candidate, "nuclear_repulsion_energy", None),
                    getattr(reference, "nuclear_repulsion_energy", None),
                    energy_tol,
                ),
                True,
            ),
            (
                "frozen core energy",
                *compare_scalar(
                    "frozen core energy",
                    getattr(candidate, "frozen_core_energy", None),
                    getattr(reference, "frozen_core_energy", None),
                    energy_tol,
                ),
                True,
            ),
            (
                "HF energy",
                *compare_scalar("HF energy", getattr(candidate, "hf_energy", None), getattr(reference, "hf_energy", None), energy_tol),
                True,
            ),
            (
                "FCI/CASCI energy",
                *compare_scalar("FCI/CASCI energy", fci_energy(candidate), fci_energy(reference), energy_tol),
                True,
            ),
            (
                "HF reference occupation",
                *compare_exact_sequence(
                    "HF reference occupation",
                    getattr(candidate, "hf_reference", None),
                    getattr(reference, "hf_reference", None),
                ),
                True,
            ),
            (
                "orbital irrep ids",
                *compare_exact_sequence(
                    "orbital irrep ids",
                    getattr(candidate, "orb_irreps_to_int", None),
                    getattr(reference, "orb_irreps_to_int", None),
                ),
                not psi4_candidate,
            ),
        ]
    )
    return rows


def build_srqk_algorithm(mol, trotter_order=1):
    qf = get_qforte()
    return qf.SRQK(
        mol,
        computer_type="fci",
        apply_ham_as_tensor=True,
        trotter_number=1,
        trotter_order=trotter_order,
        verbose=False,
        print_summary_file=False,
    )


def run_srqk_algorithm(mol):
    alg = build_srqk_algorithm(mol, trotter_order=1)
    alg.run(
        s=3,
        dt="lambda_inv",
        target_root=0,
        use_exact_evolution=False,
        diagonalize_each_step=False,
        low_memory_mat_formation=False,
        qk_tmax_type="manual",
        qk_time_grid="linear",
        qk_trotter_control="fixed",
        gev_stabilization_thresh=1.0e-10,
    )
    return {
        "energy": float(alg.get_ts_energy()),
        "Hbar": np.asarray(alg._Hbar, dtype=complex),
        "S": np.asarray(alg._S, dtype=complex),
        "qk_tmax": float(alg._qk_tmax),
        "macro_dt": np.asarray(alg._qk_macro_dt_list, dtype=float),
        "trotter_numbers": np.asarray(alg._qk_macro_trotter_number_list, dtype=int),
    }


def run_uccsd_vqe_algorithm(mol):
    qf = get_qforte()
    # Keep the symmetry metadata friendly to both Python and pybind vector
    # conversion paths.
    try:
        if getattr(mol, "orb_irreps_to_int", None) is not None:
            mol.orb_irreps_to_int = [int(value) for value in mol.orb_irreps_to_int]
    except Exception:
        pass

    alg = qf.UCCNVQE(
        mol,
        computer_type="fci",
        apply_ham_as_tensor=True,
        trotter_number=1,
        trotter_order=1,
        verbose=False,
        print_summary_file=False,
    )
    # A single MP2-seeded qforte BFGS step is too sensitive to benzene AVAS
    # active-space rotations across direct and dump-backed builds.  Start from
    # the shared HF reference and let a standard BFGS solve converge both
    # Hamiltonian representations to the same variational state.
    alg.run(
        opt_thresh=1.0e-6,
        opt_ftol=1.0e-10,
        opt_maxiter=50,
        pool_type="SD",
        optimizer="BFGS",
        use_analytic_grad=True,
        init_amps="zero",
        primary_pool_order="none",
        secondary_pool_order="shell",
        general_ex_pool_order="default",
    )
    return {
        "energy": float(alg.get_gs_energy()),
        "n_params": int(getattr(alg, "_n_classical_params", 0)),
        "n_cnot": int(getattr(alg, "_n_cnot", 0)),
        "grad_norm": float(getattr(alg, "_curr_grad_norm", np.nan)),
    }


def compare_algorithm_records(
    reference_name,
    reference,
    candidate_name,
    candidate,
    *,
    include_srqk_matrices=True,
    include_srqk_time_grid=True,
):
    rows = []
    rows.append(
        (
            f"{candidate_name} energy vs {reference_name}",
            *compare_scalar(
                "energy",
                candidate["energy"],
                reference["energy"],
                ALGORITHM_TOLERANCES["energy"],
            ),
            True,
        )
    )
    if include_srqk_matrices:
        for key, label in [("Hbar", "SRQK H matrix"), ("S", "SRQK S matrix")]:
            if key not in reference or key not in candidate:
                continue
            rows.append(
                (
                    f"{candidate_name} {label}",
                    *compare_array(
                        label,
                        candidate[key],
                        reference[key],
                        ALGORITHM_TOLERANCES["matrix"],
                    ),
                    True,
                )
            )
    if include_srqk_time_grid and "macro_dt" in reference and "macro_dt" in candidate:
        rows.append(
            (
                f"{candidate_name} SRQK macro dt list",
                *compare_sequence(
                    "macro dt",
                    candidate["macro_dt"],
                    reference["macro_dt"],
                    ALGORITHM_TOLERANCES["time"],
                ),
                True,
            )
        )
    return rows


def print_dump_summary(results):
    print_section("Generated dump summary")
    header = f"{'system':34s} {'status':>9s} {'active':>12s} {'HF energy':>16s} {'FCI/CASCI':>16s} {'path'}"
    print(header)
    print("-" * len(header))
    for item in results:
        active = item.get("active")
        active_text = "-" if active is None else f"({active[0]}e,{active[1]}o)"
        hf = item.get("hf_energy")
        fci = item.get("fci_energy")
        hf_text = "-" if hf is None else f"{hf:+16.10f}"
        fci_text = "-" if fci is None else f"{fci:+16.10f}"
        print(
            f"{item['system']:34s} {item['status']:>9s} {active_text:>12s} "
            f"{hf_text:>16s} {fci_text:>16s} {item['path']}"
        )
