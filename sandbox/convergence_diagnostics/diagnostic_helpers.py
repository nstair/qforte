"""Reusable helpers for VQE/tUCC convergence sandbox diagnostics.

This module is intentionally plain Python.  The goal is to collect enough
structured data to make convergence decisions without turning the sandbox into
a production testing framework.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import qforte as qf


PH_POOL_MAX_RANK = {
    "S": 1,
    "SD": 2,
    "SDT": 3,
    "SDTQ": 4,
    "SDTQP": 5,
    "SDTQPH": 6,
}

SUMMARY_FIELDS = [
    "case_id",
    "status",
    "molecule",
    "symmetry",
    "pool_type",
    "config_name",
    "optimizer",
    "init_amps",
    "primary_pool_order",
    "secondary_pool_order",
    "general_ex_pool_order",
    "use_hessian_diag",
    "use_newton_cg",
    "use_target_block",
    "target_block_reset_history",
    "newton_cg_reset_history",
    "maxiter",
    "pool_size",
    "n_params",
    "n_nonzero",
    "initial_energy",
    "first_reported_energy",
    "final_energy",
    "best_energy",
    "error_vs_fci",
    "error_vs_best_in_table",
    "final_grad_norm",
    "nit",
    "converged",
    "nfev",
    "njev",
    "nhev",
    "res_vec_evals",
    "res_m_evals",
    "ham_measurements",
    "pauli_term_measurements",
    "cnot",
    "target_block_attempts",
    "target_block_accepted",
    "newton_cg_attempts",
    "newton_cg_accepted",
    "escape_negcurv_attempts",
    "escape_negcurv_accepted",
    "largest_drop",
    "largest_drop_iter",
    "late_drop",
    "energy_iter_10",
    "energy_iter_25",
    "energy_iter_50",
    "energy_iter_75",
    "energy_iter_100",
    "energy_iter_150",
    "energy_iter_200",
    "runtime_s",
    "log_path",
    "error",
]


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def timestamp_label() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def slugify(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "case"


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, complex):
            if abs(value.imag) > 1.0e-12:
                return None
            value = value.real
        value = float(value)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def json_safe(value: Any) -> Any:
    """Convert common scientific/Python objects into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return safe_float(value)
    if isinstance(value, complex):
        real = safe_float(value.real)
        imag = safe_float(value.imag)
        return {"real": real, "imag": imag}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def metadata(mode: str) -> Dict[str, Any]:
    qforte_version = getattr(qf, "__version__", None)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root_from_here()),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = None

    return {
        "run_mode": mode,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "qforte_version": qforte_version,
        "cwd": str(repo_root_from_here()),
    }


def build_system(spec: Dict[str, Any]):
    """Build a QForte Molecule using a small, editable spec dictionary."""
    kwargs = {
        "system_type": "molecule",
        "build_type": spec.get("build_type", "psi4"),
        "basis": spec["basis"],
        "mol_geometry": spec["geometry"],
        "symmetry": spec.get("symmetry", "c1"),
        "multiplicity": spec.get("multiplicity", 1),
        "charge": spec.get("charge", 0),
        "num_frozen_docc": spec.get("num_frozen_docc", 0),
        "num_frozen_uocc": spec.get("num_frozen_uocc", 0),
        "run_mp2": spec.get("run_mp2", True),
        "run_ccsd": spec.get("run_ccsd", False),
        "run_fci": spec.get("run_fci", True),
        "nroots_fci": spec.get("nroots_fci", 4),
        "store_mo_ints": spec.get("store_mo_ints", True),
    }
    return qf.system_factory(**kwargs)


def molecule_summary(spec: Dict[str, Any], mol: Any) -> Dict[str, Any]:
    ref = list(getattr(mol, "hf_reference", []))
    nocc = int(sum(ref) // 2) if ref else None
    norb = int(len(ref) // 2) if ref else None
    nvir = int(norb - nocc) if nocc is not None and norb is not None else None
    return {
        "name": spec["name"],
        "basis": spec["basis"],
        "symmetry": spec.get("symmetry", "c1"),
        "geometry": spec["geometry"],
        "hf_energy": safe_float(getattr(mol, "hf_energy", None)),
        "mp2_energy": safe_float(getattr(mol, "mp2_energy", None)),
        "ccsd_energy": safe_float(getattr(mol, "ccsd_energy", None)),
        "fci_energy": safe_float(getattr(mol, "fci_energy", None)),
        "fci_energy_list": json_safe(getattr(mol, "fci_energy_list", None)),
        "n_spin_orbitals": len(ref) if ref else None,
        "n_qubits": len(ref) if ref else None,
        "n_electrons": int(sum(ref)) if ref else None,
        "nocc": nocc,
        "nvir": nvir,
        "point_group": json_safe(getattr(mol, "point_group", None)),
        "orb_irreps": json_safe(getattr(mol, "orb_irreps", None)),
        "orb_irreps_to_int": json_safe(getattr(mol, "orb_irreps_to_int", None)),
    }


def make_algorithm(mol: Any, verbose: bool = False):
    return qf.UCCNVQE(
        mol,
        computer_type="fci",
        apply_ham_as_tensor=True,
        verbose=verbose,
    )


def initialize_algorithm_for_energy(
    mol: Any,
    pool_type: str,
    init_amps: str,
    primary_pool_order: str,
    secondary_pool_order: str,
    general_ex_pool_order: str,
):
    """Create an algorithm object and initialize pool/amplitudes only.

    This mirrors the beginning of UCCNVQE.run() so we can report the starting
    energy for zero vs MP2 amplitudes without doing a full optimization.
    """
    alg = make_algorithm(mol, verbose=False)
    alg._opt_thresh = 1.0e-5
    alg._opt_ftol = 1.0e-8
    alg._opt_maxiter = 0
    alg._use_analytic_grad = True
    alg._optimizer = "lbfgs_qf"
    alg._pool_type = pool_type
    alg._noise_factor = 0.0
    alg._init_amps = init_amps
    alg._primary_pool_order = primary_pool_order
    alg._secondary_pool_order = secondary_pool_order
    alg._general_ex_pool_order = general_ex_pool_order
    alg._tops = []
    alg._tamps = []
    alg._conmutator_pool = []
    alg._converged = 0
    alg._n_classical_params = 0
    alg._n_cnot = 0
    alg._n_pauli_trm_measures = 0
    alg._res_vec_evals = 0
    alg._res_m_evals = 0
    alg._k_counter = 0
    alg._curr_grad_norm = 0.0
    alg.fill_pool()
    alg.initialize_ansatz()
    alg.apply_pool_ordering()
    return alg


def compute_initial_energy(mol: Any, case: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    opts = case["run_options"]
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            alg = initialize_algorithm_for_energy(
                mol,
                pool_type=opts["pool_type"],
                init_amps=opts.get("init_amps", "zero"),
                primary_pool_order=opts.get("primary_pool_order", "none"),
                secondary_pool_order=opts.get("secondary_pool_order", "lexical"),
                general_ex_pool_order=opts.get("general_ex_pool_order", "default"),
            )
            energy = alg.energy_feval(alg._tamps)
        return safe_float(energy), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def result_attr(alg: Any, name: str, default: Any = None) -> Any:
    result = getattr(alg, "_final_result", None)
    if result is None:
        return default
    return getattr(result, name, default)


@contextlib.contextmanager
def redirect_native_output(target):
    """Redirect Python and extension-library stdout/stderr to an open file.

    SciPy's L-BFGS-B Fortran output writes directly to file descriptors, so
    contextlib.redirect_stdout alone is not enough to keep the driver output
    readable.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    target.flush()

    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(target.fileno(), 1)
        os.dup2(target.fileno(), 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        target.flush()
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)


def get_energy_trajectory(alg: Any) -> List[float]:
    values = []
    for energy in list(getattr(alg, "_energies", []) or []):
        val = safe_float(energy)
        if val is not None:
            values.append(val)
    final_energy = safe_float(getattr(alg, "_Egs", None))
    if final_energy is not None and (not values or abs(values[-1] - final_energy) > 1.0e-12):
        values.append(final_energy)
    return values


def parse_energy_trajectory_from_log(log_path: Path) -> List[float]:
    """Extract the printed optimizer iteration energies from a QForte log.

    QForte does not currently expose a uniformly populated trajectory list for
    all optimizer paths.  The sandbox therefore parses the readable iteration
    tables that are already printed by the VQE drivers.  This is intentionally
    conservative and ignores summary/final-energy lines.
    """
    if not log_path.exists():
        return []

    energies: List[float] = []
    lbfgs_row = re.compile(r"^\s*\d+\s+\|\s*([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)\s+\|")
    scipy_row = re.compile(
        r"^\s*\d+\s+([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)\s+[-+]?\d+\.\d+"
    )

    for line in log_path.read_text(errors="replace").splitlines():
        match = lbfgs_row.match(line)
        if match is None:
            match = scipy_row.match(line)
        if match is None:
            continue
        val = safe_float(match.group(1))
        if val is not None:
            energies.append(val)
    return energies


def energy_at_iteration(energies: Sequence[float], iteration: int) -> Optional[float]:
    if not energies:
        return None
    idx = min(max(iteration - 1, 0), len(energies) - 1)
    return safe_float(energies[idx])


def analyze_trajectory(energies: Sequence[float]) -> Dict[str, Any]:
    energies = [float(e) for e in energies if safe_float(e) is not None]
    out = {
        "first_reported_energy": energies[0] if energies else None,
        "best_energy": min(energies) if energies else None,
        "largest_drop": None,
        "largest_drop_iter": None,
        "late_drop": False,
    }
    for milestone in [10, 25, 50, 75, 100, 150, 200]:
        out[f"energy_iter_{milestone}"] = energy_at_iteration(energies, milestone)

    if len(energies) < 2:
        return out

    diffs = np.diff(np.asarray(energies, dtype=float))
    min_idx = int(np.argmin(diffs))
    largest_drop = float(diffs[min_idx])
    largest_drop_iter = min_idx + 2
    out["largest_drop"] = largest_drop
    out["largest_drop_iter"] = largest_drop_iter
    out["late_drop"] = bool(largest_drop_iter > 75 and abs(largest_drop) > 1.0e-4)
    return out


def count_nonzero_amplitudes(alg: Any, threshold: float = 1.0e-10) -> Optional[int]:
    tamps = getattr(alg, "_tamps", None)
    if tamps is None:
        return None
    arr = np.asarray(tamps, dtype=float)
    return int(np.count_nonzero(np.abs(arr) > threshold))


def get_pool_size(alg: Any) -> Optional[int]:
    pool = getattr(alg, "_pool_obj", None)
    if pool is None:
        return None
    try:
        return int(len(pool))
    except Exception:
        return None


def run_case(case: Dict[str, Any], mol: Any, results_dir: Path) -> Dict[str, Any]:
    """Run one diagnostic calculation, recording failure instead of raising."""
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{case['case_id']}.log"
    run_options = dict(case["run_options"])
    initial_energy, initial_error = compute_initial_energy(mol, case)

    row = {
        "case_id": case["case_id"],
        "molecule": case["molecule_name"],
        "symmetry": case["symmetry"],
        "pool_type": run_options.get("pool_type"),
        "config_name": case["config_name"],
        "optimizer": run_options.get("optimizer"),
        "init_amps": run_options.get("init_amps", "zero"),
        "primary_pool_order": run_options.get("primary_pool_order", "none"),
        "secondary_pool_order": run_options.get("secondary_pool_order", "lexical"),
        "general_ex_pool_order": run_options.get("general_ex_pool_order", "default"),
        "use_hessian_diag": bool(run_options.get("lbfgs_qf_use_hessian_diag", False)),
        "use_newton_cg": bool(run_options.get("lbfgs_qf_use_newton_cg", False)),
        "use_target_block": bool(run_options.get("lbfgs_qf_use_target_block", False)),
        "target_block_reset_history": run_options.get("lbfgs_qf_target_block_reset_lbfgs_history"),
        "newton_cg_reset_history": run_options.get("lbfgs_qf_newton_cg_reset_lbfgs_history"),
        "maxiter": run_options.get("opt_maxiter"),
        "initial_energy": initial_energy,
        "initial_energy_error": initial_error,
        "log_path": str(log_path.relative_to(results_dir)),
    }

    start = time.perf_counter()
    trajectory: List[float] = []
    try:
        with log_path.open("w") as log_file:
            with redirect_native_output(log_file), contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                alg = make_algorithm(mol, verbose=False)
                alg.run(**run_options)

        runtime = time.perf_counter() - start
        trajectory = get_energy_trajectory(alg)
        log_trajectory = parse_energy_trajectory_from_log(log_path)
        if len(log_trajectory) > len(trajectory):
            trajectory = log_trajectory
        traj_info = analyze_trajectory(trajectory)
        fci_energy = safe_float(getattr(mol, "fci_energy", None))
        final_energy = safe_float(getattr(alg, "_Egs", None))

        row.update(
            {
                "status": "ok",
                "error": None,
                "runtime_s": runtime,
                "pool_size": get_pool_size(alg),
                "n_params": len(getattr(alg, "_tamps", []) or []),
                "n_nonzero": count_nonzero_amplitudes(alg),
                "final_energy": final_energy,
                "best_energy": traj_info["best_energy"] if traj_info["best_energy"] is not None else final_energy,
                "error_vs_fci": (
                    final_energy - fci_energy
                    if final_energy is not None and fci_energy is not None
                    else None
                ),
                "final_grad_norm": safe_float(result_attr(alg, "grad_norm", getattr(alg, "_curr_grad_norm", None))),
                "nit": result_attr(alg, "nit", getattr(alg, "_k_counter", None)),
                "converged": bool(result_attr(alg, "success", False)),
                "nfev": result_attr(alg, "nfev", None),
                "njev": result_attr(alg, "njev", None),
                "nhev": result_attr(alg, "nhev", None),
                "res_vec_evals": getattr(alg, "_res_vec_evals", None),
                "res_m_evals": getattr(alg, "_res_m_evals", None),
                "ham_measurements": (
                    alg.get_num_ham_measurements()
                    if hasattr(alg, "get_num_ham_measurements")
                    else None
                ),
                "pauli_term_measurements": getattr(alg, "_n_pauli_trm_measures", None),
                "cnot": getattr(alg, "_n_cnot", None),
                "target_block_attempts": result_attr(alg, "target_block_attempts", 0),
                "target_block_accepted": result_attr(alg, "target_block_accepted", 0),
                "newton_cg_attempts": result_attr(alg, "newton_cg_attempts", 0),
                "newton_cg_accepted": result_attr(alg, "newton_cg_accepted", 0),
                "escape_negcurv_attempts": result_attr(alg, "escape_negcurv_attempts", 0),
                "escape_negcurv_accepted": result_attr(alg, "escape_negcurv_accepted", 0),
            }
        )
        row.update(traj_info)
        row["run_options"] = json_safe(run_options)
        row["trajectory"] = trajectory
    except Exception as exc:
        runtime = time.perf_counter() - start
        tb = traceback.format_exc()
        with log_path.open("a") as log_file:
            log_file.write("\n\n=== Exception ===\n")
            log_file.write(tb)
        row.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": tb,
                "runtime_s": runtime,
                "run_options": json_safe(run_options),
                "trajectory": trajectory,
            }
        )
    return row


def expected_ph_rank_count(nocc: int, nvir: int, rank: int) -> int:
    total = 0
    for nalpha in range(rank + 1):
        nbeta = rank - nalpha
        if nalpha > nocc or nalpha > nvir or nbeta > nocc or nbeta > nvir:
            continue
        total += (
            math.comb(nocc, nalpha)
            * math.comb(nvir, nalpha)
            * math.comb(nocc, nbeta)
            * math.comb(nvir, nbeta)
        )
    return total


def expected_ph_pool_count(nocc: int, nvir: int, max_rank: int) -> int:
    return sum(expected_ph_rank_count(nocc, nvir, rank) for rank in range(1, max_rank + 1))


def build_raw_pool(mol: Any, pool_type: str):
    pool = qf.SQOpPool()
    pool.set_orb_spaces(mol.hf_reference)
    irreps = getattr(mol, "orb_irreps_to_int", None)
    if irreps is not None and hasattr(pool, "set_orb_irreps"):
        pool.set_orb_irreps(irreps, 0)

    match = re.match(r"^([0-9]+)-UpCCGSD(x?)$", pool_type)
    if match:
        k = int(match.group(1))
        if match.group(2):
            pool.fill_pool_kUpCCGSDx(k)
        else:
            pool.fill_pool_kUpCCGSD(k)
    else:
        pool.fill_pool(pool_type)
    return pool


def clean_coeff(value: Any) -> Tuple[float, float]:
    z = complex(value)
    real = 0.0 if abs(z.real) < 1.0e-12 else round(float(z.real), 12)
    imag = 0.0 if abs(z.imag) < 1.0e-12 else round(float(z.imag), 12)
    return real, imag


def op_signature(op: Any, sign: float = 1.0) -> Tuple[Any, ...]:
    pieces = []
    for coeff, creators, annihilators in op.terms():
        real, imag = clean_coeff(sign * complex(coeff))
        pieces.append((real, imag, tuple(int(i) for i in creators), tuple(int(i) for i in annihilators)))
    return tuple(sorted(pieces))


def canonical_op_signature(op: Any) -> Tuple[Any, ...]:
    """Canonicalize modulo an overall sign, equivalent to theta -> -theta."""
    return min(op_signature(op, +1.0), op_signature(op, -1.0))


def pool_signature_map(pool: Any) -> Dict[Tuple[Any, ...], str]:
    out = {}
    for _, op in pool.terms():
        out.setdefault(canonical_op_signature(op), compact_op_description(op))
    return out


def signature_irrep_products(signature: Tuple[Any, ...], irreps: Sequence[int]) -> List[int]:
    products = []
    for piece in signature:
        if len(piece) < 4:
            continue
        creators = piece[2]
        annihilators = piece[3]
        products.append(term_irrep(irreps, creators, annihilators))
    return sorted(set(products))


def compact_op_description(op: Any) -> str:
    terms = list(op.terms())
    if not terms:
        return "<empty>"
    coeff, creators, annihilators = terms[0]
    real, imag = clean_coeff(coeff)
    coeff_txt = f"{real:+.1f}" if imag == 0.0 else f"{real:+.1f}{imag:+.1f}i"
    return f"{coeff_txt} C{list(creators)} A{list(annihilators)}"


def term_irrep(irreps: Sequence[int], creators: Sequence[int], annihilators: Sequence[int]) -> int:
    sym = 0
    for idx in list(creators) + list(annihilators):
        sym ^= int(irreps[int(idx) // 2])
    return sym


def pool_irrep_bad_terms(pool: Any, irreps: Sequence[int], target_irrep: int = 0) -> List[Dict[str, Any]]:
    bad = []
    for op_idx, (_, op) in enumerate(pool.terms()):
        for term_idx, (_, creators, annihilators) in enumerate(op.terms()):
            if term_irrep(irreps, creators, annihilators) != target_irrep:
                bad.append(
                    {
                        "op_idx": op_idx,
                        "term_idx": term_idx,
                        "creators": list(creators),
                        "annihilators": list(annihilators),
                    }
                )
    return bad


def run_pool_validity_checks(molecule_records: Dict[str, Any], pool_types: Sequence[str]) -> Dict[str, Any]:
    """Build raw pools and compare expected counts/signature sets."""
    pool_rows = []
    signature_maps: Dict[Tuple[str, str], Dict[Tuple[Any, ...], str]] = {}
    pool_lengths: Dict[Tuple[str, str], int] = {}

    for key, rec in molecule_records.items():
        mol = rec["mol"]
        summary = rec["summary"]
        nocc = summary.get("nocc")
        nvir = summary.get("nvir")
        irreps = getattr(mol, "orb_irreps_to_int", None)
        for pool_type in pool_types:
            try:
                pool = build_raw_pool(mol, pool_type)
                sig_map = pool_signature_map(pool)
                signature_maps[(key, pool_type)] = sig_map
                pool_lengths[(key, pool_type)] = len(pool)
                expected = None
                if pool_type in PH_POOL_MAX_RANK and nocc is not None and nvir is not None:
                    expected = expected_ph_pool_count(nocc, nvir, PH_POOL_MAX_RANK[pool_type])
                bad_terms = (
                    pool_irrep_bad_terms(pool, irreps, 0)
                    if irreps is not None
                    else []
                )
                pool_rows.append(
                    {
                        "molecule_key": key,
                        "molecule": summary["name"],
                        "symmetry": summary["symmetry"],
                        "pool_type": pool_type,
                        "pool_size": len(pool),
                        "expected_c1_count": expected if summary["symmetry"].lower() == "c1" else None,
                        "count_status": (
                            "PASS"
                            if summary["symmetry"].lower() == "c1" and expected is not None and len(pool) == expected
                            else "unchecked"
                        ),
                        "bad_irrep_terms": len(bad_terms),
                        "first_ops": list(sig_map.values())[:10],
                    }
                )
            except Exception as exc:
                pool_rows.append(
                    {
                        "molecule_key": key,
                        "molecule": summary["name"],
                        "symmetry": summary["symmetry"],
                        "pool_type": pool_type,
                        "pool_size": None,
                        "expected_c1_count": None,
                        "count_status": "failed",
                        "bad_irrep_terms": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    x_rows = []
    for key in molecule_records:
        for base, xname in x_variant_pairs(pool_types):
            base_map = signature_maps.get((key, base))
            x_map = signature_maps.get((key, xname))
            if base_map is None or x_map is None:
                continue
            base_set = set(base_map)
            x_set = set(x_map)
            x_rows.append(
                {
                    "molecule_key": key,
                    "base_pool": base,
                    "x_pool": xname,
                    "base_size": pool_lengths.get((key, base)),
                    "x_size": pool_lengths.get((key, xname)),
                    "base_unique_signatures": len(base_map),
                    "x_unique_signatures": len(x_map),
                    "signature_sets_equal": base_set == x_set,
                    "only_base_count": len(base_set - x_set),
                    "only_x_count": len(x_set - base_set),
                    "only_base_examples": [base_map[sig] for sig in list(base_set - x_set)[:10]],
                    "only_x_examples": [x_map[sig] for sig in list(x_set - base_set)[:10]],
                }
            )

    symmetry_rows = []
    by_name: Dict[Tuple[str, str], str] = {}
    for key, rec in molecule_records.items():
        summary = rec["summary"]
        by_name[(summary["name"], summary["symmetry"].lower())] = key

    for (name, sym), c1_key in list(by_name.items()):
        if sym != "c1":
            continue
        for d2h_sym in ["d2h"]:
            d2h_key = by_name.get((name, d2h_sym))
            if d2h_key is None:
                continue
            for pool_type in pool_types:
                c1_map = signature_maps.get((c1_key, pool_type))
                d2h_map = signature_maps.get((d2h_key, pool_type))
                if c1_map is None or d2h_map is None:
                    continue
                d2h_irreps = getattr(molecule_records[d2h_key]["mol"], "orb_irreps_to_int", None)
                c1_set = set(c1_map)
                d2h_set = set(d2h_map)
                only_c1 = list(c1_set - d2h_set)
                only_d2h = list(d2h_set - c1_set)
                only_c1_irrep_examples = []
                if d2h_irreps is not None:
                    for sig in only_c1[:10]:
                        only_c1_irrep_examples.append(
                            f"{c1_map[sig]} irrep_products={signature_irrep_products(sig, d2h_irreps)}"
                        )
                symmetry_rows.append(
                    {
                        "molecule": name,
                        "pool_type": pool_type,
                        "c1_size": pool_lengths.get((c1_key, pool_type)),
                        "d2h_size": pool_lengths.get((d2h_key, pool_type)),
                        "c1_unique_signatures": len(c1_map),
                        "d2h_unique_signatures": len(d2h_map),
                        "only_c1_count": len(only_c1),
                        "only_d2h_count": len(only_d2h),
                        "only_c1_examples": [c1_map[sig] for sig in only_c1[:10]],
                        "only_c1_irrep_examples": only_c1_irrep_examples,
                        "only_d2h_examples": [d2h_map[sig] for sig in only_d2h[:10]],
                    }
                )

    return {
        "pool_rows": pool_rows,
        "x_variant_rows": x_rows,
        "symmetry_rows": symmetry_rows,
    }


def x_variant_pairs(pool_types: Sequence[str]) -> List[Tuple[str, str]]:
    pool_set = set(pool_types)
    pairs = []
    if "GSD" in pool_set and "GSDx" in pool_set:
        pairs.append(("GSD", "GSDx"))
    for k in [1, 2, 3]:
        base = f"{k}-UpCCGSD"
        xname = f"{k}-UpCCGSDx"
        if base in pool_set and xname in pool_set:
            pairs.append((base, xname))
    return pairs


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    extra = sorted(
        key
        for row in rows
        for key in row
        if key not in SUMMARY_FIELDS and key not in {"trajectory", "run_options", "traceback"}
    )
    fields = SUMMARY_FIELDS + extra
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json_safe(row.get(field)) for field in fields})


def write_trajectories_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "iteration", "energy"])
        writer.writeheader()
        for row in rows:
            for idx, energy in enumerate(row.get("trajectory", []) or [], start=1):
                writer.writerow({"case_id": row.get("case_id"), "iteration": idx, "energy": energy})


def write_json(payload: Dict[str, Any], path: Path) -> None:
    with path.open("w") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)


def format_value(value: Any, precision: int = 8) -> str:
    val = safe_float(value)
    if val is not None:
        return f"{val:+.{precision}f}"
    if value is None:
        return "-"
    return str(value)


def format_sci(value: Any) -> str:
    val = safe_float(value)
    if val is None:
        return "-"
    return f"{val:.3e}"


def md_table(rows: Sequence[Dict[str, Any]], columns: Sequence[Tuple[str, str]], limit: Optional[int] = None) -> str:
    rows = list(rows)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        return "_No rows._\n"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                cells.append(format_value(value, 8))
            else:
                cells.append(str(value) if value is not None else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def successful_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("status") == "ok" and safe_float(row.get("final_energy")) is not None]


def group_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (row.get("molecule"), row.get("symmetry"), row.get("pool_type"))


def by_group(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)
    return groups


def find_config(rows: Sequence[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for row in rows:
        if row.get("config_name") == name and row.get("status") == "ok":
            return row
    return None


def compare_energy(lhs: Optional[Dict[str, Any]], rhs: Optional[Dict[str, Any]], lhs_name: str, rhs_name: str) -> str:
    if lhs is None or rhs is None:
        return f"{lhs_name} vs {rhs_name}: not enough matching successful rows."
    d = safe_float(lhs.get("final_energy")) - safe_float(rhs.get("final_energy"))
    g_l = format_sci(lhs.get("final_grad_norm"))
    g_r = format_sci(rhs.get("final_grad_norm"))
    return (
        f"{lhs_name} - {rhs_name}: final energy difference {d:+.6e} Eh; "
        f"grad norms {g_l} vs {g_r}; evals g {lhs.get('njev')} vs {rhs.get('njev')}."
    )


def decision_comparisons(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    rows_ok = successful_rows(rows)
    sections = {
        "hessian_diag": [],
        "bfgs_vs_lbfgsb": [],
        "late_drops": [],
        "mp2": [],
        "accelerators": [],
        "reset_history": [],
        "pool_ordering": [],
        "x_variant_energy": [],
        "recommendations": [],
    }

    groups = by_group(rows_ok)
    for key, group_rows in groups.items():
        label = " / ".join(str(x) for x in key)
        sections["bfgs_vs_lbfgsb"].append(
            f"{label}: "
            + compare_energy(find_config(group_rows, "scipy_bfgs"), find_config(group_rows, "scipy_lbfgsb"), "scipy_bfgs", "scipy_lbfgsb")
        )
        sections["hessian_diag"].append(
            f"{label}: "
            + compare_energy(find_config(group_rows, "lbfgs_qf_hdiag"), find_config(group_rows, "lbfgs_qf"), "lbfgs_qf_hdiag", "lbfgs_qf")
        )
        sections["mp2"].append(
            f"{label}: "
            + compare_energy(find_config(group_rows, "lbfgs_qf_mp2_hdiag"), find_config(group_rows, "lbfgs_qf_hdiag"), "lbfgs_qf_mp2_hdiag", "lbfgs_qf_hdiag")
        )
        sections["accelerators"].append(
            f"{label}: "
            + compare_energy(find_config(group_rows, "lbfgs_qf_hdiag_block_no_reset"), find_config(group_rows, "lbfgs_qf_hdiag"), "BLOCK no-reset", "hdiag baseline")
        )
        sections["accelerators"].append(
            f"{label}: "
            + compare_energy(find_config(group_rows, "lbfgs_qf_hdiag_ncg_no_reset"), find_config(group_rows, "lbfgs_qf_hdiag"), "NCG no-reset", "hdiag baseline")
        )
        sections["reset_history"].append(
            f"{label}: "
            + compare_energy(find_config(group_rows, "lbfgs_qf_hdiag_block_reset"), find_config(group_rows, "lbfgs_qf_hdiag_block_no_reset"), "BLOCK reset", "BLOCK no-reset")
        )
        sections["reset_history"].append(
            f"{label}: "
            + compare_energy(find_config(group_rows, "lbfgs_qf_hdiag_ncg_reset"), find_config(group_rows, "lbfgs_qf_hdiag_ncg_no_reset"), "NCG reset", "NCG no-reset")
        )
        sections["pool_ordering"].append(
            f"{label}: "
            + compare_energy(find_config(group_rows, "lbfgs_qf_mp2_order_mp2_shell"), find_config(group_rows, "lbfgs_qf_mp2_hdiag"), "MP2-shell ordering", "MP2+hdiag baseline")
        )
        sections["pool_ordering"].append(
            f"{label}: "
            + compare_energy(find_config(group_rows, "lbfgs_qf_mp2_order_grad_shell"), find_config(group_rows, "lbfgs_qf_mp2_hdiag"), "gradient-shell ordering", "MP2+hdiag baseline")
        )

        file_benchmark_names = [
            "pool_order_file_default_500",
            "pool_order_file_shell_500",
            "pool_order_file_mp2_shell_500",
            "pool_order_file_gradient_shell_500",
            "pool_order_file_gradient_shell_ph_first_500",
        ]
        file_rows = [
            row for row in group_rows if row.get("config_name") in file_benchmark_names
        ]
        if file_rows:
            best_file = min(file_rows, key=lambda r: safe_float(r.get("final_energy")) or float("inf"))
            baseline = find_config(group_rows, "scipy_bfgs")
            sections["pool_ordering"].append(
                f"{label}: test_compare_pool_orderings.py mirror best was "
                f"{best_file['config_name']} at {format_value(best_file.get('final_energy'), 10)} Eh "
                f"with maxiter={best_file.get('maxiter')} and ordering "
                f"({best_file.get('primary_pool_order')}, {best_file.get('secondary_pool_order')}, "
                f"{best_file.get('general_ex_pool_order')})."
            )
            if baseline is not None:
                d = safe_float(best_file.get("final_energy")) - safe_float(baseline.get("final_energy"))
                sections["pool_ordering"].append(
                    f"{label}: file-mirror best minus medium scipy_bfgs baseline = {d:+.6e} Eh. "
                    "This isolates whether the lower energy is coming from the longer 500-iteration "
                    "BFGS/order benchmark rather than a different molecule/pool."
                )

        late_rows = [row for row in group_rows if row.get("late_drop")]
        if late_rows:
            for row in late_rows:
                sections["late_drops"].append(
                    f"{label} / {row['config_name']}: largest drop "
                    f"{format_sci(row.get('largest_drop'))} at iteration {row.get('largest_drop_iter')}."
                )
        else:
            sections["late_drops"].append(f"{label}: no late drop detected in successful runs.")

        best = min(group_rows, key=lambda r: safe_float(r.get("final_energy")) or float("inf"))
        best_eff = min(
            group_rows,
            key=lambda r: (
                abs(safe_float(r.get("error_vs_fci")) or 0.0)
                / max(float(r.get("njev") or r.get("res_vec_evals") or 1), 1.0)
            ),
        )
        sections["recommendations"].append(
            f"{label}: lowest final energy was {best['config_name']} at "
            f"{format_value(best.get('final_energy'), 10)} Eh."
        )
        sections["recommendations"].append(
            f"{label}: lowest FCI-error per gradient-eval proxy was "
            f"{best_eff['config_name']}."
        )

    x_pairs = {}
    for row in rows_ok:
        x_pairs.setdefault((row.get("molecule"), row.get("symmetry"), row.get("config_name")), {})[
            row.get("pool_type")
        ] = row
    for (mol_name, sym, config_name), by_pool_name in x_pairs.items():
        for base, xname in [("GSD", "GSDx"), ("1-UpCCGSD", "1-UpCCGSDx"), ("2-UpCCGSD", "2-UpCCGSDx"), ("3-UpCCGSD", "3-UpCCGSDx")]:
            if base in by_pool_name and xname in by_pool_name:
                d = safe_float(by_pool_name[xname]["final_energy"]) - safe_float(by_pool_name[base]["final_energy"])
                sections["x_variant_energy"].append(
                    f"{mol_name} / {sym} / {config_name}: E({xname}) - E({base}) = {d:+.6e} Eh."
                )

    for key in sections:
        if not sections[key]:
            sections[key].append("No matching successful rows were available for this comparison.")
    return sections


def add_best_in_table_errors(rows: Sequence[Dict[str, Any]]) -> None:
    groups = by_group(successful_rows(rows))
    best_by_group = {
        key: min(safe_float(row.get("final_energy")) for row in group_rows)
        for key, group_rows in groups.items()
    }
    for row in rows:
        best = best_by_group.get(group_key(row))
        final = safe_float(row.get("final_energy"))
        row["error_vs_best_in_table"] = final - best if final is not None and best is not None else None


def write_markdown_summary(
    rows: Sequence[Dict[str, Any]],
    pool_checks: Dict[str, Any],
    molecule_summaries: Sequence[Dict[str, Any]],
    config_defs: Dict[str, Any],
    meta: Dict[str, Any],
    path: Path,
) -> None:
    add_best_in_table_errors(rows)
    comparisons = decision_comparisons(rows)
    rows_ok = successful_rows(rows)
    failures = [row for row in rows if row.get("status") != "ok"]

    result_columns = [
        ("molecule", "molecule"),
        ("symmetry", "sym"),
        ("pool_type", "pool"),
        ("config_name", "config"),
        ("status", "status"),
        ("final_energy", "E_final"),
        ("best_energy", "E_best"),
        ("error_vs_fci", "E-FCI"),
        ("final_grad_norm", "||g||"),
        ("nit", "nit"),
        ("nfev", "nfev"),
        ("njev", "njev"),
        ("n_nonzero", "nnz"),
        ("largest_drop", "max_drop"),
        ("largest_drop_iter", "drop_iter"),
    ]

    lines: List[str] = []
    lines.append("# QForte VQE/tUCC Convergence Diagnostics\n")
    lines.append("## Run Metadata\n")
    for key, value in meta.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")

    lines.append("## System Summary\n")
    system_columns = [
        ("name", "name"),
        ("symmetry", "sym"),
        ("basis", "basis"),
        ("hf_energy", "RHF"),
        ("mp2_energy", "MP2"),
        ("ccsd_energy", "CCSD"),
        ("fci_energy", "FCI0"),
        ("nocc", "nocc"),
        ("nvir", "nvir"),
        ("n_qubits", "qubits"),
        ("orb_irreps_to_int", "irreps"),
    ]
    lines.append(md_table(molecule_summaries, system_columns))

    lines.append("## Pool Validity Summary\n")
    pool_rows = pool_checks.get("pool_rows", [])
    pool_columns = [
        ("molecule", "molecule"),
        ("symmetry", "sym"),
        ("pool_type", "pool"),
        ("pool_size", "size"),
        ("expected_c1_count", "expected C1"),
        ("count_status", "count"),
        ("bad_irrep_terms", "bad irrep terms"),
    ]
    lines.append(md_table(pool_rows, pool_columns, limit=80))

    lines.append("### x Variant Signature Checks\n")
    x_columns = [
        ("molecule_key", "molecule_key"),
        ("base_pool", "base"),
        ("x_pool", "x"),
        ("base_size", "base pool"),
        ("x_size", "x pool"),
        ("base_unique_signatures", "base unique"),
        ("x_unique_signatures", "x unique"),
        ("signature_sets_equal", "set equal"),
        ("only_base_count", "only base"),
        ("only_x_count", "only x"),
    ]
    lines.append(md_table(pool_checks.get("x_variant_rows", []), x_columns))

    lines.append("### C1 vs D2h Pool Differences\n")
    sym_columns = [
        ("molecule", "molecule"),
        ("pool_type", "pool"),
        ("c1_size", "C1 size"),
        ("d2h_size", "D2h size"),
        ("c1_unique_signatures", "C1 unique"),
        ("d2h_unique_signatures", "D2h unique"),
        ("only_c1_count", "only C1"),
        ("only_d2h_count", "only D2h"),
    ]
    lines.append(md_table(pool_checks.get("symmetry_rows", []), sym_columns, limit=80))
    for row in pool_checks.get("symmetry_rows", []):
        if row.get("only_c1_irrep_examples"):
            lines.append(
                f"- {row['molecule']} {row['pool_type']} examples only in C1 with D2h irrep products: "
                + "; ".join(row["only_c1_irrep_examples"][:5])
            )
        elif row.get("only_c1_examples"):
            lines.append(
                f"- {row['molecule']} {row['pool_type']} examples only in C1: "
                + "; ".join(row["only_c1_examples"][:5])
            )
    lines.append("")

    lines.append("## Optimizer Result Table\n")
    lines.append(md_table(rows, result_columns, limit=200))

    lines.append("## Trajectory Diagnostics\n")
    lines.append(
        "Largest drops are computed from consecutive reported energies. "
        "`late_drop=True` means the largest drop happened after iteration 75 "
        "and was larger than 1e-4 Eh."
    )
    traj_columns = [
        ("molecule", "molecule"),
        ("symmetry", "sym"),
        ("pool_type", "pool"),
        ("config_name", "config"),
        ("largest_drop", "largest drop"),
        ("largest_drop_iter", "iter"),
        ("late_drop", "late?"),
        ("energy_iter_75", "E75"),
        ("energy_iter_100", "E100"),
        ("energy_iter_150", "E150"),
        ("energy_iter_200", "E200"),
    ]
    lines.append(md_table(rows_ok, traj_columns, limit=200))

    lines.append("## Decision-Oriented Comparisons\n")
    question_map = [
        ("Did exact Hessian diagonal preconditioning help?", "hessian_diag"),
        ("Did scipy BFGS beat scipy L-BFGS-B?", "bfgs_vs_lbfgsb"),
        ("Was the late energy drop real?", "late_drops"),
        ("Did MP2 initialization help?", "mp2"),
        ("Did target BLOCK or NCG acceleration help?", "accelerators"),
        ("Did resetting L-BFGS history after BLOCK/NCG help or hurt?", "reset_history"),
        ("Did pool ordering matter?", "pool_ordering"),
        ("Did x variants only reorder the ansatz?", "x_variant_energy"),
    ]
    for title, key in question_map:
        lines.append(f"### {title}\n")
        for note in comparisons[key]:
            lines.append(f"- {note}")
        if title == "Did x variants only reorder the ansatz?":
            for row in pool_checks.get("x_variant_rows", []):
                lines.append(
                    f"- Signature check {row['molecule_key']} {row['base_pool']}/{row['x_pool']}: "
                    f"set_equal={row['signature_sets_equal']}, sizes={row['base_size']}/{row['x_size']}."
                )
        lines.append("")

    lines.append("### Is C1 vs D2h a pool-size issue or an optimization issue?\n")
    if pool_checks.get("symmetry_rows"):
        for row in pool_checks["symmetry_rows"]:
            if row["c1_size"] != row["d2h_size"]:
                reason = "pool-size difference"
            else:
                reason = "same pool size; inspect final nonzero amplitudes/optimizer path"
            lines.append(
                f"- {row['molecule']} {row['pool_type']}: C1 size {row['c1_size']}, "
                f"D2h size {row['d2h_size']} -> {reason}."
            )
    else:
        lines.append("- No C1/D2h pair was available in this run.")
    lines.append("")

    lines.append("## Recommended Observations\n")
    for note in comparisons["recommendations"]:
        lines.append(f"- {note}")
    if failures:
        lines.append(f"- {len(failures)} calculations failed; inspect logs and raw_results.json before trusting gaps.")
    lines.append("")

    lines.append("## Failed Cases\n")
    if failures:
        fail_columns = [
            ("case_id", "case"),
            ("molecule", "molecule"),
            ("pool_type", "pool"),
            ("config_name", "config"),
            ("error", "error"),
            ("log_path", "log"),
        ]
        lines.append(md_table(failures, fail_columns, limit=80))
    else:
        lines.append("_No failed cases._\n")

    lines.append("## Full Raw Configuration List\n")
    for name, cfg in config_defs.items():
        lines.append(f"### {name}\n")
        lines.append("```json")
        lines.append(json.dumps(json_safe(cfg), indent=2, sort_keys=True))
        lines.append("```")
    lines.append("")

    path.write_text("\n".join(lines))
