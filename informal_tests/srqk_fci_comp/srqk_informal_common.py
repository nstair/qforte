"""Shared helpers for informal SRQK regression checks."""

from __future__ import annotations

import contextlib
import json
import math
import os
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import numpy as np
import qforte as qf


THIS_DIR = Path(__file__).resolve().parent
EXPECTED_PATH = THIS_DIR / "expected_srqk_fci_comp.json"
LOG_DIR = THIS_DIR / "logs"
BACKEND_LOG_DIR = THIS_DIR / "backend_logs"
LOW_MEMORY_LOG_DIR = THIS_DIR / "low_memory_logs"

S = 5
DT = "lambda_inv"
TARGET_ROOT = 0
USE_EXACT_EVOLUTION = False
DIAGONALIZE_EACH_STEP = False
TROTTER_NUMBER = 1
TROTTER_ORDERS = (1, 2)

QK_TARGET_TROTTER_ERROR = 1.0e-3
QK_TROTTER_BOUND_SCALE = 1.0e-6
QK_VARIANCE_BETA = 2.0
GEV_STABILIZATION_THRESH = 1.0e-10

BACKENDS = ("fock", "fqe", "fci_gpu", "cusv")
OPTIONAL_RUNTIME_SKIP_BACKENDS = {"fqe", "fci_gpu", "cusv"}

TOLERANCES = {
    "energy": 1.0e-8,
    "spin_squared": 1.0e-8,
    "noons": 1.0e-8,
    "matrix": 1.0e-8,
    "time": 1.0e-10,
}

PARITY_TOLERANCES = {
    "energy": 1.0e-6,
    "spin_squared": 1.0e-6,
    "noons": 1.0e-6,
    "matrix": 1.0e-6,
    "time": 1.0e-9,
}


SYMMETRIES = ("c1", "d2h")


CASE_STYLES = [
    {
        "style": "manual_linear_old_behavior",
        "qk_tmax_type": "manual",
        "qk_time_grid": "linear",
        "qk_trotter_control": "fixed",
    },
    {
        "style": "variance_linear_fixed_trotter",
        "qk_tmax_type": "variance",
        "qk_time_grid": "linear",
        "qk_trotter_control": "fixed",
    },
    {
        "style": "variance_linear_auto_trotter",
        "qk_tmax_type": "variance",
        "qk_time_grid": "linear",
        "qk_trotter_control": "auto",
    },
    {
        "style": "variance_power_auto_trotter",
        "qk_tmax_type": "variance",
        "qk_time_grid": "power",
        "qk_time_power": 2.0,
        "qk_trotter_control": "auto",
    },
]


def linear_h4_geometry():
    return [
        ("H", (0.0, 0.0, 1.0)),
        ("H", (0.0, 0.0, 2.0)),
        ("H", (0.0, 0.0, 3.0)),
        ("H", (0.0, 0.0, 4.0)),
    ]


def print_molecule_header(symmetries=SYMMETRIES):
    print("\n==> Molecule Systems <==")
    print("  system: linear H4")
    print("  basis:  sto-3g")
    print(f"  symmetries: {', '.join(symmetries)}")
    print("  geometry:")
    print(f"{'atom':>6s} {'x':>14s} {'y':>14s} {'z':>14s}")
    print("-" * 52)
    for atom, xyz in linear_h4_geometry():
        print(f"{atom:>6s} {xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f}")


def build_linear_h4(symmetry):
    return qf.system_factory(
        system_type="molecule",
        build_type="psi4",
        basis="sto-3g",
        mol_geometry=linear_h4_geometry(),
        symmetry=symmetry,
        multiplicity=1,
        charge=0,
        num_frozen_docc=0,
        num_frozen_uocc=0,
        run_mp2=False,
        run_ccsd=False,
        run_cisd=False,
        run_fci=True,
        store_mo_ints=True,
        store_mo_ints_np=True,
    )


def case_matrix():
    cases = []
    for symmetry in SYMMETRIES:
        for order in TROTTER_ORDERS:
            for style in CASE_STYLES:
                case = dict(style)
                case["symmetry"] = symmetry
                case["trotter_order"] = order
                case["label"] = f"{symmetry}_rho{order}_{style['style']}"
                cases.append(case)
    return cases


def selected_cases(labels):
    cases = case_matrix()
    if not labels:
        return cases
    wanted = set(labels)
    selected = [case for case in cases if case["label"] in wanted]
    found = {case["label"] for case in selected}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"Unknown case label(s): {missing}")
    return selected


def common_run_kwargs(case, low_memory=False):
    kwargs = {
        "s": S,
        "dt": DT,
        "target_root": TARGET_ROOT,
        "use_exact_evolution": USE_EXACT_EVOLUTION,
        "diagonalize_each_step": DIAGONALIZE_EACH_STEP,
        "low_memory_mat_formation": low_memory,
        "qk_tmax": None,
        "qk_target_trotter_error": QK_TARGET_TROTTER_ERROR,
        "qk_trotter_bound_scale": QK_TROTTER_BOUND_SCALE,
        "qk_variance_beta": QK_VARIANCE_BETA,
        "gev_stabilization_thresh": GEV_STABILIZATION_THRESH,
        "qk_tmax_type": case["qk_tmax_type"],
        "qk_time_grid": case["qk_time_grid"],
        "qk_trotter_control": case["qk_trotter_control"],
    }
    if "qk_time_power" in case:
        kwargs["qk_time_power"] = case["qk_time_power"]
    return kwargs


def apply_ham_as_tensor(backend):
    return backend not in {"fock", "cusv"}


def build_alg(mol, backend, trotter_order, low_memory=False):
    return qf.SRQK(
        mol,
        computer_type=backend,
        apply_ham_as_tensor=apply_ham_as_tensor(backend),
        trotter_number=TROTTER_NUMBER,
        trotter_order=trotter_order,
        verbose=False,
        print_summary_file=False,
    )


def run_srqk_case(mol, case, backend="fci", low_memory=False):
    alg = build_alg(mol, backend, case["trotter_order"], low_memory=low_memory)
    alg.run(**common_run_kwargs(case, low_memory=low_memory))
    return alg


def round_float(value, digits=12):
    if value is None:
        return None
    value = float(np.real(value))
    if not math.isfinite(value):
        return value
    return round(value, digits)


def round_sequence(values, digits=12):
    if values is None:
        return None
    return [round_float(value, digits=digits) for value in values]


def complex_matrix_to_json(matrix, digits=12):
    matrix = np.asarray(matrix, dtype=complex)
    return [
        [[round_float(value.real, digits), round_float(value.imag, digits)] for value in row]
        for row in matrix
    ]


def complex_matrix_from_json(matrix):
    return np.asarray(
        [[complex(value[0], value[1]) for value in row] for row in matrix],
        dtype=complex,
    )


def analytical_cnot_count(alg):
    pool = qf.SQOpPool()
    pool.add_hermitian_pairs(1.0, alg._sq_ham)
    base_cnot = int(pool.count_cnot_for_jw_exponential(False, 1))
    order_factor = 1 if alg._trotter_order <= 1 else 2 ** (alg._trotter_order - 1)
    total_trotter_steps = int(sum(alg._qk_macro_trotter_number_list))
    return base_cnot * order_factor * total_trotter_steps


def final_diagnostics_record(alg):
    try:
        spin_squared = alg.compute_final_spin_squared_expectation()
    except Exception as exc:
        spin_squared = None
        spin_error = f"{type(exc).__name__}: {exc}"
    else:
        spin_error = getattr(alg, "_spin_squared_error", None)

    try:
        noons = alg.compute_final_noons()
    except Exception as exc:
        noons = None
        noons_error = f"{type(exc).__name__}: {exc}"
    else:
        noons_error = getattr(alg, "_noons_error", None)

    return {
        "spin_squared_available": spin_squared is not None,
        "spin_squared": round_float(spin_squared) if spin_squared is not None else None,
        "spin_squared_skip_reason": None if spin_squared is not None else spin_error,
        "noons_available": noons is not None,
        "noons": round_sequence(noons) if noons is not None else None,
        "noons_skip_reason": None if noons is not None else noons_error,
    }


def result_record(case, mol, alg, backend="fci", low_memory=False):
    macro_dt = np.asarray(alg._qk_macro_dt_list, dtype=float)
    macro_trotter = np.asarray(alg._qk_macro_trotter_number_list, dtype=int)
    longest_idx = int(np.argmax(np.abs(macro_dt)))

    record = {
        "case": case["label"],
        "symmetry": case["symmetry"],
        "style": case["style"],
        "backend": backend,
        "low_memory": bool(low_memory),
        "trotter_order": int(case["trotter_order"]),
        "system": {
            "name": "linear H4",
            "symmetry": case["symmetry"],
            "fci_energy": round_float(mol.fci_energy),
            "hf_energy": round_float(getattr(mol, "hf_energy", None)),
        },
        "run_options": common_run_kwargs(case, low_memory=low_memory),
        "energy": round_float(alg.get_ts_energy()),
        "abs_error_vs_fci": round_float(abs(alg.get_ts_energy() - mol.fci_energy)),
        "Hbar": complex_matrix_to_json(alg._Hbar),
        "S": complex_matrix_to_json(alg._S),
        "final_reduced_rank": int(alg._qk_geig_reduced_rank(alg._S)),
        "cnot_count": analytical_cnot_count(alg),
        "reported_cnot_count": int(alg._n_cnot),
        "qk_tmax": round_float(alg._qk_tmax),
        "qk_macro_dt_list": round_sequence(alg._qk_macro_dt_list),
        "qk_effective_micro_dt_list": round_sequence(alg._qk_effective_micro_dt_list),
        "qk_macro_trotter_number_list": [
            int(value) for value in alg._qk_macro_trotter_number_list
        ],
        "longest_macro_dt": round_float(abs(macro_dt[longest_idx])),
        "longest_macro_trotter_number": int(macro_trotter[longest_idx]),
        "largest_cnot": analytical_cnot_count(alg),
    }
    record.update(final_diagnostics_record(alg))
    return record


def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def compare_scalar(name, observed, expected, tol):
    delta = abs(float(observed) - float(expected))
    return delta <= tol, delta


def compare_sequence(name, observed, expected, tol):
    if len(observed) != len(expected):
        return False, f"length {len(observed)} != {len(expected)}"
    if not observed:
        return True, 0.0
    diffs = [abs(float(a) - float(b)) for a, b in zip(observed, expected)]
    return max(diffs) <= tol, max(diffs)


def compare_integer(name, observed, expected):
    return int(observed) == int(expected), abs(int(observed) - int(expected))


def compare_matrix(name, observed, expected, tol):
    obs = complex_matrix_from_json(observed)
    exp = complex_matrix_from_json(expected)
    if obs.shape != exp.shape:
        return False, f"shape {obs.shape} != {exp.shape}"
    max_diff = float(np.max(np.abs(obs - exp))) if obs.size else 0.0
    return max_diff <= tol, max_diff


def compact_skip_reason(reason):
    if not reason:
        return "diagnostic subroutine is unavailable"
    reason = str(reason)
    lowered = reason.lower()
    if "not yet implemented" in lowered or "only computer_type=\"fci\"" in lowered:
        return "diagnostic subroutine is not implemented for this backend"
    return reason


def comparison_rows(observed, expected, tolerances):
    rows = []

    checks = [
        ("energy", "srqk target E diff", compare_scalar, tolerances["energy"]),
        ("qk_tmax", "srqk Tmax diff", compare_scalar, tolerances["time"]),
        ("qk_macro_dt_list", "srqk macro dt diff", compare_sequence, tolerances["time"]),
        (
            "qk_effective_micro_dt_list",
            "srqk micro dt diff",
            compare_sequence,
            tolerances["time"],
        ),
        ("Hbar", "srqk H mat diff", compare_matrix, tolerances["matrix"]),
        ("S", "srqk S mat diff", compare_matrix, tolerances["matrix"]),
    ]

    for key, label, func, tol in checks:
        passed, diff = func(label, observed[key], expected[key], tol)
        rows.append((label, passed, diff, tol))

    optional_checks = [
        (
            "spin_squared",
            "spin_squared_available",
            "spin_squared_skip_reason",
            "srqk <S^2> diff",
            compare_scalar,
            tolerances["spin_squared"],
        ),
        (
            "noons",
            "noons_available",
            "noons_skip_reason",
            "srqk NOONs diff",
            compare_sequence,
            tolerances["noons"],
        ),
    ]
    for key, available_key, reason_key, label, func, tol in optional_checks:
        expected_available = expected.get(available_key, expected.get(key) is not None)
        observed_available = observed.get(available_key, observed.get(key) is not None)
        if not expected_available:
            reason = compact_skip_reason(
                expected.get(reason_key) or "expected reference diagnostic is unavailable"
            )
            rows.append((label, None, reason, tol))
            continue
        if not observed_available:
            reason = compact_skip_reason(
                observed.get(reason_key) or "backend diagnostic subroutine is unavailable"
            )
            rows.append((label, False, f"missing diagnostic: {reason}", tol))
            continue

        passed, diff = func(label, observed[key], expected[key], tol)
        rows.append((label, passed, diff, tol))

    for key, label in [
        ("final_reduced_rank", "srqk final RR diff"),
        ("cnot_count", "srqk CNOT count diff"),
    ]:
        passed, diff = compare_integer(label, observed[key], expected[key])
        rows.append((label, passed, diff, 0))

    passed, diff = compare_sequence(
        "srqk macro Trotter steps diff",
        observed["qk_macro_trotter_number_list"],
        expected["qk_macro_trotter_number_list"],
        0.0,
    )
    rows.append(("srqk macro Trotter steps diff", passed, diff, 0))

    return rows


def print_comparison_table(case_label, rows):
    print(f"\n==> Comparison: {case_label} <==")
    header = f"{'check':34s} {'status':>8s} {'max |diff| / reason':>36s} {'tolerance':>12s}"
    print(header)
    print("-" * len(header))
    for label, passed, diff, tol in rows:
        if passed is None:
            status = "SKIP"
        else:
            status = "PASS" if passed else "FAIL"
        if isinstance(diff, str):
            diff_text = diff
        else:
            diff_text = f"{float(diff):14.6e}"
        tol_text = f"{float(tol):12.2e}" if isinstance(tol, float) else str(tol)
        print(f"{label:34s} {status:>8s} {diff_text:>36s} {tol_text:>12s}")


def compare_records(observed, expected, tolerances, display_label):
    if observed.get("symmetry") != expected.get("symmetry"):
        raise AssertionError(
            f"SRQK informal comparison failed for {display_label}: "
            f"symmetry {observed.get('symmetry')!r} != {expected.get('symmetry')!r}"
        )
    rows = comparison_rows(observed, expected, tolerances)
    print_comparison_table(display_label, rows)
    failures = [row for row in rows if row[1] is False]
    if failures:
        details = "\n".join(f"  - {label}: diff={diff}, tol={tol}" for label, _, diff, tol in failures)
        raise AssertionError(f"SRQK informal comparison failed for {display_label}:\n{details}")


def load_expected(cases):
    if not EXPECTED_PATH.exists():
        raise FileNotFoundError(
            f"Missing expected results file: {EXPECTED_PATH}. "
            "Run run_srqk_fci_comp.py --write-expected after reviewing the output."
        )

    expected = json.loads(EXPECTED_PATH.read_text())
    expected_cases = {record["case"]: record for record in expected["cases"]}
    wanted = {case["label"] for case in cases}
    missing = sorted(wanted - set(expected_cases))
    if missing:
        raise ValueError(f"Expected SRQK data is missing case label(s): {missing}")
    return {label: expected_cases[label] for label in sorted(wanted)}


def build_systems(cases, log_root):
    systems = {}
    build_logs = {}
    for symmetry in sorted({case["symmetry"] for case in cases}):
        build_log = log_root / f"build_linear_h4_{symmetry}_sto3g.log"
        with build_log.open("w") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                systems[symmetry] = build_linear_h4(symmetry)
        build_logs[symmetry] = build_log
    return systems, build_logs


def run_all_cases(cases, backend="fci", low_memory=False, log_root=LOG_DIR):
    log_root.mkdir(parents=True, exist_ok=True)
    systems, build_logs = build_systems(cases, log_root)

    print_molecule_header(sorted(systems))
    print("\n  Reference systems:")
    for symmetry in sorted(systems):
        print(f"    {symmetry:<4s} FCI energy: {systems[symmetry].fci_energy:+18.12f}")
        print(f"         build log: {build_logs[symmetry]}")

    records = []
    skipped = []
    for index, case in enumerate(cases, start=1):
        mol = systems[case["symmetry"]]
        log_path = log_root / f"{backend}_{'lowmem_' if low_memory else ''}{case['label']}.log"
        print(
            f"\n[{index:02d}/{len(cases):02d}] "
            f"backend={backend:<7s} low_memory={str(low_memory):<5s} "
            f"case={case['label']} -> {log_path}"
        )
        try:
            with log_path.open("w") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    try:
                        alg = run_srqk_case(
                            mol,
                            case,
                            backend=backend,
                            low_memory=low_memory,
                        )
                    except Exception:
                        traceback.print_exc(file=log)
                        raise
            records.append(result_record(case, mol, alg, backend=backend, low_memory=low_memory))
            print("  status: PASS run completed")
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"  status: FAIL/SKIP run did not complete: {reason}")
            skipped.append({"backend": backend, "case": case["label"], "reason": reason})
            if backend == "fci":
                raise

    return {
        "description": "Informal linear-H4/STO-3G SRQK FCIComputer consistency data",
        "system_name": "linear H4",
        "symmetries": list(sorted(systems)),
        "s": S,
        "dt": DT,
        "trotter_number": TROTTER_NUMBER,
        "trotter_orders": list(TROTTER_ORDERS),
        "tolerances": TOLERANCES,
        "cases": records,
    }, skipped


def print_case_summary(records, systems):
    print("\n\n==> Case summary <==")
    print("  FCI reference energies:")
    for symmetry in sorted(systems):
        print(f"    {symmetry:<4s} {systems[symmetry].fci_energy:+16.10f}")

    header = (
        f"{'case':46s} {'final E':>16s} {'|E-FCI|':>12s} "
        f"{'Tmax':>12s} {'max macro dt':>14s} {'m(max dt)':>10s} "
        f"{'max CNOT':>12s} {'final RR':>9s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for record in records:
        print(
            f"{record['case']:46s} "
            f"{record['energy']:+16.10f} "
            f"{record['abs_error_vs_fci']:12.4e} "
            f"{record['qk_tmax']:12.4e} "
            f"{record['longest_macro_dt']:14.4e} "
            f"{record['longest_macro_trotter_number']:10d} "
            f"{record['largest_cnot']:12d} "
            f"{record['final_reduced_rank']:9d}"
        )


def backend_available(backend):
    if backend == "fock":
        return True, None

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
            from qforte.cusv_api.cusv_computer import (
                _CUPY_AVAILABLE,
                _CUSV_AVAILABLE,
            )
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


def should_skip_runtime_error(backend, exc):
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


def selected_backends(values):
    if not values:
        return list(BACKENDS)
    unknown = sorted(set(values) - set(BACKENDS))
    if unknown:
        raise ValueError(f"Unknown backend(s): {unknown}. Expected one of {BACKENDS}.")
    return values


def write_json(path, payload):
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
