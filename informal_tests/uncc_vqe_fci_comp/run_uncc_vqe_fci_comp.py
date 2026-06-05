"""Informal UCCN-VQE FCIComputer regression harness.

Run from the repository root with qfe_env_v1:

    conda run -n qfe_env_v1 python informal_tests/uncc_vqe_fci_comp/run_uncc_vqe_fci_comp.py

To intentionally refresh the reference data after reviewing expected changes:

    conda run -n qfe_env_v1 python informal_tests/uncc_vqe_fci_comp/run_uncc_vqe_fci_comp.py --write-expected

This is intentionally not a pytest test.  It keeps compact reference data for a
small set of H4/C1 and H4/D2h UCCN-VQE FCIComputer runs.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import numpy as np
import qforte as qf


THIS_DIR = Path(__file__).resolve().parent
EXPECTED_PATH = THIS_DIR / "expected_uncc_vqe_fci_comp.json"
LOG_DIR = THIS_DIR / "logs"

MAXITER = 5
OPT_THRESH = 1.0e-5
OPT_FTOL = 1.0e-10

TOLERANCES = {
    "energy": 1.0e-8,
    "spin_squared": 1.0e-8,
    "noons": 1.0e-8,
    "amplitudes": 1.0e-8,
    "gradient": 1.0e-8,
}


CASE_MATRIX = [
    {
        "label": "c1_sd_bfgs_mp2_hdiag_mp2",
        "symmetry": "c1",
        "pool_type": "SD",
        "optimizer": "bfgs_qf",
        "init_amps": "mp2",
        "primary_pool_order": "mp2_amps",
        "secondary_pool_order": "shell",
        "general_ex_pool_order": "default",
        "hdiag_method": "mp2",
    },
    {
        "label": "c1_sd_jacobi_zero",
        "symmetry": "c1",
        "pool_type": "SD",
        "optimizer": "jacobi",
        "init_amps": "zero",
    },
    {
        "label": "c1_gsd_lbfgs_grad_order_batch",
        "symmetry": "c1",
        "pool_type": "GSD",
        "optimizer": "lbfgs_qf",
        "init_amps": "zero",
        "primary_pool_order": "gradients",
        "secondary_pool_order": "shell",
        "general_ex_pool_order": "particle_hole_first",
        "batched_opt_type": "half_sweep",
        "hdiag_method": "analytic",
    },
    {
        "label": "c1_1up_bfgs_mp2_batch_hdiag_analytic",
        "symmetry": "c1",
        "pool_type": "1-UpCCGSD",
        "optimizer": "bfgs_qf",
        "init_amps": "mp2",
        "primary_pool_order": "mp2_amps",
        "secondary_pool_order": "shell",
        "general_ex_pool_order": "particle_hole_first",
        "batched_opt_type": "half_sweep_then_all",
        "hdiag_method": "analytic",
    },
    {
        "label": "c1_2up_lbfgs_zero_batch",
        "symmetry": "c1",
        "pool_type": "2-UpCCGSD",
        "optimizer": "lbfgs_qf",
        "init_amps": "zero",
        "secondary_pool_order": "shell",
        "general_ex_pool_order": "particle_hole_first",
        "batched_opt_type": "half_sweep",
        "hdiag_method": "analytic",
    },
    {
        "label": "d2h_sd_lbfgs_zero_hdiag_analytic",
        "symmetry": "d2h",
        "pool_type": "SD",
        "optimizer": "lbfgs_qf",
        "init_amps": "zero",
        "secondary_pool_order": "shell",
        "hdiag_method": "analytic",
    },
    {
        "label": "d2h_sd_jacobi_mp2",
        "symmetry": "d2h",
        "pool_type": "SD",
        "optimizer": "jacobi",
        "init_amps": "mp2",
        "primary_pool_order": "mp2_amps",
    },
    {
        "label": "d2h_gsd_bfgs_batch_hdiag_analytic",
        "symmetry": "d2h",
        "pool_type": "GSD",
        "optimizer": "bfgs_qf",
        "init_amps": "zero",
        "secondary_pool_order": "shell",
        "general_ex_pool_order": "particle_hole_first",
        "batched_opt_type": "half_sweep_then_all",
        "hdiag_method": "analytic",
    },
    {
        "label": "d2h_1up_lbfgs_mp2_batch",
        "symmetry": "d2h",
        "pool_type": "1-UpCCGSD",
        "optimizer": "lbfgs_qf",
        "init_amps": "mp2",
        "primary_pool_order": "mp2_amps",
        "secondary_pool_order": "shell",
        "general_ex_pool_order": "particle_hole_first",
        "batched_opt_type": "half_sweep",
        "hdiag_method": "analytic",
    },
    {
        "label": "d2h_2up_jacobi_zero",
        "symmetry": "d2h",
        "pool_type": "2-UpCCGSD",
        "optimizer": "jacobi",
        "init_amps": "zero",
        "general_ex_pool_order": "particle_hole_first",
    },
]


def h4_geometry():
    rhh = 1.5
    return [
        ("H", (0.0, -rhh / 2.0, -rhh / 2.0)),
        ("H", (0.0, -rhh / 2.0, +rhh / 2.0)),
        ("H", (0.0, +rhh / 2.0, -rhh / 2.0)),
        ("H", (0.0, +rhh / 2.0, +rhh / 2.0)),
    ]


def build_h4(symmetry: str):
    return qf.system_factory(
        system_type="molecule",
        build_type="psi4",
        basis="sto-6g",
        mol_geometry=h4_geometry(),
        symmetry=symmetry,
        multiplicity=1,
        charge=0,
        num_frozen_docc=0,
        num_frozen_uocc=0,
        run_mp2=True,
        run_ccsd=False,
        run_cisd=False,
        run_fci=True,
        store_mo_ints=True,
    )


def qf_optimizer_options(case: dict[str, Any]) -> dict[str, Any]:
    optimizer = case["optimizer"].lower()
    hdiag_method = case.get("hdiag_method")
    use_hdiag = hdiag_method is not None and optimizer in {"bfgs_qf", "lbfgs_qf"}

    if optimizer == "bfgs_qf":
        return {
            "bfgs_qf_maxiter": MAXITER,
            "bfgs_qf_max_ls": 10,
            "bfgs_qf_alpha0": 1.0,
            "bfgs_qf_max_step_norm": 0.5,
            "bfgs_qf_reset_on_bad_curvature": True,
            "bfgs_qf_reset_on_nondescent": True,
            "bfgs_qf_use_hessian_diag": use_hdiag,
            "bfgs_qf_hdiag_method": hdiag_method or "analytic",
            "bfgs_qf_hdiag_start": 1,
            "bfgs_qf_hdiag_stop": 1,
            "bfgs_qf_hdiag_mode": "abs",
            "bfgs_qf_hdiag_floor": 1.0e-3,
            "bfgs_qf_print_aux": False,
            "bfgs_qf_use_gradient_energy": True,
        }

    if optimizer == "lbfgs_qf":
        return {
            "lbfgs_qf_memory": 5,
            "lbfgs_qf_max_ls": 10,
            "lbfgs_qf_alpha0": 1.0,
            "lbfgs_qf_max_step_norm": 0.5,
            "lbfgs_qf_use_hessian_diag": use_hdiag,
            "lbfgs_qf_hdiag_method": hdiag_method or "analytic",
            "lbfgs_qf_hdiag_start": 1,
            "lbfgs_qf_hdiag_stop": 1,
            "lbfgs_qf_hdiag_mode": "abs",
            "lbfgs_qf_hdiag_floor": 1.0e-3,
            "lbfgs_qf_print_aux": False,
            "lbfgs_qf_use_gradient_energy": True,
        }

    return {}


def make_run_options(case: dict[str, Any]) -> dict[str, Any]:
    options = {
        "pool_type": case["pool_type"],
        "optimizer": case["optimizer"],
        "opt_maxiter": MAXITER,
        "opt_thresh": OPT_THRESH,
        "opt_ftol": OPT_FTOL,
        "use_analytic_grad": True,
        "init_amps": case.get("init_amps", "zero"),
        "primary_pool_order": case.get("primary_pool_order", "none"),
        "secondary_pool_order": case.get("secondary_pool_order", "lexical"),
        "general_ex_pool_order": case.get("general_ex_pool_order", "default"),
        "batched_opt_type": case.get("batched_opt_type", "none"),
        "batched_opt_cycles": 1,
        "batch_opt_thresh": OPT_THRESH,
        "batched_opt_batch_maxiter": MAXITER,
        "batched_opt_final_maxiter": MAXITER,
        "batched_opt_verbose": False,
    }
    options.update(qf_optimizer_options(case))
    return options


def round_float(value: Any, digits: int = 12):
    if value is None:
        return None
    value = float(np.real(value))
    if not math.isfinite(value):
        return value
    return round(value, digits)


def round_list(values: Any, digits: int = 12):
    if values is None:
        return None
    return [round_float(value, digits=digits) for value in values]


def json_safe(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def final_result_field(alg, name: str):
    result = getattr(alg, "_final_result", None)
    if result is None:
        return None
    return getattr(result, name, None)


def result_record(case: dict[str, Any], mol, alg) -> dict[str, Any]:
    gradient = np.asarray(alg.gradient_ary_feval(alg._tamps), dtype=float)
    spin_squared = alg.compute_final_spin_squared_expectation()
    noons = alg.compute_final_noons()

    return json_safe({
        "case": case["label"],
        "symmetry": case["symmetry"],
        "pool_type": case["pool_type"],
        "optimizer": case["optimizer"],
        "run_options": make_run_options(case),
        "system": {
            "hf_energy": round_float(getattr(mol, "hf_energy", None)),
            "mp2_energy": round_float(getattr(mol, "mp2_energy", None)),
            "fci_energy": round_float(getattr(mol, "fci_energy", None)),
        },
        "energy": round_float(alg.get_gs_energy()),
        "spin_squared": round_float(spin_squared),
        "noons": round_list(noons),
        "cnot_count": int(alg._n_cnot),
        "gradient_norm": round_float(np.linalg.norm(gradient)),
        "gradient": round_list(gradient),
        "amplitudes": round_list(alg._tamps),
        "pool_indices": [int(top) for top in alg._tops],
        "n_pool": int(len(alg._pool_obj)),
        "n_params": int(len(alg._tamps)),
        "n_nonzero_params": int(sum(abs(t) > 1.0e-12 for t in alg._tamps)),
        "optimizer_result": {
            "nit": final_result_field(alg, "nit"),
            "nfev": final_result_field(alg, "nfev"),
            "njev": final_result_field(alg, "njev"),
            "success": final_result_field(alg, "success"),
        },
        "batched_history": compact_batch_history(
            getattr(alg, "_batched_opt_history", [])
        ),
    })


def compact_batch_history(history):
    compacted = []
    for item in history:
        compacted.append(
            {
                "label": item.get("batch_label"),
                "active_size": item.get("active_size"),
                "initial_energy": round_float(item.get("initial_energy")),
                "final_energy": round_float(item.get("final_energy")),
                "final_reduced_grad_norm": round_float(
                    item.get("final_reduced_grad_norm")
                ),
                "final_full_grad_norm": round_float(item.get("final_full_grad_norm")),
                "iterations": item.get("iterations"),
            }
        )
    return compacted


def run_case(case: dict[str, Any], systems: dict[str, Any]) -> dict[str, Any]:
    mol = systems[case["symmetry"]]
    alg = qf.UCCNVQE(
        mol,
        computer_type="fci",
        apply_ham_as_tensor=True,
        verbose=False,
        print_summary_file=False,
    )
    alg.run(**make_run_options(case))
    return result_record(case, mol, alg)


def compare_scalar(path: str, observed: Any, expected: Any, tol: float, failures: list[str]):
    if observed is None or expected is None:
        if observed != expected:
            failures.append(f"{path}: observed {observed!r}, expected {expected!r}")
        return
    delta = abs(float(observed) - float(expected))
    if delta > tol:
        failures.append(
            f"{path}: observed {observed:.12g}, expected {expected:.12g}, |diff|={delta:.3e}"
        )


def compare_sequence(path: str, observed: Any, expected: Any, tol: float, failures: list[str]):
    if observed is None or expected is None:
        if observed != expected:
            failures.append(f"{path}: observed {observed!r}, expected {expected!r}")
        return
    if len(observed) != len(expected):
        failures.append(f"{path}: length {len(observed)} != expected {len(expected)}")
        return
    if not observed:
        return
    diffs = [abs(float(a) - float(b)) for a, b in zip(observed, expected)]
    max_diff = max(diffs)
    if max_diff > tol:
        idx = diffs.index(max_diff)
        failures.append(
            f"{path}: max |diff|={max_diff:.3e} at index {idx}; "
            f"observed {observed[idx]:.12g}, expected {expected[idx]:.12g}"
        )


def compare_results(observed: dict[str, Any], expected: dict[str, Any]):
    failures = []
    observed_cases = {record["case"]: record for record in observed["cases"]}
    expected_cases = {record["case"]: record for record in expected["cases"]}

    missing = sorted(set(expected_cases) - set(observed_cases))
    extra = sorted(set(observed_cases) - set(expected_cases))
    if missing:
        failures.append(f"Missing cases: {missing}")
    if extra:
        failures.append(f"Unexpected cases: {extra}")

    for label in sorted(set(observed_cases) & set(expected_cases)):
        obs = observed_cases[label]
        exp = expected_cases[label]
        prefix = f"{label}"

        for key in ["symmetry", "pool_type", "optimizer", "n_pool", "n_params"]:
            if obs.get(key) != exp.get(key):
                failures.append(f"{prefix}.{key}: {obs.get(key)!r} != {exp.get(key)!r}")

        if obs.get("cnot_count") != exp.get("cnot_count"):
            failures.append(
                f"{prefix}.cnot_count: {obs.get('cnot_count')} != {exp.get('cnot_count')}"
            )

        compare_scalar(
            f"{prefix}.energy",
            obs["energy"],
            exp["energy"],
            TOLERANCES["energy"],
            failures,
        )
        compare_scalar(
            f"{prefix}.spin_squared",
            obs["spin_squared"],
            exp["spin_squared"],
            TOLERANCES["spin_squared"],
            failures,
        )
        compare_sequence(
            f"{prefix}.noons",
            obs["noons"],
            exp["noons"],
            TOLERANCES["noons"],
            failures,
        )
        compare_sequence(
            f"{prefix}.amplitudes",
            obs["amplitudes"],
            exp["amplitudes"],
            TOLERANCES["amplitudes"],
            failures,
        )
        compare_sequence(
            f"{prefix}.gradient",
            obs["gradient"],
            exp["gradient"],
            TOLERANCES["gradient"],
            failures,
        )

    if failures:
        message = "\n".join(f"  - {failure}" for failure in failures)
        raise AssertionError(f"Informal UCCN-VQE FCIComputer check failed:\n{message}")


def selected_cases(labels: list[str] | None):
    if not labels:
        return copy.deepcopy(CASE_MATRIX)
    wanted = set(labels)
    cases = [copy.deepcopy(case) for case in CASE_MATRIX if case["label"] in wanted]
    found = {case["label"] for case in cases}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"Unknown case label(s): {missing}")
    return cases


def run_all(cases: list[dict[str, Any]]):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    symmetries = sorted({case["symmetry"] for case in cases})
    systems = {}

    for symmetry in symmetries:
        log_path = LOG_DIR / f"build_h4_{symmetry}.log"
        with log_path.open("w") as log, contextlib.redirect_stdout(log):
            systems[symmetry] = build_h4(symmetry)

    records = []
    for index, case in enumerate(cases, start=1):
        log_path = LOG_DIR / f"{case['label']}.log"
        print(f"[{index:02d}/{len(cases):02d}] {case['label']} -> {log_path}")
        with log_path.open("w") as log, contextlib.redirect_stdout(log):
            records.append(run_case(case, systems))

    return json_safe({
        "description": "Informal H4 UCCN-VQE FCIComputer consistency data",
        "maxiter": MAXITER,
        "tolerances": TOLERANCES,
        "cases": records,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-expected",
        action="store_true",
        help="Write observed results to the expected JSON file.",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Run only one named case. May be supplied more than once.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print case labels and exit.",
    )
    args = parser.parse_args()

    if args.list_cases:
        for case in CASE_MATRIX:
            print(case["label"])
        return

    cases = selected_cases(args.cases)
    observed = run_all(cases)

    if args.write_expected:
        EXPECTED_PATH.write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote expected results: {EXPECTED_PATH}")
        return

    if not EXPECTED_PATH.exists():
        raise FileNotFoundError(
            f"Missing expected results file: {EXPECTED_PATH}. "
            "Run with --write-expected after reviewing the current output."
        )

    expected = json.loads(EXPECTED_PATH.read_text())
    if args.cases:
        wanted = {case["label"] for case in cases}
        expected["cases"] = [
            record for record in expected["cases"] if record["case"] in wanted
        ]

    compare_results(observed, expected)
    print("\nInformal UCCN-VQE FCIComputer check passed.")


if __name__ == "__main__":
    main()
