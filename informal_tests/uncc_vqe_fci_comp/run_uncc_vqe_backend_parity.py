"""Informal UCCN-VQE backend parity check against saved FCIComputer data.

Run from the repository root with qfe_env_v1:

    conda run -n qfe_env_v1 python informal_tests/uncc_vqe_fci_comp/run_uncc_vqe_backend_parity.py

This is intentionally not a pytest test.  It reuses the H4/C1 and H4/D2h
cases in run_uncc_vqe_fci_comp.py, runs them with alternate computer backends,
and compares the final observables against expected_uncc_vqe_fci_comp.json.
Unavailable optional backends are skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import numpy as np
import qforte as qf

import run_uncc_vqe_fci_comp as fci_ref


BACKENDS = ("fock", "fqe", "fci_gpu", "cusv")
OPTIONAL_RUNTIME_SKIP_BACKENDS = {"fci_gpu", "cusv"}
LOG_DIR = Path(__file__).resolve().parent / "backend_logs"

PARITY_TOLERANCES = {
    "energy": 1.0e-6,
    "spin_squared": 1.0e-6,
    "noons": 1.0e-6,
    "amplitudes": 1.0e-6,
    "gradient": 1.0e-6,
    "gradient_norm": 1.0e-6,
}


def backend_available(backend: str) -> tuple[bool, str | None]:
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

    raise ValueError(f"Unknown backend {backend!r}. Expected one of {BACKENDS}.")


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


def apply_ham_as_tensor(backend: str) -> bool:
    return backend == "fci_gpu"


def backend_run_options(case: dict[str, Any], backend: str) -> dict[str, Any]:
    return fci_ref.make_run_options(case)


def requests_unsupported_analytic_hdiag(case: dict[str, Any], backend: str) -> bool:
    if backend == "fci":
        return False

    options = backend_run_options(case, backend)
    optimizer = case["optimizer"].lower()
    if optimizer not in {"bfgs_qf", "lbfgs_qf"}:
        return False

    use_key = f"{optimizer}_use_hessian_diag"
    method_key = f"{optimizer}_hdiag_method"
    return bool(options.get(use_key)) and options.get(method_key) in {
        "analytic",
        "recursive",
    }


def analytical_cnot_count(alg) -> int:
    if hasattr(alg, "count_jw_cnot_ladders"):
        return int(alg.count_jw_cnot_ladders())
    return int(alg._n_cnot)


def result_record(case: dict[str, Any], mol, alg, backend: str) -> dict[str, Any]:
    record = fci_ref.result_record(case, mol, alg)
    record["backend"] = backend
    record["run_options"] = backend_run_options(case, backend)
    record["cnot_count"] = analytical_cnot_count(alg)
    return record


def run_case(case: dict[str, Any], systems: dict[str, Any], backend: str) -> dict[str, Any]:
    mol = systems[case["symmetry"]]
    alg = qf.UCCNVQE(
        mol,
        computer_type=backend,
        apply_ham_as_tensor=apply_ham_as_tensor(backend),
        verbose=False,
        print_summary_file=False,
    )
    alg.run(**backend_run_options(case, backend))
    return result_record(case, mol, alg, backend)


def compare_record(backend: str, observed: dict[str, Any], expected: dict[str, Any]):
    failures = []
    prefix = f"{backend}.{observed['case']}"

    for key in ["symmetry", "pool_type", "optimizer", "n_pool", "n_params"]:
        if observed.get(key) != expected.get(key):
            failures.append(f"{prefix}.{key}: {observed.get(key)!r} != {expected.get(key)!r}")

    for key in ["pool_indices"]:
        if observed.get(key) != expected.get(key):
            failures.append(f"{prefix}.{key}: {observed.get(key)!r} != {expected.get(key)!r}")

    if observed.get("cnot_count") != expected.get("cnot_count"):
        failures.append(
            f"{prefix}.cnot_count: {observed.get('cnot_count')} != {expected.get('cnot_count')}"
        )

    fci_ref.compare_scalar(
        f"{prefix}.energy",
        observed["energy"],
        expected["energy"],
        PARITY_TOLERANCES["energy"],
        failures,
    )
    fci_ref.compare_scalar(
        f"{prefix}.spin_squared",
        observed["spin_squared"],
        expected["spin_squared"],
        PARITY_TOLERANCES["spin_squared"],
        failures,
    )
    fci_ref.compare_scalar(
        f"{prefix}.gradient_norm",
        observed["gradient_norm"],
        expected["gradient_norm"],
        PARITY_TOLERANCES["gradient_norm"],
        failures,
    )
    fci_ref.compare_sequence(
        f"{prefix}.noons",
        observed["noons"],
        expected["noons"],
        PARITY_TOLERANCES["noons"],
        failures,
    )
    fci_ref.compare_sequence(
        f"{prefix}.amplitudes",
        observed["amplitudes"],
        expected["amplitudes"],
        PARITY_TOLERANCES["amplitudes"],
        failures,
    )
    fci_ref.compare_sequence(
        f"{prefix}.gradient",
        observed["gradient"],
        expected["gradient"],
        PARITY_TOLERANCES["gradient"],
        failures,
    )

    if failures:
        message = "\n".join(f"  - {failure}" for failure in failures)
        raise AssertionError(f"Informal UCCN-VQE backend parity failed:\n{message}")


def load_expected(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not fci_ref.EXPECTED_PATH.exists():
        raise FileNotFoundError(
            f"Missing expected results file: {fci_ref.EXPECTED_PATH}. "
            "Run run_uncc_vqe_fci_comp.py --write-expected after reviewing the FCI data."
        )

    expected = json.loads(fci_ref.EXPECTED_PATH.read_text())
    expected_cases = {record["case"]: record for record in expected["cases"]}
    wanted = {case["label"] for case in cases}
    missing = sorted(wanted - set(expected_cases))
    if missing:
        raise ValueError(f"Expected data is missing case label(s): {missing}")
    return {label: expected_cases[label] for label in sorted(wanted)}


def build_systems(cases: list[dict[str, Any]]) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    systems = {}
    for symmetry in sorted({case["symmetry"] for case in cases}):
        log_path = LOG_DIR / f"build_h4_{symmetry}.log"
        with log_path.open("w") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                systems[symmetry] = fci_ref.build_h4(symmetry)
    return systems


def run_all(cases: list[dict[str, Any]], backends: list[str]):
    expected_cases = load_expected(cases)
    systems = build_systems(cases)
    skipped = []
    expected_unsupported = []
    passed = []

    for backend in backends:
        available, reason = backend_available(backend)
        if not available:
            print(f"[skip] {backend}: {reason}")
            skipped.append({"backend": backend, "case": "*", "reason": reason})
            continue

        backend_log_dir = LOG_DIR / backend
        backend_log_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n==> backend={backend}")

        for index, case in enumerate(cases, start=1):
            label = case["label"]
            log_path = backend_log_dir / f"{label}.log"
            print(f"[{index:02d}/{len(cases):02d}] {label} -> {log_path}")

            try:
                with log_path.open("w") as log:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        try:
                            record = run_case(case, systems, backend)
                        except Exception:
                            traceback.print_exc(file=log)
                            raise
            except Exception as exc:
                if requests_unsupported_analytic_hdiag(case, backend) and isinstance(
                    exc, NotImplementedError
                ):
                    reason = (
                        "analytical Hessian diagonals are intentionally FCI-only "
                        f"for now ({type(exc).__name__}: {exc})"
                    )
                    print(f"  [expected-unsupported] {backend}.{label}: {reason}")
                    expected_unsupported.append(
                        {"backend": backend, "case": label, "reason": reason}
                    )
                    continue
                if should_skip_runtime_error(backend, exc):
                    reason = f"{type(exc).__name__}: {exc}"
                    print(f"  [skip] {backend}.{label}: {reason}")
                    skipped.append({"backend": backend, "case": label, "reason": reason})
                    continue
                raise

            compare_record(backend, record, expected_cases[label])
            passed.append({"backend": backend, "case": label})

    required_backends = {backend for backend in backends if backend in {"fock"}}
    for backend in required_backends:
        if not any(item["backend"] == backend for item in passed):
            raise AssertionError(f"Required backend {backend!r} did not complete any cases.")

    return passed, skipped, expected_unsupported


def selected_backends(values: list[str] | None) -> list[str]:
    if not values:
        return list(BACKENDS)
    unknown = sorted(set(values) - set(BACKENDS))
    if unknown:
        raise ValueError(f"Unknown backend(s): {unknown}. Expected one of {BACKENDS}.")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        help="Run one backend. May be supplied more than once.",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Run only one named case. May be supplied more than once.",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="Print backend strings and exit.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print case labels and exit.",
    )
    args = parser.parse_args()

    if args.list_backends:
        for backend in BACKENDS:
            print(backend)
        return

    if args.list_cases:
        for case in fci_ref.CASE_MATRIX:
            print(case["label"])
        return

    cases = fci_ref.selected_cases(args.cases)
    backends = selected_backends(args.backends)
    passed, skipped, expected_unsupported = run_all(cases, backends)

    print(f"\nBackend parity passed for {len(passed)} run(s).")
    if expected_unsupported:
        print(f"Expected unsupported analytical-hdiag run(s): {len(expected_unsupported)}")
        for item in expected_unsupported:
            print(f"  - {item['backend']}.{item['case']}: {item['reason']}")
    if skipped:
        print(f"Skipped {len(skipped)} run(s):")
        for item in skipped:
            print(f"  - {item['backend']}.{item['case']}: {item['reason']}")


if __name__ == "__main__":
    main()
