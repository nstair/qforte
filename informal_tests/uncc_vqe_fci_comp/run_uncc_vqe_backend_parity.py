"""Informal UCCN-VQE backend parity check against saved FCIComputer data.

Run from the repository root with qfe_env_v1:

    conda run -n qfe_env_v1 python informal_tests/uncc_vqe_fci_comp/run_uncc_vqe_backend_parity.py

This is intentionally not a pytest test.  It reuses the HF/C1 and HF/C2v
cases in run_uncc_vqe_fci_comp.py, runs them with alternate computer backends,
and compares the final observables against expected_uncc_vqe_fci_comp.json.
Unavailable optional backends are skipped.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import numpy as np
import qforte as qf

import run_uncc_vqe_fci_comp as fci_ref


BACKENDS = ("fock", "fqe", "fci_gpu", "cusv")
OPTIONAL_RUNTIME_SKIP_BACKENDS = {"fqe", "fci_gpu", "cusv"}
LOG_DIR = Path(__file__).resolve().parent / "backend_logs"
WORKER_PREFIX = "UNCC_PARITY_WORKER_RESULT="

PARITY_TOLERANCES = {
    "energy": 1.0e-6,
    "spin_squared": 1.0e-6,
    "noons": 1.0e-6,
    "amplitudes": 1.0e-6,
    "gradient": 1.0e-6,
    "gradient_norm": 1.0e-6,
}

DISPLAY_BACKEND_ORDER = ("fci", "fci_gpu", "fock", "fqe", "cusv")
BACKEND_LABELS = {
    "fci": "FCI",
    "fci_gpu": "FCI_GPU",
    "fock": "FOCK",
    "fqe": "FQE",
    "cusv": "CUSV",
}
OPTIMIZER_LABELS = {
    "jacobi": "jacobi",
    "bfgs_qf": "qf_bfgs",
    "lbfgs_qf": "qf_lbfgs",
}
ANSATZ_LABELS = {
    "SD": "SD",
    "GSD": "GSD",
    "1-UpCCGSD": "1-Up",
    "2-UpCCGSD": "2-Up",
}
METRIC_COLUMNS = (
    ("energy", "dE", "energy_diff"),
    ("spin_squared", "dS^2", "spin_diff"),
    ("gradient_norm", "d|g|", "gradient_norm_diff"),
    ("noons", "dNOON", "noons_diff"),
    ("amplitudes", "dAmp", "amplitudes_diff"),
    ("gradient", "dGrad", "gradient_diff"),
    ("cnot_count", "dCNOT", "cnot_delta"),
)
METRIC_DIFF_KEYS = {metric_key: diff_key for metric_key, _, diff_key in METRIC_COLUMNS}
METRIC_TOLERANCE_KEYS = {
    "energy": "energy",
    "spin_squared": "spin_squared",
    "gradient_norm": "gradient_norm",
    "noons": "noons",
    "amplitudes": "amplitudes",
    "gradient": "gradient",
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
    if getattr(alg, "_computer_type", None) == "fci_gpu":
        cpu_pool = qf.SQOpPool()
        for coeff, sq_op in alg._pool_obj.terms():
            cpu_pool.add_term(coeff, sq_op)
        n_cnot = alg._Uprep.get_num_cnots()
        qubit_excitations = getattr(alg, "_qubit_excitations", False)
        trotter_number = getattr(alg, "_trotter_number", 1)
        for top in alg._tops:
            n_cnot += cpu_pool.count_cnot_for_term_jw_exponential(
                int(top),
                qubit_excitations,
                trotter_number,
            )
        return int(n_cnot)

    if hasattr(alg, "count_jw_cnot_ladders"):
        try:
            return int(alg.count_jw_cnot_ladders())
        except Exception:
            pass
    return int(alg._n_cnot)


def result_record(case: dict[str, Any], mol, alg, backend: str) -> dict[str, Any]:
    record = fci_ref.result_record(case, mol, alg)
    record["backend"] = backend
    record["run_options"] = backend_run_options(case, backend)
    record["cnot_count"] = analytical_cnot_count(alg)
    return record


def run_case(case: dict[str, Any], systems: dict[str, Any], backend: str) -> dict[str, Any]:
    mol = systems[case["symmetry"]]
    if backend == "fci_gpu":
        # GPU runs mutate the tensor residency on the molecule object, so reuse
        # across cases can poison subsequent constructor calls.
        mol = fci_ref.build_hf(case["symmetry"])
    alg = qf.UCCNVQE(
        mol,
        computer_type=backend,
        apply_ham_as_tensor=apply_ham_as_tensor(backend),
        verbose=False,
        print_summary_file=False,
    )
    if backend == "fci_gpu":
        # The GPU SQOp pool does not expose the CPU-side ladder-count helper that
        # jacobi_solver() tries to call at the end of each iteration. Let the run
        # complete and recompute the count from a CPU mirror afterward.
        alg.count_jw_cnot_ladders = lambda: int(getattr(alg, "_n_cnot", 0))
    alg.run(**backend_run_options(case, backend))
    return result_record(case, mol, alg, backend)


def scalar_diff(observed: Any, expected: Any):
    if observed is None or expected is None:
        if observed == expected:
            return None
        return f"{observed!r} != {expected!r}"
    return abs(float(observed) - float(expected))


def sequence_diff(observed: Any, expected: Any):
    if observed is None or expected is None:
        if observed == expected:
            return None
        return f"{observed!r} != {expected!r}"
    if len(observed) != len(expected):
        return f"len {len(observed)} != {len(expected)}"
    if not observed:
        return 0.0
    return max(abs(float(a) - float(b)) for a, b in zip(observed, expected))


def compact_reason(text: str, limit: int = 78) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def evaluate_record(backend: str, observed: dict[str, Any], expected: dict[str, Any]):
    failures = []
    prefix = f"{backend}.{observed['case']}"
    unsupported_metrics = set()
    if backend == "fci_gpu" and observed.get("spin_squared") is None:
        unsupported_metrics.add("spin_squared")
    if backend == "fci_gpu" and observed.get("noons") is None:
        unsupported_metrics.add("noons")

    metrics = {
        "energy": observed.get("energy"),
        "energy_diff": scalar_diff(observed.get("energy"), expected.get("energy")),
        "spin_diff": (
            None
            if "spin_squared" in unsupported_metrics
            else scalar_diff(observed.get("spin_squared"), expected.get("spin_squared"))
        ),
        "gradient_norm_diff": scalar_diff(
            observed.get("gradient_norm"),
            expected.get("gradient_norm"),
        ),
        "noons_diff": (
            None
            if "noons" in unsupported_metrics
            else sequence_diff(observed.get("noons"), expected.get("noons"))
        ),
        "amplitudes_diff": sequence_diff(
            observed.get("amplitudes"),
            expected.get("amplitudes"),
        ),
        "gradient_diff": sequence_diff(observed.get("gradient"), expected.get("gradient")),
        "cnot_delta": (
            None
            if observed.get("cnot_count") is None or expected.get("cnot_count") is None
            else int(observed.get("cnot_count")) - int(expected.get("cnot_count"))
        ),
        "unsupported_metrics": sorted(unsupported_metrics),
    }

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
    if not (backend == "fci_gpu" and observed.get("spin_squared") is None):
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
    if not (backend == "fci_gpu" and observed.get("noons") is None):
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

    return metrics, failures


def system_label(case: dict[str, Any]) -> str:
    return f"HF/{case['symmetry'].upper()}"


def ansatz_label(pool_type: str) -> str:
    return ANSATZ_LABELS.get(pool_type, pool_type)


def optimizer_label(name: str) -> str:
    return OPTIMIZER_LABELS.get(name, name)


def backend_label(name: str) -> str:
    return BACKEND_LABELS.get(name, name.upper())


def metric_short_status(item: dict[str, Any], metric_key: str) -> str:
    if item.get("backend") == "fci":
        return "PS"

    if metric_key in set(item.get("unsupported_metrics", [])):
        return "US"

    if item["status"] in {"SKIP", "UNSUPPORTED"}:
        return "US"

    diff = item.get(METRIC_DIFF_KEYS[metric_key])
    if diff is None:
        return "FL" if item["status"] == "FAIL" else "PS"
    if isinstance(diff, str):
        return "FL"
    if metric_key == "cnot_count":
        return "PS" if int(diff) == 0 else "FL"

    tol_key = METRIC_TOLERANCE_KEYS[metric_key]
    return "PS" if float(diff) <= PARITY_TOLERANCES[tol_key] else "FL"


def metric_display_text(item: dict[str, Any], metric_key: str) -> str:
    status = metric_short_status(item, metric_key)
    diff = item.get(METRIC_DIFF_KEYS[metric_key])

    if status == "US":
        return "US n/a"
    if diff is None or isinstance(diff, str):
        return f"{status} n/a"
    if metric_key == "cnot_count":
        return f"{status} {int(diff):+d}"
    return f"{status} {float(diff):.3e}"


def table_note(item: dict[str, Any]) -> str:
    note = item.get("note", "")
    if note == "matched FCI reference":
        return "matched reference"
    if note == "analytical Hessian diagonals are intentionally FCI-only for now":
        return "analytic hdiag is FCI-only"
    if "dependency is not installed" in note:
        return "dependency unavailable"
    if "could not inspect" in note:
        return "availability check failed"
    if "was not compiled" in note or "is not available" in note:
        return "backend unavailable"
    return compact_reason(note, limit=40)


def reference_display_entry(case: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": "fci",
        "case": case["label"],
        "status": "PASS",
        "energy": expected.get("energy"),
        "energy_diff": 0.0,
        "spin_diff": 0.0,
        "gradient_norm_diff": 0.0,
        "noons_diff": 0.0,
        "amplitudes_diff": 0.0,
        "gradient_diff": 0.0,
        "cnot_delta": 0,
        "unsupported_metrics": [],
        "note": "reference",
        "log_path": "-",
        "failures": [],
    }


def display_backend_order(backends: list[str]) -> list[str]:
    requested = {"fci", *backends}
    ordered = [backend for backend in DISPLAY_BACKEND_ORDER if backend in requested]
    extras = [backend for backend in backends if backend not in ordered]
    return ordered + extras


def render_text_table(headers: list[str], rows: list[list[str]]):
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    print(separator)
    print(
        "| "
        + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
        + " |"
    )
    print(separator)
    for row in rows:
        print(
            "| "
            + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
            + " |"
        )
    print(separator)


def build_display_rows(
    cases: list[dict[str, Any]],
    expected_cases: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    backends: list[str],
) -> list[list[str]]:
    result_map = {(item["backend"], item["case"]): item for item in results}
    rows = []

    for case in cases:
        label = case["label"]
        entries = [reference_display_entry(case, expected_cases[label])]
        for backend in display_backend_order(backends):
            if backend == "fci":
                continue
            entry = result_map.get((backend, label))
            if entry is not None:
                entries.append(entry)

        for entry in entries:
            rows.append(
                [
                    system_label(case),
                    ansatz_label(case["pool_type"]),
                    optimizer_label(case["optimizer"]),
                    backend_label(entry["backend"]),
                    metric_display_text(entry, "energy"),
                    metric_display_text(entry, "spin_squared"),
                    metric_display_text(entry, "gradient_norm"),
                    metric_display_text(entry, "noons"),
                    metric_display_text(entry, "amplitudes"),
                    metric_display_text(entry, "gradient"),
                    metric_display_text(entry, "cnot_count"),
                    table_note(entry),
                ]
            )

    return rows


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
        log_path = LOG_DIR / f"build_hf_{symmetry}.log"
        with log_path.open("w") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                systems[symmetry] = fci_ref.build_hf(symmetry)
    return systems


def worker_payload_for_exception(exc: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "exc_type": type(exc).__name__,
        "exc_message": str(exc),
    }


def parse_worker_payload(stdout: str) -> dict[str, Any]:
    for line in stdout.splitlines():
        if line.startswith(WORKER_PREFIX):
            return json.loads(line[len(WORKER_PREFIX) :])
    raise RuntimeError(
        "Backend worker did not return a result payload. "
        f"Captured stdout was:\n{stdout}"
    )


def raise_worker_exception(payload: dict[str, Any]):
    exc_name = payload.get("exc_type", "RuntimeError")
    exc_message = payload.get("exc_message", "")
    exc_map = {
        "AssertionError": AssertionError,
        "AttributeError": AttributeError,
        "ImportError": ImportError,
        "ModuleNotFoundError": ModuleNotFoundError,
        "NotImplementedError": NotImplementedError,
        "RuntimeError": RuntimeError,
        "TypeError": TypeError,
        "ValueError": ValueError,
    }
    raise exc_map.get(exc_name, RuntimeError)(exc_message)


def run_case_isolated(case: dict[str, Any], backend: str, log_path: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-backend",
        backend,
        "--worker-case",
        case["label"],
        "--worker-log",
        str(log_path),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent),
    )
    captured = "\n".join(part for part in [proc.stdout, proc.stderr] if part)
    try:
        payload = parse_worker_payload(captured)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc}\nWorker log: {log_path}"
        ) from None
    if payload.get("ok"):
        return payload["record"]
    raise_worker_exception(payload)


def worker_main(case_label: str, backend: str, log_path: Path):
    case = fci_ref.selected_cases([case_label])[0]
    systems = {}
    payload = None
    with log_path.open("w") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                systems[case["symmetry"]] = fci_ref.build_hf(case["symmetry"])
                record = run_case(case, systems, backend)
                payload = {"ok": True, "record": record}
            except Exception as exc:
                traceback.print_exc(file=log)
                payload = worker_payload_for_exception(exc)
    print(WORKER_PREFIX + json.dumps(payload), flush=True)


def print_reference_summary(systems: dict[str, Any]):
    print("\n==> Reference systems <==")
    for symmetry in sorted(systems):
        mol = systems[symmetry]
        print(
            f"  {symmetry:<4s} HF = {mol.hf_energy:+18.12f} "
            f"MP2 = {mol.mp2_energy:+18.12f} "
            f"FCI = {mol.fci_energy:+18.12f}"
        )


def print_backend_summary(results: list[dict[str, Any]], backends: list[str]):
    print("\n==> Backend summary <==")
    header = (
        f"{'backend':10s} {'status':>10s} {'pass':>6s} {'fail':>6s} "
        f"{'skip':>6s} {'unsup':>6s} {'ran':>6s}"
    )
    print(header)
    print("-" * len(header))
    for backend in backends:
        subset = [item for item in results if item["backend"] == backend]
        counts = collections.Counter(item["status"] for item in subset)
        ran = counts["PASS"] + counts["FAIL"]
        if counts["FAIL"]:
            overall = "FAIL"
        elif ran and (counts["SKIP"] or counts["UNSUPPORTED"]):
            overall = "PARTIAL"
        elif ran:
            overall = "PASS"
        else:
            overall = "SKIP"
        print(
            f"{backend:10s} {overall:>10s} {counts['PASS']:6d} {counts['FAIL']:6d} "
            f"{counts['SKIP']:6d} {counts['UNSUPPORTED']:6d} {ran:6d}"
        )


def print_parity_table(
    cases: list[dict[str, Any]],
    expected_cases: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    backends: list[str],
):
    print("\n==> Backend parity matrix <==")
    print("Legend: each check cell is 'status diff' with PS=pass, FL=fail, US=unsupported/unavailable.")
    headers = [
        "System",
        "Ansatz",
        "Optimizer",
        "Backend",
        "dE",
        "dS^2",
        "d|g|",
        "dNOON",
        "dAmp",
        "dGrad",
        "dCNOT",
        "Note",
    ]
    render_text_table(headers, build_display_rows(cases, expected_cases, results, backends))


def print_issue_details(results: list[dict[str, Any]]):
    failures = [item for item in results if item["status"] == "FAIL"]
    blocked = [item for item in results if item["status"] in {"SKIP", "UNSUPPORTED"}]
    if not failures and not blocked:
        return
    print("\n==> Notes <==")
    for item in failures:
        print(f"  - {item['backend']}.{item['case']} [FAIL]")
        print(f"    log: {item['log_path']}")
        print(f"    note: {item['note']}")
        for failure in item.get("failures", []):
            print(f"    * {failure}")
    if blocked:
        grouped = collections.Counter((item["status"], item["note"]) for item in blocked)
        for (status, note), count in sorted(grouped.items()):
            print(f"  - {status:<11s} x{count:2d}: {compact_reason(note, limit=120)}")


def run_all(
    cases: list[dict[str, Any]],
    backends: list[str],
    expected_cases: dict[str, dict[str, Any]],
):
    systems = build_systems(cases)
    print_reference_summary(systems)
    results = []

    for backend in backends:
        available, reason = backend_available(backend)
        if not available:
            print(f"\n==> backend={backend} <==")
            print(f"  status: SKIP all cases")
            print(f"  reason: {reason}")
            for case in cases:
                results.append(
                    {
                        "backend": backend,
                        "case": case["label"],
                        "status": "SKIP",
                        "energy": None,
                        "energy_diff": None,
                        "amplitudes_diff": None,
                        "gradient_diff": None,
                        "gradient_norm_diff": None,
                        "noons_diff": None,
                        "spin_diff": None,
                        "cnot_delta": None,
                        "unsupported_metrics": [],
                        "note": reason,
                        "log_path": "-",
                        "failures": [],
                    }
                )
            continue

        backend_log_dir = LOG_DIR / backend
        backend_log_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n==> backend={backend} <==")

        for index, case in enumerate(cases, start=1):
            label = case["label"]
            log_path = backend_log_dir / f"{label}.log"
            print(f"[{index:02d}/{len(cases):02d}] {label} -> {log_path}")
            entry = {
                "backend": backend,
                "case": label,
                "status": "FAIL",
                "energy": None,
                "energy_diff": None,
                "amplitudes_diff": None,
                "gradient_diff": None,
                "gradient_norm_diff": None,
                "noons_diff": None,
                "spin_diff": None,
                "cnot_delta": None,
                "unsupported_metrics": [],
                "note": "",
                "log_path": str(log_path),
                "failures": [],
            }

            if requests_unsupported_analytic_hdiag(case, backend):
                reason = (
                    "analytical Hessian diagonals are intentionally FCI-only "
                    "for now"
                )
                print(f"  status: UNSUPPORTED {backend}.{label}")
                print(f"  reason: {reason}")
                entry["status"] = "UNSUPPORTED"
                entry["note"] = reason
                results.append(entry)
                continue

            try:
                if backend == "fci_gpu":
                    record = run_case_isolated(case, backend, log_path)
                else:
                    with log_path.open("w") as log:
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            try:
                                record = run_case(case, systems, backend)
                            except Exception:
                                traceback.print_exc(file=log)
                                raise
            except Exception as exc:
                if should_skip_runtime_error(backend, exc):
                    reason = f"{type(exc).__name__}: {exc}"
                    print(f"  status: SKIP {backend}.{label}")
                    print(f"  reason: {reason}")
                    entry["status"] = "SKIP"
                    entry["note"] = reason
                    results.append(entry)
                    continue
                reason = f"{type(exc).__name__}: {exc}"
                print(f"  status: FAIL {backend}.{label}")
                print(f"  reason: {reason}")
                entry["status"] = "FAIL"
                entry["note"] = reason
                entry["failures"] = [reason]
                results.append(entry)
                continue

            metrics, failures = evaluate_record(backend, record, expected_cases[label])
            entry.update(metrics)
            if failures:
                print(f"  status: FAIL parity mismatch")
                print(f"  reason: {compact_reason(failures[0])}")
                entry["status"] = "FAIL"
                entry["note"] = f"{len(failures)} mismatch(es)"
                entry["failures"] = failures
            else:
                print(f"  status: PASS energy = {record['energy']:+18.12f}")
                entry["status"] = "PASS"
                entry["note"] = "matched FCI reference"
            results.append(entry)

    required_backends = {backend for backend in backends if backend in {"fock"}}
    for backend in required_backends:
        if not any(
            not requests_unsupported_analytic_hdiag(case, backend) for case in cases
        ):
            continue
        if not any(
            item["backend"] == backend and item["status"] in {"PASS", "FAIL"}
            for item in results
        ):
            raise AssertionError(f"Required backend {backend!r} did not complete any cases.")

    return results


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
    parser.add_argument(
        "--worker-backend",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-case",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-log",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.worker_backend or args.worker_case or args.worker_log:
        if not (args.worker_backend and args.worker_case and args.worker_log):
            raise ValueError(
                "--worker-backend, --worker-case, and --worker-log must be supplied together."
            )
        worker_main(args.worker_case, args.worker_backend, Path(args.worker_log))
        return

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
    expected_cases = load_expected(cases)
    results = run_all(cases, backends, expected_cases)

    print_backend_summary(results, backends)
    print_parity_table(cases, expected_cases, results, backends)
    print_issue_details(results)

    failures = [item for item in results if item["status"] == "FAIL"]
    skipped = [item for item in results if item["status"] == "SKIP"]
    unsupported = [item for item in results if item["status"] == "UNSUPPORTED"]
    passed = [item for item in results if item["status"] == "PASS"]

    print(f"\nCompleted {len(passed)} passing backend parity run(s).")
    if unsupported:
        print(f"Marked {len(unsupported)} run(s) as intentionally unsupported.")
    if skipped:
        print(f"Skipped {len(skipped)} run(s).")
    if failures:
        details = "\n".join(
            f"  - {item['backend']}.{item['case']}: {item['note']}"
            for item in failures
        )
        raise AssertionError(f"Informal UCCN-VQE backend parity failed:\n{details}")


if __name__ == "__main__":
    main()
