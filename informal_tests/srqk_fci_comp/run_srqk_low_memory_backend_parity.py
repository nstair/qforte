"""Informal SRQK low-memory backend parity check.

Run from the repository root with qfe_env_v1:

    conda run -n qfe_env_v1 python informal_tests/srqk_fci_comp/run_srqk_low_memory_backend_parity.py

This is intentionally not a pytest test.  It reuses the saved H4/STO-3G
FCIComputer reference data from run_srqk_fci_comp.py and checks that
low-memory SRQK matrix construction agrees for each available backend.
Unavailable optional CUDA/CUSV/FQE backends are skipped with clear output.
"""

from __future__ import annotations

import argparse
import contextlib
import traceback

import srqk_informal_common as common


def run_backend_cases(cases, backend, expected_cases):
    available, reason = common.backend_available(backend)
    if not available:
        print(f"\n==> backend={backend} low_memory=True <==")
        print(f"  status: SKIP all cases")
        print(f"  reason: {reason}")
        return [], [{"backend": backend, "case": "*", "reason": reason}]

    log_root = common.LOW_MEMORY_LOG_DIR / backend
    log_root.mkdir(parents=True, exist_ok=True)

    build_log = log_root / "build_h4_sto3g.log"
    with build_log.open("w") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            mol = common.build_h4()

    print(f"\n==> backend={backend} low_memory=True <==")
    print(f"  build log: {build_log}")
    print(f"  FCI reference energy: {mol.fci_energy:+18.12f}")

    passed = []
    skipped = []
    for index, case in enumerate(cases, start=1):
        log_path = log_root / f"{backend}_lowmem_{case['label']}.log"
        print(f"\n[{index:02d}/{len(cases):02d}] {case['label']} -> {log_path}")
        try:
            with log_path.open("w") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    try:
                        alg = common.run_srqk_case(
                            mol,
                            case,
                            backend=backend,
                            low_memory=True,
                        )
                    except Exception:
                        traceback.print_exc(file=log)
                        raise
            record = common.result_record(
                case,
                mol,
                alg,
                backend=backend,
                low_memory=True,
            )
        except Exception as exc:
            if common.should_skip_runtime_error(backend, exc):
                reason = f"{type(exc).__name__}: {exc}"
                print(f"  status: SKIP low-memory {backend}.{case['label']}")
                print(f"  reason: {reason}")
                skipped.append({"backend": backend, "case": case["label"], "reason": reason})
                continue
            raise

        common.compare_records(
            record,
            expected_cases[case["label"]],
            common.PARITY_TOLERANCES,
            f"low_memory.{backend}.{case['label']}",
        )
        passed.append({"backend": backend, "case": case["label"]})

    return passed, skipped


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
        for backend in common.BACKENDS:
            print(backend)
        return

    if args.list_cases:
        for case in common.case_matrix():
            print(case["label"])
        return

    cases = common.selected_cases(args.cases)
    backends = common.selected_backends(args.backends)
    expected_cases = common.load_expected(cases)

    print("\n==> Informal SRQK low-memory backend parity check <==")
    print(f"  expected file: {common.EXPECTED_PATH}")
    print(f"  backends:      {', '.join(backends)}")
    common.print_molecule_header()

    passed = []
    skipped = []
    for backend in backends:
        backend_passed, backend_skipped = run_backend_cases(
            cases,
            backend,
            expected_cases,
        )
        passed.extend(backend_passed)
        skipped.extend(backend_skipped)

    print(f"\nLow-memory backend parity passed for {len(passed)} run(s).")
    if skipped:
        print(f"Skipped {len(skipped)} run(s):")
        for item in skipped:
            print(f"  - {item['backend']}.{item['case']}: {item['reason']}")


if __name__ == "__main__":
    main()
