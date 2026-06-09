"""Informal SRQK FCIComputer regression harness.

Run from the repository root with qfe_env_v1:

    conda run -n qfe_env_v1 python informal_tests/srqk_fci_comp/run_srqk_fci_comp.py

To intentionally refresh the reference data after reviewing expected changes:

    conda run -n qfe_env_v1 python informal_tests/srqk_fci_comp/run_srqk_fci_comp.py --write-expected

This is intentionally not a pytest test.  It keeps compact reference data for
linear-H4/STO-3G SRQK FCIComputer runs in both c1 and d2h symmetry, covering
several QK time-grid/Trotter schedules
for first- and second-order Trotterization.
"""

from __future__ import annotations

import argparse
import json

import srqk_informal_common as common


def compare_to_expected(observed, cases):
    expected_cases = common.load_expected(cases)
    observed_cases = {record["case"]: record for record in observed["cases"]}
    for case in cases:
        label = case["label"]
        common.compare_records(
            observed_cases[label],
            expected_cases[label],
            common.TOLERANCES,
            label,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-expected",
        "--save-results",
        action="store_true",
        help="Write observed SRQK FCIComputer results to the expected JSON file.",
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
        for case in common.case_matrix():
            print(case["label"])
        return

    cases = common.selected_cases(args.cases)

    print("\n==> Informal SRQK FCIComputer check <==")
    print(f"  s:                 {common.S}")
    print(f"  dt:                {common.DT}")
    print(f"  trotter orders:    {common.TROTTER_ORDERS}")
    print(f"  expected file:     {common.EXPECTED_PATH}")

    observed, skipped = common.run_all_cases(
        cases,
        backend="fci",
        low_memory=False,
        log_root=common.LOG_DIR,
    )

    if skipped:
        print("\nUnexpected skipped/error cases:")
        for item in skipped:
            print(f"  - {item['backend']}.{item['case']}: {item['reason']}")
        raise AssertionError("FCIComputer SRQK reference runs should not be skipped.")

    systems = {
        record["system"]["symmetry"]: type("SystemInfo", (), {"fci_energy": record["system"]["fci_energy"]})()
        for record in observed["cases"]
    }
    common.print_case_summary(observed["cases"], systems)

    if args.write_expected:
        common.write_json(common.EXPECTED_PATH, observed)
        print(f"\nWrote expected SRQK results: {common.EXPECTED_PATH}")
        return

    compare_to_expected(observed, cases)
    print("\nInformal SRQK FCIComputer check passed.")


if __name__ == "__main__":
    main()
