"""Generate qforte-readable PySCF dump files used by the informal checks.

Run from the repository root with qfe_env_v1:

    conda run -n qfe_env_v1 python informal_tests/pyscf_fci_dump_api/run_generate_pyscf_dumps.py

Existing dump files are reused by default.  Use ``--overwrite`` only when you
intend to refresh the saved dump data after reviewing the resulting output.
"""

from __future__ import annotations

import argparse

import pyscf_dump_informal_common as common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate dumps even when the .npz files already exist.",
    )
    parser.add_argument(
        "--skip-benzene-casci",
        action="store_true",
        help="Write the benzene AVAS dump without the optional CASCI reference energy.",
    )
    args = parser.parse_args()

    print("\n==> Informal PySCF dump generation <==")
    print(f"  dump directory: {common.DUMP_DIR}")
    print(f"  overwrite:      {args.overwrite}")

    results = common.ensure_dumps(
        overwrite=args.overwrite,
        run_benzene_casci=not args.skip_benzene_casci,
    )
    common.print_dump_summary(results)


if __name__ == "__main__":
    main()
