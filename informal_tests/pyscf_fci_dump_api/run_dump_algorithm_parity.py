"""Informal algorithm parity check across molecule-build paths.

The check runs compact UCCSD-VQE and SRQK calculations from molecules built
directly with PySCF, from Psi4 when available, and from PySCF dump files.
BeH2 compares PySCF/Psi4/PySCF-dump.  Benzene compares PySCF/PySCF-dump.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

import pyscf_dump_informal_common as common


THIS_DIR = common.THIS_DIR


def build_available_molecules(system, build_types, log_root):
    molecules = {}
    for build_type in build_types:
        mol, log_path, error = common.maybe_build_molecule(system, build_type, log_root=log_root)
        print(f"\n  system={system:<8s} build_type={build_type:<10s} log={log_path}")
        if error:
            print(f"  status: SKIP build failed: {error}")
            continue
        molecules[build_type] = mol
        print("  status: PASS build completed")
    return molecules


def restore_json_arrays(value):
    if isinstance(value, dict):
        if value.get("__complex_ndarray__"):
            return np.asarray(value["real"]) + 1.0j * np.asarray(value["imag"])
        if value.get("__complex__"):
            return complex(value["real"], value["imag"])
        return {key: restore_json_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return [restore_json_arrays(item) for item in value]
    return value


def run_algorithm_child(algorithm, system, build_type, log_path):
    output_json = log_path.with_suffix(".json")
    if output_json.exists():
        output_json.unlink()

    cmd = [
        sys.executable,
        str(THIS_DIR / "run_vqe_worker.py"),
        algorithm,
        system,
        build_type,
        str(output_json),
    ]
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("KMP_INIT_AT_FORK", "FALSE")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("KMP_LIBRARY", "serial")
    env.setdefault("KMP_AFFINITY", "disabled")
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=log, env=env)

    if proc.returncode != 0:
        return None, f"child process exited with code {proc.returncode}"
    if not output_json.exists():
        return None, "child process completed but did not write a result JSON"
    return restore_json_arrays(json.loads(output_json.read_text())), None


def run_algorithm_set(system, molecules, log_root):
    records = {"vqe": {}, "srqk": {}}
    skipped = []
    for build_type, mol in molecules.items():
        for algorithm, label in [("vqe", "VQE "), ("srqk", "SRQK")]:
            log_path = log_root / f"{system}_{build_type}_{algorithm}.log"
            print(f"\n  running {label} system={system:<8s} build_type={build_type:<10s} -> {log_path}")
            record, reason = run_algorithm_child(algorithm, system, build_type, log_path)
            if reason is None:
                records[algorithm][build_type] = record
                print(f"  status: PASS {label.strip()} energy = {record['energy']:+18.12f}")
            else:
                print(f"  status: SKIP {label.strip()} {reason}")
                print(f"  reason/log: {log_path}")
                skipped.append({
                    "system": system,
                    "algorithm": algorithm,
                    "build_type": build_type,
                    "reason": reason,
                })
    return records, skipped


def print_algorithm_summary(system, records):
    common.print_section(f"{system} algorithm energy summary")
    header = f"{'algorithm':>10s} {'build':>12s} {'energy':>18s} {'delta vs PySCF':>18s}"
    print(header)
    print("-" * len(header))
    for alg_name, alg_records in records.items():
        reference_name = next(iter(alg_records), None)
        if reference_name is None:
            print(f"{alg_name:>10s} {'*':>12s} {'SKIPPED':>18s} {'no completed runs':>18s}")
            continue
        reference = alg_records[reference_name]["energy"]
        for build_type, record in alg_records.items():
            delta = record["energy"] - reference
            print(f"{alg_name:>10s} {build_type:>12s} {record['energy']:+18.12f} {delta:+18.6e}")


def compare_algorithm_set(system, records, build_types):
    for alg_name, alg_records in records.items():
        reference_name = next((name for name in build_types if name in alg_records), None)
        if reference_name is None:
            print(f"\n==> {system} {alg_name.upper()} comparisons <==")
            print("  status: SKIP no completed runs.")
            continue
        reference = alg_records[reference_name]
        for build_type in build_types:
            if build_type == reference_name or build_type not in alg_records:
                continue
            benzene_rotation_relaxed_srqk = system == "benzene" and alg_name == "srqk"
            rows = common.compare_algorithm_records(
                reference_name,
                reference,
                build_type,
                alg_records[build_type],
                include_srqk_matrices=not benzene_rotation_relaxed_srqk,
                include_srqk_time_grid=not benzene_rotation_relaxed_srqk,
            )
            if benzene_rotation_relaxed_srqk:
                print(
                    "\n  note: benzene AVAS SRQK compares the final energy only; "
                    "independent active-space rotations can change the raw Krylov "
                    "H/S blocks and lambda_inv time grid without changing the "
                    "variational root."
                )
            common.print_check_table(f"{system} {alg_name.upper()} {build_type} vs PySCF", rows)


def main():
    print("\n==> Informal PySCF dump algorithm parity check <==")
    missing_dumps = [
        path
        for path in (common.BEH2_DUMP_PATH, common.BENZENE_AVAS_DUMP_PATH)
        if not path.exists()
    ]
    if missing_dumps:
        print("  one or more dump files are missing; attempting to generate them")
        common.ensure_dumps(overwrite=False, run_benzene_casci=True)

    systems = [
        ("beh2", ("pyscf", "psi4", "pyscf_dump")),
        ("benzene", ("pyscf", "pyscf_dump")),
    ]

    for system, build_types in systems:
        if system == "beh2":
            common.print_geometry("bent BeH2", common.beh2_geometry())
        else:
            common.print_geometry("benzene C 2pz AVAS", common.benzene_geometry())

        log_root = common.LOG_DIR / f"{system}_algorithm_parity"
        molecules = build_available_molecules(system, build_types, log_root)
        if "pyscf_dump" not in molecules:
            raise AssertionError(f"{system} requires a PySCF dump molecule.")
        if "pyscf" not in molecules:
            print(
                f"\n  note: direct PySCF build for {system} is unavailable in this "
                "environment; algorithm comparisons will use the first completed "
                "available build as the reference."
            )

        records, skipped = run_algorithm_set(system, molecules, log_root)
        print_algorithm_summary(system, records)
        compare_algorithm_set(system, records, build_types)
        if skipped:
            print(f"\nSkipped {len(skipped)} {system} run(s):")
            for item in skipped:
                print(
                    f"  - {item['algorithm']} {item['build_type']}: "
                    f"{item['reason']}"
                )

    print("\nInformal PySCF dump algorithm parity check passed.")


if __name__ == "__main__":
    main()
