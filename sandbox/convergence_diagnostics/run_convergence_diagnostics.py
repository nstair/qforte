"""Run sandbox VQE/tUCC convergence diagnostics.

Usage from the repository root:

    conda run -n qfe_env_v1 python sandbox/convergence_diagnostics/run_convergence_diagnostics.py --mode fast

The script is deliberately configurable near the top.  It writes a timestamped
results directory containing summary.md, summary.csv, raw_results.json, and
trajectory data.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from diagnostic_helpers import (
    build_system,
    json_safe,
    metadata,
    molecule_summary,
    run_case,
    run_pool_validity_checks,
    slugify,
    timestamp_label,
    write_csv,
    write_json,
    write_markdown_summary,
    write_trajectories_csv,
)


# ============================================================
# User toggles
# ============================================================

RUN_MODE = "fast"  # "fast", "medium", or "slow"

WRITE_TRAJECTORIES = True
RUN_SYMMETRY_CHECKS = True
RUN_POOL_VALIDITY_CHECKS = True

MAXITER_BY_MODE = {
    "fast": 75,
    "medium": 200,
    "slow": 400,
}

OPT_THRESH_BY_MODE = {
    "fast": 1.0e-4,
    "medium": 1.0e-5,
    "slow": 1.0e-6,
}

# Set any of these to a list of names to restrict the run without editing the
# libraries below.  Leave as None to use the mode defaults.
MOLECULES_TO_RUN = None
POOLS_TO_RUN = None
CONFIGS_TO_RUN = None


# ============================================================
# Molecule definitions
# ============================================================

MOLECULE_LIBRARY: Dict[str, Dict[str, Any]] = {
    "beh2_c1": {
        "name": "BeH2",
        "basis": "sto-3g",
        "symmetry": "c1",
        "geometry": [
            ("H", (0.0, 0.0, -1.0)),
            ("Be", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 1.0)),
        ],
        "run_mp2": True,
        "run_ccsd": True,
        "run_fci": True,
        "nroots_fci": 4,
        "store_mo_ints": True,
    },
    "beh2_d2h": {
        "name": "BeH2",
        "basis": "sto-3g",
        "symmetry": "d2h",
        "geometry": [
            ("H", (0.0, 0.0, -1.0)),
            ("Be", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 1.0)),
        ],
        "run_mp2": True,
        "run_ccsd": True,
        "run_fci": True,
        "nroots_fci": 4,
        "store_mo_ints": True,
    },
    "h4_c1": {
        "name": "H4",
        "basis": "sto-3g",
        "symmetry": "c1",
        "geometry": [
            ("H", (0.0, -1.0, -1.0)),
            ("H", (0.0, -1.0, +1.0)),
            ("H", (0.0, +1.0, -1.0)),
            ("H", (0.0, +1.0, +1.0)),
        ],
        "run_mp2": True,
        "run_ccsd": True,
        "run_fci": True,
        "nroots_fci": 4,
        "store_mo_ints": True,
    },
    "h4_d2h": {
        "name": "H4",
        "basis": "sto-3g",
        "symmetry": "d2h",
        "geometry": [
            ("H", (0.0, -1.0, -1.0)),
            ("H", (0.0, -1.0, +1.0)),
            ("H", (0.0, +1.0, -1.0)),
            ("H", (0.0, +1.0, +1.0)),
        ],
        "run_mp2": True,
        "run_ccsd": True,
        "run_fci": True,
        "nroots_fci": 4,
        "store_mo_ints": True,
    },
}


# ============================================================
# Optimizer/configuration definitions
# ============================================================

BASE_RUN_OPTIONS = {
    "opt_ftol": 1.0e-8,
    "use_analytic_grad": True,
    "init_amps": "zero",
    "primary_pool_order": "none", # none
    "secondary_pool_order": "shell", # lexical
    "general_ex_pool_order": "default", # default
}

LBFGS_COMMON = {
    "lbfgs_qf_memory": 8,
    "lbfgs_qf_max_step_norm": 0.5,
    "lbfgs_qf_use_newton_cg": False,
    "lbfgs_qf_use_target_block": False,
    "lbfgs_qf_use_hessian_diag": False,
}

HESSIAN_DIAG = {
    "lbfgs_qf_use_hessian_diag": True,
    "lbfgs_qf_hdiag_mode": "abs",
}

TARGET_BLOCK_NO_RESET = {
    "lbfgs_qf_use_target_block": True,
    "lbfgs_qf_target_block_trigger": "periodic_or_stalled",
    "lbfgs_qf_target_block_start": 10,
    "lbfgs_qf_target_block_every": 10,
    "lbfgs_qf_target_block_size": 20,
    "lbfgs_qf_target_block_max_size": 20,
    "lbfgs_qf_target_block_fd_type": "forward",
    "lbfgs_qf_target_block_max_step_norm": 0.05,
    "lbfgs_qf_target_block_reset_lbfgs_history": False,
}

TARGET_BLOCK_RESET = dict(TARGET_BLOCK_NO_RESET, lbfgs_qf_target_block_reset_lbfgs_history=True)

NEWTON_CG_NO_RESET = {
    "lbfgs_qf_use_newton_cg": True,
    "lbfgs_qf_newton_cg_trigger": "periodic_or_stalled",
    "lbfgs_qf_newton_cg_start": 10,
    "lbfgs_qf_newton_cg_every": 10,
    "lbfgs_qf_newton_cg_max_step_norm": 0.25,
    "lbfgs_qf_newton_cg_reset_lbfgs_history": False,
}

NEWTON_CG_RESET = dict(NEWTON_CG_NO_RESET, lbfgs_qf_newton_cg_reset_lbfgs_history=True)

CONFIG_LIBRARY: Dict[str, Dict[str, Any]] = {
    "scipy_bfgs": {
        "optimizer": "BFGS",
    },
    "scipy_lbfgsb": {
        "optimizer": "L-BFGS-B",
    },
    "lbfgs_qf": {
        "optimizer": "lbfgs_qf",
        **LBFGS_COMMON,
    },
    "lbfgs_qf_hdiag": {
        "optimizer": "lbfgs_qf",
        **LBFGS_COMMON,
        **HESSIAN_DIAG,
    },
    "lbfgs_qf_mp2": {
        "optimizer": "lbfgs_qf",
        "init_amps": "mp2",
        **LBFGS_COMMON,
    },
    "lbfgs_qf_mp2_hdiag": {
        "optimizer": "lbfgs_qf",
        "init_amps": "mp2",
        **LBFGS_COMMON,
        **HESSIAN_DIAG,
    },
    "lbfgs_qf_hdiag_block_no_reset": {
        "optimizer": "lbfgs_qf",
        **LBFGS_COMMON,
        **HESSIAN_DIAG,
        **TARGET_BLOCK_NO_RESET,
    },
    "lbfgs_qf_hdiag_block_reset": {
        "optimizer": "lbfgs_qf",
        **LBFGS_COMMON,
        **HESSIAN_DIAG,
        **TARGET_BLOCK_RESET,
    },
    "lbfgs_qf_hdiag_ncg_no_reset": {
        "optimizer": "lbfgs_qf",
        **LBFGS_COMMON,
        **HESSIAN_DIAG,
        **NEWTON_CG_NO_RESET,
    },
    "lbfgs_qf_hdiag_ncg_reset": {
        "optimizer": "lbfgs_qf",
        **LBFGS_COMMON,
        **HESSIAN_DIAG,
        **NEWTON_CG_RESET,
    },
    "lbfgs_qf_mp2_order_mp2_shell": {
        "optimizer": "lbfgs_qf",
        "init_amps": "mp2",
        "primary_pool_order": "mp2_amps",
        "secondary_pool_order": "shell",
        **LBFGS_COMMON,
        **HESSIAN_DIAG,
    },
    "lbfgs_qf_mp2_order_grad_shell": {
        "optimizer": "lbfgs_qf",
        "init_amps": "mp2",
        "primary_pool_order": "gradients",
        "secondary_pool_order": "shell",
        **LBFGS_COMMON,
        **HESSIAN_DIAG,
    },
    "lbfgs_qf_mp2_order_grad_shell_ph_first": {
        "optimizer": "lbfgs_qf",
        "init_amps": "mp2",
        "primary_pool_order": "gradients",
        "secondary_pool_order": "shell",
        "general_ex_pool_order": "particle_hole_first",
        **LBFGS_COMMON,
        **HESSIAN_DIAG,
    },
    # These mirror sandbox/compare_pool_orderings/test_compare_pool_orderings.py
    # so the convergence summary can compare its lower BeH2 energies against
    # the broader benchmark rows without manually cross-reading terminal output.
    "pool_order_file_default_500": {
        "optimizer": "bfgs",
        "opt_maxiter": 500,
        "opt_thresh": 1.0e-4,
        "init_amps": "zero",
        "primary_pool_order": "none",
        "secondary_pool_order": "shell",
        "general_ex_pool_order": "particle_hole_first",
    },
    "pool_order_file_shell_500": {
        "optimizer": "bfgs",
        "opt_maxiter": 500,
        "opt_thresh": 1.0e-4,
        "init_amps": "zero",
        "primary_pool_order": "none",
        "secondary_pool_order": "shell",
        "general_ex_pool_order": "default",
    },
    "pool_order_file_mp2_shell_500": {
        "optimizer": "bfgs",
        "opt_maxiter": 500,
        "opt_thresh": 1.0e-4,
        "init_amps": "zero",
        "primary_pool_order": "mp2_amps",
        "secondary_pool_order": "shell",
        "general_ex_pool_order": "default",
    },
    "pool_order_file_gradient_shell_500": {
        "optimizer": "bfgs",
        "opt_maxiter": 500,
        "opt_thresh": 1.0e-4,
        "init_amps": "zero",
        "primary_pool_order": "gradients",
        "secondary_pool_order": "shell",
        "general_ex_pool_order": "default",
    },
    "pool_order_file_gradient_shell_ph_first_500": {
        "optimizer": "bfgs",
        "opt_maxiter": 500,
        "opt_thresh": 1.0e-4,
        "init_amps": "zero",
        "primary_pool_order": "gradients",
        "secondary_pool_order": "shell",
        "general_ex_pool_order": "particle_hole_first",
    },
}


# ============================================================
# Mode definitions
# ============================================================

MODE_MOLECULES = {
    "fast": ["beh2_c1", "h4_c1"],
    "medium": ["beh2_c1", "beh2_d2h", "h4_c1"],
    "slow": ["beh2_c1", "beh2_d2h", "h4_c1", "h4_d2h"],
}

MODE_POOLS = {
    "fast": ["SD", "1-UpCCGSD"],
    "medium": [
        "SD",
        "1-UpCCGSD",
        "1-UpCCGSDx",
        "2-UpCCGSD",
        "2-UpCCGSDx",
        "3-UpCCGSD",
        "3-UpCCGSDx",
    ],
    "slow": [
        "SD",
        "GSD",
        "GSDx",
        "1-UpCCGSD",
        "1-UpCCGSDx",
        "2-UpCCGSD",
        "2-UpCCGSDx",
        "3-UpCCGSD",
        "3-UpCCGSDx",
    ],
}

MODE_EXTRA_CASES = {
    "fast": [],
    "medium": [
        {
            "molecule_key": "beh2_c1",
            "pool_type": "1-UpCCGSD",
            "config_names": [
                "pool_order_file_default_500",
                "pool_order_file_shell_500",
                "pool_order_file_mp2_shell_500",
                "pool_order_file_gradient_shell_500",
                "pool_order_file_gradient_shell_ph_first_500",
            ],
        }
    ],
    "slow": [
        {
            "molecule_key": "beh2_c1",
            "pool_type": "1-UpCCGSD",
            "config_names": [
                "pool_order_file_default_500",
                "pool_order_file_shell_500",
                "pool_order_file_mp2_shell_500",
                "pool_order_file_gradient_shell_500",
                "pool_order_file_gradient_shell_ph_first_500",
            ],
        }
    ],
}

MODE_CONFIGS = {
    "fast": [
        "scipy_bfgs",
        "scipy_lbfgsb",
        "lbfgs_qf",
        "lbfgs_qf_hdiag",
        "lbfgs_qf_mp2_hdiag",
    ],
    "medium": [
        "scipy_bfgs",
        "scipy_lbfgsb",
        "lbfgs_qf",
        "lbfgs_qf_hdiag",
        "lbfgs_qf_mp2",
        "lbfgs_qf_mp2_hdiag",
        "lbfgs_qf_hdiag_block_no_reset",
        "lbfgs_qf_hdiag_ncg_no_reset",
    ],
    "slow": [
        "scipy_bfgs",
        "scipy_lbfgsb",
        "lbfgs_qf",
        "lbfgs_qf_hdiag",
        "lbfgs_qf_mp2",
        "lbfgs_qf_mp2_hdiag",
        "lbfgs_qf_hdiag_block_no_reset",
        "lbfgs_qf_hdiag_block_reset",
        "lbfgs_qf_hdiag_ncg_no_reset",
        "lbfgs_qf_hdiag_ncg_reset",
        "lbfgs_qf_mp2_order_mp2_shell",
        "lbfgs_qf_mp2_order_grad_shell",
        "lbfgs_qf_mp2_order_grad_shell_ph_first",
    ],
}


def selected(mode: str, values: Sequence[str], override: Sequence[str] | None) -> List[str]:
    if override is not None:
        return list(override)
    return list(values)


def run_options_for_case(pool_type: str, config_name: str, mode: str) -> Dict[str, Any]:
    options = copy.deepcopy(BASE_RUN_OPTIONS)
    options.update(copy.deepcopy(CONFIG_LIBRARY[config_name]))
    options["pool_type"] = pool_type
    options.setdefault("opt_maxiter", MAXITER_BY_MODE[mode])
    options.setdefault("opt_thresh", OPT_THRESH_BY_MODE[mode])
    return options


def build_cases(mode: str, molecules: Sequence[str], pools: Sequence[str], configs: Sequence[str]) -> List[Dict[str, Any]]:
    cases = []
    idx = 0
    for molecule_key in molecules:
        spec = MOLECULE_LIBRARY[molecule_key]
        for pool_type in pools:
            for config_name in configs:
                idx += 1
                case_id = slugify(
                    f"{idx:04d}_{molecule_key}_{pool_type}_{config_name}"
                )
                cases.append(
                    {
                        "case_id": case_id,
                        "molecule_key": molecule_key,
                        "molecule_name": spec["name"],
                        "symmetry": spec.get("symmetry", "c1"),
                        "pool_type": pool_type,
                        "config_name": config_name,
                        "run_options": run_options_for_case(pool_type, config_name, mode),
                    }
                )
    for extra in MODE_EXTRA_CASES.get(mode, []):
        molecule_key = extra["molecule_key"]
        if molecule_key not in molecules:
            continue
        spec = MOLECULE_LIBRARY[molecule_key]
        pool_type = extra["pool_type"]
        if pool_type not in pools:
            continue
        for config_name in extra["config_names"]:
            idx += 1
            case_id = slugify(
                f"{idx:04d}_{molecule_key}_{pool_type}_{config_name}"
            )
            cases.append(
                {
                    "case_id": case_id,
                    "molecule_key": molecule_key,
                    "molecule_name": spec["name"],
                    "symmetry": spec.get("symmetry", "c1"),
                    "pool_type": pool_type,
                    "config_name": config_name,
                    "run_options": run_options_for_case(pool_type, config_name, mode),
                }
            )
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fast", "medium", "slow"], default=RUN_MODE)
    parser.add_argument("--results-dir", default=None, help="Optional output directory.")
    parser.add_argument(
        "--limit-runs",
        type=int,
        default=None,
        help="Debug helper: run only the first N cases, but still write summaries.",
    )
    parser.add_argument(
        "--skip-pool-checks",
        action="store_true",
        help="Skip raw pool count/signature checks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode = args.mode
    root = Path(__file__).resolve().parents[2]
    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else root / "sandbox" / "convergence_diagnostics" / "results" / timestamp_label()
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    molecule_keys = selected(mode, MODE_MOLECULES[mode], MOLECULES_TO_RUN)
    pool_types = selected(mode, MODE_POOLS[mode], POOLS_TO_RUN)
    config_names = selected(mode, MODE_CONFIGS[mode], CONFIGS_TO_RUN)

    print(f"Writing diagnostics to: {results_dir}")
    print(f"mode={mode}")
    print(f"molecules={molecule_keys}")
    print(f"pools={pool_types}")
    print(f"configs={config_names}")

    molecule_records: Dict[str, Any] = {}
    for key in molecule_keys:
        spec = MOLECULE_LIBRARY[key]
        print(f"\nBuilding molecule {key} ({spec['name']} / {spec.get('symmetry', 'c1')}) ...")
        build_log = results_dir / f"build_{key}.log"
        with build_log.open("w") as handle:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                mol = build_system(spec)
        summary = molecule_summary(spec, mol)
        molecule_records[key] = {"spec": spec, "mol": mol, "summary": summary}
        print(
            f"  RHF={summary.get('hf_energy')} FCI={summary.get('fci_energy')} "
            f"nocc={summary.get('nocc')} nvir={summary.get('nvir')}"
        )

    cases = build_cases(mode, molecule_keys, pool_types, config_names)
    if args.limit_runs is not None:
        cases = cases[: args.limit_runs]
        print(f"Debug limit active: running first {len(cases)} cases.")

    rows: List[Dict[str, Any]] = []
    for i, case in enumerate(cases, start=1):
        print(
            f"\n[{i}/{len(cases)}] {case['molecule_key']} "
            f"{case['pool_type']} {case['config_name']}"
        )
        mol = molecule_records[case["molecule_key"]]["mol"]
        row = run_case(case, mol, results_dir)
        rows.append(row)
        status = row.get("status")
        energy = row.get("final_energy")
        grad = row.get("final_grad_norm")
        print(f"  status={status} E={energy} ||g||={grad} log={row.get('log_path')}")

        # Write lightweight checkpoint summaries as the run proceeds.  These
        # intentionally omit final pool/signature checks, which are only cheap
        # to do once the calculation matrix has finished.
        checkpoint_meta = metadata(mode)
        checkpoint_meta["results_dir"] = str(results_dir)
        checkpoint_meta["n_cases"] = len(cases)
        checkpoint_meta["n_completed_or_failed"] = len(rows)
        checkpoint_meta["n_failures"] = sum(1 for item in rows if item.get("status") != "ok")
        checkpoint_config_defs = {
            name: CONFIG_LIBRARY[name] for name in sorted({item["config_name"] for item in cases[:i]})
        }
        checkpoint_pool_checks = {"pool_rows": [], "x_variant_rows": [], "symmetry_rows": []}
        write_markdown_summary(
            rows=rows,
            pool_checks=checkpoint_pool_checks,
            molecule_summaries=[record["summary"] for record in molecule_records.values()],
            config_defs=checkpoint_config_defs,
            meta=checkpoint_meta,
            path=results_dir / "summary_in_progress.md",
        )
        write_csv(rows, results_dir / "summary_in_progress.csv")
        if WRITE_TRAJECTORIES:
            write_trajectories_csv(rows, results_dir / "trajectories_in_progress.csv")

    if RUN_POOL_VALIDITY_CHECKS and not args.skip_pool_checks:
        print("\nRunning pool validity/signature checks ...")
        pool_checks = run_pool_validity_checks(molecule_records, pool_types)
    else:
        pool_checks = {"pool_rows": [], "x_variant_rows": [], "symmetry_rows": []}

    molecule_summaries = [record["summary"] for record in molecule_records.values()]
    meta = metadata(mode)
    meta["results_dir"] = str(results_dir)
    meta["n_cases"] = len(cases)
    meta["n_failures"] = sum(1 for row in rows if row.get("status") != "ok")

    used_config_names = sorted({case["config_name"] for case in cases})
    selected_config_defs = {name: CONFIG_LIBRARY[name] for name in used_config_names}

    summary_md = results_dir / "summary.md"
    summary_csv = results_dir / "summary.csv"
    raw_json = results_dir / "raw_results.json"
    trajectories_csv = results_dir / "trajectories.csv"

    write_markdown_summary(
        rows=rows,
        pool_checks=pool_checks,
        molecule_summaries=molecule_summaries,
        config_defs=selected_config_defs,
        meta=meta,
        path=summary_md,
    )
    write_csv(rows, summary_csv)
    if WRITE_TRAJECTORIES:
        write_trajectories_csv(rows, trajectories_csv)
    write_json(
        {
            "metadata": meta,
            "molecules": molecule_summaries,
            "configs": selected_config_defs,
            "cases": cases,
            "results": rows,
            "pool_checks": pool_checks,
        },
        raw_json,
    )

    print("\nDiagnostics complete.")
    print(f"  {summary_md}")
    print(f"  {summary_csv}")
    print(f"  {raw_json}")
    if WRITE_TRAJECTORIES:
        print(f"  {trajectories_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
