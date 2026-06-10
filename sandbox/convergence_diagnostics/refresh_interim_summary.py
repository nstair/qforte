"""Refresh an in-progress convergence diagnostic summary from completed logs.

This is useful for a long run that was started before checkpoint summaries were
enabled.  It reads completed per-case logs and writes summary_in_progress.md/csv
without touching the running process.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from diagnostic_helpers import (
    analyze_trajectory,
    format_value,
    metadata,
    parse_energy_trajectory_from_log,
    safe_float,
    write_csv,
    write_json,
    write_markdown_summary,
    write_trajectories_csv,
)
from run_convergence_diagnostics import (
    CONFIG_LIBRARY,
    MODE_CONFIGS,
    MODE_MOLECULES,
    MODE_POOLS,
    MOLECULE_LIBRARY,
    build_cases,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", help="Active results directory to refresh.")
    parser.add_argument("--mode", default="medium", choices=["fast", "medium", "slow"])
    parser.add_argument(
        "--also-summary",
        action="store_true",
        help="Also write summary.md/summary.csv, knowing the final driver may overwrite them.",
    )
    return parser.parse_args()


def parse_fci_energy(results_dir: Path, molecule_key: str) -> Optional[float]:
    path = results_dir / f"build_{molecule_key}.log"
    if not path.exists():
        return None
    match = re.search(r"i:\s*0\s+Ei:\s*([-+]?\d+\.\d+)", path.read_text(errors="replace"))
    return safe_float(match.group(1)) if match else None


def parse_summary_scalar(text: str, label: str) -> Optional[float]:
    pattern = re.compile(re.escape(label) + r"\s*:?\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
    match = pattern.search(text)
    return safe_float(match.group(1)) if match else None


def parse_summary_int(text: str, label: str) -> Optional[int]:
    value = parse_summary_scalar(text, label)
    return int(value) if value is not None else None


def parse_iteration_info(text: str) -> Dict[str, Any]:
    last = {
        "nit": None,
        "final_grad_norm": None,
        "nfev": None,
        "njev": None,
        "res_m_evals": None,
        "target_block_accepted": 0,
        "newton_cg_accepted": 0,
    }

    lbfgs_row = re.compile(
        r"^\s*(\d+)\s+\|\s*([-+]?\d+\.\d+)\s+\|\s*([-+]?\d+\.\d+)\s+\|\s*([-+]?\d+\.\d+).*?\|\s*([A-Z]+)\s+\|.*?\|\s*(\d+)/(\d+)/(\d+)",
        re.MULTILINE,
    )
    scipy_row = re.compile(
        r"^\s*(\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+(\d+)\s+(\d+)\s+([-+]?\d+\.\d+)",
        re.MULTILINE,
    )

    for match in lbfgs_row.finditer(text):
        last["nit"] = int(match.group(1))
        last["final_grad_norm"] = safe_float(match.group(4))
        step_type = match.group(5)
        last["nfev"] = int(match.group(6))
        last["njev"] = int(match.group(7))
        last["res_m_evals"] = int(match.group(8))
        if step_type == "BLOCK":
            last["target_block_accepted"] += 1
        elif step_type == "NCG":
            last["newton_cg_accepted"] += 1

    for match in scipy_row.finditer(text):
        last["nit"] = int(match.group(1))
        last["final_grad_norm"] = safe_float(match.group(6))
        last["njev"] = int(match.group(4))
        last["res_m_evals"] = int(match.group(5))

    f_eval_match = re.search(r"Function evaluations:\s*(\d+)", text)
    g_eval_match = re.search(r"Gradient evaluations:\s*(\d+)", text)
    if f_eval_match:
        last["nfev"] = int(f_eval_match.group(1))
    if g_eval_match:
        last["njev"] = int(g_eval_match.group(1))

    return last


def row_from_completed_log(
    case: Dict[str, Any],
    log_path: Path,
    fci_by_molecule_key: Dict[str, Optional[float]],
) -> Optional[Dict[str, Any]]:
    text = log_path.read_text(errors="replace")
    if "==> UCCN-VQE summary <==" not in text and "=== Exception ===" not in text:
        return None

    opts = case["run_options"]
    row = {
        "case_id": case["case_id"],
        "molecule": case["molecule_name"],
        "symmetry": case["symmetry"],
        "pool_type": opts.get("pool_type"),
        "config_name": case["config_name"],
        "optimizer": opts.get("optimizer"),
        "init_amps": opts.get("init_amps", "zero"),
        "primary_pool_order": opts.get("primary_pool_order", "none"),
        "secondary_pool_order": opts.get("secondary_pool_order", "lexical"),
        "general_ex_pool_order": opts.get("general_ex_pool_order", "default"),
        "use_hessian_diag": bool(opts.get("lbfgs_qf_use_hessian_diag", False)),
        "use_newton_cg": bool(opts.get("lbfgs_qf_use_newton_cg", False)),
        "use_target_block": bool(opts.get("lbfgs_qf_use_target_block", False)),
        "target_block_reset_history": opts.get("lbfgs_qf_target_block_reset_lbfgs_history"),
        "newton_cg_reset_history": opts.get("lbfgs_qf_newton_cg_reset_lbfgs_history"),
        "maxiter": opts.get("opt_maxiter"),
        "log_path": str(Path("logs") / log_path.name),
        "run_options": opts,
    }

    if "=== Exception ===" in text and "==> UCCN-VQE summary <==" not in text:
        row.update({"status": "failed", "error": "See log for exception."})
        return row

    trajectory = parse_energy_trajectory_from_log(log_path)
    traj_info = analyze_trajectory(trajectory)
    iter_info = parse_iteration_info(text)
    final_energy = parse_summary_scalar(text, "Final UCCN-VQE Energy")
    fci_energy = fci_by_molecule_key.get(case["molecule_key"])

    row.update(
        {
            "status": "ok",
            "error": None,
            "trajectory": trajectory,
            "final_energy": final_energy,
            "best_energy": traj_info.get("best_energy"),
            "error_vs_fci": (
                final_energy - fci_energy
                if final_energy is not None and fci_energy is not None
                else None
            ),
            "pool_size": parse_summary_int(text, "Number of operators in pool"),
            "n_params": parse_summary_int(text, "Final number of amplitudes in ansatz"),
            "n_nonzero": parse_summary_int(text, "Number of non-zero parameters used"),
            "pauli_term_measurements": parse_summary_int(text, "Number of Pauli term measurements"),
            "res_vec_evals": parse_summary_int(text, "Number of grad vector evaluations"),
            "res_m_evals": parse_summary_int(text, "Number of individual grad evaluations"),
            "target_block_attempts": iter_info["target_block_accepted"],
            "target_block_accepted": iter_info["target_block_accepted"],
            "newton_cg_attempts": iter_info["newton_cg_accepted"],
            "newton_cg_accepted": iter_info["newton_cg_accepted"],
        }
    )
    row.update(iter_info)
    row.update(traj_info)
    return row


def molecule_summaries_for_mode(mode: str, results_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for key in MODE_MOLECULES[mode]:
        spec = MOLECULE_LIBRARY[key]
        rows.append(
            {
                "name": spec["name"],
                "basis": spec["basis"],
                "symmetry": spec.get("symmetry", "c1"),
                "geometry": spec["geometry"],
                "fci_energy": parse_fci_energy(results_dir, key),
                "hf_energy": None,
                "mp2_energy": None,
                "ccsd_energy": None,
                "nocc": None,
                "nvir": None,
                "n_qubits": None,
                "orb_irreps_to_int": None,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    cases = build_cases(args.mode, MODE_MOLECULES[args.mode], MODE_POOLS[args.mode], MODE_CONFIGS[args.mode])
    cases_by_id = {case["case_id"]: case for case in cases}
    fci_by_molecule_key = {
        key: parse_fci_energy(results_dir, key) for key in MODE_MOLECULES[args.mode]
    }

    rows = []
    for log_path in sorted((results_dir / "logs").glob("*.log")):
        case_id = log_path.stem
        case = cases_by_id.get(case_id)
        if case is None:
            continue
        row = row_from_completed_log(case, log_path, fci_by_molecule_key)
        if row is not None:
            rows.append(row)

    meta = metadata(args.mode)
    meta["results_dir"] = str(results_dir)
    meta["partial_refresh"] = True
    meta["n_completed_or_failed"] = len(rows)
    meta["n_failures"] = sum(1 for row in rows if row.get("status") != "ok")

    used_config_names = sorted({row["config_name"] for row in rows})
    config_defs = {name: CONFIG_LIBRARY[name] for name in used_config_names}
    pool_checks = {"pool_rows": [], "x_variant_rows": [], "symmetry_rows": []}
    molecule_summaries = molecule_summaries_for_mode(args.mode, results_dir)

    write_markdown_summary(
        rows,
        pool_checks,
        molecule_summaries,
        config_defs,
        meta,
        results_dir / "summary_in_progress.md",
    )
    write_csv(rows, results_dir / "summary_in_progress.csv")
    write_trajectories_csv(rows, results_dir / "trajectories_in_progress.csv")
    write_json(
        {"metadata": meta, "results": rows},
        results_dir / "raw_results_in_progress.json",
    )

    if args.also_summary:
        write_markdown_summary(
            rows,
            pool_checks,
            molecule_summaries,
            config_defs,
            meta,
            results_dir / "summary.md",
        )
        write_csv(rows, results_dir / "summary.csv")

    print(
        f"Refreshed {len(rows)} completed/failed rows in {results_dir}. "
        f"Latest final energy: {format_value(rows[-1].get('final_energy'), 10) if rows else '-'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

