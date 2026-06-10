"""Simple batched-optimization sandbox for VQE/tUCC.

Edit the toggles below and run:

    python sandbox/test_batched_optimization/test_batched_optimization.py

This is a manual diagnostic, not a formal pytest test.  Batched optimization
requires a generalized pool ordered with particle-hole terms first.
"""

from pathlib import Path
import traceback

import numpy as np
import qforte as qf


# ============================================================
# User toggles
# ============================================================

POOL_TYPE = "1-UpCCGSD"
# POOL_TYPE = "2-UpCCGSD"

OPTIMIZERS_TO_RUN = ["bfgs_qf", "lbfgs_qf", "BFGS", "L-BFGS-B"]
BATCH_TYPES_TO_RUN = [
    "none",
    "half_sweep",
    "full_sweep",
    "half_sweep_then_all",
    "full_sweep_then_all",
]

BATCHED_OPT_CYCLES = 1
MAXITER = 100
BATCH_MAXITER = 50
FINAL_MAXITER = 100

INIT_AMPS = "zero"  # "zero" or "mp2"
PRIMARY_POOL_ORDER = "none"  # "none", "mp2_amps", "gradients"
SECONDARY_POOL_ORDER = "lexical"  # "lexical", "shell"
GENERAL_EX_POOL_ORDER = "particle_hole_first"

USE_HESSIAN_DIAG = True
RUN_ERROR_DEMOS = True


def build_beh2():
    geom = [
        ("H", (0.0, 0.0, -1.0)),
        ("Be", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 1.0)),
    ]
    return qf.system_factory(
        build_type="psi4",
        mol_geometry=geom,
        basis="sto-3g",
        symmetry="c1",
        run_fci=True,
        run_mp2=True,
        store_mo_ints=True,
    )


def make_optimizer_options(optimizer_name):
    opts = {}
    if optimizer_name.lower() == "bfgs_qf":
        opts.update(
            {
                "bfgs_qf_use_hessian_diag": USE_HESSIAN_DIAG,
                "bfgs_qf_hdiag_mode": "abs",
                "bfgs_qf_hdiag_floor": 1.0e-3,
                "bfgs_qf_max_step_norm": 0.5,
            }
        )
    elif optimizer_name.lower() == "lbfgs_qf":
        opts.update(
            {
                "lbfgs_qf_use_hessian_diag": USE_HESSIAN_DIAG,
                "lbfgs_qf_hdiag_mode": "abs",
                "lbfgs_qf_hdiag_floor": 1.0e-3,
                "lbfgs_qf_memory": 10,
            }
        )
    return opts


def build_run_options(optimizer_name, batch_type):
    opts = {
        "pool_type": POOL_TYPE,
        "optimizer": optimizer_name,
        "opt_maxiter": MAXITER,
        "opt_thresh": 1.0e-4,
        "opt_ftol": 1.0e-8,
        "use_analytic_grad": True,
        "init_amps": INIT_AMPS,
        "primary_pool_order": PRIMARY_POOL_ORDER,
        "secondary_pool_order": SECONDARY_POOL_ORDER,
        "general_ex_pool_order": GENERAL_EX_POOL_ORDER,
        "batched_opt_type": batch_type,
        "batched_opt_cycles": BATCHED_OPT_CYCLES,
        "batched_opt_batch_maxiter": BATCH_MAXITER,
        "batched_opt_final_maxiter": FINAL_MAXITER,
        "batched_opt_verbose": True,
    }
    opts.update(make_optimizer_options(optimizer_name))
    return opts


def classify_batch_sizes(alg):
    ph = 0
    gen = 0
    for top in alg._tops:
        sq_op = alg._pool_obj[top][1]
        if alg._is_clean_particle_hole_excitation(sq_op) is not None:
            ph += 1
        else:
            gen += 1
    return ph, gen


def run_case(mol, optimizer_name, batch_type):
    alg = qf.UCCNVQE(
        mol,
        computer_type="fci",
        apply_ham_as_tensor=True,
        verbose=False,
    )

    opts = build_run_options(optimizer_name, batch_type)
    alg.run(**opts)

    result = getattr(alg, "_final_result", None)
    ph_size, gen_size = classify_batch_sizes(alg)
    amps = np.asarray(alg._tamps, dtype=float)
    history = getattr(alg, "_batched_opt_history", [])
    return {
        "status": "ok",
        "optimizer": optimizer_name,
        "batched_opt_type": batch_type,
        "pool": POOL_TYPE,
        "n_params": len(alg._tamps),
        "ph_params": ph_size,
        "gen_params": gen_size,
        "final_energy": float(alg.get_gs_energy()),
        "best_energy": min([h["final_energy"] for h in history], default=float(alg.get_gs_energy())),
        "error_vs_fci": float(alg.get_gs_energy() - mol.fci_energy),
        "final_grad_norm": getattr(result, "grad_norm", getattr(alg, "_curr_grad_norm", None)),
        "iterations": getattr(result, "nit", None),
        "nfev": getattr(result, "nfev", None),
        "njev": getattr(result, "njev", None),
        "nonzero": int(np.count_nonzero(np.abs(amps) > 1.0e-10)),
        "batch_sequence": " -> ".join(getattr(result, "batched_opt_schedule", ["ALL"])),
        "batch_history": history,
        "error": "",
    }


def run_error_demo(mol, label, options):
    alg = qf.UCCNVQE(
        mol,
        computer_type="fci",
        apply_ham_as_tensor=True,
        verbose=False,
    )
    try:
        alg.run(**options)
    except Exception as exc:
        print(f"\nExpected error demo [{label}]: {exc}")
        return str(exc)
    raise AssertionError(f"Expected error demo {label} did not raise.")


def write_summary(rows, error_demos):
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "summary.md"

    lines = []
    lines.append("# Batched Optimization Summary\n")
    lines.append(f"- pool: `{POOL_TYPE}`")
    lines.append(f"- init_amps: `{INIT_AMPS}`")
    lines.append(f"- general_ex_pool_order: `{GENERAL_EX_POOL_ORDER}`")
    lines.append(f"- cycles: `{BATCHED_OPT_CYCLES}`")
    lines.append("")
    lines.append(
        "| optimizer | batch type | status | E_final | err FCI | ||g|| | "
        "nit | nfev | njev | params | PH | GEN | nnz | sequence |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    for row in rows:
        grad = row["final_grad_norm"]
        grad_text = f"{grad:.3e}" if grad is not None else "-"
        lines.append(
            f"| {row['optimizer']} | {row['batched_opt_type']} | {row['status']} | "
            f"{row['final_energy']:+.10f} | {row['error_vs_fci']:+.3e} | "
            f"{grad_text} | {row['iterations']} | {row['nfev']} | {row['njev']} | "
            f"{row['n_params']} | {row['ph_params']} | {row['gen_params']} | "
            f"{row['nonzero']} | {row['batch_sequence']} |"
        )

    lines.append("\n## Per-Batch Details\n")
    for row in rows:
        if not row.get("batch_history"):
            continue
        lines.append(f"### {row['optimizer']} / {row['batched_opt_type']}\n")
        for batch in row["batch_history"]:
            lines.append(
                f"- batch {batch['batch_number']} `{batch['batch_label']}`: "
                f"active={batch['active_size']}, "
                f"E={batch['initial_energy']:+.10f} -> {batch['final_energy']:+.10f}, "
                f"reduced ||g||={batch['final_reduced_grad_norm']:.3e}, "
                f"full ||g||={batch['final_full_grad_norm']:.3e}, "
                f"nit={batch['iterations']}"
            )
        lines.append("")

    if error_demos:
        lines.append("## Expected Error Demos\n")
        for label, message in error_demos.items():
            lines.append(f"- `{label}`: {message}")

    path.write_text("\n".join(lines))
    print(f"\nWrote summary: {path}")


def main():
    mol = build_beh2()
    rows = []
    for optimizer_name in OPTIMIZERS_TO_RUN:
        for batch_type in BATCH_TYPES_TO_RUN:
            print(f"\n=== {optimizer_name} / {batch_type} ===")
            try:
                rows.append(run_case(mol, optimizer_name, batch_type))
            except Exception as exc:
                traceback.print_exc()
                rows.append(
                    {
                        "status": "failed",
                        "optimizer": optimizer_name,
                        "batched_opt_type": batch_type,
                        "pool": POOL_TYPE,
                        "n_params": 0,
                        "ph_params": 0,
                        "gen_params": 0,
                        "final_energy": np.nan,
                        "best_energy": np.nan,
                        "error_vs_fci": np.nan,
                        "final_grad_norm": None,
                        "iterations": None,
                        "nfev": None,
                        "njev": None,
                        "nonzero": 0,
                        "batch_sequence": "-",
                        "batch_history": [],
                        "error": str(exc),
                    }
                )

    error_demos = {}
    if RUN_ERROR_DEMOS:
        base = build_run_options("bfgs_qf", "half_sweep")
        bad_pool = dict(base, pool_type="SD")
        error_demos["non-generalized SD pool"] = run_error_demo(mol, "SD", bad_pool)

        bad_order = dict(base, general_ex_pool_order="default")
        error_demos["missing particle_hole_first"] = run_error_demo(
            mol, "default-order", bad_order
        )

        bad_optimizer = dict(base, optimizer="jacobi")
        error_demos["jacobi unsupported"] = run_error_demo(
            mol, "jacobi", bad_optimizer
        )

    write_summary(rows, error_demos)


if __name__ == "__main__":
    main()
