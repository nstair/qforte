"""Manual BeH2 sandbox for the in-house dense bfgs_qf optimizer.

The toggles near the top are meant to be edited directly.  This is not a
pytest-style test; it is a compact experiment script for comparing scipy BFGS,
scipy L-BFGS-B, lbfgs_qf, and the new full-memory bfgs_qf on the same molecule.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import re
import time
from pathlib import Path

import numpy as np
import qforte as qf


# ============================================================
# User toggles
# ============================================================

POOLS_TO_RUN = [
    "SD",
    "1-UpCCGSD",
    # "2-UpCCGSD",
    # "3-UpCCGSD",
]

CONFIGS_TO_RUN = [
    # "scipy_bfgs",
    # "scipy_lbfgsb",
    # "lbfgs_qf",
    # "lbfgs_qf_hdiag",
    "bfgs_qf",
    # "bfgs_qf_hdiag",
    # "bfgs_qf_hdiag_ncg",
    # "bfgs_qf_hdiag_block",
]

MAXITER = 200
OPT_THRESH = 1.0e-5
OPT_FTOL = 1.0e-8
RUN_NCG = True
RUN_BLOCK = True
PRINT_TRAJECTORIES = False

# Keep these conservative for manual experiments.  The known BeH2/k-UpCCGSD
# hard case may need 200-500 iterations to expose the lower basin.
LBFGS_HDIAG_OPTIONS = {
    "lbfgs_qf_use_hessian_diag": True,
    "lbfgs_qf_hdiag_mode": "abs",
    "lbfgs_qf_hdiag_floor": 1.0e-3,
}

BFGS_HDIAG_OPTIONS = {
    "bfgs_qf_use_hessian_diag": True,
    "bfgs_qf_hdiag_mode": "abs",
    "bfgs_qf_hdiag_floor": 1.0e-3,
}

BFGS_NCG_OPTIONS = {
    "bfgs_qf_use_newton_cg": True,
    "bfgs_qf_newton_cg_trigger": "periodic_or_stalled",
    "bfgs_qf_newton_cg_start": 10,
    "bfgs_qf_newton_cg_every": 10,
    "bfgs_qf_newton_cg_max_step_norm": 0.25,
    "bfgs_qf_newton_cg_reset_bfgs_hessian": False,
}

BFGS_BLOCK_OPTIONS = {
    "bfgs_qf_use_target_block": True,
    "bfgs_qf_target_block_trigger": "periodic_or_stalled",
    "bfgs_qf_target_block_start": 10,
    "bfgs_qf_target_block_every": 10,
    "bfgs_qf_target_block_size": 20,
    "bfgs_qf_target_block_max_step_norm": 0.05,
    "bfgs_qf_target_block_reset_bfgs_hessian": False,
}


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
        nroots_fci=4,
        run_mp2=True,
        run_ccsd=True,
        store_mo_ints=True,
    )


def config_options(name):
    """Return editable run options for one optimizer configuration."""
    if name == "scipy_bfgs":
        return {"optimizer": "BFGS"}
    if name == "scipy_lbfgsb":
        return {"optimizer": "L-BFGS-B"}
    if name == "lbfgs_qf":
        return {"optimizer": "lbfgs_qf", "lbfgs_qf_max_step_norm": 0.5}
    if name == "lbfgs_qf_hdiag":
        opts = {"optimizer": "lbfgs_qf", "lbfgs_qf_max_step_norm": 0.5}
        opts.update(LBFGS_HDIAG_OPTIONS)
        return opts
    if name == "bfgs_qf":
        return {"optimizer": "bfgs_qf", "bfgs_qf_max_step_norm": 0.5}
    if name == "bfgs_qf_hdiag":
        opts = {"optimizer": "bfgs_qf", "bfgs_qf_max_step_norm": 0.5}
        opts.update(BFGS_HDIAG_OPTIONS)
        return opts
    if name == "bfgs_qf_hdiag_ncg":
        if not RUN_NCG:
            return None
        opts = {"optimizer": "bfgs_qf", "bfgs_qf_max_step_norm": 0.5}
        opts.update(BFGS_HDIAG_OPTIONS)
        opts.update(BFGS_NCG_OPTIONS)
        return opts
    if name == "bfgs_qf_hdiag_block":
        if not RUN_BLOCK:
            return None
        opts = {"optimizer": "bfgs_qf", "bfgs_qf_max_step_norm": 0.5}
        opts.update(BFGS_HDIAG_OPTIONS)
        opts.update(BFGS_BLOCK_OPTIONS)
        return opts
    raise ValueError(f"Unknown config {name!r}")


def parse_trajectory(output):
    """Read iteration energies from qforte's printed optimizer table."""
    energies = []
    # New qforte in-house table: "  12 | -15.123 | ..."
    pipe_pat = re.compile(r"^\s*(\d+)\s*\|\s*([+-]?\d+\.\d+)")
    # SciPy callback table: "     12        -15.123..."
    space_pat = re.compile(r"^\s*(\d+)\s+([+-]?\d+\.\d+)")
    for line in output.splitlines():
        match = pipe_pat.match(line) or space_pat.match(line)
        if match:
            try:
                energies.append(float(match.group(2)))
            except ValueError:
                pass
    return energies


def trajectory_stats(energies):
    if not energies:
        return None, None, False, None
    best = min(energies)
    largest_drop = 0.0
    largest_drop_iter = None
    for idx in range(1, len(energies)):
        drop = energies[idx] - energies[idx - 1]
        if drop < largest_drop:
            largest_drop = drop
            largest_drop_iter = idx + 1
    late_drop = largest_drop_iter is not None and largest_drop_iter > 75 and abs(largest_drop) > 1.0e-4
    return best, largest_drop, late_drop, largest_drop_iter


def count_nonzero(values, threshold=1.0e-10):
    return sum(1 for value in values if abs(value) > threshold)


def run_one(mol, pool_type, config_name, maxiter):
    opts = config_options(config_name)
    if opts is None:
        return None

    alg = qf.UCCNVQE(
        mol,
        computer_type="fci",
        apply_ham_as_tensor=True,
        verbose=False,
    )

    run_options = {
        "pool_type": pool_type,
        "opt_maxiter": maxiter,
        "opt_thresh": OPT_THRESH,
        "opt_ftol": OPT_FTOL,
        "use_analytic_grad": True,
        "init_amps": "zero",
    }
    run_options.update(opts)

    stream = io.StringIO()
    t0 = time.time()
    status = "ok"
    error = ""
    try:
        with contextlib.redirect_stdout(stream):
            alg.run(**run_options)
    except Exception as exc:
        status = "failed"
        error = repr(exc)
    runtime = time.time() - t0
    output = stream.getvalue()
    print(output, end="")

    result = getattr(alg, "_final_result", None)
    energies = parse_trajectory(output)
    best_energy, largest_drop, late_drop, largest_drop_iter = trajectory_stats(energies)
    final_energy = getattr(alg, "_Egs", None)
    if final_energy is not None:
        final_energy = float(final_energy)
    if best_energy is None and final_energy is not None:
        best_energy = final_energy

    return {
        "pool": pool_type,
        "config": config_name,
        "status": status,
        "final_energy": final_energy,
        "best_energy": best_energy,
        "error_vs_fci": None if final_energy is None else final_energy - mol.fci_energy,
        "final_grad_norm": getattr(result, "grad_norm", getattr(alg, "_curr_grad_norm", None)),
        "nit": getattr(result, "nit", getattr(alg, "_k_counter", None)),
        "nfev": getattr(result, "nfev", None),
        "njev": getattr(result, "njev", None),
        "nhev": getattr(result, "nhev", 0),
        "target_block_attempts": getattr(result, "target_block_attempts", 0),
        "target_block_accepted": getattr(result, "target_block_accepted", 0),
        "newton_cg_attempts": getattr(result, "newton_cg_attempts", 0),
        "newton_cg_accepted": getattr(result, "newton_cg_accepted", 0),
        "bfgs_updates": getattr(result, "bfgs_updates_accepted", None),
        "bfgs_resets": getattr(result, "bfgs_hinv_resets", None),
        "n_nonzero": count_nonzero(getattr(alg, "_tamps", [])),
        "largest_drop": largest_drop,
        "largest_drop_iter": largest_drop_iter,
        "late_drop": late_drop,
        "runtime_s": runtime,
        "error": error,
        "trajectory": energies if PRINT_TRAJECTORIES else [],
    }


def fmt(value, precision=10):
    if value is None:
        return "-"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        return f"{value:+.{precision}f}"
    return str(value)


def write_summary(results, outdir, mol):
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.md"
    with summary_path.open("w") as handle:
        handle.write("# bfgs_qf Sandbox Summary\n\n")
        handle.write(f"RHF energy:  {mol.hf_energy:+.12f}\n\n")
        handle.write(f"MP2 energy:  {mol.mp2_energy:+.12f}\n\n")
        handle.write(f"CCSD energy: {mol.ccsd_energy:+.12f}\n\n")
        handle.write(f"FCI energy:  {mol.fci_energy:+.12f}\n\n")
        if hasattr(mol, "fci_energy_list"):
            handle.write("Lowest FCI roots:\n\n")
            for idx, energy in enumerate(mol.fci_energy_list[:4]):
                handle.write(f"- root {idx}: {energy:+.12f}\n")
            handle.write("\n")

        handle.write(
            "| pool | config | status | final E | best E | err FCI | ||g|| | "
            "nit | f/g/Hv | nonzero | BLOCK | NCG | BFGS upd/reset | max drop |\n"
        )
        handle.write(
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for row in results:
            handle.write(
                f"| {row['pool']} | {row['config']} | {row['status']} | "
                f"{fmt(row['final_energy'])} | {fmt(row['best_energy'])} | "
                f"{fmt(row['error_vs_fci'], 3)} | {fmt(row['final_grad_norm'], 3)} | "
                f"{row['nit']} | {row['nfev']}/{row['njev']}/{row['nhev']} | "
                f"{row['n_nonzero']} | "
                f"{row['target_block_attempts']}/{row['target_block_accepted']} | "
                f"{row['newton_cg_attempts']}/{row['newton_cg_accepted']} | "
                f"{row['bfgs_updates']}/{row['bfgs_resets']} | "
                f"{fmt(row['largest_drop'], 3)} @ {row['largest_drop_iter']} |\n"
            )

        failures = [row for row in results if row["status"] != "ok"]
        if failures:
            handle.write("\n## Failures\n\n")
            for row in failures:
                handle.write(f"- {row['pool']} / {row['config']}: {row['error']}\n")

    print(f"\nWrote summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxiter", type=int, default=MAXITER)
    parser.add_argument("--quick", action="store_true", help="Run only SD with bfgs_qf and bfgs_qf_hdiag.")
    args = parser.parse_args()

    pools = ["SD"] if args.quick else POOLS_TO_RUN
    configs = ["bfgs_qf", "bfgs_qf_hdiag"] if args.quick else CONFIGS_TO_RUN

    mol = build_beh2()
    results = []
    for pool_type in pools:
        for config_name in configs:
            print("\n" + "=" * 80)
            print(f"pool={pool_type}  config={config_name}")
            print("=" * 80)
            row = run_one(mol, pool_type, config_name, args.maxiter)
            if row is not None:
                results.append(row)
                print(
                    f"summary: E={fmt(row['final_energy'])} "
                    f"||g||={fmt(row['final_grad_norm'], 3)} "
                    f"nit={row['nit']} "
                    f"BLOCK={row['target_block_attempts']}/{row['target_block_accepted']} "
                    f"NCG={row['newton_cg_attempts']}/{row['newton_cg_accepted']}"
                )

    outdir = Path(__file__).resolve().parent / "results"
    write_summary(results, outdir, mol)


if __name__ == "__main__":
    main()
