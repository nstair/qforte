"""Very small, toggleable batched-optimization diagnostic.

Edit the options in the first block and run:

    python sandbox/test_batched_optimization/test_batched_optimization_simple.py

This is intentionally a simple sandbox script, not a formal test.  Batched
optimization requires a generalized pool with
general_ex_pool_order="particle_hole_first".
"""

import numpy as np
import qforte as qf


# ============================================================
# Main toggles
# ============================================================


# Batching needs a generalized pool.  For SD/SDT use BATCHED_OPT_TYPE = "none".
POOL_TYPE = "1-UpCCGSD"
# POOL_TYPE = "2-UpCCGSD"
# POOL_TYPE = "3-UpCCGSD"
# POOL_TYPE = "4-UpCCGSD"
# POOL_TYPE = "GSD"
# POOL_TYPE = "SD"

SYMMETRY = 'd2h'

OPTIMIZER = "bfgs_qf"
# OPTIMIZER = "lbfgs_qf"
# OPTIMIZER = "BFGS"
# OPTIMIZER = "L-BFGS-B"

MAXITER = 5000
OPT_THRESH = 1.0e-4
OPT_FTOL = 1.0e-12

INIT_AMPS = "zero"  # "zero" or "mp2"

# Pool ordering changes the factorized UCC product order, not pool membership.
PRIMARY_POOL_ORDER = "none"       # "none", "mp2_amps", "gradients"
SECONDARY_POOL_ORDER = "lexical"  # "lexical", "shell"
GENERAL_EX_POOL_ORDER = "particle_hole_first"  # "default", "particle_hole_first"


# ============================================================
# Batch toggles
# ============================================================



BATCHED_OPT_TYPE = "none"
# BATCHED_OPT_TYPE = "half_sweep"
# BATCHED_OPT_TYPE = "full_sweep"
# BATCHED_OPT_TYPE = "half_sweep_then_all"
# BATCHED_OPT_TYPE = "full_sweep_then_all"

BATCHED_OPT_CYCLES = 1
BATCH_OPT_THRESH = 1.0e-5
BATCH_MAXITER = 200
FINAL_MAXITER = MAXITER
BATCHED_OPT_VERBOSE = True


# ============================================================
# Optimizer toggles
# ============================================================


USE_HESSIAN_DIAG = False
HDIAG_METHOD = 'mp2' # analytic, mp2

USE_NEWTON_CG = False
USE_TARGET_BLOCK = False
USE_NEGATIVE_CURVATURE_ESCAPE = False

N_FCI_ROOTS = 1


# ============================================================
# Detailed bfgs_qf options
# ============================================================


# These are the currently supported bfgs_qf run kwargs, exposed here so this
# file can be used as a small manual optimizer control panel.  The high-level
# booleans above are wired into the most common on/off options below.
BFGS_QF_OPTIONS = {
    # Core dense-BFGS controls.
    "bfgs_qf_maxiter": None,          # None -> use opt_maxiter
    "bfgs_qf_gconv": None,            # None -> use opt_thresh
    "bfgs_qf_econv": None,            # None -> use opt_ftol
    "bfgs_qf_max_ls": 20,
    "bfgs_qf_c1": 1.0e-4, # 1e-4 
    "bfgs_qf_armijo_c1": None,        # None -> use bfgs_qf_c1
    "bfgs_qf_alpha0": 1.0, # 1.0
    "bfgs_qf_line_search": "armijo",
    "bfgs_qf_max_step_norm": 0.5, # 0.5
    "bfgs_qf_max_abs_step": None,
    "bfgs_qf_curvature_tol": 1.0e-12,
    "bfgs_qf_step_tol": 1.0e-12,
    "bfgs_qf_init_scale": 1.0,
    "bfgs_qf_reset_on_bad_curvature": True,
    "bfgs_qf_reset_on_nondescent": True,
    "bfgs_qf_max_params_dense": 5000,

    # Exact Hessian diagonal initialization/preconditioning.
    "bfgs_qf_use_hessian_diag": USE_HESSIAN_DIAG,
    "bfgs_qf_hdiag_start": 1,
    "bfgs_qf_hdiag_stop": 3, # WOAH!!
    "bfgs_qf_hdiag_update_freq": 1,
    "bfgs_qf_hdiag_floor": 1.0e-3,
    "bfgs_qf_hdiag_mode": "abs", # abs     # "positive", "abs", or "none"
    # "analytic"/"recursive", "finite_difference"/"fd", or "mp2".
    "bfgs_qf_hdiag_method": HDIAG_METHOD, # analytic
    "bfgs_qf_hdiag_fd_step": 1.0e-4,

    # Newton-CG acceleration.  Mutually exclusive with target block.
    "bfgs_qf_use_newton_cg": USE_NEWTON_CG,
    "bfgs_qf_newton_cg_trigger": "periodic_or_stalled", # "periodic" or "stalled"
    "bfgs_qf_newton_cg_start": 30,
    "bfgs_qf_newton_cg_every": 8,
    "bfgs_qf_newton_cg_maxiter": 5, #10 # DRIVES Ngrad cost up!
    "bfgs_qf_newton_cg_tol": 1.0e-3,
    "bfgs_qf_newton_cg_fd_delta": 1.0e-4,
    "bfgs_qf_newton_cg_level_shift": 1.0e-3,
    "bfgs_qf_newton_cg_max_step_norm": 0.25,
    "bfgs_qf_newton_cg_max_abs_step": None,
    "bfgs_qf_newton_cg_armijo_c1": 1.0e-4,
    "bfgs_qf_newton_cg_max_ls": 20,
    "bfgs_qf_newton_cg_reset_bfgs_hessian": False,
    "bfgs_qf_newton_cg_min_gnorm": None,

    # Target dense Hessian-block acceleration.  Mutually exclusive with NCG.
    "bfgs_qf_use_target_block": USE_TARGET_BLOCK,
    "bfgs_qf_target_block_trigger": "periodic_or_stalled",
    "bfgs_qf_target_block_start": 10,
    "bfgs_qf_target_block_every": 10,
    "bfgs_qf_target_block_size": 20,
    "bfgs_qf_target_block_min_size": 4,
    "bfgs_qf_target_block_max_size": 20,
    "bfgs_qf_target_block_size_frac": 0.10,
    "bfgs_qf_target_block_selection": "preconditioned_gradient",
    "bfgs_qf_target_block_fd_type": "forward",  # "forward" or "central"
    "bfgs_qf_target_block_fd_delta": 1.0e-4,
    "bfgs_qf_target_block_eig_floor": 1.0e-3,
    "bfgs_qf_target_block_max_step_norm": 0.05,
    "bfgs_qf_target_block_max_abs_step": None,
    "bfgs_qf_target_block_armijo_c1": 1.0e-4,
    "bfgs_qf_target_block_max_ls": 20,
    "bfgs_qf_target_block_reset_bfgs_hessian": False,
    "bfgs_qf_target_block_require_descent": True,

    # Negative-curvature escape inside target-block attempts.
    "bfgs_qf_escape_negative_curvature": USE_NEGATIVE_CURVATURE_ESCAPE,
    "bfgs_qf_escape_negcurv_trigger": "persistent",
    "bfgs_qf_escape_negcurv_eig_thresh": -1.0e-5,
    "bfgs_qf_escape_negcurv_persist_count": 3,
    "bfgs_qf_escape_negcurv_gnorm_thresh": 1.0e-3,
    "bfgs_qf_escape_negcurv_step": 5.0e-2,
    "bfgs_qf_escape_negcurv_max_ls": 12,
    "bfgs_qf_escape_negcurv_shrink": 0.5,
    "bfgs_qf_escape_negcurv_max_escapes": 5,
    "bfgs_qf_escape_negcurv_reset_bfgs_hessian": True,
    "bfgs_qf_escape_negcurv_require_energy_decrease": True,

    # Shared stall/logging controls used by NCG and target block.
    "bfgs_qf_stall_window": 5,
    "bfgs_qf_stall_gnorm_ratio": 0.8,
    "bfgs_qf_print_aux": True,
    "bfgs_qf_use_gradient_energy": True,
}


def build_beh2():
    # geom = [
    #     ("H", (0.0, 0.0, -1.0)),
    #     ("Be", (0.0, 0.0, 0.0)),
    #     ("H", (0.0, 0.0, 1.0)),
    # ]

    # geom = [
    #     ("N", (0.0, 0.0, -0.5)),
    #     ("N", (0.0, 0.0, 0.5)),
    # ]

    R = 1.0

    geom = [
        ("H", (0.0, 0.0, 1*R)),
        ("H", (0.0, 0.0, 2*R)),
        ("H", (0.0, 0.0, 3*R)),
        ("H", (0.0, 0.0, 4*R)),
        ("H", (0.0, 0.0, 5*R)),
        ("H", (0.0, 0.0, 6*R)),
        ("H", (0.0, 0.0, 7*R)),
        ("H", (0.0, 0.0, 8*R)),
        ("H", (0.0, 0.0, 9*R)),
        ("H", (0.0, 0.0, 10*R)),
    ]

    return qf.system_factory(
        build_type="psi4",
        mol_geometry=geom,
        basis="sto-3g",
        symmetry=SYMMETRY,
        run_fci=True,
        nroots_fci=N_FCI_ROOTS,
        run_mp2=True,
        run_ccsd=True,
        store_mo_ints=True,
    )


def qf_optimizer_options():
    """Return qforte in-house optimizer options for the selected optimizer."""
    name = OPTIMIZER.lower()
    if name == "bfgs_qf":
        options = dict(BFGS_QF_OPTIONS)
        use_ncg = bool(options["bfgs_qf_use_newton_cg"])
        use_block = bool(options["bfgs_qf_use_target_block"])
        use_escape = bool(options["bfgs_qf_escape_negative_curvature"])
        if use_ncg and use_block:
            raise ValueError("Use either Newton-CG or target-block acceleration, not both.")
        if use_escape and not use_block:
            raise ValueError("Negative-curvature escape requires target-block acceleration.")
        return options
    if name == "lbfgs_qf":
        if USE_NEWTON_CG and USE_TARGET_BLOCK:
            raise ValueError("Use either Newton-CG or target-block acceleration, not both.")
        if USE_NEGATIVE_CURVATURE_ESCAPE and not USE_TARGET_BLOCK:
            raise ValueError("Negative-curvature escape requires target-block acceleration.")
        return {
            "lbfgs_qf_memory": 10,
            "lbfgs_qf_use_hessian_diag": USE_HESSIAN_DIAG,
            "lbfgs_qf_hdiag_mode": "abs",
            "lbfgs_qf_hdiag_floor": 1.0e-3,
            # "lbfgs_qf_hdiag_method": "analytic",  # or "finite_difference"/"fd", "mp2"
            "lbfgs_qf_use_newton_cg": USE_NEWTON_CG,
            "lbfgs_qf_newton_cg_trigger": "stalled",
            "lbfgs_qf_newton_cg_start": 10,
            "lbfgs_qf_newton_cg_every": 10,
            "lbfgs_qf_newton_cg_reset_lbfgs_history": True,
            "lbfgs_qf_use_target_block": USE_TARGET_BLOCK,
            "lbfgs_qf_target_block_trigger": "periodic_or_stalled",
            "lbfgs_qf_target_block_start": 10,
            "lbfgs_qf_target_block_every": 10,
            "lbfgs_qf_target_block_size": 20,
            "lbfgs_qf_target_block_fd_type": "forward",
            "lbfgs_qf_target_block_max_step_norm": 0.05,
            "lbfgs_qf_target_block_reset_lbfgs_history": False,
            "lbfgs_qf_escape_negative_curvature": USE_NEGATIVE_CURVATURE_ESCAPE,
            "lbfgs_qf_escape_negcurv_trigger": "persistent",
            "lbfgs_qf_escape_negcurv_persist_count": 3,
            "lbfgs_qf_escape_negcurv_step": 0.05,
            "lbfgs_qf_use_gradient_energy": True,
        }
    return {}


def make_run_options():
    options = {
        "pool_type": POOL_TYPE,
        "optimizer": OPTIMIZER,
        "opt_maxiter": MAXITER,
        "opt_thresh": OPT_THRESH,
        "opt_ftol": OPT_FTOL,
        "use_analytic_grad": True,
        "init_amps": INIT_AMPS,
        "primary_pool_order": PRIMARY_POOL_ORDER,
        "secondary_pool_order": SECONDARY_POOL_ORDER,
        "general_ex_pool_order": GENERAL_EX_POOL_ORDER,
        "batched_opt_type": BATCHED_OPT_TYPE,
        "batched_opt_cycles": BATCHED_OPT_CYCLES,
        "batch_opt_thresh": BATCH_OPT_THRESH,
        "batched_opt_batch_maxiter": BATCH_MAXITER,
        "batched_opt_final_maxiter": FINAL_MAXITER,
        "batched_opt_verbose": BATCHED_OPT_VERBOSE,
    }
    options.update(qf_optimizer_options())
    return options


def batch_sizes(alg):
    ph = 0
    gen = 0
    for top in alg._tops:
        sq_op = alg._pool_obj[top][1]
        if alg._is_clean_particle_hole_excitation(sq_op) is not None:
            ph += 1
        else:
            gen += 1
    return ph, gen


def best_energy_from_batches(alg, final_energy):
    history = getattr(alg, "_batched_opt_history", [])
    if not history:
        return min(getattr(alg, "_energies", [final_energy]))
    return min([entry["final_energy"] for entry in history] + [final_energy])


def main():
    run_options = make_run_options()
    opt_prefix = "bfgs_qf" if OPTIMIZER.lower() == "bfgs_qf" else "lbfgs_qf"

    print("\n== Simple batched optimization diagnostic ==")
    print(f"pool_type:              {POOL_TYPE}")
    print(f"optimizer:              {OPTIMIZER}")
    print(f"init_amps:              {INIT_AMPS}")
    print(f"primary_pool_order:     {PRIMARY_POOL_ORDER}")
    print(f"secondary_pool_order:   {SECONDARY_POOL_ORDER}")
    print(f"general_ex_pool_order:  {GENERAL_EX_POOL_ORDER}")
    print(f"batched_opt_type:       {BATCHED_OPT_TYPE}")
    print(f"batched_opt_cycles:     {BATCHED_OPT_CYCLES}")
    print(f"batch/final opt_thresh: {BATCH_OPT_THRESH}/{OPT_THRESH}")
    print(f"batch/final maxiter:    {BATCH_MAXITER}/{FINAL_MAXITER}")
    print(f"use_hessian_diag:       {run_options.get(f'{opt_prefix}_use_hessian_diag', '-')}")
    print(f"use_newton_cg:          {run_options.get(f'{opt_prefix}_use_newton_cg', '-')}")
    print(f"use_target_block:       {run_options.get(f'{opt_prefix}_use_target_block', '-')}")
    print(
        "use_negcurv_escape:     "
        f"{run_options.get(f'{opt_prefix}_escape_negative_curvature', '-')}"
    )
    if OPTIMIZER.lower() == "bfgs_qf":
        print(f"bfgs alpha0:            {run_options['bfgs_qf_alpha0']}")
        print(f"bfgs max_step_norm:     {run_options['bfgs_qf_max_step_norm']}")
        print(f"bfgs max_abs_step:      {run_options['bfgs_qf_max_abs_step']}")
        print(f"bfgs hdiag mode/floor:  {run_options['bfgs_qf_hdiag_mode']}/"
              f"{run_options['bfgs_qf_hdiag_floor']}")

    mol = build_beh2()
    alg = qf.UCCNVQE(
        mol,
        computer_type="fci",
        apply_ham_as_tensor=True,
        verbose=False,
    )

    alg.run(**run_options)

    final_energy = alg.get_gs_energy()
    final_result = getattr(alg, "_final_result", None)
    grad_norm = getattr(final_result, "grad_norm", getattr(alg, "_curr_grad_norm", None))
    best_energy = best_energy_from_batches(alg, final_energy)
    n_nonzero = sum(abs(t) > 1.0e-10 for t in alg._tamps)
    ph_size, gen_size = batch_sizes(alg)

    print("\nResult")
    print("------")
    print(f"Final energy:           {final_energy:+.12f}")
    print(f"Best stored energy:     {best_energy:+.12f}")
    print(f"Error vs FCI:           {final_energy - mol.fci_energy:+.6e}")
    print(f"Final ||g||:            {grad_norm}")
    print(f"Iterations:             {getattr(final_result, 'nit', alg._k_counter)}")
    print(f"Energy evals:           {getattr(final_result, 'nfev', None)}")
    print(f"Gradient evals:         {getattr(final_result, 'njev', None)}")
    print(f"PH/GEN parameters:      {ph_size}/{gen_size}")
    print(f"Nonzero amplitudes:     {n_nonzero}")
    if INIT_AMPS == "mp2":
        dup_counts = sorted(
            set(getattr(alg, "_mp2_init_duplicate_counts", {}).values())
        )
        print(f"MP2 init nonzeros:      {getattr(alg, '_mp2_init_nonzero', 0)}")
        print(f"MP2 duplicate counts:   {dup_counts}")

    batch_history = getattr(alg, "_batched_opt_history", [])
    if batch_history:
        print("\nBatch history")
        print("-------------")
        for item in batch_history:
            threshold = item.get("opt_thresh")
            threshold_text = f"{threshold:.1e}" if threshold is not None else "-"
            print(
                f"{item['batch_number']:2d} {item['batch_label']:>3s}: "
                f"active={item['active_size']:4d}  "
                f"E={item['initial_energy']:+.10f} -> "
                f"{item['final_energy']:+.10f}  "
                f"red|g|={item['final_reduced_grad_norm']:.3e}  "
                f"full|g|={item['final_full_grad_norm']:.3e}  "
                f"thr={threshold_text}  "
                f"nit={item['iterations']}"
            )

    print("\nReference energies")
    print("------------------")
    print(f"RHF:                    {mol.hf_energy:+.12f}")
    print(f"MP2:                    {mol.mp2_energy:+.12f}")
    print(f"CCSD:                   {mol.ccsd_energy:+.12f}")
    print(f"FCI:                    {mol.fci_energy:+.12f}")

    if hasattr(mol, "fci_energy_list"):
        print("\nFCI roots")
        print("---------")
        for root, energy in enumerate(mol.fci_energy_list[:N_FCI_ROOTS]):
            print(f"root {root}:                 {energy:+.12f}")


if __name__ == "__main__":
    main()

