"""Very small, toggleable bfgs_qf diagnostic.

Edit the options in the first block and run:

    python sandbox/test_bfgs_qf/test_bfgs_qf_simple.py

This is intentionally a simple sandbox script, not a formal test.
"""

import numpy as np
import qforte as qf


# ============================================================
# Main toggles
# ============================================================


POOL_TYPE = "SD"
# POOL_TYPE = "SDT"
# POOL_TYPE = "SDTQ"
# POOL_TYPE = "1-UpCCGSD"
# POOL_TYPE = "2-UpCCGSD"
# POOL_TYPE = "3-UpCCGSD"

OPTIMIZER = "bfgs_qf"
# OPTIMIZER = "BFGS"
# OPTIMIZER = "L-BFGS-B"
# OPTIMIZER = "lbfgs_qf"

MAXITER = 1000
OPT_THRESH = 1.0e-4
OPT_FTOL = 1.0e-8

INIT_AMPS = "mp2"  # "zero" or "mp2"

# Pool ordering changes the factorized UCC product order, not pool membership.
PRIMARY_POOL_ORDER = "none"       # "none", "mp2_amps", "gradients"
SECONDARY_POOL_ORDER = "shell"  # "lexical", "shell"
GENERAL_EX_POOL_ORDER = "particle_hole_first" # "default", "particle_hole_first"

USE_HESSIAN_DIAG = True
USE_NEWTON_CG = True
USE_TARGET_BLOCK = False
USE_NEGATIVE_CURVATURE_ESCAPE = False

N_FCI_ROOTS = 1


# ============================================================
# bfgs_qf options
# ============================================================

BFGS_QF_OPTIONS = {
    "bfgs_qf_max_step_norm": 0.5,
    # "bfgs_qf_max_abs_step": None,
    # "bfgs_qf_max_ls": 20,
    # "bfgs_qf_armijo_c1": 1.0e-4,
    # "bfgs_qf_curvature_tol": 1.0e-12,
    # "bfgs_qf_reset_on_bad_curvature": False,
    # "bfgs_qf_reset_on_nondescent": True,
    # "bfgs_qf_max_params_dense": 5000,
    # "bfgs_qf_use_gradient_energy": True,
}

HESSIAN_DIAG_OPTIONS = {
    "bfgs_qf_use_hessian_diag": USE_HESSIAN_DIAG,
    "bfgs_qf_hdiag_mode": "abs",
    "bfgs_qf_hdiag_floor": 1.0e-3,
    # "bfgs_qf_hdiag_start": 1,
    # "bfgs_qf_hdiag_stop": 1,
    # "bfgs_qf_hdiag_method": "analytic",  # or "finite_difference"/"fd", "mp2"
}

NEWTON_CG_OPTIONS = {
    "bfgs_qf_use_newton_cg": USE_NEWTON_CG,
    "bfgs_qf_newton_cg_trigger": "stalled", #periodic_or_stalled
    "bfgs_qf_newton_cg_start": 10,
    "bfgs_qf_newton_cg_every": 10,
    "bfgs_qf_newton_cg_max_step_norm": 0.25,
    "bfgs_qf_newton_cg_reset_bfgs_hessian": True, # False
}

TARGET_BLOCK_OPTIONS = {
    "bfgs_qf_use_target_block": USE_TARGET_BLOCK,
    "bfgs_qf_target_block_trigger": "periodic_or_stalled",
    "bfgs_qf_target_block_start": 10,
    "bfgs_qf_target_block_every": 10,
    "bfgs_qf_target_block_size": 20,
    "bfgs_qf_target_block_fd_type": "forward",  # "forward", "backward", "central"
    "bfgs_qf_target_block_max_step_norm": 0.05,
    "bfgs_qf_target_block_reset_bfgs_hessian": False,
}

NEGATIVE_CURVATURE_OPTIONS = {
    "bfgs_qf_escape_negative_curvature": USE_NEGATIVE_CURVATURE_ESCAPE,
    "bfgs_qf_escape_negcurv_trigger": "persistent",
    "bfgs_qf_escape_negcurv_persist_count": 3,
    "bfgs_qf_escape_negcurv_step": 0.05,
}


def build_beh2():
    geom = [
        ("H", (0.0, 0.0, -1.0)),
        ("Be", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 1.0)),
    ]

    # geom = [
    #     ("H", (0.0, 0.0, -1.0)),
    #     ("Be", (0.0, 0.0, 0.0)),
    #     ("H", (0.0, 0.0, 1.0)),
    # ]
    return qf.system_factory(
        build_type="psi4",
        mol_geometry=geom,
        basis="sto-3g",
        symmetry="c1",
        run_fci=True,
        nroots_fci=N_FCI_ROOTS,
        run_mp2=True,
        run_ccsd=True,
        store_mo_ints=True,
    )


def make_run_options():
    if USE_NEWTON_CG and USE_TARGET_BLOCK:
        raise ValueError("Use either Newton-CG or target-block acceleration, not both.")
    if USE_NEGATIVE_CURVATURE_ESCAPE and not USE_TARGET_BLOCK:
        raise ValueError("Negative-curvature escape requires target-block acceleration.")

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
    }

    if OPTIMIZER.lower() == "bfgs_qf":
        options.update(BFGS_QF_OPTIONS)
        options.update(HESSIAN_DIAG_OPTIONS)
        options.update(NEWTON_CG_OPTIONS)
        options.update(TARGET_BLOCK_OPTIONS)
        options.update(NEGATIVE_CURVATURE_OPTIONS)

    return options


def largest_energy_drop(energies):
    if len(energies) < 2:
        return None, None
    drops = [energies[i] - energies[i - 1] for i in range(1, len(energies))]
    idx = int(np.argmin(drops))
    return drops[idx], idx + 2


def main():
    print("\n== Simple bfgs_qf diagnostic ==")
    print(f"pool_type:              {POOL_TYPE}")
    print(f"optimizer:              {OPTIMIZER}")
    print(f"init_amps:              {INIT_AMPS}")
    print(f"primary_pool_order:     {PRIMARY_POOL_ORDER}")
    print(f"secondary_pool_order:   {SECONDARY_POOL_ORDER}")
    print(f"general_ex_pool_order:  {GENERAL_EX_POOL_ORDER}")
    print(f"use_hessian_diag:       {USE_HESSIAN_DIAG}")
    print(f"use_newton_cg:          {USE_NEWTON_CG}")
    print(f"use_target_block:       {USE_TARGET_BLOCK}")
    print(f"use_negcurv_escape:     {USE_NEGATIVE_CURVATURE_ESCAPE}")

    mol = build_beh2()
    alg = qf.UCCNVQE(
        mol,
        computer_type="fci",
        apply_ham_as_tensor=True,
        verbose=False,
    )

    run_options = make_run_options()
    alg.run(**run_options)

    final_energy = alg.get_gs_energy()
    final_result = getattr(alg, "_final_result", None)
    grad_norm = getattr(final_result, "grad_norm", getattr(alg, "_curr_grad_norm", None))
    best_energy = min(getattr(alg, "_energies", [final_energy]))
    largest_drop, largest_drop_iter = largest_energy_drop(getattr(alg, "_energies", []))
    n_nonzero = sum(abs(t) > 1.0e-10 for t in alg._tamps)

    print("\nResult")
    print("------")
    print(f"Final energy:           {final_energy:+.12f}")
    print(f"Best stored energy:     {best_energy:+.12f}")
    print(f"Error vs FCI:           {final_energy - mol.fci_energy:+.6e}")
    print(f"Final ||g||:            {grad_norm}")
    print(f"Iterations:             {getattr(final_result, 'nit', alg._k_counter)}")
    print(f"Energy evals:           {getattr(final_result, 'nfev', None)}")
    print(f"Gradient evals:         {getattr(final_result, 'njev', None)}")
    print(f"Nonzero amplitudes:     {n_nonzero}")
    print(f"Largest stored dE:      {largest_drop} at iter {largest_drop_iter}")
    if INIT_AMPS == "mp2":
        dup_counts = sorted(
            set(getattr(alg, "_mp2_init_duplicate_counts", {}).values())
        )
        print(f"MP2 init nonzeros:      {getattr(alg, '_mp2_init_nonzero', 0)}")
        print(f"MP2 duplicate counts:   {dup_counts}")

    if final_result is not None:
        print(f"BFGS updates/skips:     {getattr(final_result, 'bfgs_updates_accepted', '-')}/"
              f"{getattr(final_result, 'bfgs_updates_skipped', '-')}")
        print(f"BFGS Hinv resets:       {getattr(final_result, 'bfgs_hinv_resets', '-')}")
        print(f"NCG attempts/accepted:  {getattr(final_result, 'newton_cg_attempts', 0)}/"
              f"{getattr(final_result, 'newton_cg_accepted', 0)}")
        print(f"BLOCK attempts/accepted:{getattr(final_result, 'target_block_attempts', 0)}/"
              f"{getattr(final_result, 'target_block_accepted', 0)}")

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
