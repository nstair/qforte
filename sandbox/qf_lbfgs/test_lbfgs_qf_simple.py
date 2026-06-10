"""Small, user-editable lbfgs_qf VQE example."""

import qforte as qf


USE_HESSIAN_DIAG = False

USE_NEWTON_CG = True

USE_TARGET_BLOCK = False
USE_NEGATIVE_CURVATURE_ESCAPE = False

USE_MP2_INIT_AMPS = False
N_FCI_ROOTS = 4

PRIMARY_POOL_ORDER = "none"              # "none", "mp2_amps", "gradients"
SECONDARY_POOL_ORDER = "shell"               # "lexical", "shell"
GENERAL_EX_POOL_ORDER = "particle_hole_first" # "default", "particle_hole_first"

LBFGS_OPTIONS = {
    "lbfgs_qf_memory": 6,
    "lbfgs_qf_max_step_norm": 0.5,
    # Other general lbfgs_qf options you can uncomment/tune:
    # "lbfgs_qf_max_ls": 20,
    # "lbfgs_qf_c1": 1.0e-4,
    # "lbfgs_qf_alpha0": 1.0,
    # "lbfgs_qf_max_abs_step": None,
    # "lbfgs_qf_curvature_tol": 1.0e-12,
    # "lbfgs_qf_step_tol": 1.0e-12,
    # "lbfgs_qf_stall_window": 5,
    # "lbfgs_qf_stall_gnorm_ratio": 0.8,
    # "lbfgs_qf_print_aux": True,
    # "lbfgs_qf_use_gradient_energy": True,
}

HESSIAN_DIAG_OPTIONS = {
    "lbfgs_qf_hdiag_mode": "abs",
    # Other Hessian-diagonal preconditioner options:
    # "lbfgs_qf_hdiag_start": 1,
    # "lbfgs_qf_hdiag_stop": None,
    # "lbfgs_qf_hdiag_update_freq": 1,
    # "lbfgs_qf_hdiag_floor": 1.0e-3,
    # "lbfgs_qf_hdiag_method": "analytic",  # or "finite_difference"/"fd", "mp2"
    # "lbfgs_qf_hdiag_fd_step": 1.0e-4,
}

NEWTON_CG_OPTIONS = {
    "lbfgs_qf_newton_cg_trigger": "periodic_or_stalled",
    "lbfgs_qf_newton_cg_start": 10,
    "lbfgs_qf_newton_cg_every": 10,
    "lbfgs_qf_newton_cg_max_step_norm": 0.5,
    # Other Newton-CG acceleration options:
    # "lbfgs_qf_newton_cg_trigger": "stalled",
    # "lbfgs_qf_newton_cg_maxiter": 10,
    # "lbfgs_qf_newton_cg_tol": 1.0e-3,
    # "lbfgs_qf_newton_cg_fd_delta": 1.0e-4,
    # "lbfgs_qf_newton_cg_level_shift": 1.0e-3,
    # "lbfgs_qf_newton_cg_max_abs_step": None,
    # "lbfgs_qf_newton_cg_armijo_c1": 1.0e-4,
    # "lbfgs_qf_newton_cg_max_ls": 20,
    "lbfgs_qf_newton_cg_reset_lbfgs_history": False,
    "lbfgs_qf_newton_cg_min_gnorm": None,
}

TARGET_BLOCK_OPTIONS = {
    "lbfgs_qf_target_block_trigger": "periodic_or_stalled",
    "lbfgs_qf_target_block_start": 8,
    "lbfgs_qf_target_block_every": 8,
    "lbfgs_qf_target_block_size": 40, # default heuristic caps at 20
    # finite-difference options for the target Hessian block:
    # choose "forward", "backward", or "central"
    "lbfgs_qf_target_block_fd_type": "forward",
    "lbfgs_qf_target_block_fd_delta": 1.0e-4,
    "lbfgs_qf_target_block_max_step_norm": 0.05,
    # Other target-block options:
    # "lbfgs_qf_target_block_trigger": "stalled",
    # "lbfgs_qf_target_block_min_size": 4,
    "lbfgs_qf_target_block_max_size": 40, # default 10
    # "lbfgs_qf_target_block_size_frac": 0.10,
    # "lbfgs_qf_target_block_selection": "preconditioned_gradient",
    # "lbfgs_qf_target_block_eig_floor": 1.0e-3,
    # "lbfgs_qf_target_block_max_abs_step": None,
    # "lbfgs_qf_target_block_armijo_c1": 1.0e-4,
    # "lbfgs_qf_target_block_max_ls": 20,
    "lbfgs_qf_target_block_reset_lbfgs_history": False,
    # "lbfgs_qf_target_block_require_descent": True,
}

NEGATIVE_CURVATURE_ESCAPE_OPTIONS = {
    "lbfgs_qf_escape_negcurv_trigger": "persistent",
    "lbfgs_qf_escape_negcurv_persist_count": 3,
    "lbfgs_qf_escape_negcurv_eig_thresh": -1.0e-5,
    "lbfgs_qf_escape_negcurv_step": 0.1, # default 0.05
    "lbfgs_qf_escape_negcurv_max_escapes": 20,
    # Other negative-curvature escape options:
    # "lbfgs_qf_escape_negcurv_trigger": "stalled_or_persistent",
    # "lbfgs_qf_escape_negcurv_gnorm_thresh": 1.0e-3,
    # "lbfgs_qf_escape_negcurv_max_ls": 12,
    # "lbfgs_qf_escape_negcurv_shrink": 0.5,
    # "lbfgs_qf_escape_negcurv_reset_lbfgs_history": True,
    # "lbfgs_qf_escape_negcurv_require_energy_decrease": True,
}




def assert_close(label, value, reference, tol=1.0e-8):
    diff = abs(value - reference)
    print(f"{label:24s} value={value:+.12f}  reference={reference:+.12f}  diff={diff:.3e}")
    if diff > tol:
        raise AssertionError(f"{label} differs by {diff:.3e}")


def lbfgs_qf_run_options():
    if USE_TARGET_BLOCK and USE_NEWTON_CG:
        raise ValueError("Use either target-block acceleration or Newton-CG, not both.")
    if USE_NEGATIVE_CURVATURE_ESCAPE and not USE_TARGET_BLOCK:
        raise ValueError("Negative-curvature escape requires target-block acceleration.")

    options = dict(LBFGS_OPTIONS)

    options["lbfgs_qf_use_hessian_diag"] = USE_HESSIAN_DIAG
    if USE_HESSIAN_DIAG:
        options.update(HESSIAN_DIAG_OPTIONS)

    options["lbfgs_qf_use_target_block"] = USE_TARGET_BLOCK
    if USE_TARGET_BLOCK:
        options.update(TARGET_BLOCK_OPTIONS)
        options["lbfgs_qf_escape_negative_curvature"] = USE_NEGATIVE_CURVATURE_ESCAPE
        if USE_NEGATIVE_CURVATURE_ESCAPE:
            options.update(NEGATIVE_CURVATURE_ESCAPE_OPTIONS)

    options["lbfgs_qf_use_newton_cg"] = USE_NEWTON_CG
    if USE_NEWTON_CG:
        options.update(NEWTON_CG_OPTIONS)

    return options


def run_lbfgs_qf_test():
    print("\n== lbfgs_qf VQE ==")
    print(f"use_hessian_diag: {USE_HESSIAN_DIAG}")
    print(f"use_target_block: {USE_TARGET_BLOCK}")
    print(f"use_negcurv_escape: {USE_NEGATIVE_CURVATURE_ESCAPE}")
    print(f"use_newton_cg:    {USE_NEWTON_CG}")
    print(f"use_mp2_init_amps: {USE_MP2_INIT_AMPS}")
    print(f"n_fci_roots:       {N_FCI_ROOTS}")
    print(f"primary_pool_order:    {PRIMARY_POOL_ORDER}")
    print(f"secondary_pool_order:  {SECONDARY_POOL_ORDER}")
    print(f"general_ex_pool_order: {GENERAL_EX_POOL_ORDER}")

    geom = [
        ("H", (0.0, 0.0, -1.0)),
        ("Be", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 1.0)),
    ]


    # geom = [
    #     ("H", (0.0, 0.0, 1.0)),
    #     ("H", (0.0, 0.0, 2.0)),
    #     ("H", (0.0, 0.0, 3.0)),
    #     ("H", (0.0, 0.0, 4.0)),
    #     ("H", (0.0, 0.0, 5.0)),
    #     ("H", (0.0, 0.0, 6.0)),
    # ]

    mol = qf.system_factory(
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

    alg = qf.UCCNVQE(
        mol,
        computer_type="fci",
        apply_ham_as_tensor=True,
        verbose=False,
    )

    pool_str = "SD"
    # pool_str = "SDT"
    # pool_str = "1-UpCCGSD"
    # For the old 1-UpCCGSDx ordering, use:
    # pool_str = "1-UpCCGSD"
    # GENERAL_EX_POOL_ORDER = "particle_hole_first"
    # pool_str = "GSD"

    run_options = {
        "opt_thresh": 1.0e-5,
        "opt_ftol": 1.0e-8,
        "opt_maxiter": 200,
        "pool_type": pool_str,
        "optimizer": "lbfgs_qf",
        "use_analytic_grad": True,
        "init_amps": "mp2" if USE_MP2_INIT_AMPS else "zero",
        "primary_pool_order": PRIMARY_POOL_ORDER,
        "secondary_pool_order": SECONDARY_POOL_ORDER,
        "general_ex_pool_order": GENERAL_EX_POOL_ORDER,
    }
    run_options.update(lbfgs_qf_run_options())
    alg.run(**run_options)

    print(f"\nAlgorithm ground state energy: {alg.get_gs_energy():.12f}")
    print(f"FCI energy:                    {mol.fci_energy:.12f}")
    print("\nReference energies")
    print("------------------")
    print(f"RHF energy:                    {mol.hf_energy:.12f}")
    print(f"MP2 energy:                    {mol.mp2_energy:.12f}")
    print(f"CCSD energy:                   {mol.ccsd_energy:.12f}")

    fci_roots = mol.fci_energy_list[:N_FCI_ROOTS]
    print(f"\nLowest {len(fci_roots)} FCI energies")
    print("---------------------")
    for root, energy in enumerate(fci_roots):
        print(f"FCI root {root}:                   {energy:.12f}")

    if USE_TARGET_BLOCK:
        print(f"Target-block attempts:         {alg._final_result.target_block_attempts}")
        print(f"Target-block accepted:         {alg._final_result.target_block_accepted}")
    if USE_NEGATIVE_CURVATURE_ESCAPE:
        print(f"Neg-curvature escapes tried:   {alg._final_result.escape_negcurv_attempts}")
        print(f"Neg-curvature escapes accepted:{alg._final_result.escape_negcurv_accepted}")
    if USE_NEWTON_CG:
        print(f"Newton-CG attempts:            {alg._final_result.newton_cg_attempts}")
        print(f"Newton-CG accepted:            {alg._final_result.newton_cg_accepted}")
    # print(f"Final amplitudes: {alg._tamps}")
    # assert_close("lbfgs_qf energy", alg.get_gs_energy(), mol.fci_energy)


def main():
    run_lbfgs_qf_test()


if __name__ == "__main__":
    main()
