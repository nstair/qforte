"""Sandbox checks for the in-house lbfgs_qf optimizer."""

import time
import numpy as np

import qforte as qf


def _build_molecule(geom, system_options):
    return qf.system_factory(
        build_type=system_options["build_type"],
        mol_geometry=geom,
        basis=system_options["basis"],
        symmetry=system_options["symmetry"],
        multiplicity=system_options["multiplicity"],
        charge=system_options["charge"],
        run_fci=system_options["run_fci"],
        run_ccsd=system_options["run_ccsd"],
        store_mo_ints=system_options["store_mo_ints"],
        build_df_ham=system_options["build_df_ham"],
    )


def _make_uccnvqe(system, algorithm_options):
    return qf.UCCNVQE(
        system,
        computer_type=algorithm_options["computer_type"],
        apply_ham_as_tensor=algorithm_options["apply_ham_as_tensor"],
    )


def _base_run_options(algorithm_options, pool_type, optimizer):
    return {
        "opt_thresh": algorithm_options["opt_thresh"],
        "opt_ftol": algorithm_options["opt_ftol"],
        "opt_maxiter": algorithm_options["opt_maxiter"],
        "pool_type": pool_type,
        "optimizer": optimizer,
        "use_analytic_grad": algorithm_options["use_analytic_grad"],
        "noise_factor": algorithm_options["noise_factor"],
    }


def _lbfgs_qf_run_options(algorithm_options, use_hdiag=False, use_newton_cg=False):
    return {
        "lbfgs_qf_use_hessian_diag": use_hdiag,
        "lbfgs_qf_hdiag_method": algorithm_options["lbfgs_qf_hdiag_method"],
        "lbfgs_qf_hdiag_fd_step": algorithm_options["lbfgs_qf_hdiag_fd_step"],
        "lbfgs_qf_hdiag_mode": algorithm_options["lbfgs_qf_hdiag_mode"],
        "lbfgs_qf_hdiag_floor": algorithm_options["lbfgs_qf_hdiag_floor"],
        "lbfgs_qf_max_step_norm": algorithm_options["lbfgs_qf_max_step_norm"],
        "lbfgs_qf_max_abs_step": algorithm_options["lbfgs_qf_max_abs_step"],
        "lbfgs_qf_memory": algorithm_options["lbfgs_qf_memory"],
        "lbfgs_qf_alpha0": algorithm_options["lbfgs_qf_alpha0"],
        "lbfgs_qf_max_ls": algorithm_options["lbfgs_qf_max_ls"],
        "lbfgs_qf_use_newton_cg": use_newton_cg,
        "lbfgs_qf_newton_cg_trigger": algorithm_options["lbfgs_qf_newton_cg_trigger"],
        "lbfgs_qf_newton_cg_start": algorithm_options["lbfgs_qf_newton_cg_start"],
        "lbfgs_qf_newton_cg_every": algorithm_options["lbfgs_qf_newton_cg_every"],
        "lbfgs_qf_newton_cg_maxiter": algorithm_options["lbfgs_qf_newton_cg_maxiter"],
        "lbfgs_qf_newton_cg_tol": algorithm_options["lbfgs_qf_newton_cg_tol"],
        "lbfgs_qf_newton_cg_fd_delta": algorithm_options["lbfgs_qf_newton_cg_fd_delta"],
        "lbfgs_qf_newton_cg_level_shift": algorithm_options["lbfgs_qf_newton_cg_level_shift"],
        "lbfgs_qf_newton_cg_max_step_norm": algorithm_options["lbfgs_qf_newton_cg_max_step_norm"],
        "lbfgs_qf_newton_cg_max_abs_step": algorithm_options["lbfgs_qf_newton_cg_max_abs_step"],
        "lbfgs_qf_newton_cg_armijo_c1": algorithm_options["lbfgs_qf_newton_cg_armijo_c1"],
        "lbfgs_qf_newton_cg_max_ls": algorithm_options["lbfgs_qf_newton_cg_max_ls"],
        "lbfgs_qf_newton_cg_reset_lbfgs_history": algorithm_options["lbfgs_qf_newton_cg_reset_lbfgs_history"],
        "lbfgs_qf_newton_cg_min_gnorm": algorithm_options["lbfgs_qf_newton_cg_min_gnorm"],
        "lbfgs_qf_stall_window": algorithm_options["lbfgs_qf_stall_window"],
        "lbfgs_qf_stall_gnorm_ratio": algorithm_options["lbfgs_qf_stall_gnorm_ratio"],
    }


def _run_uccnvqe(
    mol,
    algorithm_options,
    pool_type,
    optimizer,
    use_hdiag=False,
    use_newton_cg=False,
    opt_maxiter=None,
):
    alg = _make_uccnvqe(mol, algorithm_options)
    run_options = _base_run_options(algorithm_options, pool_type, optimizer)
    if opt_maxiter is not None:
        run_options["opt_maxiter"] = opt_maxiter
    if optimizer.lower() == "lbfgs_qf":
        run_options.update(
            _lbfgs_qf_run_options(
                algorithm_options,
                use_hdiag=use_hdiag,
                use_newton_cg=use_newton_cg,
            )
        )

    alg.run(**run_options)
    return alg


def _gradient_fd_hdiag(alg, params, step):
    params = np.array(params, dtype=float)
    h_diag = np.zeros_like(params)
    for mu in range(len(params)):
        params_p = params.copy()
        params_m = params.copy()
        params_p[mu] += step
        params_m[mu] -= step
        grad_p = alg.gradient_ary_feval(params_p)
        grad_m = alg.gradient_ary_feval(params_m)
        h_diag[mu] = (grad_p[mu] - grad_m[mu]) / (2.0 * step)
    return h_diag


def test_hessian_diag_validation(
    mol,
    algorithm_options,
    pool_type,
    reference_step,
    tolerance,
    label,
):
    print(f"\n== Hessian diagonal validation: {label} ==")
    alg = _run_uccnvqe(
        mol,
        algorithm_options,
        pool_type=pool_type,
        optimizer="lbfgs_qf",
        opt_maxiter=0,
    )
    params = np.zeros(len(alg._tamps))
    print(f"Number of pool amplitudes: {len(params)}")

    t0 = time.perf_counter()
    h_diag = alg.hessian_diag_ary_feval(params)
    analytic_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    h_fd = _gradient_fd_hdiag(alg, params, step=reference_step)
    fd_time = time.perf_counter() - t0

    abs_diff = np.abs(h_diag - h_fd)
    worst = int(np.argmax(abs_diff))
    speedup = fd_time / analytic_time if analytic_time > 0.0 else np.inf

    print(f"analytic hdiag:          {h_diag}")
    print(f"gradient-FD reference:   {h_fd}")
    print(f"worst index {worst}: diff = {abs_diff[worst]:.6e}")
    print(f"analytic hdiag time:     {analytic_time:.6f} s")
    print(f"gradient-FD hdiag time:  {fd_time:.6f} s")
    print(f"analytic speedup:        {speedup:.2f}x")

    if abs_diff[worst] > tolerance:
        raise AssertionError(
            f"Hessian diagonal mismatch at index {worst}: "
            f"{h_diag[worst]} vs {h_fd[worst]}"
        )


def test_hess_vec_fd_validation(
    mol,
    algorithm_options,
    pool_type,
    fd_delta,
    tolerance,
    label,
):
    print(f"\n== Hessian-vector finite-difference validation: {label} ==")
    alg = _run_uccnvqe(
        mol,
        algorithm_options,
        pool_type=pool_type,
        optimizer="lbfgs_qf",
        opt_maxiter=0,
    )

    rng = np.random.default_rng(7)
    x = rng.normal(scale=0.03, size=len(alg._tamps))
    v = rng.normal(size=len(alg._tamps))
    hv = alg._lbfgs_qf_hess_vec_fd(x, v, fd_delta)

    if hv.shape != x.shape:
        raise AssertionError(f"Hv shape {hv.shape} does not match x shape {x.shape}.")
    if not np.all(np.isfinite(hv)):
        raise AssertionError("Hessian-vector product contains non-finite values.")

    eps = fd_delta / np.linalg.norm(v)
    e_p = alg.energy_feval(x + eps * v)
    e_0 = alg.energy_feval(x)
    e_m = alg.energy_feval(x - eps * v)
    second_dir_fd = (e_p - 2.0 * e_0 + e_m) / (eps * eps)
    second_dir_hv = float(np.dot(v, hv))
    diff = abs(second_dir_fd - second_dir_hv)

    print(f"v.Hv from gradient FD:   {second_dir_hv:+.12e}")
    print(f"v.Hv from energy FD:     {second_dir_fd:+.12e}")
    print(f"absolute difference:     {diff:.3e}")

    if diff > tolerance:
        raise AssertionError(
            f"Hessian-vector directional check differs by {diff:.3e}."
        )


def test_newton_cg_direction_descent(
    mol,
    algorithm_options,
    pool_type,
    label,
):
    print(f"\n== Newton-CG descent-direction check: {label} ==")
    alg = _run_uccnvqe(
        mol,
        algorithm_options,
        pool_type=pool_type,
        optimizer="lbfgs_qf",
        use_newton_cg=True,
        opt_maxiter=0,
    )

    rng = np.random.default_rng(11)
    x = rng.normal(scale=0.03, size=len(alg._tamps))
    g = np.asarray(alg.gradient_ary_feval(x), dtype=float)
    p = alg._lbfgs_qf_newton_cg_direction(x, g)

    if p is None:
        print("Newton-CG direction was rejected cleanly.")
        return

    gdotp = float(np.dot(g, p))
    print(f"g.p = {gdotp:+.12e}")
    if gdotp >= 0.0:
        raise AssertionError("Newton-CG direction is not a descent direction.")

    e0 = alg.energy_feval(x)
    alpha = 1.0
    lowered = False
    for _ in range(20):
        if alg.energy_feval(x + alpha * p) < e0:
            lowered = True
            break
        alpha *= 0.5

    if not lowered:
        raise AssertionError("Backtracking along the Newton-CG direction did not lower the energy.")


def _run_optimizer(
    mol,
    algorithm_options,
    pool_type,
    name,
    optimizer,
    use_hdiag,
    use_newton_cg=False,
):
    alg = _run_uccnvqe(
        mol,
        algorithm_options,
        pool_type=pool_type,
        optimizer=optimizer,
        use_hdiag=use_hdiag,
        use_newton_cg=use_newton_cg,
    )
    final_grad_norm = np.linalg.norm(alg.gradient_ary_feval(alg._tamps))
    result = getattr(alg, "_final_result", None)
    return {
        "name": name,
        "energy": alg.get_gs_energy(),
        "grad_norm": final_grad_norm,
        "nit": getattr(result, "nit", alg._k_counter),
        "nfev": getattr(result, "nfev", None),
        "njev": getattr(result, "njev", None),
        "nhev": getattr(result, "nhev", None),
        "newton_cg_attempts": getattr(result, "newton_cg_attempts", 0),
        "newton_cg_accepted": getattr(result, "newton_cg_accepted", 0),
        "converged": getattr(result, "success", final_grad_norm < alg._opt_thresh),
    }


def test_optimizer_smoke(
    mol,
    algorithm_options,
    pool_type,
    energy_tolerance,
    label,
):
    print(f"\n== Optimizer smoke comparison: {label} ==")
    rows = [
        _run_optimizer(
            mol,
            algorithm_options,
            pool_type,
            "SciPy L-BFGS-B",
            "L-BFGS-B",
            False,
        ),
        _run_optimizer(
            mol,
            algorithm_options,
            pool_type,
            "lbfgs_qf",
            "lbfgs_qf",
            False,
        ),
        _run_optimizer(
            mol,
            algorithm_options,
            pool_type,
            "lbfgs_qf + hdiag",
            "lbfgs_qf",
            True,
        ),
    ]

    ref_energy = rows[0]["energy"]
    for row in rows:
        print(
            f"{row['name']:18s} E = {row['energy']:+16.10f}  "
            f"||g|| = {row['grad_norm']:.3e}  nit = {row['nit']}  "
            f"nfev = {row['nfev']}  njev = {row['njev']}  nhev = {row['nhev']}  "
            f"NCG = {row['newton_cg_attempts']}/{row['newton_cg_accepted']}  "
            f"converged = {row['converged']}"
        )
        if abs(row["energy"] - ref_energy) > energy_tolerance:
            raise AssertionError(
                f"{row['name']} final energy differs from SciPy L-BFGS-B by "
                f"{row['energy'] - ref_energy:+.6e} Eh."
            )


def test_lbfgs_qf_hybrid_smoke(
    mol,
    algorithm_options,
    pool_type,
    energy_tolerance,
    label,
):
    print(f"\n== lbfgs_qf hybrid smoke comparison: {label} ==")
    plain = _run_optimizer(
        mol,
        algorithm_options,
        pool_type,
        "lbfgs_qf",
        "lbfgs_qf",
        False,
    )
    hybrid = _run_optimizer(
        mol,
        algorithm_options,
        pool_type,
        "lbfgs_qf + NCG",
        "lbfgs_qf",
        False,
        use_newton_cg=True,
    )

    for row in [plain, hybrid]:
        print(
            f"{row['name']:18s} E = {row['energy']:+16.10f}  "
            f"||g|| = {row['grad_norm']:.3e}  nit = {row['nit']}  "
            f"nfev = {row['nfev']}  njev = {row['njev']}  nhev = {row['nhev']}  "
            f"NCG = {row['newton_cg_attempts']}/{row['newton_cg_accepted']}"
        )

    if abs(plain["energy"] - hybrid["energy"]) > energy_tolerance:
        raise AssertionError(
            "Hybrid lbfgs_qf final energy differs from plain lbfgs_qf by "
            f"{hybrid['energy'] - plain['energy']:+.6e} Eh."
        )
    if hybrid["newton_cg_attempts"] < 1:
        raise AssertionError("Hybrid lbfgs_qf did not attempt a Newton-CG correction.")


def test_lbfgs_qf_run_option_convention(mol, algorithm_options, pool_type, label):
    print(f"\n== lbfgs_qf run-option convention: {label} ==")

    try:
        qf.UCCNVQE(
            mol,
            computer_type=algorithm_options["computer_type"],
            apply_ham_as_tensor=algorithm_options["apply_ham_as_tensor"],
            lbfgs_qf_memory=3,
        )
    except TypeError as exc:
        print(f"constructor rejected lbfgs_qf option as expected: {exc}")
    else:
        raise AssertionError("lbfgs_qf_* options should be passed to run(), not the constructor.")

    alg = _make_uccnvqe(mol, algorithm_options)
    run_options = _base_run_options(algorithm_options, pool_type, "lbfgs_qf")
    run_options["opt_maxiter"] = 0
    run_options.update(_lbfgs_qf_run_options(algorithm_options, use_hdiag=True))
    run_options["lbfgs_qf_memory"] = 3
    run_options["lbfgs_qf_max_step_norm"] = 0.25
    alg.run(**run_options)

    if alg._lbfgs_qf_memory != 3:
        raise AssertionError("lbfgs_qf_memory was not applied from run().")
    if alg._lbfgs_qf_max_step_norm != 0.25:
        raise AssertionError("lbfgs_qf_max_step_norm was not applied from run().")
    if not alg._lbfgs_qf_use_hessian_diag:
        raise AssertionError("lbfgs_qf_use_hessian_diag was not applied from run().")


def test_beh2_tucc_comparison(
    mol,
    algorithm_options,
    pool_type,
    energy_tolerance,
    label,
):
    print(f"\n== BeH2 tUCC comparison: {label} ==")
    rows = [
        _run_optimizer(
            mol,
            algorithm_options,
            pool_type,
            "SciPy L-BFGS-B",
            "L-BFGS-B",
            False,
        ),
        _run_optimizer(
            mol,
            algorithm_options,
            pool_type,
            "lbfgs_qf",
            "lbfgs_qf",
            False,
        ),
        _run_optimizer(
            mol,
            algorithm_options,
            pool_type,
            "lbfgs_qf + hdiag",
            "lbfgs_qf",
            True,
        ),
        _run_optimizer(
            mol,
            algorithm_options,
            pool_type,
            "lbfgs_qf + hdiag + NCG",
            "lbfgs_qf",
            True,
            use_newton_cg=True,
        ),
    ]

    ref_energy = rows[0]["energy"]
    for row in rows:
        print(
            f"{row['name']:24s} E = {row['energy']:+16.10f}  "
            f"||g|| = {row['grad_norm']:.3e}  nit = {row['nit']}  "
            f"nfev = {row['nfev']}  njev = {row['njev']}  nhev = {row['nhev']}  "
            f"NCG = {row['newton_cg_attempts']}/{row['newton_cg_accepted']}"
        )
        if abs(row["energy"] - ref_energy) > energy_tolerance:
            raise AssertionError(
                f"{row['name']} final energy differs from SciPy L-BFGS-B by "
                f"{row['energy'] - ref_energy:+.6e} Eh."
            )


def main():
    run_api_convention_check = True
    run_hess_vec_validation = True
    run_newton_cg_direction_check = True
    run_hessian_diag_validation = True
    run_h4_optimizer_smoke = False
    run_h4_hybrid_smoke = True
    run_beh2_gsd_optimizer_comparison = False

    system_options = {
        "build_type": "psi4",
        "basis": "sto-3g",
        "symmetry": "c1",
        "multiplicity": 1,
        "charge": 0,
        "run_fci": 1,
        "run_ccsd": False,
        "store_mo_ints": True,
        "build_df_ham": False,
    }

    algorithm_options = {
        "computer_type": "fci",
        "apply_ham_as_tensor": False,
        "use_analytic_grad": True,
        "noise_factor": 0.0,
        "opt_thresh": 1.0e-6,
        "opt_ftol": 1.0e-8,
        "opt_maxiter": 80,
        "lbfgs_qf_memory": 10,
        "lbfgs_qf_max_ls": 20,
        "lbfgs_qf_alpha0": 1.0,
        "lbfgs_qf_max_step_norm": 0.5,
        "lbfgs_qf_max_abs_step": None,
        "lbfgs_qf_hdiag_method": "analytic",
        "lbfgs_qf_hdiag_fd_step": 1.0e-4,
        "lbfgs_qf_hdiag_mode": "abs",
        "lbfgs_qf_hdiag_floor": 1.0e-3,
        "lbfgs_qf_newton_cg_trigger": "periodic",
        "lbfgs_qf_newton_cg_start": 1,
        "lbfgs_qf_newton_cg_every": 5,
        "lbfgs_qf_newton_cg_maxiter": 10,
        "lbfgs_qf_newton_cg_tol": 1.0e-3,
        "lbfgs_qf_newton_cg_fd_delta": 1.0e-4,
        "lbfgs_qf_newton_cg_level_shift": 1.0e-3,
        "lbfgs_qf_newton_cg_max_step_norm": 0.25,
        "lbfgs_qf_newton_cg_max_abs_step": None,
        "lbfgs_qf_newton_cg_armijo_c1": 1.0e-4,
        "lbfgs_qf_newton_cg_max_ls": 20,
        "lbfgs_qf_newton_cg_reset_lbfgs_history": True,
        "lbfgs_qf_newton_cg_min_gnorm": None,
        "lbfgs_qf_stall_window": 5,
        "lbfgs_qf_stall_gnorm_ratio": 0.8,
    }

    # pool_type = "GSD"
    pool_type = "SD"

    # kmax = 1
    # pool_type = f'{kmax}-UpCCGSD'

    # pool_type = "SD"
    energy_tolerance = 1.0e-5
    hessian_diag_tolerance = 5.0e-4
    hessian_diag_reference_step = 5.0e-5
    hess_vec_tolerance = 1.0e-2
    hess_vec_fd_delta = 1.0e-4

    h4_geom = [
        ("H", (0.0, 0.0, 1.0)),
        ("H", (0.0, 0.0, 2.0)),
        ("H", (0.0, 0.0, 3.0)),
        ("H", (0.0, 0.0, 4.0)),
    ]
    beh2_geom = [
        ("H", (0.0, 0.0, -1.3264)),
        ("Be", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 1.3264)),
    ]

    h4_mol = _build_molecule(h4_geom, system_options)
    h4_label = f"H4/{system_options['basis']} {system_options['build_type']} {pool_type}"

    if run_api_convention_check:
        test_lbfgs_qf_run_option_convention(
            h4_mol,
            algorithm_options,
            pool_type,
            h4_label,
        )

    if run_hess_vec_validation:
        test_hess_vec_fd_validation(
            h4_mol,
            algorithm_options,
            pool_type,
            hess_vec_fd_delta,
            hess_vec_tolerance,
            h4_label,
        )

    if run_newton_cg_direction_check:
        test_newton_cg_direction_descent(
            h4_mol,
            algorithm_options,
            pool_type,
            h4_label,
        )

    if run_hessian_diag_validation:
        test_hessian_diag_validation(
            h4_mol,
            algorithm_options,
            pool_type,
            hessian_diag_reference_step,
            hessian_diag_tolerance,
            h4_label,
        )

    if run_h4_optimizer_smoke:
        test_optimizer_smoke(
            h4_mol,
            algorithm_options,
            pool_type,
            energy_tolerance,
            h4_label,
        )

    if run_h4_hybrid_smoke:
        test_lbfgs_qf_hybrid_smoke(
            h4_mol,
            algorithm_options,
            pool_type,
            energy_tolerance,
            h4_label,
        )

    if run_beh2_gsd_optimizer_comparison:
        beh2_pool_type = "GSD"
        beh2_mol = _build_molecule(beh2_geom, system_options)
        beh2_label = f"BeH2/{system_options['basis']} {system_options['build_type']} {beh2_pool_type}"
        test_beh2_tucc_comparison(
            beh2_mol,
            algorithm_options,
            beh2_pool_type,
            energy_tolerance,
            beh2_label,
        )
    else:
        print("\nBeH2/GSD comparison skipped. Set run_beh2_gsd_optimizer_comparison = True in main() to run it.")


if __name__ == "__main__":
    main()
