import copy
import numpy as np
from scipy.optimize import minimize, OptimizeResult

_LBFGS_QF_DEFAULTS = {
    "lbfgs_qf_memory": 10,
    "lbfgs_qf_max_ls": 20,
    "lbfgs_qf_c1": 1.0e-4,
    "lbfgs_qf_alpha0": 1.0,
    "lbfgs_qf_max_step_norm": None,
    "lbfgs_qf_max_abs_step": None,
    "lbfgs_qf_curvature_tol": 1.0e-12,
    "lbfgs_qf_step_tol": 1.0e-12,
    "lbfgs_qf_use_hessian_diag": False,
    "lbfgs_qf_hdiag_start": 1,
    "lbfgs_qf_hdiag_stop": None,
    "lbfgs_qf_hdiag_update_freq": 1,
    "lbfgs_qf_hdiag_floor": 1.0e-3,
    "lbfgs_qf_hdiag_mode": "positive",
    "lbfgs_qf_hdiag_method": "analytic",
    "lbfgs_qf_hdiag_fd_step": 1.0e-4,
    "lbfgs_qf_use_newton_cg": False,
    "lbfgs_qf_newton_cg_trigger": "stalled",
    "lbfgs_qf_newton_cg_start": 10,
    "lbfgs_qf_newton_cg_every": 10,
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
    "lbfgs_qf_use_target_block": False,
    "lbfgs_qf_target_block_trigger": "stalled",
    "lbfgs_qf_target_block_start": 10,
    "lbfgs_qf_target_block_every": 10,
    "lbfgs_qf_target_block_size": None,
    "lbfgs_qf_target_block_min_size": 4,
    "lbfgs_qf_target_block_max_size": 20,
    "lbfgs_qf_target_block_size_frac": 0.10,
    "lbfgs_qf_target_block_selection": "preconditioned_gradient",
    "lbfgs_qf_target_block_fd_type": "forward",
    "lbfgs_qf_target_block_fd_delta": 1.0e-4,
    "lbfgs_qf_target_block_eig_floor": 1.0e-3,
    "lbfgs_qf_target_block_max_step_norm": 0.05,
    "lbfgs_qf_target_block_max_abs_step": None,
    "lbfgs_qf_target_block_armijo_c1": 1.0e-4,
    "lbfgs_qf_target_block_max_ls": 20,
    "lbfgs_qf_target_block_reset_lbfgs_history": True,
    "lbfgs_qf_target_block_require_descent": True,
    "lbfgs_qf_escape_negative_curvature": False,
    "lbfgs_qf_escape_negcurv_trigger": "persistent",
    "lbfgs_qf_escape_negcurv_eig_thresh": -1.0e-5,
    "lbfgs_qf_escape_negcurv_persist_count": 3,
    "lbfgs_qf_escape_negcurv_gnorm_thresh": 1.0e-3,
    "lbfgs_qf_escape_negcurv_step": 5.0e-2,
    "lbfgs_qf_escape_negcurv_max_ls": 12,
    "lbfgs_qf_escape_negcurv_shrink": 0.5,
    "lbfgs_qf_escape_negcurv_max_escapes": 5,
    "lbfgs_qf_escape_negcurv_reset_lbfgs_history": True,
    "lbfgs_qf_escape_negcurv_require_energy_decrease": True,
    "lbfgs_qf_stall_window": 5,
    "lbfgs_qf_stall_gnorm_ratio": 0.8,
    "lbfgs_qf_print_aux": True,
    "lbfgs_qf_use_gradient_energy": True,
}

_LBFGS_QF_OPTION_ATTRS = {
    key: f"_{key}" for key in _LBFGS_QF_DEFAULTS
}

_BFGS_QF_DEFAULTS = {
    "bfgs_qf_maxiter": None,
    "bfgs_qf_gconv": None,
    "bfgs_qf_econv": None,
    "bfgs_qf_max_ls": 20,
    "bfgs_qf_c1": 1.0e-4,
    "bfgs_qf_armijo_c1": None,
    "bfgs_qf_alpha0": 1.0,
    "bfgs_qf_line_search": "armijo",
    "bfgs_qf_max_step_norm": 0.5,
    "bfgs_qf_max_abs_step": None,
    "bfgs_qf_curvature_tol": 1.0e-12,
    "bfgs_qf_step_tol": 1.0e-12,
    "bfgs_qf_init_scale": 1.0,
    "bfgs_qf_reset_on_bad_curvature": False,
    "bfgs_qf_reset_on_nondescent": True,
    "bfgs_qf_max_params_dense": 5000,
    "bfgs_qf_use_hessian_diag": False,
    "bfgs_qf_hdiag_start": 1,
    "bfgs_qf_hdiag_stop": 1,
    "bfgs_qf_hdiag_update_freq": 1,
    "bfgs_qf_hdiag_floor": 1.0e-3,
    "bfgs_qf_hdiag_mode": "positive",
    "bfgs_qf_hdiag_method": "analytic",
    "bfgs_qf_hdiag_fd_step": 1.0e-4,
    "bfgs_qf_use_newton_cg": False,
    "bfgs_qf_newton_cg_trigger": "stalled",
    "bfgs_qf_newton_cg_start": 10,
    "bfgs_qf_newton_cg_every": 10,
    "bfgs_qf_newton_cg_maxiter": 10,
    "bfgs_qf_newton_cg_tol": 1.0e-3,
    "bfgs_qf_newton_cg_fd_delta": 1.0e-4,
    "bfgs_qf_newton_cg_level_shift": 1.0e-3,
    "bfgs_qf_newton_cg_max_step_norm": 0.25,
    "bfgs_qf_newton_cg_max_abs_step": None,
    "bfgs_qf_newton_cg_armijo_c1": 1.0e-4,
    "bfgs_qf_newton_cg_max_ls": 20,
    "bfgs_qf_newton_cg_reset_bfgs_hessian": True,
    "bfgs_qf_newton_cg_min_gnorm": None,
    "bfgs_qf_use_target_block": False,
    "bfgs_qf_target_block_trigger": "stalled",
    "bfgs_qf_target_block_start": 10,
    "bfgs_qf_target_block_every": 10,
    "bfgs_qf_target_block_size": None,
    "bfgs_qf_target_block_min_size": 4,
    "bfgs_qf_target_block_max_size": 20,
    "bfgs_qf_target_block_size_frac": 0.10,
    "bfgs_qf_target_block_selection": "preconditioned_gradient",
    "bfgs_qf_target_block_fd_type": "forward",
    "bfgs_qf_target_block_fd_delta": 1.0e-4,
    "bfgs_qf_target_block_eig_floor": 1.0e-3,
    "bfgs_qf_target_block_max_step_norm": 0.05,
    "bfgs_qf_target_block_max_abs_step": None,
    "bfgs_qf_target_block_armijo_c1": 1.0e-4,
    "bfgs_qf_target_block_max_ls": 20,
    "bfgs_qf_target_block_reset_bfgs_hessian": True,
    "bfgs_qf_target_block_require_descent": True,
    "bfgs_qf_escape_negative_curvature": False,
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
    "bfgs_qf_stall_window": 5,
    "bfgs_qf_stall_gnorm_ratio": 0.8,
    "bfgs_qf_print_aux": True,
    "bfgs_qf_use_gradient_energy": True,
}

_BFGS_QF_OPTION_ATTRS = {
    key: f"_{key}" for key in _BFGS_QF_DEFAULTS
}


def set_lbfgs_qf_options(self, options):
    """Apply qforte in-house BFGS/L-BFGS keyword options supplied to run()."""
    all_option_attrs = {}
    all_option_attrs.update(_LBFGS_QF_OPTION_ATTRS)
    all_option_attrs.update(_BFGS_QF_OPTION_ATTRS)

    unknown_options = [key for key in options if key not in all_option_attrs]
    if unknown_options:
        raise TypeError(f"Unexpected run keyword option(s): {unknown_options}")

    self._qf_optimizer_supplied_options = set(options)

    run_options = copy.deepcopy(_LBFGS_QF_DEFAULTS)
    run_options.update(
        {key: value for key, value in options.items() if key in _LBFGS_QF_OPTION_ATTRS}
    )

    for key, value in run_options.items():
        setattr(self, _LBFGS_QF_OPTION_ATTRS[key], value)

    run_options = copy.deepcopy(_BFGS_QF_DEFAULTS)
    run_options.update(
        {key: value for key, value in options.items() if key in _BFGS_QF_OPTION_ATTRS}
    )

    for key, value in run_options.items():
        setattr(self, _BFGS_QF_OPTION_ATTRS[key], value)


def _cnot_count_for_current_ansatz(self):
    """Return CNOT count without building Uvqc when an analytical path exists."""
    if getattr(self, "_computer_type", None) == "fock" and hasattr(self, "build_Uvqc"):
        return self.build_Uvqc().get_num_cnots()
    if hasattr(self, "count_jw_cnot_ladders"):
        return self.count_jw_cnot_ladders()
    if hasattr(self, "build_Uvqc"):
        return self.build_Uvqc().get_num_cnots()
    return "N/A"


def _qf_active_hdiag_method(self):
    """Return the Hessian-diagonal method for the active qforte optimizer."""
    optimizer_name = getattr(self, "_optimizer", "").lower()
    if optimizer_name == "bfgs_qf":
        return getattr(
            self,
            "_bfgs_qf_hdiag_method",
            getattr(self, "_lbfgs_qf_hdiag_method", "analytic"),
        )
    return getattr(
        self,
        "_lbfgs_qf_hdiag_method",
        getattr(self, "_bfgs_qf_hdiag_method", "analytic"),
    )


def diis(diis_max_dim, t_diis, e_diis):
    """This function implements the direct inversion of iterative subspace
    (DIIS) convergence accelerator. Draws heavy insiration from Daniel
    Smith's ccsd_diss.py code in psi4 numpy
    """

    if len(t_diis) > diis_max_dim:
        del t_diis[0]
        del e_diis[0]

    diis_dim = len(t_diis) - 1

    # Construct diis B matrix (following Crawford Group github tutorial)
    B = np.ones((diis_dim+1, diis_dim+1)) * -1
    bsol = np.zeros(diis_dim+1)

    B[-1, -1] = 0.0
    bsol[-1] = -1.0
    for i, ei in enumerate(e_diis):
        for j, ej in enumerate(e_diis):
            B[i,j] = np.dot(np.real(ei), np.real(ej))

    B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

    x = np.linalg.lstsq(B, bsol, rcond=None)[0][:-1]

    t_new = np.zeros(( len(t_diis[0]) ))
    for l in range(diis_dim):
        temp_ary = x[l] * np.asarray(t_diis[l+1])
        t_new = np.add(t_new, temp_ary)

    return copy.deepcopy(list(np.real(t_new)))

def jacobi_solver(self):
    """
    This function minimizes the norm of the residual/gradient vector
    by using a quasi-Newton update procedure for the amplitudes
    """

    t_diis = [copy.deepcopy(self._tamps)]
    e_diis = []
    Ek0 = self.energy_feval(self._tamps)

    if self.__class__.__name__ in ['UCCNPQE', 'SPQE']:
        print('\n    k iteration         Energy               dE           Nrvec ev      Nrm ev*         ||r||')
    elif self.__class__.__name__ in ['UCCNVQE', 'ADAPTVQE']:
        print('\n    k iteration         Energy               dE           Ngvec ev      Ngm ev*         ||g||')
    print('---------------------------------------------------------------------------------------------------', flush=True)

    for k in range(1, self._opt_maxiter+1):

        t_old = copy.deepcopy(self._tamps)

        #do regular update
        if self.__class__.__name__ in ['UCCNPQE', 'SPQE']:
            r_k = self.get_residual_vector(self._tamps)
        elif self.__class__.__name__ in ['UCCNVQE', 'ADAPTVQE']:
            r_k = self.gradient_ary_feval(self._tamps)
        r_k = self.get_res_over_mpdenom(r_k)

        self._tamps = list(np.add(self._tamps, r_k))

        Ek = self.energy_feval(self._tamps)
        dE = Ek - Ek0
        Ek0 = Ek

        if self.__class__.__name__ in ['UCCNPQE', 'SPQE']:
            print(f'     {k:7}        {Ek:+12.10f}      {dE:+12.10f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {self._res_vec_norm:+12.10f}')
            if(self._res_vec_norm < self._opt_thresh):
                self._Egs = Ek
                break
        elif self.__class__.__name__ in ['UCCNVQE', 'ADAPTVQE']:
            print(f'     {k:7}        {Ek:+12.10f}      {dE:+12.10f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {self._curr_grad_norm:+12.10f}')
            if(self._curr_grad_norm < self._opt_thresh):
                self._Egs = Ek
                break

        t_diis.append(copy.deepcopy(self._tamps))
        e_diis.append(np.subtract(copy.deepcopy(self._tamps), t_old))

        if(k >= 1 and self._diis_max_dim >= 2):
            self._tamps = diis(self._diis_max_dim, t_diis, e_diis)

    self._Egs = Ek
    if k == self._opt_maxiter:
        print("\nMaximum number of Jacobi iterations reached!")
    if hasattr(self, '_energies'):
        self._energies.append(Ek)
    if hasattr(self, '_n_classical_params'):
        self._n_classical_params = len(self._tamps)
    if hasattr(self, '_n_pauli_measures_k'):
        self._n_pauli_measures_k += self._Nl*k * (2*len(self._tamps) + 1)
    if hasattr(self, '_n_pauli_trm_measures'):
        self._n_pauli_trm_measures += 2*self._Nl*k*len(self._tamps) + self._Nl*k
    if hasattr(self, '_n_pauli_trm_measures_lst'):
        self._n_pauli_trm_measures_lst.append(self._n_pauli_measures_k)
    if hasattr(self, '_n_cnot'):
        self._n_cnot = _cnot_count_for_current_ansatz(self)
    if hasattr(self, '_n_cnot_lst'):
        self._n_cnot_lst.append(_cnot_count_for_current_ansatz(self))

def scipy_solver(self, function_to_minimize):

    # Construct arguments to hand to the minimizer.
    opts = {}

    # Options common to all minimization algorithms
    opts['disp'] = True
    opts['maxiter'] = self._opt_maxiter

    # Optimizer-specific options
    if self._optimizer.lower() in ['bfgs', 'cg', 'l-bfgs-b']:
        opts['gtol'] = self._opt_thresh
    if self._optimizer.lower() == 'nelder-mead':
        opts['fatol'] = self._opt_thresh
        opts['adaptive'] = True
    if self._optimizer.lower() in ['powell', 'l-bfgs-b', 'slsqp']:
        opts['ftol'] = self._opt_thresh

    x0 = copy.deepcopy(self._tamps)
    self._prev_energy = self.energy_feval(x0)
    self._k_counter = 0

    res = minimize(function_to_minimize, x0,
            method=self._optimizer,
            options=opts,
            callback=self.report_iteration)

    if(res.success):
        print('  => Minimization successful!')
    else:
        print('  => WARNING: minimization result may not be tightly converged.')

    self._tamps = list(res.x)
    self._Egs = self.energy_feval(self._tamps)
    if hasattr(self, '_energies'):
        self._energies.append(self._Egs)
    if hasattr(self, '_n_classical_params'):
        self._n_classical_params = len(self._tamps)
    if hasattr(self, '_n_pauli_measures_k'):
        self._n_pauli_measures_k += self._Nl*self._k_counter * (2*len(self._tamps) + 1)
    if hasattr(self, '_n_pauli_trm_measures'):
        self._n_pauli_trm_measures += 2*self._Nl*self._k_counter*len(self._tamps) + self._Nl*self._k_counter
    if hasattr(self, '_n_pauli_trm_measures_lst'):
        self._n_pauli_trm_measures_lst.append(self._n_pauli_measures_k)
    if hasattr(self, '_n_cnot'):
        self._n_cnot = _cnot_count_for_current_ansatz(self)
    if hasattr(self, '_n_cnot_lst'):
        self._n_cnot_lst.append(_cnot_count_for_current_ansatz(self))


def _lbfgs_qf_inf_if_none(value):
    """Return np.inf for None-valued optional step caps."""
    if value is None:
        return np.inf
    return value


def _lbfgs_qf_capped_step(step, max_abs_step, max_step_norm):
    """Apply elementwise and Euclidean-norm caps to a proposed step."""
    capped_step = np.array(step, dtype=float, copy=True)

    if np.isfinite(max_abs_step):
        capped_step = np.clip(capped_step, -max_abs_step, max_abs_step)

    step_norm = np.linalg.norm(capped_step)
    if np.isfinite(max_step_norm) and step_norm > max_step_norm and step_norm > 0.0:
        capped_step *= max_step_norm / step_norm

    return capped_step


def _lbfgs_qf_hdiag_is_due(k, use_hdiag, hdiag_start, hdiag_stop, hdiag_update_freq, cached_hdiag, hdiag_mode):
    """Return whether the Hessian diagonal should be evaluated at iteration k."""
    if not use_hdiag or hdiag_mode == "none":
        return False
    if k < hdiag_start:
        return False
    if hdiag_stop is not None and k > hdiag_stop:
        return False
    if cached_hdiag is None:
        return True
    return (k - hdiag_start) % hdiag_update_freq == 0


def _lbfgs_qf_regularized_hdiag(h_diag, hdiag_floor, hdiag_mode):
    """Build the effective positive Hessian diagonal used as H0^{-1} = diag(1/h)."""
    if hdiag_mode == "positive":
        return np.maximum(h_diag, hdiag_floor)
    if hdiag_mode == "abs":
        return np.maximum(np.abs(h_diag), hdiag_floor)
    if hdiag_mode == "none":
        return None
    raise ValueError(
        'Unknown lbfgs_qf_hdiag_mode. Expected "positive", "abs", or "none"; '
        f"got {hdiag_mode!r}."
    )


def _lbfgs_qf_hess_vec_fd(self, x, v, fd_delta):
    """Finite-difference Hessian-vector product using analytical gradients."""
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)

    vnorm = np.linalg.norm(v)
    if vnorm == 0.0:
        return np.zeros_like(v)

    fd_delta = float(fd_delta)
    if fd_delta <= 0.0:
        raise ValueError("lbfgs_qf_newton_cg_fd_delta must be positive.")

    eps = fd_delta / vnorm
    gp = np.asarray(self.gradient_ary_feval(x + eps * v), dtype=float)
    gm = np.asarray(self.gradient_ary_feval(x - eps * v), dtype=float)

    self._lbfgs_qf_newton_cg_hv_evals = getattr(
        self, "_lbfgs_qf_newton_cg_hv_evals", 0
    ) + 1

    return np.real((gp - gm) / (2.0 * eps))


def _qf_gradient_energy_feval(self, x, objective=None, use_gradient_energy=True):
    """Return (E, g, nfev_inc, njev_inc), using combined backend work if available.

    The evaluation counters are logical optimizer/resource counters.  A combined
    backend call still consumes both an energy value and a gradient vector from
    the optimizer's perspective, so it increments both counters even though it
    avoids a separate expensive energy build internally.
    """
    if objective is None:
        objective = self.energy_feval

    if use_gradient_energy:
        try:
            result = self.gradient_ary_feval(x, return_energy=True)
        except TypeError as exc:
            if "return_energy" not in str(exc):
                raise
        else:
            if isinstance(result, tuple) and len(result) == 2:
                E, g = result
                return float(np.real(E)), np.asarray(g, dtype=float), 1, 1

    E = float(np.real(objective(x)))
    g = np.asarray(self.gradient_ary_feval(x), dtype=float)
    return E, g, 1, 1


def _qf_gradient_hdiag_feval(
    self,
    x,
    objective=None,
    return_energy=False,
    need_hdiag=False,
    use_gradient_energy=True,
):
    """Return gradient plus optional energy/hdiag, using a bundled backend if possible."""
    if objective is None:
        objective = self.energy_feval

    hdiag_method = _qf_active_hdiag_method(self)
    can_bundle_hdiag = hdiag_method in ["analytic", "recursive"]

    if need_hdiag and can_bundle_hdiag and hasattr(self, "derivative_ary_feval"):
        try:
            bundle = self.derivative_ary_feval(
                x,
                return_energy=return_energy,
                return_gradient=True,
                return_hessian_diag=True,
            )
        except TypeError as exc:
            if (
                "return_energy" not in str(exc)
                and "return_gradient" not in str(exc)
                and "return_hessian_diag" not in str(exc)
            ):
                raise
        else:
            if isinstance(bundle, dict) and "gradient" in bundle and "hessian_diag" in bundle:
                E = None
                nfev = 0
                if return_energy:
                    if "energy" not in bundle:
                        E = float(np.real(objective(x)))
                    else:
                        E = float(np.real(bundle["energy"]))
                    nfev = 1
                return (
                    E,
                    np.asarray(bundle["gradient"], dtype=float),
                    np.asarray(bundle["hessian_diag"], dtype=float),
                    nfev,
                    1,
                    True,
                )

    if return_energy:
        E, g, nfev, njev = _qf_gradient_energy_feval(
            self,
            x,
            objective=objective,
            use_gradient_energy=use_gradient_energy,
        )
    else:
        E = None
        g = np.asarray(self.gradient_ary_feval(x), dtype=float)
        nfev = 0
        njev = 1

    return E, g, None, nfev, njev, False


def _qf_hessian_diag_feval(self, x, expected_len, optimizer_name):
    """Evaluate and validate the Hessian diagonal, returning (h_diag, njev_inc)."""
    if not hasattr(self, 'hessian_diag_ary_feval'):
        raise ValueError(
            f'{optimizer_name}_use_hessian_diag=True requires '
            'hessian_diag_ary_feval(params) on the VQE class.'
        )

    grad_count_before = getattr(self, '_res_vec_evals', None)
    h_diag = np.array(self.hessian_diag_ary_feval(x), dtype=float)
    njev_inc = 0
    if grad_count_before is not None:
        njev_inc = max(
            0,
            getattr(self, '_res_vec_evals', grad_count_before) - grad_count_before,
        )
    if len(h_diag) != expected_len:
        raise ValueError(
            'hessian_diag_ary_feval(params) returned a vector of length '
            f'{len(h_diag)}, but there are {expected_len} amplitudes.'
        )
    return h_diag, njev_inc


def _lbfgs_qf_apply_pcg_preconditioner(r, h_eff):
    """Apply optional positive diagonal preconditioner for Newton-CG."""
    if h_eff is None:
        return r.copy()

    h_eff = np.asarray(h_eff, dtype=float)
    if h_eff.shape != r.shape or not np.all(np.isfinite(h_eff)) or np.any(h_eff <= 0.0):
        return r.copy()

    return r / h_eff


def _lbfgs_qf_newton_cg_direction(self, x, g, h_eff=None):
    """Return a truncated Newton-CG correction direction, or None if unsafe."""
    x = np.asarray(x, dtype=float)
    g = np.asarray(g, dtype=float)

    maxiter = int(getattr(self, "_lbfgs_qf_newton_cg_maxiter", 10))
    cg_tol = float(getattr(self, "_lbfgs_qf_newton_cg_tol", 1.0e-3))
    fd_delta = float(getattr(self, "_lbfgs_qf_newton_cg_fd_delta", 1.0e-4))
    level_shift = float(getattr(self, "_lbfgs_qf_newton_cg_level_shift", 1.0e-3))
    max_step_norm = _lbfgs_qf_inf_if_none(
        getattr(self, "_lbfgs_qf_newton_cg_max_step_norm", 0.25)
    )
    max_abs_step = _lbfgs_qf_inf_if_none(
        getattr(self, "_lbfgs_qf_newton_cg_max_abs_step", None)
    )

    gnorm = np.linalg.norm(g)
    info = {
        "cg_iters": 0,
        "hv_evals": 0,
        "negative_curvature": False,
        "residual_norm": gnorm,
        "preconditioned": h_eff is not None,
        "accepted": False,
        "reason": "not_started",
    }
    self._lbfgs_qf_newton_cg_last_info = info

    if maxiter <= 0:
        info["reason"] = "maxiter_zero"
        return None
    if gnorm == 0.0:
        info["reason"] = "zero_gradient"
        return np.zeros_like(g)

    b = -g
    s = np.zeros_like(g)
    r = b.copy()
    z = _lbfgs_qf_apply_pcg_preconditioner(r, h_eff)
    p = z.copy()
    rz_old = float(np.dot(r, z))

    if not np.isfinite(rz_old) or rz_old <= 0.0:
        info["reason"] = "bad_preconditioner"
        return None

    candidate = None
    tiny_curvature = 1.0e-14

    for it in range(maxiter):
        hv_before = getattr(self, "_lbfgs_qf_newton_cg_hv_evals", 0)
        Hp = _lbfgs_qf_hess_vec_fd(self, x, p, fd_delta)
        hv_after = getattr(self, "_lbfgs_qf_newton_cg_hv_evals", hv_before)
        info["hv_evals"] += max(0, hv_after - hv_before)

        Ap = Hp + level_shift * p
        pAp = float(np.dot(p, Ap))
        info["cg_iters"] = it + 1

        if not np.isfinite(pAp) or pAp <= tiny_curvature:
            info["negative_curvature"] = True
            info["reason"] = "negative_curvature"
            candidate = p.copy() if it == 0 else s.copy()
            break

        alpha = rz_old / pAp
        s = s + alpha * p
        r = r - alpha * Ap

        residual_norm = np.linalg.norm(r)
        info["residual_norm"] = residual_norm
        if residual_norm < cg_tol * gnorm:
            info["reason"] = "cg_converged"
            candidate = s.copy()
            break

        z = _lbfgs_qf_apply_pcg_preconditioner(r, h_eff)
        rz_new = float(np.dot(r, z))
        if not np.isfinite(rz_new) or rz_new <= 0.0:
            info["reason"] = "bad_pcg_update"
            candidate = s.copy()
            break

        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    if candidate is None:
        info["reason"] = "maxiter"
        candidate = s.copy()

    candidate = _lbfgs_qf_capped_step(candidate, max_abs_step, max_step_norm)
    step_norm = np.linalg.norm(candidate)
    info["step_norm"] = step_norm

    if step_norm == 0.0 or not np.all(np.isfinite(candidate)):
        info["reason"] = "zero_or_nonfinite_step"
        return None
    if np.dot(g, candidate) >= 0.0:
        info["reason"] = "not_descent"
        return None

    return candidate


def _lbfgs_qf_try_newton_cg_step(self, x, E, g, h_eff=None, objective=None):
    """Try an Armijo-globalized Newton-CG correction without mutating x/E/g."""
    if objective is None:
        objective = self.energy_feval

    self._lbfgs_qf_newton_cg_attempts = getattr(
        self, "_lbfgs_qf_newton_cg_attempts", 0
    ) + 1

    p = _lbfgs_qf_newton_cg_direction(self, x, g, h_eff=h_eff)
    info = copy.deepcopy(getattr(self, "_lbfgs_qf_newton_cg_last_info", {}))
    info.setdefault("hv_evals", 0)
    info.setdefault("cg_iters", 0)
    info["nfev"] = 0
    info["grad_evals"] = 2 * info["hv_evals"]
    info["accepted"] = False
    info["alpha"] = 0.0

    if p is None:
        info.setdefault("reason", "direction_rejected")
        self._lbfgs_qf_newton_cg_last_info = info
        return False, x, E, g, 0.0, 0.0, info

    gdotp = float(np.dot(g, p))
    if not np.isfinite(gdotp) or gdotp >= 0.0:
        info["reason"] = "not_descent"
        self._lbfgs_qf_newton_cg_last_info = info
        return False, x, E, g, 0.0, 0.0, info

    c1 = float(getattr(self, "_lbfgs_qf_newton_cg_armijo_c1", 1.0e-4))
    max_ls = int(getattr(self, "_lbfgs_qf_newton_cg_max_ls", 20))
    alpha = 1.0

    for _ in range(max_ls):
        step = alpha * p
        step_norm = np.linalg.norm(step)
        if step_norm == 0.0:
            break

        x_trial = x + step
        E_trial = float(np.real(objective(x_trial)))
        g_trial = None
        info["nfev"] += 1

        if E_trial < E and E_trial <= E + c1 * alpha * gdotp:
            if g_trial is None:
                g_trial = np.asarray(self.gradient_ary_feval(x_trial), dtype=float)
                info["grad_evals"] += 1
            info["accepted"] = True
            info["alpha"] = alpha
            info["step_norm"] = step_norm
            info["reason"] = "accepted"
            self._lbfgs_qf_newton_cg_accepted = getattr(
                self, "_lbfgs_qf_newton_cg_accepted", 0
            ) + 1
            self._lbfgs_qf_newton_cg_last_info = info
            return True, x_trial, E_trial, g_trial, alpha, step_norm, info

        alpha *= 0.5

    info["reason"] = "line_search_failed"
    self._lbfgs_qf_newton_cg_last_info = info
    return False, x, E, g, 0.0, 0.0, info


def _lbfgs_qf_should_attempt_newton_cg(self, k, gnorm_history):
    """Return whether the hybrid Newton-CG correction should be tried."""
    if not bool(getattr(self, "_lbfgs_qf_use_newton_cg", False)):
        return False

    trigger = getattr(self, "_lbfgs_qf_newton_cg_trigger", "stalled")
    allowed = {"never", "periodic", "stalled", "periodic_or_stalled"}
    if trigger not in allowed:
        raise ValueError(
            "Unknown lbfgs_qf_newton_cg_trigger. Expected one of "
            f"{sorted(allowed)}; got {trigger!r}."
        )
    if trigger == "never":
        return False

    start = int(getattr(self, "_lbfgs_qf_newton_cg_start", 10))
    if k < start:
        return False

    grad_norm = float(gnorm_history[-1])
    min_gnorm = getattr(self, "_lbfgs_qf_newton_cg_min_gnorm", None)
    if min_gnorm is not None and grad_norm < float(min_gnorm):
        return False

    every = max(1, int(getattr(self, "_lbfgs_qf_newton_cg_every", 10)))
    periodic = (k - start) % every == 0

    window = max(1, int(getattr(self, "_lbfgs_qf_stall_window", 5)))
    ratio = float(getattr(self, "_lbfgs_qf_stall_gnorm_ratio", 0.8))
    stalled = False
    if len(gnorm_history) > window:
        g_start = float(gnorm_history[-window - 1])
        if g_start > 0.0:
            stalled = grad_norm / g_start > ratio

    if trigger == "periodic":
        return periodic
    if trigger == "stalled":
        return stalled
    return periodic or stalled


def _lbfgs_qf_direction_from_history(g, s_hist, y_hist, rho_hist, h_eff=None):
    """Return the current L-BFGS inverse-Hessian direction without mutation."""
    g = np.asarray(g, dtype=float)

    if len(s_hist) == 0:
        if h_eff is not None:
            h_eff = np.asarray(h_eff, dtype=float)
            if h_eff.shape == g.shape and np.all(np.isfinite(h_eff)) and np.all(h_eff > 0.0):
                return -g / h_eff
        return -g

    q = g.copy()
    alpha_hist = []
    for s_i, y_i, rho_i in reversed(list(zip(s_hist, y_hist, rho_hist))):
        alpha_i = rho_i * np.dot(s_i, q)
        alpha_hist.append(alpha_i)
        q -= alpha_i * y_i

    if h_eff is not None:
        h_eff = np.asarray(h_eff, dtype=float)
        if h_eff.shape == g.shape and np.all(np.isfinite(h_eff)) and np.all(h_eff > 0.0):
            r = q / h_eff
        else:
            r = q.copy()
    else:
        s_last = s_hist[-1]
        y_last = y_hist[-1]
        yy_last = np.dot(y_last, y_last)
        gamma = np.dot(s_last, y_last) / yy_last if yy_last > 0.0 else 1.0
        r = gamma * q

    for i, (s_i, y_i, rho_i) in enumerate(zip(s_hist, y_hist, rho_hist)):
        beta_i = rho_i * np.dot(y_i, r)
        alpha_i = alpha_hist[len(s_hist) - 1 - i]
        r += s_i * (alpha_i - beta_i)

    return -r


def _lbfgs_qf_target_block_size(self, n_params):
    """Return the active Hessian block size from explicit value or heuristic."""
    n_params = int(n_params)
    if n_params <= 0:
        return 0

    explicit_size = getattr(self, "_lbfgs_qf_target_block_size", None)
    if explicit_size is not None:
        block_size = int(explicit_size)
    else:
        frac = float(getattr(self, "_lbfgs_qf_target_block_size_frac", 0.10))
        min_size = int(getattr(self, "_lbfgs_qf_target_block_min_size", 4))
        max_size = int(getattr(self, "_lbfgs_qf_target_block_max_size", 20))
        block_size = int(np.ceil(frac * n_params))
        block_size = max(min_size, block_size)
        block_size = min(max_size, block_size)

    block_size = max(1, block_size)
    return min(block_size, n_params)


def _lbfgs_qf_safe_preconditioned_gradient_score(g, h_eff):
    """Return abs(g / h_eff) if h_eff is safe; otherwise abs(g)."""
    g = np.asarray(g, dtype=float)
    if h_eff is None:
        return np.abs(g)

    h_eff = np.asarray(h_eff, dtype=float)
    if h_eff.shape != g.shape:
        return np.abs(g)

    safe_h = np.where(np.isfinite(h_eff) & (np.abs(h_eff) > 0.0), h_eff, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.abs(g / safe_h)
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def _lbfgs_qf_normalized_score(score):
    """Scale a nonnegative score vector to max 1, defensively handling zeros."""
    score = np.nan_to_num(np.asarray(score, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    score = np.abs(score)
    max_score = np.max(score) if score.size else 0.0
    if max_score <= 0.0 or not np.isfinite(max_score):
        return np.zeros_like(score)
    return score / max_score


def _lbfgs_qf_select_target_block(self, x, g, h_eff=None, lbfgs_step=None):
    """Select a reproducible active block of important parameter indices."""
    del x  # currently unused, kept in the signature for future local heuristics
    g = np.asarray(g, dtype=float)
    n_params = len(g)
    block_size = _lbfgs_qf_target_block_size(self, n_params)
    if block_size == 0:
        return np.array([], dtype=int)

    mode = getattr(self, "_lbfgs_qf_target_block_selection", "preconditioned_gradient")
    allowed = {"gradient", "preconditioned_gradient", "lbfgs_step", "mixed"}
    if mode not in allowed:
        raise ValueError(
            "Unknown lbfgs_qf_target_block_selection. Expected one of "
            f"{sorted(allowed)}; got {mode!r}."
        )

    if mode == "gradient":
        score = np.abs(g)
    elif mode == "preconditioned_gradient":
        score = _lbfgs_qf_safe_preconditioned_gradient_score(g, h_eff)
    elif mode == "lbfgs_step":
        if lbfgs_step is not None:
            lbfgs_step = np.asarray(lbfgs_step, dtype=float)
            if lbfgs_step.shape == g.shape and np.all(np.isfinite(lbfgs_step)):
                score = np.abs(lbfgs_step)
            else:
                score = _lbfgs_qf_safe_preconditioned_gradient_score(g, h_eff)
        else:
            score = _lbfgs_qf_safe_preconditioned_gradient_score(g, h_eff)
    else:
        if h_eff is None:
            score = np.abs(g)
        else:
            score1 = _lbfgs_qf_normalized_score(g)
            score2 = _lbfgs_qf_normalized_score(
                _lbfgs_qf_safe_preconditioned_gradient_score(g, h_eff)
            )
            score = 0.5 * score1 + 0.5 * score2

    score = np.nan_to_num(np.asarray(score, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if block_size >= n_params:
        block = np.arange(n_params, dtype=int)
    else:
        block = np.argpartition(-score, block_size - 1)[:block_size]
        block = np.sort(block.astype(int))

    if bool(getattr(self, "_verbose", False)):
        selected_scores = score[block] if len(block) else np.array([])
        print(
            "  => Target block indices: "
            f"{block.tolist()} scores={selected_scores.tolist()}",
            flush=True,
        )

    return block


def _lbfgs_qf_build_target_hessian_block(self, x, g, block):
    """Build a dense finite-difference Hessian block from analytical gradients."""
    fd_type = getattr(self, "_lbfgs_qf_target_block_fd_type", "forward")
    allowed = {"forward", "backward", "central"}
    if fd_type not in allowed:
        raise ValueError(
            "Unknown lbfgs_qf_target_block_fd_type. Expected one of "
            f"{sorted(allowed)}; got {fd_type!r}."
        )

    delta = float(getattr(self, "_lbfgs_qf_target_block_fd_delta", 1.0e-4))
    if delta <= 0.0:
        raise ValueError("lbfgs_qf_target_block_fd_delta must be positive.")

    x = np.asarray(x, dtype=float)
    g = np.asarray(g, dtype=float)
    block = np.asarray(block, dtype=int)
    block_size = len(block)
    H = np.zeros((block_size, block_size), dtype=float)

    for col, param_idx in enumerate(block):
        if fd_type == "forward":
            xp = x.copy()
            xp[param_idx] += delta
            gp = np.asarray(self.gradient_ary_feval(xp), dtype=float)
            H[:, col] = (gp[block] - g[block]) / delta
            self._lbfgs_qf_target_block_grad_evals = getattr(
                self, "_lbfgs_qf_target_block_grad_evals", 0
            ) + 1
        elif fd_type == "backward":
            xm = x.copy()
            xm[param_idx] -= delta
            gm = np.asarray(self.gradient_ary_feval(xm), dtype=float)
            H[:, col] = (g[block] - gm[block]) / delta
            self._lbfgs_qf_target_block_grad_evals = getattr(
                self, "_lbfgs_qf_target_block_grad_evals", 0
            ) + 1
        else:
            xp = x.copy()
            xm = x.copy()
            xp[param_idx] += delta
            xm[param_idx] -= delta
            gp = np.asarray(self.gradient_ary_feval(xp), dtype=float)
            gm = np.asarray(self.gradient_ary_feval(xm), dtype=float)
            H[:, col] = (gp[block] - gm[block]) / (2.0 * delta)
            self._lbfgs_qf_target_block_grad_evals = getattr(
                self, "_lbfgs_qf_target_block_grad_evals", 0
            ) + 2

    H = 0.5 * (H + H.T)
    return np.real(H)


def _lbfgs_qf_target_block_step(self, g, H_block, block, evals=None, evecs=None):
    """Return an eigenvalue-clipped dense Newton step in the selected block."""
    g = np.asarray(g, dtype=float)
    H_block = np.asarray(H_block, dtype=float)
    block = np.asarray(block, dtype=int)

    if len(block) == 0 or H_block.shape != (len(block), len(block)):
        return None, np.array([], dtype=float)
    if not np.all(np.isfinite(H_block)):
        return None, np.array([], dtype=float)

    eig_floor = float(getattr(self, "_lbfgs_qf_target_block_eig_floor", 1.0e-3))
    if eig_floor <= 0.0:
        raise ValueError("lbfgs_qf_target_block_eig_floor must be positive.")

    if evals is None or evecs is None:
        evals, evecs = np.linalg.eigh(H_block)
    else:
        evals = np.asarray(evals, dtype=float)
        evecs = np.asarray(evecs, dtype=float)
    if not np.all(np.isfinite(evals)) or not np.all(np.isfinite(evecs)):
        return None, evals

    g_block = g[block]
    evals_eff = np.maximum(evals, eig_floor)
    step_block = -(evecs @ ((evecs.T @ g_block) / evals_eff))

    max_abs_step = _lbfgs_qf_inf_if_none(
        getattr(self, "_lbfgs_qf_target_block_max_abs_step", None)
    )
    max_step_norm = _lbfgs_qf_inf_if_none(
        getattr(self, "_lbfgs_qf_target_block_max_step_norm", 0.05)
    )

    if np.isfinite(max_abs_step):
        step_block = np.clip(step_block, -max_abs_step, max_abs_step)

    step_block_norm = np.linalg.norm(step_block)
    if np.isfinite(max_step_norm) and step_block_norm > max_step_norm and step_block_norm > 0.0:
        step_block *= max_step_norm / step_block_norm

    step = np.zeros_like(g)
    step[block] = step_block

    if not np.all(np.isfinite(step)) or np.linalg.norm(step) == 0.0:
        return None, evals

    require_descent = bool(getattr(self, "_lbfgs_qf_target_block_require_descent", True))
    if require_descent and np.dot(g, step) >= 0.0:
        return None, evals

    return step, evals


def _lbfgs_qf_is_stalled(self, gnorm_history):
    """Return whether recent gradient norms indicate a stalled optimization."""
    window = max(1, int(getattr(self, "_lbfgs_qf_stall_window", 5)))
    ratio = float(getattr(self, "_lbfgs_qf_stall_gnorm_ratio", 0.8))
    if len(gnorm_history) <= window:
        return False

    g_start = float(gnorm_history[-window - 1])
    g_now = float(gnorm_history[-1])
    return g_start > 0.0 and g_now / g_start > ratio


def _lbfgs_qf_should_escape_negative_curvature(
    self,
    k,
    gnorm,
    stalled,
    lam_min,
    negcurv_count,
    negcurv_escape_count,
):
    """Return whether to try a controlled step along negative curvature."""
    del k  # kept for future iteration-based escape policies
    if not bool(getattr(self, "_lbfgs_qf_escape_negative_curvature", False)):
        return False

    trigger = getattr(self, "_lbfgs_qf_escape_negcurv_trigger", "persistent")
    allowed = {
        "never",
        "always",
        "small_gradient",
        "stalled",
        "persistent",
        "stalled_or_persistent",
    }
    if trigger not in allowed:
        raise ValueError(
            "Unknown lbfgs_qf_escape_negcurv_trigger. Expected one of "
            f"{sorted(allowed)}; got {trigger!r}."
        )
    if trigger == "never":
        return False

    max_escapes = int(getattr(self, "_lbfgs_qf_escape_negcurv_max_escapes", 5))
    if negcurv_escape_count >= max_escapes:
        return False

    eig_thresh = float(getattr(self, "_lbfgs_qf_escape_negcurv_eig_thresh", -1.0e-5))
    if lam_min is None or not np.isfinite(lam_min) or lam_min >= eig_thresh:
        return False

    if trigger == "always":
        return True
    if trigger == "small_gradient":
        gnorm_thresh = float(
            getattr(self, "_lbfgs_qf_escape_negcurv_gnorm_thresh", 1.0e-3)
        )
        return gnorm < gnorm_thresh
    if trigger == "stalled":
        return bool(stalled)
    if trigger == "persistent":
        persist_count = int(getattr(self, "_lbfgs_qf_escape_negcurv_persist_count", 3))
        return negcurv_count >= persist_count

    persist_count = int(getattr(self, "_lbfgs_qf_escape_negcurv_persist_count", 3))
    return bool(stalled) or negcurv_count >= persist_count


def _lbfgs_qf_try_negative_curvature_escape(
    self,
    x,
    E,
    g,
    block,
    evals,
    evecs,
    objective=None,
):
    """Try ± the raw most-negative block eigenvector to escape a saddle point."""
    if objective is None:
        objective = self.energy_feval

    info = {
        "accepted": False,
        "reason": "not_started",
        "nfev": 0,
        "grad_evals": 0,
        "alpha": 0.0,
        "step_norm": 0.0,
        "sign": None,
    }

    evals = np.asarray(evals, dtype=float)
    evecs = np.asarray(evecs, dtype=float)
    block = np.asarray(block, dtype=int)

    if len(evals) == 0 or evecs.shape != (len(block), len(block)):
        info["reason"] = "bad_eigenpairs"
        return False, x, E, g, info

    imin = int(np.argmin(evals))
    lam_min = float(evals[imin])
    info["lam_min"] = lam_min

    eig_thresh = float(getattr(self, "_lbfgs_qf_escape_negcurv_eig_thresh", -1.0e-5))
    if lam_min >= eig_thresh:
        info["reason"] = "no_negative_curvature"
        return False, x, E, g, info

    q = np.zeros_like(x, dtype=float)
    q[block] = np.asarray(evecs[:, imin], dtype=float)
    qnorm = np.linalg.norm(q)
    if qnorm == 0.0 or not np.isfinite(qnorm):
        info["reason"] = "bad_escape_direction"
        return False, x, E, g, info
    q /= qnorm

    alpha0 = float(getattr(self, "_lbfgs_qf_escape_negcurv_step", 5.0e-2))
    max_ls = int(getattr(self, "_lbfgs_qf_escape_negcurv_max_ls", 12))
    shrink = float(getattr(self, "_lbfgs_qf_escape_negcurv_shrink", 0.5))
    if alpha0 <= 0.0:
        raise ValueError("lbfgs_qf_escape_negcurv_step must be positive.")
    if max_ls <= 0:
        raise ValueError("lbfgs_qf_escape_negcurv_max_ls must be positive.")
    if shrink <= 0.0 or shrink >= 1.0:
        raise ValueError("lbfgs_qf_escape_negcurv_shrink must be in (0, 1).")

    # This is intentionally not a Newton minimization step: it probes the raw
    # negative-curvature eigenvector in both signs and accepts only real energy
    # decrease, which is the conservative saddle/excited-state escape behavior.
    best_x = None
    best_E = E
    best_alpha = 0.0
    best_sign = None
    best_step_norm = 0.0

    for sign in (+1.0, -1.0):
        alpha = alpha0
        for _ in range(max_ls):
            x_trial = x + sign * alpha * q
            E_trial = float(np.real(objective(x_trial)))
            info["nfev"] += 1
            if np.isfinite(E_trial) and E_trial < best_E:
                best_x = x_trial
                best_E = E_trial
                best_alpha = alpha
                best_sign = sign
                best_step_norm = alpha
            alpha *= shrink

    if best_x is None:
        info["reason"] = "line_search_failed"
        return False, x, E, g, info

    g_new = np.asarray(self.gradient_ary_feval(best_x), dtype=float)
    info["grad_evals"] += 1
    info["accepted"] = True
    info["reason"] = "accepted"
    info["alpha"] = best_alpha
    info["sign"] = best_sign
    info["step_norm"] = best_step_norm
    info["dE"] = best_E - E
    return True, best_x, best_E, g_new, info


def _lbfgs_qf_try_target_block_step(
    self,
    x,
    E,
    g,
    h_eff=None,
    lbfgs_step=None,
    objective=None,
    k=0,
    stalled=False,
    negcurv_count=0,
    negcurv_escape_count=0,
):
    """Try an Armijo-globalized dense target-block Hessian correction."""
    if objective is None:
        objective = self.energy_feval

    self._lbfgs_qf_target_block_attempts = getattr(
        self, "_lbfgs_qf_target_block_attempts", 0
    ) + 1

    info = {
        "accepted": False,
        "reason": "not_started",
        "nfev": 0,
        "grad_evals": 0,
        "alpha": 0.0,
        "step_norm": 0.0,
        "fd_type": getattr(self, "_lbfgs_qf_target_block_fd_type", "forward"),
        "step_type": "BLOCK",
        "negcurv_count": int(negcurv_count),
        "negcurv_escape_count": int(negcurv_escape_count),
    }

    block = _lbfgs_qf_select_target_block(self, x, g, h_eff=h_eff, lbfgs_step=lbfgs_step)
    info["block"] = block.tolist()
    info["block_size"] = int(len(block))
    if len(block) == 0:
        info["reason"] = "empty_block"
        self._lbfgs_qf_target_block_last_info = info
        return False, x, E, g, 0.0, 0.0, info

    grad_evals_before = getattr(self, "_lbfgs_qf_target_block_grad_evals", 0)
    try:
        H_block = _lbfgs_qf_build_target_hessian_block(self, x, g, block)
    except (ValueError, np.linalg.LinAlgError) as exc:
        info["reason"] = f"hessian_build_failed: {exc}"
        info["grad_evals"] = max(
            0,
            getattr(self, "_lbfgs_qf_target_block_grad_evals", grad_evals_before)
            - grad_evals_before,
        )
        self._lbfgs_qf_target_block_last_info = info
        return False, x, E, g, 0.0, 0.0, info

    info["grad_evals"] = max(
        0,
        getattr(self, "_lbfgs_qf_target_block_grad_evals", grad_evals_before)
        - grad_evals_before,
    )

    if H_block.shape != (len(block), len(block)) or not np.all(np.isfinite(H_block)):
        info["reason"] = "nonfinite_hessian_block"
        self._lbfgs_qf_target_block_last_info = info
        return False, x, E, g, 0.0, 0.0, info

    try:
        evals, evecs = np.linalg.eigh(H_block)
    except (ValueError, np.linalg.LinAlgError) as exc:
        info["reason"] = f"block_diagonalization_failed: {exc}"
        self._lbfgs_qf_target_block_last_info = info
        return False, x, E, g, 0.0, 0.0, info

    eig_floor = float(getattr(self, "_lbfgs_qf_target_block_eig_floor", 1.0e-3))
    if not np.all(np.isfinite(evals)) or not np.all(np.isfinite(evecs)):
        info["reason"] = "nonfinite_block_eigenpairs"
        self._lbfgs_qf_target_block_last_info = info
        return False, x, E, g, 0.0, 0.0, info

    info["eig_min"] = float(np.min(evals))
    info["eig_max"] = float(np.max(evals))
    info["lam_min"] = info["eig_min"]
    info["n_negative_eigs"] = int(np.sum(evals < 0.0))
    info["n_near_zero_eigs"] = int(np.sum(np.abs(evals) < eig_floor))

    eig_thresh = float(getattr(self, "_lbfgs_qf_escape_negcurv_eig_thresh", -1.0e-5))
    if info["lam_min"] < eig_thresh:
        negcurv_count = int(negcurv_count) + 1
    else:
        negcurv_count = 0
    info["negcurv_count"] = negcurv_count

    if _lbfgs_qf_should_escape_negative_curvature(
        self,
        k,
        np.linalg.norm(g),
        stalled,
        info["lam_min"],
        negcurv_count,
        negcurv_escape_count,
    ):
        self._lbfgs_qf_escape_negcurv_attempts = getattr(
            self, "_lbfgs_qf_escape_negcurv_attempts", 0
        ) + 1
        (
            escape_accepted,
            x_escape,
            E_escape,
            g_escape,
            escape_info,
        ) = _lbfgs_qf_try_negative_curvature_escape(
            self,
            x,
            E,
            g,
            block,
            evals,
            evecs,
            objective=objective,
        )
        info["escape_attempted"] = True
        info["escape_reason"] = escape_info.get("reason", "unknown")
        info["nfev"] += escape_info.get("nfev", 0)
        info["grad_evals"] += escape_info.get("grad_evals", 0)
        if "alpha" in escape_info:
            info["escape_alpha"] = escape_info["alpha"]
        if "sign" in escape_info:
            info["escape_sign"] = escape_info["sign"]

        if escape_accepted:
            trigger_count = negcurv_count
            negcurv_escape_count = int(negcurv_escape_count) + 1
            info["accepted"] = True
            info["alpha"] = escape_info.get("alpha", 0.0)
            info["step_norm"] = escape_info.get("step_norm", 0.0)
            info["reason"] = "accepted"
            info["step_type"] = "ESCAPE"
            info["sign"] = escape_info.get("sign", None)
            info["dE"] = E_escape - E
            info["negcurv_trigger_count"] = trigger_count
            info["negcurv_count"] = 0
            info["negcurv_escape_count"] = negcurv_escape_count
            self._lbfgs_qf_escape_negcurv_accepted = getattr(
                self, "_lbfgs_qf_escape_negcurv_accepted", 0
            ) + 1
            self._lbfgs_qf_target_block_accepted = getattr(
                self, "_lbfgs_qf_target_block_accepted", 0
            ) + 1
            self._lbfgs_qf_escape_negcurv_last_info = info
            self._lbfgs_qf_target_block_last_info = info
            return (
                True,
                x_escape,
                E_escape,
                g_escape,
                info["alpha"],
                info["step_norm"],
                info,
            )

        print(
            "  => Negative-curvature escape rejected: "
            f'lam_min={info["lam_min"]:+.2e}, '
            f'reason={escape_info.get("reason", "unknown")}'
        )
        self._lbfgs_qf_escape_negcurv_last_info = info

    try:
        step, evals = _lbfgs_qf_target_block_step(
            self,
            g,
            H_block,
            block,
            evals=evals,
            evecs=evecs,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        info["reason"] = f"block_step_failed: {exc}"
        self._lbfgs_qf_target_block_last_info = info
        return False, x, E, g, 0.0, 0.0, info

    if step is None:
        info["reason"] = "non_descent_or_bad_block_step"
        self._lbfgs_qf_target_block_last_info = info
        return False, x, E, g, 0.0, 0.0, info

    gdotp = float(np.dot(g, step))
    if not np.isfinite(gdotp) or gdotp >= 0.0:
        info["reason"] = "non_descent_block_step"
        self._lbfgs_qf_target_block_last_info = info
        return False, x, E, g, 0.0, 0.0, info

    c1 = float(getattr(self, "_lbfgs_qf_target_block_armijo_c1", 1.0e-4))
    max_ls = int(getattr(self, "_lbfgs_qf_target_block_max_ls", 20))
    alpha = 1.0

    for _ in range(max_ls):
        trial_step = alpha * step
        step_norm = np.linalg.norm(trial_step)
        if step_norm == 0.0:
            break

        x_trial = x + trial_step
        E_trial = float(np.real(objective(x_trial)))
        g_trial = None
        info["nfev"] += 1

        if E_trial < E and E_trial <= E + c1 * alpha * gdotp:
            if g_trial is None:
                g_trial = np.asarray(self.gradient_ary_feval(x_trial), dtype=float)
                info["grad_evals"] += 1
            info["accepted"] = True
            info["alpha"] = alpha
            info["step_norm"] = step_norm
            info["reason"] = "accepted"
            self._lbfgs_qf_target_block_accepted = getattr(
                self, "_lbfgs_qf_target_block_accepted", 0
            ) + 1
            self._lbfgs_qf_target_block_last_info = info
            return True, x_trial, E_trial, g_trial, alpha, step_norm, info

        alpha *= 0.5

    info["reason"] = "line_search_failed"
    self._lbfgs_qf_target_block_last_info = info
    return False, x, E, g, 0.0, 0.0, info


def _lbfgs_qf_should_attempt_target_block(self, k, gnorm_history):
    """Return whether the dense target-block correction should be tried."""
    if not bool(getattr(self, "_lbfgs_qf_use_target_block", False)):
        return False

    trigger = getattr(self, "_lbfgs_qf_target_block_trigger", "stalled")
    allowed = {"never", "periodic", "stalled", "periodic_or_stalled"}
    if trigger not in allowed:
        raise ValueError(
            "Unknown lbfgs_qf_target_block_trigger. Expected one of "
            f"{sorted(allowed)}; got {trigger!r}."
        )
    if trigger == "never":
        return False

    start = int(getattr(self, "_lbfgs_qf_target_block_start", 10))
    if k < start:
        return False

    every = max(1, int(getattr(self, "_lbfgs_qf_target_block_every", 10)))
    periodic = (k - start) % every == 0

    stalled = _lbfgs_qf_is_stalled(self, gnorm_history)

    if trigger == "periodic":
        return periodic
    if trigger == "stalled":
        return stalled
    return periodic or stalled


def _lbfgs_qf_print_iteration(
    self,
    k,
    E,
    dE,
    nfev,
    grad_norm,
    step_type,
    alpha,
    step_norm,
    status,
    hist_len,
    block_info=None,
):
    """Print one compact lbfgs_qf iteration row."""
    evals = (
        f'{nfev}/'
        f'{getattr(self, "_res_vec_evals", 0)}/'
        f'{getattr(self, "_res_m_evals", 0)}'
    )
    status_short = {
        "accepted": "ok",
        "initial": "init",
        "failed": "fail",
    }.get(str(status), str(status))
    status_short = status_short[:6]

    step_label = str(step_type)[:6]
    row = (
        f' {k:5d} | {E:+16.10f} | {dE:+16.10f} | {grad_norm:+16.10f} | '
        f'{step_label:<6} | {status_short:<6} | {alpha:7.1e} | '
        f'{step_norm:8.2e} | {hist_len:4d} | {evals:>15}'
    )
    if bool(getattr(self, "_lbfgs_qf_print_aux", False)):
        aux = "-"
        hv_evals = getattr(self, "_lbfgs_qf_newton_cg_hv_evals", 0)
        if hv_evals:
            aux = f'Hv={hv_evals}'
        if block_info:
            block_size = block_info.get("block_size", "-")
            if block_info.get("step_type") == "ESCAPE":
                lam_min = block_info.get("lam_min", block_info.get("eig_min", np.nan))
                lam_text = f"{lam_min:+.1e}" if np.isfinite(lam_min) else "-"
                sign = block_info.get("sign", block_info.get("escape_sign", None))
                sign_text = "+" if sign is not None and sign > 0 else "-"
                count = block_info.get(
                    "negcurv_trigger_count",
                    block_info.get("negcurv_count", "-"),
                )
                aux = f"K={block_size} lam={lam_text} s={sign_text} c={count}"
            else:
                eig_min = "-"
                eig_max = "-"
                if "eig_min" in block_info and np.isfinite(block_info["eig_min"]):
                    eig_min = f'{block_info["eig_min"]:+.1e}'
                if "eig_max" in block_info and np.isfinite(block_info["eig_max"]):
                    eig_max = f'{block_info["eig_max"]:+.1e}'
                aux = f'K={block_size} eig=[{eig_min},{eig_max}]'
        row += f' | {aux}'
    print(row, flush=True)


def _lbfgs_qf_finalize(self, res):
    """Mirror the optimizer bookkeeping conventions used by Jacobi/SciPy solvers."""
    self._tamps = list(np.real(res.x))
    self._Egs = float(np.real(res.fun))
    self._curr_energy = self._Egs
    self._final_result = res

    if hasattr(self, '_energies'):
        self._energies.append(self._Egs)
    if hasattr(self, '_results'):
        self._results.append(res)
    if hasattr(self, '_n_classical_params'):
        self._n_classical_params = len(self._tamps)

    nfev = getattr(res, "nfev", 0)
    njev = getattr(res, "njev", 0)

    if hasattr(self, '_n_pauli_measures_k'):
        self._n_pauli_measures_k += self._Nl * nfev
        if hasattr(self, '_Nm') and hasattr(self, '_tops'):
            for m in self._tops:
                self._n_pauli_measures_k += self._Nm[m] * self._Nl * njev

    if hasattr(self, '_n_pauli_trm_measures'):
        n_active_params = 0
        for tmu in self._tamps:
            if np.abs(tmu) > 1.0e-12:
                n_active_params += 1
        self._n_pauli_trm_measures += int(2 * self._Nl * njev * n_active_params)
        self._n_pauli_trm_measures += int(self._Nl * nfev)

    if hasattr(self, '_n_pauli_trm_measures_lst'):
        if hasattr(self, '_n_pauli_measures_k'):
            self._n_pauli_trm_measures_lst.append(self._n_pauli_measures_k)
        else:
            self._n_pauli_trm_measures_lst.append(self._n_pauli_trm_measures)

    try:
        cnot_count = _cnot_count_for_current_ansatz(self)
    except Exception:
        cnot_count = 'N/A'

    if hasattr(self, '_n_cnot'):
        self._n_cnot = cnot_count
    if hasattr(self, '_n_cnot_lst'):
        self._n_cnot_lst.append(cnot_count)


def _qf_gradient_optimizer_checks(self, optimizer_name):
    """Shared validation for in-house gradient-based optimizer drivers."""
    supported_classes = ['UCCNVQE', 'ADAPTVQE']
    if self.__class__.__name__ not in supported_classes:
        raise ValueError(
            f'optimizer="{optimizer_name}" currently supports gradient-based VQE '
            'classes UCCNVQE and ADAPTVQE only. UCCNPQE/SPQE are residual-based '
            'and do not yet provide the required energy-gradient objective.'
        )
    if getattr(self, "_computer_type", None) == "fci_gpu":
        raise NotImplementedError(
            f'optimizer="{optimizer_name}" is not yet implemented for '
            'computer_type="fci_gpu". The in-house qf optimizer path currently '
            "depends on CPU FCIComputer derivative/diagnostic helpers; matching "
            "FCIComputerGPU support will be added in a future PR."
        )
    if hasattr(self, '_use_analytic_grad') and not self._use_analytic_grad:
        raise ValueError(f'optimizer="{optimizer_name}" requires use_analytic_grad=True.')
    if not hasattr(self, 'gradient_ary_feval'):
        raise ValueError(f'optimizer="{optimizer_name}" requires gradient_ary_feval(params).')


def _bfgs_qf_option(self, suffix, default=None, lbfgs_suffix=None):
    """Return bfgs_qf option, falling back to explicitly supplied lbfgs_qf aliases."""
    supplied = getattr(self, "_qf_optimizer_supplied_options", set())
    bfgs_key = f"bfgs_qf_{suffix}"
    lbfgs_key = f"lbfgs_qf_{lbfgs_suffix if lbfgs_suffix is not None else suffix}"

    if bfgs_key in supplied:
        return getattr(self, f"_{bfgs_key}")
    if lbfgs_key in supplied and hasattr(self, f"_{lbfgs_key}"):
        return getattr(self, f"_{lbfgs_key}")
    return getattr(self, f"_{bfgs_key}", default)


def _bfgs_qf_activate_shared_lbfgs_options(self):
    """Mirror bfgs_qf options into the lbfgs_qf helper namespace.

    The NCG/target-block/negative-curvature helpers predate bfgs_qf and read
    _lbfgs_qf_* attributes. Mirroring keeps those helpers shared without changing
    lbfgs_qf behavior. The caller restores the old values before returning.
    """
    mappings = {
        "use_hessian_diag": "use_hessian_diag",
        "hdiag_start": "hdiag_start",
        "hdiag_stop": "hdiag_stop",
        "hdiag_update_freq": "hdiag_update_freq",
        "hdiag_floor": "hdiag_floor",
        "hdiag_mode": "hdiag_mode",
        "hdiag_method": "hdiag_method",
        "hdiag_fd_step": "hdiag_fd_step",
        "use_newton_cg": "use_newton_cg",
        "newton_cg_trigger": "newton_cg_trigger",
        "newton_cg_start": "newton_cg_start",
        "newton_cg_every": "newton_cg_every",
        "newton_cg_maxiter": "newton_cg_maxiter",
        "newton_cg_tol": "newton_cg_tol",
        "newton_cg_fd_delta": "newton_cg_fd_delta",
        "newton_cg_level_shift": "newton_cg_level_shift",
        "newton_cg_max_step_norm": "newton_cg_max_step_norm",
        "newton_cg_max_abs_step": "newton_cg_max_abs_step",
        "newton_cg_armijo_c1": "newton_cg_armijo_c1",
        "newton_cg_max_ls": "newton_cg_max_ls",
        "newton_cg_min_gnorm": "newton_cg_min_gnorm",
        "use_target_block": "use_target_block",
        "target_block_trigger": "target_block_trigger",
        "target_block_start": "target_block_start",
        "target_block_every": "target_block_every",
        "target_block_size": "target_block_size",
        "target_block_min_size": "target_block_min_size",
        "target_block_max_size": "target_block_max_size",
        "target_block_size_frac": "target_block_size_frac",
        "target_block_selection": "target_block_selection",
        "target_block_fd_type": "target_block_fd_type",
        "target_block_fd_delta": "target_block_fd_delta",
        "target_block_eig_floor": "target_block_eig_floor",
        "target_block_max_step_norm": "target_block_max_step_norm",
        "target_block_max_abs_step": "target_block_max_abs_step",
        "target_block_armijo_c1": "target_block_armijo_c1",
        "target_block_max_ls": "target_block_max_ls",
        "target_block_require_descent": "target_block_require_descent",
        "escape_negative_curvature": "escape_negative_curvature",
        "escape_negcurv_trigger": "escape_negcurv_trigger",
        "escape_negcurv_eig_thresh": "escape_negcurv_eig_thresh",
        "escape_negcurv_persist_count": "escape_negcurv_persist_count",
        "escape_negcurv_gnorm_thresh": "escape_negcurv_gnorm_thresh",
        "escape_negcurv_step": "escape_negcurv_step",
        "escape_negcurv_max_ls": "escape_negcurv_max_ls",
        "escape_negcurv_shrink": "escape_negcurv_shrink",
        "escape_negcurv_max_escapes": "escape_negcurv_max_escapes",
        "escape_negcurv_require_energy_decrease": "escape_negcurv_require_energy_decrease",
        "stall_window": "stall_window",
        "stall_gnorm_ratio": "stall_gnorm_ratio",
        "print_aux": "print_aux",
        "use_gradient_energy": "use_gradient_energy",
    }
    backup = {}
    missing = object()
    for bfgs_suffix, lbfgs_suffix in mappings.items():
        lbfgs_attr = f"_lbfgs_qf_{lbfgs_suffix}"
        backup[lbfgs_attr] = getattr(self, lbfgs_attr, missing)
        setattr(
            self,
            lbfgs_attr,
            _bfgs_qf_option(self, bfgs_suffix, lbfgs_suffix=lbfgs_suffix),
        )

    reset_mappings = {
        "newton_cg_reset_bfgs_hessian": "newton_cg_reset_lbfgs_history",
        "target_block_reset_bfgs_hessian": "target_block_reset_lbfgs_history",
        "escape_negcurv_reset_bfgs_hessian": "escape_negcurv_reset_lbfgs_history",
    }
    for bfgs_suffix, lbfgs_suffix in reset_mappings.items():
        lbfgs_attr = f"_lbfgs_qf_{lbfgs_suffix}"
        backup[lbfgs_attr] = getattr(self, lbfgs_attr, missing)
        setattr(self, lbfgs_attr, _bfgs_qf_option(self, bfgs_suffix))

    return backup, missing


def _bfgs_qf_restore_shared_lbfgs_options(self, backup, missing):
    """Restore _lbfgs_qf_* attributes after bfgs_qf used shared helpers."""
    for attr, value in backup.items():
        if value is missing:
            try:
                delattr(self, attr)
            except AttributeError:
                pass
        else:
            setattr(self, attr, value)


def _bfgs_qf_initial_hinv(self, n_params, h_eff=None):
    """Return the dense inverse-Hessian approximation used to start/restart BFGS."""
    n_params = int(n_params)
    if h_eff is not None:
        h_eff = np.asarray(h_eff, dtype=float)
        if h_eff.shape == (n_params,) and np.all(np.isfinite(h_eff)) and np.all(h_eff > 0.0):
            return np.diag(1.0 / h_eff)

    gamma = float(_bfgs_qf_option(self, "init_scale", 1.0))
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("bfgs_qf_init_scale must be a positive finite number.")
    return gamma * np.eye(n_params)


def _bfgs_qf_reset_hinv(self, n_params, h_eff=None):
    """Reset and count the dense inverse-Hessian approximation."""
    self._bfgs_qf_hinv_resets = getattr(self, "_bfgs_qf_hinv_resets", 0) + 1
    return _bfgs_qf_initial_hinv(self, n_params, h_eff=h_eff)


def _bfgs_qf_update_hinv(self, Hinv, s, y, curvature_tol):
    """Apply the full inverse-Hessian BFGS update when curvature is safe."""
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    ys = float(np.dot(y, s))
    info = {"ys": ys, "accepted": False, "reason": "bad_curvature"}

    if not np.isfinite(ys) or ys <= curvature_tol:
        self._bfgs_qf_updates_skipped = getattr(self, "_bfgs_qf_updates_skipped", 0) + 1
        return Hinv, info

    rho = 1.0 / ys
    I = np.eye(len(s))
    V = I - rho * np.outer(s, y)
    Hnew = V @ Hinv @ V.T + rho * np.outer(s, s)
    Hnew = 0.5 * (Hnew + Hnew.T)

    if not np.all(np.isfinite(Hnew)):
        self._bfgs_qf_updates_skipped = getattr(self, "_bfgs_qf_updates_skipped", 0) + 1
        info["reason"] = "nonfinite_hinv_update"
        return Hinv, info

    self._bfgs_qf_updates_accepted = getattr(self, "_bfgs_qf_updates_accepted", 0) + 1
    info["accepted"] = True
    info["reason"] = "accepted"
    return Hnew, info


def lbfgs_qf_solve(self, function_to_minimize=None):
    """In-house L-BFGS optimizer for gradient-based tUCC/VQE calculations.

    User-facing controls are stored on the algorithm instance:
    _lbfgs_qf_memory, _lbfgs_qf_max_ls, _lbfgs_qf_c1, _lbfgs_qf_alpha0,
    _lbfgs_qf_max_step_norm, _lbfgs_qf_max_abs_step,
    _lbfgs_qf_curvature_tol, _lbfgs_qf_use_hessian_diag,
    _lbfgs_qf_hdiag_start, _lbfgs_qf_hdiag_stop,
    _lbfgs_qf_hdiag_update_freq, _lbfgs_qf_hdiag_floor,
    _lbfgs_qf_hdiag_mode, optional Newton-CG or target-block acceleration controls, and
    related run-level lbfgs_qf_* options.
    """
    _qf_gradient_optimizer_checks(self, "lbfgs_qf")

    objective = self.energy_feval if function_to_minimize is None else function_to_minimize

    memory = int(getattr(self, "_lbfgs_qf_memory", 10))
    max_ls = int(getattr(self, "_lbfgs_qf_max_ls", 20))
    c1 = float(getattr(self, "_lbfgs_qf_c1", 1.0e-4))
    alpha0 = float(getattr(self, "_lbfgs_qf_alpha0", 1.0))
    max_step_norm = _lbfgs_qf_inf_if_none(getattr(self, "_lbfgs_qf_max_step_norm", np.inf))
    max_abs_step = _lbfgs_qf_inf_if_none(getattr(self, "_lbfgs_qf_max_abs_step", np.inf))
    curvature_tol = float(getattr(self, "_lbfgs_qf_curvature_tol", 1.0e-12))
    step_tol = float(getattr(self, "_lbfgs_qf_step_tol", 1.0e-12))
    use_hdiag = bool(getattr(self, "_lbfgs_qf_use_hessian_diag", False))
    hdiag_start = int(getattr(self, "_lbfgs_qf_hdiag_start", 1))
    hdiag_stop = getattr(self, "_lbfgs_qf_hdiag_stop", None)
    hdiag_update_freq = max(1, int(getattr(self, "_lbfgs_qf_hdiag_update_freq", 1)))
    hdiag_floor = float(getattr(self, "_lbfgs_qf_hdiag_floor", 1.0e-3))
    hdiag_mode = getattr(self, "_lbfgs_qf_hdiag_mode", "positive")
    use_newton_cg = bool(getattr(self, "_lbfgs_qf_use_newton_cg", False))
    use_target_block = bool(getattr(self, "_lbfgs_qf_use_target_block", False))
    use_escape_negcurv = bool(getattr(self, "_lbfgs_qf_escape_negative_curvature", False))
    use_gradient_energy = bool(getattr(self, "_lbfgs_qf_use_gradient_energy", True))
    if use_escape_negcurv and use_newton_cg:
        raise ValueError(
            "lbfgs_qf_escape_negative_curvature uses the target-block protocol "
            "and cannot be combined with lbfgs_qf_use_newton_cg=True."
        )
    if use_escape_negcurv and not use_target_block:
        raise ValueError(
            "lbfgs_qf_escape_negative_curvature=True requires "
            "lbfgs_qf_use_target_block=True because the escape direction uses "
            "the target-block Hessian eigenvectors."
        )
    if use_newton_cg and use_target_block:
        raise ValueError(
            "Only one lbfgs_qf acceleration protocol may be active at a time: "
            "set either lbfgs_qf_use_newton_cg=True or "
            "lbfgs_qf_use_target_block=True, not both."
        )
    reset_ncg_history = bool(getattr(self, "_lbfgs_qf_newton_cg_reset_lbfgs_history", True))
    reset_block_history = bool(getattr(self, "_lbfgs_qf_target_block_reset_lbfgs_history", True))
    reset_escape_history = bool(
        getattr(self, "_lbfgs_qf_escape_negcurv_reset_lbfgs_history", True)
    )
    trigger = getattr(self, "_lbfgs_qf_newton_cg_trigger", "stalled")
    allowed_triggers = {"never", "periodic", "stalled", "periodic_or_stalled"}
    if trigger not in allowed_triggers:
        raise ValueError(
            "Unknown lbfgs_qf_newton_cg_trigger. Expected one of "
            f"{sorted(allowed_triggers)}; got {trigger!r}."
        )
    target_trigger = getattr(self, "_lbfgs_qf_target_block_trigger", "stalled")
    if target_trigger not in allowed_triggers:
        raise ValueError(
            "Unknown lbfgs_qf_target_block_trigger. Expected one of "
            f"{sorted(allowed_triggers)}; got {target_trigger!r}."
        )
    escape_trigger = getattr(self, "_lbfgs_qf_escape_negcurv_trigger", "persistent")
    allowed_escape_triggers = {
        "never",
        "always",
        "small_gradient",
        "stalled",
        "persistent",
        "stalled_or_persistent",
    }
    if escape_trigger not in allowed_escape_triggers:
        raise ValueError(
            "Unknown lbfgs_qf_escape_negcurv_trigger. Expected one of "
            f"{sorted(allowed_escape_triggers)}; got {escape_trigger!r}."
        )

    x = np.array(copy.deepcopy(self._tamps), dtype=float)
    s_hist = []
    y_hist = []
    rho_hist = []
    cached_h_eff = None
    cached_hdiag_iter = None

    nfev = 0
    njev = 0
    self._k_counter = 0
    self._lbfgs_qf_newton_cg_attempts = 0
    self._lbfgs_qf_newton_cg_accepted = 0
    self._lbfgs_qf_newton_cg_hv_evals = 0
    self._lbfgs_qf_newton_cg_last_info = {}
    self._lbfgs_qf_target_block_attempts = 0
    self._lbfgs_qf_target_block_accepted = 0
    self._lbfgs_qf_target_block_grad_evals = 0
    self._lbfgs_qf_target_block_last_info = {}
    self._lbfgs_qf_escape_negcurv_attempts = 0
    self._lbfgs_qf_escape_negcurv_accepted = 0
    self._lbfgs_qf_escape_negcurv_last_info = {}
    negcurv_count = 0
    negcurv_escape_count = 0
    last_block_lam_min = None

    initial_hdiag_due = _lbfgs_qf_hdiag_is_due(
        1,
        use_hdiag,
        hdiag_start,
        hdiag_stop,
        hdiag_update_freq,
        cached_h_eff,
        hdiag_mode,
    )
    if initial_hdiag_due:
        E, g, h_diag, nfev_inc, njev_inc, combined_hdiag = _qf_gradient_hdiag_feval(
            self,
            x,
            objective=objective,
            return_energy=True,
            need_hdiag=True,
            use_gradient_energy=use_gradient_energy,
        )
        if h_diag is not None:
            if len(h_diag) != len(x):
                raise ValueError(
                    'hessian_diag_ary_feval(params) returned a vector of length '
                    f'{len(h_diag)}, but there are {len(x)} amplitudes.'
                )
            cached_h_eff = _lbfgs_qf_regularized_hdiag(h_diag, hdiag_floor, hdiag_mode)
            cached_hdiag_iter = 1
        elif not combined_hdiag:
            h_diag, hdiag_njev = _qf_hessian_diag_feval(self, x, len(x), "lbfgs_qf")
            njev_inc += hdiag_njev
            cached_h_eff = _lbfgs_qf_regularized_hdiag(h_diag, hdiag_floor, hdiag_mode)
            cached_hdiag_iter = 1
    else:
        E, g, nfev_inc, njev_inc = _qf_gradient_energy_feval(
            self,
            x,
            objective=objective,
            use_gradient_energy=use_gradient_energy,
        )
    nfev += nfev_inc
    njev += njev_inc
    self._prev_energy = E
    grad_norm = np.linalg.norm(g)
    self._curr_grad_norm = grad_norm
    gnorm_history = [grad_norm]

    print('  \n--> Begin qforte L-BFGS optimization:')
    print(f" Initial guess energy:              {E:+12.10f}")
    if use_newton_cg:
        print(
            ' Newton-CG acceleration:          enabled '
            f'(trigger={trigger}, start={getattr(self, "_lbfgs_qf_newton_cg_start", 10)})'
        )
    if use_target_block:
        print(
            ' Target-block acceleration:      enabled '
            f'(trigger={target_trigger}, start={getattr(self, "_lbfgs_qf_target_block_start", 10)})'
        )
    if use_escape_negcurv:
        print(
            ' Negative-curvature escape:      enabled '
            f'(trigger={escape_trigger}, '
            f'persist={getattr(self, "_lbfgs_qf_escape_negcurv_persist_count", 3)})'
        )
    header = (
        f' {"iter":>5} | {"Energy":^16} | {"dE":^16} | {"||g||":^16} | '
        f'{"step":^6} | {"stat":^6} | {"alpha":^7} | {"||step||":^8} | '
        f'{"Hist":^4} | {"evals(f/g/m)":^15}'
    )
    if bool(getattr(self, "_lbfgs_qf_print_aux", False)):
        header += ' | aux'
    print('\n' + header)
    print('-' * len(header))

    success = grad_norm < self._opt_thresh
    message = 'Initial gradient norm below threshold.' if success else 'Maximum iterations reached.'
    last_status = 'initial'
    last_step_norm = 0.0
    last_alpha = 0.0

    if not success:
        for k in range(1, self._opt_maxiter + 1):
            self._k_counter = k

            if _lbfgs_qf_hdiag_is_due(
                k,
                use_hdiag,
                hdiag_start,
                hdiag_stop,
                hdiag_update_freq,
                cached_h_eff,
                hdiag_mode,
            ):
                if cached_hdiag_iter != k:
                    h_diag, hdiag_njev = _qf_hessian_diag_feval(
                        self,
                        x,
                        len(x),
                        "lbfgs_qf",
                    )
                    njev += hdiag_njev
                    cached_h_eff = _lbfgs_qf_regularized_hdiag(h_diag, hdiag_floor, hdiag_mode)
                    cached_hdiag_iter = k
                    self._curr_grad_norm = grad_norm

            if _lbfgs_qf_should_attempt_target_block(self, k, gnorm_history):
                stalled = _lbfgs_qf_is_stalled(self, gnorm_history)
                lbfgs_step_for_selection = None
                if getattr(self, "_lbfgs_qf_target_block_selection", "preconditioned_gradient") == "lbfgs_step":
                    candidate_step = _lbfgs_qf_direction_from_history(
                        g,
                        s_hist,
                        y_hist,
                        rho_hist,
                        h_eff=cached_h_eff,
                    )
                    if np.all(np.isfinite(candidate_step)):
                        lbfgs_step_for_selection = candidate_step

                (
                    block_accepted,
                    x_block,
                    E_block,
                    g_block,
                    block_alpha,
                    block_step_norm,
                    block_info,
                ) = _lbfgs_qf_try_target_block_step(
                    self,
                    x,
                    E,
                    g,
                    h_eff=cached_h_eff,
                    lbfgs_step=lbfgs_step_for_selection,
                    objective=objective,
                    k=k,
                    stalled=stalled,
                    negcurv_count=negcurv_count,
                    negcurv_escape_count=negcurv_escape_count,
                )
                nfev += block_info.get("nfev", 0)
                njev += block_info.get("grad_evals", 0)
                negcurv_count = block_info.get("negcurv_count", negcurv_count)
                negcurv_escape_count = block_info.get(
                    "negcurv_escape_count",
                    negcurv_escape_count,
                )
                last_block_lam_min = block_info.get(
                    "lam_min",
                    block_info.get("eig_min", last_block_lam_min),
                )

                if block_accepted:
                    target_step_type = block_info.get("step_type", "BLOCK")
                    s = x_block - x
                    y = g_block - g
                    ys = np.dot(y, s)
                    reset_target_history = (
                        reset_escape_history
                        if target_step_type == "ESCAPE"
                        else reset_block_history
                    )

                    if reset_target_history:
                        s_hist = []
                        y_hist = []
                        rho_hist = []
                    elif ys > curvature_tol:
                        s_hist.append(s)
                        y_hist.append(y)
                        rho_hist.append(1.0 / ys)
                        if len(s_hist) > memory:
                            del s_hist[0]
                            del y_hist[0]
                            del rho_hist[0]

                    dE = E_block - E
                    x = x_block
                    E = E_block
                    g = g_block
                    grad_norm = np.linalg.norm(g)
                    self._curr_energy = E
                    self._curr_grad_norm = grad_norm
                    self._prev_energy = E
                    gnorm_history.append(grad_norm)
                    last_status = "accepted"
                    last_step_norm = block_step_norm
                    last_alpha = block_alpha

                    _lbfgs_qf_print_iteration(
                        self,
                        k,
                        E,
                        dE,
                        nfev,
                        grad_norm,
                        target_step_type,
                        last_alpha,
                        last_step_norm,
                        last_status,
                        len(s_hist),
                        block_info=block_info,
                    )

                    if grad_norm < self._opt_thresh:
                        success = True
                        message = 'Gradient norm below threshold.'
                        break

                    if last_step_norm < step_tol:
                        message = 'Step norm below lbfgs_qf step tolerance.'
                        break

                    continue

                print(
                    '  => Target block rejected: '
                    f'{block_info.get("reason", "unknown")}'
                )

            if _lbfgs_qf_should_attempt_newton_cg(self, k, gnorm_history):
                (
                    ncg_accepted,
                    x_ncg,
                    E_ncg,
                    g_ncg,
                    ncg_alpha,
                    ncg_step_norm,
                    ncg_info,
                ) = _lbfgs_qf_try_newton_cg_step(
                    self,
                    x,
                    E,
                    g,
                    h_eff=cached_h_eff,
                    objective=objective,
                )
                nfev += ncg_info.get("nfev", 0)
                njev += ncg_info.get("grad_evals", 0)

                if ncg_accepted:
                    s = x_ncg - x
                    y = g_ncg - g
                    ys = np.dot(y, s)

                    if reset_ncg_history:
                        s_hist = []
                        y_hist = []
                        rho_hist = []
                    elif ys > curvature_tol:
                        s_hist.append(s)
                        y_hist.append(y)
                        rho_hist.append(1.0 / ys)
                        if len(s_hist) > memory:
                            del s_hist[0]
                            del y_hist[0]
                            del rho_hist[0]

                    dE = E_ncg - E
                    x = x_ncg
                    E = E_ncg
                    g = g_ncg
                    grad_norm = np.linalg.norm(g)
                    self._curr_energy = E
                    self._curr_grad_norm = grad_norm
                    self._prev_energy = E
                    gnorm_history.append(grad_norm)
                    last_status = "accepted"
                    last_step_norm = ncg_step_norm
                    last_alpha = ncg_alpha

                    _lbfgs_qf_print_iteration(
                        self,
                        k,
                        E,
                        dE,
                        nfev,
                        grad_norm,
                        "NCG",
                        last_alpha,
                        last_step_norm,
                        last_status,
                        len(s_hist),
                    )

                    if grad_norm < self._opt_thresh:
                        success = True
                        message = 'Gradient norm below threshold.'
                        break

                    if last_step_norm < step_tol:
                        message = 'Step norm below lbfgs_qf step tolerance.'
                        break

                    continue

                print(
                    '  => Newton-CG trial rejected: '
                    f'{ncg_info.get("reason", "unknown")}'
                )

            # Standard two-loop L-BFGS recursion. The optional Hessian diagonal
            # preconditioner replaces the scalar gamma initialization for H0.
            if len(s_hist) == 0:
                if cached_h_eff is not None:
                    p = -g / cached_h_eff
                else:
                    p = -g
            else:
                q = g.copy()
                alpha_hist = []
                for s_i, y_i, rho_i in reversed(list(zip(s_hist, y_hist, rho_hist))):
                    alpha_i = rho_i * np.dot(s_i, q)
                    alpha_hist.append(alpha_i)
                    q -= alpha_i * y_i

                if cached_h_eff is not None:
                    r = q / cached_h_eff
                else:
                    s_last = s_hist[-1]
                    y_last = y_hist[-1]
                    yy_last = np.dot(y_last, y_last)
                    gamma = np.dot(s_last, y_last) / yy_last if yy_last > 0.0 else 1.0
                    r = gamma * q

                for i, (s_i, y_i, rho_i) in enumerate(zip(s_hist, y_hist, rho_hist)):
                    beta_i = rho_i * np.dot(y_i, r)
                    alpha_i = alpha_hist[len(s_hist) - 1 - i]
                    r += s_i * (alpha_i - beta_i)
                p = -r

            gdotp = np.dot(g, p)
            if not np.all(np.isfinite(p)) or gdotp >= 0.0:
                s_hist = []
                y_hist = []
                rho_hist = []
                if cached_h_eff is not None:
                    p = -g / cached_h_eff
                else:
                    p = -g
                gdotp = np.dot(g, p)
                last_status = 'reset'
            else:
                last_status = 'trial'

            accepted = False
            alpha = alpha0
            step = np.zeros_like(x)
            E_trial = E
            g_trial = None

            for _ in range(max_ls):
                step = _lbfgs_qf_capped_step(alpha * p, max_abs_step, max_step_norm)
                step_norm = np.linalg.norm(step)
                if step_norm == 0.0:
                    break
                x_trial = x + step
                E_trial = float(np.real(objective(x_trial)))
                g_trial = None
                nfev += 1
                if E_trial <= E + c1 * np.dot(g, step):
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                s_hist = []
                y_hist = []
                rho_hist = []
                if cached_h_eff is not None:
                    p = -g / cached_h_eff
                else:
                    p = -g
                alpha = max(alpha, alpha0 * (0.5 ** max_ls))
                for _ in range(max_ls):
                    step = _lbfgs_qf_capped_step(alpha * p, max_abs_step, max_step_norm)
                    step_norm = np.linalg.norm(step)
                    if step_norm == 0.0:
                        break
                    x_trial = x + step
                    E_trial = float(np.real(objective(x_trial)))
                    g_trial = None
                    nfev += 1
                    if E_trial <= E + c1 * np.dot(g, step):
                        accepted = True
                        last_status = 'sd'
                        break
                    alpha *= 0.5

            if not accepted:
                message = 'Line search failed to find an acceptable descent step.'
                last_status = 'failed'
                _lbfgs_qf_print_iteration(
                    self,
                    k,
                    E,
                    0.0,
                    nfev,
                    grad_norm,
                    "LBFGS",
                    0.0,
                    0.0,
                    last_status,
                    len(s_hist),
                )
                break

            if g_trial is None:
                next_k = k + 1
                need_next_hdiag = (
                    next_k <= self._opt_maxiter
                    and _lbfgs_qf_hdiag_is_due(
                        next_k,
                        use_hdiag,
                        hdiag_start,
                        hdiag_stop,
                        hdiag_update_freq,
                        cached_h_eff,
                        hdiag_mode,
                    )
                )
                _, g_trial, h_diag_next, _, njev_inc, _ = _qf_gradient_hdiag_feval(
                    self,
                    x_trial,
                    objective=objective,
                    return_energy=False,
                    need_hdiag=need_next_hdiag,
                    use_gradient_energy=use_gradient_energy,
                )
                njev += njev_inc
                if h_diag_next is not None:
                    if len(h_diag_next) != len(x_trial):
                        raise ValueError(
                            'hessian_diag_ary_feval(params) returned a vector of length '
                            f'{len(h_diag_next)}, but there are {len(x_trial)} amplitudes.'
                        )
                    cached_h_eff = _lbfgs_qf_regularized_hdiag(
                        h_diag_next,
                        hdiag_floor,
                        hdiag_mode,
                    )
                    cached_hdiag_iter = next_k
            s = x_trial - x
            y = g_trial - g
            ys = np.dot(y, s)

            if ys > curvature_tol:
                s_hist.append(s)
                y_hist.append(y)
                rho_hist.append(1.0 / ys)
                if len(s_hist) > memory:
                    del s_hist[0]
                    del y_hist[0]
                    del rho_hist[0]
                if last_status == 'trial':
                    last_status = 'accepted'
            else:
                print(f'  => Skipping L-BFGS update: y.s = {ys:+.6e}')
                if last_status == 'trial':
                    last_status = 'skip_y'

            dE = E_trial - E
            x = x_trial
            E = E_trial
            g = g_trial
            grad_norm = np.linalg.norm(g)
            self._curr_energy = E
            self._curr_grad_norm = grad_norm
            last_step_norm = np.linalg.norm(step)
            last_alpha = alpha
            self._prev_energy = E
            gnorm_history.append(grad_norm)

            _lbfgs_qf_print_iteration(
                self,
                k,
                E,
                dE,
                nfev,
                grad_norm,
                "LBFGS" if last_status != "sd" else "SD",
                last_alpha,
                last_step_norm,
                last_status,
                len(s_hist),
            )

            if grad_norm < self._opt_thresh:
                success = True
                message = 'Gradient norm below threshold.'
                break

            if last_step_norm < step_tol:
                message = 'Step norm below lbfgs_qf step tolerance.'
                break

    if success:
        print('  => Minimization successful!')
    else:
        print('  => WARNING: minimization result may not be tightly converged.')
    print(f'  => Minimum Energy: {E:+12.10f}')

    res = OptimizeResult(
        x=np.array(x, dtype=float),
        fun=E,
        jac=np.array(g, dtype=float),
        success=success,
        message=message,
        nit=self._k_counter,
        nfev=nfev,
        njev=njev,
        grad_norm=grad_norm,
        step_norm=last_step_norm,
        alpha=last_alpha,
        status=last_status,
        nhev=getattr(self, "_lbfgs_qf_newton_cg_hv_evals", 0),
        newton_cg_attempts=getattr(self, "_lbfgs_qf_newton_cg_attempts", 0),
        newton_cg_accepted=getattr(self, "_lbfgs_qf_newton_cg_accepted", 0),
        newton_cg_last_info=getattr(self, "_lbfgs_qf_newton_cg_last_info", {}),
        target_block_attempts=getattr(self, "_lbfgs_qf_target_block_attempts", 0),
        target_block_accepted=getattr(self, "_lbfgs_qf_target_block_accepted", 0),
        target_block_grad_evals=getattr(self, "_lbfgs_qf_target_block_grad_evals", 0),
        target_block_last_info=getattr(self, "_lbfgs_qf_target_block_last_info", {}),
        escape_negcurv_attempts=getattr(self, "_lbfgs_qf_escape_negcurv_attempts", 0),
        escape_negcurv_accepted=getattr(self, "_lbfgs_qf_escape_negcurv_accepted", 0),
        escape_negcurv_last_info=getattr(self, "_lbfgs_qf_escape_negcurv_last_info", {}),
        escape_negcurv_count=negcurv_count,
        escape_negcurv_last_lam_min=last_block_lam_min,
    )

    _lbfgs_qf_finalize(self, res)
    return res


def bfgs_qf_solve(self, function_to_minimize=None):
    """In-house dense full-memory inverse-Hessian BFGS optimizer.

    This driver deliberately reuses the lbfgs_qf accelerator helpers. Public
    options are bfgs_qf_*; explicitly supplied lbfgs_qf_* options are accepted
    as aliases for shared accelerator/preconditioner controls.
    """
    _qf_gradient_optimizer_checks(self, "bfgs_qf")

    backup, missing = _bfgs_qf_activate_shared_lbfgs_options(self)
    try:
        objective = self.energy_feval if function_to_minimize is None else function_to_minimize

        maxiter_option = _bfgs_qf_option(self, "maxiter", None)
        maxiter = self._opt_maxiter if maxiter_option is None else int(maxiter_option)
        gconv_option = _bfgs_qf_option(self, "gconv", None)
        gconv = self._opt_thresh if gconv_option is None else float(gconv_option)
        econv_option = _bfgs_qf_option(self, "econv", None)
        econv = self._opt_ftol if econv_option is None else float(econv_option)
        line_search = _bfgs_qf_option(self, "line_search", "armijo")
        if line_search != "armijo":
            raise ValueError('bfgs_qf_line_search currently supports only "armijo".')
        max_ls = int(_bfgs_qf_option(self, "max_ls", 20))
        c1 = float(_bfgs_qf_option(self, "c1", 1.0e-4))
        armijo_c1 = _bfgs_qf_option(self, "armijo_c1", None)
        if armijo_c1 is not None:
            c1 = float(armijo_c1)
        alpha0 = float(_bfgs_qf_option(self, "alpha0", 1.0))
        max_step_norm = _lbfgs_qf_inf_if_none(_bfgs_qf_option(self, "max_step_norm", 0.5))
        max_abs_step = _lbfgs_qf_inf_if_none(_bfgs_qf_option(self, "max_abs_step", None))
        curvature_tol = float(_bfgs_qf_option(self, "curvature_tol", 1.0e-12))
        step_tol = float(_bfgs_qf_option(self, "step_tol", 1.0e-12))
        max_params_dense = int(_bfgs_qf_option(self, "max_params_dense", 5000))
        reset_on_bad_curv = bool(_bfgs_qf_option(self, "reset_on_bad_curvature", False))
        reset_on_nondescent = bool(_bfgs_qf_option(self, "reset_on_nondescent", True))
        use_hdiag = bool(_bfgs_qf_option(self, "use_hessian_diag", False))
        use_gradient_energy = bool(_bfgs_qf_option(self, "use_gradient_energy", True))
        hdiag_start = int(_bfgs_qf_option(self, "hdiag_start", 1))
        hdiag_stop = _bfgs_qf_option(self, "hdiag_stop", 1)
        hdiag_update_freq = max(1, int(_bfgs_qf_option(self, "hdiag_update_freq", 1)))
        hdiag_floor = float(_bfgs_qf_option(self, "hdiag_floor", 1.0e-3))
        hdiag_mode = _bfgs_qf_option(self, "hdiag_mode", "positive")
        use_newton_cg = bool(_bfgs_qf_option(self, "use_newton_cg", False))
        use_target_block = bool(_bfgs_qf_option(self, "use_target_block", False))
        use_escape_negcurv = bool(_bfgs_qf_option(self, "escape_negative_curvature", False))
        reset_ncg_hinv = bool(
            _bfgs_qf_option(
                self,
                "newton_cg_reset_bfgs_hessian",
                True,
                lbfgs_suffix="newton_cg_reset_lbfgs_history",
            )
        )
        reset_block_hinv = bool(
            _bfgs_qf_option(
                self,
                "target_block_reset_bfgs_hessian",
                True,
                lbfgs_suffix="target_block_reset_lbfgs_history",
            )
        )
        reset_escape_hinv = bool(
            _bfgs_qf_option(
                self,
                "escape_negcurv_reset_bfgs_hessian",
                True,
                lbfgs_suffix="escape_negcurv_reset_lbfgs_history",
            )
        )

        if use_escape_negcurv and use_newton_cg:
            raise ValueError(
                "bfgs_qf_escape_negative_curvature uses the target-block protocol "
                "and cannot be combined with bfgs_qf_use_newton_cg=True."
            )
        if use_escape_negcurv and not use_target_block:
            raise ValueError(
                "bfgs_qf_escape_negative_curvature=True requires "
                "bfgs_qf_use_target_block=True because the escape direction uses "
                "the target-block Hessian eigenvectors."
            )
        if use_newton_cg and use_target_block:
            raise ValueError(
                "Only one bfgs_qf acceleration protocol may be active at a time: "
                "set either bfgs_qf_use_newton_cg=True or "
                "bfgs_qf_use_target_block=True, not both."
            )

        self._bfgs_qf_effective_options = {
            "maxiter": maxiter,
            "gconv": gconv,
            "econv": econv,
            "max_ls": max_ls,
            "c1": c1,
            "alpha0": alpha0,
            "line_search": line_search,
            "max_step_norm": None if not np.isfinite(max_step_norm) else max_step_norm,
            "max_abs_step": None if not np.isfinite(max_abs_step) else max_abs_step,
            "curvature_tol": curvature_tol,
            "step_tol": step_tol,
            "init_scale": float(_bfgs_qf_option(self, "init_scale", 1.0)),
            "reset_on_bad_curvature": reset_on_bad_curv,
            "reset_on_nondescent": reset_on_nondescent,
            "max_params_dense": max_params_dense,
            "use_hessian_diag": use_hdiag,
            "hdiag_start": hdiag_start,
            "hdiag_stop": hdiag_stop,
            "hdiag_update_freq": hdiag_update_freq,
            "hdiag_floor": hdiag_floor,
            "hdiag_mode": hdiag_mode,
            "hdiag_method": _bfgs_qf_option(self, "hdiag_method", "analytic"),
            "hdiag_fd_step": _bfgs_qf_option(self, "hdiag_fd_step", 1.0e-4),
            "use_gradient_energy": use_gradient_energy,
        }

        allowed_triggers = {"never", "periodic", "stalled", "periodic_or_stalled"}
        trigger = _bfgs_qf_option(self, "newton_cg_trigger", "stalled")
        target_trigger = _bfgs_qf_option(self, "target_block_trigger", "stalled")
        if trigger not in allowed_triggers:
            raise ValueError(
                "Unknown bfgs_qf_newton_cg_trigger. Expected one of "
                f"{sorted(allowed_triggers)}; got {trigger!r}."
            )
        if target_trigger not in allowed_triggers:
            raise ValueError(
                "Unknown bfgs_qf_target_block_trigger. Expected one of "
                f"{sorted(allowed_triggers)}; got {target_trigger!r}."
            )
        escape_trigger = _bfgs_qf_option(self, "escape_negcurv_trigger", "persistent")
        allowed_escape_triggers = {
            "never",
            "always",
            "small_gradient",
            "stalled",
            "persistent",
            "stalled_or_persistent",
        }
        if escape_trigger not in allowed_escape_triggers:
            raise ValueError(
                "Unknown bfgs_qf_escape_negcurv_trigger. Expected one of "
                f"{sorted(allowed_escape_triggers)}; got {escape_trigger!r}."
            )

        x = np.array(copy.deepcopy(self._tamps), dtype=float)
        n_params = len(x)
        if n_params > max_params_dense:
            raise ValueError(
                f'optimizer="bfgs_qf" would allocate a dense {n_params}x{n_params} '
                "inverse Hessian. Increase bfgs_qf_max_params_dense only if this "
                'is intentional, or use optimizer="lbfgs_qf" instead.'
            )

        cached_h_eff = None
        cached_hdiag_iter = None
        Hinv = _bfgs_qf_initial_hinv(self, n_params, h_eff=None)

        nfev = 0
        njev = 0
        self._k_counter = 0
        self._bfgs_qf_updates_accepted = 0
        self._bfgs_qf_updates_skipped = 0
        self._bfgs_qf_hinv_resets = 0
        self._bfgs_qf_nondescent_fallbacks = 0
        self._lbfgs_qf_newton_cg_attempts = 0
        self._lbfgs_qf_newton_cg_accepted = 0
        self._lbfgs_qf_newton_cg_hv_evals = 0
        self._lbfgs_qf_newton_cg_last_info = {}
        self._lbfgs_qf_target_block_attempts = 0
        self._lbfgs_qf_target_block_accepted = 0
        self._lbfgs_qf_target_block_grad_evals = 0
        self._lbfgs_qf_target_block_last_info = {}
        self._lbfgs_qf_escape_negcurv_attempts = 0
        self._lbfgs_qf_escape_negcurv_accepted = 0
        self._lbfgs_qf_escape_negcurv_last_info = {}
        negcurv_count = 0
        negcurv_escape_count = 0
        last_block_lam_min = None

        initial_hdiag_due = _lbfgs_qf_hdiag_is_due(
            1,
            use_hdiag,
            hdiag_start,
            hdiag_stop,
            hdiag_update_freq,
            cached_h_eff,
            hdiag_mode,
        )
        if initial_hdiag_due:
            E, g, h_diag, nfev_inc, njev_inc, combined_hdiag = _qf_gradient_hdiag_feval(
                self,
                x,
                objective=objective,
                return_energy=True,
                need_hdiag=True,
                use_gradient_energy=use_gradient_energy,
            )
            if h_diag is not None:
                if len(h_diag) != n_params:
                    raise ValueError(
                        'hessian_diag_ary_feval(params) returned a vector of length '
                        f'{len(h_diag)}, but there are {n_params} amplitudes.'
                    )
                cached_h_eff = _lbfgs_qf_regularized_hdiag(h_diag, hdiag_floor, hdiag_mode)
                cached_hdiag_iter = 1
                Hinv = _bfgs_qf_reset_hinv(self, n_params, h_eff=cached_h_eff)
            elif not combined_hdiag:
                h_diag, hdiag_njev = _qf_hessian_diag_feval(self, x, n_params, "bfgs_qf")
                njev_inc += hdiag_njev
                cached_h_eff = _lbfgs_qf_regularized_hdiag(h_diag, hdiag_floor, hdiag_mode)
                cached_hdiag_iter = 1
                Hinv = _bfgs_qf_reset_hinv(self, n_params, h_eff=cached_h_eff)
        else:
            E, g, nfev_inc, njev_inc = _qf_gradient_energy_feval(
                self,
                x,
                objective=objective,
                use_gradient_energy=use_gradient_energy,
            )
        nfev += nfev_inc
        njev += njev_inc
        self._prev_energy = E
        grad_norm = np.linalg.norm(g)
        self._curr_grad_norm = grad_norm
        gnorm_history = [grad_norm]

        print('  \n--> Begin qforte full BFGS optimization:')
        print(f" Initial guess energy:              {E:+12.10f}")
        print(
            " BFGS controls:                    "
            f"alpha0={alpha0:.2e}, max_step_norm="
            f"{self._bfgs_qf_effective_options['max_step_norm']}, "
            f"max_abs_step={self._bfgs_qf_effective_options['max_abs_step']}, "
            f"max_ls={max_ls}"
        )
        print(
            " BFGS tolerances:                  "
            f"gconv={gconv:.2e}, econv={econv:.2e}, step_tol={step_tol:.2e}"
        )
        if use_hdiag:
            print(
                " Hessian diagonal:                 "
                f"mode={hdiag_mode}, floor={hdiag_floor:.2e}, "
                f"start={hdiag_start}, stop={hdiag_stop}, "
                f"freq={hdiag_update_freq}, "
                f"method={self._bfgs_qf_effective_options['hdiag_method']}"
            )
        if use_newton_cg:
            print(
                ' Newton-CG acceleration:          enabled '
                f'(trigger={trigger}, start={getattr(self, "_lbfgs_qf_newton_cg_start", 10)})'
            )
        if use_target_block:
            print(
                ' Target-block acceleration:      enabled '
                f'(trigger={target_trigger}, start={getattr(self, "_lbfgs_qf_target_block_start", 10)})'
            )
        if use_escape_negcurv:
            print(
                ' Negative-curvature escape:      enabled '
                f'(trigger={escape_trigger}, '
                f'persist={getattr(self, "_lbfgs_qf_escape_negcurv_persist_count", 3)})'
            )

        header = (
            f' {"iter":>5} | {"Energy":^16} | {"dE":^16} | {"||g||":^16} | '
            f'{"step":^6} | {"stat":^6} | {"alpha":^7} | {"||step||":^8} | '
            f'{"Upd":^4} | {"evals(f/g/m)":^15}'
        )
        if bool(getattr(self, "_lbfgs_qf_print_aux", False)):
            header += ' | aux'
        print('\n' + header)
        print('-' * len(header))

        success = grad_norm < gconv
        message = 'Initial gradient norm below threshold.' if success else 'Maximum iterations reached.'
        last_status = 'initial'
        last_step_norm = 0.0
        last_alpha = 0.0

        if not success:
            for k in range(1, maxiter + 1):
                self._k_counter = k

                if _lbfgs_qf_hdiag_is_due(
                    k,
                    use_hdiag,
                    hdiag_start,
                    hdiag_stop,
                    hdiag_update_freq,
                    cached_h_eff,
                    hdiag_mode,
                ):
                    if cached_hdiag_iter != k:
                        h_diag, hdiag_njev = _qf_hessian_diag_feval(
                            self,
                            x,
                            n_params,
                            "bfgs_qf",
                        )
                        njev += hdiag_njev
                        cached_h_eff = _lbfgs_qf_regularized_hdiag(h_diag, hdiag_floor, hdiag_mode)
                        cached_hdiag_iter = k
                    Hinv = _bfgs_qf_reset_hinv(self, n_params, h_eff=cached_h_eff)
                    self._curr_grad_norm = grad_norm

                if _lbfgs_qf_should_attempt_target_block(self, k, gnorm_history):
                    stalled = _lbfgs_qf_is_stalled(self, gnorm_history)
                    bfgs_step_for_selection = None
                    if getattr(self, "_lbfgs_qf_target_block_selection", "preconditioned_gradient") == "lbfgs_step":
                        candidate_step = -(Hinv @ g)
                        if np.all(np.isfinite(candidate_step)):
                            bfgs_step_for_selection = candidate_step

                    (
                        block_accepted,
                        x_block,
                        E_block,
                        g_block,
                        block_alpha,
                        block_step_norm,
                        block_info,
                    ) = _lbfgs_qf_try_target_block_step(
                        self,
                        x,
                        E,
                        g,
                        h_eff=cached_h_eff,
                        lbfgs_step=bfgs_step_for_selection,
                        objective=objective,
                        k=k,
                        stalled=stalled,
                        negcurv_count=negcurv_count,
                        negcurv_escape_count=negcurv_escape_count,
                    )
                    nfev += block_info.get("nfev", 0)
                    njev += block_info.get("grad_evals", 0)
                    negcurv_count = block_info.get("negcurv_count", negcurv_count)
                    negcurv_escape_count = block_info.get(
                        "negcurv_escape_count",
                        negcurv_escape_count,
                    )
                    last_block_lam_min = block_info.get(
                        "lam_min",
                        block_info.get("eig_min", last_block_lam_min),
                    )

                    if block_accepted:
                        target_step_type = block_info.get("step_type", "BLOCK")
                        s = x_block - x
                        y = g_block - g
                        reset_target_hinv = (
                            reset_escape_hinv
                            if target_step_type == "ESCAPE"
                            else reset_block_hinv
                        )

                        if reset_target_hinv:
                            Hinv = _bfgs_qf_reset_hinv(self, n_params, h_eff=cached_h_eff)
                            curv_status = "reset"
                        else:
                            Hinv, curv_info = _bfgs_qf_update_hinv(
                                self, Hinv, s, y, curvature_tol
                            )
                            curv_status = "accepted" if curv_info["accepted"] else "skip_y"

                        dE = E_block - E
                        x = x_block
                        E = E_block
                        g = g_block
                        grad_norm = np.linalg.norm(g)
                        self._curr_energy = E
                        self._curr_grad_norm = grad_norm
                        self._prev_energy = E
                        gnorm_history.append(grad_norm)
                        last_status = curv_status
                        last_step_norm = block_step_norm
                        last_alpha = block_alpha

                        _lbfgs_qf_print_iteration(
                            self,
                            k,
                            E,
                            dE,
                            nfev,
                            grad_norm,
                            target_step_type,
                            last_alpha,
                            last_step_norm,
                            last_status,
                            getattr(self, "_bfgs_qf_updates_accepted", 0),
                            block_info=block_info,
                        )

                        if grad_norm < gconv:
                            success = True
                            message = 'Gradient norm below threshold.'
                            break

                        if econv is not None and abs(dE) < econv:
                            success = True
                            message = 'Energy change below bfgs_qf energy tolerance.'
                            break

                        if last_step_norm < step_tol:
                            message = 'Step norm below bfgs_qf step tolerance.'
                            break

                        continue

                    print(
                        '  => Target block rejected: '
                        f'{block_info.get("reason", "unknown")}'
                    )

                if _lbfgs_qf_should_attempt_newton_cg(self, k, gnorm_history):
                    (
                        ncg_accepted,
                        x_ncg,
                        E_ncg,
                        g_ncg,
                        ncg_alpha,
                        ncg_step_norm,
                        ncg_info,
                    ) = _lbfgs_qf_try_newton_cg_step(
                        self,
                        x,
                        E,
                        g,
                        h_eff=cached_h_eff,
                        objective=objective,
                    )
                    nfev += ncg_info.get("nfev", 0)
                    njev += ncg_info.get("grad_evals", 0)

                    if ncg_accepted:
                        s = x_ncg - x
                        y = g_ncg - g

                        if reset_ncg_hinv:
                            Hinv = _bfgs_qf_reset_hinv(self, n_params, h_eff=cached_h_eff)
                            curv_status = "reset"
                        else:
                            Hinv, curv_info = _bfgs_qf_update_hinv(
                                self, Hinv, s, y, curvature_tol
                            )
                            curv_status = "accepted" if curv_info["accepted"] else "skip_y"

                        dE = E_ncg - E
                        x = x_ncg
                        E = E_ncg
                        g = g_ncg
                        grad_norm = np.linalg.norm(g)
                        self._curr_energy = E
                        self._curr_grad_norm = grad_norm
                        self._prev_energy = E
                        gnorm_history.append(grad_norm)
                        last_status = curv_status
                        last_step_norm = ncg_step_norm
                        last_alpha = ncg_alpha

                        _lbfgs_qf_print_iteration(
                            self,
                            k,
                            E,
                            dE,
                            nfev,
                            grad_norm,
                            "NCG",
                            last_alpha,
                            last_step_norm,
                            last_status,
                            getattr(self, "_bfgs_qf_updates_accepted", 0),
                        )

                        if grad_norm < gconv:
                            success = True
                            message = 'Gradient norm below threshold.'
                            break

                        if econv is not None and abs(dE) < econv:
                            success = True
                            message = 'Energy change below bfgs_qf energy tolerance.'
                            break

                        if last_step_norm < step_tol:
                            message = 'Step norm below bfgs_qf step tolerance.'
                            break

                        continue

                    print(
                        '  => Newton-CG trial rejected: '
                        f'{ncg_info.get("reason", "unknown")}'
                    )

                p = -(Hinv @ g)
                gdotp = float(np.dot(g, p))
                last_status = "trial"
                if not np.all(np.isfinite(p)) or not np.isfinite(gdotp) or gdotp >= 0.0:
                    self._bfgs_qf_nondescent_fallbacks += 1
                    if reset_on_nondescent:
                        Hinv = _bfgs_qf_reset_hinv(self, n_params, h_eff=cached_h_eff)
                        p = -(Hinv @ g)
                        gdotp = float(np.dot(g, p))
                        last_status = "reset"
                    if not np.all(np.isfinite(p)) or not np.isfinite(gdotp) or gdotp >= 0.0:
                        p = -g
                        gdotp = float(np.dot(g, p))
                        last_status = "sd"

                accepted = False
                alpha = alpha0
                step = np.zeros_like(x)
                E_trial = E
                x_trial = x
                g_trial = None

                for _ in range(max_ls):
                    step = _lbfgs_qf_capped_step(alpha * p, max_abs_step, max_step_norm)
                    step_norm = np.linalg.norm(step)
                    gdotstep = float(np.dot(g, step))
                    if step_norm == 0.0 or not np.isfinite(gdotstep) or gdotstep >= 0.0:
                        break
                    x_trial = x + step
                    E_trial = float(np.real(objective(x_trial)))
                    g_trial = None
                    nfev += 1
                    if np.isfinite(E_trial) and E_trial < E and E_trial <= E + c1 * gdotstep:
                        accepted = True
                        break
                    alpha *= 0.5

                if not accepted:
                    message = 'Line search failed to find an acceptable descent step.'
                    last_status = 'failed'
                    _lbfgs_qf_print_iteration(
                        self,
                        k,
                        E,
                        0.0,
                        nfev,
                        grad_norm,
                        "BFGS",
                        0.0,
                        0.0,
                        last_status,
                        getattr(self, "_bfgs_qf_updates_accepted", 0),
                    )
                    break

                if g_trial is None:
                    next_k = k + 1
                    need_next_hdiag = (
                        next_k <= maxiter
                        and _lbfgs_qf_hdiag_is_due(
                            next_k,
                            use_hdiag,
                            hdiag_start,
                            hdiag_stop,
                            hdiag_update_freq,
                            cached_h_eff,
                            hdiag_mode,
                        )
                    )
                    _, g_trial, h_diag_next, _, njev_inc, _ = _qf_gradient_hdiag_feval(
                        self,
                        x_trial,
                        objective=objective,
                        return_energy=False,
                        need_hdiag=need_next_hdiag,
                        use_gradient_energy=use_gradient_energy,
                    )
                    njev += njev_inc
                    if h_diag_next is not None:
                        if len(h_diag_next) != n_params:
                            raise ValueError(
                                'hessian_diag_ary_feval(params) returned a vector of length '
                                f'{len(h_diag_next)}, but there are {n_params} amplitudes.'
                            )
                        cached_h_eff = _lbfgs_qf_regularized_hdiag(
                            h_diag_next,
                            hdiag_floor,
                            hdiag_mode,
                        )
                        cached_hdiag_iter = next_k
                s = x_trial - x
                y = g_trial - g

                Hinv, curv_info = _bfgs_qf_update_hinv(self, Hinv, s, y, curvature_tol)
                if curv_info["accepted"]:
                    if last_status == "trial":
                        last_status = "accepted"
                else:
                    print(f'  => Skipping BFGS update: y.s = {curv_info["ys"]:+.6e}')
                    if reset_on_bad_curv:
                        Hinv = _bfgs_qf_reset_hinv(self, n_params, h_eff=cached_h_eff)
                        last_status = "reset_y"
                    elif last_status == "trial":
                        last_status = "skip_y"

                if not np.all(np.isfinite(Hinv)):
                    Hinv = _bfgs_qf_reset_hinv(self, n_params, h_eff=cached_h_eff)
                    last_status = "reset"

                dE = E_trial - E
                x = x_trial
                E = E_trial
                g = g_trial
                grad_norm = np.linalg.norm(g)
                self._curr_energy = E
                self._curr_grad_norm = grad_norm
                last_step_norm = np.linalg.norm(step)
                last_alpha = alpha
                self._prev_energy = E
                gnorm_history.append(grad_norm)

                _lbfgs_qf_print_iteration(
                    self,
                    k,
                    E,
                    dE,
                    nfev,
                    grad_norm,
                    "BFGS" if last_status != "sd" else "SD",
                    last_alpha,
                    last_step_norm,
                    last_status,
                    getattr(self, "_bfgs_qf_updates_accepted", 0),
                )

                if grad_norm < gconv:
                    success = True
                    message = 'Gradient norm below threshold.'
                    break

                if econv is not None and abs(dE) < econv:
                    success = True
                    message = 'Energy change below bfgs_qf energy tolerance.'
                    break

                if last_step_norm < step_tol:
                    message = 'Step norm below bfgs_qf step tolerance.'
                    break

        if success:
            print('  => Minimization successful!')
        else:
            print('  => WARNING: minimization result may not be tightly converged.')
        print(f'  => Minimum Energy: {E:+12.10f}')

        res = OptimizeResult(
            x=np.array(x, dtype=float),
            fun=E,
            jac=np.array(g, dtype=float),
            success=success,
            message=message,
            nit=self._k_counter,
            nfev=nfev,
            njev=njev,
            grad_norm=grad_norm,
            step_norm=last_step_norm,
            alpha=last_alpha,
            status=last_status,
            nhev=getattr(self, "_lbfgs_qf_newton_cg_hv_evals", 0),
            bfgs_updates_accepted=getattr(self, "_bfgs_qf_updates_accepted", 0),
            bfgs_updates_skipped=getattr(self, "_bfgs_qf_updates_skipped", 0),
            bfgs_hinv_resets=getattr(self, "_bfgs_qf_hinv_resets", 0),
            bfgs_nondescent_fallbacks=getattr(self, "_bfgs_qf_nondescent_fallbacks", 0),
            newton_cg_attempts=getattr(self, "_lbfgs_qf_newton_cg_attempts", 0),
            newton_cg_accepted=getattr(self, "_lbfgs_qf_newton_cg_accepted", 0),
            newton_cg_last_info=getattr(self, "_lbfgs_qf_newton_cg_last_info", {}),
            target_block_attempts=getattr(self, "_lbfgs_qf_target_block_attempts", 0),
            target_block_accepted=getattr(self, "_lbfgs_qf_target_block_accepted", 0),
            target_block_grad_evals=getattr(self, "_lbfgs_qf_target_block_grad_evals", 0),
            target_block_last_info=getattr(self, "_lbfgs_qf_target_block_last_info", {}),
            escape_negcurv_attempts=getattr(self, "_lbfgs_qf_escape_negcurv_attempts", 0),
            escape_negcurv_accepted=getattr(self, "_lbfgs_qf_escape_negcurv_accepted", 0),
            escape_negcurv_last_info=getattr(self, "_lbfgs_qf_escape_negcurv_last_info", {}),
            escape_negcurv_count=negcurv_count,
            escape_negcurv_last_lam_min=last_block_lam_min,
            bfgs_qf_effective_options=copy.deepcopy(
                getattr(self, "_bfgs_qf_effective_options", {})
            ),
        )

        _lbfgs_qf_finalize(self, res)
        return res
    finally:
        _bfgs_qf_restore_shared_lbfgs_options(self, backup, missing)


def lbfgs_solver(self, function_to_minimize=None):
    """Backward-compatible wrapper for the in-house lbfgs_qf driver."""
    return lbfgs_qf_solve(self, function_to_minimize=function_to_minimize)
