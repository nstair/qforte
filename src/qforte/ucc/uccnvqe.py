"""
UCCNVQE classes
====================================
Classes for using an experiment to execute the variational quantum eigensolver
for a Trotterized (disentangeld) UCCN ansatz with fixed operators.
"""

import qforte
from qforte.abc.uccvqeabc import UCCVQE

from qforte.experiment import *
from qforte.maths import optimizer
from qforte.maths.optimizer import set_lbfgs_qf_options
from qforte.utils.transforms import *
from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize
from qforte.utils import moment_energy_corrections

import numpy as np
from scipy.optimize import minimize, OptimizeResult

class UCCNVQE(UCCVQE):
    """A class that encompasses the three components of using the variational
    quantum eigensolver to optimize a parameterized disentangled UCCN-like
    wave function. (1) prepares a quantum state on the quantum computer
    representing the wave function to be simulated, (2) evauates the energy and
    gradients (3) optemizes the the wave funciton by minimizing the energy

    Attributes
    ----------
    _results : list
        The optimizer result objects from each iteration of UCCN-VQE.

    _energies : list
        The optimized energies from each iteration of UCCN-VQE.

    _grad_norms : list
        The gradient norms from each iteration of UCCN-VQE.

    """
    def run(self,
            opt_thresh=1.0e-5,
            opt_ftol=1.0e-5,
            opt_maxiter=200,
            pool_type='SD',
            optimizer='BFGS',
            use_analytic_grad = True,
            noise_factor = 0.0,
            psi_i=None,
            init_amps='zero',
            primary_pool_order='none',
            secondary_pool_order='lexical',
            general_ex_pool_order='default',
            batched_opt_type='none',
            batched_opt_cycles=1,
            batch_opt_thresh=None,
            batched_opt_batch_maxiter=None,
            batched_opt_final_maxiter=None,
            batched_opt_freeze_inactive=True,
            batched_opt_verbose=True,
            batched_opt_reset_optimizer_state_between_batches=True,
            batched_opt_keep_full_tamps=True,
            batched_opt_require_nonempty_batches=True,
            **kwargs):

        if "initial_amplitudes" in kwargs:
            if init_amps != 'zero':
                raise ValueError(
                    'Specify only one initial-amplitude option: '
                    'init_amps or initial_amplitudes.'
                )
            init_amps = kwargs.pop("initial_amplitudes")

        set_lbfgs_qf_options(self, kwargs)

        init_amps = str(init_amps).lower()
        if init_amps not in ["zero", "mp2"]:
            raise ValueError('init_amps must be "zero" or "mp2".')

        primary_pool_order = str(primary_pool_order).lower()
        secondary_pool_order = str(secondary_pool_order).lower()
        general_ex_pool_order = str(general_ex_pool_order).lower()
        if primary_pool_order not in ["none", "mp2_amps", "gradients"]:
            raise ValueError(
                'primary_pool_order must be "none", "mp2_amps", or '
                '"gradients".'
            )
        if secondary_pool_order not in ["lexical", "shell"]:
            raise ValueError('secondary_pool_order must be "lexical" or "shell".')
        if general_ex_pool_order not in ["default", "particle_hole_first"]:
            raise ValueError(
                'general_ex_pool_order must be "default" or '
                '"particle_hole_first".'
            )
        batched_opt_type = str(batched_opt_type).lower()
        allowed_batched_opt_types = [
            "none",
            "half_sweep",
            "full_sweep",
            "half_sweep_then_all",
            "full_sweep_then_all",
        ]
        if batched_opt_type not in allowed_batched_opt_types:
            raise ValueError(
                "batched_opt_type must be one of "
                f"{allowed_batched_opt_types}; got {batched_opt_type!r}."
            )

        if isinstance(pool_type, str):
            if pool_type == "GSDx":
                print(
                    'Warning: pool_type="GSDx" is deprecated. Use '
                    'pool_type="GSD" with '
                    'general_ex_pool_order="particle_hole_first".'
                )
                pool_type = "GSD"
                general_ex_pool_order = "particle_hole_first"
            elif pool_type.endswith("-UpCCGSDx"):
                k_label = pool_type[:-len("-UpCCGSDx")]
                if k_label.isdigit():
                    print(
                        f'Warning: pool_type="{pool_type}" is deprecated. '
                        f'Use pool_type="{k_label}-UpCCGSD" with '
                        'general_ex_pool_order="particle_hole_first".'
                    )
                    pool_type = f"{k_label}-UpCCGSD"
                    general_ex_pool_order = "particle_hole_first"

        self._opt_thresh = opt_thresh
        self._opt_ftol = opt_ftol
        self._opt_maxiter = opt_maxiter
        self._use_analytic_grad = use_analytic_grad
        self._optimizer = optimizer
        self._pool_type = pool_type
        self._noise_factor = noise_factor
        self._init_amps = init_amps
        self._primary_pool_order = primary_pool_order
        self._secondary_pool_order = secondary_pool_order
        self._general_ex_pool_order = general_ex_pool_order
        self._batched_opt_type = batched_opt_type
        self._batched_opt_cycles = int(batched_opt_cycles)
        self._batch_opt_thresh = float(
            opt_thresh if batch_opt_thresh is None else batch_opt_thresh
        )
        self._batched_opt_batch_maxiter = batched_opt_batch_maxiter
        self._batched_opt_final_maxiter = batched_opt_final_maxiter
        self._batched_opt_freeze_inactive = bool(batched_opt_freeze_inactive)
        self._batched_opt_verbose = bool(batched_opt_verbose)
        self._batched_opt_reset_optimizer_state_between_batches = bool(
            batched_opt_reset_optimizer_state_between_batches
        )
        self._batched_opt_keep_full_tamps = bool(batched_opt_keep_full_tamps)
        self._batched_opt_require_nonempty_batches = bool(
            batched_opt_require_nonempty_batches
        )
        self._batched_opt_history = []

        self._tops = []
        self._tamps = []
        self._conmutator_pool = []
        self._converged = 0

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0
        self._res_vec_evals = 0
        self._res_m_evals = 0
        self._k_counter = 0

        self._curr_grad_norm = 0.0

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        self._timer = qforte.local_timer()

        ######### UCCN-VQE #########

        self._timer.reset()
        self.fill_pool()
        self._timer.record("fill_pool")

        if self._verbose:
            print(self._pool_obj.str())

        self._timer.reset()
        self.initialize_ansatz()
        self._timer.record("initialize_ansatz")

        self._timer.reset()
        self.apply_pool_ordering()
        self._timer.record("apply_pool_ordering")

        if self._computer_type == 'fci_gpu':
            optimizer_name = self._optimizer.lower()
            if optimizer_name in {"lbfgs_qf", "bfgs_qf"}:
                raise NotImplementedError(
                    f'optimizer="{self._optimizer}" is not yet implemented for '
                    'computer_type="fci_gpu". The in-house qf optimizer path '
                    "needs FCIComputerGPU derivative support and will be added "
                    "in a future PR."
                )
            if getattr(self, "_batched_opt_type", "none") != "none":
                raise NotImplementedError(
                    'batched_opt_type is not yet implemented for computer_type="fci_gpu". '
                    "The batched optimizer path needs reduced-space FCIComputerGPU "
                    "energy/gradient support and will be added in a future PR."
                )

        # Initialize reusable pool/computer state before optimizer callbacks.
        if self._computer_type == 'fci_gpu':
            qforte.gpu_only(True)

            self._timer.reset()
            self.initialize_gpu_pool()
            self._timer.record("initialize_gpu_pool")
        elif self._computer_type in {'fock', 'fci', 'fqe', 'cusv'}:
            self._timer.reset()
            self.initialize_reusable_ucc_objects()
            self._timer.record("initialize_reusable_ucc_objects")

        if(self._verbose):
            print('\nt operators included from pool: \n', self._tops)
            print('\nInitial tamplitudes for tops: \n', self._tamps)

        self._timer.reset()
        self.solve()
        self._timer.record("solve")

        if self._max_moment_rank:
            print('\nConstructing Moller-Plesset and Epstein-Nesbet denominators')
            self.construct_moment_space()
            print('\nComputing non-iterative energy corrections')
            self.compute_moment_energies()

        if(self._verbose):
            print('\nt operators included from pool: \n', self._tops)
            print('\nFinal tamplitudes for tops: \n', self._tamps)

        ######### UCCSD-VQE #########
        self._n_nonzero_params = 0
        for tmu in self._tamps:
            if(np.abs(tmu) > 1.0e-12):
                self._n_nonzero_params += 1

        self.psi_i = psi_i

        # verify that required attributes were defined
        # (should be called for all algorithms!)
        self.verify_run()

        self.print_summary_banner()

    # Define Algorithm abstract methods.
    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not fully implemented for UCCN-VQE.')

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_VQE_attributes()
        self.verify_required_UCCVQE_attributes()

    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('          Unitary Coupled Cluster VQE   ')
        print('-----------------------------------------------------')

        print('\n\n               ==> UCCN-VQE options <==')
        print('---------------------------------------------------------')
        # General algorithm options.
        print('Trial reference state:                   ',  ref_string(self._ref, self._nqb))
        print('Number of Hamiltonian Pauli terms:       ',  self._Nl)
        print('Trial state preparation method:          ',  self._state_prep_type)
        print('Trotter order (rho):                     ',  self._trotter_order)
        print('Trotter number (m):                      ',  self._trotter_number)
        print('Use fast version of algorithm:           ',  str(self._fast))
        if(self._fast):
            print('Measurement variance thresh:             ',  'NA')
        else:
            print('Measurement variance thresh:             ',  0.01)

        print('Use qubit excitations:                   ', self._qubit_excitations)
        print('Use compact excitation circuits:         ', self._compact_excitations)

        # VQE options.
        opt_thrsh_str = '{:.2e}'.format(self._opt_thresh)
        print('Optimization algorithm:                  ',  self._optimizer)
        print('Optimization maxiter:                    ',  self._opt_maxiter)
        print('Optimizer grad-norm threshold (theta):   ',  opt_thrsh_str)

        # UCCVQE options.
        print('Use analytic gradient:                   ',  str(self._use_analytic_grad))
        print('Operator pool type:                      ',  str(self._pool_type))
        print('Initial amplitudes:                      ',  str(self._init_amps))
        print('Primary pool order:                      ',  str(self._primary_pool_order))
        print('Secondary pool order:                    ',  str(self._secondary_pool_order))
        print('Generalized pool order:                  ',  str(self._general_ex_pool_order))
        print('Batched optimization type:               ',  str(self._batched_opt_type))
        if self._batched_opt_type != "none":
            batch_thrsh_str = '{:.2e}'.format(self._batch_opt_thresh)
            print('Batch grad-norm threshold:                ',  batch_thrsh_str)
        print(f"Computer type:                            {self._computer_type}")
        b = False
        if (self._apply_ham_as_tensor):
            b = True
        print('Apply ham as tensor                      ', str(b))

    def print_summary_banner(self):

        print('\n\n                ==> UCCN-VQE summary <==')
        print('-----------------------------------------------------------')
        # print('Final UCCN-VQE Energy:                      ', round(self._Egs, 10))
        print(f"Final UCCN-VQE Energy:                      {self._Egs:12.10f}")
        print(f"Final <S^2>:                                {self.final_spin_squared_summary_string()}")
        if self._max_moment_rank:
            print('Moment-corrected (MP) UCCN-VQE Energy:      ', round(self._E_mmcc_mp[0], 10))
            print('Moment-corrected (EN) UCCN-VQE Energy:      ', round(self._E_mmcc_en[0], 10))
        print('Number of operators in pool:                 ', len(self._pool_obj))
        print('Final number of amplitudes in ansatz:        ', len(self._tamps))
        print('Total number of Hamiltonian measurements:    ', self.get_num_ham_measurements())
        print('Total number of commutator measurements:     ', self.get_num_commut_measurements())
        print('Number of classical parameters used:         ', self._n_classical_params)
        print('Number of non-zero parameters used:          ', self._n_nonzero_params)
        print('Number of CNOT gates in deepest circuit:     ', self._n_cnot)
        print('Number of Pauli term measurements:           ', self._n_pauli_trm_measures)

        print('Number of grad vector evaluations:           ', self._res_vec_evals)
        print('Number of individual grad evaluations:       ', self._res_m_evals)

        if self._gpu_timers is not None:
            print('\n\n                ==> GPU Profiling <==')
            print('-----------------------------------------------------------')
            sorted_timers = sorted(self._gpu_timers.items(), key=lambda x: x[1], reverse=True)
            total_gpu_time = sum(self._gpu_timers.values())
            print(f'Total GPU time:                              {total_gpu_time:12.6f} s')
            print('\nGPU call breakdown (sorted by time):')
            for name, time in sorted_timers:
                if time > 0:
                    percent = 100.0 * time / total_gpu_time if total_gpu_time > 0 else 0.0
                    print(f'  {name:40s} {time:12.6f} s  ({percent:5.1f}%)')

        if self._fci_timers is not None:
            print('\n\n                ==> FCI Profiling <==')
            print('-----------------------------------------------------------')
            sorted_timers = sorted(self._fci_timers.items(), key=lambda x: x[1], reverse=True)
            total_fci_time = sum(self._fci_timers.values())
            print(f'Total FCI time:                              {total_fci_time:12.6f} s')
            print('\nFCI call breakdown (sorted by time):')
            for name, time in sorted_timers:
                if time > 0:
                    percent = 100.0 * time / total_fci_time if total_fci_time > 0 else 0.0
                    print(f'  {name:40s} {time:12.6f} s  ({percent:5.1f}%)')

        if self._fqe_timers is not None:
            print('\n\n                ==> FQE Profiling <==')
            print('-----------------------------------------------------------')
            sorted_timers = sorted(self._fqe_timers.items(), key=lambda x: x[1], reverse=True)
            total_fqe_time = sum(self._fqe_timers.values())
            print(f'Total FQE time:                              {total_fqe_time:12.6f} s')
            print('\nFQE call breakdown (sorted by time):')
            for name, time in sorted_timers:
                if time > 0:
                    percent = 100.0 * time / total_fqe_time if total_fqe_time > 0 else 0.0
                    print(f'  {name:40s} {time:12.6f} s  ({percent:5.1f}%)')

        if self._cusv_timers is not None:
            print('\n\n                ==> CUSV Profiling <==')
            print('-----------------------------------------------------------')
            sorted_timers = sorted(self._cusv_timers.items(), key=lambda x: x[1], reverse=True)
            total_cusv_time = sum(self._cusv_timers.values())
            print(f'Total CUSV time:                             {total_cusv_time:12.6f} s')
            print('\nCUSV call breakdown (sorted by time):')
            for name, time in sorted_timers:
                if time > 0:
                    percent = 100.0 * time / total_cusv_time if total_cusv_time > 0 else 0.0
                    print(f'  {name:40s} {time:12.6f} s  ({percent:5.1f}%)')

        print("\n\n")
        print(self._timer)
        print(self.final_noons_summary_table())

    def _validate_batched_optimization_options(self):
        """Validate opt-in batched optimization settings."""
        optimizer_name = self._optimizer.lower()
        if self._computer_type == "fci_gpu":
            raise NotImplementedError(
                'batched_opt_type is not yet implemented for computer_type="fci_gpu". '
                "The batched optimizer path needs reduced-space FCIComputerGPU "
                "energy/gradient support and will be added in a future PR."
            )
        supported = {"bfgs_qf", "lbfgs_qf", "bfgs", "l-bfgs-b"}
        if optimizer_name not in supported:
            raise ValueError(
                "batched_opt_type is only supported for optimizer=\"bfgs_qf\", "
                "optimizer=\"lbfgs_qf\", scipy \"BFGS\", and scipy \"L-BFGS-B\"."
            )
        if self._general_ex_pool_order != "particle_hole_first":
            raise ValueError(
                'batched_opt_type requires '
                'general_ex_pool_order="particle_hole_first" so PH/GEN batches '
                'are unambiguous.'
            )
        if int(self._batched_opt_cycles) < 1:
            raise ValueError("batched_opt_cycles must be at least 1.")
        if float(self._batch_opt_thresh) <= 0.0:
            raise ValueError("batch_opt_thresh must be positive.")
        if not self._batched_opt_freeze_inactive:
            raise ValueError(
                "batched_opt_freeze_inactive=False is not currently supported; "
                "inactive batch parameters are held fixed."
            )
        if not self._batched_opt_keep_full_tamps:
            raise ValueError(
                "batched_opt_keep_full_tamps=False is not currently supported; "
                "the full parameter vector is retained and updated between batches."
            )

    def _build_batched_opt_indices(self):
        """Return contiguous PH/GEN blocks in the current ordered ansatz.

        With general_ex_pool_order="particle_hole_first", GSD should look like
        PH, GEN, and k-UpCCGSD should look like PH(k=1), GEN(k=1),
        PH(k=2), GEN(k=2), ... .  Batching follows those contiguous blocks
        rather than merging all particle-hole terms across all k layers.
        """
        ph_indices = []
        gen_indices = []
        blocks = []
        current_kind = None
        current_indices = []
        kind_counts = {"PH": 0, "GEN": 0}

        def flush_block():
            if not current_indices:
                return
            kind_counts[current_kind] += 1
            block_number = kind_counts[current_kind]
            label = (
                current_kind
                if block_number == 1 and len(blocks) < 2
                else f"{current_kind}(k={block_number})"
            )
            blocks.append(
                {
                    "kind": current_kind,
                    "label": label,
                    "indices": list(current_indices),
                    "block_number": block_number,
                }
            )

        for pos, top in enumerate(self._tops):
            sq_op = self._pool_obj[top][1]
            if self._is_clean_particle_hole_excitation(sq_op) is not None:
                kind = "PH"
                ph_indices.append(pos)
            else:
                kind = "GEN"
                gen_indices.append(pos)

            if current_kind is None:
                current_kind = kind
                current_indices = [pos]
            elif kind == current_kind:
                current_indices.append(pos)
            else:
                flush_block()
                current_kind = kind
                current_indices = [pos]

        flush_block()

        if len(blocks) > 2:
            for block in blocks:
                block["label"] = f"{block['kind']}(k={block['block_number']})"

        if self._batched_opt_require_nonempty_batches:
            if not ph_indices:
                raise ValueError(
                    "Batched optimization requested, but the PH batch is empty."
                )
            if not gen_indices:
                raise ValueError(
                    "Batched optimization requested, but the GEN batch is empty. "
                    "Use a generalized pool with particle-hole-first ordering."
                )

        if not gen_indices:
            raise ValueError(
                "batched_opt_type requires a pool with generalized/non-particle-hole "
                "operators; no GEN operators were found."
            )

        if len(blocks) < 2:
            raise ValueError(
                "batched_opt_type requires separate PH and GEN blocks. "
                "Check that a generalized pool is being used with "
                'general_ex_pool_order="particle_hole_first".'
            )

        return {
            "PH": ph_indices,
            "GEN": gen_indices,
            "ALL": list(range(len(self._tamps))),
            "blocks": blocks,
        }

    def _build_batched_opt_schedule(self, batch_indices):
        """Return the requested batch sequence over contiguous PH/GEN blocks."""
        cycles = int(self._batched_opt_cycles)
        batch_type = self._batched_opt_type
        schedule = []
        forward = [
            {
                "label": block["label"],
                "kind": block["kind"],
                "indices": block["indices"],
            }
            for block in batch_indices["blocks"]
        ]

        def append_block(block):
            if schedule and schedule[-1]["indices"] == block["indices"]:
                return
            schedule.append(block)

        if batch_type in ["half_sweep", "half_sweep_then_all"]:
            for _ in range(cycles):
                for block in forward:
                    append_block(block)
        elif batch_type in ["full_sweep", "full_sweep_then_all"]:
            for _ in range(cycles):
                # Sweep out and back without immediately re-optimizing the
                # turning-point block. For k=2:
                # PH1 -> GEN1 -> PH2 -> GEN2 -> PH2 -> GEN1 -> PH1
                for block in forward:
                    append_block(block)
                for block in reversed(forward[:-1]):
                    append_block(block)
        else:
            raise ValueError(f"Unknown batched_opt_type: {batch_type}")

        if batch_type.endswith("_then_all"):
            schedule.append(
                {
                    "label": "ALL",
                    "kind": "ALL",
                    "indices": batch_indices["ALL"],
                }
            )
        return schedule

    def _batch_scipy_options(self, maxiter, opt_thresh):
        """Return scipy minimize options matching the ordinary scipy path."""
        opts = {"disp": True, "maxiter": maxiter}
        if self._optimizer in ['BFGS', 'CG', 'L-BFGS-B', 'TNC', 'trust-constr']:
            opts['gtol'] = opt_thresh
        if self._optimizer == 'Nelder-Mead':
            opts['fatol'] = self._opt_ftol
        if self._optimizer in ['Powell', 'L-BFGS-B', 'TNC', 'SLSQP']:
            opts['ftol'] = self._opt_ftol
        if self._optimizer == 'COBYLA':
            opts['tol'] = self._opt_ftol
        if self._optimizer in ['L-BFGS-B', 'TNC']:
            opts['maxfun'] = maxiter
        return opts

    def _with_full_batch_state(self, full_tops, full_params, func, arg):
        """Evaluate a full-space method while a reduced batch optimizer is active."""
        saved_tops = self._tops
        saved_tamps = self._tamps
        self._tops = full_tops
        self._tamps = list(np.asarray(full_params, dtype=float))
        try:
            return func(arg)
        finally:
            self._tops = saved_tops
            self._tamps = saved_tamps

    def _optimize_parameter_subset(
        self,
        x_full,
        active_indices,
        maxiter,
        label,
        batch_number,
        total_batches,
        opt_thresh,
    ):
        """Optimize only `active_indices`, embedding trial vectors in x_full."""
        active_indices = np.asarray(active_indices, dtype=int)
        x_base = np.asarray(x_full, dtype=float).copy()
        z0 = x_base[active_indices].copy()
        full_tops = list(self._tops)
        full_tamps_before = list(self._tamps)
        active_tops = [full_tops[i] for i in active_indices]

        orig_energy = self.energy_feval
        orig_grad = self.gradient_ary_feval
        orig_hdiag = getattr(self, "hessian_diag_ary_feval", None)
        orig_deriv = getattr(self, "derivative_ary_feval", None)
        orig_opt_maxiter = self._opt_maxiter
        orig_opt_thresh = self._opt_thresh
        orig_tops = self._tops
        orig_tamps = self._tamps
        orig_final_result = getattr(self, "_final_result", None)

        def embed(z):
            x_trial = x_base.copy()
            x_trial[active_indices] = np.asarray(z, dtype=float)
            return x_trial

        def energy_batch(z):
            x_trial = embed(z)
            return self._with_full_batch_state(full_tops, x_trial, orig_energy, x_trial)

        def grad_batch(z, return_energy=False):
            x_trial = embed(z)
            if return_energy:
                try:
                    E_full, g_full = self._with_full_batch_state(
                        full_tops,
                        x_trial,
                        lambda arg: orig_grad(arg, return_energy=True),
                        x_trial,
                    )
                except TypeError as exc:
                    if "return_energy" not in str(exc):
                        raise
                    E_full = self._with_full_batch_state(
                        full_tops,
                        x_trial,
                        orig_energy,
                        x_trial,
                    )
                    g_full = self._with_full_batch_state(
                        full_tops,
                        x_trial,
                        orig_grad,
                        x_trial,
                    )
                return float(np.real(E_full)), np.asarray(g_full, dtype=float)[active_indices]

            g_full = self._with_full_batch_state(full_tops, x_trial, orig_grad, x_trial)
            return np.asarray(g_full, dtype=float)[active_indices]

        def derivative_batch(
            z,
            return_energy=False,
            return_gradient=True,
            return_hessian_diag=False,
        ):
            if orig_deriv is None:
                raise NotImplementedError(
                    "This algorithm does not expose derivative_ary_feval."
                )

            x_trial = embed(z)
            saved_energy = self.energy_feval
            saved_grad = self.gradient_ary_feval
            saved_hdiag = getattr(self, "hessian_diag_ary_feval", None)
            saved_deriv = getattr(self, "derivative_ary_feval", None)
            self.energy_feval = orig_energy
            self.gradient_ary_feval = orig_grad
            if orig_hdiag is not None:
                self.hessian_diag_ary_feval = orig_hdiag
            self.derivative_ary_feval = orig_deriv
            try:
                bundle = self._with_full_batch_state(
                    full_tops,
                    x_trial,
                    lambda arg: orig_deriv(
                        arg,
                        return_energy=return_energy,
                        return_gradient=return_gradient,
                        return_hessian_diag=return_hessian_diag,
                    ),
                    x_trial,
                )
            finally:
                self.energy_feval = saved_energy
                self.gradient_ary_feval = saved_grad
                if saved_hdiag is not None:
                    self.hessian_diag_ary_feval = saved_hdiag
                elif hasattr(self, "hessian_diag_ary_feval"):
                    try:
                        delattr(self, "hessian_diag_ary_feval")
                    except AttributeError:
                        pass
                if saved_deriv is not None:
                    self.derivative_ary_feval = saved_deriv

            out = {}
            if return_energy and "energy" in bundle:
                out["energy"] = bundle["energy"]
            if return_gradient and "gradient" in bundle:
                out["gradient"] = np.asarray(bundle["gradient"], dtype=float)[active_indices]
            if return_hessian_diag and "hessian_diag" in bundle:
                out["hessian_diag"] = np.asarray(
                    bundle["hessian_diag"],
                    dtype=float,
                )[active_indices]
            return out

        def hdiag_batch(z):
            if orig_hdiag is None:
                raise ValueError(
                    "Hessian-diagonal batching requires hessian_diag_ary_feval(params)."
                )
            x_trial = embed(z)
            saved_energy = self.energy_feval
            saved_grad = self.gradient_ary_feval
            saved_hdiag = getattr(self, "hessian_diag_ary_feval", None)
            self.energy_feval = orig_energy
            self.gradient_ary_feval = orig_grad
            self.hessian_diag_ary_feval = orig_hdiag
            try:
                h_full = self._with_full_batch_state(
                    full_tops,
                    x_trial,
                    orig_hdiag,
                    x_trial,
                )
            finally:
                self.energy_feval = saved_energy
                self.gradient_ary_feval = saved_grad
                if saved_hdiag is not None:
                    self.hessian_diag_ary_feval = saved_hdiag
            return np.asarray(h_full, dtype=float)[active_indices]

        E_initial = float(np.real(orig_energy(x_base)))

        if self._batched_opt_verbose:
            print("\n---------------------------------------------------------")
            print(f"Batch {batch_number}/{total_batches}: {label}")
            print(f"Active parameters: {len(active_indices)} / {len(x_base)}")
            print(f"Initial full energy: {E_initial:+12.10f}")
            print(f"Optimizer: {self._optimizer}")
            print("---------------------------------------------------------")

        self._tops = active_tops
        self._tamps = list(z0)
        self._opt_maxiter = int(maxiter)
        self._opt_thresh = opt_thresh
        self.energy_feval = energy_batch
        self.gradient_ary_feval = grad_batch
        if orig_deriv is not None:
            self.derivative_ary_feval = derivative_batch
        if orig_hdiag is not None:
            self.hessian_diag_ary_feval = hdiag_batch

        try:
            if self._optimizer.lower() == "lbfgs_qf":
                res = self.lbfgs_qf_solve(function_to_minimize=energy_batch)
            elif self._optimizer.lower() == "bfgs_qf":
                res = self.bfgs_qf_solve(function_to_minimize=energy_batch)
            else:
                opts = self._batch_scipy_options(int(maxiter), opt_thresh)
                res = minimize(
                    energy_batch,
                    z0,
                    method=self._optimizer,
                    jac=grad_batch if self._use_analytic_grad else None,
                    options=opts,
                )
                res.grad_norm = (
                    float(np.linalg.norm(res.jac))
                    if getattr(res, "jac", None) is not None
                    else None
                )
        finally:
            self.energy_feval = orig_energy
            self.gradient_ary_feval = orig_grad
            if orig_deriv is not None:
                self.derivative_ary_feval = orig_deriv
            if orig_hdiag is not None:
                self.hessian_diag_ary_feval = orig_hdiag
            self._opt_maxiter = orig_opt_maxiter
            self._opt_thresh = orig_opt_thresh
            self._tops = orig_tops
            self._tamps = full_tamps_before
            if orig_final_result is not None and not hasattr(self, "_final_result"):
                self._final_result = orig_final_result

        x_updated = x_base.copy()
        x_updated[active_indices] = np.asarray(res.x, dtype=float)
        E_final = float(np.real(res.fun))
        g_full = np.asarray(orig_grad(x_updated), dtype=float)
        full_grad_norm = float(np.linalg.norm(g_full))
        reduced_grad = g_full[active_indices]
        reduced_grad_norm = float(np.linalg.norm(reduced_grad))

        self._tamps = list(x_updated)
        self._tops = full_tops
        self._Egs = E_final
        self._curr_energy = E_final
        self._curr_grad_norm = full_grad_norm
        self._n_classical_params = len(self._tamps)

        info = {
            "batch_number": int(batch_number),
            "batch_label": label,
            "active_indices": [int(i) for i in active_indices],
            "active_size": int(len(active_indices)),
            "optimizer": self._optimizer,
            "opt_thresh": float(opt_thresh),
            "initial_energy": E_initial,
            "final_energy": E_final,
            "best_energy": min(E_initial, E_final),
            "final_reduced_grad_norm": reduced_grad_norm,
            "final_full_grad_norm": full_grad_norm,
            "iterations": getattr(res, "nit", None),
            "nfev": getattr(res, "nfev", None),
            "njev": getattr(res, "njev", None),
            "status": getattr(res, "status", None),
            "success": bool(getattr(res, "success", False)),
            "message": str(getattr(res, "message", "")),
        }

        if self._batched_opt_verbose:
            print(f"\nBatch {label} complete:")
            print(f"    E_initial       = {E_initial:+12.10f}")
            print(f"    E_final         = {E_final:+12.10f}")
            print(f"    dE              = {E_final - E_initial:+12.10f}")
            print(f"    reduced ||g||   = {reduced_grad_norm:.6e}")
            print(f"    full ||g||      = {full_grad_norm:.6e}")
            print(f"    opt threshold   = {opt_thresh:.6e}")
            print(f"    iterations      = {getattr(res, 'nit', None)}")

        return x_updated, info, res

    def _batched_optimizer_driver(self):
        """Run opt-in PH/GEN batched optimization over the current ansatz."""
        self._validate_batched_optimization_options()
        batch_indices = self._build_batched_opt_indices()
        schedule = self._build_batched_opt_schedule(batch_indices)
        x_full = np.asarray(self._tamps, dtype=float).copy()
        self._batched_opt_history = []

        if (
            not self._batched_opt_reset_optimizer_state_between_batches
            and self._batched_opt_verbose
        ):
            print(
                "Warning: optimizer-state reuse across batches is not implemented; "
                "each batch starts with fresh optimizer state."
            )

        if self._batched_opt_verbose:
            print(f"\n==> Begin batched optimization: {self._batched_opt_type}")
            print(f"    cycles: {self._batched_opt_cycles}")
            print(f"    batches: {' -> '.join(item['label'] for item in schedule)}")
            print(f"    batch opt threshold: {self._batch_opt_thresh:.6e}")
            print(f"    final ALL threshold: {self._opt_thresh:.6e}")
            print(
                f"    PH params: {len(batch_indices['PH'])}; "
                f"GEN params: {len(batch_indices['GEN'])}; "
                f"ALL params: {len(batch_indices['ALL'])}"
            )

        total_nfev = 0
        total_njev = 0
        total_nit = 0
        last_res = None
        for batch_number, batch in enumerate(schedule, start=1):
            label = batch["label"]
            if batch["kind"] == "ALL":
                maxiter = (
                    self._opt_maxiter
                    if self._batched_opt_final_maxiter is None
                    else int(self._batched_opt_final_maxiter)
                )
                opt_thresh = self._opt_thresh
            else:
                maxiter = (
                    self._opt_maxiter
                    if self._batched_opt_batch_maxiter is None
                    else int(self._batched_opt_batch_maxiter)
                )
                opt_thresh = self._batch_opt_thresh
            x_full, info, last_res = self._optimize_parameter_subset(
                x_full,
                batch["indices"],
                maxiter=maxiter,
                label=label,
                batch_number=batch_number,
                total_batches=len(schedule),
                opt_thresh=opt_thresh,
            )
            self._batched_opt_history.append(info)
            total_nfev += int(info["nfev"] or 0)
            total_njev += int(info["njev"] or 0)
            total_nit += int(info["iterations"] or 0)

        final_energy = float(np.real(self.energy_feval(x_full)))
        final_grad = np.asarray(self.gradient_ary_feval(x_full), dtype=float)
        final_grad_norm = float(np.linalg.norm(final_grad))
        success = (
            schedule[-1]["kind"] == "ALL"
            and final_grad_norm < self._opt_thresh
        )
        message = (
            "Batched optimization completed with final ALL convergence."
            if success
            else "Batched optimization completed; final full gradient may not be converged."
        )

        res = OptimizeResult(
            x=np.asarray(x_full, dtype=float),
            fun=final_energy,
            jac=final_grad,
            success=success,
            message=message,
            nit=total_nit,
            nfev=total_nfev,
            njev=total_njev,
            grad_norm=final_grad_norm,
            batched_opt_type=self._batched_opt_type,
            batched_opt_schedule=[item["label"] for item in schedule],
            batched_opt_history=list(self._batched_opt_history),
            last_batch_result=last_res,
        )

        self._tamps = list(x_full)
        self._Egs = final_energy
        self._curr_energy = final_energy
        self._curr_grad_norm = final_grad_norm
        self._final_result = res
        self._n_classical_params = len(self._tamps)

        if self._batched_opt_verbose:
            print("\nBatched optimization summary:")
            print(f"    total batches:        {len(schedule)}")
            print(f"    total energy evals:   {total_nfev}")
            print(f"    total gradient evals: {total_njev}")
            print(f"    final energy:         {final_energy:+12.10f}")
            print(f"    final full ||g||:     {final_grad_norm:.6e}")

        return res

    def solve(self):
        optimizer_name = self._optimizer.lower()
        if getattr(self, "_batched_opt_type", "none") != "none":
            if optimizer_name == "jacobi":
                raise ValueError(
                    "batched_opt_type is only supported for energy/gradient "
                    "optimizers, not Jacobi/residual optimizers."
                )
            return self._batched_optimizer_driver()
        if optimizer_name == "jacobi":
            self.build_orb_energies()
            return self.jacobi_solver()
        elif optimizer_name == "lbfgs_qf":
            return self.lbfgs_qf_solve()
        elif optimizer_name == "bfgs_qf":
            return self.bfgs_qf_solve()
        else:
            return self.scipy_solve()

    def scipy_solve(self):
        # Construct arguments to hand to the minimizer.
        opts = {}

        # Options common to all minimization algorithms
        opts['disp'] = True
        opts['maxiter'] = self._opt_maxiter

        # Optimizer-specific options
        if self._optimizer in ['BFGS', 'CG', 'L-BFGS-B', 'TNC', 'trust-constr']:
            opts['gtol'] = self._opt_thresh
        if self._optimizer == 'Nelder-Mead':
            opts['fatol'] = self._opt_ftol
        if self._optimizer in ['Powell', 'L-BFGS-B', 'TNC', 'SLSQP']:
            opts['ftol'] = self._opt_ftol
        if self._optimizer == 'COBYLA':
            opts['tol'] = self._opt_ftol
        if self._optimizer in ['L-BFGS-B', 'TNC']:
            opts['maxfun']  = self._opt_maxiter

        x0 = copy.deepcopy(self._tamps)
        init_gues_energy = self.energy_feval(x0)
        self._prev_energy = init_gues_energy

        if self._use_analytic_grad:
            print('  \n--> Begin opt with analytic gradient:')
            print(f" Initial guess energy:              {init_gues_energy:+12.10f}")
            res =  minimize(self.energy_feval, x0,
                                    method=self._optimizer,
                                    jac=self.gradient_ary_feval,
                                    options=opts,
                                    callback=self.report_iteration)

            # account for paulit term measurement for gradient evaluations
            # for m in range(len(self._tamps)):
            #     self._n_pauli_trm_measures += self._Nm[m] * self._Nl * res.njev
            if(self._optimizer not in ['Powell', 'Nelder-Mead']):
                for tmu in res.x:
                    if(np.abs(tmu) > 1.0e-12):
                        self._n_pauli_trm_measures += int(2 * self._Nl * res.njev)

            self._n_pauli_trm_measures += int(self._Nl * res.nfev)


        else:
            print('  \n--> Begin opt with grad estimated using first-differences:')
            print(f" Initial guess energy:              {init_gues_energy:+12.10f}")
            res =  minimize(self.energy_feval, x0,
                                    method=self._optimizer,
                                    options=opts,
                                    callback=self.report_iteration)

            # account for pauli term measurement for energy evaluations
            self._n_pauli_trm_measures += self._Nl * res.nfev

        if(res.success):
            print('  => Minimization successful!')
        else:
            print('  => WARNING: minimization result may not be tightly converged.')
        print(f'  => Minimum Energy: {res.fun:+12.10f}')
        self._Egs = res.fun
        if(self._optimizer == 'POWELL'):
            print(type(res.fun))
            print(res.fun)
            self._Egs = res.fun[()]
        self._final_result = res
        self._tamps = list(res.x)

        self._n_classical_params = len(self._tamps)

        self._n_cnot = optimizer._cnot_count_for_current_ansatz(self)


    def initialize_ansatz(self):
        """Adds all operators in the pool to the list of operators in the circuit,
        with zero amplitudes by default, or MP2 doubles if requested.
        """
        self._tops = list(range(len(self._pool_obj)))
        self._tamps = [0.0] * len(self._pool_obj)
        self._mp2_init_records = []
        self._mp2_init_nonzero = 0
        self._mp2_init_skipped_small_denom = 0
        self._mp2_init_duplicate_counts = {}

        if getattr(self, "_init_amps", "zero") == "mp2":
            self._initialize_mp2_amplitudes()

    def _spin_orbital_energies_for_mp2_init(self):
        """Return active-space spin-orbital energies matching self._ref."""
        if not hasattr(self._sys, "hf_orbital_energies"):
            raise ValueError(
                'init_amps="mp2" requires system.hf_orbital_energies.'
            )

        # Molecule adapters use interleaved spin orbitals, 2*p = alpha and
        # 2*p + 1 = beta, in ascending spatial-orbital-energy order. The same
        # convention is used by the SQOperator pools and by self._ref.
        spatial_eps = list(self._sys.hf_orbital_energies)
        if 2 * len(spatial_eps) != len(self._ref):
            frozen_core = int(getattr(self._sys, "frozen_core", 0))
            frozen_virtual = int(getattr(self._sys, "frozen_virtual", 0))
            end = len(spatial_eps) - frozen_virtual if frozen_virtual else len(spatial_eps)
            spatial_eps = spatial_eps[frozen_core:end]

        spin_eps = []
        for eps in spatial_eps:
            spin_eps.extend([float(eps), float(eps)])

        if len(spin_eps) != len(self._ref):
            raise ValueError(
                'init_amps="mp2" could not align orbital energies with the '
                f'reference: got {len(spin_eps)} spin energies for '
                f'{len(self._ref)} spin orbitals.'
            )

        return spin_eps

    @staticmethod
    def _apply_sq_term_to_det(creators, annihilators, occupation):
        """Apply one SQ term to a determinant, returning phase and new occupation.

        SQOperator stores a term as creators followed by annihilators. The
        rightmost operator acts first, so application to a ket proceeds through
        reversed annihilators and then reversed creators.
        """
        occ = list(occupation)
        phase = 1.0

        for idx in reversed([int(i) for i in annihilators]):
            if occ[idx] == 0:
                return 0.0, None
            phase *= -1.0 if sum(occ[:idx]) % 2 else 1.0
            occ[idx] = 0

        for idx in reversed([int(i) for i in creators]):
            if occ[idx] == 1:
                return 0.0, None
            phase *= -1.0 if sum(occ[:idx]) % 2 else 1.0
            occ[idx] = 1

        return phase, occ

    def _sq_operator_det_coupling(self, sq_op, ref_occ, target_occ):
        """Return <target|sq_op|ref> using the SQOperator term convention."""
        coupling = 0.0 + 0.0j
        target_tuple = tuple(target_occ)

        for coeff, creators, annihilators in sq_op.terms():
            phase, new_occ = self._apply_sq_term_to_det(
                creators,
                annihilators,
                ref_occ,
            )
            if new_occ is not None and tuple(new_occ) == target_tuple:
                coupling += complex(coeff) * phase

        return coupling

    def _pool_operator_particle_hole_double(self, sq_op):
        """Return particle-hole double metadata, or None if ambiguous.

        This intentionally accepts only a single determinant double excitation
        relative to the HF/reference occupation. Generalized terms, singles, and
        spin-adapted combinations that create more than one excited determinant
        are left at zero in this first simple implementation.
        """
        ref_occ = list(self._ref)
        target_to_excitation = {}

        for coeff, creators, annihilators in sq_op.terms():
            creators = [int(i) for i in creators]
            annihilators = [int(i) for i in annihilators]

            if len(creators) != 2 or len(annihilators) != 2:
                return None
            if any(idx < 0 or idx >= len(ref_occ) for idx in creators + annihilators):
                return None

            is_excitation = (
                all(ref_occ[p] == 0 for p in creators)
                and all(ref_occ[h] == 1 for h in annihilators)
            )
            is_deexcitation = (
                all(ref_occ[p] == 1 for p in creators)
                and all(ref_occ[h] == 0 for h in annihilators)
            )

            if is_excitation:
                phase, target_occ = self._apply_sq_term_to_det(
                    creators,
                    annihilators,
                    ref_occ,
                )
                if target_occ is None or phase == 0.0:
                    return None
                target_to_excitation[tuple(target_occ)] = {
                    "creators": creators,
                    "annihilators": annihilators,
                }
            elif not is_deexcitation:
                return None

        if len(target_to_excitation) != 1:
            return None

        target_occ, excitation = next(iter(target_to_excitation.items()))
        excitation["target_occ"] = list(target_occ)
        return excitation

    def _classify_pool_operator_for_ordering(self, sq_op):
        """Classify a pool operator for product-ordering heuristics.

        The ordering options change only the factorized UCC product order:
        the sorted `_tops` list is the order in which the existing ansatz
        appends/applies operators. Pool membership and SQOperator definitions
        are left untouched.
        """
        clean_ph = self._is_clean_particle_hole_excitation(sq_op)
        rank = self._get_excitation_rank(sq_op)

        classification = {
            "rank": rank,
            "is_clean_particle_hole": clean_ph is not None,
            "particles": [],
            "holes": [],
            "particle_spatial": [],
            "hole_spatial": [],
            "shell_score": 10**9,
        }

        if clean_ph is None:
            return classification

        particles = clean_ph["particles"]
        holes = clean_ph["holes"]
        classification.update(
            {
                "rank": clean_ph["rank"],
                "particles": particles,
                "holes": holes,
                "particle_spatial": [p // 2 for p in particles],
                "hole_spatial": [h // 2 for h in holes],
                "shell_score": self._get_shell_score(clean_ph),
            }
        )
        return classification

    def _is_clean_particle_hole_excitation(self, sq_op):
        """Return oriented p-h excitation metadata, or None if not clean.

        UCC generators are often anti-Hermitian pairs, e.g.
        excitation - deexcitation. Each SQ term is oriented relative to the HF
        occupation before comparison, so either side of the pair can appear
        first in the stored SQOperator.
        """
        ref_occ = list(self._ref)
        rank = None
        excitation_patterns = {}

        for coeff, creators, annihilators in sq_op.terms():
            creators = [int(i) for i in creators]
            annihilators = [int(i) for i in annihilators]

            if len(creators) != len(annihilators) or len(creators) == 0:
                return None
            if rank is None:
                rank = len(creators)
            elif rank != len(creators):
                return None
            if any(idx < 0 or idx >= len(ref_occ) for idx in creators + annihilators):
                return None

            is_excitation = (
                all(ref_occ[p] == 0 for p in creators)
                and all(ref_occ[h] == 1 for h in annihilators)
            )
            is_deexcitation = (
                all(ref_occ[h] == 1 for h in creators)
                and all(ref_occ[p] == 0 for p in annihilators)
            )

            if is_excitation:
                particles = creators
                holes = annihilators
            elif is_deexcitation:
                particles = annihilators
                holes = creators
            else:
                return None

            key = (tuple(sorted(particles)), tuple(sorted(holes)))
            excitation_patterns[key] = {
                "rank": rank,
                "particles": list(key[0]),
                "holes": list(key[1]),
            }

        if len(excitation_patterns) != 1:
            return None

        return next(iter(excitation_patterns.values()))

    def _is_particle_hole_double(self, sq_op):
        """Return True when an SQOperator is a clean p-h double excitation."""
        clean_ph = self._is_clean_particle_hole_excitation(sq_op)
        return clean_ph is not None and clean_ph["rank"] == 2

    def _get_excitation_rank(self, sq_op):
        """Return the common body rank of an SQOperator, or None if mixed."""
        rank = None
        for coeff, creators, annihilators in sq_op.terms():
            if len(creators) != len(annihilators):
                return None
            if rank is None:
                rank = len(creators)
            elif rank != len(creators):
                return None
        return rank

    def _get_shell_score(self, excitation):
        """Return smaller scores for excitations closer to HOMO/LUMO.

        This simple active-space heuristic sums distances from the HOMO among
        annihilated occupied orbitals and from the LUMO among created virtual
        orbitals. It works for clean p-h excitations of any rank represented by
        the pool, including triples through hextuples.
        """
        if excitation is None:
            return 10**9

        nocc = int(sum(self._ref) // 2)
        if nocc <= 0:
            return 10**9

        shell_score = 0
        for hole in excitation["holes"]:
            hole_spatial = int(hole) // 2
            if hole_spatial < 0 or hole_spatial >= nocc:
                return 10**9
            shell_score += (nocc - 1) - hole_spatial

        for particle in excitation["particles"]:
            particle_spatial = int(particle) // 2
            if particle_spatial < nocc:
                return 10**9
            shell_score += particle_spatial - nocc

        return int(shell_score)

    def _mp2_amplitude_info_for_pool_operator(self, sq_op, spin_eps=None, ref_occ=None):
        """Return MP2 amplitude metadata for a clean p-h double, or None."""
        if not hasattr(self._sys, "sq_hamiltonian"):
            raise ValueError('primary_pool_order="mp2_amps" requires system.sq_hamiltonian.')

        excitation = self._pool_operator_particle_hole_double(sq_op)
        if excitation is None:
            return None

        if spin_eps is None:
            spin_eps = self._spin_orbital_energies_for_mp2_init()
        if ref_occ is None:
            ref_occ = list(self._ref)

        creators = excitation["creators"]
        annihilators = excitation["annihilators"]
        target_occ = excitation["target_occ"]

        denom = (
            sum(spin_eps[h] for h in annihilators)
            - sum(spin_eps[p] for p in creators)
        )
        if abs(denom) < 1.0e-12:
            return None

        h_coupling = self._sq_operator_det_coupling(
            self._sys.sq_hamiltonian,
            ref_occ,
            target_occ,
        )
        k_coupling = self._sq_operator_det_coupling(
            sq_op,
            ref_occ,
            target_occ,
        )
        if abs(k_coupling) < 1.0e-14:
            return None

        # This is the same convention as MP2 initial amplitudes: use the
        # already-built SQ Hamiltonian coupling to inherit QForte's integral
        # ordering, signs, and 1/2 factors, then divide by the generator
        # coupling so theta_mu is the first-order determinant coefficient.
        amp_complex = (h_coupling / denom) / k_coupling
        if abs(np.imag(amp_complex)) > 1.0e-10:
            return None

        amp = float(np.real(amp_complex))
        if not np.isfinite(amp):
            return None

        return {
            "amplitude": amp,
            "denominator": denom,
            "h_coupling": h_coupling,
            "k_coupling": k_coupling,
            "creators": creators,
            "annihilators": annihilators,
            "target_occ": target_occ,
        }

    def _get_mp2_importance_score(self, sq_op, spin_eps=None, ref_occ=None):
        """Return |MP2 amplitude| for clean p-h doubles; otherwise zero."""
        amp_info = self._mp2_amplitude_info_for_pool_operator(
            sq_op,
            spin_eps=spin_eps,
            ref_occ=ref_occ,
        )
        if amp_info is None:
            return 0.0
        return abs(amp_info["amplitude"])

    def _pool_ordering_gradient_scores(self):
        """Return |initial analytical gradient| without changing eval counts."""
        saved_res_vec_evals = getattr(self, "_res_vec_evals", None)
        saved_res_m_evals = getattr(self, "_res_m_evals", None)
        saved_curr_grad_norm = getattr(self, "_curr_grad_norm", None)

        grad = np.asarray(self.gradient_ary_feval(self._tamps), dtype=float)

        if saved_res_vec_evals is not None:
            self._res_vec_evals = saved_res_vec_evals
        if saved_res_m_evals is not None:
            self._res_m_evals = saved_res_m_evals
        if saved_curr_grad_norm is not None:
            self._curr_grad_norm = saved_curr_grad_norm

        if grad.shape != (len(self._tops),):
            raise ValueError(
                "Initial gradient size does not match the current ansatz size "
                "for primary_pool_order=\"gradients\"."
            )
        return np.abs(grad)

    def _remap_mp2_init_records_after_ordering(self, permutation):
        """Update stored MP2 record parameter positions after `_tops` sorting."""
        if not getattr(self, "_mp2_init_records", None):
            return

        old_to_new = {
            int(old_position): int(new_position)
            for new_position, old_position in enumerate(permutation)
        }
        new_records = []
        for record in self._mp2_init_records:
            old_mu = int(record["mu"])
            if old_mu not in old_to_new:
                continue
            new_record = dict(record)
            new_record["mu"] = old_to_new[old_mu]
            new_records.append(new_record)

        self._mp2_init_records = sorted(new_records, key=lambda rec: rec["mu"])

    def apply_pool_ordering(self):
        """Stably reorder `_tops`/`_tamps` according to user pool-order options.

        The SQOpPool itself is not rebuilt or mutated. These options only change
        product-form order in the factorized/trotterized UCC ansatz, which can
        change finite-product behavior even though the variational span is the
        same in the infinitesimal limit.
        """
        primary_order = getattr(self, "_primary_pool_order", "none")
        secondary_order = getattr(self, "_secondary_pool_order", "lexical")
        general_order = getattr(self, "_general_ex_pool_order", "default")

        if len(self._tops) != len(self._tamps):
            raise ValueError("_tops and _tamps must have matching lengths before pool ordering.")

        old_tops = list(self._tops)
        old_tamps = list(self._tamps)
        self._pool_ordering_records = []

        default_ordering = (
            primary_order == "none"
            and secondary_order == "lexical"
            and general_order == "default"
        )
        if default_ordering:
            if old_tops != list(range(len(self._pool_obj))):
                raise ValueError("Default pool ordering expected lexical pool indices.")
            return

        fill_applied_general_order = getattr(
            self, "_general_ex_pool_order_applied_in_fill", False
        )
        if (
            fill_applied_general_order
            and primary_order == "none"
            and secondary_order == "lexical"
        ):
            # For generalized pools, particle_hole_first is applied during
            # C++ pool construction so k-UpCCGSD keeps the layered ordering
            # PH(k0), GEN(k0), PH(k1), GEN(k1), ... .  A Python-level global
            # regrouping would flatten all particle-hole terms ahead of all
            # generalized terms, which is not the intended product order.
            return

        gradient_scores = None
        if primary_order == "gradients":
            gradient_scores = self._pool_ordering_gradient_scores()

        spin_eps = None
        ref_occ = None
        if primary_order == "mp2_amps":
            spin_eps = self._spin_orbital_energies_for_mp2_init()
            ref_occ = list(self._ref)

        records = []
        for old_position, top in enumerate(old_tops):
            sq_op = self._pool_obj[top][1]
            classification = self._classify_pool_operator_for_ordering(sq_op)

            if primary_order == "none":
                primary_score = 0.0
            elif primary_order == "mp2_amps":
                primary_score = self._get_mp2_importance_score(
                    sq_op,
                    spin_eps=spin_eps,
                    ref_occ=ref_occ,
                )
            elif primary_order == "gradients":
                primary_score = float(abs(gradient_scores[old_position]))
            else:
                raise ValueError(f"Unknown primary_pool_order: {primary_order}")

            if not np.isfinite(primary_score):
                primary_score = 0.0

            if secondary_order == "shell":
                secondary_score = classification["shell_score"]
            elif secondary_order == "lexical":
                secondary_score = 0
            else:
                raise ValueError(f"Unknown secondary_pool_order: {secondary_order}")

            if general_order == "particle_hole_first" and not fill_applied_general_order:
                group_key = 0 if classification["is_clean_particle_hole"] else 1
            elif general_order == "default":
                group_key = 0
            elif general_order == "particle_hole_first" and fill_applied_general_order:
                group_key = 0
            else:
                raise ValueError(f"Unknown general_ex_pool_order: {general_order}")

            records.append(
                {
                    "old_position": old_position,
                    "top": top,
                    "primary_score": float(primary_score),
                    "secondary_score": int(secondary_score),
                    "group_key": int(group_key),
                    "is_clean_particle_hole": classification["is_clean_particle_hole"],
                    "rank": classification["rank"],
                }
            )

        if fill_applied_general_order:
            # The C++ generalized-pool fill has already created the desired
            # block structure.  For k-UpCCGSD this is:
            #   PH(k0), GEN(k0), PH(k1), GEN(k1), ...
            # Keep those contiguous blocks fixed and apply primary/secondary
            # heuristics only inside each block.
            current_block = -1
            previous_is_ph = None
            for rec in records:
                is_ph = rec["is_clean_particle_hole"]
                if previous_is_ph is None or is_ph != previous_is_ph:
                    current_block += 1
                    previous_is_ph = is_ph
                rec["group_key"] = current_block

        if secondary_order == "shell":
            sort_key = lambda rec: (
                rec["group_key"],
                -rec["primary_score"],
                rec["secondary_score"],
                rec["old_position"],
            )
        else:
            sort_key = lambda rec: (
                rec["group_key"],
                -rec["primary_score"],
                rec["old_position"],
            )

        sorted_records = sorted(records, key=sort_key)
        permutation = [rec["old_position"] for rec in sorted_records]
        self._tops = [old_tops[pos] for pos in permutation]
        self._tamps = [old_tamps[pos] for pos in permutation]

        if len(self._tops) != len(self._tamps):
            raise ValueError("_tops and _tamps length mismatch after pool ordering.")

        for new_position, rec in enumerate(sorted_records):
            rec["new_position"] = new_position
        self._pool_ordering_records = sorted_records
        self._remap_mp2_init_records_after_ordering(permutation)

        if self._verbose:
            print("\nPool ordering applied:")
            print(f"  primary_pool_order:    {primary_order}")
            print(f"  secondary_pool_order:  {secondary_order}")
            print(f"  general_ex_pool_order: {general_order}")
            print(f"  first ordered tops:    {self._tops[:10]}")

    def _initialize_mp2_amplitudes(self):
        """Populate MP2 amplitudes for unambiguous particle-hole doubles only.

        The molecule adapters build the second-quantized Hamiltonian directly
        from their integral tensors after applying the internal spin-orbital
        ordering, 1/2 factors, and Psi4/PySCF/OpenFermion reshuffles. To avoid
        duplicating that convention-sensitive logic here, the numerator is the
        Hamiltonian coupling <Phi_ij^ab|H|Phi_0> computed from the stored
        SQOperator Hamiltonian. The denominator is the usual active spin-orbital
        MP2 denominator eps_i + eps_j - eps_a - eps_b.
        """
        if not hasattr(self._sys, "sq_hamiltonian"):
            raise ValueError('init_amps="mp2" requires system.sq_hamiltonian.')

        spin_eps = self._spin_orbital_energies_for_mp2_init()
        ref_occ = list(self._ref)

        records = []
        skipped_small_denom = 0
        amp_candidates = []

        for mu, top in enumerate(self._tops):
            sq_op = self._pool_obj[top][1]
            amp_info = self._mp2_amplitude_info_for_pool_operator(
                sq_op,
                spin_eps=spin_eps,
                ref_occ=ref_occ,
            )
            if amp_info is None:
                excitation = self._pool_operator_particle_hole_double(sq_op)
                if excitation is not None:
                    denom = (
                        sum(spin_eps[h] for h in excitation["annihilators"])
                        - sum(spin_eps[p] for p in excitation["creators"])
                    )
                    if abs(denom) < 1.0e-12:
                        skipped_small_denom += 1
                continue

            # Group repeated k-UpCCGSD layers by the actual excited determinant.
            # The total first-order MP2 coefficient for this p-h double is the
            # same regardless of how many identical k blocks are present, so an
            # excitation repeated k times receives t_ij^ab / k in each block.
            amp_key = tuple(int(i) for i in amp_info["target_occ"])
            amp_candidates.append(
                {
                    "mu": mu,
                    "top": top,
                    "amp_key": amp_key,
                    "amp_info": amp_info,
                }
            )

        amp_key_counts = {}
        for candidate in amp_candidates:
            amp_key = candidate["amp_key"]
            amp_key_counts[amp_key] = amp_key_counts.get(amp_key, 0) + 1

        self._mp2_init_duplicate_counts = dict(amp_key_counts)

        for candidate in amp_candidates:
            mu = candidate["mu"]
            top = candidate["top"]
            amp_info = candidate["amp_info"]
            denom = amp_info["denominator"]
            duplicate_count = amp_key_counts[candidate["amp_key"]]
            amp = amp_info["amplitude"] / duplicate_count

            self._tamps[mu] = amp
            if abs(amp) > 1.0e-12:
                records.append(
                    {
                        "mu": mu,
                        "top": top,
                        "amplitude": amp,
                        "undistributed_amplitude": amp_info["amplitude"],
                        "duplicate_count": duplicate_count,
                        "denominator": denom,
                        "h_coupling": amp_info["h_coupling"],
                        "k_coupling": amp_info["k_coupling"],
                        "creators": amp_info["creators"],
                        "annihilators": amp_info["annihilators"],
                        "target_occ": amp_info["target_occ"],
                    }
                )

        self._mp2_init_records = records
        self._mp2_init_nonzero = len(records)
        self._mp2_init_skipped_small_denom = skipped_small_denom

        if self._verbose:
            print(
                "\nMP2 initial amplitudes: "
                f"{self._mp2_init_nonzero} nonzero particle-hole doubles "
                f"({skipped_small_denom} skipped for small denominators)."
            )
            repeated = [
                count for count in amp_key_counts.values()
                if count > 1
            ]
            if repeated:
                print(
                    "  Repeated p-h doubles split across k blocks; "
                    f"max duplicate count = {max(repeated)}."
                )

    def initialize_gpu_pool(self):
        """Initialize reusable GPU pool and computers for optimization.
        This should be called after initialize_ansatz() and before solve().
        Precomputes index arrays and creates persistent GPU computers to avoid:
        1. Pool recreation overhead (major optimization via precomputation)
        2. GPU memory allocation/deallocation overhead each iteration
        """
        if self._computer_type != 'fci_gpu':
            return
        
        # Create reusable GPU pool with initial coefficients (zeros)
        self._reusable_pool_gpu = qforte.SQOpPoolGPU(data_type=self.data_type)
        
        # Add all operators from pool with initial zero coefficients
        for tamp, top in zip(self._tamps, self._tops):
            self._reusable_pool_gpu.add(tamp, self._pool_obj[top][1])
        
        # Create reusable GPU computers for gradient evaluation
        # qc_psi: builds evolved state
        # qc_sig: builds H|psi> for gradient computation
        self._reusable_qc_psi = qforte.FCIComputerGPU(
            self._nel, 
            self._2_spin, 
            self._norb,
            on_gpu=True,
            data_type=self.data_type,
            gpu_only=True)
        
        self._reusable_qc_psi.hartree_fock_gpu()
        
        self._reusable_qc_sig = qforte.FCIComputerGPU(
            self._nel, 
            self._2_spin, 
            self._norb,
            on_gpu=True,
            data_type=self.data_type,
            gpu_only=True)
        
        # Precompute index arrays for fast evolution
        # This sets device_vecs_populated_ = true and precomputes all index mappings
        self._reusable_qc_psi.populate_index_arrays_for_pool_evo(self._reusable_pool_gpu)
        self._reusable_qc_sig.populate_index_arrays_for_pool_evo(self._reusable_pool_gpu)
        
        if self._verbose:
            print('\n==> GPU optimization infrastructure initialized')
            print(f'    Pool size: {len(self._tops)} operators')
            print(f'    Index arrays precomputed for fast evolution')
            print(f'    Reusable GPU computers created (avoids allocation overhead)')

    # TODO: change to get_num_pt_evals
    def get_num_ham_measurements(self):
        """Returns the total number of times the energy was evaluated via
        measurement of the Hamiltonian.
        """
        try:
            self._n_ham_measurements = self._final_result.nfev
            return self._n_ham_measurements
        except AttributeError:
            # TODO: Determine the number of Hamiltonian measurements
            return "Not Yet Implemented"

    # TODO: depricate this function
    def get_num_commut_measurements(self):
        # if self._use_analytic_grad:
        #     self._n_commut_measurements = self._final_result.njev * (len(self._pool_obj))
        #     return self._n_commut_measurements
        # else:
        #     return 0
        return 0

UCCNVQE.jacobi_solver = optimizer.jacobi_solver
UCCNVQE.lbfgs_qf_solve = optimizer.lbfgs_qf_solve
UCCNVQE.bfgs_qf_solve = optimizer.bfgs_qf_solve
UCCNVQE.lbfgs_solver = optimizer.lbfgs_solver
UCCNVQE._lbfgs_qf_hess_vec_fd = optimizer._lbfgs_qf_hess_vec_fd
UCCNVQE._lbfgs_qf_newton_cg_direction = optimizer._lbfgs_qf_newton_cg_direction
UCCNVQE._lbfgs_qf_try_newton_cg_step = optimizer._lbfgs_qf_try_newton_cg_step
UCCNVQE._lbfgs_qf_select_target_block = optimizer._lbfgs_qf_select_target_block
UCCNVQE._lbfgs_qf_build_target_hessian_block = optimizer._lbfgs_qf_build_target_hessian_block
UCCNVQE._lbfgs_qf_target_block_step = optimizer._lbfgs_qf_target_block_step
UCCNVQE._lbfgs_qf_try_target_block_step = optimizer._lbfgs_qf_try_target_block_step
UCCNVQE._lbfgs_qf_should_escape_negative_curvature = optimizer._lbfgs_qf_should_escape_negative_curvature
UCCNVQE._lbfgs_qf_try_negative_curvature_escape = optimizer._lbfgs_qf_try_negative_curvature_escape
UCCNVQE.construct_moment_space = moment_energy_corrections.construct_moment_space
UCCNVQE.compute_moment_energies = moment_energy_corrections.compute_moment_energies
