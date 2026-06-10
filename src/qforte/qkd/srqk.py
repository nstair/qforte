"""
SRQK classes
=================================================
Classes for calculating reference states for quantum
mechanical systems for the single referece selected
quantum Krylov algorithm.
"""

import qforte
from qforte.abc.qsdabc import QSD
from qforte.helper.printing import matprint

from qforte.maths.eigsolve import canonical_geig_solve

from qforte.utils.state_prep import *
from qforte.utils.trotterization import (trotterize,
                                         trotterize_w_cRz)

import numpy as np
import time
import math

class SRQK(QSD):
    """A quantum subspace diagonalization algorithm that generates the many-body
    basis from different durations of real time evolution:

    .. math::
        | \Psi_n \\rangle = e^{-i n \Delta t \hat{H}} | \Phi_0 \\rangle

    In practice Trotterization is used to approximate the time evolution operator.

    Attributes
    ----------

    _dt : float
        The time step used in the time evolution unitaries.

    _nstates : int
        The total number of basis states (s + 1).

    _s : int
        The greatest m to use in unitaries, equal to the number of time evolutions.


    """
    def run(self,
            s=3,
            dt=0.5,
            target_root=0,
            use_exact_evolution=False,
            diagonalize_each_step=True,
            low_memory_mat_formation=False,
            qk_time_grid="linear",
            qk_time_power=2.0,
            qk_tmax_type="manual",
            qk_tmax=None,
            qk_trotter_control="fixed",
            qk_target_trotter_error=1.0e-3,
            qk_trotter_norm_type="integral_l1",
            qk_trotter_include_scalar=False,
            qk_trotter_two_body_factor=0.5,
            qk_trotter_bound_scale=1.0,
            qk_variance_beta=4.0,
            qk_gap_alpha=4.0,
            qk_gap=None,
            gev_stabilization_thresh=None,
            use_legacy_fock_srqk_protocal=False,
            ):

        if not isinstance(s, (int, np.integer)):
            raise ValueError("SRQK run option s must be an integer >= 1.")
        self._s = int(s)
        self._nstates = self._s + 1

        self._target_root = target_root
        self._use_exact_evolution = use_exact_evolution
        self._diagonalize_each_step = diagonalize_each_step
        self._low_memory_mat_formation = low_memory_mat_formation

        self._qk_time_grid = qk_time_grid
        self._qk_time_power = qk_time_power
        self._qk_tmax_type = qk_tmax_type
        self._qk_tmax_override = qk_tmax
        self._qk_trotter_control = qk_trotter_control

        self._qk_target_trotter_error = qk_target_trotter_error
        self._qk_trotter_norm_type = qk_trotter_norm_type
        self._qk_trotter_include_scalar = qk_trotter_include_scalar
        self._qk_trotter_two_body_factor = qk_trotter_two_body_factor
        self._qk_trotter_bound_scale = qk_trotter_bound_scale

        self._qk_variance_beta = qk_variance_beta
        self._qk_gap_alpha = qk_gap_alpha
        self._qk_gap = qk_gap

        self._gev_stabilization_thresh = gev_stabilization_thresh
        self._use_legacy_fock_srqk_protocal = bool(use_legacy_fock_srqk_protocal)
        if (
            self._use_legacy_fock_srqk_protocal
            and self._computer_type != 'fock'
        ):
            raise ValueError(
                "use_legacy_fock_srqk_protocal=True is only valid when "
                "computer_type='fock'."
            )

        if (dt == "lambda_inv"):
            print("\nSRQK option dt='lambda_inv' selected: using the inverse of the integral norm proxy for the Trotter error as the time step.")
            self._dt = 1.0 / self._compute_qk_trotter_lambda_int()
        else:
            self._dt = dt

        self._setup_qk_times_and_trotter_schedule()

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        # If GPU then create global QC & SQOP pool for precomutation path
        if (self._computer_type == 'fci_gpu'):
            self.initialize_gpu_global()

        self.common_run()

    def _reset_qk_time_grid_diagnostics(self):
        self._qk_tmax = None
        self._qk_time_points = None
        self._qk_time_increments = None
        self._qk_macro_dt_list = None
        self._qk_base_step_counts = None
        self._qk_increment_counts = None
        self._qk_effective_dt_list = None
        self._qk_effective_trotter_number_list = None
        self._qk_effective_micro_dt_list = None
        self._qk_micro_dt_target = None
        self._qk_macro_trotter_number_list = None

        self._qk_hf_energy = None
        self._qk_hf_h2 = None
        self._qk_hf_variance = None
        self._qk_hf_sigma_h = None
        self._qk_tmax_variance = None

        self._qk_trotter_lambda_int = None
        self._qk_trotter_micro_dt = None
        self._qk_trotter_step_strength = None
        self._qk_trotter_micro_dt_allowed = None
        self._qk_trotter_step_strength_allowed = None
        self._qk_trotter_base_proxy_error = None
        self._qk_trotter_proxy_error_rate = None
        self._qk_trotter_proxy_estimate = None
        self._qk_tmax_trotter = None
        self._qk_trotter_n_base_max = None

        self._qk_tmax_limiter = None
        self._qk_uses_legacy_uniform_grid = False
        self._qk_base_first_order_cnot = None

    def _validate_qk_time_grid_inputs(self):
        if not isinstance(self._s, (int, np.integer)):
            raise ValueError("SRQK run option s must be an integer >= 1.")
        if self._s < 1:
            raise ValueError("SRQK run option s must be >= 1.")

        self._dt = float(self._dt)
        if self._dt <= 0.0:
            raise ValueError("SRQK run option dt must be positive.")

        self._qk_time_grid = str(self._qk_time_grid).lower()
        allowed_grids = ("linear", "quadratic", "power")
        if self._qk_time_grid not in allowed_grids:
            raise ValueError(
                f"qk_time_grid must be one of {allowed_grids}; "
                f"got {self._qk_time_grid!r}."
            )

        self._qk_tmax_type = str(self._qk_tmax_type).lower()
        if self._qk_tmax_type == "trotter":
            raise ValueError(
                "qk_tmax_type='trotter' has been removed. Trotter error now "
                "controls per-interval trotter numbers through "
                "qk_trotter_control='auto', not Tmax."
            )
        allowed_tmax_types = ("manual", "variance", "auto", "egap")
        if self._qk_tmax_type not in allowed_tmax_types:
            raise ValueError(
                f"qk_tmax_type must be one of {allowed_tmax_types}; "
                f"got {self._qk_tmax_type!r}."
            )
        if self._qk_tmax_type == "egap":
            raise NotImplementedError(
                "qk_tmax_type='egap' is not implemented yet."
            )

        if self._qk_tmax_override is not None:
            self._qk_tmax_override = float(self._qk_tmax_override)
            if not np.isfinite(self._qk_tmax_override) or self._qk_tmax_override <= 0.0:
                raise ValueError("qk_tmax must be positive when provided.")

        self._qk_trotter_control = str(self._qk_trotter_control).lower()
        allowed_trotter_controls = ("fixed", "auto")
        if self._qk_trotter_control not in allowed_trotter_controls:
            raise ValueError(
                "qk_trotter_control must be one of "
                f"{allowed_trotter_controls}; got {self._qk_trotter_control!r}."
            )

        self._qk_time_power = float(self._qk_time_power)
        if self._qk_time_grid == "power" and self._qk_time_power <= 0.0:
            raise ValueError("qk_time_power must be positive for qk_time_grid='power'.")

        self._qk_target_trotter_error = float(self._qk_target_trotter_error)
        if self._qk_target_trotter_error <= 0.0:
            raise ValueError("qk_target_trotter_error must be positive.")

        self._qk_trotter_norm_type = str(self._qk_trotter_norm_type).lower()
        if self._qk_trotter_norm_type != "integral_l1":
            raise ValueError(
                "Only qk_trotter_norm_type='integral_l1' is currently supported."
            )

        self._qk_trotter_two_body_factor = float(self._qk_trotter_two_body_factor)
        if self._qk_trotter_two_body_factor < 0.0:
            raise ValueError("qk_trotter_two_body_factor must be nonnegative.")

        self._qk_trotter_bound_scale = float(self._qk_trotter_bound_scale)
        if self._qk_trotter_bound_scale <= 0.0:
            raise ValueError("qk_trotter_bound_scale must be positive.")

        self._qk_variance_beta = float(self._qk_variance_beta)
        if self._qk_variance_beta <= 0.0:
            raise ValueError("qk_variance_beta must be positive.")

        self._qk_gap_alpha = float(self._qk_gap_alpha)
        if self._qk_gap_alpha <= 0.0:
            raise ValueError("qk_gap_alpha must be positive.")

        if self._gev_stabilization_thresh is not None:
            self._gev_stabilization_thresh = float(self._gev_stabilization_thresh)
            if self._gev_stabilization_thresh < 0.0:
                raise ValueError("gev_stabilization_thresh must be nonnegative.")

        if self._trotter_number <= 0:
            raise ValueError("SRQK requires trotter_number to be positive.")
        if self._trotter_order < 0:
            raise ValueError("SRQK requires trotter_order to be nonnegative.")
        if self._qk_trotter_control == "auto" and self._trotter_order <= 0:
            raise ValueError(
                "qk_trotter_control='auto' requires a positive trotter_order."
            )

    def _qk_tensor_abs_sum(self, tensor):
        if isinstance(tensor, np.ndarray):
            return float(np.sum(np.abs(tensor)))
        if hasattr(tensor, "read_data"):
            data = np.asarray(tensor.read_data(), dtype=np.complex128)
            return float(np.sum(np.abs(data)))
        if hasattr(tensor, "data"):
            data = np.asarray(tensor.data(), dtype=np.complex128)
            return float(np.sum(np.abs(data)))
        raise ValueError("Could not read tensor data for qk_trotter_norm_type='integral_l1'.")

    def _qk_get_integral_tensor(self, attr_names):
        for obj in (self, self._sys):
            for attr_name in attr_names:
                if hasattr(obj, attr_name):
                    return getattr(obj, attr_name)
        return None

    def _compute_qk_hf_variance_tmax(self):
        qc = qforte.FCIComputer(self._nel, self._2_spin, self._norb)
        qc.hartree_fock()
        hf_state = qc.get_state_deep()

        if self._apply_ham_as_tensor:
            mo_oeis = self._qk_get_integral_tensor(("_mo_oeis", "mo_oeis"))
            mo_teis = self._qk_get_integral_tensor(("_mo_teis", "mo_teis"))
            mo_teis_einsum = self._qk_get_integral_tensor(
                ("_mo_teis_einsum", "mo_teis_einsum")
            )
            if mo_oeis is None or mo_teis is None or mo_teis_einsum is None:
                raise ValueError(
                    "Variance-based QK Tmax requires stored qforte MO integral "
                    "tensors when apply_ham_as_tensor=True."
                )
            qc.apply_tensor_spat_012bdy(
                self._zero_body_energy,
                mo_oeis,
                mo_teis,
                mo_teis_einsum,
                self._norb)
        else:
            qc.apply_sqop(self._sq_ham)

        h_hf_state = qc.get_state_deep()
        hf_energy = hf_state.vector_dot(h_hf_state)
        hf_h2 = h_hf_state.vector_dot(h_hf_state)
        variance = float(np.real(hf_h2) - np.real(hf_energy) ** 2)
        variance = max(variance, 0.0)
        sigma_h = float(np.sqrt(variance))

        if sigma_h <= 1.0e-14:
            raise ValueError(
                "Could not determine a variance-based QK Tmax because the "
                "reference-state Hamiltonian variance is too small."
            )

        self._qk_hf_energy = float(np.real(hf_energy))
        self._qk_hf_h2 = float(np.real(hf_h2))
        self._qk_hf_variance = variance
        self._qk_hf_sigma_h = sigma_h
        self._qk_tmax_variance = self._qk_variance_beta / sigma_h
        return self._qk_tmax_variance

    def _compute_qk_trotter_lambda_int(self):
        return self._compute_qk_trotter_lambda_int_from_herm_pairs()

    def _compute_qk_trotter_lambda_int_from_herm_pairs(self):
        mo_oeis = self._qk_get_integral_tensor(("_mo_oeis", "mo_oeis"))
        mo_teis = self._qk_get_integral_tensor(("_mo_teis", "mo_teis"))
        if mo_oeis is None or mo_teis is None:
            raise ValueError(
                "qk_trotter_control='auto' requires stored qforte MO integral "
                "tensors for the Hermitian-pair norm proxy."
            )

        norb = int(self._norb)
        lambda_int = 0.0
        h_vec = set()
        hd_vec = set()

        def add_hmu(coeff, creators, annihilators):
            nonlocal lambda_int

            coeff = complex(coeff)
            if abs(coeff) <= 0.0:
                return

            h = (tuple(sorted(creators)), tuple(sorted(annihilators)))
            hd = (h[1], h[0])
            if h in h_vec or h in hd_vec:
                return

            lambda_int += abs(coeff)
            h_vec.add(h)
            hd_vec.add(hd)

        for i in range(norb):
            ia = 2 * i
            ib = 2 * i + 1
            for j in range(norb):
                ja = 2 * j
                jb = 2 * j + 1

                h1 = mo_oeis.get([i, j])
                add_hmu(h1, [ia], [ja])
                add_hmu(h1, [ib], [jb])

                for k in range(norb):
                    ka = 2 * k
                    kb = 2 * k + 1
                    for l in range(norb):
                        la = 2 * l
                        lb = 2 * l + 1

                        h2 = 0.5 * mo_teis.get([i, l, k, j])

                        if ia != jb and kb != la:
                            add_hmu(h2, [ia, jb], [kb, la])  # abba
                        if ib != ja and ka != lb:
                            add_hmu(h2, [ib, ja], [ka, lb])  # baab
                        if ia != ja and ka != la:
                            add_hmu(h2, [ia, ja], [ka, la])  # aaaa
                        if ib != jb and kb != lb:
                            add_hmu(h2, [ib, jb], [kb, lb])  # bbbb

        if not np.isfinite(lambda_int) or lambda_int <= 0.0:
            raise ValueError(
                "Could not determine an automatic QK Trotter schedule because "
                "the Hermitian-pair norm proxy is zero or invalid."
            )

        self._qk_trotter_lambda_int = float(lambda_int)
        return self._qk_trotter_lambda_int

    def _compute_qk_trotter_tmax(self):
        raise ValueError(
            "qk_tmax_type='trotter' has been removed. Trotter error now "
            "controls per-interval trotter numbers through "
            "qk_trotter_control='auto', not Tmax."
        )

    def _qk_round_step_count(self, value):
        return int(round(float(value)))

    def _repair_qk_step_counts(self, step_counts, final_max_step):
        if final_max_step < self._s:
            raise ValueError(
                "The selected QK Tmax is too small to build a strictly increasing "
                "integer time grid for the requested s and dt. Reduce s, reduce dt, "
                "increase qk_target_trotter_error, or use a larger Tmax."
            )

        repaired = [0]
        for idx in range(1, self._s + 1):
            min_allowed = repaired[-1] + 1
            remaining = self._s - idx
            max_allowed = final_max_step - remaining
            candidate = int(step_counts[idx])
            candidate = max(candidate, min_allowed)
            candidate = min(candidate, max_allowed)
            if candidate < min_allowed:
                raise ValueError(
                    "Could not repair the rounded QK time grid into usable "
                    "integer base-step counts. Reduce s, reduce dt, increase "
                    "qk_target_trotter_error, or use a larger Tmax."
                )
            repaired.append(candidate)

        if repaired[-1] != final_max_step:
            repaired[-1] = final_max_step

        return np.asarray(repaired, dtype=int)

    def _build_qk_step_counts(self, tmax):
        if self._qk_tmax_type == "manual" and self._qk_time_grid == "linear":
            return np.arange(self._s + 1, dtype=int)

        step_counts = []
        for j in range(self._s + 1):
            x = float(j) / float(self._s)
            if self._qk_time_grid == "linear":
                time_point = tmax * x
            elif self._qk_time_grid == "quadratic":
                time_point = tmax * x ** 2
            elif self._qk_time_grid == "power":
                time_point = tmax * x ** self._qk_time_power
            else:
                raise ValueError(f"Unrecognized qk_time_grid {self._qk_time_grid!r}.")
            step_counts.append(self._qk_round_step_count(time_point / self._dt))

        step_counts[0] = 0
        if self._qk_tmax_type == "manual":
            final_max_step = self._s
        else:
            final_max_step = max(1, self._qk_round_step_count(tmax / self._dt))
        step_counts[-1] = final_max_step

        return self._repair_qk_step_counts(step_counts, final_max_step)

    def _build_qk_continuous_time_points(self, tmax):
        time_points = []
        for j in range(self._s + 1):
            x = float(j) / float(self._s)
            if self._qk_time_grid == "linear":
                time_point = tmax * x
            elif self._qk_time_grid == "quadratic":
                time_point = tmax * x ** 2
            elif self._qk_time_grid == "power":
                time_point = tmax * x ** self._qk_time_power
            else:
                raise ValueError(f"Unrecognized qk_time_grid {self._qk_time_grid!r}.")
            time_points.append(float(time_point))

        time_points[0] = 0.0
        time_points[-1] = float(tmax)
        time_points = np.asarray(time_points, dtype=float)
        if np.any(np.diff(time_points) <= 0.0):
            raise ValueError(
                "Automatic QK time-grid generation produced non-increasing "
                "time points. Use a positive qk_time_power or adjust the grid."
            )
        return time_points

    def _qk_effective_trotter_number_for_dt(self, delta_t):
        if self._qk_micro_dt_target is None:
            self._qk_micro_dt_target = self._dt / self._trotter_number
        return max(1, int(np.ceil(abs(float(delta_t)) / self._qk_micro_dt_target)))

    def _select_qk_tmax(self):
        if self._qk_tmax_override is not None:
            return float(self._qk_tmax_override)

        if self._qk_tmax_type == "manual":
            tmax = self._s * self._dt
        elif self._qk_tmax_type == "variance":
            tmax = self._compute_qk_hf_variance_tmax()
        elif self._qk_tmax_type == "auto":
            tmax = self._compute_qk_hf_variance_tmax()
            self._qk_tmax_limiter = "variance"
        else:
            raise ValueError(f"Unrecognized qk_tmax_type {self._qk_tmax_type!r}.")

        if not np.isfinite(tmax) or tmax <= 0.0:
            raise ValueError(
                "QK Tmax generation produced a nonpositive or invalid Tmax. "
                "Use qk_tmax_type='manual' or provide a positive qk_tmax."
            )

        return float(tmax)

    def _setup_qk_trotter_schedule(self):
        if self._qk_trotter_control == "fixed":
            self._qk_macro_trotter_number_list = [
                int(self._trotter_number) for _ in self._qk_macro_dt_list
            ]
            self._qk_effective_micro_dt_list = [
                abs(float(delta_t)) / self._trotter_number
                for delta_t in self._qk_macro_dt_list
            ]
            return

        lambda_int = self._compute_qk_trotter_lambda_int()
        eps = float(self._qk_target_trotter_error)
        eta = float(self._qk_trotter_bound_scale)
        tmax = float(self._qk_tmax)
        order = int(self._trotter_order)

        denom = eta * tmax * lambda_int ** (order + 1)
        if (
            not np.isfinite(denom)
            or denom <= 0.0
            or not np.isfinite(eps)
            or eps <= 0.0
        ):
            raise ValueError(
                "Could not determine an automatic QK Trotter schedule because "
                "the target error, bound scale, Tmax, or integral norm is invalid."
            )

        micro_dt_allowed = (eps / denom) ** (1.0 / order)
        if not np.isfinite(micro_dt_allowed) or micro_dt_allowed <= 0.0:
            raise ValueError(
                "Could not determine an automatic QK Trotter schedule because "
                "the allowed microstep is zero or invalid."
            )

        macro_trotter_numbers = [
            max(1, int(np.ceil(abs(float(delta_t)) / micro_dt_allowed)))
            for delta_t in self._qk_macro_dt_list
        ]
        effective_micro_dt = [
            abs(float(delta_t)) / m
            for delta_t, m in zip(self._qk_macro_dt_list, macro_trotter_numbers)
        ]
        proxy_estimate = eta * sum(
            m * (lambda_int * micro_dt) ** (order + 1)
            for m, micro_dt in zip(macro_trotter_numbers, effective_micro_dt)
        )

        self._qk_trotter_micro_dt_allowed = float(micro_dt_allowed)
        self._qk_trotter_step_strength_allowed = float(lambda_int * micro_dt_allowed)
        self._qk_trotter_proxy_estimate = float(proxy_estimate)
        self._qk_macro_trotter_number_list = macro_trotter_numbers
        self._qk_effective_micro_dt_list = effective_micro_dt

    def _setup_qk_times_and_trotter_schedule(self):
        self._reset_qk_time_grid_diagnostics()
        self._validate_qk_time_grid_inputs()

        tmax = self._select_qk_tmax()
        old_manual_linear_grid = (
            self._qk_tmax_type == "manual"
            and self._qk_tmax_override is None
            and self._qk_time_grid == "linear"
        )

        if old_manual_linear_grid:
            self._qk_base_step_counts = np.arange(self._s + 1, dtype=int)
            self._qk_time_points = self._dt * self._qk_base_step_counts.astype(float)
        else:
            self._qk_time_points = self._build_qk_continuous_time_points(tmax)
            self._qk_base_step_counts = np.rint(
                self._qk_time_points / self._dt
            ).astype(int)
            self._qk_base_step_counts[0] = 0

        self._qk_tmax = float(self._qk_time_points[-1])
        if old_manual_linear_grid:
            self._qk_macro_dt_list = np.full(self._s, self._dt, dtype=float)
        else:
            self._qk_macro_dt_list = np.diff(self._qk_time_points)
        self._qk_time_increments = self._qk_macro_dt_list
        self._qk_increment_counts = np.diff(self._qk_base_step_counts).astype(int)
        self._qk_micro_dt_target = self._dt / self._trotter_number

        self._qk_effective_dt_list = [
            float(delta_t) for delta_t in self._qk_macro_dt_list
        ]
        self._setup_qk_trotter_schedule()
        self._qk_effective_trotter_number_list = self._qk_macro_trotter_number_list

        legacy_counts = np.arange(self._s + 1, dtype=int)
        self._qk_uses_legacy_uniform_grid = (
            self._qk_tmax_type == "manual"
            and self._qk_tmax_override is None
            and self._qk_time_grid == "linear"
            and self._qk_trotter_control == "fixed"
            and np.array_equal(self._qk_base_step_counts, legacy_counts)
            and np.allclose(self._qk_macro_dt_list, self._dt)
            and all(
                m == self._trotter_number
                for m in self._qk_macro_trotter_number_list
            )
        )

        if (
            self._computer_type == "fock"
            and self._use_legacy_fock_srqk_protocal
            and not self._qk_uses_legacy_uniform_grid
        ):
            raise NotImplementedError(
                "Legacy fock SRQK only supports the old/default uniform QK "
                "grid: qk_tmax_type='manual', qk_time_grid='linear', "
                "qk_trotter_control='fixed', and no qk_tmax override."
            )
        if (not self._fast) and not self._qk_uses_legacy_uniform_grid:
            raise NotImplementedError(
                "SRQK realistic matrix-element construction does not yet support "
                "non-default QK time grids/Tmax heuristics/Trotter controls."
            )

    def _setup_qk_time_grid_and_tmax(self):
        self._setup_qk_times_and_trotter_schedule()

    # Define Algorithm abstract methods.
    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not fully implemented for SRQK.')

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_QSD_attributes()

    def _qk_format_sequence(self, seq, precision=6):
        if seq is None:
            return "None"
        values = list(seq)

        def fmt(value):
            if isinstance(value, (int, np.integer)):
                return str(int(value))
            return f"{float(value):.{precision}g}"

        if len(values) <= 10:
            return "[" + ", ".join(fmt(value) for value in values) + "]"

        head = ", ".join(fmt(value) for value in values[:5])
        tail = ", ".join(fmt(value) for value in values[-5:])
        return "[" + head + ", ..., " + tail + f"] (len={len(values)})"

    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('           Single Reference Quantum Krylov   ')
        print('-----------------------------------------------------')

        print('\n\n                     ==> QK options <==')
        print('-----------------------------------------------------------')
        # General algorithm options.
        print('Trial reference state:                   ',  ref_string(self._ref, self._nqb))
        print('Number of Hamiltonian Pauli terms:       ',  self._Nl)
        print('Trial state preparation method:          ',  self._state_prep_type)
        print('Trotter order (rho):                     ',  self._trotter_order)
        print('Trotter number (m):                      ',  self._trotter_number)
        print('Use exact time evolution?:               ',  self._use_exact_evolution)
        print('Use fast version of algorithm:           ',  str(self._fast))
        if(self._fast):
            print('Measurement varience thresh:             ',  'NA')
        else:
            print('Measurement varience thresh:             ',  0.01)

        # Specific SRQK options.
        print('Dimension of Krylov space (N):           ',  self._nstates)
        print('Target root:                             ',  str(self._target_root))
        print('Use low-memory matrix formation?:        ',  self._low_memory_mat_formation)
        print('Use legacy fock SRQK protocal?:          ',  self._use_legacy_fock_srqk_protocal)
        print('Input dt (in a.u.):                      ',  self._dt)
        print('QK Tmax type:                            ',  self._qk_tmax_type)
        if(self._qk_tmax_override is not None):
            print('Explicit QK Tmax override:               ',  self._qk_tmax_override)
        if(self._qk_tmax_limiter is not None):
            print('QK Tmax limiter:                         ',  self._qk_tmax_limiter)
        print('Selected QK Tmax (in a.u.):              ',  self._qk_tmax)
        print('QK time grid:                            ',  self._qk_time_grid)
        if(self._qk_time_grid == "power"):
            print('QK time-grid power:                      ',  self._qk_time_power)
        print('QK time points:                          ',  self._qk_format_sequence(self._qk_time_points))
        print('QK macro dt list:                        ',  self._qk_format_sequence(self._qk_macro_dt_list))
        print('QK Trotter control:                      ',  self._qk_trotter_control)
        print('QK macro Trotter numbers:                ',  self._qk_format_sequence(self._qk_macro_trotter_number_list))
        print('QK effective micro dt values:            ',  self._qk_format_sequence(self._qk_effective_micro_dt_list))
        print('QK base step counts:                     ',  self._qk_format_sequence(self._qk_base_step_counts))
        print('QK increment counts:                     ',  self._qk_format_sequence(self._qk_increment_counts))
        print('QK target Trotter error:                 ',  self._qk_target_trotter_error)
        print('QK Trotter bound scale:                  ',  self._qk_trotter_bound_scale)
        print('QK Trotter norm type:                    ',  self._qk_trotter_norm_type)
        if(self._qk_trotter_lambda_int is not None):
            print('QK Trotter Lambda_int:                   ',  self._qk_trotter_lambda_int)
            print('QK Trotter allowed micro dt:             ',  self._qk_trotter_micro_dt_allowed)
            print('QK Trotter allowed step strength:        ',  self._qk_trotter_step_strength_allowed)
            print('QK Trotter rounded proxy estimate:       ',  self._qk_trotter_proxy_estimate)
        if(self._qk_hf_variance is not None):
            print('QK HF energy:                            ',  self._qk_hf_energy)
            print('QK HF <H^2>:                             ',  self._qk_hf_h2)
            print('QK HF variance:                          ',  self._qk_hf_variance)
            print('QK HF sigma_H:                           ',  self._qk_hf_sigma_h)
            print('QK variance Tmax candidate:              ',  self._qk_tmax_variance)
        print('GEV stabilization threshold:             ',  self._gev_stabilization_thresh)


    def print_summary_banner(self):
        cs_str = '{:.2e}'.format(self._Scond)

        print('\n\n                     ==> QK summary <==')
        print('-----------------------------------------------------------')
        print('Condition number of overlap mat k(S):      ', cs_str)
        print('Final SRQK ground state Energy:           ', round(self._Egs, 10))
        print('Final SRQK target state Energy:           ', round(self._Ets, 10))
        print(f"Final target-state <S^2>:                 {self.final_spin_squared_summary_string()}")
        print('Number of classical parameters used:       ', self._n_classical_params)
        print('Number of CNOT gates in deepest circuit:   ', self._n_cnot)
        print('Number of Pauli term measurements:         ', self._n_pauli_trm_measures)

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

        print(self.final_noons_summary_table())

    def _unsupported_final_diagnostics_message(self):
        if getattr(self, "_use_exact_evolution", False):
            return (
                "SRQK final-state <S^2>/NOON diagnostics are not yet implemented "
                "for use_exact_evolution=True. The current Python-side diagnostic "
                "reconstruction replays the Trotterized Fock-space schedule only."
            )
        return (
            "SRQK final-state <S^2>/NOON diagnostics are not available for the "
            "current configuration."
        )

    def _build_final_fci_krylov_states(self):
        """Return FCI Krylov basis tensors for final-state diagnostics."""
        if (
            hasattr(self, "_omega_lst")
            and len(self._omega_lst) == self._nstates
            and all(hasattr(omega, "vector_dot") for omega in self._omega_lst)
        ):
            return self._omega_lst

        hermitian_pairs = qforte.SQOpPool()
        hermitian_pairs.add_hermitian_pairs(1.0, self._sq_ham)

        qc = qforte.FCIComputer(self._nel, self._2_spin, self._norb)
        qc.hartree_fock()

        omega_lst = []
        for m in range(self._nstates):
            if m > 0:
                self._qk_evolve_interval(
                    qc,
                    hermitian_pairs,
                    None,
                    None,
                    m - 1,
                    adjoint=False)

            omega_lst.append(qc.get_state_deep())

        return omega_lst

    def build_final_fci_state(self):
        """Build an FCIComputer holding the target-root SRQK wave function."""
        if self._computer_type != "fci":
            raise NotImplementedError(self._unsupported_final_diagnostics_message())

        omega_lst = self._build_final_fci_krylov_states()
        root_coeffs = self._eigenvectors[:, self._target_root]

        qc = qforte.FCIComputer(self._nel, self._2_spin, self._norb)
        final_state = qc.get_state_deep()
        final_state.zero()

        for coeff, omega in zip(root_coeffs, omega_lst):
            final_state.zaxpy(x=omega, alpha=coeff)

        norm = final_state.norm()
        if norm < 1.0e-14:
            raise ValueError("Cannot compute final SRQK diagnostics for a zero-norm state.")
        final_state.scale(1.0 / norm)

        qc.set_state(final_state)
        return qc

    def _backend_state_vector_to_numpy(self, state):
        """Convert a backend full-vector state copy into a NumPy array."""
        if hasattr(state, "get"):
            return np.asarray(state.get(), dtype=complex)
        return np.asarray(state, dtype=complex)

    def _backend_tensor_state_to_cpu_tensor(self, state):
        """Convert a determinant-tensor backend state copy into a CPU Tensor."""
        if hasattr(state, "copy_to_tensor"):
            tensor_shape = [
                int(math.comb(self._norb, self._nael)),
                int(math.comb(self._norb, self._nbel)),
            ]
            if hasattr(state, "to_cpu") and hasattr(state, "on_gpu") and state.on_gpu():
                state.to_cpu()
            tensor = qforte.Tensor(tensor_shape, "srqk_backend_state_cpu")
            state.copy_to_tensor(tensor)
            return tensor

        state_array = np.asarray(state, dtype=complex)
        tensor = qforte.Tensor(list(state_array.shape), "srqk_backend_state_cpu")
        tensor.fill_from_nparray(state_array.ravel(), list(state_array.shape))
        return tensor

    def _build_final_backend_fci_state(self):
        """Build a CPU FCIComputer from stored backend determinant-basis Krylov states."""
        if not (
            hasattr(self, "_omega_lst")
            and len(self._omega_lst) == self._nstates
        ):
            raise NotImplementedError(
                "Stored backend Krylov states are unavailable for final-state diagnostics."
            )

        root_coeffs = self._eigenvectors[:, self._target_root]
        final_state = self._backend_tensor_state_to_cpu_tensor(self._omega_lst[0])
        final_state.zero()

        for coeff, omega in zip(root_coeffs, self._omega_lst):
            omega_cpu = self._backend_tensor_state_to_cpu_tensor(omega)
            final_state.zaxpy(x=omega_cpu, alpha=coeff)

        norm = final_state.norm()
        if norm < 1.0e-14:
            raise ValueError("Cannot compute final SRQK diagnostics for a zero-norm state.")
        final_state.scale(1.0 / norm)

        qc = qforte.FCIComputer(self._nel, self._2_spin, self._norb)
        qc.set_state(final_state)
        return qc

    def _build_final_fock_krylov_states(self):
        """Return Fock-space Krylov basis vectors for final-state diagnostics."""
        if getattr(self, "_use_exact_evolution", False):
            raise NotImplementedError(self._unsupported_final_diagnostics_message())

        if (
            self._computer_type in {"fock", "cusv"}
            and hasattr(self, "_omega_lst")
            and len(self._omega_lst) == self._nstates
        ):
            return [self._backend_state_vector_to_numpy(omega) for omega in self._omega_lst]

        if self._use_legacy_fock_srqk_protocal:
            return [self._fock_state_for_krylov_power_legacy(m)[0] for m in range(self._nstates)]

        hermitian_pairs = qforte.SQOpPool()
        hermitian_pairs.add_hermitian_pairs(1.0, self._sq_ham)
        return [
            self._fock_build_basis_state(hermitian_pairs, m) for m in range(self._nstates)
        ]

    def build_final_fock_state(self):
        """Build a Fock-space Computer holding the target-root SRQK wave function."""
        omega_lst = self._build_final_fock_krylov_states()
        root_coeffs = self._eigenvectors[:, self._target_root]

        final_coeffs = np.zeros_like(np.asarray(omega_lst[0], dtype=complex), dtype=complex)
        for coeff, omega in zip(root_coeffs, omega_lst):
            final_coeffs += coeff * np.asarray(omega, dtype=complex)

        norm = np.linalg.norm(final_coeffs)
        if norm < 1.0e-14:
            raise ValueError("Cannot compute final SRQK diagnostics for a zero-norm state.")

        qc = qforte.Computer(self._nqb)
        qc.set_coeff_vec((final_coeffs / norm).tolist())
        return qc

    def _fock_noons_from_state(self, qc):
        """Return spin-summed spatial NOONs from a final Fock-space state."""
        coeffs = np.asarray(qc.get_coeff_vec(), dtype=complex)
        norm_sq = float(np.real(np.vdot(coeffs, coeffs)))
        if norm_sq < 1.0e-14:
            raise ValueError("Cannot compute natural occupations for a zero-norm state.")

        gamma = np.zeros((self._norb, self._norb), dtype=complex)
        for p in range(self._norb):
            for q in range(self._norb):
                element = 0.0 + 0.0j
                for spin in [0, 1]:
                    sqop = qforte.SQOperator()
                    sqop.add(1.0, [2 * p + spin], [2 * q + spin])
                    element += qc.direct_op_exp_val(sqop.jw_transform())
                gamma[p, q] = element / norm_sq

        gamma = 0.5 * (gamma + gamma.conj().T)
        noons = np.linalg.eigvalsh(np.real(gamma))
        return [float(noon) for noon in noons[::-1]]

    def compute_final_fock_diagnostics(self):
        """Compute and store final-state diagnostics using a Fock reconstruction."""
        try:
            qc = self.build_final_fock_state()
        except Exception as exc:
            err = str(exc)
            self._spin_squared = None
            self._spin_squared_error = err
            self._natural_orbital_occupation_numbers = None
            self._noons_error = err
            return

        try:
            s_squared = qforte.total_spin_squared(self._nqb)
            self._spin_squared = float(np.real(qc.direct_op_exp_val(s_squared)))
            self._spin_squared_error = None
        except Exception as exc:
            self._spin_squared = None
            self._spin_squared_error = str(exc)

        try:
            self._natural_orbital_occupation_numbers = self._fock_noons_from_state(qc)
            self._noons_error = None
        except Exception as exc:
            self._natural_orbital_occupation_numbers = None
            self._noons_error = str(exc)

    def compute_final_fci_diagnostics(self):
        """Compute and store final-state diagnostics when possible."""
        if self._computer_type in {"fock", "cusv"}:
            self.compute_final_fock_diagnostics()
            return

        if self._computer_type in {"fqe", "fci_gpu"}:
            try:
                qc = self._build_final_backend_fci_state()
            except Exception:
                self.compute_final_fock_diagnostics()
                return

            try:
                self._spin_squared = float(np.real(qc.get_spin_squared_expectation()))
                self._spin_squared_error = None
            except Exception as exc:
                self._spin_squared = None
                self._spin_squared_error = str(exc)

            try:
                noons = qc.get_natural_orbital_occupation_numbers()
                self._natural_orbital_occupation_numbers = [
                    float(np.real(noon)) for noon in noons
                ]
                self._noons_error = None
            except Exception as exc:
                self._natural_orbital_occupation_numbers = None
                self._noons_error = str(exc)
            return

        if self._computer_type != "fci":
            err = self._unsupported_final_diagnostics_message()
            self._spin_squared = None
            self._spin_squared_error = err
            self._natural_orbital_occupation_numbers = None
            self._noons_error = err
            return

        try:
            qc = self.build_final_fci_state()
        except Exception as exc:
            err = str(exc)
            self._spin_squared = None
            self._spin_squared_error = err
            self._natural_orbital_occupation_numbers = None
            self._noons_error = err
            return

        try:
            self._spin_squared = float(np.real(qc.get_spin_squared_expectation()))
            self._spin_squared_error = None
        except Exception as exc:
            self._spin_squared = None
            self._spin_squared_error = str(exc)

        try:
            noons = qc.get_natural_orbital_occupation_numbers()
            self._natural_orbital_occupation_numbers = [
                float(np.real(noon)) for noon in noons
            ]
            self._noons_error = None
        except Exception as exc:
            self._natural_orbital_occupation_numbers = None
            self._noons_error = str(exc)

    def compute_final_spin_squared_expectation(self):
        """Compute and store <S^2> for the target-root SRQK state."""
        self.compute_final_fci_diagnostics()
        return self._spin_squared

    def compute_final_noons(self):
        """Compute and store target-root natural orbital occupation numbers."""
        self.compute_final_fci_diagnostics()
        return self._natural_orbital_occupation_numbers

    def final_spin_squared_summary_string(self):
        """Return a printable target-root <S^2> summary value."""
        if not hasattr(self, "_spin_squared"):
            self.compute_final_spin_squared_expectation()
        if self._spin_squared is None:
            return f"N/A ({getattr(self, '_spin_squared_error', 'unavailable')})"
        return f"{self._spin_squared:12.10f}"

    def final_noons_summary_table(self):
        """Return a readable indexed table of target-root natural occupations."""
        if not hasattr(self, "_natural_orbital_occupation_numbers"):
            self.compute_final_noons()

        noons = self._natural_orbital_occupation_numbers
        lines = [
            "\n\n                ==> Final NOONs <==",
            "-----------------------------------------------------------",
        ]

        if noons is None:
            lines.append(f"N/A ({getattr(self, '_noons_error', 'unavailable')})")
            return "\n".join(lines)

        lines.append(" Natural orbital        Occupation")
        lines.append(" ---------------        ----------")
        for idx, noon in enumerate(noons):
            lines.append(f" {idx:15d}        {noon:10.8f}")
        lines.append(" ---------------        ----------")
        lines.append(f" {'sum':>15s}        {sum(noons):10.8f}")
        return "\n".join(lines)

    def build_qk_mats(self):
        if (self._fast):
            return self.build_qk_mats_fast()
        else:
            return self.build_qk_mats_realistic()

    # def build_qk_mats_fast(self):

    def build_qk_mats_fast(self):
        if(self._computer_type == 'fock'):
            if(self._use_legacy_fock_srqk_protocal):
                if(self._trotter_order != 1):
                    raise ValueError(
                        "Legacy fock SRQK is only compatible with 1st order "
                        "Trotter currently."
                    )
                if(self._low_memory_mat_formation):
                    return self.build_qk_mats_fast_fock_low_memory_legacy()

                return self.build_qk_mats_fast_fock_legacy()

            if(self._use_exact_evolution):
                raise NotImplementedError(
                    "Exact evolution is not implemented for the fock SRQK "
                    "Hermitian-pair protocol."
                )
            if(self._trotter_order not in [1, 2]):
                raise ValueError(
                    "fock computer SRQK Hermitian-pair protocol is only "
                    "compatible with 1st and 2nd order Trotter currently."
                )
            if(self._low_memory_mat_formation):
                return self.build_qk_mats_fast_fock_low_memory()

            return self.build_qk_mats_fast_fock()

        elif(self._computer_type == 'fci'):
            if(self._trotter_order not in [1, 2]):
                raise ValueError("fci computer SRQK only compatible with 1st and 2nd order trotter currently")

            if(self._low_memory_mat_formation):
                return self.build_qk_mats_fast_low_memory()

            return self.build_qk_mats_fast_fci()

        elif(self._computer_type == 'fqe'):
            if(self._trotter_order not in [1, 2]):
                raise ValueError("fqe computer SRQK only compatible with 1st and 2nd order trotter currently")

            if(self._low_memory_mat_formation):
                return self.build_qk_mats_fast_low_memory()

            return self.build_qk_mats_fast_fqe()

        elif (self._computer_type == 'fci_gpu'):
            if(self._trotter_order not in [1, 2]):
                raise ValueError("fci gpu computer SRQK only compatible with 1st and 2nd order trotter currently")

            if(self._low_memory_mat_formation):
                return self.build_qk_mats_fast_low_memory()

            return self.build_qk_mats_fast_fci_gpu()

        elif (self._computer_type == 'cusv'):
            if(self._trotter_order not in [1, 2]):
                raise ValueError("cusv computer SRQK only compatible with 1st and 2nd order trotter currently")

            if(self._low_memory_mat_formation):
                return self.build_qk_mats_fast_low_memory()

            return self.build_qk_mats_fast_cusv()

        else:
            raise ValueError(f"{self._computer_type} is an unrecognized computer type.")

    def _time_qk_backend_call(self, timers, clock, label, func, *args, **kwargs):
        if timers is None or clock is None:
            return func(*args, **kwargs)

        t0 = clock.time()
        value = func(*args, **kwargs)
        if label not in timers:
            timers[label] = 0.0
        timers[label] += clock.time() - t0
        return value

    def _solve_qk_geig(self, s_mat, h_mat, print_mats=False, sort_ret_vals=False):
        return canonical_geig_solve(
            s_mat,
            h_mat,
            print_mats=print_mats,
            sort_ret_vals=sort_ret_vals,
            stabilization_thresh=self._gev_stabilization_thresh)

    def _qk_evolve_increment(
            self,
            qc,
            hermitian_pairs,
            timers,
            clock,
            time_increment,
            effective_trotter_number,
            adjoint=False):
        time_increment = float(time_increment)
        effective_trotter_number = int(effective_trotter_number)
        if effective_trotter_number < 1:
            raise ValueError("QK effective Trotter number must be >= 1.")
        if abs(time_increment) <= 0.0:
            return None

        if(self._use_exact_evolution):
            # Exact Taylor evolution is applied over the total interval requested
            # by the nonuniform grid; very large intervals may need tighter
            # Taylor controls in future.
            signed_dt = -time_increment if adjoint else time_increment

            if(self._computer_type == 'fci'):
                if(self._apply_ham_as_tensor):
                    label = 'evolve_tensor_taylor_adjoint' if adjoint else 'evolve_tensor_taylor'
                    return self._time_qk_backend_call(
                        timers,
                        clock,
                        label,
                        qc.evolve_tensor_taylor,
                        self._zero_body_energy,
                        self._mo_oeis,
                        self._mo_teis,
                        self._mo_teis_einsum,
                        self._norb,
                        signed_dt,
                        1.0e-15,
                        30,
                        False)

                label = 'evolve_op_taylor_adjoint' if adjoint else 'evolve_op_taylor'
                return self._time_qk_backend_call(
                    timers,
                    clock,
                    label,
                    qc.evolve_op_taylor,
                    self._sq_ham,
                    signed_dt,
                    1.0e-15,
                    30,
                    False)

            elif(self._computer_type == 'fqe'):
                label = 'evolve_tensor_taylor_adjoint' if adjoint else 'evolve_tensor_taylor'
                return self._time_qk_backend_call(
                    timers,
                    clock,
                    label,
                    qc.evolve_tensor_taylor,
                    self._zero_body_energy,
                    self._mo_oeis_np,
                    self._mo_teis_np,
                    signed_dt,
                    1.0e-15,
                    30,
                    False)

            elif(self._computer_type == 'fci_gpu'):
                if(self._apply_ham_as_tensor):
                    raise NotImplementedError("Exact tensor evolution not implemented for FCI_GPU computer type.")

                label = 'evolve_op_taylor_cpu_adjoint' if adjoint else 'evolve_op_taylor_cpu'
                return self._time_qk_backend_call(
                    timers,
                    clock,
                    label,
                    qc.evolve_op_taylor_cpu,
                    self._sq_ham,
                    signed_dt,
                    1.0e-15,
                    30,
                    False)

            elif(self._computer_type == 'cusv'):
                raise NotImplementedError("Exact evolution not implemented for CUSV computer type.")

            raise ValueError(f"{self._computer_type} is an unrecognized computer type.")

        if(self._computer_type == 'fci_gpu'):
            label = 'evolve_pool_trotter_gpu_adjoint' if adjoint else 'evolve_pool_trotter_gpu'
            return self._time_qk_backend_call(
                timers,
                clock,
                label,
                qc.evolve_pool_trotter_gpu,
                hermitian_pairs,
                time_increment,
                effective_trotter_number,
                self._trotter_order,
                antiherm=False,
                adjoint=adjoint)

        label = 'evolve_pool_trotter_adjoint' if adjoint else 'evolve_pool_trotter'
        return self._time_qk_backend_call(
            timers,
            clock,
            label,
            qc.evolve_pool_trotter,
            hermitian_pairs,
            time_increment,
            effective_trotter_number,
            self._trotter_order,
            antiherm=False,
            adjoint=adjoint)

    def _qk_evolve_interval(
            self,
            qc,
            hermitian_pairs,
            timers,
            clock,
            interval_index,
            adjoint=False):
        interval_index = int(interval_index)
        if interval_index < 0 or interval_index >= len(self._qk_macro_dt_list):
            raise IndexError(
                f"QK interval index {interval_index} is out of range for "
                f"{len(self._qk_macro_dt_list)} intervals."
            )

        return self._qk_evolve_increment(
            qc,
            hermitian_pairs,
            timers,
            clock,
            self._qk_macro_dt_list[interval_index],
            self._qk_macro_trotter_number_list[interval_index],
            adjoint=adjoint)

    def _qk_summary_file(self):
        if(not self._diagonalize_each_step):
            return None

        self._qk_last_step_energy = None
        print('\n\n')
        header, separator = self._qk_iteration_table_header()
        print(header)
        print(separator)

        if (self._print_summary_file):
            f = open("summary.dat", "w+", buffering=1)
            f.write("#" + header + "\n")
            f.write("#" + separator + "\n")
            return f

        return None

    def _qk_iteration_table_header(self):
        header = (
            f"{'k(S)':>10}"
            f"{'E(Npar)':>18}"
            f"{'dE':>14}"
            f"{'N(params)':>12}"
            f"{'RR':>6}"
            f"{'Ttot':>12}"
            f"{'N(CNOT)':>14}"
            f"{'N(measure)':>14}"
        )
        return header, "-" * len(header)

    def _qk_geig_reduced_rank(self, s_mat):
        threshold = (
            1.0e-15
            if self._gev_stabilization_thresh is None
            else self._gev_stabilization_thresh
        )
        s_eigs = np.linalg.eigvals(s_mat)
        return int(sum(np.real(sii) > threshold for sii in s_eigs))

    def _qk_total_time_for_step(self, k):
        if self._qk_time_points is not None and len(self._qk_time_points) >= k:
            return float(self._qk_time_points[k - 1])
        return float(k - 1) * self._dt

    def _qk_trotter_order_cnot_factor(self):
        if self._trotter_order <= 1:
            return 1
        return 2 ** (self._trotter_order - 1)

    def _qk_base_first_order_trotter_cnot(self, hermitian_pairs):
        if hermitian_pairs is None:
            return 0
        if getattr(self, "_qk_base_first_order_cnot", None) is None:
            try:
                self._qk_base_first_order_cnot = int(
                    hermitian_pairs.count_cnot_for_jw_exponential(False, 1)
                )
            except AttributeError:
                self._qk_base_first_order_cnot = 0
        return self._qk_base_first_order_cnot

    def _qk_trotter_cnot_for_step(self, k, hermitian_pairs=None):
        basis_index = max(0, int(k) - 1)
        if basis_index == 0:
            return 0
        if self._qk_macro_trotter_number_list is None:
            total_trotter_steps = basis_index * int(self._trotter_number)
        else:
            total_trotter_steps = int(
                sum(self._qk_macro_trotter_number_list[:basis_index])
            )
        base_cnot = self._qk_base_first_order_trotter_cnot(hermitian_pairs)
        return base_cnot * self._qk_trotter_order_cnot_factor() * total_trotter_steps

    def _set_qk_resource_counts(self, k, n_cnot):
        self._n_classical_params = k
        self._n_cnot = n_cnot
        self._n_pauli_trm_measures  = k * self._Nl
        self._n_pauli_trm_measures += k * (k-1) * self._Nl
        self._n_pauli_trm_measures += k * (k-1)

    def _diagonalize_qk_step(
            self,
            k,
            s_mat,
            h_mat,
            n_cnot=None,
            summary_file=None,
            hermitian_pairs=None):
        if(not self._diagonalize_each_step):
            return

        evals, evecs = self._solve_qk_geig(s_mat[0:k, 0:k],
                           h_mat[0:k, 0:k],
                           print_mats=False,
                           sort_ret_vals=True)

        scond = np.linalg.cond(s_mat[0:k, 0:k])
        rr = self._qk_geig_reduced_rank(s_mat[0:k, 0:k])
        energy = float(np.real(evals[self._target_root]))
        previous_energy = getattr(self, "_qk_last_step_energy", None)
        dE = None if previous_energy is None else energy - previous_energy
        self._qk_last_step_energy = energy
        if n_cnot is None:
            n_cnot = self._qk_trotter_cnot_for_step(k, hermitian_pairs)
        ttot = self._qk_total_time_for_step(k)
        self._set_qk_resource_counts(k, n_cnot)

        dE_str = f"{dE:+.3e}" if dE is not None else "--"
        line = (
            f"{scond:10.2e}"
            f"{energy:+18.9f}"
            f"{dE_str:>14}"
            f"{self._n_classical_params:12d}"
            f"{rr:6d}"
            f"{ttot:12.6g}"
            f"{self._n_cnot:14d}"
            f"{self._n_pauli_trm_measures:14d}"
        )
        print(line)
        if (summary_file is not None):
            summary_file.write(line + "\n")

    def _low_memory_backend_objects(self):
        if(self._computer_type == 'fci'):
            hermitian_pairs = qforte.SQOpPool()
            hermitian_pairs.add_hermitian_pairs(1.0, self._sq_ham)

            if self._fci_timers is None:
                self._fci_timers = {}
            if self._fci_time is None:
                self._fci_time = time

            qc = qforte.FCIComputer(
                    self._nel,
                    self._2_spin,
                    self._norb)
            return qc, hermitian_pairs, self._fci_timers, self._fci_time

        elif(self._computer_type == 'fqe'):
            hermitian_pairs = qforte.SQOpPool()
            hermitian_pairs.add_hermitian_pairs(1.0, self._sq_ham)

            if self._fqe_timers is None:
                self._fqe_timers = {}
            if self._fqe_time is None:
                self._fqe_time = time

            qc = qforte.FQEComputer(
                    self._nel,
                    self._2_spin,
                    self._norb)
            return qc, hermitian_pairs, self._fqe_timers, self._fqe_time

        elif(self._computer_type == 'fci_gpu'):
            if self._gpu_timers is None:
                self._gpu_timers = {}
            if self._gpu_time is None:
                self._gpu_time = time

            if (hasattr(self, '_gpu_global_initialized') and self._gpu_global_initialized):
                return self._reusable_qc, self._hermitian_pairs, self._gpu_timers, self._gpu_time

            hermitian_pairs = qforte.SQOpPoolGPU(data_type=self.data_type)
            hermitian_pairs.add_hermitian_pairs(1.0, self._sq_ham)
            qc = qforte.FCIComputerGPU(
                    self._nel,
                    self._2_spin,
                    self._norb,
                    on_gpu=True,
                    data_type=self.data_type)
            return qc, hermitian_pairs, self._gpu_timers, self._gpu_time

        elif(self._computer_type == 'cusv'):
            hermitian_pairs = qforte.SQOpPool()
            hermitian_pairs.add_hermitian_pairs(1.0, self._sq_ham)

            if self._cusv_timers is None:
                self._cusv_timers = {}
            if self._cusv_time is None:
                self._cusv_time = time

            qc = qforte.CUSVComputer(
                    self._nel,
                    self._2_spin,
                    self._norb)
            return qc, hermitian_pairs, self._cusv_timers, self._cusv_time

        else:
            raise ValueError(f"{self._computer_type} is an unrecognized low-memory computer type.")

    def _low_memory_hartree_fock(self, qc, timers, clock):
        if(self._computer_type == 'fci_gpu'):
            return self._time_qk_backend_call(
                timers, clock, 'hartree_fock_gpu', qc.hartree_fock_gpu)

        return self._time_qk_backend_call(
            timers, clock, 'hartree_fock', qc.hartree_fock)

    def _low_memory_hf_dot(self, qc):
        try:
            return qc.get_hf_dot()
        except NotImplementedError:
            if(hasattr(qc, '_hf_idx') and hasattr(qc, 'get_element')):
                return qc.get_element(qc._hf_idx)
            raise

    def _low_memory_get_state_deep(self, qc, timers, clock):
        return self._time_qk_backend_call(
            timers, clock, 'get_state_deep', qc.get_state_deep)

    def _low_memory_state_dot(self, left_state, right_state):
        if(hasattr(left_state, 'vector_dot')):
            return left_state.vector_dot(right_state)

        if(hasattr(left_state, 'get') and not isinstance(left_state, np.ndarray)):
            left_state = left_state.get()
        if(hasattr(right_state, 'get') and not isinstance(right_state, np.ndarray)):
            right_state = right_state.get()

        return np.vdot(left_state, right_state)

    def _low_memory_build_time_state(self, qc, hermitian_pairs, timers, clock, total_time):
        self._low_memory_hartree_fock(qc, timers, clock)
        effective_trotter_number = self._qk_effective_trotter_number_for_dt(total_time)
        self._qk_evolve_increment(
            qc,
            hermitian_pairs,
            timers,
            clock,
            total_time,
            effective_trotter_number,
            adjoint=False)

    def _low_memory_build_basis_state(self, qc, hermitian_pairs, timers, clock, basis_index):
        basis_index = int(basis_index)
        if basis_index < 0 or basis_index >= self._nstates:
            raise IndexError(
                f"QK basis index {basis_index} is out of range for "
                f"{self._nstates} states."
            )

        self._low_memory_hartree_fock(qc, timers, clock)
        for interval_index in range(basis_index):
            self._qk_evolve_interval(
                qc,
                hermitian_pairs,
                timers,
                clock,
                interval_index,
                adjoint=False)

    def _low_memory_build_step_count_state(self, qc, hermitian_pairs, timers, clock, step_count):
        self._low_memory_build_time_state(
            qc,
            hermitian_pairs,
            timers,
            clock,
            step_count * self._dt)

    def _low_memory_build_power_state(self, qc, hermitian_pairs, timers, clock, power):
        self._low_memory_build_basis_state(
            qc,
            hermitian_pairs,
            timers,
            clock,
            power)

    def _low_memory_evolve(self, qc, hermitian_pairs, timers, clock, adjoint=False):
        return self._qk_evolve_increment(
            qc,
            hermitian_pairs,
            timers,
            clock,
            self._dt,
            self._trotter_number,
            adjoint=adjoint)

    def _low_memory_apply_ham(self, qc, timers, clock):
        if(self._computer_type == 'fci'):
            if(self._apply_ham_as_tensor):
                return self._time_qk_backend_call(
                    timers,
                    clock,
                    'apply_tensor_spat_012bdy',
                    qc.apply_tensor_spat_012bdy,
                    self._zero_body_energy,
                    self._mo_oeis,
                    self._mo_teis,
                    self._mo_teis_einsum,
                    self._norb)

            return self._time_qk_backend_call(
                timers, clock, 'apply_sqop', qc.apply_sqop, self._sq_ham)

        elif(self._computer_type == 'fqe'):
            if(self._apply_ham_as_tensor):
                return self._time_qk_backend_call(
                    timers,
                    clock,
                    'apply_tensor_spat_012bdy',
                    qc.apply_tensor_spat_012bdy,
                    self._zero_body_energy,
                    self._mo_oeis_np,
                    self._mo_teis_np)

            return self._time_qk_backend_call(
                timers, clock, 'apply_sqop', qc.apply_sqop, self._sq_ham)

        elif(self._computer_type == 'fci_gpu'):
            if(self._apply_ham_as_tensor):
                return self._time_qk_backend_call(
                    timers,
                    clock,
                    'apply_tensor_spat_012bdy_gpu',
                    qc.apply_tensor_spat_012bdy_gpu,
                    self._zero_body_energy,
                    self._mo_oeis_gpu,
                    self._mo_teis_gpu,
                    self._mo_teis_einsum_gpu,
                    self._norb)

            return self._time_qk_backend_call(
                timers, clock, 'apply_sqop_gpu', qc.apply_sqop_gpu, self._sq_ham)

        elif(self._computer_type == 'cusv'):
            return self._time_qk_backend_call(
                timers, clock, 'apply_sqop', qc.apply_sqop, self._sq_ham)

        raise ValueError(f"{self._computer_type} is an unrecognized low-memory computer type.")

    def build_qk_mats_fast_low_memory(self):
        """Build SRQK matrices without caching Krylov states.

        This path rebuilds each requested QK basis state by replaying the stored
        macro intervals.  That is deliberately simple: it is correct for both
        uniform and nonuniform grids, including interval-dependent Trotter
        numbers, without relying on a fixed-unitary lag shortcut.
        """

        K = self._nstates
        h_mat = np.zeros((K, K), dtype=complex)
        s_mat = np.zeros((K, K), dtype=complex)

        self._omega_lst = []

        qc, hermitian_pairs, timers, clock = self._low_memory_backend_objects()
        summary_file = self._qk_summary_file()

        for n in range(K):
            self._low_memory_build_basis_state(
                qc,
                hermitian_pairs,
                timers,
                clock,
                n)
            omega_n = self._low_memory_get_state_deep(qc, timers, clock)
            self._low_memory_apply_ham(qc, timers, clock)
            Homega_n = self._low_memory_get_state_deep(qc, timers, clock)

            for m in range(n + 1):
                self._low_memory_build_basis_state(
                    qc,
                    hermitian_pairs,
                    timers,
                    clock,
                    m)
                omega_m = self._low_memory_get_state_deep(qc, timers, clock)

                s_mat[m, n] = self._low_memory_state_dot(omega_m, omega_n)
                s_mat[n, m] = np.conj(s_mat[m, n])

                h_mat[m, n] = self._low_memory_state_dot(omega_m, Homega_n)
                h_mat[n, m] = np.conj(h_mat[m, n])

            self._diagonalize_qk_step(
                n + 1,
                s_mat,
                h_mat,
                summary_file=summary_file,
                hermitian_pairs=hermitian_pairs)

        if(self._diagonalize_each_step and self._print_summary_file):
            summary_file.close()

        self._set_qk_resource_counts(
            K,
            self._qk_trotter_cnot_for_step(K, hermitian_pairs))

        return s_mat, h_mat

    def _fock_state_for_krylov_power_legacy(self, power):
        U = qforte.Circuit()
        U.add(self._Uprep)
        phase = 1.0

        if(power > 0):
            fact = (0.0-1.0j) * power * self._dt
            expn_op, phase = trotterize(
                self._qb_ham,
                factor=fact,
                trotter_number=self._trotter_number)
            U.add(expn_op)

        qc = qforte.Computer(self._nqb)
        qc.apply_circuit(U)
        qc.apply_constant(phase)
        return np.asarray(qc.get_coeff_vec(), dtype=complex), U.get_num_cnots()

    def build_qk_mats_fast_fock_low_memory_legacy(self):
        """Build legacy fock SRQK matrices without storing Krylov states."""

        K = self._nstates
        h_mat = np.zeros((K, K), dtype=complex)
        s_mat = np.zeros((K, K), dtype=complex)

        self._omega_lst = []

        summary_file = self._qk_summary_file()
        max_n_cnot = 0

        for n in range(K):
            omega_n, n_cnot = self._fock_state_for_krylov_power_legacy(n)
            max_n_cnot = max(max_n_cnot, 2 * n_cnot)

            qc_h = qforte.Computer(self._nqb)
            qc_h.set_coeff_vec(omega_n.tolist())
            qc_h.apply_operator(self._qb_ham)
            Homega_n = np.asarray(qc_h.get_coeff_vec(), dtype=complex)

            for m in range(n + 1):
                omega_m, m_cnot = self._fock_state_for_krylov_power_legacy(m)
                max_n_cnot = max(max_n_cnot, 2 * m_cnot)

                h_mat[m, n] = np.vdot(omega_m, Homega_n)
                h_mat[n, m] = np.conj(h_mat[m, n])

                s_mat[m, n] = np.vdot(omega_m, omega_n)
                s_mat[n, m] = np.conj(s_mat[m, n])

            self._diagonalize_qk_step(
                n + 1,
                s_mat,
                h_mat,
                max_n_cnot,
                summary_file)

        if(self._diagonalize_each_step and self._print_summary_file):
            summary_file.close()

        self._set_qk_resource_counts(K, max_n_cnot)

        return s_mat, h_mat

    def build_qk_mats_fast_fock_legacy(self):
        """Returns matrices S and H needed for the QK algorithm using the Trotterized
        form of the unitary operators U_n = exp(-i n dt H)

        The mathematical operations of this function are unphysical for a quantum
        computer, but efficient for a simulator.

        Returns
        -------
        s_mat : ndarray
            A numpy array containing the elements S_mn = <Phi | Um^dag Un | Phi>.
            _nstates by _nstates

        h_mat : ndarray
            A numpy array containing the elements H_mn = <Phi | Um^dag H Un | Phi>
            _nstates by _nstates
        """

        h_mat = np.zeros((self._nstates,self._nstates), dtype=complex)
        s_mat = np.zeros((self._nstates,self._nstates), dtype=complex)

        # Store these vectors for the aid of MRSQK
        self._omega_lst = []
        Homega_lst = []

        summary_file = self._qk_summary_file()

        for m in range(self._nstates):
            # Compute U_m = exp(-i m dt H)
            Um = qforte.Circuit()
            Um.add(self._Uprep)
            phase1 = 1.0

            if(m>0):
                fact = (0.0-1.0j) * m * self._dt
                expn_op1, phase1 = trotterize(self._qb_ham, factor=fact, trotter_number=self._trotter_number)
                Um.add(expn_op1)

            # Compute U_m |φ>
            QC = qforte.Computer(self._nqb)
            QC.apply_circuit(Um)
            QC.apply_constant(phase1)
            self._omega_lst.append(np.asarray(QC.get_coeff_vec(), dtype=complex))

            # Compute H U_m |φ>
            QC.apply_operator(self._qb_ham)
            Homega_lst.append(np.asarray(QC.get_coeff_vec(), dtype=complex))

            # Compute S_mn = <φ| U_m^\dagger U_n |φ> and H_mn = <φ| U_m^\dagger H U_n |φ>
            for n in range(len(self._omega_lst)):
                h_mat[m][n] = np.vdot(self._omega_lst[m], Homega_lst[n])
                h_mat[n][m] = np.conj(h_mat[m][n])
                s_mat[m][n] = np.vdot(self._omega_lst[m], self._omega_lst[n])
                s_mat[n][m] = np.conj(s_mat[m][n])

            self._diagonalize_qk_step(
                m + 1,
                s_mat,
                h_mat,
                2 * Um.get_num_cnots(),
                summary_file)

        if (self._diagonalize_each_step and self._print_summary_file):
            summary_file.close()

        self._set_qk_resource_counts(
            self._nstates,
            2 * Um.get_num_cnots())


        return s_mat, h_mat

    def _fock_add_jw_sqop_evolution(self, circuit, sq_op, factor):
        if abs(complex(factor)) <= 0.0:
            return 1.0

        jw_op = sq_op.jw_transform()
        if len(jw_op.terms()) == 0:
            return 1.0

        exp_op, phase = trotterize(
            jw_op,
            factor=factor,
            trotter_number=1,
            trotter_order=1)
        circuit.add(exp_op)
        return phase

    def _fock_hermitian_pair_interval_circuit(
            self,
            hermitian_pairs,
            interval_index,
            adjoint=False):
        macro_dt = float(self._qk_macro_dt_list[interval_index])
        trotter_steps = int(self._qk_macro_trotter_number_list[interval_index])
        if trotter_steps < 1:
            raise ValueError("Fock SRQK interval Trotter number must be >= 1.")

        terms = list(hermitian_pairs.terms())
        circuit = qforte.Circuit()
        total_phase = 1.0
        sign = 1.0j if adjoint else -1.0j

        if self._trotter_order == 1:
            prefactor = macro_dt / float(trotter_steps)
            ordered_terms = list(reversed(terms)) if adjoint else terms
            for _ in range(trotter_steps):
                for coeff, sq_op in ordered_terms:
                    phase = self._fock_add_jw_sqop_evolution(
                        circuit,
                        sq_op,
                        sign * prefactor * coeff)
                    total_phase *= phase

        elif self._trotter_order == 2:
            prefactor = 0.5 * macro_dt / float(trotter_steps)
            ordered_terms = list(reversed(terms)) if adjoint else terms
            for _ in range(trotter_steps):
                for coeff, sq_op in ordered_terms:
                    phase = self._fock_add_jw_sqop_evolution(
                        circuit,
                        sq_op,
                        sign * prefactor * coeff)
                    total_phase *= phase
                for coeff, sq_op in reversed(ordered_terms):
                    phase = self._fock_add_jw_sqop_evolution(
                        circuit,
                        sq_op,
                        sign * prefactor * coeff)
                    total_phase *= phase

        else:
            raise ValueError(
                "fock computer SRQK Hermitian-pair protocol is only compatible "
                "with 1st and 2nd order Trotter currently."
            )

        return circuit, total_phase

    def _fock_build_basis_state(self, hermitian_pairs, basis_index):
        basis_index = int(basis_index)
        if basis_index < 0 or basis_index >= self._nstates:
            raise IndexError(
                f"QK basis index {basis_index} is out of range for "
                f"{self._nstates} states."
            )

        qc = qforte.Computer(self._nqb)
        qc.apply_circuit(self._Uprep)
        for interval_index in range(basis_index):
            interval_circuit, phase = self._fock_hermitian_pair_interval_circuit(
                hermitian_pairs,
                interval_index,
                adjoint=False)
            qc.apply_circuit(interval_circuit)
            qc.apply_constant(phase)
        return np.asarray(qc.get_coeff_vec(), dtype=complex)

    def build_qk_mats_fast_fock_low_memory(self):
        """Build fock SRQK matrices by replaying Hermitian-pair intervals."""

        K = self._nstates
        h_mat = np.zeros((K, K), dtype=complex)
        s_mat = np.zeros((K, K), dtype=complex)

        self._omega_lst = []

        hermitian_pairs = qforte.SQOpPool()
        hermitian_pairs.add_hermitian_pairs(1.0, self._sq_ham)

        summary_file = self._qk_summary_file()

        for n in range(K):
            omega_n = self._fock_build_basis_state(hermitian_pairs, n)

            qc_h = qforte.Computer(self._nqb)
            qc_h.set_coeff_vec(omega_n.tolist())
            qc_h.apply_operator(self._qb_ham)
            Homega_n = np.asarray(qc_h.get_coeff_vec(), dtype=complex)

            for m in range(n + 1):
                omega_m = self._fock_build_basis_state(hermitian_pairs, m)

                s_mat[m, n] = np.vdot(omega_m, omega_n)
                s_mat[n, m] = np.conj(s_mat[m, n])

                h_mat[m, n] = np.vdot(omega_m, Homega_n)
                h_mat[n, m] = np.conj(h_mat[m, n])

            self._diagonalize_qk_step(
                n + 1,
                s_mat,
                h_mat,
                summary_file=summary_file,
                hermitian_pairs=hermitian_pairs)

        if(self._diagonalize_each_step and self._print_summary_file):
            summary_file.close()

        self._set_qk_resource_counts(
            K,
            self._qk_trotter_cnot_for_step(K, hermitian_pairs))

        return s_mat, h_mat

    def build_qk_mats_fast_fock(self):
        """Build fock SRQK matrices with incremental Hermitian-pair evolution."""

        h_mat = np.zeros((self._nstates,self._nstates), dtype=complex)
        s_mat = np.zeros((self._nstates,self._nstates), dtype=complex)

        self._omega_lst = []
        Homega_lst = []

        hermitian_pairs = qforte.SQOpPool()
        hermitian_pairs.add_hermitian_pairs(1.0, self._sq_ham)

        QC = qforte.Computer(self._nqb)
        QC.apply_circuit(self._Uprep)

        summary_file = self._qk_summary_file()

        for m in range(self._nstates):
            if(m > 0):
                interval_circuit, phase = self._fock_hermitian_pair_interval_circuit(
                    hermitian_pairs,
                    m - 1,
                    adjoint=False)
                QC.apply_circuit(interval_circuit)
                QC.apply_constant(phase)

            C = np.asarray(QC.get_coeff_vec(), dtype=complex)
            self._omega_lst.append(C)

            QC.apply_operator(self._qb_ham)
            Homega_lst.append(np.asarray(QC.get_coeff_vec(), dtype=complex))

            QC.set_coeff_vec(C.tolist())

            for n in range(len(self._omega_lst)):
                h_mat[m][n] = np.vdot(self._omega_lst[m], Homega_lst[n])
                h_mat[n][m] = np.conj(h_mat[m][n])
                s_mat[m][n] = np.vdot(self._omega_lst[m], self._omega_lst[n])
                s_mat[n][m] = np.conj(s_mat[m][n])

            self._diagonalize_qk_step(
                m + 1,
                s_mat,
                h_mat,
                summary_file=summary_file,
                hermitian_pairs=hermitian_pairs)

        if (self._diagonalize_each_step and self._print_summary_file):
            summary_file.close()

        self._set_qk_resource_counts(
            self._nstates,
            self._qk_trotter_cnot_for_step(self._nstates, hermitian_pairs))

        return s_mat, h_mat
    
    # This function is a legacy and uses the same logic for handeling trotterizaiotn as the
    # fock implementation, build_qk_mats_fast_fci should be used in most cases.
    def build_qk_mats_fast_fci2(self):
        """Returns matrices S and H needed for the QK algorithm using the Trotterized
        form of the unitary operators U_n = exp(-i n dt H)

        The mathematical operations of this function are unphysical for a quantum
        computer, but efficient for a simulator.

        Returns
        -------
        s_mat : ndarray
            A numpy array containing the elements S_mn = <Phi | Um^dag Un | Phi>.
            _nstates by _nstates

        h_mat : ndarray
            A numpy array containing the elements H_mn = <Phi | Um^dag H Un | Phi>
            _nstates by _nstates
        """

        h_mat = np.zeros((self._nstates,self._nstates), dtype=complex)
        s_mat = np.zeros((self._nstates,self._nstates), dtype=complex)

        # Store these vectors for the aid of MRSQK
        self._omega_lst = []
        Homega_lst = []

        hermitian_pairs = qforte.SQOpPool()
        hermitian_pairs.add_hermitian_pairs(0.0, self._sq_ham)

        summary_file = self._qk_summary_file()

        """In reviewing this there is going to be an inherent ordering probelm. I want to apply 
        based on hermitian paris of SQ operators but the qb hamiltonain has been 'simplified' 
        and looses the exact correspondance to the sq hamiltonain"""
        for m in range(self._nstates):

            if(m>0):
                hermitian_pairs.set_coeffs_to_scaler(m * self._dt)

            # Compute U_m |φ>
            QC = qforte.FCIComputer(
                self._nel, 
                self._2_spin, 
                self._norb)
            
            QC.hartree_fock()

            QC.evolve_pool_trotter_basic(
                hermitian_pairs,
                antiherm=False,
                adjoint=False)

            C = QC.get_state_deep()
            self._omega_lst.append(C)

            # Compute H U_m |φ>
            QC.apply_sqop(self._sq_ham)

            Sig = QC.get_state_deep()

            Homega_lst.append(Sig)

            # Compute S_mn = <φ| U_m^\dagger U_n |φ> and H_mn = <φ| U_m^\dagger H U_n |φ>
            for n in range(len(self._omega_lst)):
                h_mat[m][n] = self._omega_lst[m].vector_dot(Homega_lst[n])
                h_mat[n][m] = np.conj(h_mat[m][n])
                s_mat[m][n] = self._omega_lst[m].vector_dot(self._omega_lst[n])
                s_mat[n][m] = np.conj(s_mat[m][n])

            self._diagonalize_qk_step(
                m + 1,
                s_mat,
                h_mat,
                summary_file=summary_file,
                hermitian_pairs=hermitian_pairs)

        if (self._diagonalize_each_step and self._print_summary_file):
            summary_file.close()

        self._set_qk_resource_counts(
            self._nstates,
            self._qk_trotter_cnot_for_step(self._nstates, hermitian_pairs))


        return s_mat, h_mat
    

    def build_qk_mats_fast_fci(self):
        """Returns matrices S and H needed for the QK algorithm using the Trotterized
        form of the unitary operators U_n = exp(-i n dt H)

        The mathematical operations of this function are unphysical for a quantum
        computer, but efficient for a simulator.

        Returns
        -------
        s_mat : ndarray
            A numpy array containing the elements S_mn = <Phi | Um^dag Un | Phi>.
            _nstates by _nstates

        h_mat : ndarray
            A numpy array containing the elements H_mn = <Phi | Um^dag H Un | Phi>
            _nstates by _nstates
        """

        h_mat = np.zeros((self._nstates,self._nstates), dtype=complex)
        s_mat = np.zeros((self._nstates,self._nstates), dtype=complex)

        # Store these vectors for the aid of MRSQK
        self._omega_lst = []
        Homega_lst = []

        hermitian_pairs = qforte.SQOpPool()

        # this is updated, evolution time is now just 1.0 here
        hermitian_pairs.add_hermitian_pairs(1.0, self._sq_ham)

        QC = qforte.FCIComputer(
                self._nel, 
                self._2_spin, 
                self._norb)
            
        QC.hartree_fock()

        # Initialize FCI timers if not already done
        if self._fci_timers is None:
            self._fci_timers = {}
        if self._fci_time is None:
            self._fci_time = time

        summary_file = self._qk_summary_file()

    
        """In reviewing this there is going to be an inherent ordering probelm. I want to apply 
        based on hermitian paris of SQ operators but the qb hamiltonain has been 'simplified' 
        and looses the exact correspondance to the sq hamiltonain"""
        for m in range(self._nstates):

            if(m>0):
                self._qk_evolve_interval(
                    QC,
                    hermitian_pairs,
                    self._fci_timers,
                    self._fci_time,
                    m - 1,
                    adjoint=False)

            t0 = self._fci_time.time()
            C = QC.get_state_deep()
            if 'get_state_deep' not in self._fci_timers:
                self._fci_timers['get_state_deep'] = 0.0
            self._fci_timers['get_state_deep'] += self._fci_time.time() - t0
         
            self._omega_lst.append(C)

            if(self._apply_ham_as_tensor):
                t0 = self._fci_time.time()
                QC.apply_tensor_spat_012bdy(
                    self._zero_body_energy, 
                    self._mo_oeis, 
                    self._mo_teis, 
                    self._mo_teis_einsum, 
                    self._norb)
                if 'apply_tensor_spat_012bdy' not in self._fci_timers:
                    self._fci_timers['apply_tensor_spat_012bdy'] = 0.0
                self._fci_timers['apply_tensor_spat_012bdy'] += self._fci_time.time() - t0
            else:
                t0 = self._fci_time.time()
                QC.apply_sqop(self._sq_ham)
                if 'apply_sqop' not in self._fci_timers:
                    self._fci_timers['apply_sqop'] = 0.0
                self._fci_timers['apply_sqop'] += self._fci_time.time() - t0

            t0 = self._fci_time.time()
            Sig = QC.get_state_deep()
            if 'get_state_deep' not in self._fci_timers:
                self._fci_timers['get_state_deep'] = 0.0
            self._fci_timers['get_state_deep'] += self._fci_time.time() - t0

            Homega_lst.append(Sig)

            # very important, was missing before!
            QC.set_state(C)

            # Compute S_mn = <φ| U_m^\dagger U_n |φ> and H_mn = <φ| U_m^\dagger H U_n |φ>
            for n in range(len(self._omega_lst)):
                h_mat[m][n] = self._omega_lst[m].vector_dot(Homega_lst[n])
                h_mat[n][m] = np.conj(h_mat[m][n])
                s_mat[m][n] = self._omega_lst[m].vector_dot(self._omega_lst[n])
                s_mat[n][m] = np.conj(s_mat[m][n])

            self._diagonalize_qk_step(
                m + 1,
                s_mat,
                h_mat,
                summary_file=summary_file,
                hermitian_pairs=hermitian_pairs)

        if (self._diagonalize_each_step and self._print_summary_file):
            summary_file.close()

        self._set_qk_resource_counts(
            self._nstates,
            self._qk_trotter_cnot_for_step(self._nstates, hermitian_pairs))


        return s_mat, h_mat

    def build_qk_mats_fast_fqe(self):
        """Returns matrices S and H needed for the QK algorithm using the Trotterized
        form of the unitary operators U_n = exp(-i n dt H)

        The mathematical operations of this function are unphysical for a quantum
        computer, but efficient for a simulator.

        Returns
        -------
        s_mat : ndarray
            A numpy array containing the elements S_mn = <Phi | Um^dag Un | Phi>.
            _nstates by _nstates

        h_mat : ndarray
            A numpy array containing the elements H_mn = <Phi | Um^dag H Un | Phi>
            _nstates by _nstates
        """

        h_mat = np.zeros((self._nstates,self._nstates), dtype=complex)
        s_mat = np.zeros((self._nstates,self._nstates), dtype=complex)

        # Store these vectors for the aid of MRSQK
        self._omega_lst = []
        Homega_lst = []

        hermitian_pairs = qforte.SQOpPool()

        # this is updated, evolution time is now just 1.0 here
        hermitian_pairs.add_hermitian_pairs(1.0, self._sq_ham)

        QC = qforte.FQEComputer(
                self._nel, 
                self._2_spin, 
                self._norb)
            
        QC.hartree_fock()

        # Initialize FQE timers if not already done
        if self._fqe_timers is None:
            self._fqe_timers = {}
        if self._fqe_time is None:
            self._fqe_time = time

        summary_file = self._qk_summary_file()

    
        """In reviewing this there is going to be an inherent ordering probelm. I want to apply 
        based on hermitian paris of SQ operators but the qb hamiltonain has been 'simplified' 
        and looses the exact correspondance to the sq hamiltonain"""
        for m in range(self._nstates):

            if(m>0):
                self._qk_evolve_interval(
                    QC,
                    hermitian_pairs,
                    self._fqe_timers,
                    self._fqe_time,
                    m - 1,
                    adjoint=False)

            t0 = self._fqe_time.time()
            C = QC.get_state_deep()
            if 'get_state_deep' not in self._fqe_timers:
                self._fqe_timers['get_state_deep'] = 0.0
            self._fqe_timers['get_state_deep'] += self._fqe_time.time() - t0
         
            self._omega_lst.append(C)

            if(self._apply_ham_as_tensor):
                t0 = self._fqe_time.time()
                QC.apply_tensor_spat_012bdy(
                    self._zero_body_energy, 
                    self._mo_oeis_np, 
                    self._mo_teis_np, 
                    )
                if 'apply_tensor_spat_012bdy' not in self._fqe_timers:
                    self._fqe_timers['apply_tensor_spat_012bdy'] = 0.0
                self._fqe_timers['apply_tensor_spat_012bdy'] += self._fqe_time.time() - t0
            else:
                t0 = self._fqe_time.time()
                QC.apply_sqop(self._sq_ham)
                if 'apply_sqop' not in self._fqe_timers:
                    self._fqe_timers['apply_sqop'] = 0.0
                self._fqe_timers['apply_sqop'] += self._fqe_time.time() - t0

            t0 = self._fqe_time.time()
            Sig = QC.get_state_deep()
            if 'get_state_deep' not in self._fqe_timers:
                self._fqe_timers['get_state_deep'] = 0.0
            self._fqe_timers['get_state_deep'] += self._fqe_time.time() - t0

            Homega_lst.append(Sig)

            # very important, was missing before!
            QC.set_state(C)

            # Compute S_mn = <φ| U_m^\dagger U_n |φ> and H_mn = <φ| U_m^\dagger H U_n |φ>
            for n in range(len(self._omega_lst)):
                
                h_mat[m][n] = np.vdot(self._omega_lst[m], Homega_lst[n])
                h_mat[n][m] = np.conj(h_mat[m][n])
                
                s_mat[m][n] = np.vdot(self._omega_lst[m], self._omega_lst[n])
                s_mat[n][m] = np.conj(s_mat[m][n])

            self._diagonalize_qk_step(
                m + 1,
                s_mat,
                h_mat,
                summary_file=summary_file,
                hermitian_pairs=hermitian_pairs)

        if (self._diagonalize_each_step and self._print_summary_file):
            summary_file.close()

        self._set_qk_resource_counts(
            self._nstates,
            self._qk_trotter_cnot_for_step(self._nstates, hermitian_pairs))

        return s_mat, h_mat
    
    def build_qk_mats_fast_fci_gpu(self):
        """Returns matrices S and H needed for the QK algorithm using the Trotterized
        form of the unitary operators U_n = exp(-i n dt H)

        The mathematical operations of this function are unphysical for a quantum
        computer, but efficient for a simulator.

        Returns
        -------
        s_mat : ndarray
            A numpy array containing the elements S_mn = <Phi | Um^dag Un | Phi>.
            _nstates by _nstates

        h_mat : ndarray
            A numpy array containing the elements H_mn = <Phi | Um^dag H Un | Phi>
            _nstates by _nstates
        """

        h_mat = np.zeros((self._nstates,self._nstates), dtype=complex)
        s_mat = np.zeros((self._nstates,self._nstates), dtype=complex)

        # Store these vectors for the aid of MRSQK
        self._omega_lst = []
        Homega_lst = []

        # Hermitian pairs already initialized
        # QC already initialized to HF state (on gpu)
        if (hasattr(self, '_gpu_global_initialized') and self._gpu_global_initialized):
            QC: qforte.FCIComputerGPU = self._reusable_qc
            hermitian_pairs: qforte.SQOpPoolGPU = self._hermitian_pairs
        else:
            # Fall back to old implementation if GPU not initialized for some reason
            print("Warning: GPU not initialized, falling back to Non optimized implementation for build_qk_mats_fast_fci_gpu.")
            QC = qforte.FCIComputerGPU(
                self._nel, 
                self._2_spin, 
                self._norb,
                on_gpu=True,
                data_type=self.data_type)
            
            hermitian_pairs = qforte.SQOpPoolGPU(data_type=self.data_type)
            hermitian_pairs.add_hermitian_pairs(1.0, self._sq_ham)

        # Initialize GPU timers if not already done
        if self._gpu_timers is None:
            self._gpu_timers = {}
        if self._gpu_time is None:
            self._gpu_time = time

        summary_file = self._qk_summary_file()

    
        """In reviewing this there is going to be an inherent ordering probelm. I want to apply 
        based on hermitian paris of SQ operators but the qb hamiltonain has been 'simplified' 
        and looses the exact correspondance to the sq hamiltonain"""
        for m in range(self._nstates):

            if(m>0):
                self._qk_evolve_interval(
                    QC,
                    hermitian_pairs,
                    self._gpu_timers,
                    self._gpu_time,
                    m - 1,
                    adjoint=False)

            # Get current state (stays on GPU for efficiency)
            t0 = self._gpu_time.time()
            C = QC.get_state_deep()
            if 'get_state_deep' not in self._gpu_timers:
                self._gpu_timers['get_state_deep'] = 0.0
            self._gpu_timers['get_state_deep'] += self._gpu_time.time() - t0
         
            self._omega_lst.append(C)

            if(self._apply_ham_as_tensor):
                t0 = self._gpu_time.time()
                QC.apply_tensor_spat_012bdy_gpu(
                    self._zero_body_energy, 
                    self._mo_oeis_gpu, 
                    self._mo_teis_gpu, 
                    self._mo_teis_einsum_gpu, 
                    self._norb)
                if 'apply_tensor_spat_012bdy_gpu' not in self._gpu_timers:
                    self._gpu_timers['apply_tensor_spat_012bdy_gpu'] = 0.0
                self._gpu_timers['apply_tensor_spat_012bdy_gpu'] += self._gpu_time.time() - t0
            else:
                t0 = self._gpu_time.time()
                QC.apply_sqop_gpu(self._sq_ham)
                if 'apply_sqop_gpu' not in self._gpu_timers:
                    self._gpu_timers['apply_sqop_gpu'] = 0.0
                self._gpu_timers['apply_sqop_gpu'] += self._gpu_time.time() - t0

            # Get H|ψ> state (stays on GPU)
            t0 = self._gpu_time.time()
            Sig = QC.get_state_deep()
            if 'get_state_deep' not in self._gpu_timers:
                self._gpu_timers['get_state_deep'] = 0.0
            self._gpu_timers['get_state_deep'] += self._gpu_time.time() - t0

            Homega_lst.append(Sig)

            # Restore state using GPU-to-GPU copy (critical: avoids CPU transfer)
            # Note: C is on GPU, so we must use set_state_gpu() not set_state()
            t0 = self._gpu_time.time()
            QC.set_state_gpu(C)
            if 'set_state_gpu' not in self._gpu_timers:
                self._gpu_timers['set_state_gpu'] = 0.0
            self._gpu_timers['set_state_gpu'] += self._gpu_time.time() - t0

            # Compute S_mn = <φ| U_m^\dagger U_n |φ> and H_mn = <φ| U_m^\dagger H U_n |φ>
            for n in range(len(self._omega_lst)):
                h_mat[m][n] = self._omega_lst[m].vector_dot(Homega_lst[n])
                h_mat[n][m] = np.conj(h_mat[m][n])
                s_mat[m][n] = self._omega_lst[m].vector_dot(self._omega_lst[n])
                s_mat[n][m] = np.conj(s_mat[m][n])

            self._diagonalize_qk_step(
                m + 1,
                s_mat,
                h_mat,
                summary_file=summary_file,
                hermitian_pairs=hermitian_pairs)

        if (self._diagonalize_each_step and self._print_summary_file):
            summary_file.close()

        self._set_qk_resource_counts(
            self._nstates,
            self._qk_trotter_cnot_for_step(self._nstates, hermitian_pairs))

        return s_mat, h_mat
    
    def build_qk_mats_fast_cusv(self):
        """Returns matrices S and H needed for the QK algorithm using the Trotterized
        form of the unitary operators U_n = exp(-i n dt H)

        The mathematical operations of this function are unphysical for a quantum
        computer, but efficient for a simulator.

        Returns
        -------
        s_mat : ndarray
            A numpy array containing the elements S_mn = <Phi | Um^dag Un | Phi>.
            _nstates by _nstates

        h_mat : ndarray
            A numpy array containing the elements H_mn = <Phi | Um^dag H Un | Phi>
            _nstates by _nstates
        """

        h_mat = np.zeros((self._nstates,self._nstates), dtype=complex)
        s_mat = np.zeros((self._nstates,self._nstates), dtype=complex)

        # Store these vectors for the aid of MRSQK
        self._omega_lst = []
        Homega_lst = []

        hermitian_pairs = qforte.SQOpPool()

        # this is updated, evolution time is now just 1.0 here
        hermitian_pairs.add_hermitian_pairs(1.0, self._sq_ham)

        QC = qforte.CUSVComputer(
                self._nel, 
                self._2_spin, 
                self._norb)
            
        QC.hartree_fock()

        # Initialize CUSV timers if not already done
        if self._cusv_timers is None:
            self._cusv_timers = {}
        if self._cusv_time is None:
            self._cusv_time = time

        summary_file = self._qk_summary_file()

    
        """In reviewing this there is going to be an inherent ordering probelm. I want to apply 
        based on hermitian paris of SQ operators but the qb hamiltonain has been 'simplified' 
        and looses the exact correspondance to the sq hamiltonain"""
        for m in range(self._nstates):

            if(m>0):
                self._qk_evolve_interval(
                    QC,
                    hermitian_pairs,
                    self._cusv_timers,
                    self._cusv_time,
                    m - 1,
                    adjoint=False)

            t0 = self._cusv_time.time()
            C = QC.get_state_deep()
            if 'get_state_deep' not in self._cusv_timers:
                self._cusv_timers['get_state_deep'] = 0.0
            self._cusv_timers['get_state_deep'] += self._cusv_time.time() - t0
         
            self._omega_lst.append(C)

            t0 = self._cusv_time.time()
            QC.apply_sqop(self._sq_ham)
            if 'apply_sqop' not in self._cusv_timers:
                self._cusv_timers['apply_sqop'] = 0.0
            self._cusv_timers['apply_sqop'] += self._cusv_time.time() - t0

            t0 = self._cusv_time.time()
            Sig = QC.get_state_deep()
            if 'get_state_deep' not in self._cusv_timers:
                self._cusv_timers['get_state_deep'] = 0.0
            self._cusv_timers['get_state_deep'] += self._cusv_time.time() - t0

            Homega_lst.append(Sig)

            # very important, was missing before!
            QC.set_state(C)

            # Compute S_mn = <φ| U_m^\dagger U_n |φ> and H_mn = <φ| U_m^\dagger H U_n |φ>
            for n in range(len(self._omega_lst)):
                
                h_mat[m][n] = np.vdot(self._omega_lst[m], Homega_lst[n])
                h_mat[n][m] = np.conj(h_mat[m][n])
                
                s_mat[m][n] = np.vdot(self._omega_lst[m], self._omega_lst[n])
                s_mat[n][m] = np.conj(s_mat[m][n])

            self._diagonalize_qk_step(
                m + 1,
                s_mat,
                h_mat,
                summary_file=summary_file,
                hermitian_pairs=hermitian_pairs)

        if (self._diagonalize_each_step and self._print_summary_file):
            summary_file.close()

        self._set_qk_resource_counts(
            self._nstates,
            self._qk_trotter_cnot_for_step(self._nstates, hermitian_pairs))

        return s_mat, h_mat

    def build_qk_mats_realistic(self):
        if not self._qk_uses_legacy_uniform_grid:
            raise NotImplementedError(
                "SRQK realistic matrix-element construction does not yet "
                "support nonuniform QK time grids/Tmax heuristics."
            )

        h_mat = np.zeros((self._nstates,self._nstates), dtype=complex)
        s_mat = np.zeros((self._nstates,self._nstates), dtype=complex)

        for p in range(self._nstates):
            for q in range(p, self._nstates):
                h_mat[p][q] = self.matrix_element(p, q, use_op=True)
                h_mat[q][p] = np.conj(h_mat[p][q])
                s_mat[p][q] = self.matrix_element(p, q, use_op=False)
                s_mat[q][p] = np.conj(s_mat[p][q])

        return s_mat, h_mat


    #TODO depricate this function
    def matrix_element(self, m, n, use_op=False):
        """Returns a single matrix element M_mn based on the evolution of
        two unitary operators Um = exp(-i * m * dt * H) and Un = exp(-i * n * dt * H)
        on a reference state |Phi_o>, (optionally) with respect to an operator A.
        Specifically, M_mn is given by <Phi_o| Um^dag Un | Phi_o> or
        (optionally if A is specified) <Phi_o| Um^dag A Un | Phi_o>.

        Arguments
        ---------

        m : int
            The number of time steps for the Um evolution.

        n : int
            The number of time steps for the Un evolution.

        Returns
        -------
        value : complex
            The outcome of measuring <X> and <Y> to determine <2*sigma_+>,
            ultimately the value of the matrix elemet.

        """
        value = 0.0
        ancilla_idx = self._nqb
        Uk = qforte.Circuit()
        temp_op1 = qforte.QubitOperator()
        # TODO (opt): move to C side.
        for t in self._qb_ham.terms():
            c, op = t
            phase = -1.0j * n * self._dt * c
            temp_op1.add(phase, op)

        expn_op1, phase1 = trotterize_w_cRz(temp_op1,
                                            ancilla_idx,
                                            trotter_number=self._trotter_number)

        for gate in expn_op1.gates():
            Uk.add(gate)

        Ub = qforte.Circuit()

        temp_op2 = qforte.QubitOperator()
        for t in self._qb_ham.terms():
            c, op = t
            phase = -1.0j * m * self._dt * c
            temp_op2.add(phase, op)

        expn_op2, phase2 = trotterize_w_cRz(temp_op2,
                                            ancilla_idx,
                                            trotter_number=self._trotter_number,
                                            Use_open_cRz=False)

        for gate in expn_op2.gates():
            Ub.add(gate)

        if not use_op:
            # TODO (opt): use Uprep
            cir = qforte.Circuit()
            for j in range(self._nqb):
                if self._ref[j] == 1:
                    cir.add(qforte.gate('X', j, j))

            cir.add(qforte.gate('H', ancilla_idx, ancilla_idx))

            cir.add(Uk)

            cir.add(qforte.gate('X', ancilla_idx, ancilla_idx))
            cir.add(Ub)
            cir.add(qforte.gate('X', ancilla_idx, ancilla_idx))

            X_op = qforte.QubitOperator()
            x_circ = qforte.Circuit()
            Y_op = qforte.QubitOperator()
            y_circ = qforte.Circuit()

            x_circ.add(qforte.gate('X', ancilla_idx, ancilla_idx))
            y_circ.add(qforte.gate('Y', ancilla_idx, ancilla_idx))

            X_op.add(1.0, x_circ)
            Y_op.add(1.0, y_circ)

            X_exp = qforte.Experiment(self._nqb+1, cir, X_op, 100)
            Y_exp = qforte.Experiment(self._nqb+1, cir, Y_op, 100)

            params = [1.0]
            x_value = X_exp.perfect_experimental_avg(params)
            y_value = Y_exp.perfect_experimental_avg(params)

            value = (x_value + 1.0j * y_value) * phase1 * np.conj(phase2)


        else:
            value = 0.0
            for t in self._qb_ham.terms():
                c, V_l = t

                # TODO (opt):
                cV_l = qforte.Circuit()
                for gate in V_l.gates():
                    gate_str = gate.gate_id()
                    target = gate.target()
                    control_gate_str = 'c' + gate_str
                    cV_l.add(qforte.gate(control_gate_str, target, ancilla_idx))

                cir = qforte.Circuit()
                # TODO (opt): use Uprep
                for j in range(self._nqb):
                    if self._ref[j] == 1:
                        cir.add(qforte.gate('X', j, j))

                cir.add(qforte.gate('H', ancilla_idx, ancilla_idx))

                cir.add(Uk)
                cir.add(cV_l)

                cir.add(qforte.gate('X', ancilla_idx, ancilla_idx))
                cir.add(Ub)
                cir.add(qforte.gate('X', ancilla_idx, ancilla_idx))

                X_op = qforte.QubitOperator()
                x_circ = qforte.Circuit()
                Y_op = qforte.QubitOperator()
                y_circ = qforte.Circuit()

                x_circ.add(qforte.gate('X', ancilla_idx, ancilla_idx))
                y_circ.add(qforte.gate('Y', ancilla_idx, ancilla_idx))

                X_op.add(1.0, x_circ)
                Y_op.add(1.0, y_circ)

                X_exp = qforte.Experiment(self._nqb+1, cir, X_op, 100)
                Y_exp = qforte.Experiment(self._nqb+1, cir, Y_op, 100)

                # TODO (cleanup): Remove params as required arg (Nick)
                params = [1.0]
                x_value = X_exp.perfect_experimental_avg(params)
                y_value = Y_exp.perfect_experimental_avg(params)

                element = (x_value + 1.0j * y_value) * phase1 * np.conj(phase2)
                value += c * element

        return value

    def initialize_gpu_global(self):
        """Initialize reusable GPU pool and computers for optimization.
        This should be called after initialize_ansatz() and before solve().
        Precomputes index arrays and creates persistent GPU computers to avoid:
        1. Pool recreation overhead (major optimization via precomputation)
        2. GPU memory allocation/deallocation overhead each iteration
        """

        if self._computer_type != 'fci_gpu':
            return
        
        # Create reusable GPU pool with initial coefficients (zeros)
        self._hermitian_pairs = qforte.SQOpPoolGPU(data_type=self.data_type)
        
        self._hermitian_pairs.add_hermitian_pairs(1.0, self._sq_ham)
        
        # Create reusable GPU computers for gradient evaluation
        # qc_psi: builds evolved state
        # qc_sig: builds H|psi> for gradient computation
        self._reusable_qc = qforte.FCIComputerGPU(
            self._nel, 
            self._2_spin, 
            self._norb,
            on_gpu=True,
            data_type=self.data_type)
        
        self._reusable_qc.hartree_fock_gpu()
        
        # Precompute index arrays for fast evolution
        # This sets device_vecs_populated_ = true and precomputes all index mappings
        self._reusable_qc.populate_index_arrays_for_pool_evo(self._hermitian_pairs)

        # Set flag to indicate GPU global initialization is done
        self._gpu_global_initialized = True
        
        if self._verbose:
            print('\n==> GPU optimization infrastructure initialized')
            print(f'    Pool size: {len(self._tops)} operators')
            print(f'    Index arrays precomputed for fast evolution')
            print(f'    Reusable GPU computers created (avoids allocation overhead)')
