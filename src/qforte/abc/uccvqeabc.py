"""
UCC-VQE base classes
====================================
The abstract base classes inheritied by any variational quantum eigensolver (VQE)
variant that utilizes a unitary coupled cluster (UCC) type ansatz.
"""

import qforte as qf
import copy
from abc import abstractmethod
from qforte.abc.vqeabc import VQE
from qforte.abc.ansatz import UCC

from qforte.experiment import *
from qforte.utils.transforms import *
from qforte.utils.state_prep import ref_to_basis_idx
from qforte.utils.trotterization import trotterize
from qforte.utils.compact_excitation_circuits import compact_excitation_circuit

import numpy as np

class UCCVQE(VQE, UCC):
    """The abstract base class inheritied by any algorithm that seeks to find
    eigenstates by variational minimization of the Energy

    .. math::
        E(\mathbf{t}) = \langle \Phi_0 | \hat{U}^\dagger(\mathbf{\mathbf{t}}) \hat{H} \hat{U}(\mathbf{\mathbf{t}}) | \Phi_0 \\rangle

    using a disentagled UCC type ansatz

    .. math::
        \hat{U}(\mathbf{t}) = \prod_\mu e^{t_\mu (\hat{\\tau}_\mu - \hat{\\tau}_\mu^\dagger)},

    were :math:`\hat{\\tau}_\mu` is a Fermionic excitation operator and
    :math:`t_\mu` is a cluster amplitude.

    Attributes
    ----------

    _pool_type : string or SQOpPool
        Specifies the kinds of tamplitudes allowed in the UCCN-VQE
        parameterization. If an SQOpPool is supplied, that is used as the
        operator pool. The following strings are allowed:
            SA_SD: At most two orbital excitations. Assumes a singlet wavefunction and closed-shell Slater determinant
                   reducing the number of amplitudes.
            SD: At most two orbital excitations.
            SDT: At most three orbital excitations.
            SDTQ: At most four orbital excitations.
            SDTQP: At most five orbital excitations.
            SDTQPH: At most six orbital excitations.
            GSD: At most two excitations, from any orbital to any orbital.
            GSDx: Deprecated compatibility alias for GSD with
                  general_ex_pool_order="particle_hole_first".
            k-UpCCGSDx: Deprecated compatibility alias for k-UpCCGSD with
                        general_ex_pool_order="particle_hole_first".

    _prev_energy : float
        The energy from the previous iteration.

    _curr_energy : float
        The energy from the current iteration.

    _curr_grad_norm : float
        The current norm of the gradient

    _Nm : int
        A list containing the number of pauli terms in each Jordan-Wigner
        transformed excitaiton/de-excitaion operator in the pool.

    _use_analytic_grad : bool
        Whether or not to use an analytic function for the gradient to pass to
        the optimizer. If false, the optimizer will use self-generated approximate
        gradients from finite differences (if BFGS algorithm is used).

    """

    @abstractmethod
    def get_num_ham_measurements(self):
        pass

    @abstractmethod
    def get_num_commut_measurements(self):
        pass

    def fill_commutator_pool(self):
        print('\n\n==> Building commutator pool for gradient measurement.')
        self._commutator_pool = self._pool_obj.get_qubit_op_pool()
        self._commutator_pool.join_as_commutator(self._qb_ham)
        print('==> Commutator pool construction complete.')

    def measure_operators(self, operators, Ucirc, idxs=[]):
        """
        Parameters
        ----------
        operators : QubitOpPool
            All operators to be measured

        Ucirc : Circuit
            The state preparation circuit.

        idxs : list of int
            The indices of select operators in the pool of operators. If provided, only these
            operators will be measured.

        """

        if self._fast:
            myQC = qforte.Computer(self._nqb)
            myQC.apply_circuit(Ucirc)
            if not idxs:
                grads = myQC.direct_oppl_exp_val(operators)
            else:
                grads = myQC.direct_idxd_oppl_exp_val(operators, idxs)

        else:
            raise NotImplementedError("Must have self._fast to measure an operator.")

        np.testing.assert_allclose(np.imag(grads), np.zeros_like(grads), atol=1e-7)

        return np.real(grads)

    def measure_gradient(self, params=None, return_energy=False):
        if(self._computer_type == 'fock'):
            return self.measure_gradient_fock(params, return_energy=return_energy)
        elif(self._computer_type == 'fci'):
            return self.measure_gradient_fci(params, return_energy=return_energy)
        elif(self._computer_type == 'fqe'):
            return self.measure_gradient_fqe(params, return_energy=return_energy)
        elif(self._computer_type == 'fci_gpu'):
            return self.measure_gradient_fci_gpu(params, return_energy=return_energy)
        elif(self._computer_type == 'cusv'):
            return self.measure_gradient_cusv(params, return_energy=return_energy)
        else:
            raise ValueError(f"{self._computer_type} is an unrecognized computer type.") 

    def _copy_pool_generator_sqop(self, mu):
        """Return a scaled copy of the selected pool generator for amplitude mu."""
        return self._copy_pool_generator_sqop_from_top(self._tops[mu])

    def _copy_pool_generator_sqop_from_top(self, top):
        """Return a scaled copy of a pool generator by pool index."""
        Kmu = qf.SQOperator()
        Kmu.add_op(self._pool_obj[top][1])
        Kmu.mult_coeffs(self._pool_obj[top][0])
        return Kmu

    def _make_reusable_ucc_signature(self):
        """Identify the ansatz/backend shape cached by reusable objects."""
        return (
            self._computer_type,
            id(self._pool_obj),
            tuple(self._tops),
            self._nqb,
            self._nel,
            self._2_spin,
            self._norb,
            getattr(self, "_qubit_excitations", False),
        )

    def initialize_reusable_ucc_objects(self):
        """Create reusable CPU backend objects for repeated UCC evaluations."""
        if self._computer_type == "fci_gpu":
            return

        if self._computer_type not in {"fock", "fci", "fqe", "cusv"}:
            return

        self._reusable_ucc_cache_signature = self._make_reusable_ucc_signature()

        if self._computer_type == "fock":
            self._reusable_qc_psi = qf.Computer(self._nqb)
            self._reusable_qc_sig = qf.Computer(self._nqb)
            self._reusable_fock_zero_state = copy.deepcopy(
                self._reusable_qc_psi.get_coeff_vec()
            )
            self._reusable_jw_generators = [
                self._scaled_jw_generator_for_top(top) for top in self._tops
            ]
            self._reusable_full_jw_generators = [
                self._scaled_jw_generator_for_top(top)
                for top in range(len(self._pool_obj))
            ]
            return

        computer_cls = {
            "fci": qf.FCIComputer,
            "fqe": qf.FQEComputer,
            "cusv": qf.CUSVComputer,
        }[self._computer_type]

        self._reusable_pool = qf.SQOpPool()
        for tamp, top in zip(self._tamps, self._tops):
            self._reusable_pool.add(tamp, self._pool_obj[top][1])

        self._reusable_sq_generators = [
            self._copy_pool_generator_sqop_from_top(top) for top in self._tops
        ]
        self._reusable_full_sq_generators = [
            self._copy_pool_generator_sqop_from_top(top)
            for top in range(len(self._pool_obj))
        ]
        self._reusable_qc_psi = computer_cls(self._nel, self._2_spin, self._norb)
        self._reusable_qc_sig = computer_cls(self._nel, self._2_spin, self._norb)

        if self._computer_type == "fci":
            self._reusable_qc_chi = computer_cls(self._nel, self._2_spin, self._norb)
            self._reusable_qc_hchi = computer_cls(self._nel, self._2_spin, self._norb)

    def _ensure_reusable_ucc_objects(self):
        """Rebuild reusable objects if the active ansatz shape changed."""
        if self._computer_type == "fci_gpu":
            return (
                hasattr(self, "_reusable_pool_gpu")
                and hasattr(self, "_reusable_qc_psi")
                and hasattr(self, "_reusable_qc_sig")
            )

        if self._computer_type not in {"fock", "fci", "fqe", "cusv"}:
            return False

        signature = self._make_reusable_ucc_signature()
        if getattr(self, "_reusable_ucc_cache_signature", None) != signature:
            self.initialize_reusable_ucc_objects()
        return hasattr(self, "_reusable_qc_psi") and hasattr(self, "_reusable_qc_sig")

    def _active_reusable_pool(self, params=None):
        """Return the reusable selected SQOpPool with current amplitudes."""
        if not self._ensure_reusable_ucc_objects():
            return None

        if not hasattr(self, "_reusable_pool"):
            return None

        amplitudes = self._tamps if params is None else params
        self._reusable_pool.set_coeffs([complex(tamp) for tamp in amplitudes])
        return self._reusable_pool

    def _scaled_jw_generator_for_top(self, top):
        """Return a cached-form Fock-space generator for a pool index."""
        Kmu = self._pool_obj[top][1].jw_transform(self._qubit_excitations)
        Kmu.mult_coeffs(self._pool_obj[top][0])
        return Kmu

    def _selected_sq_generator(self, mu):
        if self._ensure_reusable_ucc_objects() and hasattr(self, "_reusable_sq_generators"):
            return self._reusable_sq_generators[mu]
        return self._copy_pool_generator_sqop(mu)

    def _full_sq_generator(self, top):
        if self._ensure_reusable_ucc_objects() and hasattr(self, "_reusable_full_sq_generators"):
            return self._reusable_full_sq_generators[top]
        return self._copy_pool_generator_sqop_from_top(top)

    def _selected_jw_generator(self, mu):
        if self._ensure_reusable_ucc_objects() and hasattr(self, "_reusable_jw_generators"):
            return self._reusable_jw_generators[mu]
        return self._scaled_jw_generator_for_top(self._tops[mu])

    def _full_jw_generator(self, top):
        if self._ensure_reusable_ucc_objects() and hasattr(self, "_reusable_full_jw_generators"):
            return self._reusable_full_jw_generators[top]
        return self._scaled_jw_generator_for_top(top)

    def _apply_hamiltonian_fci(self, qc):
        """Apply the molecular Hamiltonian to an FCIComputer, respecting timers."""
        if(self._apply_ham_as_tensor):
            t0 = self._fci_time.time()
            qc.apply_tensor_spat_012bdy(
                self._zero_body_energy,
                self._mo_oeis,
                self._mo_teis,
                self._mo_teis_einsum,
                self._norb)
            self._fci_timers['apply_tensor_spat_012bdy'] += self._fci_time.time() - t0
        else:
            t0 = self._fci_time.time()
            qc.apply_sqop(self._sq_ham)
            self._fci_timers['apply_sqop'] += self._fci_time.time() - t0

    def build_final_fci_state(self, params=None):
        """Build an FCIComputer holding the current final UCC state.

        This is used for final-state diagnostics such as <S^2> and NOONs.  It mirrors the
        FCI/tUCC state construction used in the analytical-gradient routines.
        """
        if getattr(self, "_computer_type", None) == "fci_gpu":
            raise NotImplementedError(
                "Final-state <S^2>/NOON diagnostics are not yet implemented for "
                'computer_type="fci_gpu". These diagnostics need FCIComputerGPU '
                "implementations and will be added in a future PR."
            )
        if not getattr(self, "_ref_from_hf", True):
            raise ValueError(
                "Final-state FCI diagnostics currently require an HF reference state."
            )

        if params is None:
            params = self._tamps

        vqc_ops = qf.SQOpPool()
        for tamp, top in zip(params, self._tops):
            vqc_ops.add(tamp, self._pool_obj[top][1])

        qc = qf.FCIComputer(self._nel, self._2_spin, self._norb)
        qc.hartree_fock()
        qc.evolve_pool_trotter_basic(vqc_ops, antiherm=True, adjoint=False)
        return qc

    def build_final_fock_state(self, params=None):
        """Build a Fock-space Computer holding the current final UCC state."""
        qc = qf.Computer(self._nqb)
        qc.apply_circuit(self.build_Uvqc(amplitudes=params))
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
                    sqop = qf.SQOperator()
                    sqop.add(1.0, [2 * p + spin], [2 * q + spin])
                    element += qc.direct_op_exp_val(sqop.jw_transform())
                gamma[p, q] = element / norm_sq

        gamma = 0.5 * (gamma + gamma.conj().T)
        noons = np.linalg.eigvalsh(np.real(gamma))
        return [float(noon) for noon in noons[::-1]]

    def compute_final_fock_diagnostics(self, params=None):
        """Compute and store final-state diagnostics using the Fock backend."""
        try:
            qc = self.build_final_fock_state(params=params)
        except Exception as exc:
            err = str(exc)
            self._spin_squared = None
            self._spin_squared_error = err
            self._natural_orbital_occupation_numbers = None
            self._noons_error = err
            return

        try:
            s_squared = qf.total_spin_squared(self._nqb)
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

    def compute_final_fci_diagnostics(self, params=None):
        """Compute and store final-state FCI diagnostics when possible."""
        if getattr(self, "_computer_type", None) == "fock":
            self.compute_final_fock_diagnostics(params=params)
            return

        try:
            qc = self.build_final_fci_state(params=params)
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

    def compute_final_spin_squared_expectation(self, params=None):
        """Compute and store <S^2> for the final VQE state when possible."""
        self.compute_final_fci_diagnostics(params=params)
        return self._spin_squared

    def compute_final_noons(self, params=None):
        """Compute and store final natural orbital occupation numbers."""
        self.compute_final_fci_diagnostics(params=params)
        return self._natural_orbital_occupation_numbers

    def final_spin_squared_summary_string(self):
        """Return a printable final <S^2> summary value."""
        if not hasattr(self, "_spin_squared"):
            self.compute_final_spin_squared_expectation()
        if self._spin_squared is None:
            return f"N/A ({getattr(self, '_spin_squared_error', 'unavailable')})"
        return f"{self._spin_squared:12.10f}"

    def final_noons_summary_string(self):
        """Return a printable final NOON summary value."""
        if not hasattr(self, "_natural_orbital_occupation_numbers"):
            self.compute_final_noons()
        noons = self._natural_orbital_occupation_numbers
        if noons is None:
            return f"N/A ({getattr(self, '_noons_error', 'unavailable')})"
        return "[" + ", ".join(f"{noon:.8f}" for noon in noons) + "]"

    def final_noons_summary_table(self):
        """Return a readable indexed table of final natural occupations."""
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

    def _active_hdiag_method(self):
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

    def _active_hdiag_fd_step(self):
        """Return the finite-difference step for the active qforte optimizer."""
        optimizer_name = getattr(self, "_optimizer", "").lower()
        if optimizer_name == "bfgs_qf":
            return getattr(
                self,
                "_bfgs_qf_hdiag_fd_step",
                getattr(self, "_lbfgs_qf_hdiag_fd_step", 1.0e-4),
            )
        return getattr(
            self,
            "_lbfgs_qf_hdiag_fd_step",
            getattr(self, "_bfgs_qf_hdiag_fd_step", 1.0e-4),
        )

    def measure_gradient_fock(self, params=None, return_energy=False):
        """ Returns the disentangled (factorized) UCC gradient, using a
        recursive approach.

        Parameters
        ----------
        params : list of floats
            The variational parameters which characterize _Uvqc.
        """

        if not self._fast:
            raise ValueError("self._fast must be True for gradient measurement.")

        M = len(self._tamps)

        grads = np.zeros(M)

        # print(f"\n Grads before: {grads}")

        if params is None:
            Utot = self.build_Uvqc()
        else:
            Utot = self.build_Uvqc(params)

        if self._ensure_reusable_ucc_objects():
            qc_psi = self._reusable_qc_psi
            qc_sig = self._reusable_qc_sig
            qc_psi.set_coeff_vec(copy.deepcopy(self._reusable_fock_zero_state))
        else:
            qc_psi = qforte.Computer(self._nqb) # build | sig_N > according ADAPT-VQE analytical grad section
            qc_sig = qforte.Computer(self._nqb) # build | psi_N > according ADAPT-VQE analytical grad section

        qc_psi.apply_circuit(Utot)
        psi_i = copy.deepcopy(qc_psi.get_coeff_vec())
        qc_sig.set_coeff_vec(copy.deepcopy(psi_i)) # not sure if copy is faster or reapplication of state
        qc_sig.apply_operator(self._qb_ham)
        energy = None
        if return_energy:
            energy = np.real(np.vdot(psi_i, qc_sig.get_coeff_vec()))

        mu = M-1

        # find <sing_N | K_N | psi_N>
        Kmu_prev = self._selected_jw_generator(mu)

        qc_psi.apply_operator(Kmu_prev)
        grads[mu] = 2.0 * np.real(np.vdot(qc_sig.get_coeff_vec(), qc_psi.get_coeff_vec()))

        #reset Kmu_prev |psi_i> -> |psi_i>
        qc_psi.set_coeff_vec(copy.deepcopy(psi_i))

        for mu in reversed(range(M-1)):

            # mu => N-1 => M-2
            # mu+1 => N => M-1
            # Kmu => KN-1
            # Kmu_prev => KN

            if params is None:
                tamp = self._tamps[mu+1]
            else:
                tamp = params[mu+1]

            Kmu = self._selected_jw_generator(mu)

            if self._compact_excitations:
                Umu = qf.Circuit()
                # The minus sign is dictated by the recursive algorithm used to compute the analytic gradient
                # (see original ADAPT-VQE paper)
                Umu.add(compact_excitation_circuit(-tamp * self._pool_obj[self._tops[mu + 1]][1].terms()[1][0],
                                                           self._pool_obj[self._tops[mu + 1]][1].terms()[1][1],
                                                           self._pool_obj[self._tops[mu + 1]][1].terms()[1][2],
                                                           self._qubit_excitations))
            else:
                # The minus sign is dictated by the recursive algorithm used to compute the analytic gradient
                # (see original ADAPT-VQE paper)
                Umu, pmu = trotterize(Kmu_prev, factor=-tamp, trotter_number=self._trotter_number)

                if (pmu != 1.0 + 0.0j):
                    raise ValueError("Encountered phase change, phase not equal to (1.0 + 0.0i)")

            qc_sig.apply_circuit(Umu)
            qc_psi.apply_circuit(Umu)
            psi_i = copy.deepcopy(qc_psi.get_coeff_vec())

            qc_psi.apply_operator(Kmu)
            grads[mu] = 2.0 * np.real(np.vdot(qc_sig.get_coeff_vec(), qc_psi.get_coeff_vec()))

            #reset Kmu |psi_i> -> |psi_i>
            qc_psi.set_coeff_vec(copy.deepcopy(psi_i))
            Kmu_prev = Kmu

        np.testing.assert_allclose(np.imag(grads), np.zeros_like(grads), atol=1e-7)

        # print(f"\n Grads after: {grads}")

        if return_energy:
            return energy, grads
        return grads
    
    # TODO(Nick): think about optemization here, probably should have its own c++ function
    def measure_gradient_fci(self, params=None, return_energy=False):
        """ Returns the disentangled (factorized) UCC gradient, using a
        recursive approach.

        Parameters
        ----------
        params : list of floats
            The variational parameters which characterize _Uvqc.
        """

        if not self._fast:
            raise ValueError("self._fast must be True for gradient measurement.")
        
        if(self._pool_type == 'sa_SD'):
            raise ValueError('Must use single term particle-hole nbody operators for residual calculation')
        
        if not self._ref_from_hf:
            raise ValueError('get_residual_vector_fci_comp only compatible with hf reference at this time.')

        M = len(self._tamps)
        grads = np.zeros(M)
        vqc_ops = self._active_reusable_pool(params)

        if vqc_ops is None:
            vqc_ops = qforte.SQOpPool()
            if params is None:
                for tamp, top in zip(self._tamps, self._tops):
                    vqc_ops.add(tamp, self._pool_obj[top][1])
            else:
                for tamp, top in zip(params, self._tops):
                    vqc_ops.add(tamp, self._pool_obj[top][1])
            qc_psi = qforte.FCIComputer(
                self._nel,
                self._2_spin,
                self._norb)
            qc_sig = qforte.FCIComputer(
                self._nel,
                self._2_spin,
                self._norb)
        else:
            qc_psi = self._reusable_qc_psi
            qc_sig = self._reusable_qc_sig
        
        t0 = self._fci_time.time()
        qc_psi.hartree_fock()
        self._fci_timers['hartree_fock'] += self._fci_time.time() - t0
        
        # qc_psi.apply_circuit(Utot)
        t0 = self._fci_time.time()
        qc_psi.evolve_pool_trotter_basic(
            vqc_ops,
            antiherm=True,
            adjoint=False)
        self._fci_timers['evolve_pool_trotter_basic'] += self._fci_time.time() - t0

        t0 = self._fci_time.time()
        psi_i = qc_psi.get_state_deep()
        self._fci_timers['get_state_deep'] += self._fci_time.time() - t0

        # not sure if copy is faster or reapplication of state
        t0 = self._fci_time.time()
        qc_sig.set_state(psi_i)
        self._fci_timers['set_state'] += self._fci_time.time() - t0

        self._apply_hamiltonian_fci(qc_sig)
        energy = None
        if return_energy:
            t0 = self._fci_time.time()
            energy = np.real(qc_psi.get_state().vector_dot(qc_sig.get_state()))
            self._fci_timers['vector_dot'] += self._fci_time.time() - t0

        mu = M-1

        # find <sing_N | K_N | psi_N>
        Kmu_prev = self._selected_sq_generator(mu)

        t0 = self._fci_time.time()
        qc_psi.apply_sqop(Kmu_prev)
        self._fci_timers['apply_sqop'] += self._fci_time.time() - t0
        
        t0 = self._fci_time.time()
        grads[mu] = 2.0 * np.real(
            qc_sig.get_state().vector_dot(qc_psi.get_state())
            )
        self._fci_timers['vector_dot'] += self._fci_time.time() - t0

        #reset Kmu_prev |psi_i> -> |psi_i>
        t0 = self._fci_time.time()
        qc_psi.set_state(psi_i)
        self._fci_timers['set_state'] += self._fci_time.time() - t0

        for mu in reversed(range(M-1)):

            # mu => N-1 => M-2
            # mu+1 => N => M-1
            # Kmu => KN-1
            # Kmu_prev => KN

            if params is None:
                tamp = self._tamps[mu+1]
            else:
                tamp = params[mu+1]

            Kmu = self._selected_sq_generator(mu)

            # The minus sign is dictated by the recursive algorithm used to compute the analytic gradient
            # (see original ADAPT-VQE paper)
            t0 = self._fci_time.time()
            qc_psi.apply_sqop_evolution(
                -1.0*tamp,
                Kmu_prev,
                antiherm=True,
                adjoint=False)
            self._fci_timers['apply_sqop_evolution'] += self._fci_time.time() - t0
            
            t0 = self._fci_time.time()
            qc_sig.apply_sqop_evolution(
                -1.0*tamp,
                Kmu_prev,
                antiherm=True,
                adjoint=False)
            self._fci_timers['apply_sqop_evolution'] += self._fci_time.time() - t0

            t0 = self._fci_time.time()
            psi_i = qc_psi.get_state_deep()
            self._fci_timers['get_state_deep'] += self._fci_time.time() - t0

            t0 = self._fci_time.time()
            qc_psi.apply_sqop(Kmu)
            self._fci_timers['apply_sqop'] += self._fci_time.time() - t0
            
            t0 = self._fci_time.time()
            grads[mu] = 2.0 * np.real(
                qc_sig.get_state().vector_dot(qc_psi.get_state())
                )
            self._fci_timers['vector_dot'] += self._fci_time.time() - t0

            #reset Kmu |psi_i> -> |psi_i>
            t0 = self._fci_time.time()
            qc_psi.set_state(psi_i)
            self._fci_timers['set_state'] += self._fci_time.time() - t0
            Kmu_prev = Kmu

        np.testing.assert_allclose(np.imag(grads), np.zeros_like(grads), atol=1e-7)
        
        if return_energy:
            return energy, grads
        return grads

    def measure_gradient_fci_finite_difference(
        self,
        params=None,
        step=1.0e-4,
        stencil="five-point",
    ):
        """Compute finite-difference tUCC energy gradient.

        Parameters
        ----------
        params : list or np.ndarray
            Variational parameters. If None, self._tamps is used.
        step : float
            Finite-difference displacement.
        stencil : str
            "central", "forward", or "five-point".

        Returns
        -------
        np.ndarray
            Finite-difference gradient dE/dtheta_mu.
        """

        if params is None:
            params = np.array(self._tamps, dtype=float)
        else:
            params = np.array(params, dtype=float)

        M = len(params)
        grads_fd = np.zeros(M)

        for mu in range(M):
            if stencil == "central":
                params_p = params.copy()
                params_m = params.copy()

                params_p[mu] += step
                params_m[mu] -= step

                e_p = self.energy_feval(params_p)
                e_m = self.energy_feval(params_m)

                grads_fd[mu] = (e_p - e_m) / (2.0 * step)

            elif stencil == "forward":
                params_p = params.copy()

                params_p[mu] += step

                e_0 = self.energy_feval(params)
                e_p = self.energy_feval(params_p)

                grads_fd[mu] = (e_p - e_0) / step

            elif stencil == "five-point":
                params_p1 = params.copy()
                params_m1 = params.copy()
                params_p2 = params.copy()
                params_m2 = params.copy()

                params_p1[mu] += step
                params_m1[mu] -= step
                params_p2[mu] += 2.0 * step
                params_m2[mu] -= 2.0 * step

                e_p1 = self.energy_feval(params_p1)
                e_m1 = self.energy_feval(params_m1)
                e_p2 = self.energy_feval(params_p2)
                e_m2 = self.energy_feval(params_m2)

                grads_fd[mu] = (
                    -e_p2 + 8.0 * e_p1 - 8.0 * e_m1 + e_m2
                ) / (12.0 * step)

            else:
                raise ValueError(f"Unknown stencil: {stencil}")

        return grads_fd

    def hessian_diag_ary_feval(self, params=None):
        """Return the diagonal d^2 E / d theta_mu^2 for the current tUCC ansatz.

        For the FCI backend, hdiag_method="analytic" uses the same
        backward recursion as the analytical gradient and inserts the local
        generator twice, including the derivative-state energy term needed for
        the exact diagonal second derivative.  The "mp2" method is a cheap
        orbital-energy denominator approximation for particle-hole excitation
        pools only.
        """
        method = self._active_hdiag_method()
        if method in ["mp2", "mp_denominators", "orbital_energies"]:
            return self.hessian_diag_ary_feval_mp2_denominators(params=params)
        if method in ["finite_difference", "fd"]:
            return self.hessian_diag_ary_feval_finite_difference(params=params)
        if method in ["analytic", "recursive"]:
            if self._computer_type == 'fci':
                return self.hessian_diag_ary_feval_fci(params=params)
            raise NotImplementedError(
                'Analytical tUCC Hessian diagonals are currently implemented '
                'for computer_type="fci" only. Use '
                'hdiag_method="finite_difference" for this backend.'
                )
        raise ValueError(
            'Unknown hdiag_method. Expected "mp2", '
            '"mp_denominators", "orbital_energies", "finite_difference", '
            f'"fd", "analytic", or "recursive"; got {method!r}.'
        )

    def _hessian_diag_mp2_spin_orbital_energies(self):
        """Return active spin-orbital energies for MP-denominator Hessian guesses."""
        if not hasattr(self._sys, "hf_orbital_energies"):
            raise ValueError(
                'hdiag_method="mp2" requires system.hf_orbital_energies.'
            )

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
                'hdiag_method="mp2" could not align orbital energies with the '
                f'reference: got {len(spin_eps)} spin energies for '
                f'{len(self._ref)} spin orbitals.'
            )
        return spin_eps

    def _hessian_diag_mp2_particle_hole_excitation(self, sq_op):
        """Return oriented particle/hole indices for a clean p-h generator."""
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

    def hessian_diag_ary_feval_mp2_denominators(self, params=None):
        """Approximate Hessian diagonal from positive MP orbital-energy gaps.

        This path is intentionally restricted to ordinary particle-hole pools:
        S, SD, SDT, SDTQ, SDTQP, SDTQPH, and All.  For a clean excitation
        i,j,... -> a,b,... it returns

            eps_a + eps_b + ... - eps_i - eps_j - ...

        which is the positive gap corresponding to the usual MP denominator
        eps_i + eps_j + ... - eps_a - eps_b - ... .  The optimizer's
        hdiag_mode/floor then regularizes this curvature-like scale before it
        is used as a preconditioner.
        """
        allowed_pools = {"S", "SD", "SDT", "SDTQ", "SDTQP", "SDTQPH", "All"}
        pool_type = getattr(self, "_pool_type", None)
        if pool_type not in allowed_pools:
            raise ValueError(
                'hdiag_method="mp2" is only valid for particle-hole pool types '
                f'{sorted(allowed_pools)}; got pool_type={pool_type!r}.'
            )

        spin_eps = self._hessian_diag_mp2_spin_orbital_energies()
        h_diag = np.zeros(len(self._tops))
        records = []

        for mu, top in enumerate(self._tops):
            sq_op = self._pool_obj[top][1]
            excitation = self._hessian_diag_mp2_particle_hole_excitation(sq_op)
            if excitation is None:
                raise ValueError(
                    'hdiag_method="mp2" requires every ansatz operator to be a '
                    f'clean particle-hole excitation; failed at amplitude {mu}.'
                )

            particles = excitation["particles"]
            holes = excitation["holes"]
            gap = sum(spin_eps[p] for p in particles) - sum(spin_eps[h] for h in holes)
            if not np.isfinite(gap) or gap <= 0.0:
                raise ValueError(
                    'hdiag_method="mp2" produced a non-positive orbital-energy '
                    f'gap at amplitude {mu}: {gap!r}.'
                )

            h_diag[mu] = gap
            records.append(
                {
                    "mu": int(mu),
                    "top": int(top),
                    "rank": int(excitation["rank"]),
                    "particles": list(particles),
                    "holes": list(holes),
                    "gap": float(gap),
                }
            )

        self._curr_hdiag = h_diag
        self._mp2_hdiag_records = records
        return h_diag

    def hessian_diag_ary_feval_finite_difference(self, params=None, step=None):
        """Compute Hessian diagonals by central differences of analytic gradients.

        h_mu_mu = (g_mu(theta + delta e_mu) - g_mu(theta - delta e_mu))
                 / (2 delta)
        """
        if step is None:
            step = self._active_hdiag_fd_step()
        step = float(step)
        if step <= 0.0:
            raise ValueError("The Hessian-diagonal finite-difference step must be positive.")

        if params is None:
            params = np.array(self._tamps, dtype=float)
        else:
            params = np.array(params, dtype=float)

        M = len(params)
        h_diag = np.zeros(M)
        for mu in range(M):
            params_p = params.copy()
            params_m = params.copy()
            params_p[mu] += step
            params_m[mu] -= step

            grad_p = np.asarray(self.gradient_ary_feval(params_p), dtype=float)
            grad_m = np.asarray(self.gradient_ary_feval(params_m), dtype=float)
            h_diag[mu] = (grad_p[mu] - grad_m[mu]) / (2.0 * step)

        self._curr_hdiag = h_diag
        return h_diag

    def _hessian_diag_fci_derivative_state_energy(self, mu, chi_state, params):
        """Return 2 <d_mu psi_N | H | d_mu psi_N> for an FCI local derivative."""
        M = len(params)

        if self._ensure_reusable_ucc_objects() and hasattr(self, "_reusable_qc_chi"):
            qc_chi = self._reusable_qc_chi
            qc_hchi = self._reusable_qc_hchi
        else:
            qc_chi = qf.FCIComputer(
                self._nel,
                self._2_spin,
                self._norb)
            qc_hchi = qf.FCIComputer(
                self._nel,
                self._2_spin,
                self._norb)

        t0 = self._fci_time.time()
        qc_chi.set_state(chi_state)
        self._fci_timers['set_state'] += self._fci_time.time() - t0

        for nu in range(mu + 1, M):
            Knu = self._selected_sq_generator(nu)
            t0 = self._fci_time.time()
            qc_chi.apply_sqop_evolution(
                params[nu],
                Knu,
                antiherm=True,
                adjoint=False)
            self._fci_timers['apply_sqop_evolution'] += self._fci_time.time() - t0

        t0 = self._fci_time.time()
        chi_full = qc_chi.get_state_deep()
        self._fci_timers['get_state_deep'] += self._fci_time.time() - t0

        t0 = self._fci_time.time()
        qc_hchi.set_state(chi_full)
        self._fci_timers['set_state'] += self._fci_time.time() - t0

        self._apply_hamiltonian_fci(qc_hchi)

        t0 = self._fci_time.time()
        derivative_state_energy = 2.0 * np.real(
            qc_chi.get_state().vector_dot(qc_hchi.get_state())
            )
        self._fci_timers['vector_dot'] += self._fci_time.time() - t0

        return derivative_state_energy

    def derivative_ary_feval(self, params, return_energy=False,
                             return_gradient=True, return_hessian_diag=False):
        """Return a requested derivative bundle for optimizer hot paths.

        The FCI/analytic-Hessian path can reuse the same forward state and
        backward recursion to produce energy, gradient, and Hessian diagonal.
        Other methods fall back to the ordinary public evaluators so callers can
        safely request a bundle without committing to a specific backend.
        """
        hdiag_method = self._active_hdiag_method()
        analytic_hdiag = hdiag_method in ["analytic", "recursive"]

        if return_hessian_diag and analytic_hdiag and self._computer_type == "fci":
            return self.derivative_ary_feval_fci(
                params,
                return_energy=return_energy,
                return_gradient=return_gradient,
                return_hessian_diag=return_hessian_diag,
            )

        out = {}
        if return_energy and return_gradient:
            energy, grad = self.gradient_ary_feval(params, return_energy=True)
            out["energy"] = energy
            out["gradient"] = grad
        elif return_energy:
            out["energy"] = self.energy_feval(params)
        elif return_gradient:
            out["gradient"] = self.gradient_ary_feval(params)

        if return_hessian_diag:
            out["hessian_diag"] = self.hessian_diag_ary_feval(params)

        return out

    def derivative_ary_feval_fci(self, params=None, return_energy=False,
                                 return_gradient=True, return_hessian_diag=False):
        """Compute a combined FCI derivative bundle for factorized tUCC.

        When `return_hessian_diag` is requested this uses the analytical
        diagonal-Hessian recursion.  The gradient is available at almost no
        extra cost from the first generator insertion at each local point, while
        the Hessian diagonal then applies the same generator a second time and
        adds the derivative-state energy term.
        """
        if not self._fast:
            raise ValueError("self._fast must be True for derivative measurement.")

        if return_hessian_diag and self._pool_type == 'sa_SD':
            raise ValueError('Must use single term particle-hole nbody operators for Hessian diagonal calculation')

        if return_hessian_diag and not self._ref_from_hf:
            raise ValueError('hessian_diag_ary_feval_fci only compatible with hf reference at this time.')

        if params is None:
            params = np.array(self._tamps, dtype=float)
        else:
            params = np.array(params, dtype=float)

        M = len(params)
        grads = np.zeros(M) if return_gradient else None
        h_diag = np.zeros(M) if return_hessian_diag else None
        out = {}
        if M == 0:
            if return_energy:
                out["energy"] = self.energy_feval(params)
            if return_gradient:
                self._curr_grad_norm = 0.0
                self._res_vec_evals += 1
                self._res_m_evals += 0
                out["gradient"] = grads
            if return_hessian_diag:
                self._curr_hdiag = h_diag
                out["hessian_diag"] = h_diag
            return out

        vqc_ops = self._active_reusable_pool(params)
        if vqc_ops is None:
            vqc_ops = qf.SQOpPool()
            for tamp, top in zip(params, self._tops):
                vqc_ops.add(tamp, self._pool_obj[top][1])
            qc_psi = qf.FCIComputer(
                self._nel,
                self._2_spin,
                self._norb)
            qc_sig = qf.FCIComputer(
                self._nel,
                self._2_spin,
                self._norb)
        else:
            qc_psi = self._reusable_qc_psi
            qc_sig = self._reusable_qc_sig

        t0 = self._fci_time.time()
        qc_psi.hartree_fock()
        self._fci_timers['hartree_fock'] += self._fci_time.time() - t0

        t0 = self._fci_time.time()
        qc_psi.evolve_pool_trotter_basic(
            vqc_ops,
            antiherm=True,
            adjoint=False)
        self._fci_timers['evolve_pool_trotter_basic'] += self._fci_time.time() - t0

        t0 = self._fci_time.time()
        psi_i = qc_psi.get_state_deep()
        self._fci_timers['get_state_deep'] += self._fci_time.time() - t0

        t0 = self._fci_time.time()
        qc_sig.set_state(psi_i)
        self._fci_timers['set_state'] += self._fci_time.time() - t0

        self._apply_hamiltonian_fci(qc_sig)
        if return_energy:
            t0 = self._fci_time.time()
            energy = np.real(qc_psi.get_state().vector_dot(qc_sig.get_state()))
            self._fci_timers['vector_dot'] += self._fci_time.time() - t0
            self._curr_energy = float(energy)
            out["energy"] = self._curr_energy

        mu = M - 1
        Kmu_prev = self._selected_sq_generator(mu)

        t0 = self._fci_time.time()
        qc_psi.apply_sqop(Kmu_prev)
        self._fci_timers['apply_sqop'] += self._fci_time.time() - t0

        if return_gradient:
            t0 = self._fci_time.time()
            grads[mu] = 2.0 * np.real(
                qc_sig.get_state().vector_dot(qc_psi.get_state())
                )
            self._fci_timers['vector_dot'] += self._fci_time.time() - t0

        if return_hessian_diag:
            t0 = self._fci_time.time()
            chi_state = qc_psi.get_state_deep()
            self._fci_timers['get_state_deep'] += self._fci_time.time() - t0

            t0 = self._fci_time.time()
            qc_psi.apply_sqop(Kmu_prev)
            self._fci_timers['apply_sqop'] += self._fci_time.time() - t0

            t0 = self._fci_time.time()
            double_insertion = 2.0 * np.real(
                qc_sig.get_state().vector_dot(qc_psi.get_state())
                )
            self._fci_timers['vector_dot'] += self._fci_time.time() - t0

            h_diag[mu] = double_insertion + self._hessian_diag_fci_derivative_state_energy(
                mu,
                chi_state,
                params)

        t0 = self._fci_time.time()
        qc_psi.set_state(psi_i)
        self._fci_timers['set_state'] += self._fci_time.time() - t0

        for mu in reversed(range(M - 1)):
            tamp = params[mu + 1]
            Kmu = self._selected_sq_generator(mu)

            t0 = self._fci_time.time()
            qc_psi.apply_sqop_evolution(
                -1.0 * tamp,
                Kmu_prev,
                antiherm=True,
                adjoint=False)
            self._fci_timers['apply_sqop_evolution'] += self._fci_time.time() - t0

            t0 = self._fci_time.time()
            qc_sig.apply_sqop_evolution(
                -1.0 * tamp,
                Kmu_prev,
                antiherm=True,
                adjoint=False)
            self._fci_timers['apply_sqop_evolution'] += self._fci_time.time() - t0

            t0 = self._fci_time.time()
            psi_i = qc_psi.get_state_deep()
            self._fci_timers['get_state_deep'] += self._fci_time.time() - t0

            t0 = self._fci_time.time()
            qc_psi.apply_sqop(Kmu)
            self._fci_timers['apply_sqop'] += self._fci_time.time() - t0

            if return_gradient:
                t0 = self._fci_time.time()
                grads[mu] = 2.0 * np.real(
                    qc_sig.get_state().vector_dot(qc_psi.get_state())
                    )
                self._fci_timers['vector_dot'] += self._fci_time.time() - t0

            if return_hessian_diag:
                t0 = self._fci_time.time()
                chi_state = qc_psi.get_state_deep()
                self._fci_timers['get_state_deep'] += self._fci_time.time() - t0

                t0 = self._fci_time.time()
                qc_psi.apply_sqop(Kmu)
                self._fci_timers['apply_sqop'] += self._fci_time.time() - t0

                t0 = self._fci_time.time()
                double_insertion = 2.0 * np.real(
                    qc_sig.get_state().vector_dot(qc_psi.get_state())
                    )
                self._fci_timers['vector_dot'] += self._fci_time.time() - t0

                h_diag[mu] = double_insertion + self._hessian_diag_fci_derivative_state_energy(
                    mu,
                    chi_state,
                    params)

            t0 = self._fci_time.time()
            qc_psi.set_state(psi_i)
            self._fci_timers['set_state'] += self._fci_time.time() - t0
            Kmu_prev = Kmu

        if return_gradient:
            if self._noise_factor > 1e-14:
                grads = [
                    np.random.normal(np.real(grad_m), self._noise_factor)
                    for grad_m in grads
                ]
            grads = np.asarray(grads)
            np.testing.assert_allclose(np.imag(grads), np.zeros_like(grads), atol=1e-7)
            grads = np.real(grads)
            self._curr_grad_norm = np.linalg.norm(grads)
            self._res_vec_evals += 1
            self._res_m_evals += len(self._tamps)
            out["gradient"] = grads

        if return_hessian_diag:
            self._curr_hdiag = h_diag
            out["hessian_diag"] = h_diag

        return out

    def hessian_diag_ary_feval_fci(self, params=None):
        """Compute analytical tUCC Hessian diagonals for the FCI backend.

        At the local insertion point for coordinate mu, with local state
        |psi_mu> and backward Hamiltonian state |sigma_mu>, the diagonal is

            2 Re <sigma_mu | A_mu^2 | psi_mu>
            + 2 <d_mu psi_N | H | d_mu psi_N>,

        where |d_mu psi_N> is the full forward-propagated derivative state.
        """
        return self.derivative_ary_feval_fci(
            params=params,
            return_energy=False,
            return_gradient=False,
            return_hessian_diag=True,
        )["hessian_diag"]
    
    # TODO(Nick): think about optemization here, probably should have its own c++ function
    def measure_gradient_fqe(self, params=None, return_energy=False):
        """ Returns the disentangled (factorized) UCC gradient, using a
        recursive approach.

        Parameters
        ----------
        params : list of floats
            The variational parameters which characterize _Uvqc.
        """

        if not self._fast:
            raise ValueError("self._fast must be True for gradient measurement.")
        
        if(self._pool_type == 'sa_SD'):
            raise ValueError('Must use single term particle-hole nbody operators for residual calculation')
        
        if not self._ref_from_hf:
            raise ValueError('get_residual_vector_fci_comp only compatible with hf reference at this time.')

        M = len(self._tamps)
        grads = np.zeros(M)
        vqc_ops = self._active_reusable_pool(params)

        if vqc_ops is None:
            vqc_ops = qforte.SQOpPool()
            if params is None:
                for tamp, top in zip(self._tamps, self._tops):
                    vqc_ops.add(tamp, self._pool_obj[top][1])
            else:
                for tamp, top in zip(params, self._tops):
                    vqc_ops.add(tamp, self._pool_obj[top][1])
            qc_psi = qforte.FQEComputer(
                self._nel,
                self._2_spin,
                self._norb)
            qc_sig = qforte.FQEComputer(
                self._nel,
                self._2_spin,
                self._norb)
        else:
            qc_psi = self._reusable_qc_psi
            qc_sig = self._reusable_qc_sig
        
        t0 = self._fqe_time.time()
        qc_psi.hartree_fock()
        self._fqe_timers['hartree_fock'] += self._fqe_time.time() - t0
        
        # qc_psi.apply_circuit(Utot)
        t0 = self._fqe_time.time()
        qc_psi.evolve_pool_trotter_basic(
            vqc_ops,
            antiherm=True,
            adjoint=False)
        self._fqe_timers['evolve_pool_trotter_basic'] += self._fqe_time.time() - t0

        t0 = self._fqe_time.time()
        psi_i = qc_psi.get_state_deep()
        self._fqe_timers['get_state_deep'] += self._fqe_time.time() - t0

        # not sure if copy is faster or reapplication of state
        t0 = self._fqe_time.time()
        qc_sig.set_state(psi_i)
        self._fqe_timers['set_state'] += self._fqe_time.time() - t0

        if(self._apply_ham_as_tensor):
            t0 = self._fqe_time.time()
            qc_sig.apply_tensor_spat_012bdy(
                self._zero_body_energy, 
                self._mo_oeis_np, 
                self._mo_teis_np, 
                )
            self._fqe_timers['apply_tensor_spat_012bdy'] += self._fqe_time.time() - t0
        else:
            t0 = self._fqe_time.time()
            qc_sig.apply_sqop(self._sq_ham)
            self._fqe_timers['apply_sqop'] += self._fqe_time.time() - t0
        energy = None
        if return_energy:
            t0 = self._fqe_time.time()
            energy = np.real(np.vdot(qc_psi.get_state(), qc_sig.get_state()))
            self._fqe_timers['vector_dot'] += self._fqe_time.time() - t0

        mu = M-1

        # find <sing_N | K_N | psi_N>
        Kmu_prev = self._selected_sq_generator(mu)

        t0 = self._fqe_time.time()
        qc_psi.apply_sqop(Kmu_prev, antiherm=True)
        self._fqe_timers['apply_sqop'] += self._fqe_time.time() - t0

        # grads[mu] = 2.0 * np.real(
        #     qc_sig.get_state().vector_dot(qc_psi.get_state())
        #     )

        t0 = self._fqe_time.time()
        grads[mu] = 2.0 * np.real(np.vdot(qc_sig.get_state(), qc_psi.get_state()))
        self._fqe_timers['vector_dot'] += self._fqe_time.time() - t0
        # print(f"[FQE GRAD] grads[{mu}] = {grads[mu]:.16f}")

        #reset Kmu_prev |psi_i> -> |psi_i>
        t0 = self._fqe_time.time()
        qc_psi.set_state(psi_i)
        self._fqe_timers['set_state'] += self._fqe_time.time() - t0

        for mu in reversed(range(M-1)):

            # mu => N-1 => M-2
            # mu+1 => N => M-1
            # Kmu => KN-1
            # Kmu_prev => KN

            if params is None:
                tamp = self._tamps[mu+1]
            else:
                tamp = params[mu+1]

            Kmu = self._selected_sq_generator(mu)

            # The minus sign is dictated by the recursive algorithm used to compute the analytic gradient
            # (see original ADAPT-VQE paper)
            t0 = self._fqe_time.time()
            qc_psi.apply_sqop_evolution(
                -1.0*tamp,
                Kmu_prev,
                antiherm=True,
                adjoint=False)
            self._fqe_timers['apply_sqop_evolution'] += self._fqe_time.time() - t0
            
            t0 = self._fqe_time.time()
            qc_sig.apply_sqop_evolution(
                -1.0*tamp,
                Kmu_prev,
                antiherm=True,
                adjoint=False)
            self._fqe_timers['apply_sqop_evolution'] += self._fqe_time.time() - t0

            t0 = self._fqe_time.time()
            psi_i = qc_psi.get_state_deep()
            self._fqe_timers['get_state_deep'] += self._fqe_time.time() - t0

            t0 = self._fqe_time.time()
            qc_psi.apply_sqop(Kmu, antiherm=True)
            self._fqe_timers['apply_sqop'] += self._fqe_time.time() - t0
            # grads[mu] = 2.0 * np.real(
            #     qc_sig.get_state().vector_dot(qc_psi.get_state())
            #     )
            
            t0 = self._fqe_time.time()
            grads[mu] = 2.0 * np.real(np.vdot(qc_sig.get_state(), qc_psi.get_state()))
            self._fqe_timers['vector_dot'] += self._fqe_time.time() - t0
            # print(f"[FQE GRAD] grads[{mu}] = {grads[mu]:.16f}")

            #reset Kmu |psi_i> -> |psi_i>
            t0 = self._fqe_time.time()
            qc_psi.set_state(psi_i)
            self._fqe_timers['set_state'] += self._fqe_time.time() - t0
            Kmu_prev = Kmu

        np.testing.assert_allclose(np.imag(grads), np.zeros_like(grads), atol=1e-7)
        
        if return_energy:
            return energy, grads
        return grads
    
    def measure_gradient_cusv(self, params=None, return_energy=False):
        """ Returns the disentangled (factorized) UCC gradient, using a
        recursive approach.

        Parameters
        ----------
        params : list of floats
            The variational parameters which characterize _Uvqc.
        """

        if not self._fast:
            raise ValueError("self._fast must be True for gradient measurement.")
        
        if(self._pool_type == 'sa_SD'):
            raise ValueError('Must use single term particle-hole nbody operators for residual calculation')
        
        if not self._ref_from_hf:
            raise ValueError('get_residual_vector_fci_comp only compatible with hf reference at this time.')

        M = len(self._tamps)
        grads = np.zeros(M)
        vqc_ops = self._active_reusable_pool(params)

        if vqc_ops is None:
            vqc_ops = qforte.SQOpPool()
            if params is None:
                for tamp, top in zip(self._tamps, self._tops):
                    vqc_ops.add(tamp, self._pool_obj[top][1])
            else:
                for tamp, top in zip(params, self._tops):
                    vqc_ops.add(tamp, self._pool_obj[top][1])
            qc_psi = qforte.CUSVComputer(
                self._nel,
                self._2_spin,
                self._norb)
            qc_sig = qforte.CUSVComputer(
                self._nel,
                self._2_spin,
                self._norb)
        else:
            qc_psi = self._reusable_qc_psi
            qc_sig = self._reusable_qc_sig
        
        t0 = self._cusv_time.time()
        qc_psi.hartree_fock()
        self._cusv_timers['hartree_fock'] += self._cusv_time.time() - t0
        
        # qc_psi.apply_circuit(Utot)
        t0 = self._cusv_time.time()
        qc_psi.evolve_pool_trotter_basic(
            vqc_ops,
            antiherm=True,
            adjoint=False)
        self._cusv_timers['evolve_pool_trotter_basic'] += self._cusv_time.time() - t0

        t0 = self._cusv_time.time()
        psi_i = qc_psi.get_state_deep()
        self._cusv_timers['get_state_deep'] += self._cusv_time.time() - t0

        # not sure if copy is faster or reapplication of state
        t0 = self._cusv_time.time()
        qc_sig.set_state(psi_i)
        self._cusv_timers['set_state'] += self._cusv_time.time() - t0

        t0 = self._cusv_time.time()
        qc_sig.apply_sqop(self._sq_ham)
        self._cusv_timers['apply_sqop'] += self._cusv_time.time() - t0
        energy = None
        if return_energy:
            t0 = self._cusv_time.time()
            energy = np.real(np.vdot(qc_psi.get_state(), qc_sig.get_state()))
            self._cusv_timers['vector_dot'] += self._cusv_time.time() - t0

        mu = M-1

        # find <sing_N | K_N | psi_N>
        Kmu_prev = self._selected_sq_generator(mu)

        t0 = self._cusv_time.time()
        qc_psi.apply_sqop(Kmu_prev, antiherm=True)
        self._cusv_timers['apply_sqop'] += self._cusv_time.time() - t0

        # grads[mu] = 2.0 * np.real(
        #     qc_sig.get_state().vector_dot(qc_psi.get_state())
        #     )

        t0 = self._cusv_time.time()
        grads[mu] = 2.0 * np.real(np.vdot(qc_sig.get_state(), qc_psi.get_state()))
        self._cusv_timers['vector_dot'] += self._cusv_time.time() - t0
        # print(f"[FQE GRAD] grads[{mu}] = {grads[mu]:.16f}")

        #reset Kmu_prev |psi_i> -> |psi_i>
        t0 = self._cusv_time.time()
        qc_psi.set_state(psi_i)
        self._cusv_timers['set_state'] += self._cusv_time.time() - t0

        for mu in reversed(range(M-1)):

            # mu => N-1 => M-2
            # mu+1 => N => M-1
            # Kmu => KN-1
            # Kmu_prev => KN

            if params is None:
                tamp = self._tamps[mu+1]
            else:
                tamp = params[mu+1]

            Kmu = self._selected_sq_generator(mu)

            # The minus sign is dictated by the recursive algorithm used to compute the analytic gradient
            # (see original ADAPT-VQE paper)
            t0 = self._cusv_time.time()
            qc_psi.apply_sqop_evolution(
                -1.0*tamp,
                Kmu_prev,
                antiherm=True,
                adjoint=False)
            self._cusv_timers['apply_sqop_evolution'] += self._cusv_time.time() - t0
            
            t0 = self._cusv_time.time()
            qc_sig.apply_sqop_evolution(
                -1.0*tamp,
                Kmu_prev,
                antiherm=True,
                adjoint=False)
            self._cusv_timers['apply_sqop_evolution'] += self._cusv_time.time() - t0

            t0 = self._cusv_time.time()
            psi_i = qc_psi.get_state_deep()
            self._cusv_timers['get_state_deep'] += self._cusv_time.time() - t0

            t0 = self._cusv_time.time()
            qc_psi.apply_sqop(Kmu, antiherm=True)
            self._cusv_timers['apply_sqop'] += self._cusv_time.time() - t0
            # grads[mu] = 2.0 * np.real(
            #     qc_sig.get_state().vector_dot(qc_psi.get_state())
            #     )
            
            t0 = self._cusv_time.time()
            grads[mu] = 2.0 * np.real(np.vdot(qc_sig.get_state(), qc_psi.get_state()))
            self._cusv_timers['vector_dot'] += self._cusv_time.time() - t0
            # print(f"[FQE GRAD] grads[{mu}] = {grads[mu]:.16f}")

            #reset Kmu |psi_i> -> |psi_i>
            t0 = self._cusv_time.time()
            qc_psi.set_state(psi_i)
            self._cusv_timers['set_state'] += self._cusv_time.time() - t0
            Kmu_prev = Kmu

        np.testing.assert_allclose(np.imag(grads), np.zeros_like(grads), atol=1e-7)
        
        if return_energy:
            return energy, grads
        return grads

    def measure_gradient3(self):
        if(self._computer_type == 'fock'):
            return self.measure_gradient3_fock()
        elif(self._computer_type == 'fci'):
            return self.measure_gradient3_fci()
        elif(self._computer_type == 'fci_gpu'):
            return self.measure_gradient3_fci_gpu()
        else:
            raise ValueError(f"{self._computer_type} is an unrecognized computer type.") 


    def measure_gradient3_fock(self):
        """ Calculates 2 Re <Psi|H K_mu |Psi> for all K_mu in self._pool_obj.
        For antihermitian K_mu, this is equal to <Psi|[H, K_mu]|Psi>.
        In ADAPT-VQE, this is the 'residual gradient' used to determine
        whether to append exp(t_mu K_mu) to the iterative ansatz.
        """
        # print("use fock gradient 3")

        if not self._fast:
            raise ValueError("self._fast must be True for gradient measurement.")

        Utot = self.build_Uvqc()
        if self._ensure_reusable_ucc_objects():
            qc_psi = self._reusable_qc_psi
            qc_sig = self._reusable_qc_sig
            qc_psi.set_coeff_vec(copy.deepcopy(self._reusable_fock_zero_state))
        else:
            qc_psi = qforte.Computer(self._nqb)
            qc_sig = qforte.Computer(self._nqb)

        qc_psi.apply_circuit(Utot)
        psi_i = copy.deepcopy(qc_psi.get_coeff_vec())

        # TODO: Check if it's faster to recompute psi_i or copy it.
        qc_sig.set_coeff_vec(copy.deepcopy(psi_i))
        qc_sig.apply_operator(self._qb_ham)

        grads = np.zeros(len(self._pool_obj))

        for mu, (coeff, operator) in enumerate(self._pool_obj):
            Kmu = self._full_jw_generator(mu)
            qc_psi.apply_operator(Kmu)
            grads[mu] = 2.0 * np.real(np.vdot(qc_sig.get_coeff_vec(), qc_psi.get_coeff_vec()))
            qc_psi.set_coeff_vec(copy.deepcopy(psi_i))

        np.testing.assert_allclose(np.imag(grads), np.zeros_like(grads), atol=1e-7)

        return grads

    def measure_gradient3_fci(self):
        """ Calculates 2 Re <Psi|H K_mu |Psi> for all K_mu in self._pool_obj.
        For antihermitian K_mu, this is equal to <Psi|[H, K_mu]|Psi>.
        In ADAPT-VQE, this is the 'residual gradient' used to determine
        whether to append exp(t_mu K_mu) to the iterative ansatz.
        """

        if not self._fast:
            raise ValueError("self._fast must be True for gradient measurement.")

        vqc_ops = self._active_reusable_pool(self._tamps)
        if vqc_ops is None:
            qc_psi = qf.FCIComputer(
                self._nel,
                self._2_spin,
                self._norb)
            qc_sig = qf.FCIComputer(
                self._nel,
                self._2_spin,
                self._norb)

            vqc_ops = qf.SQOpPool()
            for tamp, top in zip(self._tamps, self._tops):
                vqc_ops.add(tamp, self._pool_obj[top][1])
        else:
            qc_psi = self._reusable_qc_psi
            qc_sig = self._reusable_qc_sig

        qc_psi.hartree_fock()

        # build wave function for current ADAPT iteration
        # using self._tamps and self._tops
        qc_psi.evolve_pool_trotter_basic(
            vqc_ops,
            antiherm=True,
            adjoint=False)

        # psi_i = copy.deepcopy(qc_psi.get_coeff_vec())
        psi_i = qc_psi.get_state_deep()
        
        qc_sig.set_state(psi_i)

        # qc_sig.apply_operator(self._qb_ham)
        if(self._apply_ham_as_tensor):
            qc_sig.apply_tensor_spat_012bdy(
            self._zero_body_energy, 
            self._mo_oeis, 
            self._mo_teis, 
            self._mo_teis_einsum, 
            self._norb)
        else:   
            qc_sig.apply_sqop(self._sq_ham)

        grads = np.zeros(len(self._pool_obj))
        for mu, (coeff, operator) in enumerate(self._pool_obj):
            Kmu = self._full_sq_generator(mu)
            qc_psi.apply_sqop(Kmu)
            grads[mu] = 2.0 * np.real(qc_sig.get_state().vector_dot(qc_psi.get_state()))
            qc_psi.set_state(psi_i)

        np.testing.assert_allclose(np.imag(grads), np.zeros_like(grads), atol=1e-7)
        
        return grads

    def gradient_ary_feval(self, params, return_energy=False):
        if(self._computer_type == 'fock'): 
            return self.gradient_ary_feval_fock(params, return_energy=return_energy)
        elif(self._computer_type == 'fci'):
            return self.gradient_ary_feval_fci(params, return_energy=return_energy)
        elif(self._computer_type == 'fqe'):
            return self.gradient_ary_feval_fqe(params, return_energy=return_energy)
        elif(self._computer_type == 'fci_gpu'):
            return self.gradient_ary_feval_fci_gpu(params, return_energy=return_energy)
        elif(self._computer_type == 'cusv'):
            return self.gradient_ary_feval_cusv(params, return_energy=return_energy)
        else:
            raise ValueError(f"{self._computer_type} is an unrecognized computer type.") 

    def _finalize_gradient_ary_result(self, grad_result, return_energy=False):
        if return_energy:
            energy, grads = grad_result
        else:
            energy = None
            grads = grad_result

        if(self._noise_factor > 1e-14):
            grads = [np.random.normal(np.real(grad_m), self._noise_factor) for grad_m in grads]

        self._curr_grad_norm = np.linalg.norm(grads)
        self._res_vec_evals += 1
        self._res_m_evals += len(self._tamps)

        grads = np.asarray(grads)
        if return_energy:
            self._curr_energy = float(np.real(energy))
            return self._curr_energy, grads
        return grads

    def gradient_ary_feval_fock(self, params, return_energy=False):
        return self._finalize_gradient_ary_result(
            self.measure_gradient(params, return_energy=return_energy),
            return_energy=return_energy,
        )

    def gradient_ary_feval_fci(self, params, return_energy=False):
        return self._finalize_gradient_ary_result(
            self.measure_gradient(params, return_energy=return_energy),
            return_energy=return_energy,
        )

    def gradient_ary_feval_fqe(self, params, return_energy=False):
        return self._finalize_gradient_ary_result(
            self.measure_gradient(params, return_energy=return_energy),
            return_energy=return_energy,
        )

    def gradient_ary_feval_fci_gpu(self, params, return_energy=False):
        return self._finalize_gradient_ary_result(
            self.measure_gradient(params, return_energy=return_energy),
            return_energy=return_energy,
        )
    
    def gradient_ary_feval_cusv(self, params, return_energy=False):
        return self._finalize_gradient_ary_result(
            self.measure_gradient(params, return_energy=return_energy),
            return_energy=return_energy,
        )

    def measure_gradient_fci_gpu(self, params=None, return_energy=False):
        """ Returns the disentangled (factorized) UCC gradient for FCIComputerGPU,
        using a recursive approach.

        Parameters
        ----------
        params : list of floats
            The variational parameters which characterize _Uvqc.
        """

        if not self._fast:
            raise ValueError("self._fast must be True for gradient measurement.")
        
        if(self._pool_type == 'sa_SD'):
            raise ValueError('Must use single term particle-hole nbody operators for residual calculation')
        
        if not self._ref_from_hf:
            raise ValueError('measure_gradient_fci_gpu only compatible with hf reference at this time.')

        M = len(self._tamps)
        grads = np.zeros(M)

        # Use reusable infrastructure if available
        if hasattr(self, '_reusable_pool_gpu') and hasattr(self, '_reusable_qc_psi'):
            # Update pool coefficients
            if params is None:
                complex_tamps = [complex(t, 0.0) for t in self._tamps]
                self._reusable_pool_gpu.update_evolution_coeffs(complex_tamps)
            else:
                complex_params = [complex(p, 0.0) for p in params]
                self._reusable_pool_gpu.update_evolution_coeffs(complex_params)
            
            # Use reusable computers (avoid GPU allocation/deallocation)
            qc_psi = self._reusable_qc_psi
            qc_sig = self._reusable_qc_sig
            vqc_ops = self._reusable_pool_gpu
            
            # Reset to HF state
            t0 = self._gpu_time.time()
            qc_psi.hartree_fock_gpu()
            self._gpu_timers['hartree_fock_gpu'] += self._gpu_time.time() - t0
            
        else:
            # Fallback: create fresh computers and pool
            vqc_ops = qforte.SQOpPoolGPU(data_type=self.data_type)
            if params is None:
                for tamp, top in zip(self._tamps, self._tops):
                    vqc_ops.add(tamp, self._pool_obj[top][1])
            else:
                for tamp, top in zip(params, self._tops):
                    vqc_ops.add(tamp, self._pool_obj[top][1])

            qc_psi = qforte.FCIComputerGPU(
                self._nel, 
                self._2_spin, 
                self._norb,
                on_gpu=False,
                data_type=self.data_type)
            
            t0 = self._gpu_time.time()
            qc_psi.hartree_fock_gpu()
            self._gpu_timers['hartree_fock_gpu'] += self._gpu_time.time() - t0
            
            t0 = self._gpu_time.time()
            qc_psi.to_gpu()
            self._gpu_timers['to_gpu'] += self._gpu_time.time() - t0
            
            qc_sig = qforte.FCIComputerGPU(
                self._nel, 
                self._2_spin, 
                self._norb,
                on_gpu=True,
                data_type=self.data_type)
        
        # Common gradient computation logic
        t0 = self._gpu_time.time()
        qc_psi.evolve_pool_trotter_basic_gpu(
            vqc_ops,
            antiherm=True,
            adjoint=False)
        self._gpu_timers['evolve_pool_trotter_basic_gpu'] += self._gpu_time.time() - t0

        # not sure if copy is faster or reapplication of state
        t0 = self._gpu_time.time()
        qc_sig.set_state_from_other_gpu(qc_psi)
        self._gpu_timers['set_state_from_other_gpu'] += self._gpu_time.time() - t0

        if(self._apply_ham_as_tensor):
            t0 = self._gpu_time.time()
            qc_sig.apply_tensor_spat_012bdy_gpu_v2(
                self._zero_body_energy, 
                self._mo_oeis_gpu, 
                self._mo_teis_gpu, 
                self._mo_teis_einsum_gpu, 
                self._norb)
            self._gpu_timers['apply_tensor_spat_012bdy_gpu'] += self._gpu_time.time() - t0
        else:
            t0 = self._gpu_time.time()
            qc_sig.apply_sqop_gpu(self._sq_ham)
            self._gpu_timers['apply_sqop_gpu'] += self._gpu_time.time() - t0

        mu = M-1

        # find <psi_N | K_N | sig_N> using precomputed pool index arrays (no per-call graph traversal)
        outer_coeff_last = self._pool_obj[self._tops[mu]][0]

        # new fused opp
        t0 = self._gpu_time.time()
        if self.data_type == 'real':
            grads[mu] = 2.0 * float(np.real(outer_coeff_last)) * qc_psi.dot_sqop_from_pool_gpu_real(qc_sig, vqc_ops, mu)
        else:
            grads[mu] = 2.0 * np.real(outer_coeff_last * qc_psi.dot_sqop_from_pool_gpu(qc_sig, vqc_ops, mu))
        self._gpu_timers['dot_sqop_gpu'] += self._gpu_time.time() - t0

        # old way
        # t0 = self._gpu_time.time()
        # qc_psi.apply_sqop_gpu(Kmu_prev)
        # self._gpu_timers['apply_sqop_gpu'] += self._gpu_time.time() - t0
        
        # t0 = self._gpu_time.time()
        # grads[mu] = 2.0 * np.real(qc_sig.state_vector_dot_gpu(qc_psi))
        # self._gpu_timers['vector_dot'] += self._gpu_time.time() - t0
        # print(f"[GPU GRAD] grads[{mu}] = {grads[mu]:.16f}")

        # don't need to reset psi_i since we are using fused dot-apply
        #reset Kmu_prev |psi_i> -> |psi_i>
        # t0 = self._gpu_time.time()
        # qc_psi.set_state_gpu(self.psi_i)
        # self._gpu_timers['set_state_gpu'] += self._gpu_time.time() - t0

        for mu in reversed(range(M-1)):

            # mu => N-1 => M-2
            # mu+1 => N => M-1
            # Kmu => KN-1
            # Kmu_prev => KN

            if params is None:
                tamp = self._tamps[mu+1]
            else:
                tamp = params[mu+1]

            outer_coeff_mu = self._pool_obj[self._tops[mu]][0]

            # The minus sign is dictated by the recursive algorithm used to compute the analytic gradient
            # (see original ADAPT-VQE paper)
            t0 = self._gpu_time.time()
            qc_psi.apply_sqop_evolution_from_pool_gpu(
                -1.0*tamp,
                vqc_ops,
                mu+1,
                antiherm=True,
                adjoint=False)
            self._gpu_timers['apply_sqop_evolution_gpu'] += self._gpu_time.time() - t0
            
            t0 = self._gpu_time.time()
            qc_sig.apply_sqop_evolution_from_pool_gpu(
                -1.0*tamp,
                vqc_ops,
                mu+1,
                antiherm=True,
                adjoint=False)
            self._gpu_timers['apply_sqop_evolution_gpu'] += self._gpu_time.time() - t0

            # new fused opp — pool-indexed: no per-call graph traversal
            t0 = self._gpu_time.time()
            if self.data_type == 'real':
                grads[mu] = 2.0 * float(np.real(outer_coeff_mu)) * qc_psi.dot_sqop_from_pool_gpu_real(qc_sig, vqc_ops, mu)
            else:
                grads[mu] = 2.0 * np.real(outer_coeff_mu * qc_psi.dot_sqop_from_pool_gpu(qc_sig, vqc_ops, mu))
            self._gpu_timers['dot_sqop_gpu'] += self._gpu_time.time() - t0

        np.testing.assert_allclose(np.imag(grads), np.zeros_like(grads), atol=1e-7)
        
        if return_energy:
            # The GPU recursive path does not currently expose a cheap state-dot
            # energy consistently across reusable/fresh computers. Preserve the
            # public return_energy API with the ordinary energy path here.
            energy_params = self._tamps if params is None else params
            return self.energy_feval(energy_params), grads
        return grads

    def measure_gradient3_fci_gpu(self):
        """ Calculates 2 Re <Psi|H K_mu |Psi> for all K_mu in self._pool_obj.
        For antihermitian K_mu, this is equal to <Psi|[H, K_mu]|Psi>.
        In ADAPT-VQE, this is the 'residual gradient' used to determine
        whether to append exp(t_mu K_mu) to the iterative ansatz.
        (GPU version)
        """

        if not self._fast:
            raise ValueError("self._fast must be True for gradient measurement.")

        qc_psi = qforte.FCIComputerGPU(
            self._nel, 
            self._2_spin, 
            self._norb,
            on_gpu=True,
            data_type=self.data_type) 

        t0 = self._gpu_time.time()
        qc_psi.hartree_fock_gpu()
        self._gpu_timers['hartree_fock_gpu'] += self._gpu_time.time() - t0

        # build wave function for current ADAPT iteration
        # using self._tamps and self._tops
        
        vqc_ops = qforte.SQOpPoolGPU(data_type=self.data_type)
        for tamp, top in zip(self._tamps, self._tops):
                vqc_ops.add(tamp, self._pool_obj[top][1])


        t0 = self._gpu_time.time()
        qc_psi.evolve_pool_trotter_basic_gpu(
            vqc_ops,
            antiherm=True,
            adjoint=False)
        self._gpu_timers['evolve_pool_trotter_basic_gpu'] += self._gpu_time.time() - t0

        # Initialize psi_i if it doesn't exist
        if not hasattr(self, 'psi_i'):
            self.psi_i = None

        t0 = self._gpu_time.time()
        if self.psi_i:
            qc_psi.copy_state_into(self.psi_i)
        else:
            self.psi_i = qc_psi.get_state_deep()
        self._gpu_timers['get_state_deep'] += self._gpu_time.time() - t0
        
        qc_sig = qforte.FCIComputerGPU(
            self._nel, 
            self._2_spin, 
            self._norb,
            on_gpu=True,
            data_type=self.data_type) 
        
        t0 = self._gpu_time.time()
        qc_sig.set_state_gpu(self.psi_i)
        self._gpu_timers['set_state_gpu'] += self._gpu_time.time() - t0

        if(self._apply_ham_as_tensor):
            t0 = self._gpu_time.time()
            qc_sig.apply_tensor_spat_012bdy_gpu_v2(
                self._zero_body_energy, 
                self._mo_oeis_gpu, 
                self._mo_teis_gpu, 
                self._mo_teis_einsum_gpu, 
                self._norb)
            self._gpu_timers['apply_tensor_spat_012bdy_gpu'] += self._gpu_time.time() - t0
        else:
            t0 = self._gpu_time.time()
            qc_sig.apply_sqop_gpu(self._sq_ham)
            self._gpu_timers['apply_sqop_gpu'] += self._gpu_time.time() - t0

        grads = np.zeros(len(self._pool_obj))
        for mu, (coeff, operator) in enumerate(self._pool_obj):
            Kmu = operator
            Kmu.mult_coeffs(coeff)
            
            t0 = self._gpu_time.time()
            qc_psi.apply_sqop_gpu(Kmu)
            self._gpu_timers['apply_sqop_gpu'] += self._gpu_time.time() - t0
            
            t0 = self._gpu_time.time()
            grads[mu] = 2.0 * np.real(qc_sig.state_vector_dot_gpu(qc_psi))
            self._gpu_timers['vector_dot'] += self._gpu_time.time() - t0
            
            t0 = self._gpu_time.time()
            qc_psi.set_state_gpu(self.psi_i)
            self._gpu_timers['set_state_gpu'] += self._gpu_time.time() - t0

        np.testing.assert_allclose(np.imag(grads), np.zeros_like(grads), atol=1e-7)
        
        return grads

    def report_iteration(self, x):

        self._k_counter += 1

        if(self._k_counter == 1):
            print('\n    k iteration         Energy               dE           Ngvec ev      Ngm ev*         ||g||')
            print('--------------------------------------------------------------------------------------------------')
            if (self._print_summary_file):
                f = open("summary.dat", "w+", buffering=1)
                f.write('\n#    k iteration         Energy               dE           Ngvec ev      Ngm ev*         ||g||')
                f.write('\n#--------------------------------------------------------------------------------------------------')
                f.close()

        # else:
        dE = self._curr_energy - self._prev_energy
        print(f'     {self._k_counter:7}        {self._curr_energy:+12.10f}      {dE:+12.10f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {self._curr_grad_norm:+12.10f}')

        if (self._print_summary_file):
            f = open("summary.dat", "a", buffering=1)
            f.write(f'\n       {self._k_counter:7}        {self._curr_energy:+12.12f}      {dE:+12.12f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {self._curr_grad_norm:+12.12f}')
            f.close()

        self._prev_energy = self._curr_energy

    def verify_required_UCCVQE_attributes(self):
        if self._use_analytic_grad is None:
            raise NotImplementedError('Concrete UCCVQE class must define self._use_analytic_grad attribute.')

        if self._pool_type is None:
            raise NotImplementedError('Concrete UCCVQE class must define self._pool_type attribute.')

        if self._pool_obj is None:
            raise NotImplementedError('Concrete UCCVQE class must define self._pool_obj attribute.')
