"""
S2PQE classes
====================================
Classes for solving the schrodinger equation via measurement of its projections
and subsequent updates of the disentangled UCC amplitudes.
"""

import qforte
from qforte.abc.uccpqeabc import UCCPQE

from qforte.experiment import *
from qforte.maths import optimizer
from qforte.utils.transforms import *
from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize
from qforte.utils import moment_energy_corrections

from qforte.helper.printing import matprint

import numpy as np
from scipy.linalg import lstsq

class UCCNPPQE(UCCPQE):
    """
    A class that encompasses the three components of using the projective
    quantum eigensolver to optimize a disentangld UCCN-like wave function.

    UCC-PQE: (1) prepares a quantum state on the quantum computer
    representing the wave function to be simulated, (2) evaluates the residuals

    .. math::
        r_\mu = \langle \Phi_\mu | \hat{U}^\dagger(\mathbf{t}) \hat{H} \hat{U}(\mathbf{t}) | \Phi_0 \\rangle

    and (3) optimizes the wave fuction via projective solution of
    the UCC Schrodinger Equation via a quazi-Newton update equation.
    Using this strategy, an amplitude :math:`t_\mu^{(k+1)}` for iteration :math:`k+1`
    is given by

    .. math::
        t_\mu^{(k+1)} = t_\mu^{(k)} + \\frac{r_\mu^{(k)}}{\Delta_\mu}

    where :math:`\Delta_\mu` is the standard Moller Plesset denominator.

    Attributes
    ----------
    _ref : list
        The set of 1s and 0s indicating the initial quantum state.

    """
    def run(self,
            pool_type='SD',
            opt_thresh = 1.0e-5,
            opt_e_thresh = 1.0e-6,
            opt_maxiter = 40,
            noise_factor = 0.0,
            time_step = 0.1,
            max_time_step = 0.5,
            use_dt_from_l1_norm = False,
            ppqe_trotter_order = np.inf,
            optimizer = 'rotation',
            update_type = 'jacobi_like'):

        if(self._state_prep_type != 'occupation_list'):
            raise ValueError("PQE implementation can only handle occupation_list Hartree-Fock reference.")

        self._pool_type = pool_type
        self._optimizer = optimizer
        self._opt_thresh = opt_thresh
        self._opt_e_thresh = opt_e_thresh
        self._opt_maxiter = opt_maxiter
        self._noise_factor = noise_factor
        self._dt = time_step
        self._dt_max = max_time_step
        self._use_dt_from_l1_norm = use_dt_from_l1_norm

        self._ppqe_trotter_order = ppqe_trotter_order

        if(self._ppqe_trotter_order in [1, 2]):
            self._hermitian_pairs = qforte.SQOpPool()
            # this is updated, evolution time is now just 1.0 here
            self._hermitian_pairs.add_hermitian_pairs(1.0, self._sq_ham)

        if(self._use_dt_from_l1_norm):
            self._dt = min(1.0 / (np.sqrt(self._ham_l1_norm_sq)), self._dt_max )
            print(f"\n ==> Using (Initial) time step: {self._dt:.6f} for next iteration. <===\n")

        self._ppqe_update_type = update_type
        
        if(self._ppqe_update_type == 'jacobi_like'):
            self._ppqe_update_type_str = 'LPU'
        elif(self._ppqe_update_type == 'two_level_rotation'):
            self._ppqe_update_type_str = 'SRU(Re)'
        elif(self._ppqe_update_type == 'two_level_rotation_im'):
            self._ppqe_update_type_str = 'SRU(Im)'
        else:
            raise ValueError(f"PPQE update type {self._ppqe_update_type} is not supported.")
        

        self._tops = []
        self._tamps = []
        self._converged = 0

        self._res_vec_evals = 0
        self._res_m_evals = 0
        # list: tuple(excited determinant, phase_factor)
        self._excited_dets = []
        self._excited_dets_fci_comp = []

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0
        self._n_shots = 0
        # self._results = [] #keep for future implementations

        self.print_options_banner()

        self._timer = qforte.local_timer()

        self._timer.reset()
        self.fill_pool()
        self._timer.record("fill_pool")


        if self._verbose:
            print('\n\n-------------------------------------')
            print('   Second Quantized Operator Pool')
            print('-------------------------------------')
            print(self._pool_obj.str())

        self._timer.reset()
        self.initialize_ansatz()
        self._timer.record("initialize_ansatz")

        if(self._verbose):
            print('\nt operators included from pool: \n', self._tops)
            print('Initial tamplitudes for tops: \n', self._tamps)

        self._timer.reset()
        self.fill_excited_dets()
        self._timer.record("fill_excited_dets")

        self._timer.reset()
        self.build_orb_energies()
        self._timer.record("build_orb_energies")
        
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

            print('Final tamplitudes for tops:')
            print('------------------------------')
            for i, tamp in enumerate( self._tamps ):
                print(f'  {i:4}      {tamp:+12.8f}')

        self._n_nonzero_params = 0
        for tmu in self._tamps:
            if(np.abs(tmu) > 1.0e-12):
                self._n_nonzero_params += 1

        self._n_pauli_trm_measures = int(2*self._Nl*self._res_vec_evals*self._n_nonzero_params + self._Nl*self._res_vec_evals)

        self.print_summary_banner()
        self.verify_run()

    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not fully implemented for UCCN-PPQE.')

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_PQE_attributes()
        self.verify_required_UCCPQE_attributes()

    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('           Unitary Coupled Cluster PPQE   ')
        print('-----------------------------------------------------')

        print('\n\n                 ==> UCC-PPQE options <==')
        print('---------------------------------------------------------')
        print('Trial reference state:                   ',  ref_string(self._ref, self._nqb))
        print('Number of Electrons:                     ',  self._nel)
        print('Multiplicity:                            ',  self._mult)
        print('Number spatial orbitals:                 ',  self._norb)
        print('Number of Hamiltonian Pauli terms:       ',  self._Nl)
        print('Hamiltonian Sq L1 norm:                  ',  round(self._ham_l1_norm_sq, 4))
        if(self._use_dt_from_l1_norm):
            print('Time Step (dt) = 1/|h1|:                 ',  round(self._dt, 6))
        else:
            print('Time step (dt):                          ',  round(self._dt, 6))
        print('Trial state preparation method:          ',  self._state_prep_type)
        print('PPQE Trotter Order:                      ',  self._ppqe_trotter_order)
        # print('Trotter order (rho):                     ',  self._trotter_order)
        # print('Trotter number (m):                      ',  self._trotter_number)
        print('Use fast version of algorithm:           ',  str(self._fast))
        if(self._fast):
            print('Measurement varience thresh:             ',  'NA')
        else:
            print('Measurement varience thresh:             ',  0.01)

        print('Use qubit excitations:                   ', self._qubit_excitations)
        print('Use compact excitation circuits:         ', self._compact_excitations)
        print('Optimizer:                               ', self._optimizer)
        print('PPQE Update Type:                        ', self._ppqe_update_type_str)
        if self._diis_max_dim >= 2 and self._optimizer.lower() == 'rotation':
            print('DIIS dimension:                          ', self._diis_max_dim)
        else:
            print('DIIS dimension:                           Disabled')

        res_thrsh_str = '{:.2e}'.format(self._opt_thresh)
        e_thrsh_str = '{:.2e}'.format(self._opt_e_thresh)
        print('Maximum number of iterations:            ',  self._opt_maxiter)
        print('Residual-norm threshold:                 ',  res_thrsh_str)
        print('Energy threshold:                        ',  e_thrsh_str)

        print('Operator pool type:                      ',  str(self._pool_type))


    def print_summary_banner(self):

        print('\n\n                   ==> UCC-PPQE summary <==')
        print('-----------------------------------------------------------')
        print('Final UCCN-PPQE Energy:                      ', round(self._Egs, 10))
        if self._max_moment_rank:
            print('Moment-corrected (MP) UCCN-PPQE Energy:      ', round(self._E_mmcc_mp[0], 10))
            print('Moment-corrected (EN) UCCN-PPQE Energy:      ', round(self._E_mmcc_en[0], 10))
        print('Number of operators in pool:                 ', len(self._pool_obj))
        print('Final number of amplitudes in ansatz:        ', len(self._tamps))
        print('Number of classical parameters used:         ', len(self._tamps))
        print('Number of non-zero parameters used:          ', self._n_nonzero_params)
        print('Number of CNOT gates in deepest circuit:     ', self._n_cnot)
        print('Number of Pauli term measurements:           ', self._n_pauli_trm_measures)
        print('Number of residual vector evaluations:       ', self._res_vec_evals)
        print('Number of residual element evaluations*:     ', self._res_m_evals)
        print('Number of non-zero res element evaluations:  ', int(self._res_vec_evals)*self._n_nonzero_params)

        print("\n\n")
        print(self._timer)

    def fill_excited_dets(self):
        if(self._computer_type == 'fock'):
            self.fill_excited_dets_fock()
        elif(self._computer_type == 'fci'):
            self.fill_excited_dets_fci()
        else:
            raise ValueError(f"{self._computer_type} is an unrecognized computer type.") 

    def fill_excited_dets_fock(self):
        for _, sq_op in self._pool_obj:
            # 1. Identify the excitation operator
            # occ => i,j,k,...
            # vir => a,b,c,...
            # sq_op is 1.0(a^ b^ i j) - 1.0(j^ i^ b a)

            temp_idx = sq_op.terms()[0][2][-1]
            # TODO: This code assumes that the first N orbitals are occupied, and the others are virtual.
            # Use some other mechanism to identify the occupied orbitals, so we can use use PQE on excited
            # determinants.
            if temp_idx < int(sum(self._ref)/2): # if temp_idx is an occupied idx
                sq_creators = sq_op.terms()[0][1]
                sq_annihilators = sq_op.terms()[0][2]
            else:
                sq_creators = sq_op.terms()[0][2]
                sq_annihilators = sq_op.terms()[0][1]

            # 2. Get the bit representation of the sq_ex_op acting on the reference.
            # We determine the projective condition for this amplitude by zero'ing this residual.

            # `destroyed` exists solely for error catching.
            destroyed = False

            excited_det = qforte.QubitBasis(self._nqb)
            for k, occ in enumerate(self._ref):
                excited_det.set_bit(k, occ)

            # loop over annihilators
            for p in reversed(sq_annihilators):
                if( excited_det.get_bit(p) == 0):
                    destroyed=True
                    break

                excited_det.set_bit(p, 0)

            # then over creators
            for p in reversed(sq_creators):
                if (excited_det.get_bit(p) == 1):
                    destroyed=True
                    break

                excited_det.set_bit(p, 1)

            if destroyed:
                raise ValueError("no ops should destroy reference, something went wrong!!")

            I = excited_det.add()

            qc_temp = qforte.Computer(self._nqb)
            qc_temp.apply_circuit(self._Uprep)
            qc_temp.apply_operator(sq_op.jw_transform(self._qubit_excitations))
            phase_factor = qc_temp.get_coeff_vec()[I]

            self._excited_dets.append((I, phase_factor))

    def fill_excited_dets_fci(self):
        qc = qforte.FCIComputer(
            self._nel, 
            self._2_spin, 
            self._norb)
        
        for _, sq_op in self._pool_obj:
            qc.hartree_fock()
            qc.apply_sqop(sq_op)
            non_zero_tidxs = qc.get_state().get_nonzero_tidxs()

            if(len(non_zero_tidxs) != 1):
                raise ValueError("Pool object elements should only create a single excitation from hf reference.")
            
            if(len(non_zero_tidxs[0]) != 2):
                raise ValueError("Tensor indxs must be from a a matrix.")
            
            phase_factor = qc.get_state().get(non_zero_tidxs[0])

            if(phase_factor != 0.0):
                self._excited_dets_fci_comp.append((non_zero_tidxs[0], phase_factor))

    def get_propogated_residual_vector(self, trial_amps):
        if(self._computer_type == 'fock'):
            return self.get_propogated_residual_vector_fock(trial_amps)
        elif(self._computer_type == 'fci'):
            return self.get_propogated_residual_vector_fci(trial_amps)
        else:
            raise ValueError(f"{self._computer_type} is an unrecognized computer type.") 
        

    def get_propogated_return_amp_vector(self, trial_amps):
        if(self._computer_type == 'fock'):
            # return self.get_propogated_residual_vector_fock(trial_amps)
            raise NotImplementedError('get_propogated_return_amp_vector_fock is not implemented for PPQE.')
        elif(self._computer_type == 'fci'):
            return self.get_propagated_return_amp_vector_fci(trial_amps)
        else:
            raise ValueError(f"{self._computer_type} is an unrecognized computer type.") 

    def get_propogated_residual_vector_fock(self, trial_amps):
        """Returns the Propogated residual vector with elements pertaining to all operators
        in the ansatz circuit.

        Parameters
        ----------
        trial_amps : list of floats
            The list of (real) floating point numbers which will characterize
            the state preparation circuit used in calculation of the residuals.
        """

        raise NotImplementedError('get_propogated_residual_vector_fock is not implemented for S2PQE.')
        if(self._pool_type == 'sa_SD'):
            raise ValueError('Must use single term particle-hole nbody operators for residual calculation')

        U = self.ansatz_circuit(trial_amps)

        qc_res = qforte.Computer(self._nqb)
        qc_res.apply_circuit(self._Uprep)
        qc_res.apply_circuit(U)
        qc_res.apply_operator(self._qb_ham)
        qc_res.apply_circuit(U.adjoint())

        coeffs = qc_res.get_coeff_vec()
        residuals = []

        for I, phase_factor in self._excited_dets:

            # Get the residual element, after accounting for numerical noise.
            res_m = coeffs[I] * phase_factor
            if(np.imag(res_m) != 0.0):
                raise ValueError("residual has imaginary component, something went wrong!!")

            if(self._noise_factor > 1e-12):
                res_m = np.random.normal(np.real(res_m), self._noise_factor)

            residuals.append(res_m)

        self._res_vec_norm = np.linalg.norm(residuals)
        self._res_vec_evals += 1
        self._res_m_evals += len(self._tamps)

        return residuals
    
    def get_propogated_residual_vector_fci(self, trial_amps):
        """Returns the propogated residual vector with elements pertaining to all operators
        in the ansatz circuit.

        Parameters
        ----------
        trial_amps : list of floats
            The list of (real) floating point numbers which will characterize
            the state preparation circuit used in calculation of the residuals.
        """
        if(self._pool_type == 'sa_SD'):
            raise ValueError('Must use single term particle-hole nbody operators for residual calculation')
        
        if not self._ref_from_hf:
            raise ValueError('get_residual_vector_fci_comp only compatible with hf reference at this time.')
        
        temp_pool = qforte.SQOpPool()

        # NICK: Write a 'updatte_coeffs' type fucntion for the op-pool.
        for tamp, top in zip(trial_amps, self._tops):
            temp_pool.add(tamp, self._pool_obj[top][1])

        qc_res = qforte.FCIComputer(
            self._nel, 
            self._2_spin, 
            self._norb)
        
        qc_res.hartree_fock()


        # function assumers first order trotter, with 1 trotter step, and time = 1.0
        qc_res.evolve_pool_trotter_basic(
            temp_pool,
            antiherm=True,
            adjoint=False)

        if(self._ppqe_trotter_order == np.inf):
            # NOTE(Nick): Adding a loop over smaller dt's here to account for large frozen core energy.
            # want self._frozen_core_energy * dt > taylor_thresh
            taylor_thresh = 1.0

            # niter = int(np.abs(self._frozen_core_energy) * self._dt / taylor_thresh) + 1
            # niter = int(np.sqrt(self._ham_l1_norm_sq) * self._dt / taylor_thresh) + 1
            niter = int((self._ham_l1_norm_sq**(0.25)) * self._dt / taylor_thresh) + 1

            # niter = 10

            # print(f"\n\nUsing {niter} iterations for Taylor expansion of the evolution operator.\n\n")

            if(self._apply_ham_as_tensor):

                global_phase = np.exp(-1j * self._frozen_core_energy * self._dt)

                for _ in range(niter):

                    qc_res.evolve_tensor_taylor(
                                0.0, 
                                self._mo_oeis, 
                                self._mo_teis, 
                                self._mo_teis_einsum, 
                                self._norb,
                                float(self._dt/niter),
                                1.0e-15,
                                30,
                                False)
                
                qc_res.scale(global_phase)


            else:
                for _ in range(niter):
                    qc_res.evolve_op_taylor(
                            self._sq_ham,
                            float(self._dt/niter),
                            1.0e-15,
                            30,
                            False)
                
        elif(self._ppqe_trotter_order in [1,2]):
            qc_res.evolve_pool_trotter(
                self._hermitian_pairs,
                self._dt,
                1,
                self._ppqe_trotter_order,
                antiherm=False,
                adjoint=False)

        else:
            raise ValueError(f"PPQE Trotter order {self._ppqe_trotter_order} is not supported.")




        qc_res.evolve_pool_trotter_basic(
            temp_pool,
            antiherm=True,
            adjoint=True)

        R = qc_res.get_state_deep()
        c = R.get([0, 0])
        prop_residuals = []

        # NOTE(Nick): this might be over all excitations, we should only include those that are in the poool.
        for IaIb, phase_factor in self._excited_dets_fci_comp:

            # Get the residual element, after accounting for numerical noise.
            res_m = R.get(IaIb) * phase_factor
            # res_m = R.get(IaIb)

            if self._noise_factor > 1e-12:
                noisy_real = np.random.normal(np.real(res_m), self._noise_factor)
                noisy_imag = np.random.normal(np.imag(res_m), self._noise_factor)
                res_m = noisy_real + 1j * noisy_imag

            prop_residuals.append(res_m)

        self._res_vec_norm = np.linalg.norm(prop_residuals)
        self._res_vec_evals += 1
        self._res_m_evals += len(self._tamps)

        # if(self._use_variable_dt):
        #     self._dt = min(1.0 * self._res_vec_evals / (np.sqrt(self._ham_l1_norm_sq)), self._dt_max )
            # print(f"\n ==> Using (Initial) variable time step: {self._dt:.6f} for next iteration. <===\n")

        if self._noise_factor > 1e-12:
            self._n_shots += 2.0 * (len(self._tamps) + 1.0) / (self._noise_factor * self._noise_factor)
        else:
            self._n_shots += np.inf

        self._curr_energy = qc_res.get_hf_dot()

        return c, prop_residuals

    def get_propagated_return_amp_vector_fci(self, trial_amps):
        """Returns the propagated return amplitude vector with elements pertaining to all operators
        in the ansatz circuit.

        Parameters
        ----------
        trial_amps : list of floats
            The list of (real) floating point numbers which will characterize
            the state preparation circuit used in calculation of the residuals.
        """
        if(self._pool_type == 'sa_SD'):
            raise ValueError('Must use single term particle-hole nbody operators for residual calculation')
        
        if not self._ref_from_hf:
            raise ValueError('get_residual_vector_fci_comp only compatible with hf reference at this time.')
        
        temp_pool = qforte.SQOpPool()

        for tamp, top in zip(trial_amps, self._tops):
            temp_pool.add(tamp, self._pool_obj[top][1])

        prop_return_amps = []

        # det_indexes = [([0, 0], 1.0)] + self._excited_dets_fci_comp
        det_indexes = self._excited_dets_fci_comp

        qc_ra = qforte.FCIComputer(
                self._nel, 
                self._2_spin, 
                self._norb)

        for IaIb, _ in det_indexes:

            qc_ra.zero_state()

            # print(f"IaIb: {IaIb}")
            
            qc_ra.set_element(IaIb, 1.0)


            # ====> New Here <====


            # function assumers first order trotter, with 1 trotter step, and time = 1.0
            qc_ra.evolve_pool_trotter_basic(
                temp_pool,
                antiherm=True,
                adjoint=False)
            
            Cmu = qc_ra.get_state_deep()


            if(self._ppqe_trotter_order == np.inf):
                
                taylor_thresh = 1.0
                # niter = int(np.abs(self._frozen_core_energy) * self._dt / taylor_thresh) + 1
                niter = int((self._ham_l1_norm_sq**(0.25)) * self._dt / taylor_thresh) + 1

                # niter = 10

                if(self._apply_ham_as_tensor):

                    global_phase = np.exp(-1j * self._frozen_core_energy * self._dt)

                    for _ in range(niter):
                        qc_ra.evolve_tensor_taylor(
                                    0.0, 
                                    self._mo_oeis, 
                                    self._mo_teis, 
                                    self._mo_teis_einsum, 
                                    self._norb,
                                    float(self._dt/niter),
                                    1.0e-15,
                                    30,
                                    False)
                    
                    qc_ra.scale(global_phase)


                else:
                    for _ in range(niter):
                        qc_ra.evolve_op_taylor(
                                self._sq_ham,
                                float(self._dt/niter),
                                1.0e-15,
                                30,
                                False)
                
            elif(self._ppqe_trotter_order in [1,2]):
                qc_ra.evolve_pool_trotter(
                    self._hermitian_pairs,
                    self._dt,
                    1,
                    self._ppqe_trotter_order,
                    antiherm=False,
                    adjoint=False)

            else:
                raise ValueError(f"PPQE Trotter order {self._ppqe_trotter_order} is not supported.")

            Lmu = qc_ra.get_state_deep()
            
            r_mu_mu = Cmu.vector_dot(Lmu)
            
            if self._noise_factor > 1e-12:
                noisy_real = np.random.normal(np.real(r_mu_mu), self._noise_factor)
                noisy_imag = np.random.normal(np.imag(r_mu_mu), self._noise_factor)
                r_mu_mu = noisy_real + 1j * noisy_imag

            prop_return_amps.append(r_mu_mu)

            # NOTE(Nick): this might be over all excitations, we should only include those that are in the pool.

        if self._noise_factor > 1e-12:
            self._n_shots += 2.0 * (len(self._tamps) + 1.0) / (self._noise_factor * self._noise_factor)
        else:
            self._n_shots += np.inf

        return prop_return_amps
    

    def get_c0_cmu_rmu(self, mu, trial_amps):
        """Returns the propagated return amplitude vector with elements pertaining to all operators
        in the ansatz circuit.

        Parameters
        ----------
        trial_amps : list of floats
            The list of (real) floating point numbers which will characterize
            the state preparation circuit used in calculation of the residuals.
        """
        if(self._pool_type == 'sa_SD'):
            raise ValueError('Must use single term particle-hole nbody operators for residual calculation')
        
        if not self._ref_from_hf:
            raise ValueError('get_residual_vector_fci_comp only compatible with hf reference at this time.')
        
        temp_pool = qforte.SQOpPool()

        for tamp, top in zip(trial_amps, self._tops):
            temp_pool.add(tamp, self._pool_obj[top][1])

        # prop_return_amps = []

        IaIb_mu = self._excited_dets_fci_comp[mu][0]
        phase_mu = self._excited_dets_fci_comp[mu][1]


        qc_0 = qforte.FCIComputer(
                self._nel, 
                self._2_spin, 
                self._norb)
        
        qc_mu = qforte.FCIComputer(
                self._nel, 
                self._2_spin, 
                self._norb)

        

        # qc_ra.zero_state()
        # print(f"IaIb: {IaIb}")

        qc_0.hartree_fock()
        qc_mu.set_element(IaIb_mu, 1.0)


        # ====> New Here <====


        # function assumers first order trotter, with 1 trotter step, and time = 1.0
        qc_0.evolve_pool_trotter_basic(
            temp_pool,
            antiherm=True,
            adjoint=False)
        
        qc_mu.evolve_pool_trotter_basic(
            temp_pool,
            antiherm=True,
            adjoint=False)
        

        
        C0 = qc_0.get_state_deep()
        Cmu = qc_mu.get_state_deep()


        if(self._ppqe_trotter_order == np.inf):
            
            taylor_thresh = 1.0
            # niter = int(np.abs(self._frozen_core_energy) * self._dt / taylor_thresh) + 1
            niter = int((self._ham_l1_norm_sq**(0.25)) * self._dt / taylor_thresh) + 1

            # niter = 10

            if(self._apply_ham_as_tensor):

                global_phase = np.exp(-1j * self._frozen_core_energy * self._dt)

                for _ in range(niter):
                    qc_0.evolve_tensor_taylor(
                                0.0, 
                                self._mo_oeis, 
                                self._mo_teis, 
                                self._mo_teis_einsum, 
                                self._norb,
                                float(self._dt/niter),
                                1.0e-15,
                                30,
                                False)
                    
                    qc_mu.evolve_tensor_taylor(
                                0.0, 
                                self._mo_oeis, 
                                self._mo_teis, 
                                self._mo_teis_einsum, 
                                self._norb,
                                float(self._dt/niter),
                                1.0e-15,
                                30,
                                False)
                
                qc_0.scale(global_phase)
                qc_mu.scale(global_phase)

            else:
                for _ in range(niter):

                    qc_0.evolve_op_taylor(
                            self._sq_ham,
                            float(self._dt/niter),
                            1.0e-15,
                            30,
                            False)

                    qc_mu.evolve_op_taylor(
                            self._sq_ham,
                            float(self._dt/niter),
                            1.0e-15,
                            30,
                            False)
            
        elif(self._ppqe_trotter_order in [1,2]):
            qc_0.evolve_pool_trotter(
                self._hermitian_pairs,
                self._dt,
                1,
                self._ppqe_trotter_order,
                antiherm=False,
                adjoint=False)
            
            qc_mu.evolve_pool_trotter(
                self._hermitian_pairs,
                self._dt,
                1,
                self._ppqe_trotter_order,
                antiherm=False,
                adjoint=False)

        else:
            raise ValueError(f"PPQE Trotter order {self._ppqe_trotter_order} is not supported.")

        L0 = qc_0.get_state_deep()
        Lmu = qc_mu.get_state_deep()
        
        
        c_0 = C0.vector_dot(L0)
        c_mu = Cmu.vector_dot(Lmu)
        r_mu = C0.vector_dot(Lmu) * phase_mu
        
        if self._noise_factor > 1e-12:
            nre1 = np.random.normal(np.real(c_0), self._noise_factor)
            nre2 = np.random.normal(np.real(c_mu), self._noise_factor)
            nre3 = np.random.normal(np.real(r_mu), self._noise_factor)

            nim1 = np.random.normal(np.imag(c_0), self._noise_factor)
            nim2 = np.random.normal(np.imag(c_mu), self._noise_factor)
            nim3 = np.random.normal(np.imag(r_mu), self._noise_factor)

            c_0 = nre1 + 1j * nim1
            c_mu = nre2 + 1j * nim2
            r_mu = nre3 + 1j * nim3

        if self._noise_factor > 1e-12:
            self._n_shots += 6.0 / (self._noise_factor * self._noise_factor)
        else:
            self._n_shots += np.inf

        return c_0, c_mu, r_mu


    def initialize_ansatz(self):
        """Adds all operators in the pool to the list of operators in the circuit,
        with amplitude 0.
        """
        for l in range(len(self._pool_obj)):
            self._tops.append(l)
            self._tamps.append(0.0)

# S2PQE.jacobi_solver = optimizer.jacobi_solver
# S2PQE.scipy_solver = optimizer.scipy_solver
UCCNPPQE.rotation_solver = optimizer.rotation_solver
UCCNPPQE.sequential_rotation_solver = optimizer.sequential_rotation_solver
UCCNPPQE.construct_moment_space = moment_energy_corrections.construct_moment_space
UCCNPPQE.compute_moment_energies = moment_energy_corrections.compute_moment_energies
