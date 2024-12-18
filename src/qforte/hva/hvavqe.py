"""
HVAVQE classes
====================================
Classes for using an experiment to execute the variational quantum eigensolver
for a Trotterized (disentangeld) HVA ansatz with fixed operators.
"""

import qforte
from qforte.abc.uccvqeabc import UCCVQE

from qforte.experiment import *
from qforte.maths import optimizer
from qforte.utils.transforms import *
from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize
from qforte.utils import moment_energy_corrections

from qforte.utils.point_groups import sq_op_find_symmetry

import numpy as np
from scipy.optimize import minimize

class HVAVQE(UCCVQE):
    """A class that encompasses the three components of using the variational
    quantum eigensolver to optimize a parameterized Hamiltonian variational ansatze (HVA):
    (1) prepares a quantum state on the quantum computer
    representing the wave function to be simulated, (2) evauates the energy and
    gradients (3) optemizes the the wave funciton by minimizing the energy

    Attributes
    ----------
    _results : list
        The optimizer result objects from each iteration of HVA-VQE.

    _energies : list
        The optimized energies from each iteration of HVA-VQE.

    _grad_norms : list
        The gradient norms from each iteration of HVA-VQE.

    """
    def run(self,
            opt_thresh=1.0e-5,
            opt_ftol=1.0e-5,
            opt_maxiter=200,
            pool_type='SQHVA',
            optimizer='L-BFGS-B',
            use_analytic_grad = True,
            start_from_ham_params = True,
            noise_factor = 0.0):
        
        raise NotImplementedError("Warning, Second Quantized HVA-VQE needs debugging, would not use black box implementaiton.")
        
        if(self._computer_type != 'fci'):
            raise ValueError(f'{self._computer_type} is an unsupported computer type at this time, only fci supported.')

        self._opt_thresh = opt_thresh
        self._opt_ftol = opt_ftol
        self._opt_maxiter = opt_maxiter
        self._use_analytic_grad = use_analytic_grad
        self._optimizer = optimizer
        self._pool_type = pool_type
        self._noise_factor = noise_factor
        self._start_from_ham_params = start_from_ham_params

        self._tops = []
        self._tamps = []
        self._converged = 0

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0
        self._res_vec_evals = 0
        self._res_m_evals = 0
        self._k_counter = 0

        self._curr_grad_norm = 0.0

        self.print_options_banner()

        self._timer = qforte.local_timer()

        ######### HVA-VQE #########

        self._timer.reset()

        self.fill_hva_pool()

        self._timer.record("fill_pool")

        if self._verbose:
            print(self._pool_obj.str())

        self._timer.reset()
        self.initialize_ansatz(start_from_ham_params=start_from_ham_params)
        self._timer.record("initialize_ansatz")

        if(self._verbose):
            print('\nt operators included from pool: \n', self._tops)
            print('\nInitial tamplitudes for tops: \n', self._tamps)

        self._qc = qforte.FCIComputer(
            self._nel, 
            self._2_spin, 
            self._norb)
        
        self._qc.hartree_fock()

        self._timer.reset()
        self.solve()
        self._timer.record("solve")

        if(self._verbose):
            print('\nt operators included from pool: \n', self._tops)
            print('\nFinal tamplitudes for tops: \n', self._tamps)

        ######### HVA-VQE #########
        self._n_nonzero_params = 0
        for tmu in self._tamps:
            if(np.abs(tmu) > 1.0e-12):
                self._n_nonzero_params += 1

        # verify that required attributes were defined
        # (should be called for all algorithms!)
        # self.verify_run()

        self.print_summary_banner()

    # Define Algorithm abstract methods.
    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not fully implemented for HVA-VQE.')

    def fill_hva_pool(self):
        """ This function populates an operator pool with SQOperator objects corrseponding to
        the HVA.
        """

        if self._pool_type in {'SQHVA'}:
            self._pool_obj = qforte.SQOpPool()
            self._pool_obj.fill_pool_sq_hva(1.0, self._sq_ham)
            # self._pool_obj.add_hermitian_pairs(1.0, self._sq_ham)
        else:
            raise ValueError('Invalid operator pool type specified.')

        # TODO: (Nick) consider point-group symmetry considerations in hamiltonain...

        # If possible, impose symmetry restriction to operator pool
        # Currently, symmetry is supported for system_type='molecule' and build_type='psi4'
        if hasattr(self._sys, 'point_group'):
            # raise ValueError("nope!")
            temp_sq_pool = qforte.SQOpPool()
            for sq_operator in self._pool_obj.terms():
                create = sq_operator[1].terms()[0][1]
                annihilate = sq_operator[1].terms()[0][2]
                if sq_op_find_symmetry(self._sys.orb_irreps_to_int, create, annihilate) == self._irrep:
                    temp_sq_pool.add(sq_operator[0], sq_operator[1])
            self._pool_obj = temp_sq_pool

        if(self._computer_type == 'fock'):
            self._Nm = [len(operator.jw_transform().terms()) for _, operator in self._pool_obj]
        elif(self._computer_type == 'fci'):
            self._Nm = [0 for _, operator in self._pool_obj]
            print("\n ==> Warning: resource estimator needs to be implemented for fci computer type <==")
        else:
            raise ValueError(f'{self._computer_type} is an unrecognized computer type.')

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_VQE_attributes()
        self.verify_required_UCCVQE_attributes()

    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('          Unitary Coupled Cluster VQE   ')
        print('-----------------------------------------------------')

        print('\n\n               ==> HVA-VQE options <==')
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
        print('Start from ham params:                   ',  self._start_from_ham_params)
        print('Optimizer grad-norm threshold (theta):   ',  opt_thrsh_str)

        # UCCVQE options.
        print('Use analytic gradient:                   ',  str(self._use_analytic_grad))
        print('Operator pool type:                      ',  str(self._pool_type))
        print(f"Computer type:                            {self._computer_type}")
        b = False
        if (self._apply_ham_as_tensor):
            b = True
        print('Apply ham as tensor                      ', str(b))

    def print_summary_banner(self):

        print('\n\n                ==> HVA-VQE summary <==')
        print('-----------------------------------------------------------')
        print('Final HVA-VQE Energy:                      ', round(self._Egs, 10))
        if self._max_moment_rank:
            print('Moment-corrected (MP) HVA-VQE Energy:      ', round(self._E_mmcc_mp[0], 10))
            print('Moment-corrected (EN) HVA-VQE Energy:      ', round(self._E_mmcc_en[0], 10))
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

        print("\n\n")
        print(self._timer)

    def solve(self):
        if self._optimizer.lower() == "jacobi":
            self.build_orb_energies()
            return self.jacobi_solver()
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
        init_gues_energy = self.hva_energy_feval_fci(x0)
        self._prev_energy = init_gues_energy

        if self._use_analytic_grad:
            print('  \n--> Begin opt with analytic gradient:')
            print(f" Initial guess energy:              {init_gues_energy:+12.10f}")
            res =  minimize(
                self.hva_energy_feval_fci, 
                x0,
                method=self._optimizer,
                jac=self.hva_grad_arry_eval, 
                options=opts,
                callback=self.report_hva_iteration)

            for tmu in res.x:
                if(np.abs(tmu) > 1.0e-12):
                    self._n_pauli_trm_measures += int(2 * self._Nl * res.njev)

            self._n_pauli_trm_measures += int(self._Nl * res.nfev)


        else:
            print('  \n--> Begin opt with grad estimated using first-differences:')
            print(f" Initial guess energy:              {init_gues_energy:+12.10f}")
            res =  minimize(
                self.hva_energy_feval_fci, 
                x0,
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
        # self._n_cnot = self.build_Uvqc().get_num_cnots()
        self._n_cnot = 0


    def initialize_ansatz(self, start_from_ham_params=True):
        """Adds all operators in the pool to the list of operators in the circuit,
        with amplitude h_mu (using values of hamiltonain coefficeints).
        """

        self._tops = list(range(len(self._pool_obj)))
        self._tamps = list(range(len(self._pool_obj)))

        if(start_from_ham_params):
            self._tamps = [1.0] * len(self._pool_obj)
        else:
            self._tamps = [0.0] * len(self._pool_obj)


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
    
    def hva_energy_feval_fci(self, params):
            temp_pool = self._pool_obj
            temp_pool.set_coeffs(params)
            
            self._qc.hartree_fock()

            self._qc.evolve_pool_trotter_basic(
                temp_pool,
                antiherm=False,
                adjoint=False)
            
            if(self._apply_ham_as_tensor):
                
                val = self._qc.get_exp_val_tensor(
                        self._zero_body_energy, 
                        self._mo_oeis, 
                        self._mo_teis, 
                        self._mo_teis_einsum, 
                        self._norb)

            else:   
                val = self._qc.get_exp_val(self._sq_ham)

            self._curr_energy = np.real(val)
            return self._curr_energy 
    
    def hva_grad_fd_eval(self, params=None):
        delta = 1.0e-6

        M = len(self._tamps)
        grads = np.zeros(M)
        

        if params is None:
            params_mu = copy.deepcopy(self.t_amps)
        else:
            params_mu = copy.deepcopy(params)

        for mu in range(M):
            params_mu_up = copy.deepcopy(params_mu)
            params_mu_dn = copy.deepcopy(params_mu)

            tmu = params_mu[mu]
            tmu_up = tmu + 0.5 * delta
            tmu_dn = tmu - 0.5 * delta

            params_mu_up[mu] = tmu_up
            params_mu_dn[mu] = tmu_dn

            Eup = self.hva_energy_feval_fci(params_mu_up)
            Edn = self.hva_energy_feval_fci(params_mu_dn)

            grads[mu] = np.real((Eup - Edn) / delta)

        return grads


    def hva_grad_arry_eval(self, params=None):
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
        vqc_ops = qforte.SQOpPool()

        if params is None:
            for tamp, top in zip(self._tamps, self._tops):
                vqc_ops.add(tamp, self._pool_obj[top][1])
        else:
            for tamp, top in zip(params, self._tops):
                vqc_ops.add(tamp, self._pool_obj[top][1])

        # build | sig_N > according ADAPT-VQE analytical grad section
        qc_psi = qforte.FCIComputer(
            self._nel, 
            self._2_spin, 
            self._norb) 
        
        qc_psi.hartree_fock()
        
        qc_psi.evolve_pool_trotter_basic(
            vqc_ops,
            antiherm=False,
            adjoint=False)

        qc_sig = qforte.FCIComputer(
            self._nel, 
            self._2_spin, 
            self._norb) 

        psi_i = qc_psi.get_state_deep()

        # not sure if copy is faster or reapplication of state
        qc_sig.set_state(psi_i) 

        if(self._apply_ham_as_tensor):
            qc_sig.apply_tensor_spat_012bdy(
                self._zero_body_energy, 
                self._mo_oeis, 
                self._mo_teis, 
                self._mo_teis_einsum, 
                self._norb)
        else:   
            qc_sig.apply_sqop(self._sq_ham)

        mu = M-1

        # find <sing_N | K_N | psi_N>
        Kmu_prev = self._pool_obj[self._tops[mu]][1]
        Kmu_prev.mult_coeffs(self._pool_obj[self._tops[mu]][0])

        qc_psi.apply_sqop(Kmu_prev)
        # qc_psi.scale(-1.0j)
        
        grads[mu] = 2.0 * np.imag(qc_sig.get_state().vector_dot(qc_psi.get_state()))
        qc_psi.set_state(psi_i)


        # ==> End First Iteration <== #

        for mu in reversed(range(M-1)):

            # mu => N-1 => M-2
            # mu+1 => N => M-1
            # Kmu => KN-1
            # Kmu_prev => KN

            if params is None:
                tamp = self._tamps[mu+1]
            else:
                tamp = params[mu+1]

            Kmu = self._pool_obj[self._tops[mu]][1]

            Kmu.mult_coeffs(self._pool_obj[self._tops[mu]][0])

            # The minus sign is dictated by the recursive algorithm used to compute the analytic gradient
            # (see original ADAPT-VQE paper)
            qc_psi.apply_sqop_evolution(
                -1.0*tamp,
                Kmu_prev,
                antiherm=False,
                adjoint=False)
            
            qc_sig.apply_sqop_evolution(
                -1.0*tamp,
                Kmu_prev,
                antiherm=False,
                adjoint=False)

            psi_i = qc_psi.get_state_deep()

            qc_psi.apply_sqop(Kmu)
            # qc_psi.scale(-1.0j)

            grads[mu] = 2.0 * np.imag(
                qc_sig.get_state().vector_dot(qc_psi.get_state())
                )
            
            #reset Kmu |psi_i> -> |psi_i>
            qc_psi.set_state(psi_i)
            Kmu_prev = Kmu

        if(self._noise_factor > 1e-14):
            grads = [np.random.normal(np.real(grad_m), self._noise_factor) for grad_m in grads]

        self._curr_grad_norm = np.linalg.norm(grads)

        return grads
    
    def report_hva_iteration(self, x):

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
    
    def get_num_commut_measurements(self):
        pass

HVAVQE.jacobi_solver = optimizer.jacobi_solver
HVAVQE.construct_moment_space = moment_energy_corrections.construct_moment_space
HVAVQE.compute_moment_energies = moment_energy_corrections.compute_moment_energies
