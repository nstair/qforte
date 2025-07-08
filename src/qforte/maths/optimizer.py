import copy
import numpy as np
from scipy.optimize import minimize

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
        print('\n    k iteration         Energy               dE           Nrvec ev      Nrm ev*         ||r||           Nshots')
    elif self.__class__.__name__ in ['UCCNVQE', 'ADAPTVQE']:
        print('\n    k iteration         Energy               dE           Ngvec ev      Ngm ev*         ||g||           Nshots')
    print('----------------------------------------------------------------------------------------------------------------------------', flush=True)

    if (self._print_summary_file):
        f = open("summary.dat", "w+", buffering=1)
        f.write('\n#    k iteration         Energy               dE           Nrvec ev      Nrm ev*         ||r||           Nshots')
        f.write('\n#--------------------------------------------------------------------------------------------------------------------')
        f.close()

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
            print(f'     {k:7}        {Ek:+12.10f}      {dE:+12.10f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {self._res_vec_norm:+12.10f}       {self._n_shots:2.3e}')

            if (self._print_summary_file):
                f = open("summary.dat", "a", buffering=1)
                f.write(f'\n       {self._k_counter:7}        {self._curr_energy:+12.12f}      {dE:+12.12f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {self._res_vec_norm:+12.12f}       {self._n_shots:2.3e}')
                f.close()

            if(self._res_vec_norm < self._opt_thresh):
                self._Egs = Ek
                break

        elif self.__class__.__name__ in ['UCCNVQE', 'ADAPTVQE']:
            print(f'     {k:7}        {Ek:+12.10f}      {dE:+12.10f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {self._curr_grad_norm:+12.10f}       {self._n_shots:2.3e}')
            
            if (self._print_summary_file):
                f = open("summary.dat", "a", buffering=1)
                f.write(f'\n       {self._k_counter:7}        {self._curr_energy:+12.12f}      {dE:+12.12f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {self._res_vec_norm:+12.12f}       {self._n_shots:2.3e}')
                f.close()
            
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

    if hasattr(self, '_n_shots_k'):
            self._n_shots_k += self._n_shots

    if hasattr(self, '_n_shots_lst'):
        self._n_shots_lst.append(self._n_shots_k)

    if hasattr(self, '_n_cnot'):
        if(self._computer_type == 'fock'):
            self._n_cnot = self.build_Uvqc().get_num_cnots()
        else:
            # TODO: Build resource estimator
            self._n_cnot = 'N/A'
    if hasattr(self, '_n_cnot_lst'):
        if(self._computer_type == 'fock'):
            self._n_cnot_lst.append(self.build_Uvqc().get_num_cnots())
        else:
            # TODO: Build resource estimator
            self._n_cnot_lst.append('N/A')

def rotation_solver(self):
    """
    This function minimizes the norm of the residual/gradient vector
    by using a quasi-Newton update procedure for the amplitudes
    """

    t_diis = [copy.deepcopy(self._tamps)]
    e_diis = []
    Ek0 = self.energy_feval(self._tamps)

    if self.__class__.__name__ in ['UCCNPPQE', 'S3PQE']:
        print('\n    k iteration         Energy               dE           Nrvec ev      Nrm ev*         ||r||           Nshots')

    print('--------------------------------------------------------------------------------------------------------------------', flush=True)

    if (self._print_summary_file):
        f = open("summary.dat", "w+", buffering=1)
        f.write('\n#    k iteration         Energy               dE           Nrvec ev      Nrm ev*         ||r||           Nshots')
        f.write('\n#--------------------------------------------------------------------------------------------------------------------')
        f.close()

    for k in range(1, self._opt_maxiter+1):

        t_old = copy.deepcopy(self._tamps)

        #do regular update
        if self.__class__.__name__ in ['UCCNPPQE', 'S3PQE']:
            c, d = self.get_propogated_residual_vector(self._tamps)
        else:
            raise ValueError('Rotation solver is not implemented for this algorithm.')
        
        
        # =====> IMPORTANT: Divide my MP denom?? <=====
        # d = self.get_res_over_mpdenom(d)
        # self._tamps = list(np.add(self._tamps, r_k))

        # =====> 1st, inspired by origional residual update <=====
        if(self._ppqe_update_type == 'jacobi_like'):
            d = self.get_im_res_over_mpdenom_dt(d, self._dt)
            # d = self.get_im_res_times_eidt_mpdenom(d, self._dt)
            # self._tamps = list(np.add(self._tamps, d))
            self._tamps = list(np.subtract(self._tamps, d))
        
        # =====> 2nd, Two determinant subspace rotation <=====
        elif(self._ppqe_update_type == 'two_level_rotation' or self._ppqe_update_type == 'two_level_rotation_im'):

            r_mu_mu_vec = self.get_propogated_return_amp_vector(self._tamps)
            delta_t = []

            if (self._ppqe_update_type == 'two_level_rotation'):
                # ====> Try solving real part of Equation <=====
                # ====> Smaller values, but more numerically stable (arctan...) <=====
                for r_mu_mu, r_mu in zip(r_mu_mu_vec, d):
                    delta_t_mu = 0.5 * np.arctan( -2.0 * np.real( r_mu ) / (np.real(r_mu_mu) - np.real(c) ) )
                    delta_t.append(delta_t_mu)

            elif (self._ppqe_update_type == 'two_level_rotation_im'):

                # ====> Try solving imaginary part of Equation <=====
                # ====> Preferable becuse imaginary part is generally larger (odd terms) <=====
                # note that it should be (+) in the equation but (-) seems to work better...
                for r_mu_mu, r_mu in zip(r_mu_mu_vec, d):
                    delta_t_mu = 0.5 * np.arcsin( 2.0 * np.imag(r_mu) / (np.imag(r_mu_mu) - np.imag(c) ) )
                    delta_t.append(delta_t_mu)

            else:
                raise ValueError(f"Unknown update type: {self._ppqe_update_type}")
            
            # ====> this may be nonsense <=====
            # for r_mu_mu, r_mu in zip(r_mu_mu_vec, d):
            #     delta_t_mu = 0.5 * np.real( np.arctan( 4.0 * r_mu / (c - r_mu_mu) ) )
            #     delta_t.append(delta_t_mu)

            # print(f"delta_t: {delta_t}\n")

            # =====> New update that does no work <=====
            self._tamps = [
                t_prev + delta_t_mu for t_prev, delta_t_mu in zip(self._tamps, delta_t)
            ]

            # =====> Incorrect old update that sort-of works <=====
            # self._tamps = [
            #     t_prev + 0.5 * np.arctan( np.real(d_mu) ) for t_prev, d_mu in zip(self._tamps, d)
            # ]


        else:
            raise ValueError(f"Unknown update type: {self._ppqe_update_type}")

        for t in self._tamps:
            if np.imag(t) > 1.0e-12:
                print(f" Im(t): {np.imag(t)}")
                print(f" Re(t): {np.real(t)}")
                print(f"\n\n")
                raise ValueError('Imaginary t_amp encountered during rotation solver.')

        Ek = self.energy_feval(self._tamps)
        dE = Ek - Ek0
        Ek0 = Ek

        if self.__class__.__name__ in ['UCCNPPQE', 'S3PQE']:
            print(f'     {k:7}        {Ek:+12.10f}      {dE:+12.10f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {self._res_vec_norm:+12.10f}       {self._n_shots:2.3e}')
            
            if (self._print_summary_file):
                f = open("summary.dat", "a", buffering=1)
                # f.write(f'\n       {self._k_counter:7}        {self._curr_energy:+12.12f}      {dE:+12.12f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {self._res_vec_norm:+12.12f}       {self._n_shots:2.3e}')
                f.write(f'\n     {k:7}        {Ek:+12.10f}      {dE:+12.10f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {self._res_vec_norm:+12.10f}       {self._n_shots:2.3e}')
                f.close()

            # NOTE(Nick): This is the original convergence criterion, but it is not     
            #  working well for the rotation solver, so we are using the dE criterion 
            # if(self._res_vec_norm < self._opt_thresh):
            #     self._Egs = Ek
            #     break

            if(np.abs(dE) < self._opt_e_thresh):
                self._Egs = Ek
                break

        t_diis.append(copy.deepcopy(self._tamps))
        e_diis.append(np.subtract(copy.deepcopy(self._tamps), t_old))

        if(k >= 1 and self._diis_max_dim >= 2):
            self._tamps = diis(self._diis_max_dim, t_diis, e_diis)

    #NOTE(Nick): Update resource estimates for UCCNPPQE and S3PQE
    #NOTE(Nick): This function only gets called once for fixed ansatz, 

    self._Egs = Ek
    if k == self._opt_maxiter:
        print("\nMaximum number of Jacobi iterations reached!")
    if hasattr(self, '_energies'):
        self._energies.append(Ek)
    if hasattr(self, '_n_classical_params'):
        self._n_classical_params = len(self._tamps)
    #NOTE(Nick): Pauli count is fishy here, seems like we should not be multiplying by k
    # if we are accumulating the number of measurements via +=.
    if hasattr(self, '_n_pauli_measures_k'):
        self._n_pauli_measures_k += self._Nl*k * (2*len(self._tamps) + 1)
    if hasattr(self, '_n_pauli_trm_measures'):
        self._n_pauli_trm_measures += 2*self._Nl*k*len(self._tamps) + self._Nl*k
    if hasattr(self, '_n_pauli_trm_measures_lst'):
        self._n_pauli_trm_measures_lst.append(self._n_pauli_measures_k)

    if hasattr(self, '_n_shots_k'):
        self._n_shots_k += self._n_shots

    if hasattr(self, '_n_shots_lst'):
        self._n_shots_lst.append(self._n_shots_k)

    if hasattr(self, '_n_cnot'):
        if(self._computer_type == 'fock'):
            self._n_cnot = self.build_Uvqc().get_num_cnots()
        else:
            # TODO: Build resource estimator for FCI Case
            self._n_cnot = 'N/A'
    if hasattr(self, '_n_cnot_lst'):
        if(self._computer_type == 'fock'):
            self._n_cnot_lst.append(self.build_Uvqc().get_num_cnots())
        else:
            # TODO: Build resource estimator
            self._n_cnot_lst.append('N/A')

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

    if hasattr(self, '_n_shots_k'):
                self._n_shots_k += self._n_shots

    if hasattr(self, '_n_shots_lst'):
        self._n_shots_lst.append(self._n_shots_k)

    if hasattr(self, '_n_cnot'):
        self._n_cnot = self.build_Uvqc().get_num_cnots()
    if hasattr(self, '_n_cnot_lst'):
        self._n_cnot_lst.append(self.build_Uvqc().get_num_cnots())
