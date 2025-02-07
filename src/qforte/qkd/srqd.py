"""
SRQD classes
=================================================
Classes for using a quantum computer to carry
out the quantum imaginary time evolution algorithm.
"""
import qforte as qf
from qforte.abc.algorithm import Algorithm
from qforte.utils.transforms import (get_jw_organizer,
                                    organizer_to_circuit)

from qforte.utils.state_prep import *
from qforte.utils.trotterization import trotterize
from qforte.helper.printing import *
import copy
import numpy as np
from scipy.linalg import lstsq

from qforte.maths.eigsolve import canonical_geig_solve

### Throughout this file, we'll refer to DOI 10.1038/s41567-019-0704-4 as Motta.

class SRQD(Algorithm):
    """A hybrid subspace diagonalization algorithm that generates the many-body
    basis from different durations of imaginary time evolution:

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
            target_root=0,
            max_itr=50,
            thresh=1e-9,
            dt=1.0e-6,
            expansion_type='SD',
            low_memorySb=False,
            b_thresh=1.0e-6,
            x_thresh=1.0e-10,
            fname=None):

        self._target_root = target_root
        self._max_itr = max_itr
        self._thresh = thresh
        self._dt = dt

        self._sz = 0

        self._expansion_type = expansion_type
        self._low_memorySb = low_memorySb
        self._b_thresh = b_thresh
        self._x_thresh = x_thresh

        self._n_classical_params = 0
        self._n_cnot = self._Uprep.get_num_cnots()
        self._n_pauli_trm_measures = 0

        self._fname = fname

        raise NotImplementedError("SRQD is currently broken")

        if(self._fname is None):
            self._fname = f'fix_me'

        if(self._computer_type!='fci'):
            raise NotImplementedError("SRQD only implemented for fci comp")

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        # Build expansion pool.
        self.build_expansion_pool()

        # big function that runs everything
        self.run_davidson()

        # Print summary banner (should done for all algorithms).
        self.print_summary_banner()

        # verify that required attributes were defined
        # (should done for all algorithms).
        self.verify_run()

    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not yet implemented for QITE.')


    def verify_run(self):
        self.verify_required_attributes()


    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('            Single Reference Quantum Davidson          ')
        print('-----------------------------------------------------')

        print('\n\n              ==> SRQD options <==                   ')
        print('-----------------------------------------------------------')
        # General algorithm options.
        print('Computer Type:                           ',  self._computer_type)
        print('Trial reference state:                   ',  ref_string(self._ref, self._nqb))
        print('Number of Hamiltonian Pauli terms:       ',  self._Nl)
        print('Trial state preparation method:          ',  self._state_prep_type)
        print('Trotter order (rho):                     ',  self._trotter_order)
        print('Trotter number (m):                      ',  self._trotter_number)
        print('Use fast version of algorithm:           ',  str(self._fast))
        if(not self._fast):
            print('Measurement variance thresh:             ',  0.01)

        # Specific QITE options.
        # print('Total imaginary evolution time (beta):   ',  self._beta)
        print('Linear mapping time step (db):            ',  self._dt)
        # print('Use exact evolution:                     ',  self._use_exact_evolution)
        print('Expansion type:                          ',  self._expansion_type)
        print('x value threshold:                       ',  self._x_thresh)
        # print('Use sparse tensors to solve Sx = b:      ',  str(self._sparseSb))
        # if(self._sparseSb):
            # print('b value threshold:                       ',  str(self._b_thresh))
        print('\n')
        print('Use low memory mode:                     ',  self._low_memorySb)
        # print('Do Quantum Lanczos                       ',  str(self._do_lanczos))
        # if(self._do_lanczos):
        #     print('Lanczos gap size                         ',  self._lanczos_gap)


    def print_summary_banner(self):
        print('\n\n                        ==> SRQD summary <==')
        print('-----------------------------------------------------------')
        print('Final QITE Energy:                        ', round(self._Egs, 10))
        print('Number of operators in pool:              ', self._NI)
        print('Number of classical parameters used:      ', self._n_classical_params)
        print('Estimated classical memory usage (GB):    ', f'{self._total_memory * 10**-9:e}')
        print('Number of CNOT gates in deepest circuit:  ', self._n_cnot)
        print('Number of Pauli term measurements:        ', self._n_pauli_trm_measures)


    def build_expansion_pool(self):
        print('\n==> Building expansion pool <==')

        self._sig = qf.SQOpPool() 
        self._sig.set_orb_spaces(self._ref)

        if(self._expansion_type in {'SD', 'GSD', 'SDT', 'SDTQ', 'SDTQP', 'SDTQPH'}):
            self._sig.fill_pool(self._expansion_type) # This automatically filters non-particle conserving terms

        else:
            raise ValueError('Invalid expansion type specified.')

        self._NI = len(self._sig.terms())


    def diagonalize_subspace(self):
        """Solve the generalized eigenvalue problem HV=ESV whithin the current iteration Krylov subspace
        """
        self._Uomega_lst = []
        self._HUomega_lst = []

        h_mat = np.zeros((len(self._omega_lst),len(self._omega_lst)), dtype=complex)
        s_mat = np.zeros((len(self._omega_lst),len(self._omega_lst)), dtype=complex)

        for i in range(len(self._omega_lst)):
            self._qc.set_state(self._omega_lst[i])

            self._qc.evolve_pool_trotter_basic(self._sig,
                                            antiherm=True,
                                            adjoint=False)
            self._Uomega_lst.append(self._qc.get_state_deep())

            if(self._apply_ham_as_tensor):
                self._qc.apply_tensor_spat_012bdy(
                    self._zero_body_energy, 
                    self._mo_oeis, 
                    self._mo_teis, 
                    self._mo_teis_einsum, 
                    self._norb)
            else:   
                self._qc.apply_sqop(self._sq_ham)

            self._HUomega_lst.append(self._qc.get_state_deep())

            for j in range(i+1):
                h_mat[j,i] = h_mat[i,j] = self._Uomega_lst[i].vector_dot(self._HUomega_lst[j])
                s_mat[j,i] = s_mat[i,j] = self._Uomega_lst[i].vector_dot(self._Uomega_lst[j])

        if(self._verbose):
            # print('\nbtot:\n ', btot)
            print('\n S_mat davidson:  \n')
            matprint(s_mat)
            print('\n H_mat davidson:  \n')
            matprint(h_mat)


        self._S_inv = np.linalg.inv(s_mat)

        self._evals, self._evecs = canonical_geig_solve(s_mat,
                                h_mat,
                                print_mats=False,
                                sort_ret_vals=True)
        self._scond = np.linalg.cond(s_mat)

        self._root_val = self._evals[self._target_root]
        self._root_vec = self._evecs[self._target_root]

        self._Egs = np.real(np.real(self._root_val))

        if(self._verbose):
            print('\nsubspace root eig val:\n', self._root_val)

        if(self._verbose):
            print('\nsubspace root eig vec:\n', self._root_vec)


    def build_unitary(self):
        """Solve the linear system for the sqop pool coefficients (approx unitary)
        """
        S, btot = self.second_order_mapping()
        x = lstsq(S, btot)[0]
        x = np.real(x)
        x_list = x.tolist()
        x_list_fci = [x*self._dt for x in x_list]

        if(self._verbose):
            print('\nbtot:\n ', btot)
            print('\n S:  \n')
            matprint(S)
            print('\n x:  \n')
            print(x)

        # self._sig.set_coeffs(x_list)
        self._sig.set_coeffs(x_list_fci)


    def second_order_mapping(self):
        """Construct the matrix M (eq. 9) and vector b (eq. 10) of Tsuchimochi, with non-local Hamiltonian approx, utilizing FCIComputer class.
        """
        Idim = self._NI

        self._n_pauli_trm_measures += int(self._NI*(self._NI+1)*0.5)
        self._n_pauli_trm_measures += self._Nl * self._NI

        # Initialize linear system
        S = np.zeros((Idim, Idim), dtype=complex)
        b = np.zeros(Idim, dtype=complex)

        # if(self._second_order):
        prefactor = -2.0
        S_factor = 2.0

        Ipsi_qc = qf.FCIComputer(self._nel, self._sz, self._norb)
        Hpsi_qc = qf.FCIComputer(self._nel, self._sz, self._norb)
        Hpsi_qc.set_state(self._qc.get_state_deep())
 
        Hpsi_qc.apply_sqop(self._temp_sqop) #FLAG

        if(self._low_memorySb):
            for i in range(Idim):
                S[i][i] = 1.0 # With Pauli strings, this is always the inner product

                # initialize state and apply pool term
                Ipsi_qc.set_state(self._qc.get_state_deep())
                Ipsi_qc.apply_sqop(self._sig.terms()[i][1])
                Ipsi_mu = Ipsi_qc.get_state_deep()

                # build b (second order variation)
                # if(self._second_order):
                exp_val = Hpsi_qc.get_state_deep().vector_dot(Ipsi_mu)
                b[i] = prefactor * exp_val

                # populate lower triangle of S and copy conjugate to upper triangle
                for j in range(i):
                    Ipsi_qc.set_state(self._qc.get_state_deep())
                    Ipsi_qc.apply_sqop(self._sig.terms()[j][1])

                    S[i][j] = Ipsi_mu.vector_dot(Ipsi_qc.get_state_deep())
                    S[j][i] = S[i][j].conj()

            return S_factor * np.real(S), np.real(b)

        else:
            rho_psi = []
            for i in range(Idim):
                S[i][i] = 1.0 # With Pauli strings, this is always the inner product
                
                # initialize state and apply pool term
                Ipsi_qc.set_state(self._qc.get_state_deep())
                Ipsi_qc.apply_sqop(self._sig.terms()[i][1])
                rho_psi.append(Ipsi_qc.get_state_deep())

                # build b (second order variation)
                # if(self._second_order):
                exp_val = Hpsi_qc.get_state_deep().vector_dot(rho_psi[i])
                b[i] = prefactor * exp_val

                # populate lower triangle of S and copy conjugate to upper triangle
                for j in range(i):
                    S[i][j] = rho_psi[i].vector_dot(rho_psi[j])
                    S[j][i] = S[i][j].conj()

            return S_factor * np.real(S), np.real(b)


    def build_correction_vector(self):
        """Build correction vector at current iteration using unitary approximation
        """
        # self._scaled_delta_vec.zero()
        self._qc.evolve_pool_trotter_basic(self._sig,
                                        antiherm=True,
                                        adjoint=False)

        delta_vec = self._qc.get_state_deep()
        # USE ZAXPY HERE IN ORDER TO AVOID STORING EXTRA COPIES OF TENSORS
        for k in range(len(self._omega_lst)):
            for j in range(len(self._omega_lst)):

                # print('\ndelta vec:\n', delta_vec)

                self._qc.set_state(self._omega_lst[j])

                s_factor = self._S_inv[k,j] * self._qc.get_state_deep().vector_dot(delta_vec)
                # print(f's-factor: {s_factor} at k: {k}, j: {j}')

                # temp_vec = delta_vec.scale(s_factor)
                # print('\ntemp vec:\n', temp_vec)

                # assert temp_vec is not None
                # assert self._scaled_delta_vec is not None
                self._qc.set_state(self._omega_lst[k])
                # self._scaled_delta_vec.add(temp_vec)
                self._scaled_delta_vec.zaxpy(x=self._qc.get_state_deep(), alpha=s_factor)

        self._dp.copy_in(delta_vec)
        
        # print('\ndelta vec:\n', self._dp)
        # print('\nscaled vec:\n', self._scaled_delta_vec)

        self._dp.subtract(self._scaled_delta_vec)
        # print(self._dp)
        # print(self._dp.norm()**2)

        # check_ortho = self._dp.norm()**2 / self._delta_norm

    def run_davidson(self):
        self._omega_lst = []
        
        self._qc = qf.FCIComputer(self._nel, self._sz, self._norb)
        PSI = self._qc.get_state_deep() # empty tensor for zaxpy routine, same dimension as element of guess space
        self._dp = self._qc.get_state_deep()
        self._scaled_delta_vec = self._qc.get_state_deep()

        self._qc.hartree_fock()
        C_0 = self._qc.get_state_deep()

        qc_size = self._qc.get_state().size()

        if(self._low_memorySb):
            self._total_memory = 5.0 * 16.0 * qc_size
        else:
            self._total_memory = self._NI * 16.0 * qc_size

        if(self._total_memory > 8.0e9 and not self._low_memorySb):
            print('\n')
            print('WARNING: ESTIMATED MEMORY USAGE EXCEEDS 8GB, SWITCHING TO LOW MEMORY MODE')
            print('\n')
            self._low_memorySb = True
            self._total_memory = 5.0 * 16.0 * qc_size # 5 corresponds to total # of tensors at any given time in memory

        timer = qf.local_timer()
        timer.reset()

        print(f"{'Iteration':>7}{'k(S)':>7}{'E(Npar)':>19}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}\n")
        print('-----------------------------------------------------------------------------------------------\n')

        if (self._print_summary_file):
            f = open(f"{self._fname}_summary.dat", "w+", buffering=1)
            f.write(f"#{'Iteration':>7}{'k(S)':>7}{'E(Npar)':>19}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}\n")
            f.write('#----------------------------------------------------------------------------------------------\n')

        for m in range(1, self._max_itr + 1):

            if(m<=1):
                self._omega_lst.append(C_0)
                self._sig.set_coeffs(np.zeros(self._NI).tolist())

            # build H, S mats and solve generalized eigenvalue problem HV = ESV within the current Krylov subspace
            self.diagonalize_subspace()

            print(f'{m}  {self._scond:7.2e}  {np.real(self._root_val):+15.9f}  {self._n_classical_params:8}   {self._n_cnot:10}   {self._n_pauli_trm_measures:12}')
            if (self._print_summary_file):
                f.write(f'{m}  {self._scond:7.2e}    {np.real(self.self._root_val):+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}\n')

            # for a, v in zip(self._root_vec, self._omega_lst):
            #     PSI.zaxpy(x = v, alpha = a)
            # if(self._verbose):
            #     print('\nPSI before zaxpy:\n', PSI)
            
            # state_list = []
            for i in range(len(self._omega_lst)):

                for j in range(len(self._omega_lst)):
                    self._qc.set_state(self._omega_lst[j])
                    PSI.zaxpy(x=self._qc.get_state_deep(), alpha=self._evecs[i][j])

                self._qc.set_state(PSI)
                curr_state = self._qc.get_state_deep()
                PSI.zero()

                # if(self._verbose):
                #     print('\nPSI after zaxpy:\n', PSI)

                # self._qc.set_state(state_list[i])
                
                # calculate norm of (H - E)|psi> and expectation value of (H - E + 1/dt)|psi> and append to EKb list (this would be done on a classical computer)
                self._temp_sqop = qf.SQOperator()
                self._temp_sqop.add_op(self._sq_ham)
                # self._temp_sqop.mult_coeffs(-1)
                self._temp_sqop.add_term(-self._evals[i], [], [])

                self._qc.apply_sqop(self._temp_sqop) # ask Nick if there is a better way to do this
                self._qc.apply_sqop(self._temp_sqop)

                self._r_norm = np.real(curr_state.vector_dot(self._qc.get_state_deep()))

                if(self._verbose):
                    print('r norm:', self._r_norm)

                if(self._r_norm > self._thresh):
                    self._qc.set_state(curr_state)

                    # self._temp_sqop.add_term(1/self._dt, [], [])

                    # if(self._verbose):
                    #     print('\nsqop used to calculate expectation value:\n', self._temp_sqop)

                    self._Ekb = np.real(self._qc.get_exp_val(self._temp_sqop))
                    
                    if(self._verbose):
                        print('\nexpectation value:\n', self._Ekb)

                    # map residual vector construction to unitary evolution
                    self.build_unitary()

                    # determine orthogonality of current iteration guess vector
                    self.build_correction_vector()

                    if(self._dp.norm()**2 < self._thresh):
                        self._omega_lst.append(self._dp)

            if(self._verbose):
                print(f'\nguess vectors at completion of iteration {m}\n')
                for i in range(len(self._omega_lst)):
                    print(f'gv {i}\n', self._omega_lst[i])


                # else:
                #     self._Egs = np.real(np.real(self._root_val))
                #     print('\n')
                #     print(f'Qdavidson converged for root {self._target_root} iteration {m}, Energy: {self._Egs}')
                #     break


        timer.record('Total davidson time')
        print(timer)

        if (self._print_summary_file):
            f.close()
    
    # def evolve(self):
    #     """Perform QITE for a time step :math:`\\Delta \\beta`.
    #     """

    #     print(f"{'beta':>7}{'E(beta)':>18}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}")
    #     print('-------------------------------------------------------------------------------')
    #     print(f' {0.0:7.3f}    {self._Ekb[0]:+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}')

    #     if (self._print_summary_file):
    #         f = open(f"qite_{self._fname}_summary.dat", "w+", buffering=1)
    #         f.write(f"#{'beta':>7}{'E(beta)':>18}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}\n")
    #         f.write('#-------------------------------------------------------------------------------\n')
    #         f.write(f'  {0.0:7.3f}    {self._Ekb[0]:+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}\n')

    #     for kb in range(1, self._number_of_subspace_vectors):
    #         self.solve_Sb()

    #         # Q DAVIDSON STUFF HERE MAYBE?

    #         print(f' {kb*self._db:7.3f}    {self._Ekb[kb]:+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}')
    #         if (self._print_summary_file):
    #             f.write(f'  {kb*self._db:7.3f}    {self._Ekb[kb]:+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}\n')
    #     self._Egs = self._Ekb[-1]

    #     if (self._print_summary_file):
    #         f.close()

    def print_expansion_ops(self):
        print('\nunitary expansion operators:')
        print('-------------------------')
        print(self._sig.str())
