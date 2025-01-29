"""
CIS classes
=================================================
Classes for calculating reference states for quantum
mechanical systems using configuration interation Singles.
"""
import copy

import qforte as qf
from qforte.abc.qsdabc import QSD
from qforte.helper.printing import matprint
from qforte.utils.point_groups import sq_op_find_symmetry

from qforte.maths.eigsolve import canonical_geig_solve
from qforte.maths import gram_schmidt

from qforte.utils.state_prep import *
from qforte.utils.trotterization import (trotterize,
                                         trotterize_w_cRz)

# from qforte.helper.printing import matprint

import numpy as np

class CIS(QSD):
    """Runs CIS (brute force). In present implementation, will generate all CIS singlet states. Uses FCI Computer.

    Attributes
    ----------

    _target_root : int
        The target eigenstate for cis.


    """
    def run(self,
            target_root=0,
            diagonalize_each_step=False,
            low_memory=False
            ):

        self._target_root = target_root
        self._diagonalize_each_step = diagonalize_each_step
        self._low_memory = low_memory

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0

        self._pool_obj = qforte.SQOpPool()
        self._pool_obj.set_orb_spaces(self._ref)

        #NOTE(Nick) May need to implement
        identity = qf.SQOperator()
        identity.add_term(1.0, [], [])
        self._pool_obj.add_term(1.0, identity)
        self._pool_obj.fill_pool('S')

        # print(f"self._sys.point_group: {self._sys.point_group}")

        #NOTE(Nick): For some reason CIS is not inheriting the irrep info, will always seek totally symmetric irrep...
        if hasattr(self._sys, 'point_group'):
            print('\nWARNING: The {0} point group was detected, but no irreducible representation was specified.\n'
                        '         Proceeding with totally symmetric.\n'.format(self._sys.point_group[0].capitalize()))
            self._irrep = 0
            
            temp_sq_pool = qf.SQOpPool()
            for sq_operator in self._pool_obj.terms():
                create = sq_operator[1].terms()[0][1]
                annihilate = sq_operator[1].terms()[0][2]
                if sq_op_find_symmetry(self._sys.orb_irreps_to_int, create, annihilate) == self._irrep:
                    temp_sq_pool.add(sq_operator[0], sq_operator[1])

            self._pool_obj = temp_sq_pool

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        self._timer = qforte.local_timer()
        self._timer.reset()
        self.run_cis()
        self._timer.record("Run")

        # Temporary solution, replacing Egs with lambda_low
        self._Egs = self._sys.hf_energy

    # Define Algorithm abstract methods.
    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not fully implemented for CIS.')

    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_QSD_attributes()

    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('           Configuaration Interaction Singles   ')
        print('-----------------------------------------------------')

        print('\n\n                     ==> CIS options <==')
        print('-----------------------------------------------------------')
        # General algorithm options.
        print('Trial reference state:                   ',  ref_string(self._ref, self._nqb))
        print('Number of Hamiltonian Pauli terms:       ',  self._Nl)
        print('Trial state preparation method:          ',  self._state_prep_type)

        # Specific CIS options.
        print('Target root:                             ',  str(self._target_root))


    def print_summary_banner(self):

        print('\n\n                     ==> CIS summary <==')
        print('-----------------------------------------------------------')
        print('Final CIS ground state Energy:           ', round(self._Egs, 10))
        print('Final CIS target state Energy:           ', round(self._Ets, 10))
        print('Estimated classical memory usage (GB):    ', f'{self._total_memory * 10**-9:e}')
        print('Number of classical parameters used:       ', self._N_cis)
        
       

        print("\n\n")
        print(self._timer)

    def build_qk_mats(self):
        return self.build_cis_mats()

    def build_cis_mats(self):
        if(self._computer_type == 'fock'):
            raise ValueError("fock computer CIS not currently implemented")
        
        elif(self._computer_type == 'fci'):
            
            return self.build_cis_mats_fci()
        
        else:
            raise ValueError(f"{self._computer_type} is an unrecognized computer type.") 
    
    
    def build_cis_mats_fci(self):
        """Returns matrices H needed for the CIS algorithm 

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

        if(self._diagonalize_each_step):
            print('\n\n')

            print(f"{'k(S)':>7}{'E(Npar)':>19}{'dE(Npar)':>19}{'N(params) =====':>14}")
            print('--------------------------------------------------------------------------')

            if (self._print_summary_file):
                f = open("summary.dat", "w+", buffering=1)
                f.write(f"#{'k(S)':>7}{'E(Npar)':>19}{'dE(Npar)':>19}{'N(params)':>14}\n")
                f.write('#-------------------------------------------------------------------------------\n')

        self._qc = qforte.FCIComputer(
                self._nel, 
                self._2_spin, 
                self._norb)

        self._N_cis = len(self._pool_obj.terms())

        H = np.zeros((self._N_cis, self._N_cis), dtype=complex)
        S = np.eye(self._N_cis, dtype=complex)

        
        qc_size = self._qc.get_state().size()
        if(self._low_memory):
            self._total_memory = 2.0 * 16.0 * qc_size
        else:
            self._total_memory = 2.0 * self._N_cis * 16.0 * qc_size

        if(self._total_memory > 8.0e9 and not self._low_memory):
            print('\n')
            print('WARNING: ESTIMATED MEMORY USAGE EXCEEDS 8GB, SWITCHING TO LOW MEMORY MODE')
            print('\n')
            self._low_memory = True
            self._total_memory = 2.0 * 16.0 * qc_size 

        if(self._low_memory):
            for i in range(self._N_cis):
                self._qc.hartree_fock()
                self._qc.apply_sqop(self._pool_obj.terms()[i][1])

                IJ_lst = self._qc.get_nonzero_idxs()
                self._qc.set_element(IJ_lst[0], 1.0)

                rho_psi = self._qc.get_state_deep()

                for j in range(i+1):
                    self._qc.hartree_fock()
                    self._qc.apply_sqop(self._pool_obj.terms()[j][1])

                    IJ_lst = self._qc.get_nonzero_idxs()
                    self._qc.set_element(IJ_lst[0], 1.0)

                    if(self._apply_ham_as_tensor):
                        self._qc.apply_tensor_spat_012bdy(
                                self._zero_body_energy, 
                                self._mo_oeis, 
                                self._mo_teis, 
                                self._mo_teis_einsum, 
                                self._norb)
                    else:
                        self._qc.apply_sqop(self._sq_ham)

                    H_rho_psi = self._qc.get_state_deep()

                    H[i][j] = rho_psi.vector_dot(H_rho_psi)
                    H[j][i] = H[i][j].conj()

            return S, H

        else:
            rho_psi = []
            H_rho_psi = []

            for i in range(self._N_cis):
                self._qc.hartree_fock()

                if(i>0):
                    
                    ex = self._pool_obj.terms()[i][1].terms()[1]
                    sqop_mu = qf.SQOperator()
                    sqop_mu.add_term(ex[0], ex[1], ex[2])

                else:
                    sqop_mu = self._pool_obj.terms()[i][1]

                self._qc.apply_sqop(sqop_mu)

                # This is relevant for larger systems!
                IJ_lst = self._qc.get_nonzero_idxs()
                self._qc.set_element(IJ_lst[0], 1.0)

                rho_psi.append(self._qc.get_state_deep())

                if(self._apply_ham_as_tensor):
                    self._qc.apply_tensor_spat_012bdy(
                            self._zero_body_energy, 
                            self._mo_oeis, 
                            self._mo_teis, 
                            self._mo_teis_einsum, 
                            self._norb)
                else:
                    self._qc.apply_sqop(self._sq_ham)

                H_rho_psi.append(self._qc.get_state_deep())

                # populate lower triangle of S and copy conjugate to upper triangle
                for j in range(i+1):
                    H[i][j] = rho_psi[i].vector_dot(H_rho_psi[j])
                    H[j][i] = H[i][j].conj()

        # print(f"\n\n==> H matrix <==\n\n")
        # matprint(H)

        return S, H
    
    def run_cis(self):
        # Build S and H matrices
        self._S, self._Hbar = self.build_qk_mats()
        self._Scond = np.linalg.cond(self._S)
        self._eigenvalues, self._eigenvectors = np.linalg.eigh(self._Hbar)
        
        # print(f'\n\n ==> Evecs <== \n\n')
        # matprint(self._eigenvectors)

        print(f'\n       ==> Lowset 20 {type(self).__name__} eigenvalues <==')
        print('----------------------------------------')
        for i, val in enumerate(self._eigenvalues):
            if(i<20):
                print('  root  {}  {:.8f}    {:.8f}j'.format(i, np.real(val), np.imag(val)))

        # Set ground state energy.
        self._Egs = np.real(self._eigenvalues[0])

        # Set target state energy.
        self._Ets = np.real(self._eigenvalues[self._target_root])

        # Build and set a Unitary that builds the CIS state from HF.
        self.set_cis_unitary_parameters()

        # Set a tensor from a fci computer as the cis state for target_root.
        self.set_cis_state_tensor()

        # Print summary banner (should done for all algorithms).
        self.print_summary_banner()

        # verify that required attributes were defined
        # (should be called for all algorithms!)
        self.verify_run()

    def set_cis_state_tensor(self):
        self._qc.hartree_fock()

        self._qc.apply_two_determinant_rotations(
            self._IJ_sources,
            self._IJ_targets,
            self._theta_cis,
            False,
        )
        
        self._Ccis = self._qc.get_state_deep()

    def get_cis_state_tensor(self):
        return copy.deepcopy(self._Cis)
    
    def get_cis_unitary_parameters(self):
        return copy.deepcopy(self._IJ_sources), copy.deepcopy(self._IJ_targets), copy.deepcopy(self._theta_cis)

    def set_cis_unitary_parameters(self):

        ft = qforte.FCIComputer(
                    self._nel, 
                    self._2_spin, 
                    self._norb)
                
        opai_ij_cij_inds = []
        self._IJ_targets = []
        self._IJ_sources = []

        for ai in range(self._N_cis):
            
            Cia = self._eigenvectors[ai, self._target_root]
            op_ai = self._pool_obj.terms()[ai][1]

            if(np.abs(Cia) > 1.0e-10):
                # NOTE(Nick): A silly way to do this but it works..
                # Really we need to use FCIGraph to tell us what the IJ
                # indicies are for a given excitation op
                ft.hartree_fock()
                ft.apply_sqop(op_ai)
                IJ_lst = ft.get_nonzero_idxs()
                opai_ij_cij_inds.append((op_ai, IJ_lst[0], Cia))

                self._IJ_targets.append(IJ_lst[0])
                self._IJ_sources.append([0, 0])

        self._theta_cis = []
        Co = 1.0
        
        for i, val in enumerate(opai_ij_cij_inds):
            Cai_cis = np.real(val[2])

            # In this version Cia is = 0, (starting from HF) 
            Cai = 0.0

            # Treat the final edge case, can be numerically 
            # unstable otherwise...
            if(i == len(opai_ij_cij_inds)-1):
                if(Cai_cis*Co > 0.0):
                    theta = np.pi/2.0
                else:
                    theta = 3.0*np.pi/2.0
            else:
                theta = np.real(np.arcsin( Cai_cis / (Co) ))

            self._theta_cis.append(theta)

            Co_prime  = np.cos(theta)*Co - np.sin(theta)*Cai
            # Cai_prime = np.sin(theta)*Co + np.cos(theta)*Cai

            Co = Co_prime


        ft4 = qforte.FCIComputer(
                    self._nel, 
                    self._2_spin, 
                    self._norb)
        

        ft4.hartree_fock()


        ft4.apply_two_determinant_rotations(
            self._IJ_sources,
            self._IJ_targets,
            self._theta_cis,
            False,
        )

        Ecis4 = ft4.get_exp_val(self._sq_ham)
        print(f"\n\nEcis_root from U|HF> state: {Ecis4}\n\n")
