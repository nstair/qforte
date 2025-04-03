"""
QITE classes
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
from qforte.helper.df_ham_helper import *
import copy
import numpy as np
from scipy.linalg import lstsq

from qforte.maths.eigsolve import canonical_geig_solve

### Throughout this file, we'll refer to DOI 10.1038/s41567-019-0704-4 as Motta.

class QITE(Algorithm):
    """This class implements the quantum imaginary time evolution (QITE)
    algorithm in a fashion amenable to non k-local hamiltonains, which is
    the focus of the origional algorithm (see DOI 10.1038/s41567-019-0704-4).

    In QITE one attepmts to approximate the action of the imaginary time evolution
    operator on a state :math:`| \Phi \\rangle` with a parameterized unitary
    operation:

    .. math::
        c(\\Delta \\beta)^{-1/2} e^{-\\Delta \\beta \hat{H}} | \Phi \\rangle \\approx e^{-i \\Delta \\beta \hat{A}(\\vec{\\theta})} | \Phi \\rangle,

    where :math:`\\Delta \\beta` is a small time step and
    :math:`c(\\Delta \\beta)^{-1/2}` is a normalization coefficient approximated
    by :math:`1-2\\Delta \\beta \\langle \Phi | \hat{H} | \Phi \\rangle`.

    The parameterized anti-hermetian operator :math:`\hat{A}(\\vec{\\theta})`
    is given by the linear combination of :math:`N_\mu` operators

    .. math::
        \hat{A}(\\vec{\\theta}) = \sum_\mu^{N_\mu} \\theta_\mu \hat{P}_\mu,

    where :math:`\hat{P}_\mu` is a product of Pauli operators. In practice the
    operators that enter in to the sum are a subset of an operator pool specified
    by the user.

    To determine the parameters :math:`\\theta_\mu` one seeks to satisfy the
    condition:

    .. math::
        c(\\beta)^{-1/2} \\langle \Phi |  \sum_{\mu} \\theta_\mu \hat{P}_\mu^\dagger \hat{H} | \Phi \\rangle
        \\approx -i  \\langle \Phi | \sum_{\mu} \\theta_\mu \\theta_\\nu \hat{P}_\mu^\dagger  \hat{P}_\\nu | \Phi \\rangle

    which corresponding to solving the linear systems

    .. math::
        \mathbf{S} \\vec{\\theta} = \\vec{b}

    where the elements

    .. math::
        S_{\mu \\nu} = \\langle \Phi | \hat{P}_\mu^\dagger \hat{P}_\\nu | \Phi \\rangle,

    .. math::
        b_\mu = \\frac{-i}{\sqrt{c(\Delta \\beta)}} \\langle \Phi | \hat{P}_\mu^\dagger \hat{H} | \Phi \\rangle

    can be measured on a quantum device.

    Note that the QITE procedure is iterative and is repated for a specified
    number of time steps to reach a target total evolution time.

    Attributes
    ----------

    _b_thresh : float
        The minimum threshold absolute vale for the elements of :math:`b_\mu` to be included
        in the solving of the linear system. Operators :math:`\hat{P}_\mu`
        corresponding to elements of :math:`|b_\mu|` < _b_thresh will not enter
        into the operator :math:`\hat{A}`.

    _x_thresh : float
        Operators :math:`\hat{P}_\mu` corresponding to elements of :math:`|\\theta_\mu|`
        < _b_thresh will not enter into the operator :math:`\hat{A}`.

    _beta : float
        The target total evolution time.

    _db : float
        The imaginary time step to use.

    _do_lanczos : bool
        Whether or not to additionaly compute the QLanczos QSD matrices and
        solve the corresponding generailzed eigenvalue problem.

    _Ekb : list of float
        The list of after each additional time step.

    _expansion_type: {'complete_qubit', 'cqoy', 'SD', 'GSD', 'SDT', SDTQ', 'SDTQP', 'SDTQPH'}
        The family of operators that each evolution operator :math:`\hat{A}` will be built of.

    _lanczos_gap : int
        The number of time steps between generation of Lanczos basis vectors.

    _nbeta: int
        How many QITE steps should be taken? (not directly specified by user).

    _NI : int
        The number of operators in _sig.

    _sig : QubitOpPool
        The basis of operators allowed in a unitary evolution step.

    _sparseSb : bool
        Use sparse tensors to solve the linear system?

    _Uqite : Circuit
        The circuit that prepares the QITE state at the current iteration.


    """
    def run(self,
            beta=1.0,
            db=0.2,
            dt=0.01,
            use_diis=False,
            max_diis_size=False,
            use_exact_evolution=False,
            expansion_type='SD',
            evolve_dfham=False,
            random_state=False,
            sparseSb=True,
            low_memorySb=False,
            second_order=False,
            selected_pool=False,
            t_thresh=1.0e-6,
            cumulative_t=False,
            b_thresh=1.0e-6,
            x_thresh=1.0e-10,
            physical_r = False,
            folded_spectrum = False,
            BeH2_guess = False,
            e_shift = None,
            update_e_shift = True,
            do_lanczos=False,
            lanczos_gap=2,
            realistic_lanczos=True,
            fname=None,
            output_path=None,
            print_pool=False,
            use_cis_reference=False,
            target_root=0,
            cis_target_root=0,
            ):
        
        # TODO(Nick): Remove BeH2 specific stuff..

        self._beta = beta
        self._db = db
        self._dt = dt
        self._use_exact_evolution = use_exact_evolution

        if(folded_spectrum):
            beta_sq = beta*beta
            db_sq = db*db
            self._nbeta = int(beta_sq/db_sq)+1
        else:
            self._nbeta = int(beta/db)+1

        self._expansion_type = expansion_type
        self._evolve_dfham = evolve_dfham
        self._random_state = random_state
        self._sparseSb = sparseSb
        self._low_memorySb = low_memorySb
        self._second_order = second_order
        self._selected_pool = selected_pool
        self._t_thresh = t_thresh
        self._cumulative_t = cumulative_t
        self._total_phase = 1.0 + 0.0j
        self._Uqite = qf.Circuit()
        self._b_thresh = b_thresh
        self._x_thresh = x_thresh
        self._physical_r = physical_r

        # DIIS options
        self._use_diis = use_diis
        self._qite_diis_max = max_diis_size

        # CIS options
        self._use_cis_reference = use_cis_reference
        self._target_root = target_root
        self._cis_target_root = cis_target_root

        # FS options
        self._folded_spectrum = folded_spectrum
        self._BeH2_guess = BeH2_guess 
        self._e_shift = e_shift
        self._update_e_shift = update_e_shift

        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_pauli_trm_measures = 0

        self._do_lanczos = do_lanczos
        self._lanczos_gap = lanczos_gap
        self._realistic_lanczos = realistic_lanczos
        self._fname = fname
        self._output_path = output_path
        self._print_pool = print_pool

        if(self._fname is None):
            if(self._use_exact_evolution):
                self._fname = f'beta_{self._beta}_db_{self._db}_EXACT_EVOLUTION'
            else:
                self._fname = f'beta_{self._beta}_db_{self._db}_{self._computer_type}_{self._expansion_type}_second_order_{self._second_order}_folded_spectrum_{self._folded_spectrum}_e_shift_{self._e_shift}_selected_pool_{self._selected_pool}_t_{self._t_thresh}_physical_r_{self._physical_r}_dfham_{self._evolve_dfham}'

        if(self._output_path is None):
            self._output_path = ''

        self._sz = 0

        if(self._computer_type=='fci'):
            qc_ref = qf.FCIComputer(self._nel, self._sz, self._norb)

            if(self._random_state):
                comp_shape = qc_ref.get_state_deep().shape()
                rand_arr = np.random.rand(*comp_shape)
                norm = np.linalg.norm(rand_arr)
                normalized_coeffs = rand_arr / norm

                # nc = normalized_coeffs.tolist()
                self._rand_tensor = qf.Tensor(shape=comp_shape, name='random')
                for i in range(comp_shape[0]):
                    for j in range(comp_shape[1]):
                        self._rand_tensor.set([i,j], normalized_coeffs[i,j])

                qc_ref.set_state(self._rand_tensor)
            elif(self._use_cis_reference):

                alg_cis = qf.CIS(
                    self._sys,
                    computer_type = self._computer_type,
                    apply_ham_as_tensor=self._apply_ham_as_tensor,
                )

                alg_cis.run(
                    target_root=self._target_root,
                    diagonalize_each_step=False,
                    low_memory=False
                )
                
                if(self._e_shift == None):
                    print(f'\n** Setting FS-QITE Eshft={alg_cis._Ets:+8.8f} from cis root: {self._cis_target_root} **\n')
                    self._e_shift = alg_cis._Ets

                self._cis_IJ_sources, self._cis_IJ_targets, self._cis_angles = alg_cis.get_cis_unitary_parameters()

                qc_ref.hartree_fock()

                qc_ref.apply_two_determinant_rotations(
                    self._cis_IJ_sources,
                    self._cis_IJ_targets,
                    self._cis_angles,
                    False
                )

            else:
                qc_ref.hartree_fock()

            if(self._folded_spectrum): # only implementing it for sq ham to start
                if(self._apply_ham_as_tensor):
                    self._shifted_0_body = self._nuclear_repulsion_energy - self._e_shift
                else:
                    self._Ofs = qf.SQOperator()
                    self._Ofs.add_op(self._sq_ham)
                    self._Ofs.add_term(-self._e_shift, [], [])

                #FOR BeH2 BENCHMARK ONLY
                if(self._BeH2_guess):
                    val = 1.0 / np.sqrt(2.0)
    
                    fci_temp = qf.FCIComputer(self._nel, self._sz, self._norb)

                    alpha_ex = qf.SQOperator()
                    alpha_ex.add_term(1.0, [6], [4])

                    beta_ex = qf.SQOperator()
                    beta_ex.add_term(1.0, [7], [5])

                    fci_temp.hartree_fock()
                    fci_temp.apply_sqop(alpha_ex)
                    a_ind = fci_temp.get_nonzero_idxs()[0]

                    fci_temp.hartree_fock()
                    fci_temp.apply_sqop(beta_ex)
                    b_ind = fci_temp.get_nonzero_idxs()[0]

                    fci_temp.zero_state()
                    fci_temp.set_element(a_ind, val)
                    fci_temp.set_element(b_ind, val)

                    self._excited_guess = fci_temp.get_state_deep()

            if(self._evolve_dfham):
                dfh = self._sys.df_ham
                time_scale_first_leaf(dfh, self._db)
                v_lst = dfh.get_scaled_density_density_matrices()
                g_lst = dfh.get_trotter_basis_change_matrices()

                self._ga0 = qf.SQOpPool()
                self._ga0.append_givens_ops_sector(
                    g_lst[0], 
                    1.0/self._db,
                    True)

                self._gb0 = qf.SQOpPool()
                self._gb0.append_givens_ops_sector(
                    g_lst[0], 
                    1.0/self._db,
                    False)

                self._d0 = qf.SQOpPool()
                self._d0.append_diagonal_ops_all(
                    v_lst[0], 
                    1.0)

                exp1 = qc_ref.get_state_deep()
                qc_ref.apply_sqop_pool(self._d0)

                self._Ekb = [np.real(exp1.vector_dot(qc_ref.get_state_deep()))]
            
            else:
                if(self._apply_ham_as_tensor):

                    self._Ekb = [np.real(qc_ref.get_exp_val_tensor(
                            self._zero_body_energy, 
                            self._mo_oeis, 
                            self._mo_teis, 
                            self._mo_teis_einsum, 
                            self._norb))]

                else:
                    self._Ekb = [np.real(qc_ref.get_exp_val(self._sq_ham))]
            
        if(self._computer_type=='fock'):
            qc_ref = qf.Computer(self._nqb)
            qc_ref.apply_circuit(self._Uprep)
            self._Ekb = [np.real(qc_ref.direct_op_exp_val(self._qb_ham))]

        # Print options banner (should done for all algorithms).
        self.print_options_banner()

        # NOTE(Nick): temporary, just want diis to work as expected.
        self._tamps = []

        # Build expansion pool.
        if(not self._use_exact_evolution):
            self.build_expansion_pool()

        self._t_diis = [copy.deepcopy(self._tamps)]
        self._e_diis = []
        

        # Do the imaginary time evolution.
        timer = qf.local_timer()
        timer.reset()

        self.evolve()

        timer.record('Total evolution time')
        print(f"\n\n{timer}\n\n")

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
        print('     Quantum Imaginary Time Evolution Algorithm   ')
        print('-----------------------------------------------------')

        print('\n\n                 ==> QITE options <==')
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
        print('\n')
        print('Total imaginary evolution time (beta):   ',  self._beta)
        print('Imaginary time step (db):                ',  self._db)

        # Options for Folded Spectrum
        print('\n')
        print('Use Folded Spectrum:                     ',  self._folded_spectrum)
        if(self._folded_spectrum):
            print('FS-QITE target root:                     ',  self._target_root)

        # Options for CIS Reference
        print('\n')
        print('Use CIS Reference:                       ',  self._use_cis_reference)
        if(self._use_cis_reference):
            print('Using CIS Reference root:                ',  self._cis_target_root)
            print('Update Eshift during QITE:               ',  self._update_e_shift)

        # Last QITE option, dictates whether more is printed
        print('\n')
        print('Use exact evolutoin:                     ', self._use_exact_evolution)

        if not self._use_exact_evolution:
            print('\n')
            print('Expansion type:                          ',  self._expansion_type)
            print('Use DIIS:                                ',  self._use_diis)
            print('Max DIIS size:                           ',  self._qite_diis_max)
        
            print('Use selected pool:                       ',  self._selected_pool)
            if self._selected_pool:
                print('Use cumulative selection:                ',  self._cumulative_t)
                print('Use physical selection:                  ',  self._physical_r)
                print('Selection time step (dt):                ',  self._dt)

        
            print('x value threshold:                       ',  self._x_thresh)
            print('Use sparse tensors to solve Sx = b:      ',  str(self._sparseSb))
            if(self._sparseSb):
                print('b value threshold:                       ',  str(self._b_thresh))
            print('\n')
            print('Use low memory mode:                     ',  self._low_memorySb)
            print('Use 2nd order derivation of QITE:        ',  self._second_order)

        print('Do Quantum Lanczos                       ',  str(self._do_lanczos))
        if(self._do_lanczos):
            print('Lanczos gap size                         ',  self._lanczos_gap)

        print('\n\n')

    def print_summary_banner(self):
        print('\n\n                        ==> QITE summary <==')
        print('-----------------------------------------------------------')
        print('Final QITE Energy:                        ', round(self._Ets, 10))
        if(not self._use_exact_evolution and self._folded_spectrum):
            print('Final Energy Shift:                       ', round(self._e_shift, 10))

        if(not self._use_exact_evolution):
            print('Number of operators in pool:              ', self._NI)
            print('Number of classical parameters used:      ', self._n_classical_params)
            print('Estimated classical memory usage (GB):    ', f'{self._total_memory * 10**-9:e}')
            print('Number of CNOT gates in deepest circuit:  ', self._n_cnot)
            print('Number of Pauli term measurements:        ', self._n_pauli_trm_measures)

    def build_expansion_pool(self):
        print('\n==> Building expansion pool <==')

        if(self._computer_type=='fci'):
            if(self._selected_pool):
                self._full_pool = qf.SQOpPool()
                self._full_pool.set_orb_spaces(self._ref)

                self._total_pool = qf.SQOpPool()

                if(self._expansion_type in {'All'}):
                    self._full_pool.fill_pool(self._expansion_type) # This automatically filters non-particle conserving terms

                else:
                    raise ValueError('Selected QITE only implemented for full expansion pool.')

                self._pool_idx_to_state_idx = {}
                self._state_idx_to_pool_idx = {}

                my_fci_comp = qforte.FCIComputer(
                                self._nel, 
                                self._sz, 
                                self._norb)

                for mu, term in enumerate(self._full_pool.terms()):
                    my_fci_comp.hartree_fock()
                    my_fci_comp.apply_sqop(term[1])
                    ij = my_fci_comp.get_nonzero_idxs()
                    # print(tuple(ij[0]))

                    self._pool_idx_to_state_idx[mu] = tuple(ij[0])
                    self._state_idx_to_pool_idx[tuple(ij[0])] = mu

                self._NI = len(self._full_pool.terms())
                # print(f'# of pool terms: {self._NI}')
                self._idx_lst = np.zeros(self._NI)
                self._R_sq_lst = np.zeros(self._NI)

                self._R = [None] * self._NI
                # self._idx = np.zeros(self._NI)

            else:
                self._sig = qf.SQOpPool() # changed this from QubitOpPool
                self._sig.set_orb_spaces(self._ref) # is this ok for starting from a random state?

                if(self._expansion_type in {'SD', 'GSD', 'SDT', 'SDTQ', 'SDTQP', 'SDTQPH', 'All'}):
                    self._sig.fill_pool(self._expansion_type) # This automatically filters non-particle conserving terms

                elif(self._expansion_type[0].isdigit() and self._expansion_type[1:] == '-UpCCGSD'):
                    self._sig.fill_pool_kUpCCGSD(int(self._expansion_type[0]))

                else:
                    raise ValueError('Invalid expansion type specified.')

                self._NI = len(self._sig.terms())
                self._O_sq_lst = np.zeros(self._NI)

        if(self._computer_type=='fock'):
            self._sig = qf.QubitOpPool()

            if(self._expansion_type == 'complete_qubit'):
                if (self._nqb > 6):
                    raise ValueError('Using complete qubits expansion will result in a very large number of terms!')
                self._sig.fill_pool("complete_qubit", self._ref)

            elif(self._expansion_type == 'cqoy'):
                self._sig.fill_pool("cqoy", self._ref)

            elif(self._expansion_type in {'SD', 'GSD', 'SDT', 'SDTQ', 'SDTQP', 'SDTQPH'}):
                P = qf.SQOpPool()
                P.set_orb_spaces(self._ref) 
                P.fill_pool(self._expansion_type)
                sig_temp = P.get_qubit_operator("commuting_grp_lex", False)

                # Filter the generated operators, so that only those with an odd number of Y gates are allowed.
                # See section "Real Hamiltonians and states" in the SI of Motta for theoretical justification.
                # Briefly, this method solves Ax=b, but all b elements with an odd number of Y gates are imaginary and
                # thus vanish. This method will not be correct for non-real Hamiltonians or states.
                for _, rho in sig_temp.terms():
                    nygates = 0
                    temp_rho = qf.Circuit()
                    for gate in rho.gates():
                        temp_rho.add(qf.gate(gate.gate_id(), gate.target(), gate.control()))
                        if (gate.gate_id() == "Y"):
                            nygates += 1

                    if (nygates % 2 == 1):
                        rho_op = qf.QubitOperator()
                        rho_op.add(1.0, temp_rho)
                        self._sig.add(1.0, rho_op)

            else:
                raise ValueError('Invalid expansion type specified.')

            self._NI = len(self._sig.terms())

        self._tamps = list(np.zeros(self._NI))


    def build_S_b_FCI(self):
        """Construct the matrix S (eq. 5a) and vector b (eq. 5b) of Motta, with h[m] the full Hamiltonian, utilizing FCIComputer class.
        """
        Idim = self._NI

        self._n_pauli_trm_measures += int(self._NI*(self._NI+1)*0.5)
        self._n_pauli_trm_measures += self._Nl * self._NI

        # Initialize linear system
        S = np.zeros((Idim, Idim), dtype=complex)
        b = np.zeros(Idim, dtype=complex)

        if(self._second_order):
            prefactor = -2.0
            S_factor = 2.0

        else:
            denom = np.sqrt(1.0 - 2.0*self._db*self._Ekb[-1])
            prefactor = -1.0 / denom
            S_factor = 1.0

        Ipsi_qc = qf.FCIComputer(self._nel, self._sz, self._norb)
        Hpsi_qc = qf.FCIComputer(self._nel, self._sz, self._norb)
        Hpsi_qc.set_state(self._qc.get_state_deep())

        if(self._evolve_dfham):
            Hpsi_qc.apply_sqop_pool(self._d0)

        else:
            if(self._folded_spectrum):
                if(self._apply_ham_as_tensor):
                    Hpsi_qc.apply_tensor_spat_012bdy(
                            self._shifted_0_body,
                            self._mo_oeis, 
                            self._mo_teis, 
                            self._mo_teis_einsum, 
                            self._norb)
                    Hpsi_qc.apply_tensor_spat_012bdy(
                            self._shifted_0_body,
                            self._mo_oeis, 
                            self._mo_teis, 
                            self._mo_teis_einsum, 
                            self._norb)
                else:
                    Hpsi_qc.apply_sqop(self._Ofs)
                    Hpsi_qc.apply_sqop(self._Ofs)
                
            else:
                if(self._apply_ham_as_tensor):
                    Hpsi_qc.apply_tensor_spat_012bdy(
                            self._zero_body_energy, 
                            self._mo_oeis, 
                            self._mo_teis, 
                            self._mo_teis_einsum, 
                            self._norb)
                else:
                    Hpsi_qc.apply_sqop(self._sq_ham)

        if(self._low_memorySb):
            for i in range(Idim):
                S[i][i] = 1.0 # With Pauli strings, this is always the inner product

                # initialize state and apply pool term
                Ipsi_qc.set_state(self._qc.get_state_deep())
                Ipsi_qc.apply_sqop(self._sig.terms()[i][1])
                Ipsi_mu = Ipsi_qc.get_state_deep()

                # build b (second order variation)
                if(self._second_order):
                    exp_val = Hpsi_qc.get_state_deep().vector_dot(Ipsi_mu)
                    b[i] = prefactor * exp_val

                # build b (original)
                else:
                    exp_val = Ipsi_mu.vector_dot(Hpsi_qc.get_state_deep())
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
                if(self._second_order):
                    exp_val = Hpsi_qc.get_state_deep().vector_dot(rho_psi[i])
                    b[i] = prefactor * exp_val

                # build b (original)
                else:
                    exp_val = rho_psi[i].vector_dot(Hpsi_qc.get_state_deep())
                    b[i] = prefactor * exp_val

                # populate lower triangle of S and copy conjugate to upper triangle
                for j in range(i):
                    S[i][j] = rho_psi[i].vector_dot(rho_psi[j])
                    S[j][i] = S[i][j].conj()

            return S_factor * np.real(S), np.real(b)


    def build_S(self):
        """Construct the matrix S (eq. 5a) of Motta.
        """
        Idim = self._NI

        S = np.zeros((Idim, Idim), dtype=complex)

        Ipsi_qc = qf.Computer(self._nqb)
        Ipsi_qc.set_coeff_vec(copy.deepcopy(self._qc.get_coeff_vec()))
        # CI[I][J] = (σ_I Ψ)_J
        self._n_pauli_trm_measures += int(self._NI*(self._NI+1)*0.5)
        CI = np.zeros(shape=(Idim, int(2**self._nqb)), dtype=complex)

        for i in range(Idim):
            S[i][i] = 1.0 # With Pauli strings, this is always the inner product
            Ipsi_qc.apply_operator(self._sig.terms()[i][1])
            CI[i,:] = copy.deepcopy(Ipsi_qc.get_coeff_vec())
            for j in range(i):
                S[i][j] = S[j][i] = np.vdot(CI[i,:], CI[j,:])
            Ipsi_qc.set_coeff_vec(copy.deepcopy(self._qc.get_coeff_vec()))

        return np.real(S)


    def build_sparse_S_b(self, b):
        b_sparse = []
        idx_sparse = []
        for I, bI in enumerate(b):
            if(np.abs(bI) > self._b_thresh):
                idx_sparse.append(I)
                b_sparse.append(bI)
        Idim = len(idx_sparse)
        self._n_pauli_trm_measures += int(Idim*(Idim+1)*0.5)

        S = np.zeros((len(b_sparse),len(b_sparse)), dtype=complex)

        Ipsi_qc = qf.Computer(self._nqb)
        Ipsi_qc.set_coeff_vec(copy.deepcopy(self._qc.get_coeff_vec()))
        CI = np.zeros(shape=(Idim, int(2**self._nqb)), dtype=complex)

        for i in range(Idim):
            S[i][i] = 1.0 # With Pauli strings, this is always the inner product
            Ii = idx_sparse[i]
            Ipsi_qc.apply_operator(self._sig.terms()[Ii][1])
            CI[i,:] = copy.deepcopy(Ipsi_qc.get_coeff_vec())
            for j in range(i):
                S[i][j] = S[j][i] = np.vdot(CI[i,:], CI[j,:])
            Ipsi_qc.set_coeff_vec(copy.deepcopy(self._qc.get_coeff_vec()))

        return idx_sparse, np.real(S), np.real(b_sparse)

    def build_b(self):
        """Construct the vector b (eq. 5b) of Motta, with h[m] the full Hamiltonian.
        """

        b  = np.zeros(self._NI, dtype=complex)

        denom = np.sqrt(1 - 2*self._db*self._Ekb[-1])
        prefactor = -1.0j / denom

        self._n_pauli_trm_measures += self._Nl * self._NI

        Hpsi_qc = qf.Computer(self._nqb)
        Hpsi_qc.set_coeff_vec(copy.deepcopy(self._qc.get_coeff_vec()))
        Hpsi_qc.apply_operator(self._qb_ham)
        C_Hpsi_qc = copy.deepcopy(Hpsi_qc.get_coeff_vec())

        for I, (op_coefficient, operator) in enumerate(self._sig.terms()):
            Hpsi_qc.apply_operator(operator)
            exp_val = np.vdot(self._qc.get_coeff_vec(), Hpsi_qc.get_coeff_vec())
            b[I] = prefactor * op_coefficient * exp_val
            Hpsi_qc.set_coeff_vec(copy.deepcopy(C_Hpsi_qc))

        return np.real(b)


    def do_qite_step(self):

        if(self._folded_spectrum and self._update_e_shift):
            self.update_e_shift()

        if(self._computer_type=='fci'):
            if(self._sparseSb):
                print(f"Warning, build sparseSb method isn't supported for FCI computer, setting option to false")
                self._sparseSb = False

            S, btot = self.build_S_b_FCI()

        if(self._computer_type=='fock'):
            if(self._low_memorySb):
                print(f"Warning, build low memory Sb method isn't supported for Fock computer, setting option to false")
                self._low_memorySb = False

            btot = self.build_b()
            A = qf.QubitOperator()

            if(self._sparseSb):
                sp_idxs, S, btot = self.build_sparse_S_b(btot)
            else:
                S = self.build_S()

        x = lstsq(S, btot)[0]
        x = np.real(x)
        x_list = x.tolist()

        self._n_classical_params += len(x_list)

        if(self._folded_spectrum):
            x_list_fci = [x*self._db*self._db for x in x_list]
        else:
            x_list_fci = [x*self._db for x in x_list]

        # Also used only for DIIS
        if(self._use_diis):
            told = copy.deepcopy(self._tamps)
            self._tamps = self._tamps = list(np.add(self._tamps, x_list_fci))
            evec = list(np.subtract(self._tamps, told))

        if(self._computer_type=='fock'):
            if(self._sparseSb):
                for I, spI in enumerate(sp_idxs):
                    if np.abs(x[I]) > self._x_thresh:
                        A.add(-1.0j * self._db * x[I], self._sig.terms()[spI][1].terms()[0][1])
                        self._n_classical_params += 1

            else:
                for I, SigI in enumerate(self._sig.terms()):
                    if np.abs(x[I]) > self._x_thresh:
                        A.add(-1.0j * self._db * x[I], SigI[1].terms()[0][1])
                        self._n_classical_params += 1

        if(self._verbose):
            print('\nbtot:\n ', btot)
            print('\n S:  \n')
            matprint(S)
            print('\n x:  \n')
            print(x)

        # added fock computer conditional
        if(self._computer_type=='fock'):
            eiA_kb, phase1 = trotterize(A, trotter_number=self._trotter_number)
            self._total_phase *= phase1
            self._Uqite.add(eiA_kb)
            self._qc.apply_circuit(eiA_kb)
            self._Ekb.append(np.real(self._qc.direct_op_exp_val(self._qb_ham)))

            self._n_cnot += eiA_kb.get_num_cnots()

        if(self._computer_type=='fci'):
            if(self._evolve_dfham):
                self._qc.evolve_pool_trotter(
                    self._ga0,
                    self._db,
                    1,
                    1)

                self._sig.set_coeffs(x_list_fci)
                self._qc.evolve_pool_trotter_basic(
                    self._sig, 
                    1, 
                    0)

                self._qc.evolve_pool_trotter(
                    self._gb0,
                    self._db,
                    1,
                    1)
                
                exp1 = self._qc.get_state_deep()
                self._qc.apply_sqop_pool(self._d0)
                self._Ekb.append(np.real(exp1.vector_dot(self._qc.get_state_deep())))
                self._qc.set_state(exp1)

            else:
                if(self._use_diis):

                    self._t_diis.append(copy.deepcopy(self._tamps))
                    self._e_diis.append(copy.deepcopy(evec))

                    self._tamps = self.qite_diis(
                        self._qite_diis_max,
                        self._t_diis,
                        self._e_diis)
                    
                    x_list_fci_diis = list(np.subtract(self._tamps, told))
                    self._sig.set_coeffs(x_list_fci_diis)
                
                else:
                    self._sig.set_coeffs(x_list_fci)

                self._qc.evolve_pool_trotter_basic(
                    self._sig, 
                    1, 
                    0)

                if(self._apply_ham_as_tensor):
                    self._Ekb.append(np.real(self._qc.get_exp_val_tensor(
                            self._zero_body_energy, 
                            self._mo_oeis, 
                            self._mo_teis, 
                            self._mo_teis_einsum, 
                            self._norb)))
                else:
                    self._Ekb.append(np.real(self._qc.get_exp_val(self._sq_ham)))


        if(self._verbose):
            print('state after operator pool evolution')
            print(self._qc)
            print('\n')

    def evolve(self):
        """Perform QITE for a time step :math:`\\Delta \\beta`.
        """
    
        if(self._computer_type=='fock'):
            self._Uqite.add(self._Uprep)
            self._qc = qf.Computer(self._nqb)
            self._qc.apply_circuit(self._Uqite)

        if(self._computer_type=='fci'):
            self._qc = qf.FCIComputer(self._nel, self._sz, self._norb)
            
            if(self._random_state):
                self._qc.set_state(self._rand_tensor)

            elif(self._use_cis_reference):

                self._qc.hartree_fock()

                self._qc.apply_two_determinant_rotations(
                    self._cis_IJ_sources,
                    self._cis_IJ_targets,
                    self._cis_angles,
                    False
                )
            else:
                self._qc.hartree_fock()

            if(not self._use_exact_evolution):
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


        if(self._do_lanczos and not self._use_exact_evolution):
            #initialize constant list to build H and S matricies
            if(self._realistic_lanczos):
                self._c_list = []
                self._c_list.append(1.0) # will always be 1.0 for 0th iteration (E_l = E_0)

            else:
                self._lanczos_vecs = []
                self._Hlanczos_vecs = []
                self._lanczos_vecs.append(self._qc.get_state_deep())

                qcSig_temp = qf.FCIComputer(self._nel, self._sz, self._norb)
                qcSig_temp.set_state(self._qc.get_state_deep())

                if(self._apply_ham_as_tensor):
                    qcSig_temp.apply_tensor_spat_012bdy(
                            self._zero_body_energy, 
                            self._mo_oeis, 
                            self._mo_teis, 
                            self._mo_teis_einsum, 
                            self._norb)
                else:
                    qcSig_temp.apply_sqop(self._sq_ham)

                self._Hlanczos_vecs.append(qcSig_temp.get_state_deep())

        

        if(self._folded_spectrum):
            print(f"{'beta^2':>8}{'E(beta)':>18}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}")
            print('-------------------------------------------------------------------------------')
            print(f' {0.0:7.3f}    {self._Ekb[0]:+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}')
        else:
            print(f"{'beta':>7}{'E(beta)':>18}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}")
            print('-------------------------------------------------------------------------------')
            print(f' {0.0:7.3f}    {self._Ekb[0]:+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}')

        if (self._print_summary_file):
            f = open(f"{self._output_path}qite_{self._fname}_summary.dat", "w+", buffering=1)
            f.write(f"#{'beta':>7}{'E(beta)':>18}{'N(params)':>14}{'N(CNOT)':>18}{'N(measure)':>20}\n")
            f.write('#-------------------------------------------------------------------------------\n')
            f.write(f'  {0.0:7.3f}    {self._Ekb[0]:+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}\n')

            if(self._print_pool):
                f_pool = open(f"pool_qite_{self._fname}_summary.dat", "w+", buffering=1)
                if(not self._selected_pool):
                    f_pool.write(f'Initial SQOP Pool:\n{self._sig}\n')

        for kb in range(1, self._nbeta):
            if(self._use_exact_evolution):
                if(self._folded_spectrum):

                    if(self._update_e_shift):
                        self.update_e_shift()

                    if(self._apply_ham_as_tensor):
                            self._qc.evolve_tensor2_taylor(
                                    self._shifted_0_body,
                                    self._mo_oeis, 
                                    self._mo_teis, 
                                    self._mo_teis_einsum, 
                                    self._norb,
                                    self._db*self._db,
                                    1.0e-15,
                                    30,
                                    True)

                            # print(f'norm before scaling: {self._qc.get_state().norm()}')

                            norm = 1.0 / self._qc.get_state().norm()
                            self._qc.scale(norm)

                            # print(f'norm after scaling: {self._qc.get_state().norm()}')

                            self._Ekb.append(np.real(self._qc.get_exp_val_tensor(
                                    self._zero_body_energy, 
                                    self._mo_oeis, 
                                    self._mo_teis, 
                                    self._mo_teis_einsum, 
                                    self._norb)))
                    else:
                        self._qc.evolve_op2_taylor(
                                self._Ofs,
                                self._db*self._db,
                                1.0e-15,
                                30,
                                True)

                        # print(f'norm before scaling: {self._qc.get_state().norm()}')

                        norm = 1.0 / self._qc.get_state().norm()
                        self._qc.scale(norm)

                        # print(f'norm after scaling: {self._qc.get_state().norm()}')

                        self._Ekb.append(np.real(self._qc.get_exp_val(self._sq_ham)))

                else:
                    if(self._apply_ham_as_tensor):
                        self._qc.evolve_tensor_taylor(
                                self._zero_body_energy, 
                                self._mo_oeis, 
                                self._mo_teis, 
                                self._mo_teis_einsum, 
                                self._norb,
                                self._db,
                                1.0e-15,
                                30,
                                True)

                        # print(f'norm before scaling: {self._qc.get_state().norm()}')

                        norm = 1.0 / self._qc.get_state().norm()
                        self._qc.scale(norm)

                        # print(f'norm after scaling: {self._qc.get_state().norm()}')

                        self._Ekb.append(np.real(self._qc.get_exp_val_tensor(
                                self._zero_body_energy, 
                                self._mo_oeis, 
                                self._mo_teis, 
                                self._mo_teis_einsum, 
                                self._norb)))
                    else:
                        self._qc.evolve_op_taylor(
                                self._sq_ham,
                                self._db,
                                1.0e-15,
                                30,
                                True)

                        # print(f'norm before scaling: {self._qc.get_state().norm()}')

                        norm = 1.0 / self._qc.get_state().norm()
                        self._qc.scale(norm)

                        # print(f'norm after scaling: {self._qc.get_state().norm()}')

                        self._Ekb.append(np.real(self._qc.get_exp_val(self._sq_ham)))

            else:
                if(self._selected_pool):

                    if(kb>=2):
                        for term in self._sig.terms():
                            self._total_pool.add_term(term[0], term[1])


                    qc_res = qf.FCIComputer(self._nel, self._sz, self._norb)
                    qc_res.hartree_fock()

                    qc_res.evolve_pool_trotter_basic(
                        self._total_pool,
                        1,
                        0)

                    if(self._physical_r):
                        if(self._apply_ham_as_tensor):
                            qc_res.evolve_tensor_taylor(
                                self._zero_body_energy, 
                                self._mo_oeis, 
                                self._mo_teis, 
                                self._mo_teis_einsum, 
                                self._norb,
                                self._dt,
                                1.0e-15,
                                30,
                                False)
                        else:
                            qc_res.evolve_op_taylor(
                                self._sq_ham,
                                self._dt,
                                1.0e-15,
                                30,
                                False)

                    # unphysical for QC!
                    else:
                        if(self._apply_ham_as_tensor):
                            qc_res.apply_tensor_spat_012bdy(
                                self._zero_body_energy, 
                                self._mo_oeis, 
                                self._mo_teis, 
                                self._mo_teis_einsum, 
                                self._norb)
                        else:
                            qc_res.apply_sqop(self._sq_ham)

                    qc_res.evolve_pool_trotter_basic(
                        self._total_pool,
                        1,
                        1)

                    res_coeffs = qc_res.get_state_deep()
                    # print(res_coeffs)

                    self._sig = qf.SQOpPool()

                    if(self._cumulative_t):
                        for i in range(len(self._full_pool.terms())):
                            state_idx = self._pool_idx_to_state_idx[i]

                            if(self._physical_r):
                                self._R[i] = (np.real(res_coeffs.get([state_idx[0],state_idx[1]])*np.conj(res_coeffs.get([state_idx[0],state_idx[1]])))/self._dt**2, i)
                            else:
                                self._R[i] = (np.real(res_coeffs.get([state_idx[0],state_idx[1]])*np.conj(res_coeffs.get([state_idx[0],state_idx[1]]))), i)

                        R_sorted = sorted(self._R, key=lambda x: x[0])
                        R_magnitude = 0.0
                        self._sig_ind = []

                        for i in range(len(R_sorted)):
                            R_magnitude += R_sorted[i][0]
                            j = R_sorted[i][1]

                            if(R_magnitude>self._t_thresh):
                                self._sig_ind.append(j)
                                self._sig.add_term(1.0, self._full_pool.terms()[j][1])

                    else:
                        for i in range(len(self._full_pool.terms())):
                            state_idx = self._pool_idx_to_state_idx[i]
                            self._R_sq_lst[i] += np.real(res_coeffs.get([state_idx[0],state_idx[1]])**2)
                            self._idx_lst[i] = i

                            if(i>0):
                                if(np.real(res_coeffs.get([state_idx[0],state_idx[1]])**2) > self._t_thresh):
                                    self._sig.add_term(1.0, self._full_pool.terms()[i][1])

                    # for i in range(res_coeffs.shape()[0]):
                    #     for j in range(res_coeffs.shape()[1]):

                    #         if((i,j) == (0,0)):
                    #             continue

                    #         if(np.real(res_coeffs.get([i,j])**2) > self._t_thresh):
                    #             state_idx = (i, j)
                    #             mu = self._state_idx_to_pool_idx[state_idx]
                    #             self._sig.add_term(1.0, self._full_pool.terms()[mu][1])

                    self._NI = len(self._sig.terms())

                #do CNOT estimation here
                nqbit = self._norb * 2
                cnot_count = {}
                for term in self._sig.terms():
                    num_exc = len(term[1].terms()[1][1])
                    cnot_count[num_exc] = cnot_count.get(num_exc, 0) + 1

                # print(f'# excitacions per excitation order: {cnot_count}')

                temp_cnot = 0.0
                for exc in cnot_count.keys():
                    temp_cnot += (nqbit/3)*exc*cnot_count[exc]

                final_cnot = round(temp_cnot)
                self._n_cnot += final_cnot
                # print(f'total cnot gate estimate: {final_cnot}')

                self.do_qite_step()

                if(self._do_lanczos):
                    if(self._realistic_lanczos):
                        c_kb = np.exp(-2.0 * self._db * (self._Ekb[kb] - self._Ekb[0]))
                        self._c_list.append(c_kb)

                    else:
                        if(kb % self._lanczos_gap == 0):
                            self._lanczos_vecs.append(self._qc.get_state_deep())
                            qcSig_temp = qf.FCIComputer(self._nel, self._sz, self._norb)
                            qcSig_temp.set_state(self._qc.get_state_deep())

                            if(self._apply_ham_as_tensor):
                                qcSig_temp.apply_tensor_spat_012bdy(
                                    self._zero_body_energy, 
                                    self._mo_oeis, 
                                    self._mo_teis, 
                                    self._mo_teis_einsum, 
                                    self._norb)
                            else:
                                qcSig_temp.apply_sqop(self._sq_ham)

                            self._Hlanczos_vecs.append(qcSig_temp.get_state_deep())

            if(self._folded_spectrum):
                print(f' {kb*(self._db)**2:7.3f}    {self._Ekb[kb]:+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}')
            else:
                print(f' {kb*self._db:7.3f}    {self._Ekb[kb]:+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}')

            if (self._print_summary_file):
                f.write(f'  {kb*self._db:7.3f}    {self._Ekb[kb]:+15.9f}    {self._n_classical_params:8}        {self._n_cnot:10}        {self._n_pauli_trm_measures:12}\n')
                
                if(self._print_pool):
                    if(not self._selected_pool):
                        sorted_pool = self._sig.terms()
                        # sorted_pool = sorted(self._sig.terms(), key=lambda t: (len(t[1].terms()[0][2]), t[1].terms()[0][2]))
                        f_pool.write(f'iteration {kb} pool coeffs: {[term[0] for term in sorted_pool]}\n')

        self._Ets = self._Ekb[-1]

        if(self._target_root == 0):
            self._Egs = self._Ets
        else:
            self._Egs = self._sys.hf_energy

        if (self._print_summary_file):
            if(self._print_pool):
                if(self._selected_pool):
                    f_pool.write(f'\n{self._idx_lst}')
                    f_pool.write(f'\n{self._R_sq_lst}')

            f.close()
            if(self._print_pool):
                f_pool.close()

    def print_expansion_ops(self):
        print('\nQITE expansion operators:')
        print('-------------------------')
        print(self._sig.str())


    # NOTE(Nick): Presently this is identical to the pqe/vqe diis, likely can be optemized for qite,
    # We will also want a version that works with selection
    def qite_diis(self, diis_max_dim, t_diis, e_diis):
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


    def update_e_shift(self):
        if(self._apply_ham_as_tensor):
            self._shifted_0_body = self._nuclear_repulsion_energy - self._Ekb[-1]
            self._e_shift = self._Ekb[-1]

        else:
            self._Ofs.add_term(-self._Ekb[-1] + self._e_shift, [], [])
            self._e_shift = self._Ekb[-1]
        
