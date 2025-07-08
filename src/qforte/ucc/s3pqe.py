"""
s3pqe.py
====================================
Streaming‐Shot Selected Projective Quantum Eigensolver (S3PQE)
"""

import copy
import numpy as np
import qforte as qf

from qforte.abc.uccpqeabc import UCCPQE
# from qforte.utils.transforms import sq_op_find_symmetry
from qforte.utils.point_groups import sq_op_find_symmetry
from qforte.utils.trotterization import trotterize

class S3PQE(UCCPQE):
    """Streaming‐Shot Selected PQE (S3PQE).

    Every single shot selects one excitation label `mu` and its ancilla sign `a`,
    then updates theta[mu] -= eta * a immediately.
    """

    def run(self,
            max_shots: int = 100000,
            dt: float = 0.001,
            eta: float = 0.01,
            conv_tol: float = 1e-4,
            max_excit_rank = None):
        # reference check
        if self._state_prep_type != 'occupation_list':
            raise ValueError("S3PQE requires occupation_list reference.")
        

        # =====> FROM SPQE START <=====

        if(self._state_prep_type != 'occupation_list'):
            raise ValueError("SPQE implementation can only handle occupation_list Hartree-Fock reference.")

        # self._spqe_thresh = spqe_thresh
        # self._spqe_maxiter = spqe_maxiter
        self._dt = dt
        M_omega = max_shots

        if(M_omega != 'inf'):
            self._M_omega = int(M_omega)
        else:
            self._M_omega = M_omega

        # self._use_cumulative_thresh = use_cumulative_thresh
        # self._optimizer = optimizer
        # self._opt_thresh = opt_thresh
        # self._opt_maxiter = opt_maxiter

        # _nbody_counts: list that contains the numbers of singles, doubles, etc. incorporated in the final ansatz
        self._nbody_counts = []
        self._n_classical_params_lst = []

        self._results = []
        self._energies = []
        self._grad_norms = []
        self._tops = []
        self._tamps = []
        self._stop_macro = False
        self._converged = False
        self._res_vec_evals = 0
        self._res_m_evals = 0

        self._curr_energy = 0.0

        # Resource estimates.
        self._n_classical_params = 0
        self._n_cnot = 0
        self._n_cnot_lst = []
        self._n_pauli_trm_measures = 0
        self._n_pauli_trm_measures_lst = []
        self._Nm = []

        # self._eiH: QuantumCircuit
        #     Used to estimate the residuals outside the zero'd set when selecting new residuals to zero.
        self._eiH, _ = trotterize(self._qb_ham, factor= self._dt*(0.0 + 1.0j), trotter_number=self._trotter_number)

        if(self._computer_type=='fci'):
            # build hermitain pairs for time evolution (with fci computer)
            self._hermitian_pairs = qf.SQOpPool()

            # this is updated, evolution time is now just 1.0 here
            self._hermitian_pairs.add_hermitian_pairs(self._dt, self._sq_ham)        

        for occupation in self._ref:
            if occupation:
                self._nbody_counts.append(0)

        # create a pool of particle number, Sz, and spatial symmetry adapted second quantized operators
        # Encode the occupation list into a bitstring
        ref = sum([b << i for i, b in enumerate(self._ref)])
       # `& mask_alpha` gives the alpha component of a bitstring. `& mask_beta` does likewise.
        mask_alpha = 0x5555555555555555
        mask_beta = mask_alpha << 1
        nalpha = sum(self._ref[0::2])
        nbeta = sum(self._ref[1::2])

        if max_excit_rank is None:
            max_excit_rank = nalpha + nbeta
        elif not isinstance(max_excit_rank, int) or max_excit_rank <= 0:
            raise TypeError("The maximum excitation rank max_excit_rank must be a positive integer!")
        elif max_excit_rank > nalpha + nbeta:
            max_excit_rank = nalpha + nbeta
            print("\nWARNING: The entered maximum excitation rank exceeds the number of particles.\n"
                    "         Proceeding with max_excit_rank = {0}.\n".format(max_excit_rank))
        self._pool_type = max_excit_rank

        idx = 0
        # Given a coefficient index, what is the index of the "corresponding" pool element? Used to compute the operator to add to the ansatz in macroiterations.
        self._coeff_idx_to_pool_idx = {}
        self._coeff_idx_to_pool_idx_fci = {}
        self._indices_of_zeroable_residuals_for_pool = set()
        self._indices_of_zeroable_residuals_for_pool_fci = set()
        self._pool_obj = qf.SQOpPool()


        for I in range(1 << self._nqb):
            alphas = [int(j) for j in bin(I & mask_alpha)[2:]]
            betas = [int(j) for j in bin(I & mask_beta)[2:]]
            # Enforce particle number and Sz symmetry
            if sum(alphas) == nalpha and sum(betas) == nbeta:
                # Enforce point group symmetry
                if sq_op_find_symmetry(self._sys.orb_irreps_to_int,
                                       [len(alphas) - i - 1 for i, x in enumerate(alphas) if x],
                                       [len(betas) -i - 1 for i, x in enumerate(betas) if x]) == self._irrep:
                   # Create the bitstring of created/annihilated orbitals
                    excit = bin(ref ^ I).replace("0b", "")
                    # Confirm excitation number is non-zero
                    if excit != "0":
                        # Consider operators with rank <= max_excit_rank
                        if int(excit.count('1')/2) <= self._pool_type:
                            occ_idx = [int(i) for i,j in enumerate(reversed(excit)) if int(j) == 1 and self._ref[i] == 1]
                            unocc_idx = [int(i) for i,j in enumerate(reversed(excit)) if int(j) == 1 and self._ref[i] == 0]
                            sq_op = qf.SQOperator()
                            sq_op.add(+1.0, unocc_idx, occ_idx)
                            sq_op.add(-1.0, occ_idx[::-1], unocc_idx[::-1])
                            sq_op.simplify()
                            self._pool_obj.add_term(0.0, sq_op)
                            self._coeff_idx_to_pool_idx[I] = idx
                            
                            my_fci_comp = qf.FCIComputer(
                                self._nel, 
                                self._2_spin, 
                                self._norb) 
                            my_fci_comp.hartree_fock()
                            my_fci_comp.apply_sqop(sq_op)
                            
                            a = my_fci_comp.get_nonzero_idxs()
                            if len(a) != 1:
                                raise ValueError("Should only have one nonzero index")
                            
                            i = a[0][0]
                            j = a[0][1]
                            
                            self._coeff_idx_to_pool_idx_fci[(i,j)] = idx
                            self._indices_of_zeroable_residuals_for_pool.add(I)
                            self._indices_of_zeroable_residuals_for_pool_fci.add((i,j))   
                            idx += 1

        # Given a pool index, what is the coefficient of the "corresponding" coefficient vector element? Used to extract significant residuals in microiterations.
        # WARNING! To support repeated operators, either replace this variable or have repeated operators in the pool (which seems an awful hack).
        self._pool_idx_to_coeff_idx = {value: key for key, value in self._coeff_idx_to_pool_idx.items()}
        self._pool_idx_to_coeff_idx_fci = {value: key for key, value in self._coeff_idx_to_pool_idx_fci.items()}

        self.print_options_banner()

        self._timer = qf.local_timer()

        self._timer.reset()
        self.build_orb_energies()
        self._timer.record("build_orb_energies")

        if self._max_moment_rank:
            print('\nConstructing Moller-Plesset and Epstein-Nesbet denominators')
            self.construct_moment_space()

        self._spqe_iter = 1

        if(self._print_summary_file):
            f = open("summary.dat", "w+", buffering=1)
            f.write(f"#{'Iter(k)':>8}{'E(k)':>14}{'N(params)':>17}{'N(CNOT)':>18}{'N(measure)':>20}\n")
            f.write('#-------------------------------------------------------------------------------\n')

        # =====> FROM SPQE END <=====


        # build operator pool and mappings
        # self._build_pool()

        # initialize counters and parameters
        N_mu = np.zeros(len(self._pool_obj), dtype=int)
        theta = np.zeros(len(self._pool_obj), dtype=float)
        shots_seen = 1
        last_step = np.inf

        # precompute trotter circuit for e^{-i dt H}
        eiH_circ, _ = trotterize(self._qb_ham,
                                 factor=1.0j * dt,
                                 trotter_number=self._trotter_number)
        self._eiH = eiH_circ

        # streaming shots
        for shot in range(1, max_shots+1):
            # perform one shot update
            if self._computer_type == 'fock':
                raise NotImplementedError("Fock‐based S3PQE is not implemented yet.")
                last_step = self.update_ansatz_fock(N_mu, theta, eta, shots_seen)
            elif self._computer_type == 'fci':
                last_step = self.update_ansatz_fci(N_mu, theta, eta, shots_seen)
            else:
                raise ValueError(f"Unknown computer_type {self._computer_type}")

            shots_seen += 1

            # convergence check
            if abs(last_step) < conv_tol:
                print(f"S3PQE converged after {shot} shots.")
                break

            if shot % 10 == 0:
                print(f"Shot {shot:6d}: last step = {last_step:.6f}, "
                      f"shots seen = {shots_seen:6d}, "
                      f"theta[0] = {theta[0]:.6f}",
                      f"E = {self.energy_feval(self._tamps):.6f}",)

        # finalize energy
        self._tamps = theta.tolist()
        self._Egs = self.energy_feval(self._tamps)
        print(f"\n\nS3PQE ground‐state energy: {self._Egs:.12f}\n\n")
        return self._Egs

    def update_ansatz_fock(self, N_mu, theta, eta, shots_seen):
        """Single‐shot update using fock‐based Computer."""
        # 1) build current ansatz circuit
        U = self.ansatz_circuit(theta)

        # 2) simulate Uprep -> U -> e^{-i dt H} -> U†
        qc = qf.Computer(self._nqb)
        qc.apply_circuit(self._Uprep)
        qc.apply_circuit(U)
        qc.apply_circuit(self._eiH)
        qc.apply_circuit(U.adjoint())
        coeffs = qc.get_coeff_vec()

        # 3) extract residuals for each pool index
        res = np.array([
            coeffs[self._pool_idx_to_coeff_idx[i]]
            for i in range(len(self._pool_obj))
        ], dtype=complex)
        probs = np.abs(res)**2
        probs /= probs.sum()

        # 4) sample one mu and ancilla sign a
        mu = np.random.choice(len(probs), p=probs)
        a = np.sign(res[mu].real) or 1

        # 5) update signed count and parameter
        N_mu[mu] += int(a)
        step = eta * a / shots_seen
        theta[mu] += step

        return step

    def update_ansatz_fci(self, N_mu, theta, eta, shots_seen):
        """Single‐shot update using FCIComputer."""
        # 1) prepare FCIComputer state
        qc_res = qf.FCIComputer(
            self._nel, 
            self._2_spin, 
            self._norb)
        
        qc_res.hartree_fock()


         # do U^dag e^iH U |Phi_o> = |Phi_res>
        qc_res = qf.FCIComputer(
            self._nel, 
            self._2_spin, 
            self._norb) 

        qc_res.hartree_fock()
        
        temp_pool = qf.SQOpPool()
        for tamp, top in zip(self._tamps, self._tops):
            temp_pool.add(tamp, self._pool_obj[top][1])

        qc_res.evolve_pool_trotter_basic(
            temp_pool,
            antiherm=True,
            adjoint=False)

        # time evolve the hamiltonain
        qc_res.evolve_pool_trotter_basic(
            self._hermitian_pairs,
            antiherm=False,
            adjoint=False)

        qc_res.evolve_pool_trotter_basic(
            temp_pool,
            antiherm=True,
            adjoint=True)

        R = qc_res.get_state_deep()

        # R2.square_modulize()

        IJ = R.sample_index_by_weight_normal(3)
        a = np.sign( np.real(R.get(IJ) )) or 1
        if(IJ == [0,0]):
            return eta / shots_seen
        
        mu = self._coeff_idx_to_pool_idx_fci[tuple(IJ)]

        # 2) apply circuits Uprep, U, e^{-i dt H}, U†
        # U = self.ansatz_circuit(theta)
        # qc_res.apply_circuit(self._Uprep)
        # qc_res.apply_circuit(U)
        # qc_res.apply_circuit(self._eiH)
        # qc_res.apply_circuit(U.adjoint())

        # # 3) extract residuals from FCI state
        # state = qc_res.get_state_deep()
        # res = np.array([
        #     state.get(self._pool_idx_to_coeff_idx_fci[i], 0.0)
        #     for i in range(len(self._pool_obj))
        # ], dtype=complex)
        # probs = np.abs(res)**2
        # probs /= probs.sum()

        # 4) sample one mu and sign a
        # mu = np.random.choice(len(probs), p=probs)
        # a = np.sign(qc_res[mu].real) or 1

        # 5) update signed count and parameter
        N_mu[mu] += int(a)
        step = eta * a / shots_seen
        theta[mu] += step

        return step
    
    def print_summary_banner(self):
        print('\n\n                ==> SP3QE summary <==')
        print('-----------------------------------------------------------')
        print('Final SPQE Energy:                           ', round(self._Egs, 10))
        if self._max_moment_rank:
            print('Moment-corrected (MP) SPQE Energy:           ', round(self._E_mmcc_mp[-1], 10))
            print('Moment-corrected (EN) SPQE Energy:           ', round(self._E_mmcc_en[-1], 10))
        print('Number of operators in pool:                 ', len(self._pool_obj))
        print('Final number of amplitudes in ansatz:        ', len(self._tamps))
        print('Number of classical parameters used:         ', self._n_classical_params)
        print('Number of CNOT gates in deepest circuit:     ', self._n_cnot)
        print('Number of Pauli term measurements:           ', self._n_pauli_trm_measures)
        print('Number of residual vector evaluations:       ', self._res_vec_evals)
        print('Number of individual residual evaluations:   ', self._res_m_evals)

        print("\n\n")
        print(self._timer)

    def print_options_banner(self):
        print('\n-----------------------------------------------------')
        print('       Selected Shot Streaming PQE (S3PQE)')
        print('-----------------------------------------------------')

        print('\n\n               ==> S3PQE options <==')
        print('---------------------------------------------------------')
        # print('Trial reference state:                   ',  ref_string(self._ref, self._nqb))
        print('Number of Hamiltonian Pauli terms:       ',  self._Nl)
        print('Trial state preparation method:          ',  self._state_prep_type)
        print('Trotter order (rho):                     ',  self._trotter_order)
        print('Trotter number (m):                      ',  self._trotter_number)
        print('Use fast version of algorithm:           ',  str(self._fast))
        if(self._fast):
            print('Measurement varience thresh:             ',  'NA')
        else:
            print('Measurement varience thresh:             ',  0.01)

        print('Use qubit excitations:                   ', self._qubit_excitations)
        print('Use compact excitation circuits:         ', self._compact_excitations)

        # opt_thrsh_str = '{:.2e}'.format(self._opt_thresh)
        # spqe_thrsh_str = '{:.2e}'.format(self._spqe_thresh)
        # print('Optimizer:                               ', self._optimizer)
        # if self._diis_max_dim >=2 and self._optimizer.lower() == 'jacobi':
        #     print('DIIS dimension:                          ', self._diis_max_dim)
        # else:
        #     print('DIIS dimension:                           Disabled')
        # print('Maximum number of micro-iterations:      ',  self._opt_maxiter)
        # print('Micro-iteration residual-norm threshold: ',  opt_thrsh_str)
        print('Maximum excitation rank in operator pool:',  self._pool_type)
        print('Number of operators in pool:             ',  len(self._pool_obj))
        # print('Macro-iteration residual-norm threshold: ',  spqe_thrsh_str)
        # print('Maximum number of macro-iterations:      ',  self._spqe_maxiter)
        print(f"Computer type:                            {self._computer_type}")
        b = False
        if (self._apply_ham_as_tensor):
            b = True
        print('Apply ham as tensor                      ', str(b))

    def run_realistic(self):
        raise NotImplementedError('run_realistic() is not fully implemented for SPQE.')
    
    def verify_run(self):
        self.verify_required_attributes()
        self.verify_required_PQE_attributes()
        self.verify_required_UCCPQE_attributes()

    # def _build_pool(self):
    #     """Construct pool of SQ operators and index mappings."""
    #     ref_int = sum([b << i for i, b in enumerate(self._ref)])
    #     mask_alpha = 0x5555555555555555
    #     mask_beta = mask_alpha << 1
    #     nalpha = sum(self._ref[0::2])
    #     nbeta = sum(self._ref[1::2])

    #     pool = qf.SQOpPool()
    #     self._pool_idx_to_coeff_idx = {}
    #     self._pool_idx_to_coeff_idx_fci = {}

    #     idx = 0
    #     for I in range(1 << self._nqb):
    #         alphas = bin(I & mask_alpha).count('1')
    #         betas = bin(I & mask_beta).count('1')
    #         if alphas == nalpha and betas == nbeta:
    #             excit = bin(ref_int ^ I)[2:].zfill(self._nqb)
    #             if excit.count('1') > 0:
    #                 # determine occupied and virtual orbitals
    #                 occ = [i for i, x in enumerate(reversed(excit)) if x == '1' and self._ref[i] == 1]
    #                 vir = [i for i, x in enumerate(reversed(excit)) if x == '1' and self._ref[i] == 0]
    #                 # symmetry check
    #                 if sq_op_find_symmetry(self._sys.orb_irreps_to_int,
    #                                        [len(occ)-1-i for i in occ],
    #                                        [len(vir)-1-i for i in vir]) != self._irrep:
    #                     continue
    #                 # build SQ operator
    #                 sqop = qf.SQOperator()
    #                 sqop.add(+1.0, vir, occ)
    #                 sqop.add(-1.0, occ[::-1], vir[::-1])
    #                 sqop.simplify()
    #                 pool.add_term(0.0, sqop)

    #                 # map indices for fock
    #                 self._pool_idx_to_coeff_idx[idx] = I

    #                 # map indices for fci
    #                 if self._computer_type == 'fci':
    #                     fci_tmp = qf.FCIComputer(self._nel, self._2_spin, self._norb)
    #                     fci_tmp.hartree_fock()
    #                     fci_tmp.apply_sqop(sqop)
    #                     nz = fci_tmp.get_nonzero_idxs()
    #                     assert len(nz) == 1, "FCI mapping error"
    #                     det_idx = nz[0][0]
    #                     self._pool_idx_to_coeff_idx_fci[idx] = det_idx

    #                 idx += 1

    #     self._pool_obj = pool
