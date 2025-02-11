"""
Algorithm and AnsatzAlgorithm base classes
==========================================
The abstract base classes inherited by all algorithm subclasses.
"""

from abc import ABC, abstractmethod
import qforte as qf
from qforte.utils.state_prep import *
from qforte.utils.point_groups import sq_op_find_symmetry

class Algorithm(ABC):
    """A class that characterizes the most basic functionality for all
    other algorithms.

    Attributes
    ----------
    _ref : list
        The set of 1s and 0s indicating the initial quantum state.

    _nqb : int
        The number of qubits the calculation empolys.

    _qb_ham : QuantumOperator
        The operator to be measured (usually the Hamiltonian), mapped to a
        qubit representation.

    _fast : bool
        Whether or not to use a faster version of the algorithm that bypasses
        measurment (unphysical for quantum computer). Most algorithms only
        have a fast implentation.

    _trotter_order : int
        The Trotter order to use for exponentiated operators.
        (exact in the infinite limit).

    _trotter_number : int
        The number of trotter steps (m) to perform when approximating the matrix
        exponentials (Um or Un). For the exponential of two non commuting terms
        e^(A + B), the approximate operator C(m) = (e^(A/m) * e^(B/m))^m is
        exact in the infinite m limit.

    _Egs : float
        The final ground state energy value.

    _Umaxdepth : QuantumCircuit
        The deepest circuit used during any part of the algorithm.

    _n_classical_params : int
        The number of classical parameters used by the algorithm.

    _n_cnot : int
        The number of controlled-not (CNOT) opperations used in the (deepest)
        quantum circuit (_Umaxdepth).

    _n_pauli_trm_measures : int
        The number of pauli terms (Hermitian products of Pauli X, Y, and/or Z gates)
        mesaured over the entire algorithm.

    _res_vec_evals : int
        The total number of times the entire residual was evaluated.

    _res_m_evals : int
        The total number of times an individual residual element was evaluated.
    """

    def __init__(self,
                 system,
                 computer_type='fock',
                 apply_ham_as_tensor=True,
                 reference=None,
                 state_prep_type='occupation_list',
                 trotter_order=1,
                 trotter_number=1,
                 fast=True,
                 verbose=False,
                 print_summary_file=False,
                 **kwargs):

        if isinstance(self, qf.QPE) and hasattr(system, 'frozen_core'):
            if system.frozen_core + system.frozen_virtual > 0:
                raise ValueError("QPE with frozen orbitals is not currently supported.")

        self._sys = system
        self._state_prep_type = state_prep_type

        self._ref_from_hf = True

        if self._state_prep_type == 'occupation_list':
            if(reference==None):
                self._ref = system.hf_reference
            else:
                if not (isinstance(reference, list)):
                    raise ValueError("occupation_list reference must be list of 1s and 0s.")
                self._ref = reference
                self._ref_from_hf = False

            self._Uprep = build_Uprep(self._ref, state_prep_type)

        elif self._state_prep_type == 'unitary_circ':
            if not isinstance(reference, qf.Circuit):
                raise ValueError("unitary_circ reference must be a Circuit.")

            self._ref = system.hf_reference
            self._Uprep = reference

            self._ref_from_hf = False

        else:
            raise ValueError("QForte only suppors references as occupation lists and Circuits.")


        self._nqb = len(self._ref)

        self._nael = 0
        self._nbel = 0

        self._nspin_orb = len(self._ref)
        # may cause issues for odd number of orbitals...
        self._norb = int(self._nspin_orb / 2) 

        for i, occ in enumerate(self._ref):
            if occ:
                if (i % 2 == 0):
                    self._nael += 1
                else:
                    self._nbel += 1

        self._nel = self._nael + self._nbel
        self._tot_spin = 0.5 * np.abs(self._nael - self._nbel)
        self._2_spin = int(np.abs(self._nael - self._nbel))

        self._mult = int(2 * self._tot_spin + 1)

        self._qb_ham = system.hamiltonian
        self._sq_ham = system.sq_hamiltonian

        self._zero_body_energy = 0.0

        if(hasattr(system, 'nuclear_repulsion_energy')):
            self._nuclear_repulsion_energy = system.nuclear_repulsion_energy
            self._zero_body_energy += system.nuclear_repulsion_energy
        else:
            print("NOTE: No nuclear repulsion enerly set! Using only provided Hamiltonain.")
            self._nuclear_repulsion_energy = 0.0

        if(hasattr(system, 'frozen_core_energy')):
            self._frozen_core_energy = system.frozen_core_energy
            self._zero_body_energy += system.frozen_core_energy
        else:
            print("NOTE: No frozen core energy repulsion enerly set! Using only provided Hamiltonain.")
            self._frozen_core_energy= 0.0

        if(computer_type=='fci'):
            if(apply_ham_as_tensor):
                self._mo_oeis = system.mo_oeis 
                self._mo_teis = system.mo_teis 
                self._mo_teis_einsum = system.mo_teis_einsum
                

        if len(self._qb_ham.terms()) > 0 and self._qb_ham.num_qubits() != self._nqb:
            raise ValueError(f"The reference has {self._nqb} qubits, but the Hamiltonian has {self._qb_ham.num_qubits()}. This is inconsistent.")
        try:
            self._hf_energy = system.hf_energy
        except AttributeError:
            self._hf_energy = 0.0

        self._Nl = len(self._qb_ham.terms())
        self._trotter_order = trotter_order
        self._trotter_number = trotter_number
        self._fast = fast

        apply_ham_as_tensor_temp = apply_ham_as_tensor

        if(computer_type=='fock'):
            if(apply_ham_as_tensor):
                print(f"Warning, can't apply hamiltonain as tensor for fock computer, setting option to false")
                apply_ham_as_tensor_temp = False
            self._computer_type = 'fock'
        elif(computer_type=='fci'):
            self._computer_type = 'fci'
        else:
            raise ValueError(f"Computer type must be fci or fock.")
        
        self._apply_ham_as_tensor = apply_ham_as_tensor_temp

        self._verbose = verbose
        self._print_summary_file = print_summary_file

        self._noise_factor = 0.0

        # Required attributes, to be defined in concrete class.
        self._Egs = None
        self._Umaxdepth = None
        self._n_classical_params = None
        self._n_cnot = None
        self._n_pauli_trm_measures = None


    @abstractmethod
    def print_options_banner(self):
        """Prints the run options used for algorithm.
        """
        pass

    @abstractmethod
    def print_summary_banner(self):
        """Prints a summary of the post-run information.
        """
        pass

    @abstractmethod
    def run(self):
        """Executes the algorithm.
        """
        pass

    @abstractmethod
    def run_realistic(self):
        """Executes the algorithm using only operations physically possible for
        quantum hardware. Not implemented for most algorithms.
        """
        pass

    @abstractmethod
    def verify_run(self):
        """Verifies that the abstract sub-class(es) define the required attributes.
        """
        pass

    def get_gs_energy(self):
        """Returns the final ground state energy.
        """
        return self._Egs

    def get_Umaxdepth(self):
        """Returns the deepest circuit used during any part of the
        algorithm (_Umaxdepth).
        """
        pass

    def get_tot_measurements(self):
        pass

    def get_tot_state_preparations(self):
        pass

    def verify_required_attributes(self):
        """Verifies that the concrete sub-class(es) define the required attributes.
        """
        if self._Egs is None:
            raise NotImplementedError('Concrete Algorithm class must define self._Egs attribute.')

#         if self._Umaxdepth is None:
#             raise NotImplementedError('Concrete Algorithm class must define self._Umaxdepth attribute.')

        if self._n_classical_params is None:
            raise NotImplementedError('Concrete Algorithm class must define self._n_classical_params attribute.')

        if self._n_cnot is None:
            raise NotImplementedError('Concrete Algorithm class must define self._n_cnot attribute.')

        if self._n_pauli_trm_measures is None:
            raise NotImplementedError('Concrete Algorithm class must define self._n_pauli_trm_measures attribute.')

class AnsatzAlgorithm(Algorithm):
    """A class that characterizes the most basic functionality for all
    other algorithms which utilize an operator ansatz such as VQE.

    Attributes
    ----------
    _curr_energy: float
        The energy at the current iteration step.

    _Nm: list of int
        Number of circuits for each operator in the pool.

    _opt_maxiter : int
        The maximum number of iterations for the classical optimizer.

    _opt_thresh : float
        The numerical convergence threshold for the specified classical
        optimization algorithm. Is usually the norm of the gradient, but
        is algorithm dependant, see scipy.minimize.optimize for details.

    _pool_obj : SQOpPool
        A pool of second quantized operators we use in the ansatz.

    _tops : list
        A list of indices representing selected operators in the pool.

    _tamps : list
        A list of amplitudes (to be optimized) representing selected
        operators in the pool.

    _qubit_excitations: bool
        Controls the use of qubit/fermionic excitations.

    _compact_excitations: bool
        Controls the use of compact quantum circuits for fermion/qubit
        excitations.
    """

    # TODO (opt major): write a C function that prepares this super efficiently
    def build_Uvqc(self, amplitudes=None):
        """ This function returns the Circuit object built
        from the appropriate amplitudes (tops)

        Parameters
        ----------
        amplitudes : list
            A list of parameters that define the variational degrees of freedom in
            the state preparation circuit Uvqc. This is needed for the scipy minimizer.
        """

        U = self.ansatz_circuit(amplitudes)

        Uvqc = qforte.Circuit()
        Uvqc.add(self._Uprep)
        Uvqc.add(U)

        return Uvqc

    def fill_pool(self):
        """ This function populates an operator pool with SQOperator objects.
        """

        timer2 = qforte.local_timer()

        if self._pool_type in {'sa_SD', 'GSD', 'SD', 'SDT', 'SDTQ', 'SDTQP', 'SDTQPH'}:
            self._pool_obj = qf.SQOpPool()

            timer2.reset()
            self._pool_obj.set_orb_spaces(self._ref)
            timer2.record("_pool_obj.set_orb_spaces")

            timer2.reset()
            self._pool_obj.fill_pool(self._pool_type)
            timer2.record("_pool_obj.fill_pool")

        elif (self._pool_type[0].isdigit() and self._pool_type[1:] == '-UpCCGSD'):
            self._pool_obj = qf.SQOpPool()

            timer2.reset()
            self._pool_obj.set_orb_spaces(self._ref)
            timer2.record("_pool_obj.set_orb_spaces")

            timer2.reset()
            self._pool_obj.fill_pool_kUpCCGSD(int(self._pool_type[0]))
            timer2.record("_pool_obj.fill_pool")

        elif isinstance(self._pool_type, qf.SQOpPool):
            self._pool_obj = self._pool_type
        else:
            raise ValueError('Invalid operator pool type specified.')

        # If possible, impose symmetry restriction to operator pool
        # Currently, symmetry is supported for system_type='molecule' and build_type='psi4'
        if hasattr(self._sys, 'point_group'):
            timer2.reset()
            temp_sq_pool = qf.SQOpPool()
            for sq_operator in self._pool_obj.terms():
                create = sq_operator[1].terms()[0][1]
                annihilate = sq_operator[1].terms()[0][2]
                if sq_op_find_symmetry(self._sys.orb_irreps_to_int, create, annihilate) == self._irrep:
                    temp_sq_pool.add(sq_operator[0], sq_operator[1])
            self._pool_obj = temp_sq_pool
            timer2.record("point_group_considerations")

        timer2.reset()
        if(self._computer_type == 'fock'):
            self._Nm = [len(operator.jw_transform().terms()) for _, operator in self._pool_obj]
        elif(self._computer_type == 'fci'):
            self._Nm = [0 for _, operator in self._pool_obj]
            print("\n ==> Warning: resource estimator needs to be implemented for fci computer type <==")
        else:
            raise ValueError(f'{self._computer_type} is an unrecognized computer type.')

        timer2.record("jw transform")

        # print(timer2)

    def measure_energy(self, Ucirc):
        """
        This function returns the energy expectation value of the state
        Uprep|0>.

        Parameters
        ----------
        Ucirc : Circuit
            The state preparation circuit.
        """
        if self._fast:
            myQC = qforte.Computer(self._nqb)
            myQC.apply_circuit(Ucirc)
            val = np.real(myQC.direct_op_exp_val(self._qb_ham))
        else:
            Exp = qforte.Experiment(self._nqb, Ucirc, self._qb_ham, 2000)
            val = Exp.perfect_experimental_avg([])

        assert np.isclose(np.imag(val), 0.0)

        return val

    def __init__(self, *args, qubit_excitations=False, compact_excitations=False, diis_max_dim=8,
            max_moment_rank = 0, moment_dt=None,
            **kwargs):
        super().__init__(*args, **kwargs)
        self._curr_energy = 0
        self._Nm = []
        self._tamps = []
        self._tops = []
        self._pool_obj = qf.SQOpPool()
        self._qubit_excitations = qubit_excitations
        self._compact_excitations = compact_excitations
        self._diis_max_dim = diis_max_dim
        # The max_moment_rank controls the calculation of non-iterative energy corrections
        # based on the method of moments of coupled-cluster theory.
        # max_moment_rank = 0: non-iterative correction skipped
        # max_moment_rank = n: projections up to n-tuply excited Slater determinants are considered
        self._max_moment_rank = max_moment_rank
        # The moment_dt variable defines the 'residual' state used to measure the residuals for the moment corrections
        self._moment_dt = moment_dt

        kwargs.setdefault('irrep', None)
        if hasattr(self._sys, 'point_group'):
            irreps = list(range(len(self._sys.point_group[1])))
            if kwargs['irrep'] is None:
                print('\nWARNING: The {0} point group was detected, but no irreducible representation was specified.\n'
                        '         Proceeding with totally symmetric.\n'.format(self._sys.point_group[0].capitalize()))
                self._irrep = 0
            elif kwargs['irrep'] in irreps:
                self._irrep = kwargs['irrep']
            else:
                raise ValueError("{0} is not an irreducible representation of {1}.\n"
                                 "               Choose one of {2} corresponding to the\n"
                                 "               {3} irreducible representations of {1}".format(kwargs['irrep'],
                                     self._sys.point_group[0].capitalize(),
                                     irreps,
                                     self._sys.point_group[1]))
        elif kwargs['irrep'] is not None:
            print('\nWARNING: Point group information not found.\n'
                    '         Ignoring "irrep" and proceeding without symmetry.\n')

    def energy_feval(self, params):
        """
        This function returns the energy expectation value of the state
        Uprep(params)|0>, where params are parameters that can be optimized
        for some purpouse such as energy minimizaiton.

        Parameters
        ----------
        params : list of floats
            The dist of (real) floating point number which will characterize
            the state preparation circuit.
        """
        if(self._computer_type == 'fock'):
            return self.energy_feval_fock(params)
        elif(self._computer_type == 'fci'):
            return self.energy_feval_fci(params)
        else:
            raise ValueError(f"{self._computer_type} is an unrecognized computer type.") 

    def energy_feval_fock(self, params):
        """
        This function returns the energy expectation value of the state
        Uprep(params)|0>, where params are parameters that can be optimized
        for some purpouse such as energy minimizaiton.

        Parameters
        ----------
        params : list of floats
            The dist of (real) floating point number which will characterize
            the state preparation circuit.
        """
        Ucirc = self.build_Uvqc(amplitudes=params)
        Energy = self.measure_energy(Ucirc)

        self._curr_energy = Energy
        return Energy
    
    def energy_feval_fci(self, params):
        if not self._ref_from_hf:
            raise ValueError('get_residual_vector_fci_comp only compatible with hf reference at this time.')
        
        temp_pool = qforte.SQOpPool()

        # NICK: Write a 'updatte_coeffs' type fucntion for the op-pool.
        for param, top in zip(params, self._tops):
            temp_pool.add(param, self._pool_obj[top][1])

        qc = qforte.FCIComputer(
            self._nel, 
            self._2_spin, 
            self._norb)
        
        qc.hartree_fock()

        qc.evolve_pool_trotter_basic(
            temp_pool,
            antiherm=True,
            adjoint=False)
        
        if(self._apply_ham_as_tensor):
            
            self._curr_energy = np.real(
                qc.get_exp_val_tensor(
                    self._zero_body_energy, 
                    self._mo_oeis, 
                    self._mo_teis, 
                    self._mo_teis_einsum, 
                    self._norb)
            )
        else:   
            self._curr_energy = np.real(qc.get_exp_val(self._sq_ham))

        
        return self._curr_energy
