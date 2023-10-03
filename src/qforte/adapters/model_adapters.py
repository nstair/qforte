import qforte as qf
from qforte.system.molecular_info import System

def create_TFIM(n: int, h: float, J: float):
    """Creates a 1D Transverse Field Ising Model hamiltonian with
    open boundary conditions, i.e., no interaction between the
    first and last spin sites.

    n: int
        Number of lattice sites

    h: float
        Strength of magnetic field

    j: float
        Interaction strength 
    """

    TFIM = System()
    TFIM.hamiltonian = qf.QubitOperator()

    circuit = [(-h, f"Z_{i}") for i in range(n)]
    circuit += [(-J, f"X_{i} X_{i+1}") for i in range(n-1)]

    for coeff, op_str in circuit:
        TFIM.hamiltonian.add(coeff, qf.build_circuit(op_str))

    TFIM.hf_reference = [0] * n

    return TFIM

def create_fermi_hubbard(
        nel,
        nsites,
        tunneling_term, # J
        coulomb_term,   # U
        basis='sto-3g', 
        run_scf=0,
        run_fci=0
        ):
    
    # describe what the funtion does...

    """Creates ...

    nel: 
        The numebr of electorns:

        .... (write descriptions for other arguments) 
    """

    fh = System()
    

    # call pyscf (see the page I sent in the link https://pyscf.org/develop/scf_developer.html) 
    # to run the hubbard model and get the integrals
    # See qforte/adapters/molecule_adapters.py if helful

    # the "integrals" come out as numpy arrays


    # Construct a sq Hamiltonain, and fill it based on the pyscf integrals (see code in )
    sq_hubbard_ham = qf.SQOperator()

    # See line 171 in qforte/adapters/molecule_adapters.py

    # Assign the SQ Hamiltonain as the system hamiltonain
    fh.sq_hamiltonian = sq_hubbard_ham

    # Assign all other atributes of the System class (ask nick if questions)

    print("Hello Emmett!")

    # return the model
    return fh
