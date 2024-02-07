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

def create_fermi_hubbard(**kwargs):
        

    
    # describe what the funtion does...

    """Creates ...

    nel: 
        The number of electorns
    nsites:
        The number of available sites
    tunneling_term:
        The energy associated with tunneling
    coulomb_term:
        The energy associated with coulombic repulsion

        .... (write descriptions for other arguments) 
    """

    nel = kwargs['nel']
    nsites = kwargs['nsites']
    tunneling_term = kwargs['tunneling_term']
    coulomb_term = kwargs['coulomb_term']
    basis = kwargs['basis']
    run_scf = kwargs['run_scf']
    run_fci = kwargs['run_fci']
    pbc = kwargs['pbc']

    fh = System()
    

    # setp 1 call pyscf (see the page I sent in the link https://pyscf.org/develop/scf_developer.html) 
    # to run the hubbard model and get the electron integrals as numpy arrays

    # gives code access to numpy and pyscf
    import numpy as np
    try:
        from pyscf import gto, scf, ao2mo, mcscf        
    except:
        raise ImportError("Psi4 was not imported correctely.")


    # an empty pyscf molecule
    mol = gto.M()

    # incore_anyway=True ensures the customized Hamiltonian (the _eri attribute)
    # is used.  Without this parameter, the MO integral transformation used in
    # subsequent post-HF calculations may
    # ignore the customized Hamiltonian if there is not enough memory.
    mol.incore_anyway = True

    # will need need to be set to what whatever is passed into this funciton
    n = nsites
    mol.nelectron = nel

    # throw an error message if nel > 2*nsites
    if nel>2*nsites:
        raise ValueError("number of electrons exceeds number of accessible positions (2 x number of sites)")
    
    # throw an error message if there is an uneven number of electrons
    if nel%2 != 0:
        raise ValueError("Please enter even number of electons")


 
    # the dimension of the:
    # 1-electorn integrals is nsites x nsites (same as norb x norb)
    # 2-electorn integrals is nsites x nsites x nsites x nsites (same as norb x norb) 
    # frozen core is 0
    # once we get 1 and 2 body hamiltonian(mo_oeis) loop through and build second quantized operators
    # then get initial guess for these second quantized operators(hamiltonians) using pyscf and finally use algorithms
    # that utilize the initial guess to generate a more accurate solution

    # update with tunneling and coulomb coefficents as appropriate
    h1 = np.zeros((n,n))
    for i in range(n-1):
        h1[i,i+1] = h1[i+1,i] = -tunneling_term
    if pbc:
        h1[n-1,0] = h1[0,n-1] = -tunneling_term  # PBC (Periodic bounty conditions, may want to make optional)

    h2 = np.zeros((n,n,n,n))
    for i in range(n):
        h2[i,i,i,i] = coulomb_term

    # run self consistant (mean) field calculation
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(n)

    # ao2mo.restore(8, eri, n) to get 8-fold permutation symmetry of the integrals
    # ._eri only supports two-electron integrals with 4-fold or 8-fold symmetry.
    mf._eri = ao2mo.restore(8, h2, n)
    mf.init_guess = '1e'
    Escf = mf.kernel() 
    
    # CHANGED SECOND TWO TERMS TO NSITES AND NEL
    mycas = mcscf.CASSCF(mf, nsites, nel)
    mycas.kernel()
    fci_energy = mycas.e_cas
    # kernel builds it
    
    # Get the 1-electron integrals

    
    mo_coeff = mycas.mo_coeff

    h1_cas = mycas.get_hcore()
    
    

    h1_cas = np.dot(mo_coeff.T, np.dot(h1_cas, mo_coeff))
    

    # Get the 2-electron integrals

    h2_cas = ao2mo.incore.full(mycas._scf._eri, mo_coeff)

    h2_cas = ao2mo.restore(1, h2_cas, n)

 

# Now h1_cas and h2_cas contain the 1-electron and 2-electron integrals

# in the CASSCF active space.

 

    # post-SCF calculation
    # may throw an error


    # no active indicies
    # active_indices = []

    # mo_oeis, constant = mycas.get_h1cas()
    # mo_teis = mycas.get_h2cas()

    #might not need this line
    #h2e = ao2mo.addons.restore('1', h2e, len(active_indices))

    # or this line ...
    #mo_teis = h2e.transpose(0, 2, 3, 1)

    # once we have the integrals (h1e and h2e from pyscf...)
    # See qforte/adapters/molecule_adapters.py if helful


    # the "integrals" come out as numpy arrays


    # Construct a sq Hamiltonain, and fill it based on the pyscf integrals 
   
    sq_hubbard_ham = qf.SQOperator()
    frozen_virtual = 0
    frozen_core = 0
    frozen_core_energy = 0

    ## CHANGED NMO TO NSITES
    nmo = nsites
    # Make hf_reference
    hf_reference = [1] * (nel - 2 * frozen_core) + [0] * (2 * (nmo - frozen_virtual) - nel)

    for i in range(frozen_core, nmo - frozen_virtual):
        ia = (i - frozen_core)*2
        ib = (i - frozen_core)*2 + 1
        for j in range(frozen_core, nmo - frozen_virtual):
            ja = (j - frozen_core)*2
            jb = (j - frozen_core)*2 + 1

            sq_hubbard_ham.add(h1_cas[i,j], [ia], [ja])
            sq_hubbard_ham.add(h1_cas[i,j], [ib], [jb])

            for k in range(frozen_core, nmo - frozen_virtual):
                ka = (k - frozen_core)*2
                kb = (k - frozen_core)*2 + 1
                for l in range(frozen_core, nmo - frozen_virtual):
                    la = (l - frozen_core)*2
                    lb = (l - frozen_core)*2 + 1

                    if(ia!=jb and kb != la):
                        sq_hubbard_ham.add( h2_cas[i,l,k,j]/2, [ia, jb], [kb, la] ) # abba
                    if(ib!=ja and ka!=lb):
                        sq_hubbard_ham.add( h2_cas[i,l,k,j]/2, [ib, ja], [ka, lb] ) # baab

                    if(ia!=ja and ka!=la):
                        sq_hubbard_ham.add( h2_cas[i,l,k,j]/2, [ia, ja], [ka, la] ) # aaaa
                    if(ib!=jb and kb!=lb):
                        sq_hubbard_ham.add( h2_cas[i,l,k,j]/2, [ib, jb], [kb, lb] ) # bbbb
    
    # Assign the SQ Hamiltonain as the system hamiltonain
    # Set attributes
    fh.nuclear_repulsion_energy = 0
    fh.hf_energy = Escf
    fh.fci_energy = fci_energy
    fh.hf_reference = hf_reference
    fh.hamiltonian = sq_hubbard_ham.jw_transform()
    # fh.point_group = [point_group, irreps]
    #fh.orb_irreps = orb_irreps
    #fh.orb_irreps_to_int = orb_irreps_to_int
    #fh.hf_orbital_energies = hf_orbital_energies #assign this "how to get orbital energies from pyscf"
    fh.frozen_core = frozen_core
    fh.frozen_virtual = frozen_virtual
    fh.frozen_core_energy = frozen_core_energy
    fh.sq_hamiltonian = sq_hubbard_ham
    fh.conv = mf.converged

    # Assign all other atributes of the System class (ask nick if questions)


    # return the model
    return fh