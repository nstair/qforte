import qforte as qf
import numpy as np
 
import time

# Define the reference and geometry lists.
geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    ('H', (0., 0., 5.0)), 
    ('H', (0., 0., 6.0)),
    # ('H', (0., 0., 7.0)), 
    # ('H', (0., 0., 8.0)),
    # ('H', (0., 0., 9.0)), 
    # ('H', (0., 0.,10.0))
    ]

geom = [
    ('H', (0., 0., 1.0)), 
    ('O', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ]

# Psi4 Localization
mol1 = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g', 
    symmetry='c1',
    localize_orbitals=True,
    localize_blocks='split', # or full
    localize_method='pipek-mezey',
    run_fci=1,
    build_qb_ham = False,
    store_mo_ints=0,
    build_df_ham=0,
    df_icut=1.0e-6
    )

# QForte Localization
mol2 = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g', 
    symmetry='c1',
    localize_orbitals=True,
    localize_blocks='qf_split', # only split for now
    localize_method='pipek-mezey',
    run_fci=1,
    build_qb_ham = False,
    store_mo_ints=0,
    build_df_ham=0,
    df_icut=1.0e-6
    )
 
print("\n Initial FCIcomp Stuff")
print("===========================")
ref1 = mol1.hf_reference


nel = sum(ref1)
sz = 0
norb = int(len(ref1) / 2)

print(f" nqbit:     {norb*2}")
print(f" nel:       {nel}")
 
fc1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb) # Standard Trotter Order


fci1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb) # Exact Taylor Evolution (for comparison)



# reference = 'random'
reference = 'hf'

if(reference == 'hf'):
    fc1.hartree_fock()
    fci1.hartree_fock()
    

elif(reference == 'random'):
    np.random.seed(42)
    random_array = np.random.rand(fci1.get_state().shape()[0], fci1.get_state().shape()[1])
    random = np.array(random_array, dtype = np.dtype(np.complex128))

    Crand = qf.Tensor(fci1.get_state().shape(), "Crand")
    Crand.fill_from_nparray(random.ravel(), Crand.shape())
    rand_nrm = Crand.norm()
    Crand.scale(1/rand_nrm)

    fc1.set_state(Crand)
    fci1.set_state(Crand)

sqham1 = mol1.sq_hamiltonian
sqham2 = mol2.sq_hamiltonian

# print intial HF energies:
print("\n Initial HF Energies")
print("===========================")
print(f"Initial HF Energy from P4 Loc: {np.real(fc1.get_exp_val(sqham1)):+4.10f}")
print(f"Initial HF Energy from QF Loc: {np.real(fci1.get_exp_val(sqham2)):+4.10f}")
print("\n\n")

hp1 = qf.SQOpPool()

hp1.add_hermitian_pairs(1.0, sqham1) # hf orbs

W1 = hp1.get_commutativity_graph()

# This may not be working properly!
# hp1.reorder_terms_from_graph(W1)



time = 0.1

r = 1
order = 1
N = 1

print(f"dt:    {time}")
print(f"r:     {r}")
print(f"order: {order}")


print("\n\n")
print("Evolving with Trotter")
print("=====================================")
for i in range(N):
    # Call Trotter for fc1
    fc1.evolve_pool_trotter(
        hp1,
        time,
        r,
        order,
        antiherm=False,
        adjoint=False)

    # print(fc1.str(print_complex=False))
    # print(fc1.get_state().norm())

    # Call full taylor evolution for fc2
    fci1.evolve_op_taylor(
        sqham1,
        time,
        1.0e-15,
        30,
        False)

    # print(fc2)
    # print(fc2.get_state().norm())

    Ct1 = fc1.get_state_deep()
    C1 = fci1.get_state_deep()

    C1.subtract(Ct1)

    # print(C1)
    print(f" t: {(i+1)*time:6.4f}  |dC|: {C1.norm():6.10f}")


