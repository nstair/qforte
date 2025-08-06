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

# geom = [
#     ('H', (0., 0., 1.0)), 
#     ('O', (0., 0., 2.0)),
#     ('H', (0., 0., 3.0)), 
#     ]

mol1 = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g', 
    localize_orbitals=False,
    # localize_blocks='split', # or full
    # localize_method='pipek-mezey',
    run_fci=1,
    build_qb_ham = False,
    store_mo_ints=0,
    build_df_ham=0,
    df_icut=1.0e-6
    )

mol2 = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g', 
    localize_orbitals=True,
    localize_blocks='split', # or full
    localize_method='pipek-mezey',
    run_fci=1,
    build_qb_ham = False,
    store_mo_ints=0,
    build_df_ham=0,
    df_icut=1.0e-6
    )

mol3 = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g', 
    localize_orbitals=True,
    localize_blocks='full', # or full
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
ref2 = mol2.hf_reference
ref3 = mol3.hf_reference

nel = sum(ref1)
sz = 0
norb = int(len(ref1) / 2)

print(f" nqbit:     {norb*2}")
print(f" nel:       {nel}")
 
fc1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb) # Standard Trotter Order
fc2 = qf.FCIComputer(nel=nel, sz=sz, norb=norb) # Optemized Trotter Order
fc3 = qf.FCIComputer(nel=nel, sz=sz, norb=norb) # Random Trotter Order

fci1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb) # Exact Taylor Evolution (for comparison)
fci2 = qf.FCIComputer(nel=nel, sz=sz, norb=norb) # Exact Taylor Evolution (for comparison)
fci3 = qf.FCIComputer(nel=nel, sz=sz, norb=norb) # Exact Taylor Evolution (for comparison)


reference = 'random'
# reference = 'hf'

if(reference == 'hf'):
    fc1.hartree_fock()
    fc2.hartree_fock()
    fc3.hartree_fock()
    fci1.hartree_fock()
    fci2.hartree_fock()
    fci3.hartree_fock()
    

elif(reference == 'random'):
    np.random.seed(42)
    random_array = np.random.rand(fci1.get_state().shape()[0], fci1.get_state().shape()[1])
    random = np.array(random_array, dtype = np.dtype(np.complex128))

    Crand = qf.Tensor(fci1.get_state().shape(), "Crand")
    Crand.fill_from_nparray(random.ravel(), Crand.shape())
    rand_nrm = Crand.norm()
    Crand.scale(1/rand_nrm)

    fc1.set_state(Crand)
    fc2.set_state(Crand)
    fc3.set_state(Crand)
    fci1.set_state(Crand)
    fci2.set_state(Crand)
    fci3.set_state(Crand)

sqham1 = mol1.sq_hamiltonian
sqham2 = mol2.sq_hamiltonian
sqham3 = mol3.sq_hamiltonian

# print intial HF energies:
print("\n Initial HF Energies")
print("===========================")
print(f"Initial HF Energy 1: {np.real(fc1.get_exp_val(sqham1)):+4.10f}")
print(f"Initial HF Energy 2: {np.real(fc2.get_exp_val(sqham2)):+4.10f}")
print(f"Initial HF Energy 3: {np.real(fc3.get_exp_val(sqham3)):+4.10f}")
print("\n\n")

hp1 = qf.SQOpPool()
hp2 = qf.SQOpPool()
hp3 = qf.SQOpPool()

hp1.add_hermitian_pairs(1.0, sqham1) # hf orbs
hp2.add_hermitian_pairs(1.0, sqham2) # split loc
hp3.add_hermitian_pairs(1.0, sqham3) # full loc

W1 = hp1.get_commutativity_graph()
W2 = hp2.get_commutativity_graph()
W3 = hp3.get_commutativity_graph()

# This may not be working properly!
# hp1.reorder_terms_from_graph(W1)
# hp2.reorder_terms_from_graph(W2)
# hp3.reorder_terms_from_graph(W3)

# seed = 42
# hp3.shuffle_terms_random(seed)

# print(W)
# print('sqham')
# print(sqham)

# print('\nhemitian_pairs_1')
# print(hp1)
# print('\nhemitian_pairs_2')
# print(hp2)

time = 0.1

r = 1
order = 1
N = 10

print(f"dt:    {time}")
print(f"r:     {r}")
print(f"order: {order}")

# sqham_used = sqham1
# sqham_used = sqham2
# sqham_used = sqham3


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
    
    fc2.evolve_pool_trotter(
        hp2,
        time,
        r,
        order,
        antiherm=False,
        adjoint=False)
    
    fc3.evolve_pool_trotter(
        hp3,
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
    
    fci2.evolve_op_taylor(
        sqham2,
        time,
        1.0e-15,
        30,
        False)
    
    fci3.evolve_op_taylor(
        sqham3,
        time,
        1.0e-15,
        30,
        False)

    # print(fc2)
    # print(fc2.get_state().norm())

    Ct1 = fc1.get_state_deep()
    Ct2 = fc2.get_state_deep()
    Ct3 = fc3.get_state_deep()
    C1 = fci1.get_state_deep()
    C2 = fci2.get_state_deep()
    C3 = fci3.get_state_deep()

    C1.subtract(Ct1)
    C2.subtract(Ct2)
    C3.subtract(Ct3)

    # print(C1)
    print(f" t: {(i+1)*time:6.4f}  |dC| HF: {C1.norm():6.10f}   |dC| SL: {C2.norm():6.10f}   |dC| FL: {C3.norm():6.10f}")




