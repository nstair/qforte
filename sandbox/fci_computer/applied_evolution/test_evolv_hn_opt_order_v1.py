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
    ('H', (0., 0., 7.0)), 
    ('H', (0., 0., 8.0)),
    # ('H', (0., 0., 9.0)), 
    # ('H', (0., 0.,10.0))
    ]

mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g', 
    run_fci=1,
    build_qb_ham = False,
    store_mo_ints=0,
    build_df_ham=0,
    df_icut=1.0e-6
    )
 
print("\n Initial FCIcomp Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

print(f" nqbit:     {norb*2}")
print(f" nel:       {nel}")
 
fc1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb) # Standard Trotter Order
fc2 = qf.FCIComputer(nel=nel, sz=sz, norb=norb) # Optemized Trotter Order
fc3 = qf.FCIComputer(nel=nel, sz=sz, norb=norb) # Random Trotter Order
fci = qf.FCIComputer(nel=nel, sz=sz, norb=norb) # Exact Taylor Evolution (for comparison)

reference = 'random'
# reference = 'hf'

if(reference == 'hf'):
    fc1.hartree_fock()
    fc2.hartree_fock()
    fc3.hartree_fock()
    fci.hartree_fock()
    

elif(reference == 'random'):
    np.random.seed(42)
    random_array = np.random.rand(fci.get_state().shape()[0], fci.get_state().shape()[1])
    random = np.array(random_array, dtype = np.dtype(np.complex128))

    Crand = qf.Tensor(fci.get_state().shape(), "Crand")
    Crand.fill_from_nparray(random.ravel(), Crand.shape())
    rand_nrm = Crand.norm()
    Crand.scale(1/rand_nrm)

    fc1.set_state(Crand)
    fc2.set_state(Crand)
    fc3.set_state(Crand)
    fci.set_state(Crand)
    
sqham = mol.sq_hamiltonian

hp1 = qf.SQOpPool()
hp2 = qf.SQOpPool()
hp3 = qf.SQOpPool()

hp1.add_hermitian_pairs(1.0, sqham)
hp2.add_hermitian_pairs(1.0, sqham)
hp3.add_hermitian_pairs(1.0, sqham)

W = hp2.get_commutativity_graph()
hp2.reorder_terms_from_graph(W)

seed = 42
hp3.shuffle_terms_random(seed)

# print(W)
# print('sqham')
# print(sqham)

# print('\nhemitian_pairs_1')
# print(hp1)
# print('\nhemitian_pairs_2')
# print(hp2)

time = 0.1


r = 1
order = 2

print(f"dt:    {time}")
print(f"r:     {r}")
print(f"order: {order}")


print("\n\n")
print("Evolving with Trotter")
print("=====================================")
for i in range(10):
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
    fci.evolve_op_taylor(
        sqham,
        time,
        1.0e-15,
        30,
        False)

    # print(fc2)
    # print(fc2.get_state().norm())

    C1 = fc1.get_state_deep()
    C2 = fc2.get_state_deep()
    C3 = fc3.get_state_deep()
    C = fci.get_state_deep()

    C1.subtract(C)
    C2.subtract(C)
    C3.subtract(C)

    # print(C1)
    print(f" t: {(i+1)*time:6.4f}  |dC1|: {C1.norm():6.10f}   |dC2|: {C2.norm():6.10f}   |dC3|: {C3.norm():6.10f}")



