import qforte as qf
import numpy as np
 
import time

# Define the reference and geometry lists.
geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    # ('H', (0., 0., 5.0)), 
    # ('H', (0., 0., 6.0)),
    # ('H', (0., 0., 7.0)), 
    # ('H', (0., 0., 8.0)),
    # ('H', (0., 0., 9.0)), 
    # ('H', (0., 0.,10.0)),
    # ('H', (0., 0.,11.0)), 
    # ('H', (0., 0.,12.0))
    ]


# geom = [
#     ('H', (0., 0., 1.0)), 
#     ('Be', (0., 0., 2.0)),
#     ('H', (0., 0., 3.0)), 
#     ]

# geom = [('Li', [0.0, 0.0, 0.0]), ('H', [0.0, 0.0, 1.45])]

# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g', 
    build_qb_ham = False, 
    run_fci=1,
    store_mo_ints=1,
    build_df_ham=1,  # NOTE(for Victor): Need to do in order build  df_ham
    df_icut=1.0e-10)
 
print("\n Initial FCIcomp Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

print(f" nqbit:     {norb*2}")
print(f" nel:       {nel}")
 
fc1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fc2 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fc1.hartree_fock()
fc2.hartree_fock()

rand = True
if(rand):
    np.random.seed(11)
    random_array = np.random.rand(fc1.get_state().shape()[0], fc1.get_state().shape()[1])
    random = np.array(random_array, dtype = np.dtype(np.complex128))
    Crand = qf.Tensor(fc1.get_state().shape(), "Crand")
    Crand.fill_from_nparray(random.ravel(), Crand.shape())
    rand_nrm = Crand.norm()
    Crand.scale(1/rand_nrm)
    fc1.set_state(Crand)
    fc2.set_state(Crand)
 
dim = 2*norb
max_nbody = 2
 
Top = qf.TensorOperator(
    max_nbody = max_nbody,
    dim = dim
    )
 
print("\n SQOP Stuff")
print("===========================")
sq0, sq1, sq2 = mol.sq_hamiltonian.split_by_rank(False)
 
# so all seems to be working except
print("\n Tensor Stuff")
print("===========================")
Top.add_sqop_of_rank(sq0, 0)
Top.add_sqop_of_rank(sq1, 2)
Top.add_sqop_of_rank(sq2, 4)
 
[H0, H1spin, H2] = Top.tensors()
# print(H1spin)

# get one body spatial tensor:
# H1spat = mol.mo_oeis # augmented ...
H1spat = mol.df_ham.get_one_body_ints()
# print(H1spat)
 

# fc1.apply_tensor_spin_012bdy(H0, H1, H2, norb)
fc1.apply_sqop(sq1)
fc2.apply_tensor_spat_1bdy(H1spat, norb)


E1 = np.real(fc1.get_hf_dot()) 
E2 = np.real(fc2.get_hf_dot()) 

 
if(norb < 6): 
    print("\n Final FCIcomp Stuff")
    print("===========================")
    print(fc1)
    print(fc2)
 
print("\n Result")
print("======================================================")
print(f" nqbit:     {norb*2}")
print(f" E1:        {E1:6.10f}")
print(f" E2:        {E2:6.10f}")