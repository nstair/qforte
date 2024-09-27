import qforte as qf
import numpy as np

from qforte.helper.df_ham_helper import *

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

tmr = qf.local_timer()

tmr.reset()
# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g', 
    build_qb_ham = False, 
    run_fci=1,
    store_mo_ints=1,
    build_df_ham=1,  # NOTE(for Victor): Need to do in order build  df_ham
    df_icut=1e-6)

tmr.record("psi4 setup")

sq_ham = mol.sq_hamiltonian
 
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
fc3 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)


fc1.hartree_fock()
fc2.hartree_fock()
fc3.hartree_fock()

rand = False
if(rand):
    random_array = np.random.rand(fc1.get_state().shape()[0], fc1.get_state().shape()[1])
    random = np.array(random_array, dtype = np.dtype(np.complex128))
    Crand = qf.Tensor(fc1.get_state().shape(), "Crand")
    Crand.fill_from_nparray(random.ravel(), Crand.shape())
    rand_nrm = Crand.norm()
    Crand.scale(1/rand_nrm)
    fc1.set_state(Crand)
    fc2.set_state(Crand)

dt = 1.0

# NOTE(for Victor): Must scale the first leaf with dt!
qf.helper.df_ham_helper.time_scale_first_leaf(mol.df_ham, dt)
v_lst = mol.df_ham.get_scaled_density_density_matrices()
gt_lst = mol.df_ham.get_trotter_basis_change_matrices()
g_lst = mol.df_ham.get_basis_change_matrices()

# sets augmented 1-body factorizaiton
do_augmented_one_body_factorization(mol.df_ham)

print(f"\nnorb {len(geom)} len v_lst {len(v_lst)} len g_lst {len(g_lst)} en gt_lst {len(gt_lst)} norb^2 {norb**2}\n")



tmr.reset()
fc1.apply_sqop(mol.sq_hamiltonian)
tmr.record("apply sq ham")

tmr.reset()
fc2.apply_df_ham(mol.df_ham, mol.nuclear_repulsion_energy)
tmr.record("apply df ham")

tmr.reset()
fc3.apply_df_ham_all_givens(mol.df_ham, mol.nuclear_repulsion_energy)
tmr.record("apply df ham all givens")

E1 = np.real(fc1.get_hf_dot())
E2 = np.real(fc2.get_hf_dot())
E3 = np.real(fc3.get_hf_dot())


C1 = fc1.get_state_deep()
C2 = fc2.get_state_deep()
C3 = fc3.get_state_deep()

# print(C1)
# print(C2)

# C1.subtract(C2)
C1.subtract(C3)

diff_norm = C1.norm()

print("")
print(f"||dC2||: {diff_norm}")
print("")

print(f"E1:      {E1:+6.10f}")
print(f"E2:      {E2:+6.10f}")
print(f"E3:      {E3:+6.10f}")
print(f"dE2:     {np.abs(E1-E2):e}")
print(f"dE3:     {np.abs(E1-E3):e}")

print("")
print(tmr)



