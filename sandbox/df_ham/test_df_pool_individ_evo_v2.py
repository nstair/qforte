import qforte as qf
import numpy as np
from qforte.helper.df_ham_helper import *

# NOTE(Nick): this sandbox file compares the evolution of the HF wfn
# under trotterized evoltuion of only the first double factorized 't-leaf' 
# relative to fqe, as of 7/12/2024 evolution matches.


def t_diff(Tqf, npt, name, print_both=False):
    print(f"\n  ===> {name} Tensor diff <=== ")
    Tnp = qf.Tensor(shape=np.shape(npt), name='Tnp')
    Tnp.fill_from_nparray(npt.ravel(), np.shape(npt))
    if(print_both):
        print(Tqf)
        print(Tnp)
    Tnp.subtract(Tqf)
    print(f"  ||dT||: {Tnp.norm()}")
    if(Tnp.norm() > 1.0e-12):
        print(Tnp)


geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    ]

nel  = 4
norb = 4
sz = 0


timer = qf.local_timer()

timer.reset()
mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g', 
    build_qb_ham = False,
    run_fci=1,
    store_mo_ints=1,
    build_df_ham=1)

dfh = mol.df_ham

timer.record('Run Psi4 and Initialize')


dt = 0.1


## ====> Set up DF and Trotter Stuff <==== ##

# NOTE(for Victor): Must scale the first leaf with dt!
time_scale_first_leaf(dfh, dt)
# sets augmented 1-body factorizaiton
do_augmented_one_body_factorization(mol.df_ham)

g_lst = mol.df_ham.get_trotter_basis_change_matrices()

v_lst = mol.df_ham.get_scaled_density_density_matrices()
u_lst = mol.df_ham.get_basis_change_matrices()
uo = mol.df_ham.get_aug_one_body_basis_change()
do = mol.df_ham.get_aug_one_body_diag()


## ====> Set up DF Trotter Individal Operators <==== ##
tmp1 = qf.SQOpPool()
tmp2 = qf.SQOpPool()


tmp1.append_givens_ops_sector(
    g_lst[0], 
    1.0/dt,
    True,
    False)

tmp1.append_givens_ops_sector(
    g_lst[0], 
    1.0/dt,
    False,
    False)

# now the new one

tmp2.append_givens_ops_sector(
    uo, 
    1.0/dt,
    True,
    False)


tmp2.append_givens_ops_sector(
    uo, 
    1.0/dt,
    False,
    False)

tmp2.append_one_body_diagonal_ops_all(
    do,
    -1.0)

tmp2.append_givens_ops_sector(
    uo, 
    1.0/dt,
    False,
    True)

tmp2.append_givens_ops_sector(
    uo, 
    1.0/dt,
    True,
    True)

tmp2.append_givens_ops_sector(
    u_lst[0], 
    1.0/dt,
    True,
    False)

tmp2.append_givens_ops_sector(
    u_lst[0], 
    1.0/dt,
    False,
    False)


## ====> set up FCIComputer <==== ##

fc1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fc2 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fc1.hartree_fock()
fc2.hartree_fock()


## ====> First leaf alpha givens <==== ##


fc1.evolve_pool_trotter(
    tmp1,
    dt,
    1,
    1)

fc2.evolve_pool_trotter(
    tmp2,
    dt,
    1,
    1)

print(fc1.str(print_complex=True))
print(fc2.str(print_complex=True))

dC = fc1.get_state_deep()
dC.subtract(fc2.get_state_deep())
print(f"||dCga||: {dC.norm():6.6f}")












