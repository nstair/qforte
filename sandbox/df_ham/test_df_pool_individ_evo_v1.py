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





## ====> Set up DF Trotter Stuff <==== ##
dt = 0.1
time_scale_first_leaf(dfh, dt)

v_lst = dfh.get_scaled_density_density_matrices()
g_lst = dfh.get_trotter_basis_change_matrices()


## ====> Set up DF Trotter Individal Operators <==== ##

ga0 = qf.SQOpPool()
ga0.append_givens_ops_sector(
    g_lst[0], 
    1.0/dt,
    True)

gb0 = qf.SQOpPool()
gb0.append_givens_ops_sector(
    g_lst[0], 
    1.0/dt,
    False)

d0 = qf.SQOpPool()
d0.append_diagonal_ops_all(
    v_lst[0], 
    1.0)



## ====> set up FCIComputer <==== ##

fc1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fc2 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fc1.hartree_fock()
fc2.hartree_fock()


## ====> First leaf alpha givens <==== ##
# fc1.hartree_fock()
# fc2.hartree_fock()

fc1.evolve_givens(
    g_lst[0],
    True # is alpha
)

fc2.evolve_pool_trotter(
    ga0,
    dt,
    1,
    1)

print(fc1)
print(fc2)

dC = fc1.get_state_deep()
dC.subtract(fc2.get_state_deep())
print(f"||dCga||: {dC.norm():6.6f}")


## ====> First leaf beta givens <==== ##
# fc1.hartree_fock()
# fc2.hartree_fock()

fc1.evolve_givens(
    g_lst[0],
    False # is alpha
)

fc2.evolve_pool_trotter(
    gb0,
    dt,
    1,
    1)

# print(fc1)
# print(fc2)

dC = fc1.get_state_deep()
dC.subtract(fc2.get_state_deep())
print(f"||dCgb||: {dC.norm():6.6f}")

## ====> First leaf diagonal <==== ##
# fc1.hartree_fock()
# fc2.hartree_fock()

fc1.evolve_diagonal_from_mat(
            v_lst[0],
            dt
        )

fc2.evolve_pool_trotter(
    d0,
    dt,
    1,
    1)

print(fc1.str(print_complex=True))
print(fc2.str(print_complex=True))

dC = fc1.get_state_deep()
dC.subtract(fc2.get_state_deep())
print(f"||dCdiag||: {dC.norm():6.6f}")

# print(d0)



## ====> Whole Enchiladda <==== ##
# fc1.hartree_fock()
# fc1.evolve_df_ham_trotter(
#       dfh,
#       dt)
# t_diff(fc1.get_state_deep(), full_df_evo_ary, "full_df_evo_ary", print_both=False)










