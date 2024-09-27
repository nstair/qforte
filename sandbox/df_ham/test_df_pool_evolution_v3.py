import qforte as qf
import numpy as np
from qforte.helper.df_ham_helper import *
import qf_fqe_helper as qfqe

# NOTE(Nick): this sandbox file compares the evolution of the HF wfn
# under trotterized evoltuion of the double factorized hamiltonain  
# to the exact time evolution (via taylor expansion of e^-itH)


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


timer = qf.local_timer()

timer.reset()

# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g', 
    symmetry='d2h',
    build_qb_ham = False, 
    run_fci=1,
    store_mo_ints=1,
    build_df_ham=1,  # NOTE(for Victor): Need to do in order build  df_ham
    df_icut=1.0e-1)  # NOTE(for Victor): Controlls the level of rank truncation in df_ham!

# NOTE(for Victor): the df_icut (integrall cutoff threshold), will make the v_lst and g_lst shorter, 
# resulting in faster applicaiton time classically (and shorter quantum circuits!)
# note that you can really make it very large as still get pretty good time evolution!

timer.record('Run Psi4 and Initialize')


## ====> Set up Time Step and number of steps <==== ##
dt = 0.5
N = 10
r = 1
order = 1

## ====> Set up DF and Trotter Stuff <==== ##

# NOTE(for Victor): Must scale the first leaf with dt!
time_scale_first_leaf(mol.df_ham, dt)
# sets augmented 1-body factorizaiton
do_augmented_one_body_factorization(mol.df_ham)

v_lst = mol.df_ham.get_scaled_density_density_matrices()
g_lst = mol.df_ham.get_trotter_basis_change_matrices()
u_lst = mol.df_ham.get_basis_change_matrices()
uo = mol.df_ham.get_aug_one_body_basis_change()

print(f"\nnorb {len(geom)} len v_lst {len(v_lst)} len g_lst {len(g_lst)}\n")
sqdf_ham_pool = qf.SQOpPool()
tp1 = qf.SQOpPool()
tp2 = qf.SQOpPool()

# Just for expectaiton values
sqham = mol.sq_hamiltonian



print("")
print(f"len(sqdf_ham_pool.terms()):  {len(sqdf_ham_pool.terms())} ")
print("")


# ===> Set which typer of pool trotterization <=== #
sqdf_ham_pool.fill_pool_df_trotter(mol.df_ham, dt)
# sqdf_ham_pool.fill_pool_df_trotter_all_diag(mol.df_ham, dt)

tp1.append_givens_ops_sector(
    uo,
    dt,
    True,
    False
)

tp1.append_givens_ops_sector(
    uo,
    dt,
    False,
    False
)

tp1.append_givens_ops_sector(
    uo,
    dt,
    True,
    True
)

tp1.append_givens_ops_sector(
    uo,
    dt,
    False,
    True
)

print(tp1)

## ====> set up FCIComputers <==== ##
ref = mol.hf_reference
nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)
print(f"nel {nel} norb {norb}")

fc1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fc2 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fc3 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)

fc1.hartree_fock()
fc2.hartree_fock()
fc3.hartree_fock()


## ====> DF Evolution <==== ##
print(f"dt:    {dt}")
print(f"r:     {r}")
print(f"order: {order}")
print(f"\nmol.nuclear_repulsion_energy: {mol.nuclear_repulsion_energy}")
print(f"\nexp(-i*dt*nre): {np.exp(-1.0j*dt*mol.nuclear_repulsion_energy)}")
print("")

# Global phase comes from the zero-body hamiltonain term (nuclear repulsion energy)
gphase = np.exp(-1.0j*dt*mol.nuclear_repulsion_energy)

fc1.evolve_pool_trotter(
    tp1,
    dt,
    r,
    order,
    antiherm=False,
    adjoint=False)

print(fc1)












