# This tests the new 2nd order trotter function, 
# there is no actyal 'fci_new' computer type, its a temporary hack 
# used to test this function agains the old one


import qforte as qf
import numpy as np

geom = [
    ('H', (0., 0., 1.00)), 
    ('H', (0., 0., 2.00)),
    ('H', (0., 0., 3.00)),
    ('H', (0., 0., 4.00)),
    ('H', (0., 0., 5.00)), 
    ('H', (0., 0., 6.00)),
    ('H', (0., 0., 7.00)),
    ('H', (0., 0., 8.00)),
    ('H', (0., 0., 9.00)),
    ('H', (0., 0., 10.00))
    ]

timer = qf.local_timer()

timer.reset()
mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    symmetry='d2h',
    basis='sto-3g',
    build_qb_ham=True,
    store_mo_ints=True,
    store_mo_ints_np=True,
    run_fci=0)
timer.record("mol build")


s = 5
dt = 0.25

r = 2
order = 2


alg_fci = qf.SRQK(
    mol,
    computer_type = 'fqe',
    trotter_number=r,
    trotter_order=order,
    )

alg_cusv = qf.SRQK(
    mol,
    computer_type = 'cusv',
    trotter_number=r,
    trotter_order=order,
    apply_ham_as_tensor=False
    )


timer.reset()
alg_fci.run(
    s=s,
    dt=dt,
    use_exact_evolution=False,
    )
timer.record("run alg fci")

Eo_fci_comp = alg_fci.get_gs_energy()

Hfci = alg_fci._Hbar
Sfci = alg_fci._S


timer.reset()
alg_cusv.run(
    s=s,
    dt=dt,
    use_exact_evolution=False,
    )
timer.record("run alg cusv")

Eo_cusv_comp = alg_cusv.get_gs_energy()

Hcusv = alg_cusv._Hbar
Scusv = alg_cusv._S

dHbar = Hfci - Hcusv
dS = Sfci - Scusv
normdH = np.linalg.norm(dHbar)
normdS = np.linalg.norm(dS)

print("\n Check H and S Matrices \n")
print("===========================")
qf.helper.printing.matprint(Sfci)

print("\n")
qf.helper.printing.matprint(Scusv)
print("\n")

print("\n Check Final Energy \n")
print("===========================")
print(f' Efci_comp:  {Eo_fci_comp:+12.10f}')
print(f' Ecusv_comp: {Eo_cusv_comp:+12.10f}')
print(f' E diff:     {Eo_fci_comp - Eo_cusv_comp:+12.10f}')
print(f' |dH|:       {normdH:+12.14f}')
print(f' |dS|:       {normdS:+12.14f}')

# Eold = alg_fci.get_gs_energy()

print(timer)

# print('\n\n')
# print(f' Efci:    {mol.fci_energy:+12.10f}')
# print(f' Eold:    {Eold:+12.10f}')
# print(f' Eold:    {Eold-mol.fci_energy:+12.10f}')


#LGTM!

