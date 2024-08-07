# This tests the new 2nd order trotter function, 
# there is no actyal 'fci_new' computer type, its a temporary hack 
# used to test this function agains the old one


import qforte as qf


# geom = [
#     ('H', (0., 0., 1.00)), 
#     ('H', (0., 0., 2.00)),
#     ('H', (0., 0., 3.00)),
#     ('H', (0., 0., 4.00)),
#     ('H', (0., 0., 5.00)), 
#     ('H', (0., 0., 6.00)),
#     ('H', (0., 0., 7.00)),
#     ('H', (0., 0., 8.00)),
#     # ('H', (0., 0., 9.00)),
#     # ('H', (0., 0., 10.00))
#     ]

geom = [
    ('H', (0., 0., 1.0)), 
    ('Be', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ]

mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g',
    run_fci=1)


s = 4
dt = 0.1

# alg_fock = qf.SRQK(
#     mol,
#     computer_type = 'fock',
#     trotter_number=4
#     )

# alg_fock.run(
#     s=s,
#     dt=dt
# )

# print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')

r = 5
order = 2


alg_fci = qf.SRQK(
    mol,
    computer_type = 'fci',
    trotter_number=r,
    trotter_order=order
    )

alg_fci.run(
    s=s,
    dt=dt
    )

Eold = alg_fci.get_gs_energy()

alg_fci_new = qf.SRQK(
    mol,
    computer_type = 'fci',
    trotter_number=r,
    trotter_order=order,
    )

alg_fci_new.run(
    s=s,
    dt=dt,
    use_exact_evolution=True
    # test_option='new'
    )

Enew = alg_fci_new.get_gs_energy()

print('\n\n')
print(f' Efci:    {mol.fci_energy:+12.10f}')
print(f' Eold:    {Eold:+12.10f}')
print(f' Enew:    {Enew:+12.10f}')

print(f' ')
print(f' dEold:   {Eold-mol.fci_energy:+12.10f}')
print(f' dEnew:   {Enew-mol.fci_energy:+12.10f}')

#LGTM!

