import qforte as qf


geom = [
    # ('Be', (0., 0., 1.00)), 
    ('H', (0., 0., 1.00)),
    ('H', (0., 0., 2.00)),
    ('H', (0., 0., 3.00)),
    ('H', (0., 0., 4.00)),
    ('H', (0., 0., 5.00)),
    ('H', (0., 0., 6.00)),
    # ('H', (0., 0., 7.00)),
    # ('H', (0., 0., 8.00)),
    # ('H', (0., 0., 9.00)),
    # ('H', (0., 0., 10.00))
    ]

# geom = [
#     ('Be', (0., 0., 2.00)), 
#     ('H', (0., 0., 1.00)),
#     ('H', (0., 0., 3.00)),
#     ]

# geom = [
#     ('N', (0., 0., 1.00)),
#     ('N', (0., 0., 2.00)),
#     ]

kmax = 3

timer = qf.local_timer()

timer.reset()

mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g',
    symmetry='d2h',
    run_fci=1)

timer.record("Psi4 Setup")


UCCSD = qf.UCCNVQE(
    mol,
    computer_type = 'fci',
    apply_ham_as_tensor = True
    )

timer.reset()
UCCSD.run(opt_thresh=1.0e-4, 
            pool_type='SD',
            optimizer='BFGS',
            )

timer.record("dUCCSD FCI")

print(timer)
            
print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')


pool_str = f'{kmax}-UpCCGSD'


pUCCGSD = qf.UCCNVQE(
    mol,
    computer_type = 'fci',
    apply_ham_as_tensor = True,
    verbose=False
    )

timer.reset()
pUCCGSD.run(opt_thresh=1.0e-4, 
            pool_type=pool_str,
            optimizer='BFGS',
            opt_maxiter=500,
            use_analytic_grad=True
            )

timer.record("k FCI")

print(timer)

print('\n\n')
print(f'Efci:        {mol.fci_energy:+12.10f}')
print(f'Euccsd:      {UCCSD.get_gs_energy():+12.10f}')
print(f'|dEuccsd|:   {mol.fci_energy - UCCSD.get_gs_energy():+12.10f}')
print(f'Ekupcc:      {pUCCGSD.get_gs_energy():+12.10f}')
print(f'|dEkupcc|:   {mol.fci_energy - pUCCGSD.get_gs_energy():+12.10f}')