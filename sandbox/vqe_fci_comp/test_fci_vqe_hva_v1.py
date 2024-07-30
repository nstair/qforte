import qforte as qf


geom = [
    ('H', (0., 0., 1.00)),
    ('H', (0., 0., 2.00)),
    ('H', (0., 0., 3.00)),
    ('H', (0., 0., 4.00)),
    # ('H', (0., 0., 5.00)),
    # ('H', (0., 0., 6.00)),
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

timer = qf.local_timer()

timer.reset()

mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g',
    run_fci=1)

timer.record("Psi4 Setup")

print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')

alg = qf.HVAVQE(
    mol,
    computer_type = 'fci',
    apply_ham_as_tensor = True,
    )

timer.reset()
alg.run(
    opt_thresh=1.0e-6,
    opt_ftol=1.0e-6,
    opt_maxiter=100,
    pool_type='SQHVA',
    optimizer='BFGS',
    use_analytic_grad=True,
    start_from_ham_params=True,
    noise_factor=1.0e-12)

timer.record("HVA FCI")

print(timer)
            
print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')

