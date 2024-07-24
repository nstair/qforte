import qforte as qf


geom = [
    # ('Be', (0., 0., 1.00)), 
    ('H', (0., 0., 1.00)),
    ('H', (0., 0., 2.00)),
    ('H', (0., 0., 3.00)),
    ('H', (0., 0., 4.00)),
    ('H', (0., 0., 5.00)),
    ('H', (0., 0., 6.00))
    ]

timer = qf.local_timer()

timer.reset()

mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g',
    run_fci=1)

timer.record("Psi4 Setup")

timer.reset()

alg_fock = qf.SPQE(
    mol,
    computer_type = 'fock'
    )

alg_fock.run(
    opt_thresh=1.0e-2,
    spqe_maxiter=1)

timer.record("SPQE Fock")
print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')

timer.reset()

alg_fock = qf.SPQE(
    mol,
    computer_type = 'fci'
    )

alg_fock.run(
    opt_thresh=1.0e-2,
    spqe_maxiter=1)

timer.record("SPQE FCI")

print(timer)
print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')