import qforte as qf


h2_geom = [
    # ('Be', (0., 0., 1.00)), 
    ('H', (0., 0., 1.00)),
    ('H', (0., 0., 2.00)),
    # ('H', (0., 0., 3.00)),
    # ('H', (0., 0., 4.00)),
    # ('H', (0., 0., 5.00)),
    # ('H', (0., 0., 6.00)),
    # ('H', (0., 0., 7.00)),
    # ('H', (0., 0., 8.00)),
    # ('H', (0., 0., 9.00)),
    # ('H', (0., 0., 10.00))
    ]

h4_geom = [
    # ('Be', (0., 0., 1.00)), 
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

h8_geom = [
    # ('Be', (0., 0., 1.00)), 
    ('H', (0., 0., 1.00)),
    ('H', (0., 0., 2.00)),
    ('H', (0., 0., 3.00)),
    ('H', (0., 0., 4.00)),
    ('H', (0., 0., 5.00)),
    ('H', (0., 0., 6.00)),
    ('H', (0., 0., 7.00)),
    ('H', (0., 0., 8.00)),
    # ('H', (0., 0., 9.00)),
    # ('H', (0., 0., 10.00))
    ]

h10_geom = [
    # ('Be', (0., 0., 1.00)), 
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

h20_geom = [

    ('H', (0., 0., 1.0)),
    ('O', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)),

]

LiH_geom = [

    ('Li', (0., 0., 1.0)),
    ('H', (0., 0., 2.0)),

    ]

geom_list = [

    # Ethylyne_geom,

    h2_geom,

    h4_geom,

    h8_geom,

    # BeH2_geom,

    LiH_geom,

    # N2_geom,

    h20_geom,

    h10_geom,

    # H12_geom,

    # H14_geom

    ]

name_list = [

    # 'Ethylyne',

    'H2',

    'H4',

    # 'H6',

    'H8',

    # 'BeH2',

    'LiH',

    # 'N2',

    'H2O',

    'H10',

    # 'H12',

    # 'H14'

    ]
timer = qf.local_timer()
for geom, name in zip(geom_list, name_list):

    mol = qf.system_factory(
        build_type='psi4', 
        mol_geometry=geom, 
        basis='sto-3g',
        run_fci=1)

    alg_fock = qf.UCCNVQE(
        mol,
        computer_type = 'fock'
        )

    timer.reset()

    alg_fock.run(
        opt_thresh=1.0e-2, 
        pool_type='SD',
        optimizer='BFGS'
        )
    timer.record(f"fock {name}")
    print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')


    alg_fci = qf.UCCNVQE(
        mol,
        computer_type = 'fci',
        apply_ham_as_tensor=True
        )
    timer.reset()
    alg_fci.run(opt_thresh=1.0e-2, 
                pool_type='SD',
                optimizer='BFGS')

    timer.record(f"fci {name}")
    print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')


print(timer)