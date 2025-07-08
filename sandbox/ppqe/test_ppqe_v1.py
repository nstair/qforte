import qforte as qf


geom = [
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

# geom = [
#     ('Be', (0., 0., 0.00)), 
#     ('H', (0., 0., -1.00)),
#     ('H', (0., 0., +1.00)),
#     ]

# geom = [
#     ('N', (0., 0.,  0.00)), 
#     ('N', (0., 0.,  1.20)),
#     ]

geom = [
        ('O', (0., 0., 0.00)), 
        ('H', (0., 0., -1.00)),
        ('H', (0., 0., +1.00)),
        ]  

timer = qf.local_timer()

timer.reset()

mol = qf.system_factory(
    build_type='psi4', 
    symmetry='d2h',
    mol_geometry=geom, 
    basis='sto-6g',
    nroots_fci=4,
    run_fci=1)

timer.record("Psi4 Setup")


r_g_opt_thresh = 1.0e-4
pool_type = 'SD'
noise_factor = 1.0e-4  # Set to zero for exact residuals
dt = 0.01
update_type = 'jacobi_like'
# update_type = 'two_level_rotation'
# update_type = 'two_level_rotation_im'

# ===> VQE <===
timer.reset()

alg_pqe = qf.UCCNVQE(
    mol,
    computer_type = 'fci'
    )

alg_pqe.run(
    opt_thresh=r_g_opt_thresh, 
    pool_type=pool_type,
    noise_factor=noise_factor,
)

timer.record("VQE FCI")

print(timer)
print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')


# ===> PQE <===
timer.reset()

alg_pqe = qf.UCCNPQE(
    mol,
    computer_type = 'fci'
    )

alg_pqe.run(
    opt_thresh=r_g_opt_thresh, 
    pool_type=pool_type,
    noise_factor=noise_factor
)

timer.record("PQE FCI")

print(timer)
print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')


# ===> PPQE <===

timer.reset()

alg_ppqe = qf.UCCNPPQE(
    mol,
    computer_type = 'fci'
    )

alg_ppqe.run(
    pool_type=pool_type,
    opt_thresh = r_g_opt_thresh,
    opt_e_thresh= max(1.0e-1 * noise_factor, 1.0e-8), # note this is an odd choice
    opt_maxiter = 200,
    noise_factor = noise_factor,
    time_step = dt,
    optimizer = 'rotation',
    update_type = update_type, 
    ppqe_trotter_order = 2, 
    )

timer.record("PPQE FCI")

print(timer)
print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')

# print(mol.fci_energy_list)