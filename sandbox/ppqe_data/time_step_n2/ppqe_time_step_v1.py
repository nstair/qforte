import qforte as qf
import numpy as np


# geom = [
#     # ('Be', (0., 0., 1.00)), 
#     ('H', (0., 0., 1.00)),
#     ('H', (0., 0., 2.00)),
#     ('H', (0., 0., 3.00)),
#     ('H', (0., 0., 4.00)),
#     ('H', (0., 0., 5.00)),
#     ('H', (0., 0., 6.00)),
#     ('H', (0., 0., 7.00)),
#     ('H', (0., 0., 8.00)),
#     ('H', (0., 0., 9.00)),
#     ('H', (0., 0., 10.00))
#     ]

geom = [
    ('Be', (0., 0., 0.00)), 
    ('H', (0., 0., -1.00)),
    ('H', (0., 0., +1.00)),
    ]

# geom = [
#     ('N', (0., 0.,  0.00)), 
#     ('N', (0., 0.,  1.20)),
#     ]

timer = qf.local_timer()

timer.reset()

mol = qf.system_factory(
    build_type='psi4', 
    symmetry='d2h',
    mol_geometry=geom, 
    basis='sto-3g',
    run_fci=1)

timer.record("Psi4 Setup")


r_g_opt_thresh = 1.0e-6
pool_type = 'SD'
noise_factor = 0.0e-4  # Set to zero for exact residuals
dt = 0.1
# update_type = 'jacobi_like'
update_type = 'two_level_rotation'


# ===> PPQE <===

timer.reset()

alg_ppqe = qf.UCCNPPQE(
    mol,
    computer_type = 'fci'
    )

alg_ppqe.run(
    pool_type=pool_type,
    opt_thresh = r_g_opt_thresh,
    opt_maxiter = 40,
    noise_factor = noise_factor,
    time_step = dt,
    use_dt_from_l1_norm = True,
    optimizer = 'rotation',
    ppqe_trotter_order = 2,
    update_type = update_type,  
    )

timer.record("PPQE FCI")

print(timer)
print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')