import qforte as qf
import numpy as np
import os

# sys_str = 'n2'
# sys_str = 'c2'
# sys_str = 'o2'

# sys_str = 'h8'
# sys_str = 'h2be'
sys_str = 'h2o'
# sys_str = 'c6h6'

tord = np.inf

# update_type = 'jacobi_like'
update_type = 'two_level_rotation'
# update_type = 'two_level_rotation_im'

if(sys_str == 'h8'):
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

elif(sys_str == 'h2be'):
    geom = [
        ('Be', (0., 0., 0.00)), 
        ('H', (0., 0., -1.00)),
        ('H', (0., 0., +1.00)),
        ]   

elif(sys_str == 'h2o'):
    geom = [
        ('O', (0., 0., 0.00)), 
        ('H', (0., 0., -1.00)),
        ('H', (0., 0., +1.00)),
        ]   

elif(sys_str == 'n2'):
    geom = [
        ('N', (0., 0., -1.00)), 
        ('N', (0., 0., +1.00)),
        ]  
    
elif(sys_str == 'c2'):
    geom = [
        ('C', (0., 0., -0.50)), 
        ('C', (0., 0., +0.50)),
        ]  
    
elif(sys_str == 'o2'):
    geom = [
        ('O', (0., 0., -1.00)), 
        ('O', (0., 0., +1.00)),
        ]  

elif(sys_str == 'c6h6'):
    geom = [
        ('C',   ( 0.000000,    1.396792,    0.000000)), 
        ('C',   ( 0.000000,   -1.396792,    0.000000)),
        ('C',   ( 1.209657,    0.698396,    0.000000)),
        ('C',   (-1.209657,   -0.698396,    0.000000)),
        ('C',   (-1.209657,    0.698396,    0.000000)),
        ('C',   ( 1.209657,   -0.698396,    0.000000)),
        ('H',   ( 0.000000,    2.484212,    0.000000)),
        ('H',   ( 2.151390,    1.242106,    0.000000)),
        ('H',   (-2.151390,   -1.242106,    0.000000)),
        ('H',   (-2.151390,    1.242106,    0.000000)),
        ('H',   ( 2.151390,   -1.242106,    0.000000)),
        ('H',   ( 0.000000,   -2.484212,    0.000000)),
                ]       


timer = qf.local_timer()

timer.reset()

if(sys_str == 'c6h6'):
    symm_str = 'c1'
    fdocc = 0
    fuocc = 0

    basis_set = 'cc-pvdz'
    avas_atoms_and_atomic_orbs = ['C 2pz']

    mol = qf.system_factory(
        build_type='pyscf', 
        symmetry=symm_str,
        mol_geometry=geom, 
        basis=basis_set, 
        run_fci=True, 
        use_avas=True, #                     <=====
        avas_atoms_or_orbitals=avas_atoms_and_atomic_orbs,
        run_ccsd=False,
        store_mo_ints=True,
        build_df_ham=False,
        num_frozen_uocc = fuocc,
        num_frozen_docc = fdocc,
        build_qb_ham = True,
        )
else:

    mol = qf.system_factory(
        build_type='psi4', 
        symmetry='d2h',
        mol_geometry=geom, 
        basis='sto-6g',
        run_fci=1)

timer.record("Setup")


# ===> Parameters <===
e_opt_thresh = 0.0e-8
r_g_opt_thresh = 1.0e-6
pool_type = 'SD'
noise_factor = 1.0e-6  # Set to zero for exact residuals
dt = 0.1

# use_dt_from_l1_norm = False 

max_diis = 12
opt_maxiter = 20



# ===> PPQE <===
timer.reset()

alg_ppqe = qf.UCCNPPQE(
    mol,
    computer_type = 'fci',
    diis_max_dim= max_diis,
    print_summary_file = True,
    )


alg_ppqe.run(
    pool_type=pool_type,
    opt_thresh = r_g_opt_thresh,
    opt_e_thresh = e_opt_thresh,
    opt_maxiter = opt_maxiter,
    noise_factor = noise_factor,
    time_step = dt,
    use_dt_from_l1_norm = False,
    optimizer = 'rotation',
    ppqe_trotter_order = tord,
    update_type = update_type,  
    )

print(timer)

print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')

#Efci:-230.7911477999
#    k iteration         Energy               dE           Nrvec ev      Nrm ev*         ||r||           Nshots
#--------------------------------------------------------------------------------------------------------------------
        #    1        -230.7379094343      -0.0160902917         1           117       +0.1090720688       inf
        #    2        -230.6827888501      +0.0551205842         2           234       +0.1084643746       inf
        #    3        -230.7871341871      -0.1043453369         3           351       +0.0286924748       inf
        #    4        -230.7894214395      -0.0022872524         4           468       +0.0151934458       inf
        #    5        -230.7901275405      -0.0007061011         5           585       +0.0063852566       inf
        #    6        -230.7902052538      -0.0000777132         6           702       +0.0029026828       inf
        #    7        -230.7902123970      -0.0000071432         7           819       +0.0013910988       inf
        #    8        -230.7902136104      -0.0000012134         8           936       +0.0010117774       inf
        #    9        -230.7902136933      -0.0000000828         9          1053       +0.0009792528       inf
        #   10        -230.7902137142      -0.0000000209        10          1170       +0.0009739425       inf
        #   11        -230.7902137268      -0.0000000127        11          1287       +0.0009728791       inf
        #   12        -230.7902137295      -0.0000000027        12          1404       +0.0009726093       inf