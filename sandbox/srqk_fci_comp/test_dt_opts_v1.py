import qforte as qf


geom = [
    ('H', (0., 0., 1.00)), 
    ('H', (0., 0., 2.00)),
    ('H', (0., 0., 3.00)),
    ('H', (0., 0., 4.00))
    ]

mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g',
    run_fci=1)


s = 4
dt = 0.2


alg_fci = qf.SRQK(
    mol,
    computer_type = 'fci'
    )


# New run options

# 

dt_grid_type = 'liner'

# liner => n * dt up to s

# quadratic? => formula


tmax_type = 'liner'

# linear => just s * dt (default), Tmax set by s and dt 

# egap => use alpha/egap to determine tmax, this and s determine dt, and trotter number at each step

# variance => use beta/variance to determine tmax, this and s determine dt, and trotter number at each step

# trotter => use trotter error to determine tmax, this and s determine dt, and trotter number at each step

# auto => use min (Tgap, Tvar, Ttrott) to determine tmax, this and s determine dt, and trotter number at each step



alg_fci.run(
    s=s,
    dt=dt
    dt_grid_type=dt_grid_type,
    tmax_type=tmax_type,
    tartet_trotter_error=1e-3,
    gev_stabilization_thres=1e-8,
    )

print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')
