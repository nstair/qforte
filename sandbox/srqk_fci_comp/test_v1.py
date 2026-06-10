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
    ('H', (0., 0., 9.00)),
    ('H', (0., 0., 10.00)),
    ('H', (0., 0., 11.00)),
    ('H', (0., 0., 12.00)),
    ('H', (0., 0., 13.00)),
    ('H', (0., 0., 14.00)),
    ]

# geom = [
#     ('N', (0., 0., 1.00)), 
#     ('N', (0., 0., 2.00)),
#     ]

mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g',
    run_fci=0)


GEV_STABILIZATION_THRESH = 1.0e-10

# COMPUTER_TYPE = 'fci'
COMPUTER_TYPE = 'fci_gpu'
# COMPUTER_TYPE = 'fqe'
# COMPUTER_TYPE = 'cusv'
# COMPUTER_TYPE = 'fock'


if(COMPUTER_TYPE in ['fock', 'cusv']):
    APPLY_HAM_AS_TENSOR = False
else:
    APPLY_HAM_AS_TENSOR = True

s = 4
dt = 'lambda_inv'
# dt = 0.01

trotter_number = 1
trotter_order = 2

timer = qf.local_timer()

alg = qf.SRQK(
    mol,
    computer_type = COMPUTER_TYPE,
    apply_ham_as_tensor=APPLY_HAM_AS_TENSOR,
    trotter_number = trotter_number,
    trotter_order = trotter_order,
    )

timer.reset()

alg.run(
    s=s,
    dt=dt,
    gev_stabilization_thresh=GEV_STABILIZATION_THRESH
    )

timer.record('SRQK {COMPUTER_TYPE})}')

# print(f'\n\n Efci:      {mol.fci_energy:+12.10f}')
# print(f'\n\n |E-FCI|:   {abs(mol.fci_energy - alg.get_ts_energy()):e}')

print('\n\n')
print(timer)
