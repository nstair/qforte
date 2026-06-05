import qforte as qf


# geom = [
#     ('H', (0., 0., 1.00)), 
#     ('H', (0., 0., 2.00)),
#     ('H', (0., 0., 3.00)),
#     ('H', (0., 0., 4.00))
#     ]

geom = [
    ('N', (0., 0., 1.00)), 
    ('N', (0., 0., 2.00)),
    ]

mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g',
    run_fci=1)


GEV_STABILIZATION_THRESH = 1.0e-10

s = 499
dt = 'lambda_inv'
# dt = 0.01

trotter_number = 1
trotter_order = 2

# alg_fock = qf.SRQK(
#     mol,
#     computer_type = 'fock'
#     )

# alg_fock.run(
#     s=s,
#     dt=dt,
#     gev_stabilization_thresh=GEV_STABILIZATION_THRESH
# )
# print(f'\n\n Efci:   {mol.fci_energy:+12.10f}')


alg_fci = qf.SRQK(
    mol,
    computer_type = 'fci',
    trotter_number = trotter_number,
    trotter_order = trotter_order,
    )

alg_fci.run(
    s=s,
    dt=dt,
    gev_stabilization_thresh=GEV_STABILIZATION_THRESH
    )

print(f'\n\n Efci:      {mol.fci_energy:+12.10f}')
print(f'\n\n |E-FCI|:   {abs(mol.fci_energy - alg_fci.get_ts_energy()):e}')

# I would like the SRQK per step diagonalzation printout modified. Currently it prints someting like:

#    k(S)            E(Npar)     N(params)           N(CNOT)          N(measure)
# -------------------------------------------------------------------------------
#  1.00e+00     -107.419532452           1                 0                2959
#  1.12e+06     -107.509980600           2                 0               11838
#  2.59e+10     -107.528734987           3                 0               26637

# In the updated version I would like additional (well formatted) columns for the following:

# dE => the change in energy from the previous step, in the column adjacent to the current total QK energy

# RR => the 'reduced rank' that was used for the GEVS at each step, in the column adjacent N(params)

# Ttot => the total evolution time used for the step, in the column adjacent to N(CNOT)

# and N(CNOT) to be populated for the computer_type = 'fci' case as well, curretly it just prints 0, 
# but the hermitian pairs sqop pool should be able to return the number of CNOTs that would be needed to implement the evolution operator for the step, 
# Care needs to be taken for accurate bookkeeping, noting the number of steps, trotter number, and trotter order, 
# the N(CNOT) for a base 1st order trotter step should only be calculated once 
# and all other estimates should be based on that base estimate, for example a 2nd order trotter step should have N(CNOT) = 2 * N(CNOT) of the base 1st order step, and a 3rd order trotter step should have N(CNOT) = 4 * N(CNOT) of the base 1st order step, etc.

# Lastly, printing of the top of the banner 

#    k(S)            E(Npar)     N(params)           N(CNOT)          N(measure)
# -------------------------------------------------------------------------------

# is hardcoded for each impl of compute qk mats, this is needles code duplication there should be a single function that prints 
# the per-iteration upper banner, and then each compute qk mats impl should call that function to print the banner, 
# and then print the per step data in the same format, this will ensure consistency across different compute qk mats impls and reduce code duplication.