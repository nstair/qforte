import qforte as qf

geom = [
    ('H', (0., 0., 1.0)),
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    # ('H', (0., 0., 5.0)),
    # ('H', (0., 0., 6.0)),
    # ('H', (0., 0., 7.0)), 
    # ('H', (0., 0., 8.0)),
    # ('H', (0., 0., 9.0)), 
    # ('H', (0., 0.,10.0)),
    # ('H', (0., 0.,11.0)), 
    # ('H', (0., 0.,12.0))
    ]

# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-6g', 
    run_fci=True)

ref = mol.hf_reference

norb = int(len(ref) / 2)
nqbit = norb*2

print(f'nqbit:     {nqbit}')

# build pool
expansion_type = 'All'
full_pool = qf.SQOpPool()
full_pool.set_orb_spaces(ref)
full_pool.fill_pool(expansion_type)

# print('\n')
# print('FULL POOL')
# print('================')
# print('\n')
# print(full_pool)
# print('================')
# print('\n')

cnot_count = 0
for term in full_pool.terms():
    cnot = term[1].count_cnot_for_exponential()
    print(cnot)
    cnot_count += cnot

print(f'total cnot gate estimate: {cnot_count}')