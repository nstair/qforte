import qforte as qf

geom = [
    ('H', (0., 0., 1.0)),
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    ('H', (0., 0., 5.0)),
    ('H', (0., 0., 6.0)),
    ('H', (0., 0., 7.0)), 
    ('H', (0., 0., 8.0)),
    ('H', (0., 0., 9.0)), 
    ('H', (0., 0.,10.0)),
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
ham = mol.sq_hamiltonian

# print(ham)
# print(type(ham))

n_pauli_m_ham = ham.count_unique_pauli_products()

print(f'# pauli associated with <Φ|H|Φ>: {n_pauli_m_ham}')

# build pool
expansion_type = 'All'
pool = qf.SQOpPool()
pool.set_orb_spaces(ref)
pool.fill_pool(expansion_type)

# print(pool)

for i, term in enumerate(pool.terms()):
    n_pm_ham_pool = ham.count_unique_pauli_products(term[1])
    print(f'# pauli associated with <Φ|Hσ_{i}|Φ>: {n_pauli_m_ham}')

    if(i>=5):
        break

# for i, term1 in enumerate(pool.terms()):
#     for j, term2 in enumerate(pool.terms()):
#         print(term1[1])
#         print(term2[1])
#         n_pm_pool = term1[1].count_unique_pauli_products(term2[1])
#         print(f'# pauli associated with <Φ|σ_{i}σ_{j}|Φ>: {n_pm_pool}')


# We find that the hamiltonian and the pauli product of a pool term with the hamiltonian have the same cost