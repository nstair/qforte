import qforte as qf

# ====> Goal of Sandbox Test <==== #
# To test the pyscf integral interface

print('\nBuild Geometry')
print('-------------------------')

symm_str = 'c1'
fdocc = 0
fuocc = 0

geom = [('H', (0., 0.,  1.00)),
        ('H', (0., 0.,  2.00)), 
        ('H', (0., 0.,  3.00)),
        ('H', (0., 0.,  4.00)),
        ('H', (0., 0.,  5.00)),
        ('H', (0., 0.,  6.00)),
        ('H', (0., 0.,  7.00)),
        ('H', (0., 0.,  8.00)),  
        # ('H', (0., 0.,  9.00)),
        # ('H', (0., 0., 10.00)),  
        ]

### ====> Build Qforte Mol with Psi4 <==== ###
# mol1 = qf.system_factory(
#     build_type='psi4', 
#     symmetry=symm_str,
#     mol_geometry=geom, 
#     basis='sto-6g', 
#     run_fci=True, 
#     run_ccsd=False,
#     store_mo_ints=True,
#     build_df_ham=False,
#     num_frozen_uocc = fuocc, # must be..
#     num_frozen_docc = fdocc,
#     )


# print(f'The FCI energy from Psi4:         {mol1.fci_energy:12.10f}')
# print(f'The SCF energy from Psi4:         {mol1.hf_energy:12.10f}')


# # Confirm HF works
# alg1 = qf.UCCNPQE(
#     mol1,
#     computer_type = 'fci',
#     apply_ham_as_tensor=True,
#     verbose=False)


# alg1.run(
#     opt_thresh=1.0e-4, 
#     pool_type='SD',
#     )



## ====> Build Qforte Mol with PySCF (using AVAS) <==== ###
mol2 = qf.system_factory(
    build_type='pyscf', 
    symmetry=symm_str,
    mol_geometry=geom, 
    basis='cc-pvdz', 
    run_fci=True, 
    use_avas=True, #                     <=====
    avas_atoms_or_orbitals=['H 1s'],
    run_ccsd=False,
    store_mo_ints=True,
    build_df_ham=False,
    num_frozen_uocc = fuocc,
    num_frozen_docc = fdocc,
    build_qb_ham = False,
    )


print(f'The FCI energy from Pyscf:         {mol2.fci_energy:12.10f}')
print(f'The SCF energy from Pyscf:         {mol2.hf_energy:12.10f}')

alg2 = qf.UCCNPQE(
    mol2,
    computer_type = 'fci',
    apply_ham_as_tensor=True, #          <=====
    verbose=False)


alg2.run(
    opt_thresh=1.0e-4, 
    pool_type='SD',
    )


# print(f'The SCF energy from Psi4:         {mol1.hf_energy:12.10f}')
# print(f'The SCF energy from Pyscf:        {mol2.hf_energy:12.10f}')
# print('')

# print(f'The FCI energy from Psi4:         {mol1.fci_energy:12.10f}')
# print(f'The PQE energy with Psi4 ints:    {alg1._Egs:12.10f}')
# print('')

print(f'The FCI energy from Pyscf:        {mol2.fci_energy:12.10f}')
print(f'The PQE energy with Pyscf ints:   {alg2._Egs:12.10f}')

