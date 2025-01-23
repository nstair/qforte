import qforte as qf

print('\nBuild Geometry')
print('-------------------------')

symm_str = 'c1'

geom = [('H', (0., 0., 1.00)),
        ('H', (0., 0., 2.00)), 
        ('H', (0., 0., 3.00)),
        ('H', (0., 0., 4.00)),
        # ('H', (0., 0., 5.00)),
        # ('H', (0., 0., 3.00)), 
        ]

### ====> Build Qforte Mol with Psi4 <==== ###
mol2 = qf.system_factory(
    build_type='pyscf', 
    symmetry=symm_str,
    mol_geometry=geom, 
    basis='sto-6g', 
    run_fci=True, 
    run_ccsd=False,
    store_mo_ints=True,
    build_df_ham=False,
    nroots_fci = 3
    # df_icut=1.0e-1
    )


print(f'The FCI energy from Pyscf:         {mol2.fci_energy:12.10f}')
print(f'The SCF energy from Pyscf:         {mol2.hf_energy:12.10f}')