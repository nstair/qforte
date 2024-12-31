from qforte import Molecule, QITE, system_factory

# The FCI energy for H2 at 1.5 Angstrom in a sto-3g basis
# E_fci = -0.9981493534
# E_fock = -0.9108735544

print('\nBuild Psi4 Geometry')
print('-------------------------')

geom = [('H', (0., 0., 0.)), 
        ('H', (0., 0., 1.00)),
        ('H', (0., 0., 2.00)), 
        ('H', (0., 0., 3.00)),
        ('H', (0., 0., 4.00)),
        ('H', (0., 0., 5.00)),
        # ('H', (0., 0., 6.00)), 
        # ('H', (0., 0., 7.00)), 
        ]


# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
mol = system_factory(build_type='psi4', 
                     mol_geometry=geom, 
                     basis='sto-6g', 
                     run_fci=True, 
                     run_ccsd=False,
                     store_mo_ints=1,
                     build_df_ham=1,
                     df_icut=1.0e-1)

# mol.ccsd_energy = 0.0

print(f'The FCI energy from Psi4:                                    {mol.fci_energy:12.10f}')
print(f'The HF energy from Psi4:                                     {mol.hf_energy:12.10f}')

print('\nBegin QITE test for H8 2nd order')
print('-------------------------')

alg = QITE(mol, 
        reference=mol.hf_reference, 
        computer_type='fci', 
        verbose=0, 
        print_summary_file=0,
        apply_ham_as_tensor=True)

alg.run(beta=10.0, 
        db=0.50,
        dt=0.001,
        use_exact_evolution=False,   # <==== Nothing else matters if using exact evo
        use_diis=True,               # <==== Manin new option here
        max_diis_size=6,             # <==== Max number of previous iterations to use in DIIS
        sparseSb=0,
        expansion_type='SD', #All 
        low_memorySb=False,
        second_order=True, 
        print_pool=1, 
        evolve_dfham=False, 
        random_state=False, 
        selected_pool=False,
        physical_r=True,
        cumulative_t=True,
        t_thresh=1e-4)

Egs_FCI = alg.get_gs_energy()


print(f'The HF energy from Psi4:                                     {mol.hf_energy:12.10f}')
print(f'The FCI energy from Psi4:                                    {mol.fci_energy:12.10f}')

print(f'The FCI energy from FCI QITE:                                {Egs_FCI:12.10f}')
print(f'The FCI energy error:                                        {Egs_FCI - mol.fci_energy:12.10f}')


