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
        # ('H', (0., 0., 4.00)),
        # ('H', (0., 0., 5.00)),
        # ('H', (0., 0., 3.00)), 
        ]
        # ('H', (0., 0., 3.50))]
# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
mol = system_factory(build_type='psi4',
                     symmetry='cs',
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

# print('\nBegin QITE test for H2 using old Fock computer')
# print('-------------------------')

# alg = QITE_FCI(mol, reference=mol.hf_reference, apply_ham_as_tensor=False, verbose=0)
# alg.run(beta=18.0, sparseSb=False)
# Egs_Fock = alg.get_gs_energy()

# alg = QITE(mol, reference=mol.hf_reference, computer_type='fci', verbose=0, print_summary_file=0)
# alg.run(beta=23.20, expansion_type='SD', use_exact_evolution=True)
# Egs_FCI_low_mem = alg.get_gs_energy()

# print('\nBegin QITE test for H4 with low memory Sb and 2nd order')
# print('-------------------------')

# alg = QITE(mol, reference=mol.hf_reference, computer_type='fci', verbose=0, print_summary_file=1)
# alg.run(beta=23.20, sparseSb=False, expansion_type='SD', low_memorySb=True, second_order=True)
# Egs_FCI_low_mem = alg.get_gs_energy()


print('\nBegin QITE test for H8 2nd order')
print('-------------------------')

alg = QITE(mol, 
        reference=mol.hf_reference, 
        computer_type='fci', 
        verbose=0, 
        print_summary_file=0,
        apply_ham_as_tensor=True)

alg.run(beta=50,
        db=0.05,
        dt=0.001,
        sparseSb=0,
        expansion_type='SD', 
        low_memorySb=0,
        second_order=1, 
        print_pool=1, 
        evolve_dfham=0, 
        random_state=0, 
        selected_pool=1,
        physical_r=1,
        folded_spectrum=1,
        e_shift=mol.fci_energy,
        cumulative_t=1,
        t_thresh=1e-3)

Egs_FCI = alg.get_gs_energy()

# print(f'The FCI energy for H2 at 1.5 Angstrom in a sto-3g basis:     {E_fci:12.10f}')
print(f'The HF energy from Psi4:                                     {mol.hf_energy:12.10f}')
print(f'The FCI energy from Psi4:                                    {mol.fci_energy:12.10f}')
# print(f'The FCI energy from Fock QITE:                               {Egs_Fock:12.10f}')
print(f'The FCI energy from FCI QITE:                                {Egs_FCI:12.10f}')
print(f'The FCI energy error:                                        {Egs_FCI - mol.fci_energy:12.10f}')

# print(f'The FCI energy from FCI QITE:                                {Egs_FCI:12.10f}')
