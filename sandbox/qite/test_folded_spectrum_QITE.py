from pyscf import gto, scf, fci
from qforte import Molecule, QITE, system_factory

print('\nRun Pyscf Excited States Calc')
print('---------------------------------')

# Define the molecule
mol = gto.Mole()
mol.atom = '''
H 0 0 0
H 0 0 1.00
H 0 0 2.00
H 0 0 3.00
'''
mol.basis = 'sto-6g'
mol.spin = 0  # Singlet state
mol.build()

# Perform Hartree-Fock calculation
mf = scf.RHF(mol)
mf.kernel()

# Perform FCI calculation
cisolver = fci.FCI(mol, mf.mo_coeff)
e_ground, c_ground = cisolver.kernel()

# Compute excited states
nroots = 5  # Number of states (ground + 4 excited states)
e_states, c_states = cisolver.kernel(nroots=nroots)

# Print results
print('\n')
print("Energies of states (Hartree):")
for i, e in enumerate(e_states):
    print(f"State {i}: {e:.6f} Hartree")


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
                     mol_geometry=geom, 
                     basis='sto-6g', 
                     run_fci=True, 
                     run_ccsd=False,
                     store_mo_ints=1,
                     build_df_ham=1,
                     df_icut=1.0e-1)
# mol.ccsd_energy = 0.0

print(mol.point_group)
print('\n')
print(f'The FCI energy from Psi4:                                    {mol.fci_energy:12.10f}')
print(f'The HF energy from Psi4:                                     {mol.hf_energy:12.10f}')


alg = QITE(mol, 
        reference=mol.hf_reference, 
        computer_type='fci', 
        verbose=0, 
        print_summary_file=0,
        apply_ham_as_tensor=False)

alg.run(beta=5.0, 
        db=0.1,
        dt=0.001,
        sparseSb=0,
        expansion_type='SD', 
        low_memorySb=0,
        second_order=1, 
        print_pool=1, 
        evolve_dfham=0, 
        random_state=0, 
        selected_pool=0,
        physical_r=0,
        folded_spectrum=1,
        e_shift=e_states[0],
        cumulative_t=0,
        t_thresh=1e-3)

Egs_FCI = alg.get_gs_energy()

print(f'The FCI energy from PySCF:                                    {e_states[0]:12.10f}')
# print(f'The FCI energy from Fock QITE:                               {Egs_Fock:12.10f}')
print(f'The FCI energy from QITE:                                     {Egs_FCI:12.10f}')
print(f'The FCI energy error:                                         {Egs_FCI - e_states[0]:12.10f}')

# print(f'The FCI energy from FCI QITE:                                {Egs_FCI:12.10f}')