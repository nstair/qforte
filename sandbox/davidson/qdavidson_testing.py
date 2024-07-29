from qforte import Molecule, SRQD, system_factory

# The FCI energy for H2 at 1.5 Angstrom in a sto-3g basis
# E_fci = -0.9981493534
# E_fock = -0.9108735544

print('\nBuild Psi4 Geometry')
print('-------------------------')

geom = [('H', (0., 0., 0.)), 
        ('H', (0., 0., 1.00)),
        ('H', (0., 0., 2.00)), 
        ('H', (0., 0., 3.00))]
        # ('H', (0., 0., 4.00)), 
        # ('H', (0., 0., 5.00))]
        # ('H', (0., 0., 3.00)), 
        # ('H', (0., 0., 3.50))]
        # ('H', (0., 0., 4.00)), 
        # ('H', (0., 0., 4.50))]

# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
mol = system_factory(build_type='psi4', mol_geometry=geom, basis='sto-6g', run_fci=True, run_ccsd=False)
# mol.ccsd_energy = 0.0

print(f'The FCI energy from Psi4:                                    {mol.fci_energy:12.10f}')
print(f'The HF energy from Psi4:                                     {mol.hf_energy:12.10f}')


print('\nBegin QDavidson test for H4')
print('-------------------------')

alg = SRQD(mol, computer_type='fci', verbose=1)
alg.run(thresh=1.0e-9,
        dt=1.0e-4,
        max_itr=3)
# Egs_FCI_low_mem = alg.get_gs_energy()

print(f'The HF energy from Psi4:                                     {mol.hf_energy:12.10f}')
print(f'The FCI energy from Psi4:                                    {mol.fci_energy:12.10f}')
# print(f'The FCI energy from QLanczos:                                {Egs_FCI:12.10f}')