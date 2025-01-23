import numpy as np
from pyscf import gto, scf, fci
from qforte import Molecule, QITE, system_factory

print('\nRun BeH2 Excited States Calc')
print('---------------------------------')

# target root
root = 0

# Define the molecule
geom = [('Be', (0., 0., 0.)),
        ('H', (0., 0., 1.334)),
        ('H', (0., 0., -1.334))]

# Build the molecule
mol = system_factory(build_type='pyscf',
                     symmetry='D2h',
                     mol_geometry=geom, 
                     basis='sto-6g', 
                     run_fci=1,
                     nroots_fci=4,
                     run_ccsd=0,
                     store_mo_ints=1,
                     build_df_ham=0,
                     df_icut=1.0e-1)

alg = QITE(mol, 
        reference=mol.hf_reference, 
        computer_type='fci', 
        verbose=0, 
        print_summary_file=0,
        apply_ham_as_tensor=1)

alg.run(beta=50,
        db=0.10,
        dt=0.001,
        sparseSb=0,
        expansion_type='SD',
        low_memorySb=0,
        second_order=1, 
        print_pool=0, 
        evolve_dfham=0, 
        random_state=0, 
        selected_pool=0,
        physical_r=0,
        folded_spectrum=1,
        BeH2_guess=0,
        e_shift=mol.fci_energy_list[root],
        cumulative_t=0,
        t_thresh=1e-3)

Egs_FCI = alg.get_gs_energy()

print(f'The FCI energy from pyscf:                                    {mol.fci_energy_list[root]:12.10f}')
print(f'The FCI energy from QITE:                                     {Egs_FCI:12.10f}')

# print(f'The FCI energy from PySCF:                                    {e_states[0]:12.10f}')
# # print(f'The FCI energy from Fock QITE:                               {Egs_Fock:12.10f}')
# print(f'The FCI energy from QITE:                                     {Egs_FCI:12.10f}')
# print(f'The FCI energy error:                                         {Egs_FCI - e_states[0]:12.10f}')

# # print(f'The FCI energy from FCI QITE:                                {Egs_FCI:12.10f}')