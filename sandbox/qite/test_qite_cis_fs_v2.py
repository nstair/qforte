import numpy as np
from pyscf import gto, scf, fci
from qforte import Molecule, QITE, system_factory

print('\nRun BeH2 Excited States Calc')
print('---------------------------------')

# target root
root = 1

# Define the molecule
geom = [('Be', (0., 0., 0.)),
        ('H', (0., 0., 1.334)),
        ('H', (0., 0., -1.334))
        ]

# geom = [('H', (0., 0., 0.)), 
#         ('H', (0., 0., 1.00)),
#         ('H', (0., 0., 2.00)), 
#         ('H', (0., 0., 3.00)),
#         ('H', (0., 0., 4.00)),
#         ('H', (0., 0., 5.00)),
        # ('H', (0., 0., 6.00)), 
        # ('H', (0., 0., 7.00)), 
        # ]


# Build the molecule
mol = system_factory(
        build_type='pyscf',
        symmetry='D2h',
        mol_geometry=geom, 
        basis='sto-6g', 
        run_fci=1,
        nroots_fci=6,
        run_ccsd=0,
        store_mo_ints=1,
        build_df_ham=0,
        df_icut=1.0e-1)

alg = QITE(
        mol, 
        reference=mol.hf_reference, 
        computer_type='fci', 
        verbose=0, 
        print_summary_file=0,
        apply_ham_as_tensor=1,
        )

alg.run(
        beta=1.0,
        db=0.1,
        dt=0.001,
        sparseSb=0,
        folded_spectrum=True,       ##
        target_root=root,
        cis_target_root=root,
        use_exact_evolution=False,  ##       
        use_cis_reference=True,
        e_shift=None, # can set to a number to override cis guess
        # e_shift=mol.fci_energy_list[root], 
        update_e_shift=True,
        expansion_type='SD',
        low_memorySb=0,
        second_order=1, 
        print_pool=False,
        selected_pool=False,
        physical_r=0,
        cumulative_t=0,
        t_thresh=1e-2,
        BeH2_guess=False, #Remove option
        )

# Egs = alg.get_gs_energy()
Ets = alg._Ets
Eroot = mol.fci_energy_list[root]

print('\n')
print(f'The FCI target_root energy from pyscf:     {Eroot:12.10f}')
print(f'The target_root energy from qite:          {Ets:12.10f}')
print(f'Delta E                                    {np.abs(Ets-Eroot):12.10f}')


