import numpy as np
import qforte as qf


geom = [('H', (0., 0., 0.)), 
        ('H', (0., 0., 1.00)),
        ('H', (0., 0., 2.00)), 
        ('H', (0., 0., 3.00)),
        ('H', (0., 0., 4.00)),
        ('H', (0., 0., 5.00)),
        ('H', (0., 0., 6.00)), 
        ('H', (0., 0., 7.00)), 
        # ('H', (0., 0., 8.00)), 
        # ('H', (0., 0., 9.00)), 
        ]


print('\nBuild Geometry')
print('-------------------------')

# symm_str = 'c1'
# fdocc = 0
# fuocc = 0

# basis_set = 'cc-pvdz'
# avas_atoms_and_atomic_orbs = ['C 2pz']

mol = qf.system_factory(
        build_type='psi4',
        symmetry='d2h',
        mol_geometry=geom, 
        basis='sto-6g', 
        run_fci=1,
        nroots_fci=1,
        run_ccsd=0,
        store_mo_ints=True,
        build_df_ham=True,
        df_icut=1.0e-0)

## ====> Build Qforte Mol with PySCF (using AVAS) <==== ###
# mol = qf.system_factory(
#     build_type='pyscf', 
#     symmetry=symm_str,
#     mol_geometry=geom, 
#     basis=basis_set, 
#     run_fci=True, 
#     use_avas=True, #                     <=====
#     avas_atoms_or_orbitals=avas_atoms_and_atomic_orbs,
#     run_ccsd=False,
#     store_mo_ints=True,
#     build_df_ham=True,        # Important, must compute here
#     num_frozen_uocc = fuocc,
#     num_frozen_docc = fdocc,
#     build_qb_ham = False,
#     )

print(f'The FCI energy from Pyscf:         {mol.fci_energy:12.10f}')
print(f'The SCF energy from Pyscf:         {mol.hf_energy:12.10f}')

alg = qf.QITE(
        mol, 
        reference=mol.hf_reference, 
        computer_type='fci', 
        verbose=0, 
        print_summary_file=0,
        apply_ham_as_tensor=True # leave as False, resource estimaton not implemented for tensor ops
        )

ty = qf.local_timer()
ty.reset()

alg.run(
        beta=1.5,                  # 
        db=0.1,                    #
        dt=0.01,                   #
        use_diis=0,                #
        max_diis_size=0,           #
        use_exact_evolution=0,     #
        expansion_type='All',      # must set to "All" for sqite
        evolve_dfham=0,            #
        random_state=0,            #
        sparseSb=0,                #
        low_memorySb=0,            # low memory Sb resource estimation may be broken
        second_order=1,            #
        selected_pool=1,           #
        t_thresh=1.0e-8,           # ===> Important
        cumulative_t=1,            #
        b_thresh=1.0e-4,           # fock computer only
        x_thresh=1.0e-10,          # filters non-contributing operators from fixed pool types 
        conv_thresh=1.0e-3,        # halts evolution once conv threshold reached
        physical_r=1,              #
        use_df_ham_selection=True, # ====> New important df ham for selection option <===
        folded_spectrum=0,         #
        BeH2_guess=0,              # remove option
        e_shift=None,              #
        update_e_shift=0,          #
        do_lanczos=0,              # broken
        lanczos_gap=2,             #
        realistic_lanczos=1,       #
        fname=None,                # 
        output_path=None,          # for data generation
        print_pool=0,              #
        use_cis_reference=0,       #
        target_root=0,             #
        cis_target_root=0,         #
        )

ty.record("QITE")
print(ty)

Ets = alg._Ets

print('\n')
print(f'The FCI target_root energy from pyscf:     {mol.fci_energy:12.10f}')
print(f'The target_root energy from qite:          {Ets:12.10f}')
print(f'Delta E                                    {np.abs(Ets-mol.fci_energy):12.10f}')

# ==> Building expansion pool <==
#    beta           E(beta)     N(params)           N(CNOT)          N(measure)
# -------------------------------------------------------------------------------
#    0.000       -2.112460699           0               120                   0

# S:
# 2  0  0  0  0  0  0  0  0  0  0  
# 0  2  0  0  0  0  0  0  0  0  0  
# 0  0  2  0  0  0  0  0  0  0  0  
# 0  0  0  2  0  0  0  0  0  0  0  
# 0  0  0  0  2  0  0  0  0  0  0  
# 0  0  0  0  0  2  0  0  0  0  0  
# 0  0  0  0  0  0  2  0  0  0  0  
# 0  0  0  0  0  0  0  2  0  0  0  
# 0  0  0  0  0  0  0  0  2  0  0  
# 0  0  0  0  0  0  0  0  0  2  0  
# 0  0  0  0  0  0  0  0  0  0  2  

# b:
# 0.274727  0.2161  0.210211  0.194078  0.10661  0.197856  0.0912456  0.0912456  0.197856  0.10661  1.3915e-10  
#    0.100       -2.127474602          11               360               58226

# S:
#           2   -0.00112065  0.000498277   0.000421414   0.000227061   0.000194349  0.000421414  0.000194349   0.000227061  0.000500652             0  
# -0.00112065             2  0.000500652   0.000421414   0.000227061   0.000194349  0.000421414  0.000194349   0.000227061  0.000498277  -1.39039e-11  
# 0.000498277   0.000500652            2   0.000463414   0.000249691   0.000213719  0.000463414  0.000213719   0.000249691  -0.00104178   -6.4593e-16  
# 0.000421414   0.000421414  0.000463414             2   0.000210672   0.000180321  -0.00118377  0.000180321   0.000210672  0.000463414  -1.26874e-13  
# 0.000227061   0.000227061  0.000249691   0.000210672             2   9.71585e-05  0.000210672  9.71585e-05   -0.00146125  0.000249691  -2.18263e-17  
# 0.000194349   0.000194349  0.000213719   0.000180321   9.71585e-05             2  0.000180321   -0.0014916   9.71585e-05  0.000213719  -2.74968e-13  
# 0.000421414   0.000421414  0.000463414   -0.00118377   0.000210672   0.000180321            2  0.000180321   0.000210672  0.000463414   -3.9443e-30  
# 0.000194349   0.000194349  0.000213719   0.000180321   9.71585e-05    -0.0014916  0.000180321            2   9.71585e-05  0.000213719             0  
# 0.000227061   0.000227061  0.000249691   0.000210672   -0.00146125   9.71585e-05  0.000210672  9.71585e-05             2  0.000249691  -2.18263e-17  
# 0.000500652   0.000498277  -0.00104178   0.000463414   0.000249691   0.000213719  0.000463414  0.000213719   0.000249691            2   5.49908e-16  
#           0  -1.39039e-11  -6.4593e-16  -1.26874e-13  -2.18263e-17  -2.74968e-13  -3.9443e-30            0  -2.18263e-17  5.49908e-16             2  

# b:
# 0.185294  0.166631  0.15449  0.17376  0.0980908  0.0756689  0.17376  0.0756689  0.0980908  0.255798  -0.00126247  
#    0.200       -2.138812407          22               600              116452


#         Process name            Time (s)             Percent
#        =============       =============       =============
# Total evolution time              0.0127              100.00

#           Total Time              0.0127              100.00