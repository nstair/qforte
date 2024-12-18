import qforte as qf

# ====> Goal of Sandbox Test <==== #
# To test the pyscf integral interface

print('\nBuild Geometry')
print('-------------------------')

symm_str = 'c1'
fdocc = 0
fuocc = 0

basis_set = 'cc-pvdz'
avas_atoms_and_atomic_orbs = ['C 2pz']

# 8-ene geom
geom = [
('C', ( 0.335863334534,    -0.638024393629,    -0.000000000000)),
('H', ( 1.432938652990,    -0.620850836653,    -0.000000000000)),
('C', (-0.296842857597,    -1.839541058429,     0.000000000000)),
('H', (-1.393739535698,    -1.860704460450,     0.000000000000)),
('C', ( 0.383204712221,    -3.118962195977,     0.000000000000)),
('H', ( 1.479554414548,    -3.087255036285,     0.000000000000)),
('C', (-0.237579216697,    -4.313790116847,     0.000000000000)),
('H', ( 0.325161369217,    -5.249189992705,     0.000000000000)),
('H', (-1.329245260575,    -4.386987278270,     0.000000000000)),
('C', (-0.335863334534,     0.638024393629,     0.000000000000)),
('H', (-1.432938652990,     0.620850836653,     0.000000000000)),
('C', ( 0.296842857597,     1.839541058429,     0.000000000000)),
('H', ( 1.393739535698,     1.860704460450,     0.000000000000)),
('C', (-0.383204712221,     3.118962195977,     0.000000000000)),
('H', (-1.479554414548,     3.087255036285,     0.000000000000)),
('C', ( 0.237579216697,     4.313790116847,     0.000000000000)),
('H', ( 1.329245260575,     4.386987278270,     0.000000000000)),
('H', (-0.325161369217,     5.249189992705,     0.000000000000)),
        ]

# 12-ene geom
# C            0.318934151643     1.838664211757     0.000000000000
# H            1.415980959638     1.843665440633     0.000000000000
# C           -0.328238108423     0.638465784864     0.000000000000
# H           -1.425382504623     0.634328966959     0.000000000000
# C            0.328238108423    -0.638465784864     0.000000000000
# H            1.425382504623    -0.634328966959     0.000000000000
# C           -0.318934151643    -1.838664211757     0.000000000000
# H           -1.415980959638    -1.843665440633     0.000000000000
# C           -0.339993000522     3.116591194820     0.000000000000
# H           -1.437280937131     3.109148808007     0.000000000000
# C            0.302193183930     4.315542102589     0.000000000000
# H            1.399191891261     4.328262148643     0.000000000000
# C           -0.368438664361     5.598270030604     0.000000000000
# H           -1.465038377840     5.574093341283     0.000000000000
# C            0.259998191265     6.789887178283     0.000000000000
# H            1.352081696730     6.856305536198     0.000000000000
# C            0.339993000522    -3.116591194820     0.000000000000
# H            1.437280937131    -3.109148808007     0.000000000000
# C           -0.302193183930    -4.315542102589     0.000000000000
# H           -1.399191891261    -4.328262148643     0.000000000000
# C            0.368438664361    -5.598270030604     0.000000000000
# H            1.465038377840    -5.574093341283     0.000000000000
# C           -0.259998191265    -6.789887178283     0.000000000000
# H           -1.352081696730    -6.856305536198     0.000000000000
# H            0.297077214646    -7.728635113123     0.000000000000
# H           -0.297077214646     7.728635113123     0.000000000000


## ====> Build Qforte Mol with PySCF (using AVAS) <==== ###
mol = qf.system_factory(
    build_type='pyscf', 
    symmetry=symm_str,
    mol_geometry=geom, 
    basis=basis_set, 
    run_fci=True, 
    use_avas=True, #                     <=====
    avas_atoms_or_orbitals=avas_atoms_and_atomic_orbs,
    run_ccsd=False,
    store_mo_ints=True,
    build_df_ham=False,
    num_frozen_uocc = fuocc,
    num_frozen_docc = fdocc,
    build_qb_ham = False,
    )


print(f'The FCI energy from Pyscf:         {mol.fci_energy:12.10f}')
print(f'The SCF energy from Pyscf:         {mol.hf_energy:12.10f}')

# alg1 = qf.UCCNPQE(
#     mol,
#     computer_type = 'fci',
#     apply_ham_as_tensor=True,  #          <=====
#     verbose=False)

# alg1.run(
#     opt_thresh=1.0e-4, 
#     pool_type='SD',
#     )

alg2 = qf.QITE(
    mol, 
    reference=mol.hf_reference, 
    computer_type='fci', 
    verbose=0, 
    print_summary_file=0,
    apply_ham_as_tensor=True)

alg2.run(
    beta=10.0, 
    db=1.0,
    dt=0.001,                     # <===== Time evo for selection process
    use_exact_evolution=False,    # <===== Exact Evo
    do_lanczos=False,              # <===== Lanczos (broken at the moment?)
    sparseSb=False,
    expansion_type='SD',         # <===== Pool Type
    low_memorySb=False,
    second_order=True, 
    print_pool=False, 
    evolve_dfham=False, 
    random_state=False, 
    selected_pool=False,           # <===== Selcct Pool?
    physical_r=True,              # <===== Not sure about this? (realistic selection?)
    cumulative_t=True,
    t_thresh=1.0e-2)

print(f'The FCI energy from Pyscf:        {mol.fci_energy:12.10f}')
# print(f'The PQE energy with Pyscf ints:   {alg1._Egs:12.10f}')
print(f'The QITE energy with Pyscf ints:  {alg2._Egs:12.10f}')

