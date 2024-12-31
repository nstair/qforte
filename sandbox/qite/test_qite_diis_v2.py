import qforte as qf

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

print('\nBegin QITE test for H8 2nd order')
print('-------------------------')

alg = qf.QITE(mol, 
        reference=mol.hf_reference, 
        computer_type='fci', 
        verbose=0, 
        print_summary_file=0,
        apply_ham_as_tensor=True)

alg.run(beta=10.0, 
        db=0.10,
        dt=0.001,
        use_exact_evolution=True,   # <==== Nothing else matters if using exact evo
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


print(f'The HF energy from Pscf:                                     {mol.hf_energy:12.10f}')
print(f'The FCI energy from Pscf:                                    {mol.fci_energy:12.10f}')

print(f'The FCI energy from FCI QITE:                                {Egs_FCI:12.10f}')
print(f'The FCI energy error:                                        {Egs_FCI - mol.fci_energy:12.10f}')


