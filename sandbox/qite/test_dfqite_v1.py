# from qforte import Molecule, QITE, system_factory

import qforte as qf

print('\nBuild Psi4 Geometry')
print('-------------------------')

geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    ('H', (0., 0., 5.0)), 
    ('H', (0., 0., 6.0)),
#     ('H', (0., 0., 7.0)), 
#     ('H', (0., 0., 8.0)),
    # ('H', (0., 0., 9.0)), 
    # ('H', (0., 0.,10.0)),
    # ('H', (0., 0.,11.0)), 
    # ('H', (0., 0.,12.0))
    ]



mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-6g', 
    run_fci=True, 
    run_ccsd=False,
    symmetry='d2h',
    build_df_ham=True,
    df_icut=1.0e-5)

alg = qf.DFQITE(
    mol, 
    computer_type='fci', 
    verbose=0)

alg.run(
    beta=3.0, 
    db = 0.1, 
    sparseSb=False, 
    expansion_type='1-UpCCGSD', 
    low_memorySb=False, 
    second_order=True)

Egs = alg.get_gs_energy()

print(f'The HF energy from Psi4:                    {mol.hf_energy:12.10f}')
print(f'The FCI energy from Psi4:                   {mol.fci_energy:12.10f}')
print(f'The FCI energy from FCI QITE:               {Egs:12.10f}')
print(f'dE:                                          {Egs-mol.fci_energy:12.10f}')
print(f'dE:                                          {Egs-mol.fci_energy:e}')
