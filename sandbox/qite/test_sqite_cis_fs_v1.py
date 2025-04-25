import numpy as np
from pyscf import gto, scf, fci
# from qforte import Molecule, QITE, system_factory, FCIComputer
import qforte as qf

print('\nRun BeH2 Excited States Calc')
print('---------------------------------')

# target root
root = 1

# Define the molecule
# geom = [('Be', (0., 0., 0.)),
#         ('H', (0., 0., 1.334)),
#         ('H', (0., 0., -1.334))
#         ]

geom = [('H', (0., 0., 0.)), 
        ('H', (0., 0., 1.00)),
        ('H', (0., 0., 2.00)), 
        ('H', (0., 0., 3.00)),
        # ('H', (0., 0., 4.00)),
        # ('H', (0., 0., 5.00)),
        # ('H', (0., 0., 6.00)), 
        # ('H', (0., 0., 7.00)), 
        ]


# Build the molecule
mol = qf.system_factory(
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

alg = qf.QITE(
        mol, 
        reference=mol.hf_reference, 
        computer_type='fci', 
        verbose=0, 
        print_summary_file=0,
        apply_ham_as_tensor=1,
        )

alg.run(
        beta=0.5,
        db=0.5,
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


# just after a few iterations, get Uqite and find the residual based on that state, 
# then get the operator pool corresponding to |redidual> and |reference>

# make a new fc1 (for residual), and fc2 (for reference)
ref = qf.FCIComputer(
        alg._nel, 
        alg._2_spin, 
        alg._norb)

res = qf.FCIComputer(
        alg._nel, 
        alg._2_spin, 
        alg._norb)

ref.hartree_fock()
res.hartree_fock()

# Get cis info
IJs =  alg._cis_IJ_sources
IJt = alg._cis_IJ_targets
angles = alg._cis_angles

ref.apply_two_determinant_rotations(
        IJs,
        IJt,
        angles,
        False)


A = alg._sig

res.apply_two_determinant_rotations(
        IJs,
        IJt,
        angles,
        False)

res.evolve_pool_trotter_basic(
        A, 
        True, 
        False)

# res = alg._qc

res.apply_tensor_spat_012bdy(
        alg._nuclear_repulsion_energy, 
        alg._mo_oeis, 
        alg._mo_teis, 
        alg._mo_teis_einsum, 
        alg._norb)

res.evolve_pool_trotter_basic(
        A, 
        True, 
        True)

res.apply_two_determinant_rotations(
        IJs,
        IJt,
        angles,
        True)

print(res)
print(f"||R|| = {res.get_state().norm():10.10f}")
print(ref)

# looks the way we want it :)

new_pool = qf.SQOpPool()
new_pool.add_connection_pairs(res, ref, 1.0e-2)

print(new_pool)



