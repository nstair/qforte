
import numpy as np
import qforte as qf

# Define the molecule
# geom = [('H', (0., 0., 0.00)), 
#         ('H', (0., 0., 1.00)),
#         ('H', (0., 0., 2.00)), 
#         ('H', (0., 0., 3.00)),
#         ('H', (0., 0., 4.00)),
#         ('H', (0., 0., 5.00)),
#         ('H', (0., 0., 6.00)),
#         ('H', (0., 0., 7.00)),
#         ]

geom = [('H', (0., 0., 0.)), 
        ('Be', (0., 0., 1.00)),
        ('H', (0., 0., 2.00)),       
        ]

# ====> Target root <======
num_roots = 9
target_root = 4

# sometimes cis target is not the same state as fci target!
# need to inspect determinental composition to find out!
cis_target_root = 2
# cis_target_root = target_root

# Build the molecule
mol = qf.system_factory(build_type='pyscf',
                        symmetry='d2h',
                        mol_geometry=geom, 
                        basis='sto-6g', 
                        run_fci=1,
                        nroots_fci=num_roots,
                        run_ccsd=0,
                        store_mo_ints=1,
                        build_df_ham=0,
                        df_icut=1.0e-1)

ref = mol.hf_reference
nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)
e_0 = mol.fci_energy_list[0]
target_energy = mol.fci_energy_list[target_root]


# ===> CIS Stuff <===
apply_ham_as_tensor = True

alg_fci = qf.CIS(
    mol,
    computer_type = 'fci',
    apply_ham_as_tensor=apply_ham_as_tensor,
    )

alg_fci.run(
    target_root=cis_target_root,
    diagonalize_each_step=False,
    low_memory=False
)

IJs, IJt, angles = alg_fci.get_cis_unitary_parameters()
cis_target_energy = alg_fci._Ets


# ====> Eshift <======
# shift = target_energy
shift = cis_target_energy
# shift = +0.0

# SQOp ham
Ofs = qf.SQOperator()
Ofs.add_op(mol.sq_hamiltonian)
Ofs.add_term(-1.0*shift, [], [])
Ofs.simplify()

H = qf.SQOperator()
H.add_op(mol.sq_hamiltonian)

# Run info
# print('\n')
# print(f'target root:         {target_root}')
# print(f'shift energy:       {shift:+10.10f}')
# print(f'root zero energy:   {e_0:+10.10f}')
# print(f'target root energy: {target_energy:+10.10f}')
# print(f'nuclear rep energy: {mol.nuclear_repulsion_energy:+10.10f}')
# print('\n')


# Instantiate FCI Comp and tensors
fc1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)

zero_body_energy = -1.0*shift
zero_body_energy += mol.nuclear_repulsion_energy
mo_oeis = mol.mo_oeis
mo_teis = mol.mo_teis 
mo_teis_einsum = mol.mo_teis_einsum


# ===========> FS METHOD BELOW <===========#


print('\n  Begin Folded Spectrum Operator Test')
print('-------------------------------------------')


fc1.hartree_fock()

# Set to CIS state
fc1.apply_two_determinant_rotations(
        IJs,
        IJt,
        angles,
        False
    )

rand = False
seed = 40
if(rand):
    # Create a new random number generator instance with a seed
    rng = np.random.default_rng(seed)
    # Use the generator to create random numbers
    shape = fc1.get_state().shape()
    random_array = rng.random(shape)
    random = np.array(random_array, dtype=np.complex128)

    # random = np.ones((fc1.get_state().shape()[0], fc1.get_state().shape()[1]))

    Crand = qf.Tensor(fc1.get_state().shape(), "Crand")
    Crand.fill_from_nparray(random.ravel(), Crand.shape())
    rand_nrm = Crand.norm()
    Crand.scale(1/rand_nrm)
    fc1.set_state(Crand)
    # print(fc1.str(print_data=True))


# Time step
# NOTE! IS NUMERICALLY UNSTABLE FOR TOO LARGE dBeta!
beta = 100.0
db = 0.1
nbeta = int(beta/db)

beta_sq = 0.0

update_shift = True

# Better convergence likely achieveable by updateing shift!
print(f'\n\n     B^2            <H>')
print('====================================:')
for kb in range(1, nbeta):

    beta_sq += db**2
    sq_exp4 = np.real(fc1.get_exp_val(mol.sq_hamiltonian))

    if(update_shift):
        zero_body_energy = mol.nuclear_repulsion_energy
        zero_body_energy += -1.0*sq_exp4

    fc1.evolve_tensor2_taylor(
        zero_body_energy, 
        mo_oeis, 
        mo_teis, 
        mo_teis_einsum, 
        norb,
        db**2,
        1.0e-15,
        30,
        True)
    

    norm1 = 1.0 / fc1.get_state().norm()
    fc1.scale(norm1)
    
    if(kb == 1 or kb%20==0):
        print(f'  {beta_sq:7.3f}    {sq_exp4:+15.9f}')

# Computer |R> = H|C> - Ek|C> = C_H - C_Ek
C_Ek = fc1.get_state_deep()
C_Ek.scale(sq_exp4)
fc1.apply_sqop(H)
C_H = fc1.get_state_deep()
C_H.subtract(C_Ek)
print(f"\n||R|| is close to zero for an eigenstate")
print(f"||R||: {C_H.norm():10.10f}\n")


print('\n')
print("     Folded Sepctrum Summary:")
print("======================================:")
print(f'target root:          {target_root}')
print(f'target root cis:      {cis_target_root}')
print(f'shift energy:         {shift:+10.10f}')
print(f'update shift in Ofs:  {update_shift}')
print(f'root zero energy:     {e_0:+10.10f}')
print(f'target root Ecis:     {cis_target_energy:+10.10f} <===')
print(f'Ek = <H>:             {sq_exp4:+10.10f} <===')
print(f'target root energy:   {target_energy:+10.10f} <===')
print(f'|Etarget - Ek|:       {np.abs(target_energy-sq_exp4):+10.10f} <===')
print('\n')

print('\n  FCI Eigenstate Energies')
print('======================================:')
for i, Ei in enumerate(mol.fci_energy_list):
    print(f"  i: {i}  Ei:       {Ei:+10.10f}")

