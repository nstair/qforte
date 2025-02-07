
# Test for the 

import numpy as np
import qforte as qf

# Define the molecule
geom = [('H', (0., 0., 0.)), 
        ('H', (0., 0., 1.00)),
        ('H', (0., 0., 2.00)), 
        ('H', (0., 0., 3.00)),
        # ('H', (0., 0., 4.00)),
        # ('H', (0., 0., 5.00)),
        # ('H', (0., 0., 3.00)), 
        ]

# ====> Target root <======
num_roots = 8
# target_root = 3 # works!
target_root = 5

# Build the molecule
mol = qf.system_factory(build_type='pyscf',
                        symmetry='C1',
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


# ====> Eshift <======
shift = target_energy
# shift = +0.9

# SQOp ham
Ofs = qf.SQOperator()
Ofs.add_op(mol.sq_hamiltonian)
Ofs.add_term(-1.0*shift, [], [])
Ofs.simplify()

H = qf.SQOperator()
H.add_op(mol.sq_hamiltonian)

# Run info
print('\n')
print(f'target root:         {target_root}')
print(f'shift energy:       {shift:+10.10f}')
print(f'root zero energy:   {e_0:+10.10f}')
print(f'target root energy: {target_energy:+10.10f}')
print(f'nuclear rep energy: {mol.nuclear_repulsion_energy:+10.10f}')
print('\n')


# Instantiate FCI Comp
fc1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fc2 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)

fc1.hartree_fock()
fc2.hartree_fock()

zero_body_energy = -1.0*shift
zero_body_energy += mol.nuclear_repulsion_energy
mo_oeis = mol.mo_oeis
mo_teis = mol.mo_teis 
mo_teis_einsum = mol.mo_teis_einsum


sq_exp = np.real(fc1.get_exp_val(mol.sq_hamiltonian))
tn_exp = np.real(fc2.get_exp_val_tensor(mol.nuclear_repulsion_energy,
                                                mol.mo_oeis, 
                                                mol.mo_teis, 
                                                mol.mo_teis_einsum, 
                                                norb))



sq_exp2 = np.real(fc1.get_exp_val(Ofs))
tn_exp2 = np.real(
        fc2.get_exp_val_tensor(
                zero_body_energy,
                mol.mo_oeis, 
                mol.mo_teis, 
                mol.mo_teis_einsum, 
                norb))

print('\n')
print(f"sq_exp2 {sq_exp2:10.10f}")
print(f"tn_exp2 {tn_exp2:10.10f}")
print('\n')

fc1.hartree_fock()
fc2.hartree_fock()

n = 100

for i in range(n):

        # sq_exp3 = np.real(fc1.get_exp_val(Ofs))
        sq_exp3 = np.real(fc1.get_exp_val(H))

        tn_exp3 = np.real(
        fc2.get_exp_val_tensor(
                mol.nuclear_repulsion_energy,
                mol.mo_oeis, 
                mol.mo_teis, 
                mol.mo_teis_einsum, 
                norb))



        if(i%4==0):
            print(f'i: {i:3} power method energy: {sq_exp3:+10.10f} tensor: {sq_exp3:+10.10f}')

        fc1.apply_sqop(Ofs)
        fc2.apply_tensor_spat_012bdy(
            zero_body_energy,
            mol.mo_oeis, 
            mol.mo_teis, 
            mol.mo_teis_einsum, 
            norb)

        norm1 = 1.0 / fc1.get_state().norm()
        fc1.scale(norm1)

        norm2 = 1.0 / fc2.get_state().norm()
        fc2.scale(norm2)

Ek = np.real(fc1.get_exp_val(H))
Lk = np.real(fc1.get_exp_val(Ofs))
print('\n')
print(f'Ek = <H>:           {Ek:+10.10f}')
print(f'Lk = <H-shift>:     {Lk:+10.10f}')
print('\n')

# Computer |R> = H|C> - Ek|C> = C_H - C_Ek
# print(f"\n||Cbef||: {fc1.get_state().norm()}\n")
C_Ek = fc1.get_state_deep()
C_Ek.scale(Ek)
# print(f"\n||Caff||: {fc1.get_state().norm()}\n")

fc1.apply_sqop(H)
C_H = fc1.get_state_deep()

C_H.subtract(C_Ek)
print(f"\n||R|| is close to zero for an eigenstate")
print(f"||R||: {C_H.norm():10.10f}\n")



# fc1.apply_sqop(Ofs)
# print(f"\n||C||: {fc1.get_state().norm()}\n")

# Get residual


print('\n')
print("     Power Method Summary:")
print("======================================:")
print(f'target root:        {target_root}')
print(f'shift energy:       {shift:+10.10f}')
print(f'root zero energy:   {e_0:+10.10f}')
print(f'Ek = <H>:           {Ek:+10.10f} <===')
print(f'target root energy: {target_energy:+10.10f} <===')
print(f'|Etarget - Ek|:     {np.abs(target_energy-Ek):+10.10f} <===')
print('\n')






print('\nBegin Folded Spectrum Operator Test')
print('#----------------------------------------#')
# print(f"#{'kb':>6}{'E SQOp':>16}{'E Tensor':>21}          #")
# print('#-----------------------------------------------------#\n')
# print(f'  {0:7.3f}    {target_energy:+15.9f}    {target_energy:+15.9f}\n')

fc1.hartree_fock()
fc2.hartree_fock()

rand = True
seed = 42
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
    fc2.set_state(Crand)
    # print(fc1.str(print_data=True))

beta_sq = 0.0

# Time step
beta = 500.0
db = 0.8
nbeta = int(beta/db)

for kb in range(1, nbeta):

    # note that apply_sqop() is called twice here.
    fc1.evolve_op2_taylor(
        Ofs,
        db**2,
        1.0e-15,
        30,
        True)

    fc2.evolve_tensor2_taylor(
        zero_body_energy, 
        mo_oeis, 
        mo_teis, 
        mo_teis_einsum, 
        norb,
        db**2,
        1.0e-15,
        30,
        True)
    
    beta_sq += db**2


    norm1 = 1.0 / fc1.get_state().norm()
    fc1.scale(norm1)

    norm2 = 1.0 / fc2.get_state().norm()
    fc2.scale(norm2)
    
    sq_exp4 = np.real(fc1.get_exp_val(mol.sq_hamiltonian))
    tn_exp4 = np.real(fc2.get_exp_val(mol.sq_hamiltonian))

    # tn_exp4 = np.real(fc2.get_exp_val_tensor(
    #     mol.nuclear_repulsion_energy,
    #     mo_oeis, 
    #     mo_teis, 
    #     mo_teis_einsum, 
    #     norb))
    
    # print(f'  {beta_sq:7.3f}    {sq_exp:+15.9f}    {tn_exp:+15.9f}')
    if(kb%100==0):
        print(f'  {beta_sq:7.3f}    {sq_exp4:+15.9f}  tensor: {tn_exp4:+15.9f}')

# Computer |R> = H|C> - Ek|C> = C_H - C_Ek
# print(f"\n||Cbef||: {fc1.get_state().norm()}\n")
C_Ek = fc1.get_state_deep()
C_Ek.scale(sq_exp4)
# print(f"\n||Caff||: {fc1.get_state().norm()}\n")

fc1.apply_sqop(H)
C_H = fc1.get_state_deep()

C_H.subtract(C_Ek)
print(f"\n||R|| is close to zero for an eigenstate")
print(f"||R||: {C_H.norm():10.10f}\n")

print('\n')
print("     Folded Sepctrum Summary:")
print("======================================:")
print(f'target root:        {target_root}')
print(f'shift energy:       {shift:+10.10f}')
print(f'root zero energy:   {e_0:+10.10f}')
print(f'Ek = <H>:           {sq_exp4:+10.10f} <===')
print(f'target root energy: {target_energy:+10.10f} <===')
print(f'|Etarget - Ek|:     {np.abs(target_energy-Ek):+10.10f} <===')
print('\n')

for i, Ei in enumerate(mol.fci_energy_list):
    print(f"  i: {i}  Ei:       {Ei:+10.10f}")

#     9.000       -1.619632790
#    18.000       -1.644496553
#    27.000       -1.648000836
#    36.000       -1.648627625
#    45.000       -1.648806257
#    54.000       -1.648912518

#     9.000       -1.530901668
#    18.000       -1.619623801
#    27.000       -1.640352596
#    36.000       -1.644086228
#    45.000       -1.644764028
#    54.000       -1.644932135