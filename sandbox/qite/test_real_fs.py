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

num_roots = 8
target_root = 0

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
target_energy = -1.0*mol.fci_energy_list[target_root]

print(f'\ntarget root:   {target_root}')
print(f'target energy: {-target_energy:+15.9f}')

# SQOp ham
Ofs = qf.SQOperator()
Ofs.add_op(mol.sq_hamiltonian)
Ofs.add_term(target_energy, [], [])
Ofs.simplify()

# Tensor ham
print(f'nuclear : {mol.nuclear_repulsion_energy}')
zero_body_energy = target_energy
zero_body_energy += mol.nuclear_repulsion_energy
mo_oeis = mol.mo_oeis
mo_teis = mol.mo_teis 
mo_teis_einsum = mol.mo_teis_einsum

# Instantiate FCI Comp
SQOp_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
Tensor_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)

# Copy in excited state wfn
shape1 = np.shape(mol.fci_wfn[target_root])
T1 = qf.Tensor(shape=shape1, name='T1')
T1.fill_from_np(mol.fci_wfn[target_root].ravel(), shape1)

SQOp_comp.set_state(T1)
Tensor_comp.set_state(T1)

SQOp_exp = np.real(SQOp_comp.get_exp_val(mol.sq_hamiltonian))
Tensor_exp = np.real(Tensor_comp.get_exp_val_tensor(mol.nuclear_repulsion_energy,
                                                mol.mo_oeis, 
                                                mol.mo_teis, 
                                                mol.mo_teis_einsum, 
                                                norb))

print(SQOp_exp)
print(target_energy + SQOp_exp)
# print(Tensor_exp)

SQOp_exp2 = np.real(SQOp_comp.get_exp_val(Ofs))
Tensor_exp2 = np.real(Tensor_comp.get_exp_val_tensor(zero_body_energy,
                                                mol.mo_oeis, 
                                                mol.mo_teis, 
                                                mol.mo_teis_einsum, 
                                                norb))

print('\n')
print(SQOp_exp2)
# print(Tensor_exp2)

# SQOp_comp.hartree_fock()
Tensor_comp.hartree_fock()

n = 100

for i in range(n):

        SQOp_exp3 = np.real(SQOp_comp.get_exp_val(Ofs))
        print(f'power method energy: {SQOp_exp3}')

        SQOp_comp.apply_sqop(Ofs)

        norm1 = 1.0 / SQOp_comp.get_state().norm()
        SQOp_comp.scale(norm1)

        # norm2 = 1.0 / Tensor_comp.get_state().norm()
        # Tensor_comp.scale(norm2)

        # Tensor_exp = np.real(Tensor_comp.get_exp_val_tensor(mol.nuclear_repulsion_energy,
        #                                                 mol.mo_oeis, 
        #                                                 mol.mo_teis, 
        #                                                 mol.mo_teis_einsum, 
        #                                                 norb))

# SQOp_comp.apply_sqop(Ofs)
print(SQOp_comp.get_state().norm())
# 0.7236580925533873 FS
#  0.7237764246979698 HF

# # Time step
# beta = 0.5
# db = 0.22
# nbeta = int(beta/db)

# print('\nBegin Folded Spectrum Operator Test')
# print('#-----------------------------------------------------#')
# print(f"#{'kb':>6}{'E SQOp':>16}{'E Tensor':>21}          #")
# print('#-----------------------------------------------------#\n')
# print(f'  {0:7.3f}    {target_energy:+15.9f}    {target_energy:+15.9f}\n')

# for kb in range(1, nbeta):
#     SQOp_comp.evolve_op_taylor(Ofs,
#                                db**2,
#                                1.0e-15,
#                                30,
#                                1,
#                                1)

#     Tensor_comp.evolve_tensor_taylor(zero_body_energy, 
#                                      mo_oeis, 
#                                      mo_teis, 
#                                      mo_teis_einsum, 
#                                      norb,
#                                      db**2,
#                                      1.0e-15,
#                                      30,
#                                      1,
#                                      1)


#     norm1 = 1.0 / SQOp_comp.get_state().norm()
#     SQOp_comp.scale(norm1)

#     norm2 = 1.0 / Tensor_comp.get_state().norm()
#     Tensor_comp.scale(norm2)
    
#     SQOp_exp = np.real(SQOp_comp.get_exp_val(mol.sq_hamiltonian))
#     Tensor_exp = np.real(Tensor_comp.get_exp_val_tensor(zero_body_energy,
#                                                         mo_oeis, 
#                                                         mo_teis, 
#                                                         mo_teis_einsum, 
#                                                         norb))
    
#     print(f'  {kb:7.3f}    {SQOp_exp:+15.9f}    {Tensor_exp:+15.9f}\n')


