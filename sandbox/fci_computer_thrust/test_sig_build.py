import qforte as qf
import numpy as np
 
import time

# Define the reference and geometry lists.
geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    # ('H', (0., 0., 5.0)), 
    # ('H', (0., 0., 6.0)),
    # ('H', (0., 0., 7.0)), 
    # ('H', (0., 0., 8.0)),
    # ('H', (0., 0., 9.0)), 
    # ('H', (0., 0.,10.0)),
    # ('H', (0., 0.,11.0)), 
    # ('H', (0., 0.,12.0)),
    # ('H', (0., 0.,13.0)), 
    # ('H', (0., 0.,14.0)),
    # ('H', (0., 0.,15.0)), 
    # ('H', (0., 0.,16.0))
    ]

mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g', 
    run_fci=0,
    build_qb_ham = False,
    store_mo_ints=True,
    store_mo_ints_np=True,
    build_df_ham=0,
    df_icut=1.0e-6
    )
 
print("\n Initial FCIcomp Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

print("\n")
print(f" nqbit:     {norb*2}")
print(f" nel:       {nel}")
print("\n")

timer = qf.local_timer()

fci_comp_cpu = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fci_comp_gpu = qf.FCIComputerGPU(nel=nel, sz=sz, norb=norb, on_gpu=False, data_type="complex")

fci_comp_cpu.hartree_fock()
fci_comp_gpu.hartree_fock_cpu()

sqham = mol.sq_hamiltonian

hermitian_pairs = qf.SQOpPool()
hermitian_pairs.add_hermitian_pairs(1.0, sqham)

time = 0.1
r = 3
order = 2

N = 4

ah = False
adj = True

# print("\n Time Evo Settings")
# print("===========================")
# print(f" time:      {time}")
# print(f" r:         {r}")
# print(f" order:     {order}")
# print(f" antiherm:  {ah}")
# print(f" adjoint:   {adj}")
# print("\n")

app_sqop = False
app_tens = True
app_exact_evo = False
app_trot = False


# ===> apply tensor <====

if(app_tens):
    fci_comp_cpu.hartree_fock()
    fci_comp_gpu.hartree_fock_cpu()

    timer.reset()
    fci_comp_cpu.apply_tensor_spat_012bdy(
        mol.nuclear_repulsion_energy, 
        mol.mo_oeis, 
        mol.mo_teis, 
        mol.mo_teis_einsum, 
        norb)
    timer.record('FCI apply tensor')

    # TODO: ask nick how GPU tensor conversion should be handled 
    # should we add parts to system factory to convert to GPU tensors or make user handle it?

    mo_oeis_gpu = qf.TensorGPU(mol.mo_oeis.shape(), "mo_oeis_gpu", False, "real")
    mo_teis_gpu = qf.TensorGPU(mol.mo_teis.shape(), "mo_teis_gpu", False, "real")
    mo_teis_einsum_gpu = qf.TensorGPU(mol.mo_teis_einsum.shape(), "mo_teis_einsum_gpu", False, "real")
    mo_oeis_gpu.fill_from_tensor_cpu(mol.mo_oeis, mol.mo_oeis.shape())
    mo_teis_gpu.fill_from_tensor_cpu(mol.mo_teis, mol.mo_teis.shape())
    mo_teis_einsum_gpu.fill_from_tensor_cpu(mol.mo_teis_einsum, mol.mo_teis_einsum.shape())

    mo_oeis_gpu.to_gpu()
    mo_teis_gpu.to_gpu()
    mo_teis_einsum_gpu.to_gpu()
    fci_comp_gpu.to_gpu()

    timer.reset()
    fci_comp_gpu.apply_tensor_spat_012bdy_gpu(
        mol.nuclear_repulsion_energy, 
        mo_oeis_gpu, 
        mo_teis_gpu, 
        mo_teis_einsum_gpu, 
        norb)
    timer.record('FCI GPU apply tensor')

    fci_comp_gpu.to_cpu()
    Cfci = fci_comp_cpu.get_state_deep()
    Cfci_gpu = qf.Tensor(Cfci.shape(), "Cfci_gpu")
    fci_comp_gpu.copy_to_tensor_cpu(Cfci_gpu)
    Cfci.subtract(Cfci_gpu)

    # fci_comp_gpu.to_cpu()
    # Cfci = fci_comp_cpu.get_state_deep()
    # Cfci_gpu = fci_comp_gpu.get_state_deep()
    # Cfci.subtract(Cfci_gpu)

    print(f"\n |dC| apply tensor: {(Cfci.norm())} \n")

    # print(fci_comp_cpu.get_state_deep())
    # print(Cfci_gpu)



    # print(fci_comp1)
    # print(fqe_comp1)


# print(f" N hp's:     {len(hermitian_pairs.terms())}")


print("\n\n")
print(timer)

fci_gpu_timer = fci_comp_gpu.get_acc_timer()
print(fci_gpu_timer.acc_str_table())

# fci_timer = fci_comp_cpu.get_acc_timer()
# print(fci_timer.acc_str_table())

# OMP_NUM_THREADS=4 OMP_PROC_BIND=TRUE OMP_DYNAMIC=FALSE \
# OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
# python sandbox/fqe_api/test_advanced_api_v1.py

#              Process name                 Time (s)                  Percent
#             =============            =============            =============
#            FCI apply sqop                   0.7209                     1.71
#            FQE apply sqop                   0.4135                     0.98
#          FCI apply tensor                   0.0947                     0.22
#          FQE apply tensor                   0.0553                     0.13
#            FCI exact step                   4.1651                     9.86
#            FQE exact step                   1.0557                     2.50
#          FCI Trotter step                   5.6837                    13.45
#  FCI Trotter step inplace                   4.3180                    10.22
#          FQE Trotter step                  24.5347                    58.07
# FCI Trotter basic inplace                   0.1675                     0.40
#         FQE Trotter basic                   1.0389                     2.46

#                Total Time                  42.2479                   100.00