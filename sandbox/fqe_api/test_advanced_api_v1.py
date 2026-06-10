import qforte as qf
import numpy as np
import psi4
 
import time

import os
# os.environ["OMP_NUM_THREADS"] = "16"
# os.environ["OMP_DYNAMIC"] = "FALSE"
# os.environ["OMP_PROC_BIND"] = "true"
# os.environ["OMP_PLACES"] = "cores"

# print("PY sees OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"))
# print("PY sees OMP_DYNAMIC =", os.environ.get("OMP_DYNAMIC"))
# print("PY sees OMP_PROC_BIND =", os.environ.get("OMP_PROC_BIND"))
# print("PY sees OMP_PLACES =", os.environ.get("OMP_PLACES"))

# Define the reference and geometry lists.
geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    ('H', (0., 0., 5.0)), 
    ('H', (0., 0., 6.0)),
    ('H', (0., 0., 7.0)), 
    ('H', (0., 0., 8.0)),
    ('H', (0., 0., 9.0)), 
    ('H', (0., 0.,10.0)),
    ('H', (0., 0.,11.0)), 
    ('H', (0., 0.,12.0)),
    ('H', (0., 0.,13.0)),
    ('H', (0., 0.,14.0)),
    ('H', (0., 0.,15.0)),
    ('H', (0., 0.,16.0)),
    ]

# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
# mol = qf.system_factory(
#     build_type='psi4', 
#     mol_geometry=geom, 
#     basis='sto-3g', 
#     run_fci=1)

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

fci_comp1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fqe_comp1 = qf.FQEComputer(nel=nel, sz=sz, norb=norb)
fci_comp_gpu = qf.FCIComputerGPU(nel=nel, sz=sz, norb=norb, on_gpu=False, data_type="complex")

fci_comp1.hartree_fock()
fqe_comp1.hartree_fock()
fci_comp_gpu.hartree_fock_cpu()

sqham = mol.sq_hamiltonian

hermitian_pairs = qf.SQOpPool()
hermitian_pairs.add_hermitian_pairs(1.0, sqham)

num_threads = 8
print(f"\n num threads: {num_threads}\n")

time = 0.1
r = 1
order = 1

N = 1

ah = False
adj = True

print("\n Time Evo Settings")
print("===========================")
print(f" time:      {time}")
print(f" r:         {r}")
print(f" order:     {order}")
print(f" antiherm:  {ah}")
print(f" adjoint:   {adj}")
print("\n")

app_sqop = False
app_tens = True
app_exact_evo = False
app_trot = False

# Psi4 clamps num threads to 1 even for fqe precesses
# if this is not explicitally set here!
# import psi4
# psi4.set_num_threads(1)



# ===> apply sqop <====

if(app_sqop):
    timer.reset()
    fci_comp1.apply_sqop(sqham)
    timer.record("FCI apply sqop")

    timer.reset()
    fqe_comp1.apply_sqop(sqham)
    timer.record("FQE apply sqop")

    Cfci = fci_comp1.get_state_deep()
    print(f"\n |dC| apply sqop: {fqe_comp1.get_tensor_diff(Cfci)} \n")




# ===> apply tensor <====

if(app_tens):
    fci_comp1.hartree_fock()
    fqe_comp1.hartree_fock()

    mo_oeis_gpu = qf.TensorGPU(mol.mo_oeis.shape(), "mo_oeis_gpu", False)
    mo_teis_gpu = qf.TensorGPU(mol.mo_teis.shape(), "mo_teis_gpu", False)
    mo_teis_einsum_gpu = qf.TensorGPU(mol.mo_teis_einsum.shape(), "mo_teis_einsum_gpu", False)
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

    # timer.reset()
    # fci_comp1.apply_tensor_spat_012bdy(
    #     mol.nuclear_repulsion_energy, 
    #     mol.mo_oeis, 
    #     mol.mo_teis, 
    #     mol.mo_teis_einsum, 
    #     norb)
    # timer.record('FCI apply tensor')

    psi4.core.set_num_threads(num_threads)

    timer.reset()
    fqe_comp1.apply_tensor_spat_012bdy(
        mol.nuclear_repulsion_energy, 
        mol.mo_oeis_np, 
        mol.mo_teis_np)
    timer.record('FQE apply tensor')

    Cfqe = fqe_comp1.get_state_deep()
    Cfci = fci_comp1.get_state_deep()
    fci_comp_gpu.to_cpu()
    Cfci_gpu = qf.Tensor(Cfci.shape(), "Cfci_gpu")
    fci_comp_gpu.copy_to_tensor_cpu(Cfci_gpu)
    print(f"\n |dC| apply tensor: {fqe_comp1.get_tensor_diff(Cfci_gpu)} \n")

    # print(fci_comp1)
    # print(fqe_comp1)


# ===> evovle exactly <====

if(app_exact_evo):
    fci_comp1.hartree_fock()
    fqe_comp1.hartree_fock()

    timer.reset()

    fci_comp1.evolve_tensor_taylor(
        mol.nuclear_repulsion_energy, 
        mol.mo_oeis, 
        mol.mo_teis, 
        mol.mo_teis_einsum, 
        norb,
        time,
        1.0e-15,
        100,
        False)

    timer.record(f"FCI exact step")

    Cfci1 = fci_comp1.get_state_deep()

    timer.reset()

    fqe_comp1.evolve_tensor_taylor(
        mol.nuclear_repulsion_energy,
        mol.mo_oeis_np,
        mol.mo_teis_np,
        time,
        1.0e-15, # was 1.0e-15
        100,
        False)

    timer.record(f"FQE exact step")
    Cfqe1 = fqe_comp1.get_state_deep()
    print(f"\n |Cfci| evolve exact: {Cfci1.norm()} \n")
    print(f"\n |Cfqe| evolve exact: {np.linalg.norm(Cfqe1)} \n")
    print(f"\n |dC1| evolve exact: {fqe_comp1.get_tensor_diff(Cfci1)} \n")



# ===> evovle pool trotter <====

if(app_trot):
    psi4.core.set_num_threads(num_threads)

    # fci_comp1.hartree_fock()
    # fqe_comp1.hartree_fock()

    # timer.reset()

    # fci_comp1.evolve_pool_trotter_not_inplace(
    #     hermitian_pairs,
    #     time,
    #     r,
    #     order,
    #     antiherm=ah,
    #     adjoint=adj)

    # timer.record(f"FCI Trotter step")

    # Cfci1 = fci_comp1.get_state_deep()
    # print(f"\n |Cfci| evolve sqop: {Cfci1.norm()} \n")

    # fci_comp1.hartree_fock()
    # fqe_comp1.hartree_fock()

    # timer.reset()

    # fci_comp1.evolve_pool_trotter(
    #     hermitian_pairs,
    #     time,
    #     r,
    #     order,
    #     antiherm=ah,
    #     adjoint=adj)

    # timer.record(f"FCI Trotter step inplace")

    # timer.reset()

    # fqe_comp1.evolve_pool_trotter(
    #     hermitian_pairs,
    #     time,
    #     r,
    #     order,
    #     antiherm=ah,
    #     adjoint=adj)

    # timer.record(f"FQE Trotter step")

    # Cfci2 = fci_comp1.get_state_deep()
    # print(f"\n |Cfqe| evolve sqop: {Cfci2.norm()} \n")
    # print(f"\n |dC1| evolve sqop: {fqe_comp1.get_tensor_diff(Cfci1)} \n")
    # print(f"\n |dC2| evolve sqop: {fqe_comp1.get_tensor_diff(Cfci2)} \n")


    # ===> evovle pool basic <====

    ah = True
    adj = False

    print("\n Time Evo Settings")
    print("===========================")
    print(f" time:      {1.0}")
    print(f" r:         {1}")
    print(f" order:     {1}")
    print(f" antiherm:  {ah}")
    print(f" adjoint:   {adj}")
    print("\n")


    sd_pool = qf.SQOpPool()
    sd_pool.set_orb_spaces(ref)
    sd_pool.fill_pool("SD")
    # print(sd_pool)

    fci_comp1.hartree_fock()
    fqe_comp1.hartree_fock()


    timer.reset()

    fci_comp1.evolve_pool_trotter_basic(
        sd_pool,
        antiherm=ah,
        adjoint=adj)
    
    # fci_comp1.evolve_pool_trotter_basic_not_inplace(
    #     sd_pool,
    #     antiherm=ah,
    #     adjoint=adj)


    timer.record(f"FCI Trotter basic inplace")

    timer.reset()

    fqe_comp1.evolve_pool_trotter_basic(
        sd_pool,
        antiherm=ah,
        adjoint=adj)

    timer.record(f"FQE Trotter basic")

    Cfci = fci_comp1.get_state_deep()
    print(f"\n |dC| evolve sqop basic: {fqe_comp1.get_tensor_diff(Cfci)} \n")



print(f" N hp's:     {len(hermitian_pairs.terms())}")


print("\n\n")
print(timer)

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