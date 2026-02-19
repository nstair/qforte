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

fqe_comp = qf.FQEComputer(nel=nel, sz=sz, norb=norb)
fci_comp_gpu = qf.FCIComputerGPU(nel=nel, sz=sz, norb=norb, on_gpu=False, data_type="complex")

fqe_comp.hartree_fock()
fci_comp_gpu.hartree_fock_cpu()

sqham = mol.sq_hamiltonian

hermitian_pairs = qf.SQOpPool()
hermitian_pairs.add_hermitian_pairs(1.0, sqham)

# Test parameters - amplitude for a single operator
amplitude = 0.1

N = 1

ah = True  # antihermitian (for UCC operators)
adj = False  # not adjoint

print("\n Time Evo Settings")
print("===========================")
print(f" amplitude: {amplitude}")
print(f" antiherm:  {ah}")
print(f" adjoint:   {adj}")
print("\n")

app_sqop = False
app_sqop_evo = True  # Test what measure_gradient uses
app_pool_evo = False


# ===> apply sqop <====

if(app_sqop):
    timer.reset()
    fqe_comp.apply_sqop(sqham, antiherm=ah)
    timer.record("FQE apply sqop")

    fci_comp_gpu.to_gpu()
    timer.reset()
    fci_comp_gpu.apply_sqop_gpu(sqham)
    timer.record("GPU apply sqop")
    fci_comp_gpu.to_cpu()

    Cfqe = fqe_comp.get_state_deep()
    Cfci = qf.Tensor(Cfqe.shape(), "Cfci")
    fci_comp_gpu.copy_to_tensor_cpu(Cfci)
    print(f"\n |dC| apply sqop: {fqe_comp.get_tensor_diff(Cfci)} \n")


# ===> apply sqop evolution (single operator) - measure_gradient <====
if(app_sqop_evo):
    # Create a simple operator pool like in measure_gradient
    sd_pool_fqe = qf.SQOpPool()
    sd_pool_fqe.set_orb_spaces(ref)
    sd_pool_fqe.fill_pool("SD")
    
    sd_pool_gpu = qf.SQOpPoolGPU(data_type="complex")
    sd_pool_gpu.set_orb_spaces(ref)
    sd_pool_gpu.fill_pool("SD")
    
    # Test with first operator from pool (like measure_gradient does)
    test_op_fqe = sd_pool_fqe[0][1]
    test_op_gpu = sd_pool_gpu[0][1]
    
    print(f"Testing apply_sqop_evolution with operator: {test_op_fqe}")
    print(f"Amplitude: {amplitude}, antiherm: {ah}, adjoint: {adj}\n")
    
    fci_comp_gpu.to_gpu()
    
    timer.reset()
    fci_comp_gpu.apply_sqop_evolution_gpu(
        amplitude,
        test_op_gpu,
        antiherm=ah,
        adjoint=adj)
    timer.record('GPU apply_sqop_evolution')

    timer.reset()
    fqe_comp.apply_sqop_evolution(
        amplitude,
        test_op_fqe,
        antiherm=ah,
        adjoint=adj)
    timer.record('FQE apply_sqop_evolution')
    
    fci_comp_gpu.to_cpu()

    Cfqe = fqe_comp.get_state_deep()
    Cfci = qf.Tensor(Cfqe.shape, "Cfci")
    fci_comp_gpu.copy_to_tensor_cpu(Cfci)
    
    diff = fqe_comp.get_tensor_diff(Cfci)
    print(f"\n |dC| after single sqop_evolution: {diff}")
    print(f" State norm GPU: {Cfci.norm():.12f}")
    print(f" State norm FQE: {Cfqe.norm():.12f}\n")


# ===> evolve pool trotter (how VQE uses it) <====
if(app_pool_evo):
    # Create an operator pool with a single operator (like VQE does)
    sd_pool_fqe = qf.SQOpPool()
    sd_pool_fqe.set_orb_spaces(ref)
    sd_pool_fqe.fill_pool("SD")
    
    sd_pool_gpu = qf.SQOpPoolGPU(data_type="complex")
    sd_pool_gpu.set_orb_spaces(ref)
    sd_pool_gpu.fill_pool("SD")
    
    # Test with first few operators from the pool
    vqc_ops_fqe = qf.SQOpPool()
    vqc_ops_gpu = qf.SQOpPoolGPU(data_type="complex")
    
    num_ops_to_test = min(5, len(sd_pool_fqe))
    print(f"Testing with {num_ops_to_test} operators from pool")
    
    for i in range(num_ops_to_test):
        vqc_ops_fqe.add(amplitude, sd_pool_fqe[i][1])
        vqc_ops_gpu.add(amplitude, sd_pool_gpu[i][1])
    
    # Apply evolution as done in VQE
    fci_comp_gpu.to_gpu()
    
    timer.reset()
    fci_comp_gpu.evolve_pool_trotter_basic_gpu(
        vqc_ops_gpu,
        antiherm=ah,
        adjoint=adj)
    timer.record('GPU Pool Evolution')
    
    timer.reset()
    fqe_comp.evolve_pool_trotter_basic(
        vqc_ops_fqe,
        antiherm=ah,
        adjoint=adj)
    timer.record('FQE Pool Evolution')
    
    fci_comp_gpu.to_cpu()

    Cfqe = fqe_comp.get_state_deep()
    Cfci = qf.Tensor(Cfqe.shape, "Cfci")
    fci_comp_gpu.copy_to_tensor_cpu(Cfci)
    print(f"\n |dC| pool evo: {fqe_comp.get_tensor_diff(Cfci)} \n")


print("\n\n")
print(timer)