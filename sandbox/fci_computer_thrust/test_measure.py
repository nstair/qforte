import numpy as np
import qforte as qf
from qforte import UCCNVQE  # or import from where class is defined
import copy

np.random.seed(1)

def get_tensor_diff_gpu_cpu(fci_comp_gpu: qf.FCIComputerGPU, fci_comp_cpu: qf.FCIComputer):
    """
    Compute the difference between FQE and FCI GPU computer state tensors.
    """
    switched = False
    if fci_comp_gpu.on_gpu():
        switched = True
        fci_comp_gpu.to_cpu()
    
    Cfci_cpu = fci_comp_cpu.get_state_deep()
    Cfci_gpu = qf.Tensor(Cfci_cpu.shape(), "Cfci")
    fci_comp_gpu.copy_to_tensor_cpu(Cfci_gpu)
    Cfci_cpu.subtract(Cfci_gpu)

    if switched:
        fci_comp_gpu.to_gpu()
    return Cfci_cpu.norm()

# 1) build a tiny molecule (use your system_factory pattern)
r = 1.0
geom = [
    ('H', (0., 0., 0.0*r)),
    ('H', (0., 0., 1.0*r)),
    ('H', (0., 0., 2.0*r)),
    ('H', (0., 0., 3.0*r)),
]
mol = qf.system_factory(
    build_type='psi4',
    mol_geometry=geom,
    symmetry='d2h',
    basis='sto-3g',
    build_qb_ham=True,
    store_mo_ints=True,
    store_mo_ints_np=True,
    run_fci=0
)

alg_gpu = qf.UCCNVQE(mol, apply_ham_as_tensor=True, computer_type='fci_gpu', verbose=False)
alg_cpu = qf.UCCNVQE(mol, apply_ham_as_tensor=True, computer_type='fci', verbose=False)

# 3) prepare minimal internal state (pool + ansatz). This *does not* run the optimizer.
for alg in (alg_gpu, alg_cpu):
    alg._fast = True                # required by measurement routines
    alg._ref_from_hf = True         # many measure_* require HF reference
    alg._pool_type = 'SD'           # set pool type
    alg.fill_pool()                 # populate _pool_obj
    alg.initialize_ansatz()         # sets _tops and _tamps
    # Give a small nonzero test vector so evolutions do non-trivial work
    alg._tamps = [0.01 * (i + 1) for i in range(len(alg._tamps))]

M = len(alg_cpu._tamps)
grads_cpu = np.zeros(M)
grads_gpu = np.zeros(M)

vqc_ops_cpu = qf.SQOpPool()
vqc_ops_gpu = qf.SQOpPoolGPU(data_type=alg_gpu.data_type)

for tamp, top in zip(alg_cpu._tamps, alg_cpu._tops):
    vqc_ops_cpu.add(tamp, alg_cpu._pool_obj[top][1])

for tamp, top in zip(alg_gpu._tamps, alg_gpu._tops):
    vqc_ops_gpu.add(tamp, alg_gpu._pool_obj[top][1])

qc_psi_cpu = qf.FCIComputer(
            alg_cpu._nel, 
            alg_cpu._2_spin, 
            alg_cpu._norb) 

qc_psi_gpu = qf.FCIComputerGPU(
            alg_gpu._nel, 
            alg_gpu._2_spin, 
            alg_gpu._norb,
            on_gpu=False,
            data_type=alg_gpu.data_type)

qc_psi_cpu.hartree_fock()
qc_psi_gpu.hartree_fock_cpu()

print("psi tensor diff after HF:", get_tensor_diff_gpu_cpu(qc_psi_gpu, qc_psi_cpu))

qc_psi_gpu.to_gpu()

qc_psi_cpu.evolve_pool_trotter_basic(
            vqc_ops_cpu,
            antiherm=True,
            adjoint=False)

qc_psi_gpu.evolve_pool_trotter_basic_gpu(
            vqc_ops_gpu,
            antiherm=True,
            adjoint=False)

print("psi tensor diff after evo:", get_tensor_diff_gpu_cpu(qc_psi_gpu, qc_psi_cpu))

# build | psi_N > according ADAPT-VQE analytical grad section
qc_sig_cpu = qf.FCIComputer(
            alg_cpu._nel, 
            alg_cpu._2_spin, 
            alg_cpu._norb) 

qc_sig_gpu = qf.FCIComputerGPU(
            alg_gpu._nel, 
            alg_gpu._2_spin, 
            alg_gpu._norb,
            on_gpu=True,
            data_type=alg_gpu.data_type) 

psi_i_cpu = qc_psi_cpu.get_state_deep()

# Get GPU state - create TensorGPU if needed
if not hasattr(alg_gpu, 'psi_i') or alg_gpu.psi_i is None:
    alg_gpu.psi_i = qf.TensorGPU(qc_psi_cpu.get_state().shape(), "psi_i", data_type=alg_gpu.data_type)

qc_psi_gpu.copy_state_into(alg_gpu.psi_i)

qc_sig_cpu.set_state(psi_i_cpu)
qc_sig_gpu.set_state_gpu(alg_gpu.psi_i)

print("sig tensor diff after set_state:", get_tensor_diff_gpu_cpu(qc_sig_gpu, qc_sig_cpu))

if(alg_cpu._apply_ham_as_tensor):
    print("Applying ham as tensor CPU")
    qc_sig_cpu.apply_tensor_spat_012bdy(
        alg_cpu._zero_body_energy, 
        alg_cpu._mo_oeis, 
        alg_cpu._mo_teis, 
        alg_cpu._mo_teis_einsum, 
        alg_cpu._norb)
else:
    qc_sig_cpu.apply_sqop(alg_cpu._sq_ham)

if(alg_gpu._apply_ham_as_tensor):
    print("Applying ham as tensor GPU")
    qc_sig_gpu.apply_tensor_spat_012bdy_gpu(
        alg_gpu._zero_body_energy,
        alg_gpu._mo_oeis_gpu,
        alg_gpu._mo_teis_gpu,
        alg_gpu._mo_teis_einsum_gpu,
        alg_gpu._norb)
else:
    qc_sig_gpu.apply_sqop_gpu(alg_gpu._sq_ham)

print("oei on_gpu:", alg_gpu._mo_oeis_gpu.on_gpu(), "dtype:", alg_gpu._mo_oeis_gpu.data_type())
print("tei on_gpu:", alg_gpu._mo_teis_gpu.on_gpu(), "dtype:", alg_gpu._mo_teis_gpu.data_type())
print("tei_einsum on_gpu:", alg_gpu._mo_teis_einsum_gpu.on_gpu(), "dtype:", alg_gpu._mo_teis_einsum_gpu.data_type())

oei_back = qf.Tensor(mol.mo_oeis.shape(), "oei_back")
alg_gpu._mo_oeis_gpu.to_cpu()
alg_gpu._mo_oeis_gpu.copy_to_tensor(oei_back)
oei_back.subtract(mol.mo_oeis)
print("||oei_cpu - oei_gpu||:", oei_back.norm())

tei_back = qf.Tensor(mol.mo_teis.shape(), "tei_back")
alg_gpu._mo_teis_gpu.to_cpu()
alg_gpu._mo_teis_gpu.copy_to_tensor(tei_back)
tei_back.subtract(mol.mo_teis)
print("||tei_cpu - tei_gpu||:", tei_back.norm())

tei_einsum_back = qf.Tensor(mol.mo_teis_einsum.shape(), "tei_einsum_back")
alg_gpu._mo_teis_einsum_gpu.to_cpu()
alg_gpu._mo_teis_einsum_gpu.copy_to_tensor(tei_einsum_back)
tei_einsum_back.subtract(mol.mo_teis_einsum)
print("||tei_einsum_cpu - tei_einsum_gpu||:", tei_einsum_back.norm())

print("sig tensor diff after apply ham:", get_tensor_diff_gpu_cpu(qc_sig_gpu, qc_sig_cpu))

mu = M-1

Kmu_prev_cpu = alg_cpu._pool_obj[alg_cpu._tops[mu]][1]
Kmu_prev_gpu = alg_gpu._pool_obj[alg_gpu._tops[mu]][1]

Kmu_prev_cpu.mult_coeffs(alg_cpu._pool_obj[alg_cpu._tops[mu]][0])
Kmu_prev_gpu.mult_coeffs(alg_gpu._pool_obj[alg_gpu._tops[mu]][0])

print ("Kmu_prev coeffs: {}, {}".format(
    Kmu_prev_cpu,
    Kmu_prev_gpu
    ))

# Apply the operator to psi for both CPU and GPU
qc_psi_cpu.apply_sqop(Kmu_prev_cpu)
qc_psi_gpu.apply_sqop_gpu(Kmu_prev_gpu)

print("psi tensor diff after apply Kmu_prev:", get_tensor_diff_gpu_cpu(qc_psi_gpu, qc_psi_cpu))

# Compute the gradient for mu=M-1
grads_cpu[mu] = 2.0 * np.real(
    qc_sig_cpu.get_state().vector_dot(qc_psi_cpu.get_state())
)

grads_gpu[mu] = 2.0 * np.real(
    qc_sig_gpu.state_vector_dot_gpu(qc_psi_gpu)
)

print(f"Gradient at mu={mu}:")
print(f"  CPU: {grads_cpu[mu]:.16f}")
print(f"  GPU: {grads_gpu[mu]:.16f}")
print(f"  Diff: {abs(grads_cpu[mu] - grads_gpu[mu]):.16e}")

# Reset psi to psi_i for both
qc_psi_cpu.set_state(psi_i_cpu)
qc_psi_gpu.set_state_gpu(alg_gpu.psi_i)

print("psi tensor diff after reset:", get_tensor_diff_gpu_cpu(qc_psi_gpu, qc_psi_cpu))

# Continue with the rest of the gradient calculation loop
for mu in reversed(range(M-1)):
    tamp_cpu = alg_cpu._tamps[mu+1]
    tamp_gpu = alg_gpu._tamps[mu+1]
    
    Kmu_cpu = alg_cpu._pool_obj[alg_cpu._tops[mu]][1]
    Kmu_gpu = alg_gpu._pool_obj[alg_gpu._tops[mu]][1]
    
    Kmu_cpu.mult_coeffs(alg_cpu._pool_obj[alg_cpu._tops[mu]][0])
    Kmu_gpu.mult_coeffs(alg_gpu._pool_obj[alg_gpu._tops[mu]][0])
    
    # Apply evolution with -tamp * Kmu_prev
    qc_psi_cpu.apply_sqop_evolution(
        -1.0*tamp_cpu,
        Kmu_prev_cpu,
        antiherm=True,
        adjoint=False)
    
    qc_psi_gpu.apply_sqop_evolution_gpu(
        -1.0*tamp_gpu,
        Kmu_prev_gpu,
        antiherm=True,
        adjoint=False)
    
    qc_sig_cpu.apply_sqop_evolution(
        -1.0*tamp_cpu,
        Kmu_prev_cpu,
        antiherm=True,
        adjoint=False)
    
    qc_sig_gpu.apply_sqop_evolution_gpu(
        -1.0*tamp_gpu,
        Kmu_prev_gpu,
        antiherm=True,
        adjoint=False)
    
    print(f"Tensor diff after evolution (mu={mu}):", get_tensor_diff_gpu_cpu(qc_psi_gpu, qc_psi_cpu))
    print(f"Sig tensor diff after evolution (mu={mu}):", get_tensor_diff_gpu_cpu(qc_sig_gpu, qc_sig_cpu))
    
    # Save current psi state
    psi_i_cpu = qc_psi_cpu.get_state_deep()
    qc_psi_gpu.copy_state_into(alg_gpu.psi_i)
    
    # Apply Kmu
    qc_psi_cpu.apply_sqop(Kmu_cpu)
    qc_psi_gpu.apply_sqop_gpu(Kmu_gpu)
    
    # Compute gradient
    grads_cpu[mu] = 2.0 * np.real(
        qc_sig_cpu.get_state().vector_dot(qc_psi_cpu.get_state())
    )
    
    grads_gpu[mu] = 2.0 * np.real(
        qc_sig_gpu.state_vector_dot_gpu(qc_psi_gpu)
    )
    
    print(f"Gradient at mu={mu}:")
    print(f"  CPU: {grads_cpu[mu]:.16f}")
    print(f"  GPU: {grads_gpu[mu]:.16f}")
    print(f"  Diff: {abs(grads_cpu[mu] - grads_gpu[mu]):.16e}")
    
    # Reset psi
    qc_psi_cpu.set_state(psi_i_cpu)
    qc_psi_gpu.set_state_gpu(alg_gpu.psi_i)
    
    # Update Kmu_prev for next iteration
    Kmu_prev_cpu = Kmu_cpu
    Kmu_prev_gpu = Kmu_gpu

# Final comparison
print("\n" + "="*60)
print("FINAL GRADIENT COMPARISON")
print("="*60)
print("CPU Gradients:", grads_cpu)
print("GPU Gradients:", grads_gpu)
print("Absolute Differences:", np.abs(grads_cpu - grads_gpu))
print("Max Absolute Difference:", np.max(np.abs(grads_cpu - grads_gpu)))

# Assert they match within tolerance
np.testing.assert_allclose(grads_cpu, grads_gpu, rtol=1e-10, atol=1e-10)
print("\n✓ CPU and GPU gradients match within tolerance!")

#  # TODO(Nick): think about optemization here, probably should have its own c++ function
#     def measure_gradient_fci(self, params=None):
#         """ Returns the disentangled (factorized) UCC gradient, using a
#         recursive approach.

#         Parameters
#         ----------
#         params : list of floats
#             The variational parameters which characterize _Uvqc.
#         """

#         if not self._fast:
#             raise ValueError("self._fast must be True for gradient measurement.")
        
#         if(self._pool_type == 'sa_SD'):
#             raise ValueError('Must use single term particle-hole nbody operators for residual calculation')
        
#         if not self._ref_from_hf:
#             raise ValueError('get_residual_vector_fci_comp only compatible with hf reference at this time.')

#         M = len(self._tamps)
#         grads = np.zeros(M)
#         vqc_ops = qforte.SQOpPool()

#         if params is None:
#             for tamp, top in zip(self._tamps, self._tops):
#                 vqc_ops.add(tamp, self._pool_obj[top][1])
#         else:
#             for tamp, top in zip(params, self._tops):
#                 vqc_ops.add(tamp, self._pool_obj[top][1])

#         # build | sig_N > according ADAPT-VQE analytical grad section
#         qc_psi = qforte.FCIComputer(
#             self._nel, 
#             self._2_spin, 
#             self._norb) 
        
#         t0 = self._fci_time.time()
#         qc_psi.hartree_fock()
#         self._fci_timers['hartree_fock'] += self._fci_time.time() - t0
        
#         # qc_psi.apply_circuit(Utot)
#         t0 = self._fci_time.time()
#         qc_psi.evolve_pool_trotter_basic(
#             vqc_ops,
#             antiherm=True,
#             adjoint=False)
#         self._fci_timers['evolve_pool_trotter_basic'] += self._fci_time.time() - t0

#         # build | psi_N > according ADAPT-VQE analytical grad section
#         qc_sig = qforte.FCIComputer(
#             self._nel, 
#             self._2_spin, 
#             self._norb) 

#         t0 = self._fci_time.time()
#         psi_i = qc_psi.get_state_deep()
#         self._fci_timers['get_state_deep'] += self._fci_time.time() - t0

#         # not sure if copy is faster or reapplication of state
#         t0 = self._fci_time.time()
#         qc_sig.set_state(psi_i)
#         self._fci_timers['set_state'] += self._fci_time.time() - t0

#         if(self._apply_ham_as_tensor):
#             t0 = self._fci_time.time()
#             qc_sig.apply_tensor_spat_012bdy(
#                 self._zero_body_energy, 
#                 self._mo_oeis, 
#                 self._mo_teis, 
#                 self._mo_teis_einsum, 
#                 self._norb)
#             self._fci_timers['apply_tensor_spat_012bdy'] += self._fci_time.time() - t0
#         else:
#             t0 = self._fci_time.time()
#             qc_sig.apply_sqop(self._sq_ham)
#             self._fci_timers['apply_sqop'] += self._fci_time.time() - t0

#         mu = M-1

#         # find <sing_N | K_N | psi_N>
#         Kmu_prev = self._pool_obj[self._tops[mu]][1]

#         Kmu_prev.mult_coeffs(self._pool_obj[self._tops[mu]][0])

#         t0 = self._fci_time.time()
#         qc_psi.apply_sqop(Kmu_prev)
#         self._fci_timers['apply_sqop'] += self._fci_time.time() - t0
        
#         t0 = self._fci_time.time()
#         grads[mu] = 2.0 * np.real(
#             qc_sig.get_state().vector_dot(qc_psi.get_state())
#             )
#         self._fci_timers['vector_dot'] += self._fci_time.time() - t0

#         #reset Kmu_prev |psi_i> -> |psi_i>
#         t0 = self._fci_time.time()
#         qc_psi.set_state(psi_i)
#         self._fci_timers['set_state'] += self._fci_time.time() - t0

#         for mu in reversed(range(M-1)):

#             # mu => N-1 => M-2
#             # mu+1 => N => M-1
#             # Kmu => KN-1
#             # Kmu_prev => KN

#             if params is None:
#                 tamp = self._tamps[mu+1]
#             else:
#                 tamp = params[mu+1]

#             # Kmu = self._pool_obj[self._tops[mu]][1].jw_transform(self._qubit_excitations)
#             Kmu = self._pool_obj[self._tops[mu]][1]

#             Kmu.mult_coeffs(self._pool_obj[self._tops[mu]][0])

#             # The minus sign is dictated by the recursive algorithm used to compute the analytic gradient
#             # (see original ADAPT-VQE paper)
#             t0 = self._fci_time.time()
#             qc_psi.apply_sqop_evolution(
#                 -1.0*tamp,
#                 Kmu_prev,
#                 antiherm=True,
#                 adjoint=False)
#             self._fci_timers['apply_sqop_evolution'] += self._fci_time.time() - t0
            
#             t0 = self._fci_time.time()
#             qc_sig.apply_sqop_evolution(
#                 -1.0*tamp,
#                 Kmu_prev,
#                 antiherm=True,
#                 adjoint=False)
#             self._fci_timers['apply_sqop_evolution'] += self._fci_time.time() - t0

#             t0 = self._fci_time.time()
#             psi_i = qc_psi.get_state_deep()
#             self._fci_timers['get_state_deep'] += self._fci_time.time() - t0

#             t0 = self._fci_time.time()
#             qc_psi.apply_sqop(Kmu)
#             self._fci_timers['apply_sqop'] += self._fci_time.time() - t0
            
#             t0 = self._fci_time.time()
#             grads[mu] = 2.0 * np.real(
#                 qc_sig.get_state().vector_dot(qc_psi.get_state())
#                 )
#             self._fci_timers['vector_dot'] += self._fci_time.time() - t0

#             #reset Kmu |psi_i> -> |psi_i>
#             t0 = self._fci_time.time()
#             qc_psi.set_state(psi_i)
#             self._fci_timers['set_state'] += self._fci_time.time() - t0
#             Kmu_prev = Kmu

#         np.testing.assert_allclose(np.imag(grads), np.zeros_like(grads), atol=1e-7)
        
#         return grads





# def measure_gradient_fci_gpu(self, params=None):
#         """ Returns the disentangled (factorized) UCC gradient for FCIComputerGPU,
#         using a recursive approach.

#         Parameters
#         ----------
#         params : list of floats
#             The variational parameters which characterize _Uvqc.
#         """

#         if not self._fast:
#             raise ValueError("self._fast must be True for gradient measurement.")
        
#         if(self._pool_type == 'sa_SD'):
#             raise ValueError('Must use single term particle-hole nbody operators for residual calculation')
        
#         if not self._ref_from_hf:
#             raise ValueError('measure_gradient_fci_gpu only compatible with hf reference at this time.')

#         M = len(self._tamps)
#         grads = np.zeros(M)
#         vqc_ops = qforte.SQOpPoolGPU(data_type=self.data_type)

#         if params is None:
#             for tamp, top in zip(self._tamps, self._tops):
#                 vqc_ops.add(tamp, self._pool_obj[top][1])
#         else:
#             for tamp, top in zip(params, self._tops):
#                 vqc_ops.add(tamp, self._pool_obj[top][1])

#         # build | sig_N > according ADAPT-VQE analytical grad section
#         qc_psi = qforte.FCIComputerGPU(
#             self._nel, 
#             self._2_spin, 
#             self._norb,
#             on_gpu=False,
#             data_type=self.data_type) 
        
#         t0 = self._gpu_time.time()
#         qc_psi.hartree_fock_cpu()
#         self._gpu_timers['hartree_fock_cpu'] += self._gpu_time.time() - t0
        
#         t0 = self._gpu_time.time()
#         qc_psi.to_gpu()
#         self._gpu_timers['to_gpu'] += self._gpu_time.time() - t0
        
#         t0 = self._gpu_time.time()
#         qc_psi.evolve_pool_trotter_basic_gpu(
#             vqc_ops,
#             antiherm=True,
#             adjoint=False)
#         self._gpu_timers['evolve_pool_trotter_basic_gpu'] += self._gpu_time.time() - t0

#         # build | psi_N > according ADAPT-VQE analytical grad section
#         qc_sig = qforte.FCIComputerGPU(
#             self._nel, 
#             self._2_spin, 
#             self._norb,
#             on_gpu=True,
#             data_type=self.data_type) 

#         # Initialize psi_i if it doesn't exist
#         if not hasattr(self, 'psi_i'):
#             self.psi_i = None

#         t0 = self._gpu_time.time()
#         if self.psi_i:
#             qc_psi.copy_state_into(self.psi_i)
#         else:
#             self.psi_i = qc_psi.get_state_deep()
#         self._gpu_timers['get_state_deep'] += self._gpu_time.time() - t0

#         # not sure if copy is faster or reapplication of state
#         t0 = self._gpu_time.time()
#         qc_sig.set_state_gpu(self.psi_i)
#         self._gpu_timers['set_state_gpu'] += self._gpu_time.time() - t0

#         if(self._apply_ham_as_tensor):
#             t0 = self._gpu_time.time()
#             qc_sig.apply_tensor_spat_012bdy_gpu(
#                 self._zero_body_energy, 
#                 self._mo_oeis_gpu, 
#                 self._mo_teis_gpu, 
#                 self._mo_teis_einsum_gpu, 
#                 self._norb)
#             self._gpu_timers['apply_tensor_spat_012bdy_gpu'] += self._gpu_time.time() - t0
#         else:
#             t0 = self._gpu_time.time()
#             qc_sig.apply_sqop_gpu(self._sq_ham)
#             self._gpu_timers['apply_sqop_gpu'] += self._gpu_time.time() - t0

#         mu = M-1

#         # find <sing_N | K_N | psi_N>
#         Kmu_prev = self._pool_obj[self._tops[mu]][1]

#         Kmu_prev.mult_coeffs(self._pool_obj[self._tops[mu]][0])

#         t0 = self._gpu_time.time()
#         qc_psi.apply_sqop_gpu(Kmu_prev)
#         self._gpu_timers['apply_sqop_gpu'] += self._gpu_time.time() - t0
        
#         t0 = self._gpu_time.time()
#         grads[mu] = 2.0 * np.real(
#             qc_sig.state_vector_dot_gpu(qc_psi)
#             )
#         self._gpu_timers['vector_dot'] += self._gpu_time.time() - t0
#         # print(f"[GPU GRAD] grads[{mu}] = {grads[mu]:.16f}")

#         #reset Kmu_prev |psi_i> -> |psi_i>
#         t0 = self._gpu_time.time()
#         qc_psi.set_state_gpu(self.psi_i)
#         self._gpu_timers['set_state_gpu'] += self._gpu_time.time() - t0

#         for mu in reversed(range(M-1)):

#             # mu => N-1 => M-2
#             # mu+1 => N => M-1
#             # Kmu => KN-1
#             # Kmu_prev => KN

#             if params is None:
#                 tamp = self._tamps[mu+1]
#             else:
#                 tamp = params[mu+1]

#             Kmu = self._pool_obj[self._tops[mu]][1]

#             Kmu.mult_coeffs(self._pool_obj[self._tops[mu]][0])

#             # The minus sign is dictated by the recursive algorithm used to compute the analytic gradient
#             # (see original ADAPT-VQE paper)
#             t0 = self._gpu_time.time()
#             qc_psi.apply_sqop_evolution_gpu(
#                 -1.0*tamp,
#                 Kmu_prev,
#                 antiherm=True,
#                 adjoint=False)
#             self._gpu_timers['apply_sqop_evolution_gpu'] += self._gpu_time.time() - t0
            
#             t0 = self._gpu_time.time()
#             qc_sig.apply_sqop_evolution_gpu(
#                 -1.0*tamp,
#                 Kmu_prev,
#                 antiherm=True,
#                 adjoint=False)
#             self._gpu_timers['apply_sqop_evolution_gpu'] += self._gpu_time.time() - t0

#             t0 = self._gpu_time.time()
#             qc_psi.copy_state_into(self.psi_i)
#             self._gpu_timers['get_state_deep'] += self._gpu_time.time() - t0

#             t0 = self._gpu_time.time()
#             qc_psi.apply_sqop_gpu(Kmu)
#             self._gpu_timers['apply_sqop_gpu'] += self._gpu_time.time() - t0
            
#             t0 = self._gpu_time.time()
#             grads[mu] = 2.0 * np.real(
#                 qc_sig.state_vector_dot_gpu(qc_psi)
#                 )
#             self._gpu_timers['vector_dot'] += self._gpu_time.time() - t0
#             # print(f"[GPU GRAD] grads[{mu}] = {grads[mu]:.16f}")

#             #reset Kmu |psi_i> -> |psi_i>
#             t0 = self._gpu_time.time()
#             qc_psi.set_state_gpu(self.psi_i)
#             self._gpu_timers['set_state_gpu'] += self._gpu_time.time() - t0
#             Kmu_prev = Kmu

#         np.testing.assert_allclose(np.imag(grads), np.zeros_like(grads), atol=1e-7)
        
#         return grads