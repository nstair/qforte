import numpy as np
import qforte as qf
from qforte import UCCNVQE  # or import from where class is defined
import copy

np.random.seed(1)

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

# 2) instantiate the two algorithm objects (no .run())
alg_fqe = qf.UCCNVQE(mol, apply_ham_as_tensor=True, computer_type='fqe', verbose=False)
alg_gpu = qf.UCCNVQE(mol, apply_ham_as_tensor=True, computer_type='fci_gpu', verbose=False)
alg_cpu = qf.UCCNVQE(mol, apply_ham_as_tensor=True, computer_type='fci', verbose=False)

# 3) prepare minimal internal state (pool + ansatz). This *does not* run the optimizer.
for alg in (alg_fqe, alg_gpu, alg_cpu):
    alg._fast = True                # required by measurement routines
    alg._ref_from_hf = True         # many measure_* require HF reference
    alg._pool_type = 'SD'           # set pool type
    alg.fill_pool()                 # populate _pool_obj
    alg.initialize_ansatz()         # sets _tops and _tamps
    # Give a small nonzero test vector so evolutions do non-trivial work
    alg._tamps = [0.01 * (i + 1) for i in range(len(alg._tamps))]

# 4) call measurement functions directly
# FQE path
grads_fqe = alg_fqe.measure_gradient(params=alg_fqe._tamps)   # calls measure_gradient_fqe internally
E_fqe = alg_fqe.energy_feval(alg_fqe._tamps)                 # should exist in VQE parent

# GPU path
# For GPU tests you can run on CPU-backed GPU object (on_gpu False) if you don't have a GPU:
# ensure the FCIComputerGPU behaves on your machine (if it requires CUDA and you don't have it set on_gpu=False).
# The UCCNVQE wrapper code will call measure_gradient_fci_gpu under the hood.
grads_gpu = alg_gpu.measure_gradient(params=alg_gpu._tamps)
E_gpu = alg_gpu.energy_feval(alg_gpu._tamps)

# CPU path
grads_cpu = alg_cpu.measure_gradient(params=alg_cpu._tamps)
E_cpu = alg_cpu.energy_feval(alg_cpu._tamps)

# --- Basic summary ---
print("\n=== Summary ===")
print("FQE grads shape, dtype:", grads_fqe.shape, grads_fqe.dtype)
print("GPU grads shape, dtype:", grads_gpu.shape, grads_gpu.dtype)
print(f"FQE energy: {E_fqe:+.15f}")
print(f"GPU energy: {E_gpu:+.15f}")
print("||grads_fqe||_2 =", np.linalg.norm(grads_fqe))
print("||grads_gpu||_2 =", np.linalg.norm(grads_gpu))
print("max abs diff grads:", np.max(np.abs(np.real(grads_fqe) - np.real(grads_gpu))))
print("max rel diff grads:",
      np.max(np.abs(np.real(grads_fqe) - np.real(grads_gpu)) / (np.abs(np.real(grads_fqe)) + 1e-16)))
print("energy diff:", E_fqe - E_gpu)

# --- elementwise cpu ---
n_show = min(80, len(grads_fqe))
print("\nIndex  |    grad_fqe (real)    imag   |    grad_cpu (real)    imag   |  abs_diff")
for i in range(n_show):
    gf = grads_fqe[i]
    gc = grads_cpu[i]
    absdiff = abs(np.real(gf) - np.real(gc))
    print(f"{i:4d}   {np.real(gf):+13.9e} {np.imag(gf):+8.1e}   {np.real(gc):+13.9e} {np.imag(gc):+8.1e}   {absdiff:9.3e}")

# --- elementwise table (first 50 elements or all if fewer) ---
n_show = min(80, len(grads_fqe))
print("\nIndex  |    grad_fqe (real)    imag   |    grad_gpu (real)    imag   |  abs_diff")
for i in range(n_show):
    gf = grads_fqe[i]
    gg = grads_gpu[i]
    absdiff = abs(np.real(gf) - np.real(gg))
    print(f"{i:4d}   {np.real(gf):+13.9e} {np.imag(gf):+8.1e}   {np.real(gg):+13.9e} {np.imag(gg):+8.1e}   {absdiff:9.3e}")

# --- show which pool operator each index corresponds to (brief) ---
print("\nPool operator summary for first entries:")
for i in range(n_show):
    coeff, op = alg_fqe._pool_obj[alg_fqe._tops[i]]
    # op.terms() or op.string() may exist depending on the SQOp representation in your codebase
    try:
        desc = op.terms()  # small summary of operator structure
    except Exception:
        try:
            desc = str(op)
        except Exception:
            desc = "<op repr unavailable>"
    print(f"{i:4d} coeff={coeff:.6e}  desc={desc}")

# --- save results for offline inspection ---
np.save("grads_fqe.npy", np.real(grads_fqe))
np.save("grads_gpu.npy", np.real(grads_gpu))
print("\nSaved grads_fqe.npy and grads_gpu.npy (real parts).")

# --- quick checks and asserts (non-fatal prints) ---
# imaginary parts should be ~0
max_im_fqe = np.max(np.abs(np.imag(grads_fqe)))
max_im_gpu = np.max(np.abs(np.imag(grads_gpu)))
print(f"\nMax imag part: FQE {max_im_fqe:.3e}, GPU {max_im_gpu:.3e}")

if max_im_fqe > 1e-8 or max_im_gpu > 1e-8:
    print("WARNING: significant imaginary components found in gradients.")

# If values differ more than you'd expect, print the indices sorted by largest diff
diffs = np.abs(np.real(grads_fqe) - np.real(grads_gpu))
idx_sort = np.argsort(-diffs)  # descending
print("\nTop 10 indices by abs gradient difference:")
for j in idx_sort[:10]:
    print(f" idx {j:4d}  diff={diffs[j]:.6e}  fqe={np.real(grads_fqe[j]):.12e}  gpu={np.real(grads_gpu[j]):.12e}")