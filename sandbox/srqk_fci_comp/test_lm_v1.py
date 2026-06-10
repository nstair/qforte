import numpy as np
import qforte as qf


# Edit-and-run controls.
BACKENDS = ["fci"]
# BACKENDS = ["fci", "fock", "fqe"]
# BACKENDS = ["fci_gpu"]  # Requires the GPU build/runtime.
# BACKENDS = ["cusv"]     # Requires the cuStateVec runtime.

S = 3
DT = 0.1
TROTTER_NUMBER = 1
TROTTER_ORDER = 2
USE_EXACT_EVOLUTION = False
DIAGONALIZE_EACH_STEP = True

MATRIX_ATOL = 1.0e-10
MATRIX_RTOL = 1.0e-10
ENERGY_ATOL = 1.0e-8


# geom = [
#     ("H", (0.0, 0.0, 1.00)),
#     ("H", (0.0, 0.0, 2.00)),
#     ("H", (0.0, 0.0, 3.00)),
#     ("H", (0.0, 0.0, 4.00)),
#     ("H", (0.0, 0.0, 5.00)),
#     ("H", (0.0, 0.0, 6.00)),
#     ("H", (0.0, 0.0, 7.00)),
#     ("H", (0.0, 0.0, 8.00)),
# ]

geom = [
    ("N", (0.0, 0.0, 1.00)),
    ("N", (0.0, 0.0, 2.00)),
]

mol = qf.system_factory(
    build_type="psi4",
    mol_geometry=geom,
    basis="sto-3g",
    run_fci=1,
    store_mo_ints_np=True,
)


def backend_kwargs(computer_type):
    kwargs = {
        "computer_type": computer_type,
        "trotter_number": TROTTER_NUMBER,
        "trotter_order": TROTTER_ORDER,
    }
    if computer_type == "cusv":
        kwargs["apply_ham_as_tensor"] = False
    return kwargs


def run_srqk(computer_type, low_memory_mat_formation):
    alg = qf.SRQK(mol, **backend_kwargs(computer_type))
    alg.run(
        s=S,
        dt=DT,
        use_exact_evolution=USE_EXACT_EVOLUTION,
        diagonalize_each_step=DIAGONALIZE_EACH_STEP,
        low_memory_mat_formation=low_memory_mat_formation,
    )
    return alg


print("\n\n==> SRQK low-memory matrix formation comparison <==")
print(f"  FCI reference: {mol.fci_energy:+12.10f}")
print(f"  s:             {S}")
print(f"  dt:            {DT}")
print(f"  trotter:       order={TROTTER_ORDER}, number={TROTTER_NUMBER}")

for backend in BACKENDS:
    print(f"\n\n==> Backend: {backend} <==")

    alg_standard = run_srqk(backend, low_memory_mat_formation=False)
    alg_low_mem = run_srqk(backend, low_memory_mat_formation=True)

    s_diff = np.max(np.abs(alg_standard._S - alg_low_mem._S))
    h_diff = np.max(np.abs(alg_standard._Hbar - alg_low_mem._Hbar))
    e_diff = abs(alg_standard.get_gs_energy() - alg_low_mem.get_gs_energy())

    print("\nComparison")
    print(f"  E standard:  {alg_standard.get_gs_energy():+12.10f}")
    print(f"  E low-mem:   {alg_low_mem.get_gs_energy():+12.10f}")
    print(f"  |dE|:        {e_diff:.6e}")
    print(f"  max |dS|:    {s_diff:.6e}")
    print(f"  max |dH|:    {h_diff:.6e}")

    np.testing.assert_allclose(
        alg_low_mem._S,
        alg_standard._S,
        atol=MATRIX_ATOL,
        rtol=MATRIX_RTOL,
    )
    np.testing.assert_allclose(
        alg_low_mem._Hbar,
        alg_standard._Hbar,
        atol=MATRIX_ATOL,
        rtol=MATRIX_RTOL,
    )
    np.testing.assert_allclose(
        alg_low_mem.get_gs_energy(),
        alg_standard.get_gs_energy(),
        atol=ENERGY_ATOL,
        rtol=0.0,
    )

print("\nAll requested SRQK low-memory checks passed.")
