import os

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OMP_DYNAMIC"] = "FALSE"
# os.environ["OMP_PROC_BIND"] = "FALSE"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qforte")

import numpy as np
import psi4
import qforte as qf


DEFAULT_NH = 14
DEFAULT_THRESHOLD = 1.0e-12

RUN_FQE = False
RUN_QFORTE_CPU = False
RUN_QFORTE_GPU_REAL = False
RUN_QFORTE_GPU_REAL_V2 = True
RUN_QFORTE_CPU_DEBUG = False


def env_flag(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "no", "off")


def tensor_diff(lhs, rhs):
    diff = lhs.get_state_deep()
    diff.subtract(rhs.get_state_deep())
    return diff.norm()


def tensor_gpu_from_cpu(tensor, name):
    tensor_gpu = qf.TensorGPU(tensor.shape(), name, False, "real")
    tensor_gpu.fill_from_tensor_cpu(tensor, tensor.shape())
    tensor_gpu.to_gpu()
    return tensor_gpu


def run_fqe(mol, nel, sz, norb, timer):
    fqe = qf.FQEComputer(nel=nel, sz=sz, norb=norb)
    fqe.hartree_fock()

    timer.reset()
    fqe.apply_tensor_spat_012bdy(
        mol.nuclear_repulsion_energy,
        mol.mo_oeis_np,
        mol.mo_teis_np,
    )
    timer.record("FQE sigma build")
    return fqe


def run_qforte_cpu(mol, nel, sz, norb, timer):
    fci = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
    fci.hartree_fock()

    timer.reset()
    fci.apply_tensor_spat_012bdy(
        mol.nuclear_repulsion_energy,
        mol.mo_oeis,
        mol.mo_teis,
        mol.mo_teis_einsum,
        norb,
    )
    timer.record("qforte CPU sigma build")
    return fci


def run_qforte_cpu_debug(mol, nel, sz, norb, timer):
    fci = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
    fci.hartree_fock()

    timer.reset()
    fci.apply_tensor_spat_012bdy_debug_elementwise(
        mol.nuclear_repulsion_energy,
        mol.mo_oeis,
        mol.mo_teis,
        mol.mo_teis_einsum,
        norb,
    )
    timer.record("qforte CPU debug elementwise sigma build")
    return fci


def run_qforte_gpu_real(mol, nel, sz, norb, timer):
    fci = qf.FCIComputerGPU(
        nel=nel,
        sz=sz,
        norb=norb,
        on_gpu=False,
        data_type="real",
    )
    fci.hartree_fock_cpu()

    mo_oeis_gpu = tensor_gpu_from_cpu(mol.mo_oeis, "mo_oeis_gpu")
    mo_teis_gpu = tensor_gpu_from_cpu(mol.mo_teis, "mo_teis_gpu")
    mo_teis_einsum_gpu = tensor_gpu_from_cpu(mol.mo_teis_einsum, "mo_teis_einsum_gpu")
    fci.to_gpu()

    timer.reset()
    fci.apply_tensor_spat_012bdy_gpu(
        mol.nuclear_repulsion_energy,
        mo_oeis_gpu,
        mo_teis_gpu,
        mo_teis_einsum_gpu,
        norb,
    )
    timer.record("qforte GPU real sigma build")

    fci.to_cpu()
    state = qf.Tensor([fci.get_Na(), fci.get_Nb()], "qforte GPU real state")
    fci.copy_to_tensor_cpu(state)
    return fci, state


def run_qforte_gpu_real_v2(mol, nel, sz, norb, timer):
    fci = qf.FCIComputerGPU(
        nel=nel,
        sz=sz,
        norb=norb,
        on_gpu=False,
        data_type="real",
    )
    fci.hartree_fock_cpu()

    mo_oeis_gpu = tensor_gpu_from_cpu(mol.mo_oeis, "mo_oeis_gpu_v2")
    mo_teis_gpu = tensor_gpu_from_cpu(mol.mo_teis, "mo_teis_gpu_v2")
    mo_teis_einsum_gpu = tensor_gpu_from_cpu(mol.mo_teis_einsum, "mo_teis_einsum_gpu_v2")
    fci.to_gpu()

    timer.reset()
    fci.apply_tensor_spat_012bdy_gpu_v2(
        mol.nuclear_repulsion_energy,
        mo_oeis_gpu,
        mo_teis_gpu,
        mo_teis_einsum_gpu,
        norb,
    )
    timer.record("qforte GPU real sigma build v2")

    fci.to_cpu()
    state = qf.Tensor([fci.get_Na(), fci.get_Nb()], "qforte GPU real state v2")
    fci.copy_to_tensor_cpu(state)
    return fci, state


def main():
    nh = int(os.environ.get("QFORTE_SIGMA_NH", DEFAULT_NH))
    threshold = float(os.environ.get("QFORTE_SIGMA_THRESHOLD", DEFAULT_THRESHOLD))
    check_errors = env_flag("QFORTE_SIGMA_CHECK_ERRORS", True)

    if not any((RUN_FQE, RUN_QFORTE_CPU, RUN_QFORTE_GPU_REAL, RUN_QFORTE_GPU_REAL_V2, RUN_QFORTE_CPU_DEBUG)):
        raise ValueError("At least one sigma-build case must be enabled.")

    geom = [("H", (0.0, 0.0, float(i + 1))) for i in range(nh)]

    mol = qf.system_factory(
        build_type="psi4",
        mol_geometry=geom,
        basis="sto-3g",
        run_fci=0,
        build_qb_ham=False,
        store_mo_ints=True,
        store_mo_ints_np=True,
        build_df_ham=0,
        df_icut=1.0e-6,
    )

    ref = mol.hf_reference
    nel = sum(ref)
    sz = 0
    norb = int(len(ref) / 2)

    print("\n Improved Sigma Build Comparison")
    print("================================")
    print(f"hydrogens:    {nh}")
    print(f"norb:         {norb}")
    print(f"nqubit:       {2 * norb}")
    print(f"nel:          {nel}")
    print(f"run FQE:      {RUN_FQE}")
    print(f"run CPU:      {RUN_QFORTE_CPU}")
    print(f"run GPU real: {RUN_QFORTE_GPU_REAL}")
    print(f"run GPU v2:   {RUN_QFORTE_GPU_REAL_V2}")
    print(f"run debug:    {RUN_QFORTE_CPU_DEBUG}")
    print(f"check errors: {check_errors}")
    print(f"threshold:    {threshold:.1e}")

    psi4.core.set_num_threads(1)
    timer = qf.local_timer()

    fqe = None
    fci_cpu = None
    fci_debug = None
    gpu_state = None
    gpu_v2_state = None
    gpu_acc_timer = None
    gpu_v2_acc_timer = None

    if RUN_FQE:
        fqe = run_fqe(mol, nel, sz, norb, timer)

    if RUN_QFORTE_CPU:
        fci_cpu = run_qforte_cpu(mol, nel, sz, norb, timer)

    if RUN_QFORTE_CPU_DEBUG:
        fci_debug = run_qforte_cpu_debug(mol, nel, sz, norb, timer)

    if RUN_QFORTE_GPU_REAL:
        fci_gpu, gpu_state = run_qforte_gpu_real(mol, nel, sz, norb, timer)
        gpu_acc_timer = fci_gpu.get_acc_timer()

    if RUN_QFORTE_GPU_REAL_V2:
        fci_gpu_v2, gpu_v2_state = run_qforte_gpu_real_v2(mol, nel, sz, norb, timer)
        gpu_v2_acc_timer = fci_gpu_v2.get_acc_timer()

    if check_errors:
        errors = {}
        printed_header = False

        def print_error(label, error):
            nonlocal printed_header
            if not printed_header:
                print("")
                printed_header = True
            print(f"|dC| {label:<31} {error:.16e}")
            errors[label] = error

        if fqe is not None and fci_cpu is not None:
            print_error("qforte CPU vs FQE:", fqe.get_tensor_diff(fci_cpu.get_state_deep()))

        if fqe is not None and fci_debug is not None:
            print_error("qforte CPU debug vs FQE:", fqe.get_tensor_diff(fci_debug.get_state_deep()))

        if fci_debug is not None and fci_cpu is not None:
            print_error("qforte CPU debug vs CPU:", tensor_diff(fci_debug, fci_cpu))

        if fqe is not None and gpu_state is not None:
            print_error("qforte GPU real vs FQE:", fqe.get_tensor_diff(gpu_state))

        if fqe is not None and gpu_v2_state is not None:
            print_error("qforte GPU real v2 vs FQE:", fqe.get_tensor_diff(gpu_v2_state))

        if fci_cpu is not None and gpu_state is not None:
            gpu_vs_cpu = fci_cpu.get_state_deep()
            gpu_vs_cpu.subtract(gpu_state)
            print_error("qforte GPU real vs CPU:", gpu_vs_cpu.norm())

        if fci_cpu is not None and gpu_v2_state is not None:
            gpu_v2_vs_cpu = fci_cpu.get_state_deep()
            gpu_v2_vs_cpu.subtract(gpu_v2_state)
            print_error("qforte GPU real v2 vs CPU:", gpu_v2_vs_cpu.norm())

        if gpu_state is not None and gpu_v2_state is not None:
            gpu_v2_vs_gpu = qf.Tensor(gpu_state.shape(), "qforte GPU real v2 minus GPU")
            gpu_v2_vs_gpu.copy_in(gpu_state)
            gpu_v2_vs_gpu.subtract(gpu_v2_state)
            print_error("qforte GPU real v2 vs GPU:", gpu_v2_vs_gpu.norm())

        for label, error in errors.items():
            if not np.isfinite(error) or error >= threshold:
                raise AssertionError(
                    f"{label} exceeded {threshold:.1e}: |dC| = {error:.16e}"
                )

    print("\n")
    print(timer)

    if gpu_acc_timer is not None:
        print("\nqforte GPU real internal timer:")
        print(gpu_acc_timer.acc_str_table())

    if gpu_v2_acc_timer is not None:
        print("\nqforte GPU real v2 internal timer:")
        print(gpu_v2_acc_timer.acc_str_table())


if __name__ == "__main__":
    main()
