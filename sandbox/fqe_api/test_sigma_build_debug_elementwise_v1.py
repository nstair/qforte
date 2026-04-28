import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["OMP_PROC_BIND"] = "FALSE"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qforte")

import numpy as np
import psi4
import qforte as qf


THRESHOLD = 1.0e-14
DEFAULT_NH = 12


def tensor_diff(lhs, rhs):
    diff = lhs.get_state_deep()
    diff.subtract(rhs.get_state_deep())
    return diff.norm()


def main():
    # Keep this deliberately smaller than test_advanced_api_v1.py.  The new
    # elementwise path is a correctness/debug reference for a future CUDA
    # kernel, not a production CPU path.
    nh = int(os.environ.get("QFORTE_SIGMA_NH", DEFAULT_NH))
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

    print("\n Sigma Build Debug Elementwise")
    print("==============================")
    print(f"norb:      {norb}")
    print(f"hydrogens: {nh}")
    print(f"nqubit:    {2 * norb}")
    print(f"nel:       {nel}")
    print(f"threshold: {THRESHOLD:.1e}")

    psi4.core.set_num_threads(1)

    fci_opt = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
    fci_dbg = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
    fci_dbg_12 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
    fqe_ref = qf.FQEComputer(nel=nel, sz=sz, norb=norb)
    fqe_ref_12 = qf.FQEComputer(nel=nel, sz=sz, norb=norb)

    fci_opt.hartree_fock()
    fci_dbg.hartree_fock()
    fci_dbg_12.hartree_fock()
    fqe_ref.hartree_fock()
    fqe_ref_12.hartree_fock()

    timer = qf.local_timer()

    # Optimized CPU path: useful as a second local cross-check, but this is not
    # the implementation under test.
    timer.reset()
    fci_opt.apply_tensor_spat_012bdy(
        mol.nuclear_repulsion_energy,
        mol.mo_oeis,
        mol.mo_teis,
        mol.mo_teis_einsum,
        norb,
    )
    timer.record("FCI optimized sigma build")

    # New CPU debug/reference path.  This is the element-by-element sigma build
    # intended to outline the eventual CUDA kernel.
    timer.reset()
    fci_dbg.apply_tensor_spat_012bdy_debug_elementwise(
        mol.nuclear_repulsion_energy,
        mol.mo_oeis,
        mol.mo_teis,
        mol.mo_teis_einsum,
        norb,
    )
    timer.record("FCI debug elementwise sigma build")

    # FQE reference uses the NumPy integrals saved by system_factory.
    timer.reset()
    fqe_ref.apply_tensor_spat_012bdy(
        mol.nuclear_repulsion_energy,
        mol.mo_oeis_np,
        mol.mo_teis_np,
    )
    timer.record("FQE sigma build")

    # Also exercise the explicit 12-body debug entry point.  FQE does not have
    # a separate adapter method for this in qforte, so use the same FQE tensor
    # apply with a zero scalar term.
    timer.reset()
    fci_dbg_12.apply_tensor_spat_12bdy_debug_elementwise(
        mol.mo_oeis,
        mol.mo_teis,
        mol.mo_teis_einsum,
        norb,
    )
    timer.record("FCI debug elementwise 12-body sigma build")

    timer.reset()
    fqe_ref_12.apply_tensor_spat_012bdy(
        0.0,
        mol.mo_oeis_np,
        mol.mo_teis_np,
    )
    timer.record("FQE 12-body sigma build")

    dbg_vs_fqe = fqe_ref.get_tensor_diff(fci_dbg.get_state_deep())
    opt_vs_fqe = fqe_ref.get_tensor_diff(fci_opt.get_state_deep())
    dbg_vs_opt = tensor_diff(fci_dbg, fci_opt)
    dbg_12_vs_fqe = fqe_ref_12.get_tensor_diff(fci_dbg_12.get_state_deep())

    print(f"\n|dC| debug elementwise vs FQE: {dbg_vs_fqe:.16e}")
    print(f"|dC| optimized CPU vs FQE:    {opt_vs_fqe:.16e}")
    print(f"|dC| debug elementwise vs opt: {dbg_vs_opt:.16e}\n")
    print(f"|dC| debug elementwise 12-body vs FQE: {dbg_12_vs_fqe:.16e}\n")

    print(timer)

    if not np.isfinite(dbg_vs_fqe) or dbg_vs_fqe >= THRESHOLD:
        raise AssertionError(
            "Debug elementwise sigma build did not meet the "
            f"{THRESHOLD:.1e} threshold: |dC| = {dbg_vs_fqe:.16e}"
        )

    if not np.isfinite(dbg_12_vs_fqe) or dbg_12_vs_fqe >= THRESHOLD:
        raise AssertionError(
            "Debug elementwise 12-body sigma build did not meet the "
            f"{THRESHOLD:.1e} threshold: |dC| = {dbg_12_vs_fqe:.16e}"
        )


if __name__ == "__main__":
    main()
