"""Informal Hamiltonian-application parity check from saved PySCF dumps.

Run from the repository root with the user-selected qforte environment, for example:

    conda run -n qf-cusv1 python informal_tests/ham_application/run_ham_application.py

This is intentionally not a pytest test. It builds BeH2 and benzene AVAS
molecules from the committed PySCF dumps, compares several backend-specific
Hamiltonian-application routes against saved FCI tensor reference sigma vectors,
and prints compact PASS/SKIP tables with per-case timings.

Only refresh the saved references after reviewing intentional numerical changes:

    conda run -n qf-cusv1 python informal_tests/ham_application/run_ham_application.py --write-reference
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import ham_application_common as common


def build_reference_sigma(mol):
    qf = common.get_qforte()
    counts = common.occupancy_counts(mol)

    comp = qf.FCIComputer(counts["nel"], counts["sz"], counts["norb"])
    comp.hartree_fock()

    start = time.perf_counter()
    comp.apply_tensor_spat_012bdy(
        common.zero_body_energy(mol),
        mol.mo_oeis,
        mol.mo_teis,
        mol.mo_teis_einsum,
        counts["norb"],
    )
    elapsed = time.perf_counter() - start
    state = common.tensor_to_numpy(comp.get_state_deep())
    return comp, state, elapsed


def build_ref_tensor(reference_state: np.ndarray):
    return common.numpy_to_tensor(reference_state, name="reference_sigma")


def build_ref_fci(mol, reference_state: np.ndarray):
    return common.make_reference_fci_computer(mol, reference_state)


def build_tensor_gpu(mol_tensor, name: str):
    qf = common.get_qforte()
    tensor_gpu = qf.TensorGPU(list(mol_tensor.shape()), name, False)
    tensor_gpu.fill_from_tensor_cpu(mol_tensor, list(mol_tensor.shape()))
    tensor_gpu.to_gpu()
    return tensor_gpu


def copy_fci_gpu_state_to_numpy(comp, shape):
    qf = common.get_qforte()
    comp.to_cpu()
    tensor = qf.Tensor(list(shape), "gpu_state")
    comp.copy_to_tensor_cpu(tensor)
    return common.tensor_to_numpy(tensor)


def run_fci_tensor_case(mol, reference_state, _ref_fci, _ref_tensor):
    _comp, state, elapsed = build_reference_sigma(mol)
    diff = float(np.linalg.norm((state - reference_state).ravel()))
    return diff, elapsed


def run_fci_sqop_case(mol, reference_state, _ref_fci, _ref_tensor):
    qf = common.get_qforte()
    counts = common.occupancy_counts(mol)
    comp = qf.FCIComputer(counts["nel"], counts["sz"], counts["norb"])
    comp.hartree_fock()

    start = time.perf_counter()
    comp.apply_sqop(mol.sq_hamiltonian)
    elapsed = time.perf_counter() - start

    state = common.tensor_to_numpy(comp.get_state_deep())
    diff = float(np.linalg.norm((state - reference_state).ravel()))
    return diff, elapsed


def run_fci_gpu_tensor_case(mol, reference_state, _ref_fci, _ref_tensor):
    qf = common.get_qforte()
    counts = common.occupancy_counts(mol)
    comp = qf.FCIComputerGPU(counts["nel"], counts["sz"], counts["norb"], False, "complex")
    comp.hartree_fock_cpu()

    mo_oeis_gpu = build_tensor_gpu(mol.mo_oeis, "mo_oeis_gpu")
    mo_teis_gpu = build_tensor_gpu(mol.mo_teis, "mo_teis_gpu")
    mo_teis_einsum_gpu = build_tensor_gpu(mol.mo_teis_einsum, "mo_teis_einsum_gpu")
    comp.to_gpu()

    start = time.perf_counter()
    comp.apply_tensor_spat_012bdy_gpu(
        common.zero_body_energy(mol),
        mo_oeis_gpu,
        mo_teis_gpu,
        mo_teis_einsum_gpu,
        counts["norb"],
    )
    elapsed = time.perf_counter() - start

    state = copy_fci_gpu_state_to_numpy(comp, reference_state.shape)
    diff = float(np.linalg.norm((state - reference_state).ravel()))
    return diff, elapsed


def run_fci_gpu_sqop_case(mol, reference_state, _ref_fci, _ref_tensor):
    qf = common.get_qforte()
    counts = common.occupancy_counts(mol)
    comp = qf.FCIComputerGPU(counts["nel"], counts["sz"], counts["norb"], False, "complex")
    comp.hartree_fock_cpu()
    comp.to_gpu()

    start = time.perf_counter()
    comp.apply_sqop_gpu(mol.sq_hamiltonian)
    elapsed = time.perf_counter() - start

    state = copy_fci_gpu_state_to_numpy(comp, reference_state.shape)
    diff = float(np.linalg.norm((state - reference_state).ravel()))
    return diff, elapsed


def run_fock_qubit_operator_case(mol, _reference_state, ref_fci, _ref_tensor):
    qf = common.get_qforte()
    counts = common.occupancy_counts(mol)
    comp = qf.Computer(counts["nqubit"])
    common.set_fock_hartree_fock(comp, mol)

    start = time.perf_counter()
    comp.apply_operator(mol.hamiltonian)
    elapsed = time.perf_counter() - start

    diff = float(comp.get_fci_comp_state_diff(ref_fci, do_phase_compare=True))
    return diff, elapsed


def run_fqe_tensor_case(mol, _reference_state, _ref_fci, ref_tensor):
    qf = common.get_qforte()
    counts = common.occupancy_counts(mol)
    comp = qf.FQEComputer(counts["nel"], counts["sz"], counts["norb"])
    comp.hartree_fock()

    start = time.perf_counter()
    comp.apply_tensor_spat_012bdy(
        common.zero_body_energy(mol),
        mol.mo_oeis_np,
        mol.mo_teis_np,
    )
    elapsed = time.perf_counter() - start

    diff = float(comp.get_tensor_diff(ref_tensor))
    return diff, elapsed


def run_fqe_sqop_case(mol, _reference_state, _ref_fci, ref_tensor):
    qf = common.get_qforte()
    counts = common.occupancy_counts(mol)
    comp = qf.FQEComputer(counts["nel"], counts["sz"], counts["norb"])
    comp.hartree_fock()

    start = time.perf_counter()
    comp.apply_sqop(mol.sq_hamiltonian)
    elapsed = time.perf_counter() - start

    diff = float(comp.get_tensor_diff(ref_tensor))
    return diff, elapsed


def run_cusv_sqop_case(mol, _reference_state, ref_fci, _ref_tensor):
    qf = common.get_qforte()
    counts = common.occupancy_counts(mol)
    comp = qf.CUSVComputer(
        counts["nel"],
        counts["sz"],
        counts["norb"],
        on_gpu=False,
        dtype=np.complex128,
    )
    comp.hartree_fock()
    comp.to_gpu()

    start = time.perf_counter()
    comp.apply_sqop(mol.sq_hamiltonian)
    elapsed = time.perf_counter() - start

    comp.to_cpu()
    diff = float(comp.get_fci_comp_state_diff(ref_fci, do_phase_compare=True))
    return diff, elapsed


CASE_RUNNERS = {
    "fci.tensor": run_fci_tensor_case,
    "fci.sqop": run_fci_sqop_case,
    "fci_gpu.tensor": run_fci_gpu_tensor_case,
    "fci_gpu.sqop": run_fci_gpu_sqop_case,
    "fock.qubit_operator": run_fock_qubit_operator_case,
    "fqe.tensor": run_fqe_tensor_case,
    "fqe.sqop": run_fqe_sqop_case,
    "cusv.sqop": run_cusv_sqop_case,
}


def execute_case(system: str, mol, case, reference_state, ref_fci, ref_tensor, log_root: Path):
    available, reason = common.backend_available(case.backend)
    log_path = log_root / system / f"{case.label.replace('.', '_')}.log"

    if not available:
        status = "SKIP" if case.optional else "FAIL"
        return {
            "case": case.label,
            "status": status,
            "elapsed_s": None,
            "diff_norm": None,
            "tol": case.tol,
            "reason": reason,
            "log_path": log_path,
        }

    def _runner():
        return CASE_RUNNERS[case.label](mol, reference_state, ref_fci, ref_tensor)

    try:
        diff, elapsed = common.run_with_log(log_path, _runner)
    except Exception as exc:
        if case.optional and common.should_skip_runtime_error(case.backend, exc):
            return {
                "case": case.label,
                "status": "SKIP",
                "elapsed_s": None,
                "diff_norm": None,
                "tol": case.tol,
                "reason": f"{type(exc).__name__}: {exc}",
                "log_path": log_path,
            }
        return {
            "case": case.label,
            "status": "FAIL",
            "elapsed_s": None,
            "diff_norm": None,
            "tol": case.tol,
            "reason": f"{type(exc).__name__}: {exc}",
            "log_path": log_path,
        }

    status = "PASS" if diff <= case.tol else "FAIL"
    reason = None
    if status == "FAIL":
        reason = f"diff {diff:.4e} exceeded tolerance {case.tol:.1e}"

    return {
        "case": case.label,
        "status": status,
        "elapsed_s": elapsed,
        "diff_norm": diff,
        "tol": case.tol,
        "reason": reason,
        "log_path": log_path,
    }


def write_references(systems: list[str]):
    print("\n==> Informal Hamiltonian-application reference refresh <==")
    for system in systems:
        log_root = common.LOG_DIR / system
        mol, build_log = common.build_dump_molecule(system, log_root=log_root, run_fci=False)
        common.print_system_header(system, mol, build_log, common.reference_path(system))
        _comp, state, elapsed = build_reference_sigma(mol)
        path = common.save_reference_state(system, mol, state)
        print(f"  wrote reference: {path}")
        print(f"  FCI tensor time: {elapsed:.6f} s")
    print("\nReference refresh completed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-reference",
        action="store_true",
        help="Refresh the saved FCI tensor sigma references for the selected systems and exit.",
    )
    parser.add_argument(
        "--system",
        action="append",
        dest="systems",
        help="Run one system. May be supplied more than once.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        help="Run only cases attached to one backend. May be supplied more than once.",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Run only one named case. May be supplied more than once.",
    )
    parser.add_argument(
        "--list-systems",
        action="store_true",
        help="Print system labels and exit.",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="Print backend strings and exit.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print case labels and exit.",
    )
    args = parser.parse_args()

    if args.list_systems:
        for system in common.SYSTEMS:
            print(system)
        return

    if args.list_backends:
        for backend in sorted({case.backend for case in common.CASE_MATRIX}):
            print(backend)
        return

    if args.list_cases:
        for case in common.CASE_MATRIX:
            print(case.label)
        return

    systems = common.selected_systems(args.systems)
    cases = common.selected_cases(args.cases, args.backends)

    if args.write_reference:
        write_references(systems)
        return

    print("\n==> Informal Hamiltonian-application parity check <==")
    print(f"  systems:  {', '.join(systems)}")
    print(f"  cases:    {', '.join(case.label for case in cases)}")

    failures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    passed = 0

    for system in systems:
        log_root = common.LOG_DIR / system
        mol, build_log = common.build_dump_molecule(system, log_root=log_root, run_fci=False)
        reference_state = common.load_reference_state(system)
        reference_file = common.reference_path(system)
        ref_fci = build_ref_fci(mol, reference_state)
        ref_tensor = build_ref_tensor(reference_state)

        common.print_system_header(system, mol, build_log, reference_file)
        records = []
        for case in cases:
            record = execute_case(system, mol, case, reference_state, ref_fci, ref_tensor, common.LOG_DIR)
            records.append(record)
            if record["status"] == "PASS":
                passed += 1
            elif record["status"] == "SKIP":
                skipped.append({"system": system, **record})
            else:
                failures.append({"system": system, **record})

        common.print_case_table(records)

    if skipped:
        print(f"\nSkipped {len(skipped)} run(s):")
        for item in skipped:
            print(f"  - {item['system']}.{item['case']}: {item['reason']}")

    if failures:
        print(f"\nFailed {len(failures)} run(s):")
        for item in failures:
            print(f"  - {item['system']}.{item['case']}: {item['reason']} (log: {item['log_path']})")
        raise AssertionError("Informal Hamiltonian-application parity check failed.")

    print(f"\nHamiltonian-application parity passed for {passed} run(s).")


if __name__ == "__main__":
    main()
