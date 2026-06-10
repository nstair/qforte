"""Informal BeH2 molecule-build parity check for PySCF dumps.

This builds BeH2/STO-3G three ways:

  1. PySCF backend
  2. Psi4 backend, when available
  3. saved PySCF dump through build_type="pyscf_dump"

The PySCF dump path is compared strictly against the direct PySCF molecule.
Psi4 is also checked, but direct MO tensor rows are printed as non-required
when orbital phase conventions differ between packages.
"""

from __future__ import annotations

import pyscf_dump_informal_common as common


def main():
    print("\n==> Informal BeH2 PySCF dump molecule parity check <==")
    common.print_geometry("bent BeH2", common.beh2_geometry())
    common.generate_beh2_dump()

    log_root = common.LOG_DIR / "beh2_molecule_parity"
    molecules = {}
    for build_type in ("pyscf", "psi4", "pyscf_dump"):
        mol, log_path, error = common.maybe_build_molecule("beh2", build_type, log_root=log_root)
        print(f"\n  build_type={build_type:<10s} log={log_path}")
        if error:
            print(f"  status: SKIP build failed: {error}")
            continue
        molecules[build_type] = mol
        print(f"  status: PASS build completed")
        print(f"  HF energy:       {mol.hf_energy:+18.12f}")
        print(f"  FCI energy:      {common.fci_energy(mol):+18.12f}")
        print(f"  scalar energy:   {common.scalar_energy(mol):+18.12f}")

    if "pyscf" not in molecules:
        raise AssertionError("The direct PySCF molecule is required for this check.")
    if "pyscf_dump" not in molecules:
        raise AssertionError("The PySCF dump molecule is required for this check.")

    rows = common.molecule_comparison_rows(molecules["pyscf"], molecules["pyscf_dump"])
    common.print_check_table("PySCF dump vs direct PySCF", rows)

    if "psi4" in molecules:
        rows = common.molecule_comparison_rows(
            molecules["pyscf"],
            molecules["psi4"],
            include_phase_relaxed=True,
            psi4_candidate=True,
        )
        common.print_check_table("Psi4 vs direct PySCF", rows)
    else:
        print("\n==> Psi4 vs direct PySCF <==")
        print("  status: SKIP psi4 backend was not available in this environment.")

    print("\nInformal BeH2 PySCF dump molecule parity check passed.")


if __name__ == "__main__":
    main()
