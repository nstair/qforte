"""Informal benzene AVAS molecule-build parity check for PySCF dumps.

This builds a benzene/STO-3G C 2pz AVAS active space directly from PySCF and
from a qforte-readable PySCF dump, then compares the qforte molecule fields.
"""

from __future__ import annotations

import pyscf_dump_informal_common as common


def main():
    print("\n==> Informal benzene AVAS PySCF dump molecule parity check <==")
    common.print_geometry("benzene C 2pz AVAS", common.benzene_geometry())
    common.generate_benzene_avas_dump()

    log_root = common.LOG_DIR / "benzene_avas_molecule_parity"
    molecules = {}
    for build_type in ("pyscf", "pyscf_dump"):
        mol, log_path, error = common.maybe_build_molecule("benzene", build_type, log_root=log_root)
        print(f"\n  build_type={build_type:<10s} log={log_path}")
        if error:
            print(f"  status: FAIL build failed: {error}")
            raise AssertionError(f"Could not build benzene with {build_type}: {error}")
        molecules[build_type] = mol
        print("  status: PASS build completed")
        print(f"  HF energy:          {mol.hf_energy:+18.12f}")
        print(f"  CASCI energy:       {common.fci_energy(mol):+18.12f}")
        print(f"  scalar energy:      {common.scalar_energy(mol):+18.12f}")
        print(f"  active orbitals:    {len(mol.hf_orbital_energies)}")
        print(f"  active electrons:   {sum(mol.hf_reference)}")

    print(
        "\n  note: symmetric benzene AVAS orbitals can rotate within the active "
        "pi subspace across independent PySCF builds; elementwise two-body "
        "tensor rows are therefore warnings, with Frobenius-norm invariants required."
    )
    rows = common.molecule_comparison_rows(
        molecules["pyscf"],
        molecules["pyscf_dump"],
        rotation_relaxed_two_body=True,
    )
    common.print_check_table("PySCF dump vs direct PySCF AVAS", rows)

    print("\nInformal benzene AVAS PySCF dump molecule parity check passed.")


if __name__ == "__main__":
    main()
