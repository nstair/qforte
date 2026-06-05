#!/usr/bin/env python3
"""Generate a qforte-readable PySCF dump for N2/STO-3G."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyscf import ao2mo, fci, gto, scf

from pyscf_dump_utils import (
    closed_shell_hf_reference,
    geometry_to_pyscf_atom,
    write_pyscf_dump,
)


THIS_DIR = Path(__file__).resolve().parent
DUMP_PATH = THIS_DIR / "dumps" / "n2_sto3g_pyscf_dump.npz"

BASIS = "sto-3g"
CHARGE = 0
SPIN = 0
GEOMETRY = [
    ("N", (0.0, 0.0, -0.5488)),
    ("N", (0.0, 0.0, 0.5488)),
]


def main() -> None:
    mol = gto.Mole()
    mol.build(
        atom=geometry_to_pyscf_atom(GEOMETRY),
        basis=BASIS,
        charge=CHARGE,
        spin=SPIN,
        unit="Angstrom",
        symmetry=False,
        verbose=4,
    )

    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("N2 RHF did not converge.")

    C = mf.mo_coeff
    hcore = mf.get_hcore()
    mo_oeis = C.T @ hcore @ C
    nmo = int(C.shape[1])
    mo_teis = ao2mo.kernel(mol, C, compact=False).reshape(nmo, nmo, nmo, nmo)

    cisolver = fci.FCI(mf)
    fci_energy, _ = cisolver.kernel()

    nalpha, nbeta = mol.nelec
    nelec = int(nalpha + nbeta)
    scalar_energy = float(mol.energy_nuc())
    metadata = {
        "system_label": "n2_sto3g",
        "source": "pyscf",
        "mol_geometry": GEOMETRY,
        "basis": BASIS,
        "charge": CHARGE,
        "multiplicity": SPIN + 1,
        "num_active_orbitals": nmo,
        "num_active_electrons": nelec,
        "num_alpha": int(nalpha),
        "num_beta": int(nbeta),
        "hf_reference": closed_shell_hf_reference(nelec, nmo),
        "scalar_energy": scalar_energy,
        "nuclear_repulsion_energy": scalar_energy,
        "frozen_core_energy": 0.0,
        "hf_energy": float(mf.e_tot),
        "fci_energy_list": [float(np.real(fci_energy))],
        "hf_orbital_energies": [float(eps) for eps in mf.mo_energy],
        "point_group": "c1",
        "irreps": ["A"],
        "orb_irreps": ["A"] * nmo,
        "orb_irreps_to_int": [0] * nmo,
        "notes": "Full-valence N2/STO-3G RHF molecular-orbital dump.",
    }

    write_pyscf_dump(DUMP_PATH, mo_oeis, mo_teis, metadata)

    print("\nWrote PySCF dump")
    print(f"  npz:      {DUMP_PATH}")
    print(f"  FCIDUMP:  {DUMP_PATH.with_suffix('.FCIDUMP')}")
    print(f"  RHF:      {float(mf.e_tot):+.12f}")
    print(f"  FCI:      {float(np.real(fci_energy)):+.12f}")
    print(f"  active:   ({nelec}e, {nmo}o)")


if __name__ == "__main__":
    main()
