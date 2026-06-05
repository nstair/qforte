#!/usr/bin/env python3
"""Generate a qforte-readable PySCF AVAS dump for naphthalene/cc-pVDZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pyscf import ao2mo, gto, mcscf, scf
from pyscf.mcscf import avas

from pyscf_dump_utils import (
    active_electron_count,
    active_spin_counts,
    closed_shell_hf_reference,
    geometry_to_pyscf_atom,
    parse_acene_geometry_file,
    write_pyscf_dump,
)


THIS_DIR = Path(__file__).resolve().parent
GEOMETRY_FILE = THIS_DIR.parent / "poly_acene_pyscf_calcs" / "acene_geoms.txt"
DUMP_PATH = THIS_DIR / "dumps" / "naphthalene_avas_ccpvdz_pyscf_dump.npz"

BASIS = "cc-pvdz"
ACENE = "naphthalene"
STATE = "singlet"
CHARGE = 0
SPIN = 0
AVAS_LABELS = ["C 2pz"]
AVAS_THRESHOLD = 0.2
AVAS_MINAO = "minao"
AVAS_WITH_IAO = False
AVAS_CANONICALIZE = True
EXPECT_ACTIVE = (10, 10)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-casci", action="store_true")
    parser.add_argument("--dump-path", type=Path, default=DUMP_PATH)
    parser.add_argument("--geometry-file", type=Path, default=GEOMETRY_FILE)
    parser.add_argument("--avas-labels", nargs="+", default=AVAS_LABELS)
    parser.add_argument("--avas-threshold", type=float, default=AVAS_THRESHOLD)
    parser.add_argument("--avas-minao", default=AVAS_MINAO)
    parser.add_argument("--avas-with-iao", action="store_true", default=AVAS_WITH_IAO)
    parser.add_argument(
        "--no-avas-canonicalize",
        dest="avas_canonicalize",
        action="store_false",
    )
    parser.set_defaults(avas_canonicalize=AVAS_CANONICALIZE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    geom = parse_acene_geometry_file(args.geometry_file, ACENE, STATE)

    mol = gto.Mole()
    mol.build(
        atom=geometry_to_pyscf_atom(geom),
        basis=BASIS,
        charge=CHARGE,
        spin=SPIN,
        unit="Angstrom",
        symmetry=False,
        verbose=4,
        max_memory=4000,
    )

    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-10
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Naphthalene RHF did not converge.")

    avas_obj = avas.AVAS(
        mf,
        args.avas_labels,
        threshold=args.avas_threshold,
        minao=args.avas_minao,
        with_iao=args.avas_with_iao,
        canonicalize=args.avas_canonicalize,
        verbose=4,
    )
    ncas, nelecas, mo_coeff = avas_obj.kernel()
    ncas = int(ncas)
    nelec_active = active_electron_count(nelecas)
    nalpha, nbeta = active_spin_counts(nelecas)

    if EXPECT_ACTIVE is not None and (nelec_active, ncas) != EXPECT_ACTIVE:
        raise RuntimeError(
            f"Unexpected AVAS active space: got ({nelec_active}e, {ncas}o), "
            f"expected ({EXPECT_ACTIVE[0]}e, {EXPECT_ACTIVE[1]}o)."
        )

    casci = mcscf.CASCI(mf, ncas, nelecas)
    casci.mo_coeff = mo_coeff
    h1eff, ecore = casci.get_h1eff()
    ncore = int(casci.ncore)
    C_cas = casci.mo_coeff[:, ncore:ncore + ncas]
    eri_cas = ao2mo.kernel(mf.mol, C_cas, compact=False).reshape(
        ncas,
        ncas,
        ncas,
        ncas,
    )

    fci_energy_list = []
    if args.run_casci:
        result = casci.kernel()
        fci_energy_list = [float(np.real(result[0]))]

    F = mf.get_fock()
    F_avas = mo_coeff.T @ F @ mo_coeff
    active_slice = slice(ncore, ncore + ncas)
    F_active = F_avas[active_slice, active_slice]
    mo_energy_active = np.linalg.eigvalsh(0.5 * (F_active + F_active.T))

    nuclear_repulsion = float(mol.energy_nuc())
    frozen_core_energy = float(ecore - nuclear_repulsion)
    scalar_energy = float(ecore)
    metadata = {
        "system_label": "naphthalene_avas_ccpvdz",
        "source": "pyscf_avas",
        "mol_geometry": geom,
        "basis": BASIS,
        "charge": CHARGE,
        "multiplicity": SPIN + 1,
        "avas_labels": args.avas_labels,
        "avas_threshold": args.avas_threshold,
        "avas_minao": args.avas_minao,
        "avas_with_iao": args.avas_with_iao,
        "avas_canonicalize": args.avas_canonicalize,
        "num_active_orbitals": ncas,
        "num_active_electrons": nelec_active,
        "num_alpha": nalpha,
        "num_beta": nbeta,
        "hf_reference": closed_shell_hf_reference(nelec_active, ncas),
        "scalar_energy": scalar_energy,
        "nuclear_repulsion_energy": nuclear_repulsion,
        "frozen_core_energy": frozen_core_energy,
        "hf_energy": float(mf.e_tot),
        "fci_energy_list": fci_energy_list,
        "hf_orbital_energies": [float(eps) for eps in mo_energy_active],
        "point_group": "c1",
        "irreps": ["A"],
        "orb_irreps": ["A"] * ncas,
        "orb_irreps_to_int": [0] * ncas,
        "notes": "Unrelaxed PySCF AVAS active-space dump targeting C 2pz.",
    }

    write_pyscf_dump(args.dump_path, h1eff, eri_cas, metadata)

    print("\nWrote PySCF AVAS dump")
    print(f"  npz:      {args.dump_path}")
    print(f"  FCIDUMP:  {args.dump_path.with_suffix('.FCIDUMP')}")
    print(f"  RHF:      {float(mf.e_tot):+.12f}")
    if fci_energy_list:
        print(f"  CASCI:    {fci_energy_list[0]:+.12f}")
    print(f"  active:   ({nelec_active}e, {ncas}o)")


if __name__ == "__main__":
    main()
