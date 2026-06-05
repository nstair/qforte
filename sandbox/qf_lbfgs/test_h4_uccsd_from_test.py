"""Plain-script version of the H4 UCCSD solver comparison from tests/test_uccsd.py."""

from qforte import system_factory, UCCNPQE


def build_h4_square(rhh=2.0):
    return system_factory(
        system_type="molecule",
        build_type="psi4",
        basis="sto-6g",
        mol_geometry=[
            ("H", (0, -rhh / 2, -rhh / 2)),
            ("H", (0, -rhh / 2, +rhh / 2)),
            ("H", (0, +rhh / 2, -rhh / 2)),
            ("H", (0, +rhh / 2, +rhh / 2)),
        ],
        symmetry="d2h",
        multiplicity=1,
        charge=0,
        num_frozen_docc=0,
        num_frozen_uocc=0,
        run_mp2=1,
        run_ccsd=0,
        run_cisd=0,
        run_fci=1,
    )


def run_jacobi(mol):
    alg = UCCNPQE(
        mol,
        compact_excitations=True,
        qubit_excitations=False,
        diis_max_dim=8,
    )
    alg.run(optimizer="jacobi", pool_type="SD")
    return alg


def run_bfgs(mol):
    alg = UCCNPQE(
        mol,
        compact_excitations=True,
        qubit_excitations=False,
    )
    alg.run(optimizer="BFGS", pool_type="SD")
    return alg


def main():
    mol = build_h4_square(rhh=2.0)

    jacobi = run_jacobi(mol)
    bfgs = run_bfgs(mol)

    jacobi_energy = jacobi.get_gs_energy()
    bfgs_energy = bfgs.get_gs_energy()
    diff = abs(jacobi_energy - bfgs_energy)

    print("\n== H4 UCCSD solver comparison ==")
    print(f"Rhh:             {2.0:.6f}")
    print(f"RHF energy:      {mol.hf_energy:+.12f}")
    print(f"MP2 energy:      {mol.mp2_energy:+.12f}")
    print(f"FCI energy:      {mol.fci_energy:+.12f}")
    print(f"Jacobi energy:   {jacobi_energy:+.12f}")
    print(f"BFGS energy:     {bfgs_energy:+.12f}")
    print(f"|Jacobi-BFGS|:   {diff:.3e}")

    if diff > 1.0e-8:
        raise AssertionError(f"Jacobi and BFGS differ by {diff:.3e}")


if __name__ == "__main__":
    main()
