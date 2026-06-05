"""Small H4 VQE sanity runs for pool-construction changes.

This is intentionally a plain sandbox script.  It is useful after changing
pool membership/order logic because it exercises SD, GSD, and 1-UpCCGSD with
the default and particle-hole-first generalized ordering.
"""

import numpy as np
import qforte as qf


POOL_CONFIGS = [
    {"name": "SD", "pool_type": "SD", "general_ex_pool_order": "default"},
    {"name": "GSD", "pool_type": "GSD", "general_ex_pool_order": "default"},
    {
        "name": "GSD_ph_first",
        "pool_type": "GSD",
        "general_ex_pool_order": "particle_hole_first",
    },
    {
        "name": "1-UpCCGSD",
        "pool_type": "1-UpCCGSD",
        "general_ex_pool_order": "default",
    },
    {
        "name": "1-UpCCGSD_ph_first",
        "pool_type": "1-UpCCGSD",
        "general_ex_pool_order": "particle_hole_first",
    },
]
MAXITER = 50


def build_h4_square(rhh=2.0):
    return qf.system_factory(
        system_type="molecule",
        build_type="psi4",
        basis="sto-6g",
        mol_geometry=[
            ("H", (0.0, -rhh / 2.0, -rhh / 2.0)),
            ("H", (0.0, -rhh / 2.0, +rhh / 2.0)),
            ("H", (0.0, +rhh / 2.0, -rhh / 2.0)),
            ("H", (0.0, +rhh / 2.0, +rhh / 2.0)),
        ],
        symmetry="c1",
        multiplicity=1,
        charge=0,
        num_frozen_docc=0,
        num_frozen_uocc=0,
        run_mp2=True,
        run_fci=True,
        store_mo_ints=True,
    )


def final_result_value(alg, name, default=None):
    result = getattr(alg, "_final_result", None)
    return getattr(result, name, default)


def run_pool(mol, config):
    alg = qf.UCCNVQE(
        mol,
        computer_type="fci",
        apply_ham_as_tensor=True,
        verbose=False,
    )

    alg.run(
        pool_type=config["pool_type"],
        optimizer="lbfgs_qf",
        opt_thresh=1.0e-4,
        opt_ftol=1.0e-8,
        opt_maxiter=MAXITER,
        use_analytic_grad=True,
        init_amps="zero",
        general_ex_pool_order=config["general_ex_pool_order"],
        lbfgs_qf_memory=8,
        lbfgs_qf_use_hessian_diag=False,
        lbfgs_qf_use_newton_cg=False,
        lbfgs_qf_use_target_block=False,
    )

    amps = np.asarray(alg._tamps, dtype=float)
    return {
        "pool": config["name"],
        "n_ops": len(alg._pool_obj),
        "energy": alg.get_gs_energy(),
        "grad_norm": final_result_value(alg, "grad_norm", None),
        "nit": final_result_value(alg, "nit", None),
        "nfev": final_result_value(alg, "nfev", None),
        "nonzero": int(np.count_nonzero(np.abs(amps) > 1.0e-12)),
    }


def print_summary(rows, mol):
    print("\n== H4 pool energy sanity summary ==")
    print(f"RHF energy: {mol.hf_energy:+.12f}")
    print(f"FCI energy: {mol.fci_energy:+.12f}")
    print()
    print(
        f"{'pool':14s} {'n_ops':>6s} {'energy':>16s} {'dE_HF':>12s} "
        f"{'||g||':>12s} {'nit':>6s} {'nfev':>6s} {'nnz':>6s}"
    )
    print("-" * 92)
    for row in rows:
        grad = row["grad_norm"]
        grad_text = f"{grad:.3e}" if grad is not None else "-"
        print(
            f"{row['pool']:14s} {row['n_ops']:6d} {row['energy']:+16.10f} "
            f"{row['energy'] - mol.hf_energy:+12.3e} {grad_text:>12s} "
            f"{str(row['nit']):>6s} {str(row['nfev']):>6s} {row['nonzero']:6d}"
        )

    by_pool = {row["pool"]: row for row in rows}
    for base, xname in [("GSD", "GSD_ph_first"), ("1-UpCCGSD", "1-UpCCGSD_ph_first")]:
        if base in by_pool and xname in by_pool:
            diff = abs(by_pool[base]["energy"] - by_pool[xname]["energy"])
            print(f"|E({base}) - E({xname})| = {diff:.3e}")


def main():
    mol = build_h4_square()
    rows = []
    for config in POOL_CONFIGS:
        print(f"\nRunning pool {config['name']} ...")
        rows.append(run_pool(mol, config))
    print_summary(rows, mol)

    for row in rows:
        if row["energy"] >= mol.hf_energy:
            raise AssertionError(f"{row['pool']} did not lower the RHF energy.")


if __name__ == "__main__":
    main()
