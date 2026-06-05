"""Compare UCC pool-ordering conventions on a small fixed problem.

This is a sandbox diagnostic, not a formal unit test. The pool string controls
which operators are included; the ordering options below only change the
factorized UCC product order.
"""

import numpy as np
import qforte as qf
# from sandbox.qf_lbfgs.test_lbfgs_qf_simple import USE_NEWTON_CG


# POOL_TYPE = "SD"
POOL_TYPE = "1-UpCCGSD"
INIT_AMPS = "zero"  # "zero", "mp2"
OPT_MAXITER = 500

ORDERINGS = [
    {
        "name": "default",
        "primary_pool_order": "none",
        "secondary_pool_order": "shell",
        "general_ex_pool_order": "particle_hole_first",
    },
    {
        "name": "shell",
        "primary_pool_order": "none",
        "secondary_pool_order": "shell",
        "general_ex_pool_order": "default",
    },
    {
        "name": "mp2_shell",
        "primary_pool_order": "mp2_amps",
        "secondary_pool_order": "shell",
        "general_ex_pool_order": "default",
    },
    {
        "name": "gradient_shell",
        "primary_pool_order": "gradients",
        "secondary_pool_order": "shell",
        "general_ex_pool_order": "default",
    },
    {
        "name": "gradient_shell_ph_first",
        "primary_pool_order": "gradients",
        "secondary_pool_order": "shell",
        "general_ex_pool_order": "particle_hole_first",
    },
]


def build_molecule():
    # geom = [
    #     ("H", (0.0, 0.0, 0.0)),
    #     ("H", (0.0, 0.0, 1.0)),
    #     ("H", (0.0, 0.0, 2.0)),
    #     ("H", (0.0, 0.0, 3.0)),
    #     ("H", (0.0, 0.0, 4.0)),
    #     ("H", (0.0, 0.0, 5.0)),
    # ]

    geom = [
        ("H", (0.0, 0.0, -1.0)),
        ("Be", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0,  1.0)),
    ]

    return qf.system_factory(
        build_type="psi4",
        mol_geometry=geom,
        basis="sto-3g",
        symmetry="c1",
        run_fci=True,
        run_mp2=True,
        store_mo_ints=True,
    )

# -15.6483581350

def first_operator_line(alg, top):
    text = str(alg._pool_obj[top][1])
    return text.splitlines()[0] if text else "<empty operator>"


def print_first_ordered_ops(alg, n=10):
    print(f"\nFirst {min(n, len(alg._tops))} ordered pool operators")
    print("----------------------------------------")
    for mu, top in enumerate(alg._tops[:n]):
        print(f"mu={mu:3d}  top={top:4d}  {first_operator_line(alg, top)}")


def final_result_value(result, name, default=None):
    return getattr(result, name, default)


def run_one_ordering(mol, ordering):
    alg = qf.UCCNVQE(
        mol,
        computer_type="fci",
        apply_ham_as_tensor=True,
        verbose=False,
    )

    alg.run(
        pool_type=POOL_TYPE,
        optimizer="bfgs",
        opt_thresh=1.0e-4,
        opt_ftol=1.0e-8,
        opt_maxiter=OPT_MAXITER,
        use_analytic_grad=True,
        init_amps=INIT_AMPS,
        lbfgs_qf_use_hessian_diag=False,
        lbfgs_qf_hdiag_mode="abs",
        lbfgs_qf_memory=12,
        lbfgs_qf_use_newton_cg=False,
        lbfgs_qf_newton_cg_reset_lbfgs_history = False,    
        **{k: v for k, v in ordering.items() if k != "name"},
    )

    tamps = np.asarray(alg._tamps, dtype=float)
    result = getattr(alg, "_final_result", None)
    row = {
        "name": ordering["name"],
        "energy": alg.get_gs_energy(),
        "nit": final_result_value(result, "nit", None),
        "nfev": final_result_value(result, "nfev", None),
        "njev": final_result_value(result, "njev", None),
        "grad_norm": final_result_value(result, "grad_norm", None),
        "nonzero": int(np.count_nonzero(np.abs(tamps) > 1.0e-12)),
        "first_tops": list(alg._tops[:10]),
    }

    print(f"\n== {ordering['name']} ==")
    print_first_ordered_ops(alg)
    print(f"Final energy:       {row['energy']:+.12f}")
    print(f"Iterations:         {row['nit']}")
    print(f"Energy evals:       {row['nfev']}")
    print(f"Gradient evals:     {row['njev']}")
    print(f"Final ||g||:        {row['grad_norm']}")
    print(f"Nonzero amplitudes: {row['nonzero']}")

    return row


def print_summary(rows, mol):
    print("\n== Pool-ordering summary ==")
    print(f"Pool type: {POOL_TYPE}")
    print(f"Initial amplitudes: {INIT_AMPS}")
    print(f"FCI energy: {mol.fci_energy:+.12f}")
    print()
    print(
        f"{'ordering':28s} {'energy':>16s} {'dE_fci':>12s} "
        f"{'nit':>6s} {'nfev':>6s} {'njev':>6s} {'||g||':>12s} {'nnz':>6s}"
    )
    print("-" * 100)
    for row in rows:
        dE = row["energy"] - mol.fci_energy
        grad_norm = row["grad_norm"]
        grad_text = f"{grad_norm:.3e}" if grad_norm is not None else "-"
        print(
            f"{row['name']:28s} {row['energy']:+16.10f} {dE:+12.3e} "
            f"{str(row['nit']):>6s} {str(row['nfev']):>6s} "
            f"{str(row['njev']):>6s} {grad_text:>12s} {row['nonzero']:6d}"
        )


def main():
    mol = build_molecule()
    rows = [run_one_ordering(mol, ordering) for ordering in ORDERINGS]
    print_summary(rows, mol)


if __name__ == "__main__":
    main()

# == Pool-ordering summary ==
# Pool type: 1-UpCCGSD
# Initial amplitudes: zero
# FCI energy: -15.481741069878

# ordering                               energy       dE_fci    nit   nfev   njev        ||g||    nnz
# ----------------------------------------------------------------------------------------------------
# shell                          -15.4792067400   +2.534e-03    311    328    328            -     63
# mp2_shell                      -15.4792144660   +2.527e-03    295    313    313            -     63
# gradient_shell                 -15.4809057106   +8.354e-04    412    428    428            -     63
# gradient_shell_ph_first        -15.4809057106   +8.354e-04    412    428    428            -     63

# == Pool-ordering summary ==
# Pool type: 1-UpCCGSD
# Initial amplitudes: zero
# FCI energy: -15.481741069878

# ordering                               energy       dE_fci    nit   nfev   njev        ||g||    nnz
# ----------------------------------------------------------------------------------------------------
# shell                          -15.4792258090   +2.515e-03    389    407    407            -     63
# mp2_shell                      -15.4792136204   +2.527e-03    300    322    322            -     63
# gradient_shell                 -15.4792262113   +2.515e-03    384    403    403            -     63
# gradient_shell_ph_first        -15.4792262113   +2.515e-03    384    403    403            -     63