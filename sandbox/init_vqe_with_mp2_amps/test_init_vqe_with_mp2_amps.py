"""Small diagnostic for MP2-initialized UCCN-VQE amplitudes.

This is intentionally a readable sandbox script rather than a formal unit test.
It builds a tiny tUCC calculation twice, once with all-zero amplitudes and once
with particle-hole double excitations initialized from MP2 amplitudes.
"""

import numpy as np
import qforte as qf


def build_h2_molecule():
    geom = [
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 0.75)),
    ]

    return qf.system_factory(
        build_type="psi4",
        mol_geometry=geom,
        basis="sto-3g",
        symmetry="c1",
        run_fci=True,
        store_mo_ints=True,
    )


def make_alg(mol):
    return qf.UCCNVQE(
        mol,
        computer_type="fci",
        apply_ham_as_tensor=True,
        verbose=False,
    )


def initialize_only(alg, init_amps):
    alg.run(
        pool_type="SD",
        optimizer="lbfgs_qf",
        opt_maxiter=0,
        use_analytic_grad=True,
        init_amps=init_amps,
    )
    return alg.energy_feval(alg._tamps)


def print_largest_mp2_amplitudes(alg, n=8):
    records = sorted(
        getattr(alg, "_mp2_init_records", []),
        key=lambda rec: abs(rec["amplitude"]),
        reverse=True,
    )

    print(f"\nLargest {min(n, len(records))} MP2 amplitudes")
    print("----------------------------------------")
    for rec in records[:n]:
        print(
            f"mu={rec['mu']:4d}  top={rec['top']:4d}  "
            f"amp={rec['amplitude']:+.10e}  "
            f"cre={rec['creators']}  ann={rec['annihilators']}  "
            f"denom={rec['denominator']:+.10e}"
        )


def main():
    mol = build_h2_molecule()

    zero_alg = make_alg(mol)
    E_zero = initialize_only(zero_alg, "zero")

    mp2_alg = make_alg(mol)
    E_mp2 = initialize_only(mp2_alg, "mp2")

    tamps = np.asarray(mp2_alg._tamps, dtype=float)
    nonzero = np.flatnonzero(np.abs(tamps) > 1.0e-12)
    records = getattr(mp2_alg, "_mp2_init_records", [])

    print("\n== MP2 initial-amplitude diagnostic ==")
    print(f"Number of pool operators:      {len(mp2_alg._pool_obj)}")
    print(f"Nonzero MP2 amplitudes:        {len(nonzero)}")
    print(f"Recorded MP2 doubles:          {len(records)}")
    print(f"E_zero:                        {E_zero:+.12f}")
    print(f"E_mp2:                         {E_mp2:+.12f}")
    print(f"E_mp2 - E_zero:                {E_mp2 - E_zero:+.12f}")

    if not np.all(np.isfinite(tamps)):
        raise AssertionError("MP2 amplitudes contain NaN or inf.")
    if len(nonzero) != len(records):
        raise AssertionError("Found nonzero amplitudes outside recorded MP2 doubles.")
    if E_mp2 >= E_zero:
        raise AssertionError("MP2-initialized state did not lower the starting energy.")

    print_largest_mp2_amplitudes(mp2_alg)

    print("\nScaling check")
    print("-------------")
    mp2_tamps = np.asarray(mp2_alg._tamps, dtype=float)
    for scale in [0.0, 0.25, 0.5, 1.0]:
        E_scale = mp2_alg.energy_feval(scale * mp2_tamps)
        print(f"scale={scale:4.2f}  E={E_scale:+.12f}")


if __name__ == "__main__":
    main()
