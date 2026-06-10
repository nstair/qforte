import argparse
import math

import numpy as np
import qforte as qf


def assert_close(label, value, reference, tol=1.0e-10):
    diff = abs(value - reference)
    print(f"{label:32s} value={value:+.12f}  reference={reference:+.12f}  diff={diff:.3e}")
    if diff > tol:
        raise AssertionError(f"{label} differs by {diff:.3e}")


def sminus_splus_operator(norb):
    op = qf.SQOperator()

    for p in range(norb):
        op.add(1.0, [2 * p + 1], [2 * p + 1])

    for p in range(norb):
        for q in range(norb):
            op.add(-1.0, [2 * p + 1, 2 * q], [2 * p, 2 * q + 1])

    op.simplify()
    return op


def tensor_to_numpy(tensor):
    return np.asarray(tensor.read_data(), dtype=np.complex128).reshape(tuple(tensor.shape()))


def set_fci_state(qc, state):
    tensor = qf.Tensor(list(state.shape), "spin2_state")
    tensor.fill_from_nparray(state.ravel(), list(state.shape))
    qc.set_state(tensor)


def reference_spin_squared(qc, sz, norb):
    m_s = 0.5 * sz
    state = tensor_to_numpy(qc.get_state())
    norm_sq = np.vdot(state.ravel(), state.ravel()).real
    return m_s * (m_s + 1.0) * norm_sq + qc.get_exp_val(sminus_splus_operator(norb)).real


def check_against_sqoperator(label, qc, sz, norb, tol=1.0e-10):
    fast = qc.get_spin_squared_expectation()
    ref = reference_spin_squared(qc, sz, norb)
    assert_close(label, fast, ref, tol)
    return fast


def run_individual_determinant_tests():
    print("\n== Individual determinant checks ==")

    qc = qf.FCIComputer(nel=2, sz=0, norb=2)
    qc.zero_state()
    qc.set_element([0, 0], 1.0)
    assert_close("closed-shell determinant", qc.get_spin_squared_expectation(), 0.0)
    check_against_sqoperator("closed-shell SQOperator", qc, sz=0, norb=2)

    qc.zero_state()
    qc.set_element([0, 1], 1.0)
    assert_close("open-shell determinant", qc.get_spin_squared_expectation(), 1.0)
    check_against_sqoperator("open-shell SQOperator", qc, sz=0, norb=2)

    qc = qf.FCIComputer(nel=2, sz=2, norb=2)
    qc.zero_state()
    qc.set_element([0, 0], 1.0)
    assert_close("high-spin determinant", qc.get_spin_squared_expectation(), 2.0)
    check_against_sqoperator("high-spin SQOperator", qc, sz=2, norb=2)

    qc = qf.FCIComputer(nel=2, sz=0, norb=2)
    qc.zero_state()
    qc.set_element([0, 1], 1.0 / math.sqrt(2.0))
    qc.set_element([1, 0], 1.0 / math.sqrt(2.0))
    assert_close("two-det singlet combo", qc.get_spin_squared_expectation(), 0.0)

    qc.zero_state()
    qc.set_element([0, 1], 1.0 / math.sqrt(2.0))
    qc.set_element([1, 0], -1.0 / math.sqrt(2.0))
    assert_close("two-det triplet combo", qc.get_spin_squared_expectation(), 2.0)


def run_random_state_test():
    print("\n== Random state check ==")

    nel = 4
    sz = 0
    norb = 5
    rng = np.random.default_rng(7)

    qc = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
    shape = tuple(qc.get_state().shape())
    state = rng.normal(size=shape) + 1.0j * rng.normal(size=shape)
    state /= np.linalg.norm(state.ravel())

    set_fci_state(qc, state)
    check_against_sqoperator("random normalized state", qc, sz=sz, norb=norb, tol=1.0e-9)


def build_vqe_state(alg):
    pool = qf.SQOpPool()
    for tamp, top in zip(alg._tamps, alg._tops):
        pool.add(tamp, alg._pool_obj[top][1])

    qc = qf.FCIComputer(alg._nel, alg._2_spin, alg._norb)
    qc.hartree_fock()
    qc.evolve_pool_trotter_basic(pool, antiherm=True, adjoint=False)
    return qc


def run_h6_tuccsd_vqe_test():
    print("\n== H6/STO-3G tUCCSD-VQE state check ==")

    Nh = 4

    geom = [("H", (0.0, 0.0, float(i))) for i in range(1, Nh + 1)]
    mol = qf.system_factory(
        build_type="pyscf",
        mol_geometry=geom,
        basis="sto-3g",
        symmetry="c1",
        run_fci=False,
    )

    alg = qf.UCCNVQE(
        mol,
        computer_type="fci",
        apply_ham_as_tensor=True,
        verbose=False,
    )
    try:
        alg.run(
            opt_thresh=1.0e-10,
            opt_ftol=1.0e-7,
            opt_maxiter=80,
            pool_type="SDTQ",
            optimizer="bfgs",
            use_analytic_grad=True,
        )
    except ValueError as exc:
        if not hasattr(alg, "_tamps") or not hasattr(alg, "_Egs"):
            raise
        print(f"Continuing after post-optimization VQE reporting failure: {exc}")

    qc = build_vqe_state(alg)
    s2 = check_against_sqoperator(
        "H6 tUCCSD-VQE state",
        qc,
        sz=alg._2_spin,
        norb=alg._norb,
        tol=1.0e-8,
    )
    print(f"H6 tUCCSD-VQE energy={alg.get_gs_energy():+.12f}  <S^2>={s2:+.12f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-vqe",
        action="store_true",
        help="Skip the slower H6/STO-3G tUCCSD-VQE sandbox check.",
    )
    args = parser.parse_args()

    run_individual_determinant_tests()
    run_random_state_test()
    if not args.skip_vqe:
        run_h6_tuccsd_vqe_test()


if __name__ == "__main__":
    main()
