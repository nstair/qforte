import numpy as np
import qforte as qf


def normalized_random_state(nbasis, seed):
    rng = np.random.default_rng(seed)
    coeff = rng.normal(size=nbasis) + 1.0j * rng.normal(size=nbasis)
    coeff /= np.linalg.norm(coeff)
    return [complex(x) for x in coeff]


def coeff_vec(computer):
    return np.array(computer.get_coeff_vec(), dtype=np.complex128)


def assert_qnp_or_matches_two_givens(qubits, theta, seed):
    nqubit = 6
    coeff = normalized_random_state(2**nqubit, seed)
    q1, q2, q3, q4 = qubits

    qnp_or = qf.gate("QNP_OR", q1, q2, q3, q4, theta)

    direct = qf.Computer(nqubit)
    safe = qf.Computer(nqubit)
    reference = qf.Computer(nqubit)
    direct.set_coeff_vec(coeff)
    safe.set_coeff_vec(coeff)
    reference.set_coeff_vec(coeff)

    direct.apply_gate(qnp_or)
    safe.apply_gate_safe(qnp_or)

    reference.apply_gate(qf.gate("Givens", q3, q1, theta))
    reference.apply_gate(qf.gate("Givens", q4, q2, theta))

    direct_diff = np.linalg.norm(coeff_vec(direct) - coeff_vec(reference))
    safe_diff = np.linalg.norm(coeff_vec(safe) - coeff_vec(reference))

    print(f"qubits = {qubits}, direct ||dC|| = {direct_diff:.3e}")
    print(f"qubits = {qubits}, safe   ||dC|| = {safe_diff:.3e}")

    assert qnp_or.nqubits() == 4
    assert qnp_or.qubits() == list(qubits)
    assert direct_diff < 1.0e-12
    assert safe_diff < 1.0e-12


if __name__ == "__main__":
    theta = 0.617
    assert_qnp_or_matches_two_givens((0, 1, 2, 3), theta, seed=7)
    assert_qnp_or_matches_two_givens((3, 0, 5, 2), theta, seed=11)
    print("QNP_OR fock Computer checks passed.")
