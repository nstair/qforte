import numpy as np
import qforte as qf


PAIR_0011 = 3
PAIR_1100 = 12


def normalized_random_state(nbasis, seed):
    rng = np.random.default_rng(seed)
    coeff = rng.normal(size=nbasis) + 1.0j * rng.normal(size=nbasis)
    coeff /= np.linalg.norm(coeff)
    return np.array(coeff, dtype=np.complex128)


def coeff_vec(computer):
    return np.array(computer.get_coeff_vec(), dtype=np.complex128)


def local_index(global_index, qubits):
    idx = 0
    for qubit in qubits:
        idx <<= 1
        idx += (global_index >> qubit) & 1
    return idx


def with_local_index(global_index, qubits, target_local_index):
    idx = int(global_index)
    nlocal = len(qubits)
    for local_pos, qubit in enumerate(qubits):
        bit = (target_local_index >> (nlocal - 1 - local_pos)) & 1
        if bit:
            idx |= 1 << qubit
        else:
            idx &= ~(1 << qubit)
    return idx


def apply_reference_qnp_px(coeff, qubits, theta):
    c = np.cos(0.5 * theta)
    s = np.sin(0.5 * theta)
    out = np.array(coeff, dtype=np.complex128, copy=True)

    for global_index in range(coeff.size):
        if local_index(global_index, qubits) != PAIR_0011:
            continue

        idx_0011 = global_index
        idx_1100 = with_local_index(global_index, qubits, PAIR_1100)
        amp_0011 = coeff[idx_0011]
        amp_1100 = coeff[idx_1100]

        out[idx_0011] = c * amp_0011 + s * amp_1100
        out[idx_1100] = -s * amp_0011 + c * amp_1100

    return out


def assert_qnp_px_matches_pair_block(qubits, theta, seed):
    nqubit = 6
    coeff = normalized_random_state(2**nqubit, seed)
    q1, q2, q3, q4 = qubits

    qnp_px = qf.gate("QNP_PX", q1, q2, q3, q4, theta)

    direct = qf.Computer(nqubit)
    safe = qf.Computer(nqubit)
    direct.set_coeff_vec([complex(x) for x in coeff])
    safe.set_coeff_vec([complex(x) for x in coeff])

    direct.apply_gate(qnp_px)
    safe.apply_gate_safe(qnp_px)

    reference = apply_reference_qnp_px(coeff, qubits, theta)
    direct_diff = np.linalg.norm(coeff_vec(direct) - reference)
    safe_diff = np.linalg.norm(coeff_vec(safe) - reference)

    print(f"qubits = {qubits}, direct ||dC|| = {direct_diff:.3e}")
    print(f"qubits = {qubits}, safe   ||dC|| = {safe_diff:.3e}")

    assert qnp_px.nqubits() == 4
    assert qnp_px.qubits() == list(qubits)
    assert direct_diff < 1.0e-12
    assert safe_diff < 1.0e-12


def assert_basis_action(qubits, theta):
    nqubit = 6
    c = np.cos(0.5 * theta)
    s = np.sin(0.5 * theta)
    qnp_px = qf.gate("QNP_PX", *qubits, theta)

    idx_0011 = with_local_index(0, qubits, PAIR_0011)
    idx_1100 = with_local_index(0, qubits, PAIR_1100)
    max_diff = 0.0

    for local in range(16):
        global_index = with_local_index(0, qubits, local)
        coeff = np.zeros(2**nqubit, dtype=np.complex128)
        coeff[global_index] = 1.0

        computer = qf.Computer(nqubit)
        computer.set_coeff_vec([complex(x) for x in coeff])
        computer.apply_gate(qnp_px)
        result = coeff_vec(computer)

        expected = coeff.copy()
        if local == PAIR_0011:
            expected[idx_0011] = c
            expected[idx_1100] = -s
        elif local == PAIR_1100:
            expected[idx_0011] = s
            expected[idx_1100] = c

        diff = np.linalg.norm(result - expected)
        max_diff = max(max_diff, diff)
        assert diff < 1.0e-12

    print(f"qubits = {qubits}, basis sweep max ||dC|| = {max_diff:.3e}")


if __name__ == "__main__":
    theta = 0.617
    assert_qnp_px_matches_pair_block((0, 1, 2, 3), theta, seed=13)
    assert_qnp_px_matches_pair_block((3, 0, 5, 2), theta, seed=17)
    assert_basis_action((0, 1, 2, 3), theta)
    assert_basis_action((3, 0, 5, 2), theta)
    print("QNP_PX fock Computer checks passed.")
