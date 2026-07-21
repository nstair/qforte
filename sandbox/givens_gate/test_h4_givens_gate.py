import numpy as np
import qforte as qf

def prepare_hartree_fock_state(ref):
    """Prepare the qubit Hartree-Fock determinant on a qforte Computer."""
    computer = qf.Computer(len(ref))
    for qubit, occupied in enumerate(ref):
        if occupied:
            computer.apply_gate(qf.gate("X", qubit))
    return computer

def print_nonzero_amplitudes(computer, label, threshold=1.0e-12):
    coeffs = computer.get_coeff_vec()
    nqubits = int(np.log2(len(coeffs)))

    print(f"\n{label}")
    print("=" * len(label))
    for idx, coeff in enumerate(coeffs):
        if abs(coeff) > threshold:
            bitstring = format(idx, f"0{nqubits}b")[::-1]
            print(f"|{bitstring}>  {coeff.real:+.12f} {coeff.imag:+.12f}i")
            
def apply_givens(computer, q1, q2, theta):
    target = q1
    control = q2
    
    givens = qf.gate("Givens", target, control, theta)
    computer.apply_gate(givens)
    
def apply_qnp_or(computer, p, q, phi):
    # alpha qubits
    qa_p = 2 * p # control
    qa_q = 2 * q # target
    
    # beta qubits
    qb_p = 2*p + 1
    qb_q = 2*q + 1
    
    apply_givens(computer, qa_q, qa_p, phi)
    apply_givens(computer, qb_q, qb_p, phi)
    
def apply_qnp_or_same_spin(computer, p, q, r, s, phi):
    apply_givens(computer, r, p, phi)
    apply_givens(computer, s, q, phi)
    
def apply_white_cnot(computer, target_q, control_q):
    
    cnot = qf.gate("cX", target_q, control_q)
    x = qf.gate("X", control_q)
    
    computer.apply_gate(x)
    computer.apply_gate(cnot)
    computer.apply_gate(x)
    
def apply_qnp_px1(computer, q1, q2, q3, q4, theta):
    apply_white_cnot(computer, q1, q2) # q1 is target
    apply_white_cnot(computer, q4, q3) # q4 is target
    
    apply_givens(computer, q3, q2, theta)
        
    apply_white_cnot(computer, q1, q2)
    apply_white_cnot(computer, q4, q3)

def apply_qnp_px2(computer, p, q, theta):
    """Apply a QNP pair-exchange rotation between spatial orbitals p and q."""
    qa_p = 2 * p
    qb_p = 2 * p + 1
    qa_q = 2 * q
    qb_q = 2 * q + 1

    mask_p = (1 << qa_p) | (1 << qb_p)
    mask_q = (1 << qa_q) | (1 << qb_q)

    coeffs = np.asarray(computer.get_coeff_vec(), dtype=complex)
    new_coeffs = coeffs.copy()
    c = np.cos(0.5 * theta)
    s = np.sin(0.5 * theta)

    for idx in range(len(coeffs)):
        p_has_pair = (idx & mask_p) == mask_p
        q_is_empty = (idx & mask_q) == 0

        if p_has_pair and q_is_empty:
            pair_exchanged_idx = (idx ^ mask_p) ^ mask_q
            p_pair_amp = coeffs[idx]
            q_pair_amp = coeffs[pair_exchanged_idx]

            new_coeffs[idx] = c * p_pair_amp + s * q_pair_amp
            new_coeffs[pair_exchanged_idx] = -s * p_pair_amp + c * q_pair_amp

    computer.set_coeff_vec(list(new_coeffs))
    
def apply_q(computer, p, q, q1, q2, q3, q4, phi, theta):
    apply_qnp_px1(computer, q1, q2, q3, q4, theta)
    apply_qnp_or(computer, p,  q, phi)

# def determinant_index(bitstring):
#     idx = 0
#     for qubit, occupied in enumerate(bitstring):
#         if occupied:
#             idx |= 1 << qubit
#     return idx

# def apply_UCC(computer, ref, cre_ops, ann_ops, theta, qubits=None):
#     """Apply one dUCC factor exp(theta * kappa_mu) from the Jacobi notes."""
#     if qubits is None:
#         qubits = list(range(computer.get_nqubit()))

#     if qubits != list(range(computer.get_nqubit())):
#         raise ValueError("apply_UCC currently builds the full-register dUCC matrix.")

#     sq = qf.SQOperator()
#     sq.add_term(1.0, cre_ops, ann_ops)

#     excited_comp = prepare_hartree_fock_state(ref)
#     excited_comp.apply_sq_operator(sq)

#     ref_idx = determinant_index(ref)
#     excited_coeffs = np.asarray(excited_comp.get_coeff_vec(), dtype=complex)
#     excited_idxs = np.flatnonzero(np.abs(excited_coeffs) > 1.0e-12)

#     if len(excited_idxs) != 1:
#         raise ValueError(
#              "The supplied creation/annihilation operators must generate exactly "
#             "one excited determinant from ref."
#         )

#     excited_idx = int(excited_idxs[0])
#     phase = excited_coeffs[excited_idx]
#     if abs(abs(phase) - 1.0) > 1.0e-12:
#         raise ValueError(f"Excitation generated non-unit phase/amplitude {phase}.")

#     nstates = 2 ** computer.get_nqubit()
#     ucc_matrix = np.eye(nstates, dtype=complex)
#     c = np.cos(theta)
#     s = np.sin(theta)

#     ucc_matrix[ref_idx, ref_idx] = c
#     ucc_matrix[excited_idx, ref_idx] = phase * s
#     ucc_matrix[ref_idx, excited_idx] = -np.conj(phase) * s
#     ucc_matrix[excited_idx, excited_idx] = c

#     computer.apply_matrix(ucc_matrix, qubits)

def main():
    r = 1.0
    
    geom = [
        ("H", (0.0, 0.0, 1.0*r)),
        ("H", (0.0, 0.0, 2.0*r)),
        ("H", (0.0, 0.0, 3.0*r)),
        ("H", (0.0, 0.0, 4.0*r)),
    ]

    mol = qf.system_factory(
        build_type="psi4",
        mol_geometry=geom,
        basis="sto-3g",
        build_qb_ham=False,
        run_fci=False,
        store_mo_ints=False,
    )
    
    # set-up HF
    ref = mol.hf_reference
    print(f"H4 HF reference: {ref}")
    
    hf_comp = prepare_hartree_fock_state(ref)
    print_nonzero_amplitudes(hf_comp, "Hartree-Fock state")
    
    phi = np.linspace(0, 2*np.pi, 7)
    # theta = np.linspace(0, 2*np.pi, 7)
    p = 1
    q = 2
    r = 4
    s = 6
    
    q1 = 2
    q2 = 3
    q3 = 4
    q4 = 5
    
    for angle in phi:
        # phi = np.pi
        theta = np.pi
        
        # givens_comp = prepare_hartree_fock_state(ref)
        # apply_givens(givens_comp, q1, q3, angle)
        
        # print(
        #     f"\nApplied givens(phi={angle}) "
        #     f"between qubits q1={q1} and q2={q3}"
        # )

        # print_nonzero_amplitudes(givens_comp, "After givens rotation")
        # print("Norm Givens:", np.linalg.norm(givens_comp.get_coeff_vec()))
        
        # qnp_or_comp = prepare_hartree_fock_state(ref)
        # apply_qnp_or(qnp_or_comp, p, q, angle)
        # # apply_qnp_or_same_spin(qnp_or_comp, p, q, r, s, angle)
        
        # print(
        #     f"\nApplied QNP_OR(phi={angle}) "
        #     f"between spatial orbitals p={p} and q={q}"
        # )

        # print_nonzero_amplitudes(qnp_or_comp, "After QNP_OR rotation")
        # print("Norm QNP_OR:", np.linalg.norm(qnp_or_comp.get_coeff_vec()))
        
        # qnp_px_comp = prepare_hartree_fock_state(ref)
        # apply_qnp_px1(qnp_px_comp, q1, q2, q3, q4, angle)
        # # apply_qnp_px2(qnp_px_comp, p, q, angle)
        
        # print(
        #     f"\nApplied QNP_PX(phi={angle}) "
        #     f"between qubits q1={q1}, q2={q2}, q3={q3}, and q4={q4}"
        #     # f"between spatial orbitals p={p} and q={q}"
        # )

        # print_nonzero_amplitudes(qnp_px_comp, "After QNP_PX rotation")
        # print("Norm QNP_PX:", np.linalg.norm(qnp_px_comp.get_coeff_vec()))
        
        q_comp = prepare_hartree_fock_state(ref)
        apply_q(q_comp, p, q, q1, q2, q3, q4, angle, theta)
        
        print(
            f"\nApplied Q(phi={angle}, theta={theta}) "
            f"with QNP_px btw spatial orbitals p={p} and q={q} and QNP_or btw qubits q1={q1}, q2={q2}, q3={q3}, and q4={q4}"
        )
        
        print_nonzero_amplitudes(q_comp, "After Q rotation")
        print("Norm Q:", np.linalg.norm(q_comp.get_coeff_vec()))
            
    # dUCC Unitary applied
    # ucc_comp = prepare_hartree_fock_state(ref)
    # apply_UCC(ucc_comp, ref, )
    
    # theta = np.pi/4
    
    # # choose one occupied(particle) and one virtual(hole) orbital
    # occ = [i for i, x in enumerate(ref) if x]
    # vir = [i for i, x in enumerate(ref) if not x]

    # i = occ[0]
    # a = vir[0]

    # # alpha spin excitation
    # sq = qf.SQOperator()
    # sq.add_term(+theta, [a], [i])
    # sq.add_term(-theta, [i], [a])

    # # qforte applies exp(SQOperator) internally when used as a gate
    # ucc_gate = qf.gate("SQOperator", sq)

    # ucc_comp.apply_gate(ucc_gate)

    # print(f"\nApplied UCC single excitation {i} -> {a}")
    # print_nonzero_amplitudes(ucc_comp, "After UCC singles")

    # normalization check
    # print("\nNorm HF :", np.linalg.norm(hf_comp.get_coeff_vec()))
    # print("Norm QNP:", np.linalg.norm(qnp_comp.get_coeff_vec()))
    # print("Norm UCC:", np.linalg.norm(ucc_comp.get_coeff_vec()))

if __name__ == "__main__":
    main()
