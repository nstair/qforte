import qforte as qf
import numpy as np

geom = [
   ('H', (0., 0., 1.0)),
   ('H', (0., 0., 2.0)),
   ('H', (0., 0., 3.0)),
   ('H', (0., 0., 4.0)),
]

mol = qf.system_factory(
   build_type='psi4',
   mol_geometry=geom,
   basis='sto-3g',
   build_qb_ham=False,
   run_fci=1)

ref = mol.hf_reference
nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

graph = qf.FCIGraph((nel + sz) // 2, nel - (nel + sz) // 2, norb)
Na = graph.get_lena()
Nb = graph.get_lenb()
R = min(Na, Nb)

rr = qf.RRFCIComputer(nel=nel, sz=sz, norb=norb, rank=R)
rr.hartree_fock()

# Save original P,Q
P0 = np.zeros((R, Na), dtype=np.complex128)
Q0 = np.zeros((R, Nb), dtype=np.complex128)
for r in range(R):
    for J in range(Na):
        P0[r, J] = rr.get_p_element(J, r)
    for K in range(Nb):
        Q0[r, K] = rr.get_q_element(K, r)

# Zero tensors
h1 = qf.Tensor([norb, norb], "h1")
h2 = qf.Tensor([norb, norb, norb, norb], "h2")
h2e = qf.Tensor([norb*norb, norb*norb], "h2e_einsum")

h0 = 2.3456789

rr.apply_tensor_spat_012bdy(h0, h1, h2, h2e, norb)

SigmaP_rr = np.zeros((R, Na), dtype=np.complex128)
SigmaQ_rr = np.zeros((R, Nb), dtype=np.complex128)
for r in range(R):
    for J in range(Na):
        SigmaP_rr[r, J] = rr.get_p_element(J, r)
    for K in range(Nb):
        SigmaQ_rr[r, K] = rr.get_q_element(K, r)

SigmaP_exact = np.zeros((R, Na), dtype=np.complex128)
SigmaQ_exact = np.zeros((R, Nb), dtype=np.complex128)

for i in range(R):
    for m in range(R):
        qdot = np.dot(Q0[m, :], Q0[i, :])
        pdot = np.dot(P0[m, :], P0[i, :])
        SigmaP_exact[i, :] += h0 * qdot * P0[m, :]
        SigmaQ_exact[i, :] += h0 * pdot * Q0[m, :]

print("||SigmaP_rr - SigmaP_exact|| =", np.linalg.norm(SigmaP_rr - SigmaP_exact))
print("||SigmaQ_rr - SigmaQ_exact|| =", np.linalg.norm(SigmaQ_rr - SigmaQ_exact))