import qforte as qf
import numpy as np

def tensor_to_numpy_2d(T):
    shp = T.shape()
    out = np.zeros((shp[0], shp[1]), dtype=np.complex128)
    for i in range(shp[0]):
        for j in range(shp[1]):
            out[i, j] = T.get([i, j])
    return out

def rr_p_to_numpy(rr):
    R = rr.get_rank()
    Na = rr.get_nalfa_strs()
    P = np.zeros((R, Na), dtype=np.complex128)
    for r in range(R):
        for J in range(Na):
            P[r, J] = rr.get_p_element(J, r)
    return P

def rr_q_to_numpy(rr):
    R = rr.get_rank()
    Nb = rr.get_nbeta_strs()
    Q = np.zeros((R, Nb), dtype=np.complex128)
    for r in range(R):
        for K in range(Nb):
            Q[r, K] = rr.get_q_element(K, r)
    return Q

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

graph = qf.FCIGraph((nel + sz)//2, nel - (nel + sz)//2, norb)
Na = graph.get_lena()
Nb = graph.get_lenb()
R = min(Na, Nb)

rr = qf.RRFCIComputer(nel=nel, sz=sz, norb=norb, rank=R)
fci = qf.FCIComputer(nel=nel, sz=sz, norb=norb)

rr.hartree_fock()

P0 = rr_p_to_numpy(rr)
Q0 = rr_q_to_numpy(rr)

Crr = rr.reconstruct_C()
fci.set_state(Crr)

h0 = 0.0

# zero 1-body
h1 = qf.Tensor([norb, norb], "h1")
h1_np = np.zeros((norb, norb), dtype=np.complex128)
h1.fill_from_nparray(h1_np.ravel(), h1.shape())

# nonzero same-spin 2-body tensor
h2 = qf.Tensor([norb, norb, norb, norb], "h2")
h2_np = np.zeros((norb, norb, norb, norb), dtype=np.complex128)

# simple sparse choice
h2_np[0, 0, 0, 0] = 1.0
h2_np[0, 1, 0, 1] = 0.7
h2_np[1, 0, 1, 0] = -0.3
h2_np[1, 1, 1, 1] = 0.4

h2.fill_from_nparray(h2_np.ravel(), h2.shape())

# zero mixed-spin effective tensor
h2einsum = qf.Tensor([norb*norb, norb*norb], "h2einsum")
h2einsum_np = np.zeros((norb*norb, norb*norb), dtype=np.complex128)
h2einsum.fill_from_nparray(h2einsum_np.ravel(), h2einsum.shape())

# Dense sigma
fci.apply_tensor_spat_012bdy(h0, h1, h2, h2einsum, norb)
Sigma_dense = tensor_to_numpy_2d(fci.get_state())

# RR sigma
rr.apply_tensor_spat_012bdy(h0, h1, h2, h2einsum, norb)

SigmaP_rr = rr_p_to_numpy(rr)
SigmaQ_rr = rr_q_to_numpy(rr)

# Dense projected sigmas
SigmaP_dense = np.zeros((R, Na), dtype=np.complex128)
SigmaQ_dense = np.zeros((R, Nb), dtype=np.complex128)

for i in range(R):
    SigmaP_dense[i, :] = Sigma_dense @ Q0[i, :]
    SigmaQ_dense[i, :] = P0[i, :] @ Sigma_dense

print("same-spin 2-body only")
print("||SigmaP_rr - SigmaP_dense|| =", np.linalg.norm(SigmaP_rr - SigmaP_dense))
print("||SigmaQ_rr - SigmaQ_dense|| =", np.linalg.norm(SigmaQ_rr - SigmaQ_dense))