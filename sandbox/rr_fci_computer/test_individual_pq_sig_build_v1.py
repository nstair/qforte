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

def rr_norm_from_factors(P, Q):
    R = P.shape[0]
    out = 0.0 + 0.0j
    for i in range(R):
        for m in range(R):
            out += np.vdot(P[i, :], P[m, :]) * np.vdot(Q[i, :], Q[m, :])
    return out

def projected_sigmas_from_dense_rr_style(Sigma_dense, P, Q):
    """
    Build the dense/exact projected RR sigma objects corresponding to:

        SigmaP_i(J) = sum_K Sigma(J,K) * conj(Q_i(K))
        SigmaQ_i(K) = sum_J conj(P_i(J)) * Sigma(J,K)

    Shapes:
        Sigma_dense : (Na, Nb)
        P           : (R, Na)
        Q           : (R, Nb)

    Returns:
        SigmaP      : (R, Na)
        SigmaQ      : (R, Nb)
    """
    R, Na = P.shape
    _, Nb = Q.shape

    SigmaP = np.zeros((R, Na), dtype=np.complex128)
    SigmaQ = np.zeros((R, Nb), dtype=np.complex128)

    for i in range(R):
        SigmaP[i, :] = Sigma_dense @ np.conj(Q[i, :])
        SigmaQ[i, :] = np.conj(P[i, :]) @ Sigma_dense

    return SigmaP, SigmaQ

def frob_norm(A):
    return np.sqrt(np.sum(np.abs(A)**2))

def rowwise_errors(A, B):
    errs = []
    for i in range(A.shape[0]):
        errs.append(np.linalg.norm(A[i, :] - B[i, :]))
    return np.array(errs)

# ============================================================
# molecule
# ============================================================

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

# ------------------------------------------------------------
# choose rank
# ------------------------------------------------------------
R = 1
# R = 2
# R = 3
# R = min(Na, Nb)

rr = qf.RRFCIComputer(nel=nel, sz=sz, norb=norb, rank=R)

# reference = "random"
reference = "hf"

rng = np.random.default_rng(7)

if reference == "hf":
    rr.hartree_fock()
else:
    # P = rng.random((R, Na)) + 1.0j * rng.random((R, Na))
    # Q = rng.random((R, Nb)) + 1.0j * rng.random((R, Nb))

    P = rng.random((R, Na)) 
    Q = rng.random((R, Nb)) 

    # normalize reconstructed dense C in Frobenius norm
    Ctmp = P.T @ Q
    nrm = np.sqrt(np.sum(np.abs(Ctmp)**2))
    P /= np.sqrt(nrm)
    Q /= np.sqrt(nrm)

    for r in range(R):
        for J in range(Na):
            rr.set_p_element(J, r, P[r, J])
        for K in range(Nb):
            rr.set_q_element(K, r, Q[r, K])

# Save original factors
P0 = rr_p_to_numpy(rr)
Q0 = rr_q_to_numpy(rr)

# ============================================================
# norm checks
# ============================================================

Crr_tensor = rr.reconstruct_C()
Crr = tensor_to_numpy_2d(Crr_tensor)

norm_rr = rr_norm_from_factors(P0, Q0)
norm_dense = np.vdot(Crr.ravel(), Crr.ravel())

print("\nNorm checks")
print("======================================================")
print("RR factor norm        =", norm_rr)
print("Dense reconstructed   =", norm_dense)
print("abs diff              =", abs(norm_rr - norm_dense))

test_same_spin_alfa = True
test_same_spin_beta = False
test_diff_spin = False

# ============================================================
# dense tensor reference 1 body
# ============================================================

fci_tensor = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fci_tensor.set_state(Crr_tensor)

fci_tensor.apply_tensor_spat_012bdy(
    mol.nuclear_repulsion_energy,
    mol.mo_oeis,
    mol.mo_teis,
    mol.mo_teis_einsum,
    norb
)

Sigma_tensor = tensor_to_numpy_2d(fci_tensor.get_state())
SigmaP_tensor, SigmaQ_tensor = projected_sigmas_from_dense_rr_style(Sigma_tensor, P0, Q0)

# ============================================================
# dense trusted sqop reference
# ============================================================

fci_sqop = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fci_sqop.set_state(Crr_tensor)
fci_sqop.apply_sqop(mol.sq_hamiltonian)
Sigma_sqop = tensor_to_numpy_2d(fci_sqop.get_state())
SigmaP_sqop, SigmaQ_sqop = projected_sigmas_from_dense_rr_style(Sigma_sqop, P0, Q0)

# ============================================================
# RR projected sigma build
# ============================================================

rr_sigma = qf.RRFCIComputer(nel=nel, sz=sz, norb=norb, rank=R)
for r in range(R):
    for J in range(Na):
        rr_sigma.set_p_element(J, r, P0[r, J])
    for K in range(Nb):
        rr_sigma.set_q_element(K, r, Q0[r, K])

rr_sigma.apply_tensor_spat_012bdy(
    mol.nuclear_repulsion_energy,
    mol.mo_oeis,
    mol.mo_teis,
    mol.mo_teis_einsum,
    norb
)

SigmaP_rr = rr_p_to_numpy(rr_sigma)
SigmaQ_rr = rr_q_to_numpy(rr_sigma)

# ============================================================
# numerator / energy consistency checks
# ============================================================

numP_rr = sum(np.vdot(P0[i, :], SigmaP_rr[i, :]) for i in range(R))
numQ_rr = sum(np.vdot(Q0[i, :], SigmaQ_rr[i, :]) for i in range(R))

num_dense_sqop = np.vdot(Crr.ravel(), Sigma_sqop.ravel())
num_dense_tensor = np.vdot(Crr.ravel(), Sigma_tensor.ravel())

Err_rr = rr.get_exp_val_tensor(
    mol.nuclear_repulsion_energy,
    mol.mo_oeis,
    mol.mo_teis,
    mol.mo_teis_einsum,
    norb
)
Err_sqop = num_dense_sqop / norm_dense
Err_tensor = num_dense_tensor / norm_dense

print("\nEnergy / numerator checks")
print("======================================================")
print("numP_rr              =", numP_rr)
print("numQ_rr              =", numQ_rr)
print("abs(numP_rr-numQ_rr) =", abs(numP_rr - numQ_rr))
print("dense sqop numerator =", num_dense_sqop)
print("dense tens numerator =", num_dense_tensor)
print("Err_rr               =", Err_rr)
print("Err_sqop             =", Err_sqop)
print("Err_tensor           =", Err_tensor)

# ============================================================
# sigma comparisons
# ============================================================

print("\nProjected sigma comparisons")
print("======================================================")
print("||SigmaP_rr - SigmaP_sqop||   =", frob_norm(SigmaP_rr - SigmaP_sqop))
print("||SigmaQ_rr - SigmaQ_sqop||   =", frob_norm(SigmaQ_rr - SigmaQ_sqop))
print("||SigmaP_rr - SigmaP_tensor|| =", frob_norm(SigmaP_rr - SigmaP_tensor))
print("||SigmaQ_rr - SigmaQ_tensor|| =", frob_norm(SigmaQ_rr - SigmaQ_tensor))

print("\nRow-wise SigmaP rr vs sqop")
print("======================================================")
print(rowwise_errors(SigmaP_rr, SigmaP_sqop))

print("\nRow-wise SigmaQ rr vs sqop")
print("======================================================")
print(rowwise_errors(SigmaQ_rr, SigmaQ_sqop))

# ============================================================
# detailed diagnostics
# ============================================================

if R == 1:
    print("\nProjected vector diagnostics")
    print("======================================================")
    print("\nSigmaP_rr[0,:]      =", SigmaP_rr[0, :])
    print("\nSigmaP_sqop[0,:]    =", SigmaP_sqop[0, :])
    print("\nDeltaP              =", SigmaP_rr[0, :] - SigmaP_sqop[0, :])
    print("\nSigmaQ_rr[0,:]      =", SigmaQ_rr[0, :])
    print("\nSigmaQ_sqop[0,:]    =", SigmaQ_sqop[0, :])
    print("\nDeltaQ              =", SigmaQ_rr[0, :] - SigmaQ_sqop[0, :])

# ============================================================
# HF-specific direct check
# ============================================================

if reference == "hf" and R == 1:
    print("\nHF direct checks")
    print("======================================================")
    print("Sigma_sqop[0,0]      =", Sigma_sqop[0,0])
    print("Sigma_tensor[0,0]    =", Sigma_tensor[0,0])
    print("SigmaP_rr[0,0]       =", SigmaP_rr[0,0])
    print("SigmaQ_rr[0,0]       =", SigmaQ_rr[0,0])
    print("Ehf                  =", mol.hf_energy)