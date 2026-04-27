import qforte as qf
from qforte.helper.printing import tensor_str as tstr
import numpy as np
import time


# ============================================================
# Helpers
# ============================================================

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


def frob_norm(A):
    return np.sqrt(np.sum(np.abs(A) ** 2))


# ============================================================
# Define the reference and geometry lists.
# ============================================================

geom = [
   ('H', (0., 0., 1.0)),
   ('H', (0., 0., 2.0)),
   ('H', (0., 0., 3.0)),
   ('H', (0., 0., 4.0)),
   ('H', (0., 0., 5.0)),
   ('H', (0., 0., 6.0)),
   ('H', (0., 0., 7.0)),
   ('H', (0., 0., 8.0)),
#    ('H', (0., 0., 9.0)),
#    ('H', (0., 0., 10.0)),
]

timer = qf.local_timer()

timer.reset()
mol = qf.system_factory(
   build_type='psi4',
   mol_geometry=geom,
   basis='sto-3g',
   build_qb_ham=False,
   run_fci=1)

timer.record('Run Psi4 and Initialize')

print("\n Initial RRFCI Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

graph = qf.FCIGraph((nel + sz) // 2, nel - (nel + sz) // 2, norb)
Na = graph.get_lena()
Nb = graph.get_lenb()

# ======> Specify Rank here <======
R = min(1, Na, Nb)
R = 1

print(f" nqubit:    {norb*2}")
print(f" nel:       {nel}")
print(f" norb:      {norb}")
print(f" Na:        {Na}")
print(f" Nb:        {Nb}")
print(f" R:         {R}")

rr_comp = qf.RRFCIComputer(nel=nel, sz=sz, norb=norb, rank=R)
fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)

# reference = 'random'
reference = 'hf'

rng = np.random.default_rng(7)

if reference == 'hf':
    rr_comp.hartree_fock()

elif reference == 'random':
    # Build random P and Q
    P = rng.random((R, Na)) + 1.0j * rng.random((R, Na))
    Q = rng.random((R, Nb)) + 1.0j * rng.random((R, Nb))

    # Optional mild normalization so numbers stay sane
    # Normalize reconstructed C in Frobenius norm.
    Crr_tmp = P.T @ Q
    nrm = np.sqrt(np.sum(np.abs(Crr_tmp)**2))
    P /= np.sqrt(nrm)
    Q /= np.sqrt(nrm)

    for r in range(R):
        for J in range(Na):
            rr_comp.set_p_element(J, r, P[r, J])
        for K in range(Nb):
            rr_comp.set_q_element(K, r, Q[r, K])

else:
    raise RuntimeError("Unknown reference choice")


# ------------------------------------------------------------
# Save the ORIGINAL P and Q before RR sigma overwrite
# ------------------------------------------------------------
P0 = rr_p_to_numpy(rr_comp)   # shape (R, Na)
Q0 = rr_q_to_numpy(rr_comp)   # shape (R, Nb)

# Build dense C from RR state
Crr_tensor = rr_comp.reconstruct_C()
Crr = tensor_to_numpy_2d(Crr_tensor)   # shape (Na, Nb)

# Copy dense C into a standard FCIComputer
fci_comp.set_state(Crr_tensor)

print("\n Initial RRFCI and FCIComputer states")
print("======================================================")
print(tstr("C0rr", Crr))
print(tstr("P0", P0))
print(tstr("Q0", Q0))

# ------------------------------------------------------------
# Dense sigma build
# ------------------------------------------------------------
timer.reset()
fci_comp.apply_tensor_spat_012bdy(
   mol.nuclear_repulsion_energy,
   mol.mo_oeis,
   mol.mo_teis,
   mol.mo_teis_einsum,
   norb)
timer.record('Dense apply spatial 012 body')

Sigma_actual_tensor = fci_comp.get_state()
Sigma_actual = tensor_to_numpy_2d(Sigma_actual_tensor)   # shape (Na, Nb)

# ------------------------------------------------------------
# RR projected sigma build
# NOTE: this overwrites rr_comp.P() and rr_comp.Q()
# with the projected sigma objects in your current implementation
# ------------------------------------------------------------
timer.reset()
rr_comp.apply_tensor_spat_012bdy(
   mol.nuclear_repulsion_energy,
   mol.mo_oeis,
   mol.mo_teis,
   mol.mo_teis_einsum,
   norb)
timer.record('RR projected sigma build')

SigmaP_rr = rr_p_to_numpy(rr_comp)   # shape (R, Na), stores projected sigma^P rows
SigmaQ_rr = rr_q_to_numpy(rr_comp)   # shape (R, Nb), stores projected sigma^Q rows

SigmaC_rr = rr_comp.reconstruct_C()

print(Sigma_actual_tensor)
print(SigmaC_rr)

       # shape (Na, Nb), stores reconstructed C from projected sigma

# ------------------------------------------------------------
# Build projected sigma from dense Sigma_actual using ORIGINAL P0,Q0
#
# Definitions:
#   SigmaP_dense[i, J] = sum_K Sigma_actual[J, K] * Q0[i, K]
#   SigmaQ_dense[i, K] = sum_J P0[i, J] * Sigma_actual[J, K]
# ------------------------------------------------------------
SigmaP_dense = np.zeros((R, Na), dtype=np.complex128)
SigmaQ_dense = np.zeros((R, Nb), dtype=np.complex128)

for i in range(R):
    # vectorized forms:
    # SigmaP_dense[i, :] = Sigma_actual @ Q0[i, :] ??? no:
    # Sigma_actual is (Na,Nb), Q0[i,:] is (Nb,)
    # result is (Na,)
    SigmaP_dense[i, :] = Sigma_actual @ Q0[i, :]

    # SigmaQ_dense[i, :] = P0[i,:] @ Sigma_actual
    SigmaQ_dense[i, :] = P0[i, :] @ Sigma_actual

# ------------------------------------------------------------
# Errors
# ------------------------------------------------------------
errP_abs = frob_norm(SigmaP_rr - SigmaP_dense)
errQ_abs = frob_norm(SigmaQ_rr - SigmaQ_dense)

normP = max(frob_norm(SigmaP_dense), 1.0e-14)
normQ = max(frob_norm(SigmaQ_dense), 1.0e-14)

errP_rel = errP_abs / normP
errQ_rel = errQ_abs / normQ

# Optional spot checks against HF projections
hf_proj_dense = Sigma_actual[0, 0]
hf_proj_rr_from_denseC = np.sum(Crr[0, 0])

print("\n Timing")
print("======================================================")
print(timer)

print("\n Shapes")
print("======================================================")
print(f"Crr shape:          {Crr.shape}")
print(f"Sigma_actual shape: {Sigma_actual.shape}")
print(f"SigmaP_rr shape:    {SigmaP_rr.shape}")
print(f"SigmaQ_rr shape:    {SigmaQ_rr.shape}")

print("\n Sanity Checks")
print("======================================================")
print(f"||Crr||_F:                       {frob_norm(Crr)}")
print(f"||Sigma_actual||_F:             {frob_norm(Sigma_actual)}")
print(f"||SigmaP_dense||_F:             {frob_norm(SigmaP_dense)}")
print(f"||SigmaQ_dense||_F:             {frob_norm(SigmaQ_dense)}")
print(f"abs err SigmaP:                 {errP_abs}")
print(f"rel err SigmaP:                 {errP_rel}")
print(f"abs err SigmaQ:                 {errQ_abs}")
print(f"rel err SigmaQ:                 {errQ_rel}")

print("\n Energetics")
print("======================================================")
print(f" Efci:                          {mol.fci_energy}")
print(f" Enr:                           {mol.nuclear_repulsion_energy}")
print(f" Eelec:                         {mol.hf_energy - mol.nuclear_repulsion_energy}")
print(f" Ehf:                           {mol.hf_energy}")

# Optional: compare RR expectation value helper vs dense expectation value
rr_comp2 = qf.RRFCIComputer(nel=nel, sz=sz, norb=norb, rank=R)
# rr_comp2.hartree_fock()  # just to initialize the state and internal buffers

for r in range(R):
    for J in range(Na):
        rr_comp2.set_p_element(J, r, P0[r, J])
    for K in range(Nb):
        rr_comp2.set_q_element(K, r, P0[r, 0]*0 + Q0[r, K])  # just to avoid accidental aliasing assumptions



Err = rr_comp2.get_exp_val_tensor(
    mol.nuclear_repulsion_energy,
    mol.mo_oeis,
    mol.mo_teis,
    mol.mo_teis_einsum,
    norb
)   

rr_comp2.apply_tensor_spat_012bdy(
    mol.nuclear_repulsion_energy,
    mol.mo_oeis,
    mol.mo_teis,
    mol.mo_teis_einsum,
    norb
)

C_flat = Crr.ravel()
S_flat = Sigma_actual.ravel()
dense_rr_exp = np.vdot(C_flat, S_flat) / np.vdot(C_flat, C_flat)


Crr_flat = Crr.ravel()
Sig_rr = rr_comp2.reconstruct_C()
Sig_rr_np = tensor_to_numpy_2d(Sig_rr)
Srr_flat = Sig_rr_np.ravel()
# Srr_flat = Sigma_actual.ravel()
Err_from_reconstruc = np.vdot(Crr_flat, Srr_flat) / np.vdot(Crr_flat, Crr_flat)

print(f" dense projected exp val:       {np.real(dense_rr_exp)}")
print(f" E from get_exp_val_tensor:     {np.real(Err)}")
print(f" RR exp val from rec:           {np.real(Err_from_reconstruc)}")


# print("\n P/Q projections of dense Sigma")
# print("======================================================")
# print(tstr("SigP RR", SigmaP_rr))
# print(tstr("SigP Actual", SigmaP_dense))
# print(tstr("SigQ RR", SigmaQ_rr))
# print(tstr("SigQ Actual", SigmaQ_dense))