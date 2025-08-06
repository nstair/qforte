'''
A module for transforming orbitals between different representations, such as
molecular orbitals (MO) and localized orbitals (LO), and for handling transformations
between different orbital representations in quantum chemistry calculations.
'''

import numpy as np
import math
import random
from qforte.helper.printing import *

def pipek_mezey(C, 
                S, 
                basis,
                maxiter=100,
                conv_tol=1e-12,
                seed=0,
                verbose=False):
    """
    Perform Pipek–Mezey localization of a block of molecular orbitals.

    This routine implements Psi4’s PMLocalizer in pure NumPy: it maximizes
    the sum of squared Mulliken charges on each atom via Jacobi rotations.
    You can localize either the occupied or virtual block (or any set of MOs)
    by passing in the corresponding coefficient matrix C.

    Parameters
    ----------
    C : ndarray, shape (nbf, nmo)
        The initial MO coefficient matrix in the AO basis.  Each column is one
        orbital (e.g. the occupied block or virtual block).
    S : ndarray, shape (nbf, nbf)
        The AO overlap matrix, ⟨χ_μ|χ_ν⟩.
    basis : psi4.core.BasisSet
        A Psi4 BasisSet object used to determine which AOs belong to each atom.
        The function will call `basis.function_to_center(μ)` to build its atom
        starts array.
    maxiter : int, optional
        Maximum number of Jacobi‐rotation sweeps before giving up (default 100).
    conv_tol : float, optional
        Convergence threshold on the _relative_ change in the PM metric
        between sweeps (default 1e-12).
    seed : int, optional
        Seed for Psi4‐style random sweep ordering (default 0).
    verbose : bool, optional
        If True, print per‐sweep metrics and convergence info to stdout.

    Raises
    ------
    RuntimeError
        If the metric fails to converge within `maxiter` Jacobi sweeps.

    Returns
    -------
    L : ndarray, shape (nbf, nmo)
        The localized MO coefficient matrix, L = C · U.

    Notes
    -----
    - The function builds an “astarts” list of AO-indices by querying
      `basis.function_to_center()`, mimicking Psi4's internal grouping.
    - The PM metric is ∑_i ∑_A (q_i^A)^2, with q_i^A the Mulliken charge of
      orbital i on atom A.
    - Uses the small-angle fallback (θ → π/4) exactly as in Psi4's C++ code.
    - Preserves Lᵀ S L = I by rotating both L and LS = S·L simultaneously.
    - The returned `L` is normalized in the overlap metric (LᵀS L ≈ I), not
      the Euclidean sense.
    """

    nbf, nmo = C.shape
    nA = basis.molecule().natom()

    astarts = []
    aoff = 0

    for m in range(nbf):
        if(basis.function_to_center(m) == aoff):
            astarts.append(m)
            aoff += 1

    astarts.append(nbf)

    # — initialize L, LS = S·L, and U = I —
    L  = C.copy()
    LS = S.dot(L)
    U  = np.eye(nmo)

    # seed the RNG
    random.seed(seed)

    # — compute initial metric —
    metric = 0.0
    for i in range(nmo):
        for A in range(nA):
            nm  = astarts[A+1] - astarts[A]
            off = astarts[A]

            PA  = np.dot(LS[off:off+nm, i], L[off:off+nm, i])
            metric += PA * PA
    
    old_metric = metric
    
    if verbose:
        print(f" Iter 0    Metric = {metric:24.16E}")

    # — Jacobi sweeps —
    for iteration in range(1, maxiter+1):

        if(iteration == maxiter):
            raise RuntimeError(f"PM Localization did nont converge in {maxiter} Iterations!")

        # build the randomized order2[] exactly as in Psi4
        order  = list(range(nmo))
        order2 = []
        for i in range(nmo):
            pivot = int((nmo - i) * random.random())
            order2.append(order.pop(pivot))

        # sweep over all pairs (i,j)
        for ii in range(nmo-1):
            for jj in range(ii+1, nmo):
                i = order2[ii]
                j = order2[jj]

                # accumulate a, b, c over atoms
                a = b = c = 0.0
                for A in range(nA):

                    nm  = astarts[A+1] - astarts[A]
                    off = astarts[A]

                    LSi = LS[off:off+nm, i]
                    LSj = LS[off:off+nm, j]
                    Li  =  L[off:off+nm, i]
                    Lj  =  L[off:off+nm, j]

                    Aii = np.dot(LSi, Li)
                    Ajj = np.dot(LSj, Lj)
                    Aij = 0.5*(np.dot(LSi, Lj) + np.dot(LSj, Li))

                    Ad = Aii - Ajj
                    Ao = 2.0 * Aij
                    a += Ad * Ad
                    b += Ao * Ao
                    c += Ad * Ao

                # compute the optimal theta
                Hd    = a - b
                Ho    = 2.0 * c
                theta = 0.5 * math.atan2(Ho, Hd + math.sqrt(Hd*Hd + Ho*Ho))

                # small‐angle fallback (θ≈0 ⇒ maybe θ=π/4)
                if abs(theta) < 1e-8:
                    O0 = O1 = 0.0
                    for A in range(nA):
                        # off = astarts[A]
                        # nm  = astarts[A+1] - off

                        nm  = astarts[A+1] - astarts[A]
                        off = astarts[A]

                        LSi = LS[off:off+nm, i]
                        LSj = LS[off:off+nm, j]
                        Li  =  L[off:off+nm, i]
                        Lj  =  L[off:off+nm, j]

                        Aii = np.dot(LSi, Li)
                        Ajj = np.dot(LSj, Lj)
                        Aij = 0.5*(np.dot(LSi, Lj) + np.dot(LSj, Li))

                        O0 += Aij*Aij
                        O1 += 0.25*(Ajj - Aii)**2

                    if O1 < O0:
                        theta = math.pi / 4.0

                if abs(theta) > 1e-12:
                    cc = math.cos(theta)
                    ss = math.sin(theta)

                    # — Rotate L columns i and j using BLAS‐style Givens:
                    #    L[:,i] ←  c·L[:,i] + s·L[:,j]
                    #    L[:,j] ←  c·L[:,j] − s·L[:,i]_old
                    Li_old = L[:, i].copy()
                    Lj_old = L[:, j].copy()
                    L[:, i] =  cc * Li_old + ss * Lj_old
                    L[:, j] =  cc * Lj_old  - ss * Li_old

                    # — Likewise keep LS = S·L in sync:
                    LSi_old = LS[:, i].copy()
                    LSj_old = LS[:, j].copy()
                    LS[:, i] =  cc * LSi_old + ss * LSj_old
                    LS[:, j] =  cc * LSj_old  - ss * LSi_old

                    # — Accumulate the same rotation into U so that L = C·U:
                    #    G = I;  G[i,i]=c; G[j,j]=c; G[i,j]= s; G[j,i]=-s
                    G = np.eye(nmo)
                    G[i, i] =  cc
                    G[j, j] =  cc
                    G[i, j] =  ss
                    G[j, i] = -ss
                    U = U.dot(G)

        # recompute metric & check convergence
        metric = 0.0
        for i in range(nmo):
            for A in range(nA):
                off = astarts[A]
                nm  = astarts[A+1] - off
                PA  = np.dot(LS[off:off+nm, i], L[off:off+nm, i])
                metric += PA * PA

        conv = abs(metric - old_metric) / abs(old_metric) if old_metric != 0 else 0.0
        
        if verbose:
            print(f" Iter {iteration:3d}    Metric = {metric:24.16E}    Conv = {conv:.3e}")
        old_metric = metric
        
        if conv < conv_tol:
            if verbose:
                print("  Converged!")
            break

    #NOTE(Nick): May want to consider returing U as well at some point

    return L
