import numpy as np
from scipy import linalg
from qforte.helper.idx_org import sorted_largest_idxs
from qforte.helper.printing import matprint 

def canonical_geig_solve(
        S,
        H,
        print_mats=False,
        sort_ret_vals=False,
        stabilization_thresh=None):
    """Solves a generalized eigenvalue problem HC = SCe in a numerically stable
    fashioin. See pq. 144 of "Modern Quantum Chemistry" by A. Szabo
    and N. S. Ostlund beginning with Eq. (3.169).

        Arguments
        ---------

        S : ndarray
            A complex valued numpy array for the overlap matrix S.

        H : ndarray
            A complex valued numpy array for the matrix H.

        print_mats : bool
            Whether or not to print the intermediate matricies.

        sort_ret_vals : bool
            Whether or not to retrun the eivenvalues (and corresponding eigenvectors)
            in order of increasing value
            (meaning index 0 pertains to lowest eigenvalue).

        stabilization_thresh : float or None
            Optional threshold for discarding small overlap eigenvalues.
            None preserves the historical default threshold and code path.

        Returns
        -------
        e_prime : ndarray
            The energy eigenvalues.

        C : ndarray
            The eigenvectors.

    """

    THRESHOLD = 1e-15 if stabilization_thresh is None else stabilization_thresh
    s, U = linalg.eig(S)
    s_prime = []
    s_prime_idxs = []

    for idx, sii in enumerate(s):
        if(np.imag(sii) > 1e-5):
            raise ValueError('S may not be hermetian, large imag. eval component.')
        if(np.real(sii) > THRESHOLD):
            s_prime.append(np.real(sii))
            s_prime_idxs.append(idx)

    if((len(s) - len(s_prime)) != 0 and print_mats):
        print('\nGeneralized eigenvalue probelm rank was reduced, matrix may be ill conditioned!')
        print('  s is of inital rank:    ', len(s))
        print('  s is of truncated rank: ', len(s_prime))

    X_prime = np.zeros((len(s), len(s_prime)), dtype=complex)
    for i in range(len(s)):
        for j in range(len(s_prime)):
            u_col = j if stabilization_thresh is None else s_prime_idxs[j]
            X_prime[i][j] = U[i][u_col] / np.sqrt(s_prime[j])

    H_prime = (((X_prime.conjugate()).transpose()).dot(H)).dot(X_prime)
    e_prime, C_prime = linalg.eig(H_prime)
    C = X_prime.dot(C_prime)

    if(print_mats):
        print('\n      -----------------------------')
        print('      Printing GEVS Mats (unsorted)')
        print('      -----------------------------')

        I_prime = (((C.conjugate()).transpose()).dot(S)).dot(C)

        print('\ns:\n')
        print(s)
        print('\nU:\n')
        matprint(U)
        print('\nX_prime:\n')
        matprint(X_prime)
        print('\nH_prime:\n')
        matprint(H_prime)
        print('\ne_prime:\n')
        print(e_prime)
        print('\nC_prime:\n')
        matprint(C_prime)
        print('\ne_prime:\n')
        print(e_prime)
        print('\nC:\n')
        matprint(C)
        print('\nIprime:\n')
        matprint(I_prime)

        print('\n      ------------------------------')
        print('          Printing GEVS Mats End    ')
        print('      ------------------------------')

    if(sort_ret_vals):
        sorted_e_prime_idxs = sorted_largest_idxs(e_prime, use_real=True, rev=False)
        sorted_e_prime = np.zeros((len(e_prime)), dtype=complex)
        sorted_C_prime = np.zeros((len(e_prime),len(e_prime)), dtype=complex)
        for n in range(len(e_prime)):
            old_idx = sorted_e_prime_idxs[n][1]
            sorted_e_prime[n]   = e_prime[old_idx]
            sorted_C_prime[:,n] = C_prime[:,old_idx]

        # X_prime is the fixed overlap-basis transform. Only the generalized
        # eigenpairs need reordering by energy; permuting X_prime here corrupts
        # the reconstructed eigenvectors/state coefficients.
        sorted_C = X_prime.dot(sorted_C_prime)
        return sorted_e_prime, sorted_C

    else:
        return e_prime, C
