import numpy as np

from qforte.maths.eigsolve import canonical_geig_solve


def test_canonical_geig_solve_sorted_vectors_satisfy_gep():
    S = np.array(
        [
            [1.67632039 - 7.64394652e-18j, -0.85075699 + 4.98608675e-02j, 0.31050733 + 2.75638583e-01j],
            [-0.85075699 - 4.98608675e-02j, 5.00478009 + 6.07087262e-17j, 0.52648249 - 1.94342761e00j],
            [0.31050733 - 2.75638583e-01j, 0.52648249 + 1.94342761e00j, 1.63841790 + 4.34421261e-18j],
        ],
        dtype=complex,
    )
    H = np.array(
        [
            [-3.80244548 + 0.0j, -1.52462887 + 1.48163482j, -1.68498395 + 0.92214623j],
            [-1.52462887 - 1.48163482j, -2.53489296 + 0.0j, 0.08433341 - 2.03941770j],
            [-1.68498395 - 0.92214623j, 0.08433341 + 2.03941770j, -5.03351942 + 0.0j],
        ],
        dtype=complex,
    )

    evals, evecs = canonical_geig_solve(
        S,
        H,
        sort_ret_vals=True,
        stabilization_thresh=1.0e-10,
    )

    assert np.all(np.diff(np.real(evals)) >= -1.0e-12)

    for idx, eval_ in enumerate(evals):
        residual = H @ evecs[:, idx] - eval_ * (S @ evecs[:, idx])
        assert np.max(np.abs(residual)) < 1.0e-10

        s_norm = np.vdot(evecs[:, idx], S @ evecs[:, idx])
        assert abs(s_norm - 1.0) < 1.0e-10
