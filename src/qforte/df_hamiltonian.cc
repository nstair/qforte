#include <map>
#include <random>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <iterator>

#include "tensor.h"
#include "sq_operator.h"
#include "blas_math.h"

#include "df_hamiltonian.h"


DFHamiltonian::DFHamiltonian(
    int nel, 
    int norb, 
    Tensor& ff_eigenvalues,
    Tensor& one_body_squares,
    Tensor& one_body_ints,
    Tensor& one_body_correction,
    std::vector<Tensor>& scaled_density_density_matrices,
    std::vector<Tensor>& basis_change_matrices,
    std::vector<Tensor>& trotter_basis_change_matrices
    ) : 
        nel_(nel), 
        norb_(norb),
        ff_eigenvalues_(ff_eigenvalues),
        one_body_squares_(one_body_squares),
        one_body_ints_(one_body_ints),
        one_body_correction_(one_body_correction),
        scaled_density_density_matrices_(scaled_density_density_matrices),
        basis_change_matrices_(basis_change_matrices),
        trotter_basis_change_matrices_(trotter_basis_change_matrices){}

std::array<std::array<std::complex<double>, 2>, 2> DFHamiltonian::givens_matrix_elements(
        std::complex<double> a, 
        std::complex<double> b, 
        std::string which) 
{
        
        double cosine, sine;
        std::complex<double> phase(1.0, 0.0);

        if (std::abs(a) < 1.0e-11) {
            cosine = 1.0;
            sine = 0.0;
        } else if (std::abs(b) < 1.0e-11) {
            cosine = 0.0;
            sine = 1.0;
        } else {
            double denominator = std::sqrt(std::norm(a) + std::norm(b));
            cosine = std::abs(b) / denominator;
            sine = std::abs(a) / denominator;
            std::complex<double> sign_b = b / std::abs(b);
            std::complex<double> sign_a = a / std::abs(a);
            phase = sign_a * std::conj(sign_b);

            if (phase.imag() == 0) {
                phase = phase.real();
            }
        }

        std::array<std::array<std::complex<double>, 2>, 2> givens_rotation;

        if (which == "left") {
            if (std::abs(a.imag()) < 1.0e-11 && std::abs(b.imag()) < 1.0e-11) {
                givens_rotation = {{
                    {cosine, -phase * sine},
                    {phase * sine, cosine}
                }};
            } else {
                givens_rotation = {{
                    {cosine, -phase * sine},
                    {sine, phase * cosine}
                }};
            }
        } else if (which == "right") {
            if (std::abs(a.imag()) < 1.0e-11 && std::abs(b.imag()) < 1.0e-11) {
                givens_rotation = {{
                    {sine, phase * cosine},
                    {-phase * cosine, sine}
                }};
            } else {
                givens_rotation = {{
                    {sine, phase * cosine},
                    {cosine, -phase * sine}
                }};
            }
        } else {
            throw std::invalid_argument("\"which\" must be equal to \"left\" or \"right\".");
        }

        return givens_rotation;
}

// NOTE(Nick): note efficiet, speed up if this proves to be a bottleneck
void DFHamiltonian::givens_rotate(
    Tensor& op,
    const std::array<std::array<std::complex<double>, 2>, 2>& givens_rotation,
    size_t i, 
    size_t j, 
    std::string which) {

    op.square_error();
    Tensor op_new = op;
    size_t n = op.shape()[0];

    if (which == "row") {

        // Rotate rows i and j
        for (size_t k = 0; k < n; ++k) {
            size_t ik = n*i + k;
            size_t jk = n*j + k;
            op_new.data()[ik] = givens_rotation[0][0] * op.data()[ik] + givens_rotation[0][1] * op.data()[jk];
            op_new.data()[jk] = givens_rotation[1][0] * op.data()[ik] + givens_rotation[1][1] * op.data()[jk];
        }

    } else if (which == "col") {
        // Rotate columns i and j
        // NOTE(Nick): projably shuld just transpose and then do row wise for speed...
        for (size_t k = 0; k < n; ++k) {
            size_t ki = n*k + i;
            size_t kj = n*k + j;
            op_new.data()[ki] = givens_rotation[0][0] * op.data()[ki] + std::conj(givens_rotation[0][1]) * op.data()[kj];
            op_new.data()[kj] = givens_rotation[1][0] * op.data()[ki] + std::conj(givens_rotation[1][1]) * op.data()[kj];
        }

    } else {
        throw std::invalid_argument("\"which\" must be equal to \"row\" or \"col\".");
    }

    op = op_new;
}

std::tuple<
    std::vector<size_t>, 
    std::vector<size_t>, 
    std::vector<double>, 
    std::vector<double>, 
    std::vector<std::complex<double>>
> DFHamiltonian::givens_decomposition_square(
    const Tensor& unitary_matrix,
    const bool always_insert) {

    unitary_matrix.square_error();

    //deep copy I think?
    Tensor current_matrix = unitary_matrix; 
    int n = current_matrix.shape()[0];

    std::vector<size_t> i_vector;
    std::vector<size_t> j_vector;
    std::vector<double> theta_vector;
    std::vector<double> phi_vector;
    std::vector<std::complex<double>> diagonal(n);

    for (int k = 0; k < 2 * (n - 1) - 1; ++k) {
        int start_row, start_column;
        if (k < n - 1) {
            start_row = 0;
            start_column = n - 1 - k;
        } else {
            start_row = k - (n - 2);
            start_column = k - (n - 3);
        }

        std::vector<size_t> column_indices, row_indices;
        for (size_t col = start_column; col < n; col += 2) {
            column_indices.push_back(col);
        }
        for (size_t row = start_row; row < start_row + column_indices.size(); ++row) {
            row_indices.push_back(row);
        }

        for (size_t idx = 0; idx < row_indices.size(); ++idx) {
            size_t i = row_indices[idx];
            size_t j = column_indices[idx];
            size_t ij_right = n*i + j;

            std::complex<double> right_element = std::conj(current_matrix.data()[ij_right]);
            if (always_insert || std::abs(right_element) > 1.0e-11) {
                size_t ij_left = n * i + j - 1;
                std::complex<double> left_element = std::conj(current_matrix.data()[ij_left]);
                auto givens_rotation = givens_matrix_elements(left_element, right_element, "right");

                double theta = std::asin(std::real(givens_rotation[1][0]));
                double phi = std::arg(givens_rotation[1][1]);
                
                i_vector.push_back(j - 1);
                j_vector.push_back(j);
                theta_vector.push_back(theta);
                phi_vector.push_back(phi);

                givens_rotate(current_matrix, givens_rotation, j - 1, j, "col");
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        diagonal[i] = current_matrix.data()[i*n + i];
    }

    return std::make_tuple(i_vector, j_vector, theta_vector, phi_vector, diagonal);
}


/**
 * @brief Estimate the total CNOT count for one first-order Trotter step of the
 *        double-factorized (DF) electronic‐structure Hamiltonian exponential.
 *
 * We decompose
 *   H = H₁ + H₂,
 *   H₁ = ∑_{pq} h_{pq}\,a_p† a_q,                   (one-body)
 *   H₂ ≈ ∑_{ℓ=1}^L λ_ℓ Γ_ℓ²,                         (DF two-body)
 *
 * and approximate
 *   U(dt) ≈ e^{-iH₁ dt} ∏_{ℓ=1}^L e^{-iλ_ℓ Γ_ℓ² dt}.
 *
 * ––– One-body term H₁ –––
 * • Diagonalize H₁ = U_h† D_h U_h with two n×n Givens‐based basis changes.
 * • Number of Givens rotations per n×n unitary:  n(n–1)/2
 * • Each Givens rotation ⇒ 2 CNOTs
 * • Two basis changes (forward + inverse) ⇒
 *     2 × [ n(n–1)/2 rotations × 2 CNOT/rotation ]
 *   = 2 n(n–1) CNOT
 *
 * ––– DF two-body terms H₂ –––
 * For each factor ℓ:
 * • Basis-change matrix V_ℓ is r_ℓ×r_ℓ.  #rotations = r_ℓ(r_ℓ–1)/2
 * • Each rotation ⇒ 2 CNOT  ⇒ one V_ℓ (forward) costs r_ℓ(r_ℓ–1) CNOT
 * • Apply forward + inverse ⇒ 2 × r_ℓ(r_ℓ–1)
 * • Diagonal block exp(–i λ_ℓ Λ_ℓ² dt) is r_ℓ single-qubit R_z ⇒ 0 CNOT
 * Summing over ℓ gives
 *   C₂ = ∑ₗ 2 r_ℓ(r_ℓ–1).
 *
 * ––– Total per Trotter step –––
 *   C_step = C₁ + C₂ = 2 n(n–1) + ∑ₗ 2 r_ℓ(r_ℓ–1).
 *
 * This function assumes a single Trotter step (N_step = 1).  To get the full
 * evolution for N steps, multiply the result by N.
 */
size_t DFHamiltonian::count_cnot_for_exponential() const {
    // Number of spin-orbitals
    const size_t n = static_cast<size_t>(norb_);

    // 1) One-body term H1:
    //    2 * n * (n – 1) CNOT per step
    size_t cnot_H1 = 2 * n * (n - 1);

    // 2) DF two-body terms H2:
    //    Sum over each factor ℓ of:
    //      2 * r_ℓ * (r_ℓ – 1)
    //    where r_ℓ = dimension of basis_change_matrices_[ℓ].
    size_t cnot_H2 = 0;
    for (const auto& V : basis_change_matrices_) {
        size_t r = V.shape()[0];
        if (r >= 2) {
            cnot_H2 += 2 * r * (r - 1);
        }
    }

    // 3) Total CNOTs per (single) Trotter step
    return cnot_H1 + cnot_H2;
}

