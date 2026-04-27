// rr_fci_computer.cc
#include <map>
#include <random>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <iterator>
#include <complex>
#include <vector>

#include "qubit_basis.h"
#include "circuit.h"
#include "gate.h"
#include "helpers.h"
#include "qubit_operator.h"
#include "tensor.h"
#include "tensor_operator.h"
#include "qubit_op_pool.h"
#include "sq_op_pool.h"
#include "timer.h"
#include "sq_operator.h"
#include "blas_math.h"

#include "rr_fci_computer.h"
#include "fci_graph.h"

RRFCIComputer::RRFCIComputer(int nel, int sz, int norb, int rank)
    : nel_(nel), sz_(sz), norb_(norb), R_(rank) {

    if (nel_ < 0) {
        throw std::invalid_argument("Cannot have negative electrons");
    }
    if (rank <= 0) {
        throw std::invalid_argument("RRFCIComputer rank must be positive");
    }
    if (nel_ < std::abs(static_cast<double>(sz_))) {
        throw std::invalid_argument("Spin quantum number exceeds physical limits");
    }
    if ((nel_ + sz_) % 2 != 0) {
        throw std::invalid_argument("Parity of spin quantum number and number of electrons is incompatible");
    }

    nalfa_el_ = (nel_ + sz_) / 2;
    nbeta_el_ = nel_ - nalfa_el_;

    nalfa_strs_ = 1;
    for (int i = 1; i <= static_cast<int>(nalfa_el_); ++i) {
        nalfa_strs_ *= norb_ - i + 1;
        nalfa_strs_ /= i;
    }
    if (nalfa_el_ > norb_) {
        nalfa_strs_ = 0;
    }

    nbeta_strs_ = 1;
    for (int i = 1; i <= static_cast<int>(nbeta_el_); ++i) {
        nbeta_strs_ *= norb_ - i + 1;
        nbeta_strs_ /= i;
    }
    if (nbeta_el_ > norb_) {
        nbeta_strs_ = 0;
    }

    // Transposed / row-major-friendly storage:
    // each rank slice is contiguous.
    P_.zero_with_shape({R_, nalfa_strs_});
    Q_.zero_with_shape({R_, nbeta_strs_});
    SigmaP_.zero_with_shape({R_, nalfa_strs_});
    SigmaQ_.zero_with_shape({R_, nbeta_strs_});

    P_.set_name("RRFCI P");
    Q_.set_name("RRFCI Q");
    SigmaP_.set_name("RRFCI SigmaP");
    SigmaQ_.set_name("RRFCI SigmaQ");

    graph_ = FCIGraph(nalfa_el_, nbeta_el_, norb_);
    timer_ = local_timer();
}

void RRFCIComputer::set_rank(size_t rank) {
    if (rank == 0) {
        throw std::invalid_argument("RRFCIComputer rank must be positive");
    }

    R_ = rank;

    P_.zero_with_shape({R_, nalfa_strs_});
    Q_.zero_with_shape({R_, nbeta_strs_});
    SigmaP_.zero_with_shape({R_, nalfa_strs_});
    SigmaQ_.zero_with_shape({R_, nbeta_strs_});

    P_.set_name("RRFCI P");
    Q_.set_name("RRFCI Q");
    SigmaP_.set_name("RRFCI SigmaP");
    SigmaQ_.set_name("RRFCI SigmaQ");
}

void RRFCIComputer::set_P(const Tensor& P) {
    if (P.shape() != std::vector<size_t>{R_, nalfa_strs_}) {
        throw std::invalid_argument("set_P shape mismatch");
    }
    P_ = P;
}

void RRFCIComputer::set_Q(const Tensor& Q) {
    if (Q.shape() != std::vector<size_t>{R_, nbeta_strs_}) {
        throw std::invalid_argument("set_Q shape mismatch");
    }
    Q_ = Q;
}

void RRFCIComputer::zero() {
    std::fill(P_.data().begin(), P_.data().end(), std::complex<double>(0.0, 0.0));
    std::fill(Q_.data().begin(), Q_.data().end(), std::complex<double>(0.0, 0.0));
    std::fill(SigmaP_.data().begin(), SigmaP_.data().end(), std::complex<double>(0.0, 0.0));
    std::fill(SigmaQ_.data().begin(), SigmaQ_.data().end(), std::complex<double>(0.0, 0.0));
}

void RRFCIComputer::clear_sigmas() {
    std::fill(SigmaP_.data().begin(), SigmaP_.data().end(), std::complex<double>(0.0, 0.0));
    std::fill(SigmaQ_.data().begin(), SigmaQ_.data().end(), std::complex<double>(0.0, 0.0));
}

void RRFCIComputer::set_element(
    const std::vector<size_t>& idxs,
    const std::complex<double> val) {

    if (idxs.size() != 3) {
        throw std::invalid_argument("RRFCIComputer::set_element expects idxs = {which,string,rank}");
    }

    const size_t which = idxs[0];
    const size_t str   = idxs[1];
    const size_t r     = idxs[2];

    if (which == 0) {
        if (str >= nalfa_strs_ || r >= R_) {
            throw std::out_of_range("P index out of range");
        }
        P_.data()[p_index(r, str)] = val;
        return;
    }

    if (which == 1) {
        if (str >= nbeta_strs_ || r >= R_) {
            throw std::out_of_range("Q index out of range");
        }
        Q_.data()[q_index(r, str)] = val;
        return;
    }

    throw std::invalid_argument("RRFCIComputer::set_element expects first index 0 (P) or 1 (Q)");
}

void RRFCIComputer::add_to_element(
    const std::vector<size_t>& idxs,
    const std::complex<double> val) {

    if (idxs.size() != 3) {
        throw std::invalid_argument("RRFCIComputer::add_to_element expects idxs = {which,string,rank}");
    }

    const size_t which = idxs[0];
    const size_t str   = idxs[1];
    const size_t r     = idxs[2];

    if (which == 0) {
        if (str >= nalfa_strs_ || r >= R_) {
            throw std::out_of_range("P index out of range");
        }
        P_.data()[p_index(r, str)] += val;
        return;
    }

    if (which == 1) {
        if (str >= nbeta_strs_ || r >= R_) {
            throw std::out_of_range("Q index out of range");
        }
        Q_.data()[q_index(r, str)] += val;
        return;
    }

    throw std::invalid_argument("RRFCIComputer::add_to_element expects first index 0 (P) or 1 (Q)");
}

void RRFCIComputer::set_p_element(size_t J, size_t r, const std::complex<double> val) {
    if (J >= nalfa_strs_ || r >= R_) {
        throw std::out_of_range("set_p_element index out of range");
    }
    P_.data()[p_index(r, J)] = val;
}

void RRFCIComputer::set_q_element(size_t K, size_t r, const std::complex<double> val) {
    if (K >= nbeta_strs_ || r >= R_) {
        throw std::out_of_range("set_q_element index out of range");
    }
    Q_.data()[q_index(r, K)] = val;
}

std::complex<double> RRFCIComputer::get_p_element(size_t J, size_t r) const {
    if (J >= nalfa_strs_ || r >= R_) {
        throw std::out_of_range("get_p_element index out of range");
    }
    return P_.read_data()[p_index(r, J)];
}

std::complex<double> RRFCIComputer::get_q_element(size_t K, size_t r) const {
    if (K >= nbeta_strs_ || r >= R_) {
        throw std::out_of_range("get_q_element index out of range");
    }
    return Q_.read_data()[q_index(r, K)];
}

std::vector<std::complex<double>> RRFCIComputer::get_P_row(size_t r) const {
    if (r >= R_) {
        throw std::out_of_range("get_P_row: rank index out of range");
    }

    std::vector<std::complex<double>> out(nalfa_strs_, 0.0);
    const auto& pdata = P_.read_data();
    const size_t off = p_index(r, 0);

    for (size_t J = 0; J < nalfa_strs_; ++J) {
        out[J] = pdata[off + J];
    }
    return out;
}

std::vector<std::complex<double>> RRFCIComputer::get_Q_row(size_t r) const {
    if (r >= R_) {
        throw std::out_of_range("get_Q_row: rank index out of range");
    }

    std::vector<std::complex<double>> out(nbeta_strs_, 0.0);
    const auto& qdata = Q_.read_data();
    const size_t off = q_index(r, 0);

    for (size_t K = 0; K < nbeta_strs_; ++K) {
        out[K] = qdata[off + K];
    }
    return out;
}

void RRFCIComputer::set_P_row(size_t r, const std::vector<std::complex<double>>& row) {
    if (r >= R_) {
        throw std::out_of_range("set_P_row: rank index out of range");
    }
    if (row.size() != nalfa_strs_) {
        throw std::invalid_argument("set_P_row length mismatch");
    }

    auto& pdata = P_.data();
    const size_t off = p_index(r, 0);
    for (size_t J = 0; J < nalfa_strs_; ++J) {
        pdata[off + J] = row[J];
    }
}

void RRFCIComputer::set_Q_row(size_t r, const std::vector<std::complex<double>>& row) {
    if (r >= R_) {
        throw std::out_of_range("set_Q_row: rank index out of range");
    }
    if (row.size() != nbeta_strs_) {
        throw std::invalid_argument("set_Q_row length mismatch");
    }

    auto& qdata = Q_.data();
    const size_t off = q_index(r, 0);
    for (size_t K = 0; K < nbeta_strs_; ++K) {
        qdata[off + K] = row[K];
    }
}

std::complex<double> RRFCIComputer::dot_vec(
    const std::vector<std::complex<double>>& a,
    const std::vector<std::complex<double>>& b) const {

    if (a.size() != b.size()) {
        throw std::invalid_argument("dot_vec size mismatch");
    }

    std::complex<double> out = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        out += std::conj(a[i]) * b[i];
    }
    return out;
}

std::complex<double> RRFCIComputer::contract_vec_no_conj(
    const std::vector<std::complex<double>>& a,
    const std::vector<std::complex<double>>& b) const {

    if (a.size() != b.size()) {
        throw std::invalid_argument("contract_vec_no_conj size mismatch");
    }

    std::complex<double> out = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        out += a[i] * b[i];
    }
    return out;
}

void RRFCIComputer::axpy_into_row(
    Tensor& T,
    size_t row,
    const std::vector<std::complex<double>>& vec,
    const std::complex<double>& alpha) {

    const auto shape = T.shape();
    if (shape.size() != 2) {
        throw std::invalid_argument("axpy_into_row expects rank-2 Tensor");
    }
    if (row >= shape[0]) {
        throw std::out_of_range("axpy_into_row row out of range");
    }
    if (vec.size() != shape[1]) {
        throw std::invalid_argument("axpy_into_row vector length mismatch");
    }

    auto& tdata = T.data();
    const size_t off = row * shape[1];
    for (size_t i = 0; i < vec.size(); ++i) {
        tdata[off + i] += alpha * vec[i];
    }
}

void RRFCIComputer::apply_array_1bdy_vector(
    std::vector<std::complex<double>>& out,
    const std::vector<int>& dexc,
    const int states,
    const int ndexc,
    const Tensor& h1e,
    const std::vector<std::complex<double>>& in,
    const int norbs) const {

    (void)norbs;

    out.assign(states, 0.0);
    const auto& h1 = h1e.read_data();

    for (int s1 = 0; s1 < states; ++s1) {
        const int* cdexc = dexc.data() + 3 * s1 * ndexc;
        const int* lim1  = cdexc + 3 * ndexc;

        std::complex<double> accum = 0.0;
        for (; cdexc < lim1; cdexc += 3) {
            const int source  = cdexc[0];
            const int ijshift = cdexc[1];
            const int parity  = cdexc[2];
            accum += static_cast<double>(parity) * h1[ijshift] * in[source];
        }
        out[s1] += accum;
    }
}

void RRFCIComputer::lm_apply_array12_same_spin_opt_vector(
    std::vector<std::complex<double>>& out,
    const std::vector<int>& dexc,
    const int states,
    const int ndexc,
    const Tensor& h1e,
    const Tensor& h2e,
    const std::vector<std::complex<double>>& in,
    const int norbs) const {

    out.assign(states, 0.0);
    std::vector<std::complex<double>> temp(states, 0.0);

    const auto& h1 = h1e.read_data();
    const auto& h2 = h2e.read_data();

    for (int s1 = 0; s1 < states; ++s1) {
        std::fill(temp.begin(), temp.end(), 0.0);

        const int* cdexc = dexc.data() + 3 * s1 * ndexc;
        const int* lim1  = cdexc + 3 * ndexc;

        for (; cdexc < lim1; cdexc += 3) {
            const int s2      = cdexc[0];
            const int ijshift = cdexc[1];
            const int parity1 = cdexc[2];

            temp[s2] += static_cast<double>(parity1) * h1[ijshift];

            const int* cdexc2 = dexc.data() + 3 * s2 * ndexc;
            const int* lim2   = cdexc2 + 3 * ndexc;
            const int h2off   = ijshift * norbs * norbs;

            for (; cdexc2 < lim2; cdexc2 += 3) {
                const int target  = cdexc2[0];
                const int klshift = cdexc2[1];
                const int parity  = cdexc2[2] * parity1;

                temp[target] += static_cast<double>(parity) * h2[h2off + klshift];
            }
        }

        std::complex<double> accum = 0.0;
        for (int ii = 0; ii < states; ++ii) {
            accum += temp[ii] * in[ii];
        }
        out[s1] += accum;
    }
}

void RRFCIComputer::build_transition_vector(
    std::vector<std::complex<double>>& gamma_proj,
    const std::vector<int>& dexc,
    const int states,
    const int ndexc,
    const std::vector<std::complex<double>>& left,
    const std::vector<std::complex<double>>& right,
    const int norbs) const {

    gamma_proj.assign(norbs * norbs, 0.0);

    for (int s1 = 0; s1 < states; ++s1) {
        const int* cdexc = dexc.data() + 3 * s1 * ndexc;
        const int* lim1  = cdexc + 3 * ndexc;

        for (; cdexc < lim1; cdexc += 3) {
            const int source  = cdexc[0];
            const int ijshift = cdexc[1];
            const int parity  = cdexc[2];
            gamma_proj[ijshift] += std::conj(left[s1]) * static_cast<double>(parity) * right[source];
        }
    }
}

void RRFCIComputer::build_transition_vector_no_conj(
    std::vector<std::complex<double>>& gamma_proj,
    const std::vector<int>& dexc,
    const int states,
    const int ndexc,
    const std::vector<std::complex<double>>& left,
    const std::vector<std::complex<double>>& right,
    const int norbs) const {

    gamma_proj.assign(norbs * norbs, 0.0);

    for (int s1 = 0; s1 < states; ++s1) {
        const int* cdexc = dexc.data() + 3 * s1 * ndexc;
        const int* lim1  = cdexc + 3 * ndexc;

        for (; cdexc < lim1; cdexc += 3) {
            const int source  = cdexc[0];
            const int ijshift = cdexc[1];
            const int parity  = cdexc[2];

            gamma_proj[ijshift] += left[s1] * static_cast<double>(parity) * right[source];
        }
    }
}

void RRFCIComputer::build_effective_one_body_from_transition(
    Tensor& heff,
    const std::vector<std::complex<double>>& transition,
    const Tensor& h2e_einsum,
    const int norbs) const {

    heff.zero_with_shape({static_cast<size_t>(norbs), static_cast<size_t>(norbs)});
    heff.set_name("rr_heff");

    auto& hout = heff.data();
    const auto& h2 = h2e_einsum.read_data();

    for (int ij = 0; ij < norbs * norbs; ++ij) {
        const std::complex<double> coeff = transition[ij];
        if (std::abs(coeff) < compute_threshold_) {
            continue;
        }

        const int off = ij * norbs * norbs;
        for (int kl = 0; kl < norbs * norbs; ++kl) {
            hout[kl] += coeff * h2[off + kl];
        }
    }
}

void RRFCIComputer::build_overlap_projections(Tensor& SP, Tensor& SQ) const {
    SP.zero_with_shape({R_, nalfa_strs_});
    SQ.zero_with_shape({R_, nbeta_strs_});
    SP.set_name("RRFCI SP");
    SQ.set_name("RRFCI SQ");

    auto& sp = SP.data();
    auto& sq = SQ.data();
    const auto& p = P_.read_data();
    const auto& q = Q_.read_data();

    std::fill(sp.begin(), sp.end(), std::complex<double>(0.0, 0.0));
    std::fill(sq.begin(), sq.end(), std::complex<double>(0.0, 0.0));

    for (size_t i = 0; i < R_; ++i) {
        for (size_t m = 0; m < R_; ++m) {
            std::complex<double> qdot = 0.0;
            for (size_t K = 0; K < nbeta_strs_; ++K) {
                qdot += std::conj(q[q_index(i, K)]) * q[q_index(m, K)];
            }

            std::complex<double> pdot = 0.0;
            for (size_t J = 0; J < nalfa_strs_; ++J) {
                pdot += std::conj(p[p_index(i, J)]) * p[p_index(m, J)];
            }

            const size_t sp_i_off = p_index(i, 0);
            const size_t sq_i_off = q_index(i, 0);
            const size_t p_m_off  = p_index(m, 0);
            const size_t q_m_off  = q_index(m, 0);

            for (size_t J = 0; J < nalfa_strs_; ++J) {
                sp[sp_i_off + J] += qdot * p[p_m_off + J];
            }

            for (size_t K = 0; K < nbeta_strs_; ++K) {
                sq[sq_i_off + K] += pdot * q[q_m_off + K];
            }
        }
    }
}

// Four updated same spin functions, unique to alfa/beta
void RRFCIComputer::rr_apply_same_spin_alpha_to_sigmaP(
    Tensor& outP,
    const Tensor& leftQ,
    const Tensor& rightQ,
    const Tensor& rightP,
    const Tensor& h1e,
    const Tensor& h2e,
    const int norbs) {

    outP.zero_with_shape({R_, nalfa_strs_});
    outP.set_name("rr_sigmaP_same_alpha");
    std::fill(outP.data().begin(), outP.data().end(), std::complex<double>(0.0, 0.0));

    const auto& lq = leftQ.read_data();
    const auto& rq = rightQ.read_data();
    const auto& rp = rightP.read_data();

    for (size_t i = 0; i < R_; ++i) {
        std::vector<std::complex<double>> Qi(nbeta_strs_, 0.0);
        for (size_t K = 0; K < nbeta_strs_; ++K) {
            Qi[K] = lq[q_index(i, K)];
        }

        for (size_t m = 0; m < R_; ++m) {
            std::vector<std::complex<double>> Qm(nbeta_strs_, 0.0);
            std::vector<std::complex<double>> Pm(nalfa_strs_, 0.0);

            for (size_t K = 0; K < nbeta_strs_; ++K) {
                Qm[K] = rq[q_index(m, K)];
            }
            for (size_t J = 0; J < nalfa_strs_; ++J) {
                Pm[J] = rp[p_index(m, J)];
            }

            const std::complex<double> qdot = dot_vec(Qi, Qm);
            // const std::complex<double> qdot = contract_vec_no_conj(Qi, Qm);
            

            std::vector<std::complex<double>> APm;
            lm_apply_array12_same_spin_opt_vector(
                APm,
                graph_.read_dexca_vec(),
                static_cast<int>(nalfa_strs_),
                graph_.get_ndexca(),
                h1e,
                h2e,
                Pm,
                norbs);

            axpy_into_row(outP, i, APm, qdot);
        }
    }
}

void RRFCIComputer::rr_apply_same_spin_beta_to_sigmaP(
    Tensor& outP,
    const Tensor& leftQ,
    const Tensor& rightQ,
    const Tensor& rightP,
    const Tensor& h1e,
    const Tensor& h2e,
    const int norbs) {

    outP.zero_with_shape({R_, nalfa_strs_});
    outP.set_name("rr_sigmaP_same_beta");
    std::fill(outP.data().begin(), outP.data().end(), std::complex<double>(0.0, 0.0));

    const auto& lq = leftQ.read_data();
    const auto& rq = rightQ.read_data();
    const auto& rp = rightP.read_data();

    for (size_t i = 0; i < R_; ++i) {
        std::vector<std::complex<double>> Qi(nbeta_strs_, 0.0);
        for (size_t K = 0; K < nbeta_strs_; ++K) {
            Qi[K] = lq[q_index(i, K)];
        }

        for (size_t m = 0; m < R_; ++m) {
            std::vector<std::complex<double>> Qm(nbeta_strs_, 0.0);
            std::vector<std::complex<double>> Pm(nalfa_strs_, 0.0);

            for (size_t K = 0; K < nbeta_strs_; ++K) {
                Qm[K] = rq[q_index(m, K)];
            }
            for (size_t J = 0; J < nalfa_strs_; ++J) {
                Pm[J] = rp[p_index(m, J)];
            }

            std::vector<std::complex<double>> AQm;
            lm_apply_array12_same_spin_opt_vector(
                AQm,
                graph_.read_dexcb_vec(),
                static_cast<int>(nbeta_strs_),
                graph_.get_ndexcb(),
                h1e,
                h2e,
                Qm,
                norbs);

            const std::complex<double> scalar = dot_vec(Qi, AQm);
            // const std::complex<double> scalar = contract_vec_no_conj(Qi, AQm);
            axpy_into_row(outP, i, Pm, scalar);
        }
    }
}

void RRFCIComputer::rr_apply_same_spin_alpha_to_sigmaQ(
    Tensor& outQ,
    const Tensor& leftP,
    const Tensor& rightP,
    const Tensor& rightQ,
    const Tensor& h1e,
    const Tensor& h2e,
    const int norbs) {

    outQ.zero_with_shape({R_, nbeta_strs_});
    outQ.set_name("rr_sigmaQ_same_alpha");
    std::fill(outQ.data().begin(), outQ.data().end(), std::complex<double>(0.0, 0.0));

    const auto& lp = leftP.read_data();
    const auto& rp = rightP.read_data();
    const auto& rq = rightQ.read_data();

    for (size_t i = 0; i < R_; ++i) {
        std::vector<std::complex<double>> Pi(nalfa_strs_, 0.0);
        for (size_t J = 0; J < nalfa_strs_; ++J) {
            Pi[J] = lp[p_index(i, J)];
        }

        for (size_t m = 0; m < R_; ++m) {
            std::vector<std::complex<double>> Pm(nalfa_strs_, 0.0);
            std::vector<std::complex<double>> Qm(nbeta_strs_, 0.0);

            for (size_t J = 0; J < nalfa_strs_; ++J) {
                Pm[J] = rp[p_index(m, J)];
            }
            for (size_t K = 0; K < nbeta_strs_; ++K) {
                Qm[K] = rq[q_index(m, K)];
            }

            std::vector<std::complex<double>> APm;
            lm_apply_array12_same_spin_opt_vector(
                APm,
                graph_.read_dexca_vec(),
                static_cast<int>(nalfa_strs_),
                graph_.get_ndexca(),
                h1e,
                h2e,
                Pm,
                norbs);

            const std::complex<double> scalar = dot_vec(Pi, APm);
            // const std::complex<double> scalar = contract_vec_no_conj(Pi, APm);
            axpy_into_row(outQ, i, Qm, scalar);
        }
    }
}

void RRFCIComputer::rr_apply_same_spin_beta_to_sigmaQ(
    Tensor& outQ,
    const Tensor& leftP,
    const Tensor& rightP,
    const Tensor& rightQ,
    const Tensor& h1e,
    const Tensor& h2e,
    const int norbs) {

    outQ.zero_with_shape({R_, nbeta_strs_});
    outQ.set_name("rr_sigmaQ_same_beta");
    std::fill(outQ.data().begin(), outQ.data().end(), std::complex<double>(0.0, 0.0));

    const auto& lp = leftP.read_data();
    const auto& rp = rightP.read_data();
    const auto& rq = rightQ.read_data();

    for (size_t i = 0; i < R_; ++i) {
        std::vector<std::complex<double>> Pi(nalfa_strs_, 0.0);
        for (size_t J = 0; J < nalfa_strs_; ++J) {
            Pi[J] = lp[p_index(i, J)];
        }

        for (size_t m = 0; m < R_; ++m) {
            std::vector<std::complex<double>> Pm(nalfa_strs_, 0.0);
            std::vector<std::complex<double>> Qm(nbeta_strs_, 0.0);

            for (size_t J = 0; J < nalfa_strs_; ++J) {
                Pm[J] = rp[p_index(m, J)];
            }
            for (size_t K = 0; K < nbeta_strs_; ++K) {
                Qm[K] = rq[q_index(m, K)];
            }

            const std::complex<double> pdot = dot_vec(Pi, Pm);
            // const std::complex<double> pdot = contract_vec_no_conj(Pi, Pm);

            std::vector<std::complex<double>> AQm;
            lm_apply_array12_same_spin_opt_vector(
                AQm,
                graph_.read_dexcb_vec(),
                static_cast<int>(nbeta_strs_),
                graph_.get_ndexcb(),
                h1e,
                h2e,
                Qm,
                norbs);

            axpy_into_row(outQ, i, AQm, pdot);
        }
    }
}

// ==> old version below, to be removed after testing new ones above

void RRFCIComputer::rr_apply_array12_same_spin_opt_P(
    Tensor& outP,
    const Tensor& leftQ,
    const Tensor& rightQ,
    const Tensor& rightP,
    const std::vector<int>& dexc,
    const int alpha_states,
    const int ndexc,
    const Tensor& h1e,
    const Tensor& h2e,
    const int norbs) {

    outP.zero_with_shape({R_, nalfa_strs_});
    outP.set_name("rr_sigmaP_same");
    std::fill(outP.data().begin(), outP.data().end(), std::complex<double>(0.0, 0.0));

    const auto& lq = leftQ.read_data();
    const auto& rq = rightQ.read_data();
    const auto& rp = rightP.read_data();

    for (size_t i = 0; i < R_; ++i) {
        std::vector<std::complex<double>> Qi(nbeta_strs_, 0.0);
        for (size_t K = 0; K < nbeta_strs_; ++K) {
            Qi[K] = lq[q_index(i, K)];
        }

        for (size_t m = 0; m < R_; ++m) {
            std::vector<std::complex<double>> Qm(nbeta_strs_, 0.0);
            std::vector<std::complex<double>> Pm(alpha_states, 0.0);

            for (size_t K = 0; K < nbeta_strs_; ++K) {
                Qm[K] = rq[q_index(m, K)];
            }
            for (int J = 0; J < alpha_states; ++J) {
                Pm[J] = rp[p_index(m, static_cast<size_t>(J))];
            }

            const std::complex<double> qdot = dot_vec(Qi, Qm);

            std::vector<std::complex<double>> APm;
            lm_apply_array12_same_spin_opt_vector(APm, dexc, alpha_states, ndexc, h1e, h2e, Pm, norbs);

            axpy_into_row(outP, i, APm, qdot);
        }
    }
}

void RRFCIComputer::rr_apply_array12_same_spin_opt_Q(
    Tensor& outQ,
    const Tensor& leftP,
    const Tensor& rightP,
    const Tensor& rightQ,
    const std::vector<int>& dexc,
    const int beta_states,
    const int ndexc,
    const Tensor& h1e,
    const Tensor& h2e,
    const int norbs) {

    outQ.zero_with_shape({R_, nbeta_strs_});
    outQ.set_name("rr_sigmaQ_same");
    std::fill(outQ.data().begin(), outQ.data().end(), std::complex<double>(0.0, 0.0));

    const auto& lp = leftP.read_data();
    const auto& rp = rightP.read_data();
    const auto& rq = rightQ.read_data();

    for (size_t i = 0; i < R_; ++i) {
        std::vector<std::complex<double>> Pi(nalfa_strs_, 0.0);
        for (size_t J = 0; J < nalfa_strs_; ++J) {
            Pi[J] = lp[p_index(i, J)];
        }

        for (size_t m = 0; m < R_; ++m) {
            std::vector<std::complex<double>> Pm(beta_states, 0.0);
            std::vector<std::complex<double>> Qm(nbeta_strs_, 0.0);

            for (int J = 0; J < beta_states; ++J) {
                Pm[J] = rp[p_index(m, static_cast<size_t>(J))];
            }
            for (size_t K = 0; K < nbeta_strs_; ++K) {
                Qm[K] = rq[q_index(m, K)];
            }

            std::vector<std::complex<double>> APm;
            lm_apply_array12_same_spin_opt_vector(APm, dexc, beta_states, ndexc, h1e, h2e, Pm, norbs);

            const std::complex<double> scalar = dot_vec(Pi, APm);
            axpy_into_row(outQ, i, Qm, scalar);
        }
    }
}

// ==> old functions above, to be removed after testing new ones 

void RRFCIComputer::rr_apply_array12_diff_spin_opt_P(
    Tensor& outP,
    const Tensor& leftQ,
    const Tensor& rightQ,
    const Tensor& rightP,
    const std::vector<int>& adexc,
    const std::vector<int>& bdexc,
    const int alpha_states,
    const int beta_states,
    const int nadexc,
    const int nbdexc,
    const Tensor& h2e_einsum,
    const int norbs) {

    (void)adexc;

    outP.zero_with_shape({R_, nalfa_strs_});
    outP.set_name("rr_sigmaP_diff");
    std::fill(outP.data().begin(), outP.data().end(), std::complex<double>(0.0, 0.0));

    const auto& lq = leftQ.read_data();
    const auto& rq = rightQ.read_data();
    const auto& rp = rightP.read_data();

    for (size_t i = 0; i < R_; ++i) {
        std::vector<std::complex<double>> Qi(beta_states, 0.0);
        for (int K = 0; K < beta_states; ++K) {
            Qi[K] = lq[q_index(i, static_cast<size_t>(K))];
        }

        for (size_t m = 0; m < R_; ++m) {
            std::vector<std::complex<double>> Qm(beta_states, 0.0);
            std::vector<std::complex<double>> Pm(alpha_states, 0.0);

            for (int K = 0; K < beta_states; ++K) {
                Qm[K] = rq[q_index(m, static_cast<size_t>(K))];
            }
            for (int J = 0; J < alpha_states; ++J) {
                Pm[J] = rp[p_index(m, static_cast<size_t>(J))];
            }

            std::vector<std::complex<double>> beta_transition;
            build_transition_vector(beta_transition, bdexc, beta_states, nbdexc, Qi, Qm, norbs);
            // build_transition_vector_no_conj(beta_transition, bdexc, beta_states, nbdexc, Qi, Qm, norbs);

            Tensor heff_alpha;
            build_effective_one_body_from_transition(heff_alpha, beta_transition, h2e_einsum, norbs);

            std::vector<std::complex<double>> contrib;
            apply_array_1bdy_vector(
                contrib,
                graph_.read_dexca_vec(),
                alpha_states,
                nadexc,
                heff_alpha,
                Pm,
                norbs);

            axpy_into_row(outP, i, contrib, 1.0);
        }
    }
}

void RRFCIComputer::rr_apply_array12_diff_spin_opt_Q(
    Tensor& outQ,
    const Tensor& leftP,
    const Tensor& rightP,
    const Tensor& rightQ,
    const std::vector<int>& adexc,
    const std::vector<int>& bdexc,
    const int alpha_states,
    const int beta_states,
    const int nadexc,
    const int nbdexc,
    const Tensor& h2e_einsum,
    const int norbs) {

    (void)bdexc;

    outQ.zero_with_shape({R_, nbeta_strs_});
    outQ.set_name("rr_sigmaQ_diff");
    std::fill(outQ.data().begin(), outQ.data().end(), std::complex<double>(0.0, 0.0));

    const auto& lp = leftP.read_data();
    const auto& rp = rightP.read_data();
    const auto& rq = rightQ.read_data();

    for (size_t i = 0; i < R_; ++i) {
        std::vector<std::complex<double>> Pi(alpha_states, 0.0);
        for (int J = 0; J < alpha_states; ++J) {
            Pi[J] = lp[p_index(i, static_cast<size_t>(J))];
        }

        for (size_t m = 0; m < R_; ++m) {
            std::vector<std::complex<double>> Pm(alpha_states, 0.0);
            std::vector<std::complex<double>> Qm(beta_states, 0.0);

            for (int J = 0; J < alpha_states; ++J) {
                Pm[J] = rp[p_index(m, static_cast<size_t>(J))];
            }
            for (int K = 0; K < beta_states; ++K) {
                Qm[K] = rq[q_index(m, static_cast<size_t>(K))];
            }

            std::vector<std::complex<double>> alpha_transition;
            build_transition_vector(alpha_transition, adexc, alpha_states, nadexc, Pi, Pm, norbs);
            // build_transition_vector_no_conj(alpha_transition, adexc, alpha_states, nadexc, Pi, Pm, norbs);

            Tensor heff_beta;
            build_effective_one_body_from_transition(heff_beta, alpha_transition, h2e_einsum, norbs);

            std::vector<std::complex<double>> contrib;
            apply_array_1bdy_vector(
                contrib,
                graph_.read_dexcb_vec(),
                beta_states,
                nbdexc,
                heff_beta,
                Qm,
                norbs);

            axpy_into_row(outQ, i, contrib, 1.0);
        }
    }
}

void RRFCIComputer::apply_tensor_spat_12bdy(
    const Tensor& h1e,
    const Tensor& h2e,
    const Tensor& h2e_einsum,
    size_t norb) {

    if (h1e.size() != norb * norb) {
        throw std::invalid_argument("Expecting h1e to be nmo x nmo for RRFCIComputer::apply_tensor_spat_12bdy");
    }

    if (h2e.size() != norb * norb * norb * norb) {
        throw std::invalid_argument("Expecting h2e to be nmo x nmo x nmo x nmo for RRFCIComputer::apply_tensor_spat_12bdy");
    }

    Tensor sigmaP_same_alpha;
    Tensor sigmaP_same_beta;
    Tensor sigmaQ_same_alpha;
    Tensor sigmaQ_same_beta;
    Tensor sigmaP_diff;
    Tensor sigmaQ_diff;

    sigmaP_same_alpha.zero_with_shape({R_, nalfa_strs_});
    sigmaP_same_beta.zero_with_shape({R_, nalfa_strs_});
    sigmaQ_same_alpha.zero_with_shape({R_, nalfa_strs_});
    sigmaQ_same_beta.zero_with_shape({R_, nalfa_strs_});
    sigmaP_diff.zero_with_shape({R_, nalfa_strs_});
    sigmaQ_diff.zero_with_shape({R_, nalfa_strs_});


    // Same-spin alpha contribution
    rr_apply_same_spin_alpha_to_sigmaP(
        sigmaP_same_alpha,
        Q_, Q_, P_,
        h1e, h2e,
        static_cast<int>(norb_));

    rr_apply_same_spin_alpha_to_sigmaQ(
        sigmaQ_same_alpha,
        P_, P_, Q_,
        h1e, h2e,
        static_cast<int>(norb_));

    // Same-spin beta contribution
    rr_apply_same_spin_beta_to_sigmaP(
        sigmaP_same_beta,
        Q_, Q_, P_,
        h1e, h2e,
        static_cast<int>(norb_));

    rr_apply_same_spin_beta_to_sigmaQ(
        sigmaQ_same_beta,
        P_, P_, Q_,
        h1e, h2e,
        static_cast<int>(norb_));

    // // Mixed-spin contribution
    rr_apply_array12_diff_spin_opt_P(
        sigmaP_diff,
        Q_, Q_, P_,
        graph_.read_dexca_vec(),
        graph_.read_dexcb_vec(),
        static_cast<int>(nalfa_strs_),
        static_cast<int>(nbeta_strs_),
        graph_.get_ndexca(),
        graph_.get_ndexcb(),
        h2e_einsum,
        static_cast<int>(norb_));

    rr_apply_array12_diff_spin_opt_Q(
        sigmaQ_diff,
        P_, P_, Q_,
        graph_.read_dexca_vec(),
        graph_.read_dexcb_vec(),
        static_cast<int>(nalfa_strs_),
        static_cast<int>(nbeta_strs_),
        graph_.get_ndexca(),
        graph_.get_ndexcb(),
        h2e_einsum,
        static_cast<int>(norb_));

    SigmaP_.zero_with_shape({R_, nalfa_strs_});
    SigmaQ_.zero_with_shape({R_, nbeta_strs_});
    SigmaP_.set_name("RRFCI SigmaP");
    SigmaQ_.set_name("RRFCI SigmaQ");

    auto& sigp = SigmaP_.data();
    auto& sigq = SigmaQ_.data();

    const auto& spa = sigmaP_same_alpha.read_data();
    const auto& spb = sigmaP_same_beta.read_data();
    const auto& spd = sigmaP_diff.read_data();

    const auto& sqa = sigmaQ_same_alpha.read_data();
    const auto& sqb = sigmaQ_same_beta.read_data();
    const auto& sqd = sigmaQ_diff.read_data();

    for (size_t idx = 0; idx < sigp.size(); ++idx) {
        sigp[idx] = spa[idx] + spb[idx] + spd[idx];
    }

    for (size_t idx = 0; idx < sigq.size(); ++idx) {
        sigq[idx] = sqa[idx] + sqb[idx] + sqd[idx];
    }

    P_ = SigmaP_;
    Q_ = SigmaQ_;
}

// void RRFCIComputer::apply_tensor_spat_12bdy(
//     const Tensor& h1e,
//     const Tensor& h2e,
//     const Tensor& h2e_einsum,
//     size_t norb) {

//     if (h1e.size() != norb * norb) {
//         throw std::invalid_argument("Expecting h1e to be nmo x nmo for RRFCIComputer::apply_tensor_spat_12bdy");
//     }

//     if (h2e.size() != norb * norb * norb * norb) {
//         throw std::invalid_argument("Expecting h2e to be nmo x nmo x nmo x nmo for RRFCIComputer::apply_tensor_spat_12bdy");
//     }

//     Tensor sigmaP_same_alpha;
//     Tensor sigmaQ_same_alpha({R_, nbeta_strs_}, "sigmaQ_same_alpha");
//     Tensor sigmaP_same_beta({R_, nalfa_strs_}, "sigmaP_same_beta");
//     Tensor sigmaQ_same_beta;
//     Tensor sigmaP_diff;
//     Tensor sigmaQ_diff;

//     rr_apply_array12_same_spin_opt_P(
//         sigmaP_same_alpha,
//         Q_, Q_, P_,
//         graph_.read_dexca_vec(),
//         static_cast<int>(nalfa_strs_),
//         graph_.get_ndexca(),
//         h1e, h2e,
//         static_cast<int>(norb_));

//     // rr_apply_array12_same_spin_opt_Q(
//     //     sigmaQ_same_alpha,
//     //     P_, P_, Q_,
//     //     graph_.read_dexca_vec(),
//     //     static_cast<int>(nbeta_strs_),
//     //     graph_.get_ndexca(),
//     //     h1e, h2e,
//     //     static_cast<int>(norb_));

//     // rr_apply_array12_same_spin_opt_P(
//     //     sigmaP_same_beta,
//     //     Q_, Q_, P_,
//     //     graph_.read_dexcb_vec(),
//     //     static_cast<int>(nalfa_strs_),
//     //     graph_.get_ndexcb(),
//     //     h1e, h2e,
//     //     static_cast<int>(norb_));

//     rr_apply_array12_same_spin_opt_Q(
//         sigmaQ_same_beta,
//         P_, P_, Q_,
//         graph_.read_dexcb_vec(),
//         static_cast<int>(nbeta_strs_),
//         graph_.get_ndexcb(),
//         h1e, h2e,
//         static_cast<int>(norb_));

//     rr_apply_array12_diff_spin_opt_P(
//         sigmaP_diff,
//         Q_, Q_, P_,
//         graph_.read_dexca_vec(),
//         graph_.read_dexcb_vec(),
//         static_cast<int>(nalfa_strs_),
//         static_cast<int>(nbeta_strs_),
//         graph_.get_ndexca(),
//         graph_.get_ndexcb(),
//         h2e_einsum,
//         static_cast<int>(norb_));

//     rr_apply_array12_diff_spin_opt_Q(
//         sigmaQ_diff,
//         P_, P_, Q_,
//         graph_.read_dexca_vec(),
//         graph_.read_dexcb_vec(),
//         static_cast<int>(nalfa_strs_),
//         static_cast<int>(nbeta_strs_),
//         graph_.get_ndexca(),
//         graph_.get_ndexcb(),
//         h2e_einsum,
//         static_cast<int>(norb_));

//     // Persistent work buffers
//     SigmaP_.zero_with_shape({R_, nalfa_strs_});
//     SigmaQ_.zero_with_shape({R_, nbeta_strs_});
//     SigmaP_.set_name("RRFCI SigmaP");
//     SigmaQ_.set_name("RRFCI SigmaQ");

//     auto& sigp = SigmaP_.data();
//     auto& sigq = SigmaQ_.data();

//     const auto& spa = sigmaP_same_alpha.read_data();
//     const auto& spb = sigmaP_same_beta.read_data();
//     const auto& spd = sigmaP_diff.read_data();

//     const auto& sqa = sigmaQ_same_alpha.read_data();
//     const auto& sqb = sigmaQ_same_beta.read_data();
//     const auto& sqd = sigmaQ_diff.read_data();

//     for (size_t idx = 0; idx < sigp.size(); ++idx) {
//         sigp[idx] = spa[idx] + spb[idx] + spd[idx];
//     }

//     for (size_t idx = 0; idx < sigq.size(); ++idx) {
//         sigq[idx] = sqa[idx] + sqb[idx] + sqd[idx];
//     }

//     // Match FCIComputer semantics: applying H overwrites the current state.
//     P_ = SigmaP_;
//     Q_ = SigmaQ_;
// }

void RRFCIComputer::apply_tensor_spat_012bdy(
    const std::complex<double> h0e,
    const Tensor& h1e,
    const Tensor& h2e,
    const Tensor& h2e_einsum,
    size_t norb) {

    Tensor Ptemp = P_;
    Tensor Qtemp = Q_;

    apply_tensor_spat_12bdy(
        h1e,
        h2e,
        h2e_einsum,
        norb);

    // Match FCIComputer semantics:
    // new_state = H old_state + h0e * old_state
    auto& p = P_.data();
    auto& q = Q_.data();
    const auto& ptemp = Ptemp.read_data();
    const auto& qtemp = Qtemp.read_data();

    if (p.size() != ptemp.size() || q.size() != qtemp.size()) {
        throw std::runtime_error("Internal RRFCIComputer shape mismatch in apply_tensor_spat_012bdy");
    }

    for (size_t idx = 0; idx < p.size(); ++idx) {
        p[idx] += h0e * ptemp[idx];
    }

    for (size_t idx = 0; idx < q.size(); ++idx) {
        q[idx] += h0e * qtemp[idx];
    }

    // Keep persistent sigmas aligned with the applied result
    SigmaP_ = P_;
    SigmaQ_ = Q_;
}

void RRFCIComputer::hartree_fock() {
    // Set the RR state to the exact rank-1 HF determinant:
    // C_{JK} = delta_{J,0} delta_{K,0}
    //
    // With the current transposed physical storage:
    //   P_(r, J), Q_(r, K)
    //
    // We use only rank channel r = 0 and zero all others.

    std::fill(P_.data().begin(), P_.data().end(), std::complex<double>(0.0, 0.0));
    std::fill(Q_.data().begin(), Q_.data().end(), std::complex<double>(0.0, 0.0));
    std::fill(SigmaP_.data().begin(), SigmaP_.data().end(), std::complex<double>(0.0, 0.0));
    std::fill(SigmaQ_.data().begin(), SigmaQ_.data().end(), std::complex<double>(0.0, 0.0));

    if (R_ == 0 || nalfa_strs_ == 0 || nbeta_strs_ == 0) {
        return;
    }

    // HF alpha string assumed to be index 0
    // HF beta  string assumed to be index 0
    P_.data()[p_index(0, 0)] = std::complex<double>(1.0, 0.0);
    Q_.data()[q_index(0, 0)] = std::complex<double>(1.0, 0.0);
}

std::complex<double> RRFCIComputer::get_hf_dot() const {
    // Overlap with the HF determinant |J=0, K=0>.
    //
    // Since C_{00} = sum_r P_r(0) Q_r(0),
    // the overlap is exactly that coefficient.

    if (nalfa_strs_ == 0 || nbeta_strs_ == 0) {
        return std::complex<double>(0.0, 0.0);
    }

    const auto& p = P_.read_data();
    const auto& q = Q_.read_data();

    std::complex<double> out = 0.0;
    for (size_t r = 0; r < R_; ++r) {
        out += p[p_index(r, 0)] * q[q_index(r, 0)];
    }

    return out;
}

std::complex<double> RRFCIComputer::get_overlap(const RRFCIComputer& other) const {
    // Computes
    //
    // <this | other>
    // =
    // sum_{r,m}
    //   ( sum_J conj(P_r(J)) P'_m(J) )
    //   ( sum_K conj(Q_r(K)) Q'_m(K) )
    //
    // assuming both objects have the same string dimensions and rank-space layout.

    if (nalfa_strs_ != other.nalfa_strs_ ||
        nbeta_strs_ != other.nbeta_strs_) {
        throw std::invalid_argument("RRFCIComputer::get_overlap dimension mismatch");
    }

    const auto& pA = P_.read_data();
    const auto& qA = Q_.read_data();
    const auto& pB = other.P_.read_data();
    const auto& qB = other.Q_.read_data();

    std::complex<double> out = 0.0;

    for (size_t r = 0; r < R_; ++r) {
        for (size_t m = 0; m < other.R_; ++m) {
            std::complex<double> pdot = 0.0;
            for (size_t J = 0; J < nalfa_strs_; ++J) {
                pdot += std::conj(pA[p_index(r, J)]) * pB[other.p_index(m, J)];
            }

            std::complex<double> qdot = 0.0;
            for (size_t K = 0; K < nbeta_strs_; ++K) {
                qdot += std::conj(qA[q_index(r, K)]) * qB[other.q_index(m, K)];
            }

            out += pdot * qdot;
        }
    }

    return out;
}

std::complex<double> RRFCIComputer::pq_dot_product(const Tensor& Pother, const Tensor& Qother) const {
    // Computes the overlap of this RR state with another specified by (Pother, Qother).
    //
    // <this | other> = sum_{r,m} (sum_J conj(P_r(J)) Pother_m(J)) (sum_K conj(Q_r(K)) Qother_m(K))
    //
    // This is a lower-level version of get_overlap that can be used when we have raw P/Q tensors on hand.

    if (nalfa_strs_ != Pother.shape()[1] || nbeta_strs_ != Qother.shape()[1]) {
        throw std::invalid_argument("RRFCIComputer::pq_dot_product dimension mismatch");
    }

    const auto& pA = P_.read_data();
    const auto& qA = Q_.read_data();
    const auto& pB = Pother.read_data();
    const auto& qB = Qother.read_data();

    std::complex<double> out = 0.0;

    for (size_t r = 0; r < R_; ++r) {
        for (size_t m = 0; m < Pother.shape()[0]; ++m) {
            std::complex<double> pdot = 0.0;
            for (size_t J = 0; J < nalfa_strs_; ++J) {
                pdot += std::conj(pA[p_index(r, J)]) * pB[m * nalfa_strs_ + J];
            }

            std::complex<double> qdot = 0.0;
            for (size_t K = 0; K < nbeta_strs_; ++K) {
                qdot += std::conj(qA[q_index(r, K)]) * qB[m * nbeta_strs_ + K];
            }

            out += pdot * qdot;
        }
    }

    return out;
}

// std::complex<double> RRFCIComputer::get_exp_val_tensor(
//     const std::complex<double> h0e,
//     const Tensor& h1e,
//     const Tensor& h2e,
//     const Tensor& h2e_einsum,
//     size_t norb) const {

//     // Returns
//     //
//     // <Psi|H|Psi> / <Psi|Psi>
//     //
//     // without mutating *this.
//     //
//     // We form a copy, apply H to that copy, then evaluate the RR overlap.

//     RRFCIComputer sigma_state(*this);
//     sigma_state.apply_tensor_spat_012bdy(h0e, h1e, h2e, h2e_einsum, norb);

//     const std::complex<double> denom = this->get_overlap(*this);
//     if (std::abs(denom) < 1.0e-14) {
//         throw std::runtime_error("RRFCIComputer::get_exp_val_tensor found near-zero norm");
//     }

//     const std::complex<double> numer = this->get_overlap(sigma_state);
//     return numer / denom;
// }


// std::complex<double> RRFCIComputer::get_exp_val_tensor(
//     const std::complex<double> h0e, 
//     const Tensor& h1e, 
//     const Tensor& h2e, 
//     const Tensor& h2e_einsum, 
//     size_t norb)  
// {
//     Tensor Pin = P_;
//     Tensor Qin = Q_;

//     apply_tensor_spat_012bdy(
//         h0e,
//         h1e, 
//         h2e, 
//         h2e_einsum, 
//         norb
//     );

//     std::complex<double> val = pq_dot_product(Pin, Qin);

//     P_ = Pin;
//     Q_ = Qin;

//     return val;
// }

std::complex<double> RRFCIComputer::get_exp_val_tensor(
    const std::complex<double> h0e,
    const Tensor& h1e,
    const Tensor& h2e,
    const Tensor& h2e_einsum,
    size_t norb) {

    // Make a copy so we do not mutate the current state.
    RRFCIComputer sigma_state(*this);

    // Apply H to the copy.
    // After this call, sigma_state.P_ holds SigmaP,
    // and sigma_state.Q_ holds SigmaQ,
    // i.e. projected sigma vectors built from the ORIGINAL state.
    sigma_state.apply_tensor_spat_012bdy(h0e, h1e, h2e, h2e_einsum, norb);

    const auto& p_orig = P_.read_data();
    const auto& sigP   = sigma_state.P_.read_data();

    // Numerator:
    // <C|HC> = sum_i <P_i | SigmaP_i>
    std::complex<double> numer = 0.0;
    for (size_t r = 0; r < R_; ++r) {
        for (size_t J = 0; J < nalfa_strs_; ++J) {
            numer += std::conj(p_orig[p_index(r, J)]) * sigP[sigma_state.p_index(r, J)];
        }
    }

    // Denominator:
    // <C|C>
    const std::complex<double> denom = this->get_overlap(*this);

    if (std::abs(denom) < 1.0e-14) {
        throw std::runtime_error("RRFCIComputer::get_exp_val_tensor found near-zero norm");
    }

    return numer / denom;
}

Tensor RRFCIComputer::reconstruct_C() const {
    // Reconstruct the full CI coefficient matrix
    //
    //   C(J,K) = sum_r P(r,J) * Q(r,K)
    //
    // Returned Tensor shape:
    //   {nalfa_strs_, nbeta_strs_}
    //
    // This is dense and intended primarily for debugging / sanity checks.

    Tensor Crr;
    Crr.zero_with_shape({nalfa_strs_, nbeta_strs_});
    Crr.set_name("RRFCI reconstructed C");

    auto& c = Crr.data();
    const auto& p = P_.read_data();
    const auto& q = Q_.read_data();

    for (size_t J = 0; J < nalfa_strs_; ++J) {
        const size_t c_row_off = J * nbeta_strs_;
        for (size_t K = 0; K < nbeta_strs_; ++K) {
            std::complex<double> val = 0.0;
            for (size_t r = 0; r < R_; ++r) {
                val += p[p_index(r, J)] * q[q_index(r, K)];
            }
            c[c_row_off + K] = val;
        }
    }

    return Crr;
}