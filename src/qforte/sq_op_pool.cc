#include "helpers.h"
#include "gate.h"
#include "circuit.h"
#include "qubit_operator.h"
#include "sq_operator.h"
#include "qubit_op_pool.h"
#include "sq_op_pool.h"
#include "df_hamiltonian.h"
#include "tensor.h"
#include "fci_computer.h"

#include "qubit_basis.h"

#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <set>
#include <sstream>
#include <tuple>
#include <random>

// #include <functional>
#include <unordered_set>
// #include <bitset>
// #include <cstdint>
// #include <math>

void SQOpPool::add_term(std::complex<double> coeff, const SQOperator& sq_op ){
    terms_.push_back(std::make_pair(coeff, sq_op));
}

void SQOpPool::set_orb_irreps(const std::vector<int>& orb_irreps_to_int, int target_irrep) {
    const int norb = nocc_ + nvir_;
    if (norb > 0 && static_cast<int>(orb_irreps_to_int.size()) != norb) {
        throw std::invalid_argument("Orbital irrep vector length must equal the number of spatial orbitals.");
    }

    orb_irreps_to_int_ = orb_irreps_to_int;
    target_irrep_ = target_irrep;
}

bool SQOpPool::excitation_irrep_allowed(
    const std::vector<size_t>& creators,
    const std::vector<size_t>& annihilators) const
{
    if (creators.size() != annihilators.size()) {
        return false;
    }

    if (orb_irreps_to_int_.empty()) {
        return true;
    }

    int sym = 0;
    for (const auto idx : creators) {
        const size_t spatial_idx = idx / 2;
        if (spatial_idx >= orb_irreps_to_int_.size()) {
            throw std::invalid_argument("Excitation creator index is outside the stored orbital irrep space.");
        }
        sym ^= orb_irreps_to_int_[spatial_idx];
    }
    for (const auto idx : annihilators) {
        const size_t spatial_idx = idx / 2;
        if (spatial_idx >= orb_irreps_to_int_.size()) {
            throw std::invalid_argument("Excitation annihilator index is outside the stored orbital irrep space.");
        }
        sym ^= orb_irreps_to_int_[spatial_idx];
    }

    return sym == target_irrep_;
}

bool SQOpPool::add_pool_operator(std::complex<double> coeff, SQOperator sq_op) {
    sq_op.simplify();
    if (sq_op.terms().empty()) {
        return false;
    }

    // Generated UCC pool operators are anti-Hermitian combinations of terms
    // with the same excitation irrep. Check every term so spin-adapted
    // combinations cannot smuggle in a disallowed component.
    for (const auto& term : sq_op.terms()) {
        if (!excitation_irrep_allowed(std::get<1>(term), std::get<2>(term))) {
            return false;
        }
    }

    terms_.push_back(std::make_pair(coeff, sq_op));
    return true;
}

/// NICK: This funcion is working but needs testing for edge cases!!
void SQOpPool::add_hermitian_pairs(std::complex<double> coeff, const SQOperator& sq_op ){
    std::vector<std::pair< std::vector<size_t>, std::vector<size_t>>> h_vec;
    std::vector<std::pair< std::vector<size_t>, std::vector<size_t>>> hd_vec;

    for (size_t l = 0; l < sq_op.terms().size(); l++){
        std::pair< std::vector<size_t>, std::vector<size_t>> h;
        std::pair< std::vector<size_t>, std::vector<size_t>> hd;

        std::complex<double> hl = std::get<0>(sq_op.terms()[l]);
        h.first  = std::get<1>(sq_op.terms()[l]);
        h.second = std::get<2>(sq_op.terms()[l]);

        hd.first = h.second;
        hd.second = h.first;

        std::reverse(hd.first.begin(), hd.first.end());
        std::reverse(hd.second.begin(), hd.second.end());

        std::sort(h.first.begin(), h.first.end());
        std::sort(h.second.begin(), h.second.end());

        std::sort(hd.first.begin(), hd.first.end());
        std::sort(hd.second.begin(), hd.second.end());
        
        // Determine if term is in current set of terms or term adjoints
        // if it isn't found in either then append the vectors
        if (std::find(h_vec.begin(), h_vec.end(), h) == h_vec.end()){
            if (std::find(hd_vec.begin(), hd_vec.end(), h) == hd_vec.end()){
                SQOperator temp;
                // if term is same as term adjoint add both
                if(h == hd or h.first == h.second){
                    // (Nick) Need this checked out for sure
                    temp.add_term(hl/2.0, h.first, h.second);
                    temp.add_term(hl/2.0, hd.first, hd.second);
                } else {
                    temp.add_term(hl, h.first, h.second);
                    temp.add_term(hl, hd.first, hd.second);
                    temp.simplify();
                }

                terms_.push_back(std::make_pair(coeff, temp));

                h_vec.push_back(h);
                hd_vec.push_back(hd);

            }
        } 
    }
}

// The code below is a helper function to add_connection_pairs
namespace {
    struct TupleHash {
        std::size_t operator()(const std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>& t) const {
            uint64_t a = std::get<0>(t);
            uint64_t b = std::get<1>(t);
            uint64_t c = std::get<2>(t);
            uint64_t d = std::get<3>(t);

            std::size_t seed = 0;
            
            // Boost-style hash combination function
            auto combine = [](std::size_t& seed, uint64_t value) {
                seed ^= std::hash<uint64_t>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            };

            combine(seed, a);
            combine(seed, b);
            combine(seed, c);
            combine(seed, d);

            return seed;
        }
    };

    double clean_signature_value(double value) {
        return (std::abs(value) < 1.0e-14) ? 0.0 : value;
    }

    std::string signed_sq_operator_signature(const SQOperator& sq_op, double sign) {
        std::ostringstream oss;
        oss << std::setprecision(17);
        for (const auto& term : sq_op.terms()) {
            const auto& coeff = std::get<0>(term);
            oss << clean_signature_value(sign * coeff.real()) << ","
                << clean_signature_value(sign * coeff.imag()) << ":";
            for (const auto idx : std::get<1>(term)) {
                oss << idx << ",";
            }
            oss << "|";
            for (const auto idx : std::get<2>(term)) {
                oss << idx << ",";
            }
            oss << ";";
        }
        return oss.str();
    }

    std::string sq_operator_signature(SQOperator sq_op) {
        sq_op.simplify();

        // The legacy GSDx and k-UpCCGSDx constructors are retained as internal
        // compatibility helpers for general_ex_pool_order="particle_hole_first".
        // Some particle-hole helpers use the opposite overall sign from the
        // generalized construction, but the generator span and UCC
        // parameterization are the same up to theta -> -theta.  Canonicalize
        // duplicate detection modulo global sign so these ordered pools reorder
        // rather than enlarge the operator set.
        const std::string positive = signed_sq_operator_signature(sq_op, +1.0);
        const std::string negative = signed_sq_operator_signature(sq_op, -1.0);
        return std::min(positive, negative);
    }

    template <typename PoolType>
    void add_unique_operator(PoolType& pool, SQOperator sq_op, std::set<std::string>& seen) {
        sq_op.simplify();
        if (sq_op.terms().empty()) {
            return;
        }

        const std::string signature = sq_operator_signature(sq_op);
        if (seen.insert(signature).second) {
            pool.add_pool_operator(1.0, sq_op);
        }
    }

    template <typename PoolType>
    void append_unique_pool_terms(PoolType& target, const PoolType& source, std::set<std::string>& seen) {
        for (const auto& term : source.terms()) {
            const std::string signature = sq_operator_signature(term.second);
            if (seen.find(signature) == seen.end()) {
                if (target.add_pool_operator(term.first, term.second)) {
                    seen.insert(signature);
                }
            }
        }
    }

    template <typename PoolType>
    void fill_gsd_particle_hole_terms(PoolType& pool, const int nocc, const int nvir) {
        std::set<std::string> seen;

        for (size_t i = 0; i < static_cast<size_t>(nocc); i++) {
            const size_t ia = 2 * i;
            const size_t ib = 2 * i + 1;

            for (size_t a = 0; a < static_cast<size_t>(nvir); a++) {
                const size_t aa = 2 * static_cast<size_t>(nocc) + 2 * a;
                const size_t ab = 2 * static_cast<size_t>(nocc) + 2 * a + 1;

                SQOperator temp1a;
                temp1a.add_term(+1.0, {aa}, {ia});
                temp1a.add_term(-1.0, {ia}, {aa});
                add_unique_operator(pool, temp1a, seen);

                SQOperator temp1b;
                temp1b.add_term(+1.0, {ab}, {ib});
                temp1b.add_term(-1.0, {ib}, {ab});
                add_unique_operator(pool, temp1b, seen);
            }
        }

        for (size_t i = 0; i < static_cast<size_t>(nocc); i++) {
            const size_t ia = 2 * i;
            const size_t ib = 2 * i + 1;
            for (size_t j = i; j < static_cast<size_t>(nocc); j++) {
                const size_t ja = 2 * j;
                const size_t jb = 2 * j + 1;
                for (size_t a = 0; a < static_cast<size_t>(nvir); a++) {
                    const size_t aa = 2 * static_cast<size_t>(nocc) + 2 * a;
                    const size_t ab = 2 * static_cast<size_t>(nocc) + 2 * a + 1;
                    for (size_t b = a; b < static_cast<size_t>(nvir); b++) {
                        const size_t ba = 2 * static_cast<size_t>(nocc) + 2 * b;
                        const size_t bb = 2 * static_cast<size_t>(nocc) + 2 * b + 1;

                        if ((aa != ba) && (ia != ja)) {
                            SQOperator temp2aaaa;
                            temp2aaaa.add_term(+1.0, {aa, ba}, {ia, ja});
                            temp2aaaa.add_term(-1.0, {ja, ia}, {ba, aa});
                            add_unique_operator(pool, temp2aaaa, seen);
                        }

                        if ((ab != bb) && (ib != jb)) {
                            SQOperator temp2bbbb;
                            temp2bbbb.add_term(+1.0, {ab, bb}, {ib, jb});
                            temp2bbbb.add_term(-1.0, {jb, ib}, {bb, ab});
                            add_unique_operator(pool, temp2bbbb, seen);
                        }

                        if ((aa != bb) && (ia != jb)) {
                            SQOperator temp2abab;
                            temp2abab.add_term(+1.0, {aa, bb}, {ia, jb});
                            temp2abab.add_term(-1.0, {jb, ia}, {bb, aa});
                            add_unique_operator(pool, temp2abab, seen);
                        }

                        if ((ab != ba) && (ib != ja)) {
                            SQOperator temp2baba;
                            temp2baba.add_term(+1.0, {ab, ba}, {ib, ja});
                            temp2baba.add_term(-1.0, {ja, ib}, {ba, ab});
                            add_unique_operator(pool, temp2baba, seen);
                        }

                        if ((aa != bb) && (ib != ja)) {
                            SQOperator temp2abba;
                            temp2abba.add_term(+1.0, {aa, bb}, {ib, ja});
                            temp2abba.add_term(-1.0, {ja, ib}, {bb, aa});
                            add_unique_operator(pool, temp2abba, seen);
                        }

                        if ((ab != ba) && (ia != jb)) {
                            SQOperator temp2baab;
                            temp2baab.add_term(+1.0, {ab, ba}, {ia, jb});
                            temp2baab.add_term(-1.0, {jb, ia}, {ba, ab});
                            add_unique_operator(pool, temp2baab, seen);
                        }
                    }
                }
            }
        }
    }

    template <typename PoolType>
    void fill_kupccgsd_particle_hole_terms(PoolType& pool, const int nocc, const int nvir) {
        std::set<std::string> seen;

        for (size_t i = 0; i < static_cast<size_t>(nocc); i++) {
            const size_t ia = 2 * i;
            const size_t ib = 2 * i + 1;

            for (size_t a = 0; a < static_cast<size_t>(nvir); a++) {
                const size_t aa = 2 * static_cast<size_t>(nocc) + 2 * a;
                const size_t ab = 2 * static_cast<size_t>(nocc) + 2 * a + 1;

                SQOperator temp1a;
                temp1a.add_term(+1.0, {aa}, {ia});
                temp1a.add_term(-1.0, {ia}, {aa});
                add_unique_operator(pool, temp1a, seen);

                SQOperator temp1b;
                temp1b.add_term(+1.0, {ab}, {ib});
                temp1b.add_term(-1.0, {ib}, {ab});
                add_unique_operator(pool, temp1b, seen);
            }
        }

        for (size_t p = static_cast<size_t>(nocc);
             p < static_cast<size_t>(nocc + nvir);
             ++p) {
            const size_t pa = 2 * p;
            const size_t pb = 2 * p + 1;
            for (size_t q = 0; q < static_cast<size_t>(nocc); ++q) {
                const size_t qa = 2 * q;
                const size_t qb = 2 * q + 1;

                if ((pa != qa) && (pb != qb)) {
                    SQOperator temp2abab;
                    temp2abab.add_term(-1.0, {pa, pb}, {qa, qb});
                    temp2abab.add_term(+1.0, {qb, qa}, {pb, pa});
                    add_unique_operator(pool, temp2abab, seen);
                }
            }
        }
    }

    void combinations_recursive(
        const size_t start,
        const size_t n,
        const size_t k,
        std::vector<size_t>& current,
        std::vector<std::vector<size_t>>& result)
    {
        if (current.size() == k) {
            result.push_back(current);
            return;
        }

        const size_t remaining = k - current.size();
        for (size_t i = start; i + remaining <= n; ++i) {
            current.push_back(i);
            combinations_recursive(i + 1, n, k, current, result);
            current.pop_back();
        }
    }

    std::vector<std::vector<size_t>> combinations(const size_t n, const size_t k) {
        std::vector<std::vector<size_t>> result;
        if (k > n) {
            return result;
        }
        std::vector<size_t> current;
        current.reserve(k);
        combinations_recursive(0, n, k, current, result);
        return result;
    }

    template <typename PoolType>
    void add_particle_hole_rank_terms(PoolType& pool, const int nocc, const int nvir, const int rank) {
        // Direct particle-hole construction assumes the same RHF occupied-block
        // layout as the direct singles/doubles code:
        //   occupied spatial orbitals: 0, ..., nocc - 1
        //   virtual spatial orbitals:  nocc, ..., nocc + nvir - 1
        //   alpha spin orbital:        2 * p
        //   beta spin orbital:         2 * p + 1
        //
        // For rank r we choose r_alpha alpha p-h substitutions and r_beta beta
        // substitutions, with r_alpha + r_beta = r. This reproduces the C1
        // count sum_a C(nocc,a)C(nvir,a)C(nocc,r-a)C(nvir,r-a).
        for (int nalpha = 0; nalpha <= rank; ++nalpha) {
            const int nbeta = rank - nalpha;
            if (nalpha > nocc || nalpha > nvir || nbeta > nocc || nbeta > nvir) {
                continue;
            }

            const auto alpha_occ_combos = combinations(static_cast<size_t>(nocc), static_cast<size_t>(nalpha));
            const auto alpha_vir_combos = combinations(static_cast<size_t>(nvir), static_cast<size_t>(nalpha));
            const auto beta_occ_combos = combinations(static_cast<size_t>(nocc), static_cast<size_t>(nbeta));
            const auto beta_vir_combos = combinations(static_cast<size_t>(nvir), static_cast<size_t>(nbeta));

            for (const auto& alpha_occ : alpha_occ_combos) {
                for (const auto& alpha_vir : alpha_vir_combos) {
                    for (const auto& beta_occ : beta_occ_combos) {
                        for (const auto& beta_vir : beta_vir_combos) {
                            std::vector<size_t> creators;
                            std::vector<size_t> annihilators;
                            creators.reserve(static_cast<size_t>(rank));
                            annihilators.reserve(static_cast<size_t>(rank));

                            // Deterministic convention: alpha sector first,
                            // then beta sector, within each lexical combination.
                            for (const auto a : alpha_vir) {
                                creators.push_back(2 * (static_cast<size_t>(nocc) + a));
                            }
                            for (const auto a : beta_vir) {
                                creators.push_back(2 * (static_cast<size_t>(nocc) + a) + 1);
                            }
                            for (const auto i : alpha_occ) {
                                annihilators.push_back(2 * i);
                            }
                            for (const auto i : beta_occ) {
                                annihilators.push_back(2 * i + 1);
                            }

                            SQOperator op;
                            op.add_term(+1.0, creators, annihilators);

                            std::vector<size_t> reversed_creators(creators.rbegin(), creators.rend());
                            std::vector<size_t> reversed_annihilators(annihilators.rbegin(), annihilators.rend());
                            op.add_term(-1.0, reversed_annihilators, reversed_creators);
                            op.simplify();
                            pool.add_pool_operator(1.0, op);
                        }
                    }
                }
            }
        }
    }
}

void SQOpPool::add_connection_pairs(
      const FCIComputer& residual, 
      const FCIComputer& reference,
      const double threshold)
{
    // Check that the residual and reference states have the same dimensions.
    if (residual.get_state().shape() != reference.get_state().shape()) {
        throw std::invalid_argument("Dimension of residual must have the same shape as reference.");
    }

    size_t n_alpha_str = residual.get_state().shape()[0];
    size_t n_beta_str  = residual.get_state().shape()[1];
    size_t Nfci = n_alpha_str * n_beta_str;

    // std::cout << "n_alpha_str = " << n_alpha_str << ", n_beta_str = " << n_beta_str
    //           << ", total determinants = " << Nfci << std::endl;
    // std::cout << "Threshold = " << threshold << std::endl;

    // 1. Create a vector of tuples (r_mu^2, alpha_index, beta_index) for the residual,
    //    but skip the HF determinant (assumed to be at (0,0)).
    std::vector<std::tuple<double, int, int>> res_sqs;
    res_sqs.reserve(Nfci);
    for (size_t I = 0; I < n_alpha_str; ++I) {
        for (size_t J = 0; J < n_beta_str; ++J) {
            // Skip the HF determinant at (0,0)
            if (I == 0 && J == 0) 
                continue;

            std::complex<double> r_mu = residual.get_state().get({I, J});
            double r_mu_sq = std::norm(r_mu);
            // res_sqs.push_back(std::make_tuple(r_mu_sq, static_cast<int>(I), static_cast<int>(J)));
            res_sqs.push_back(std::make_tuple(r_mu_sq, I, J));
        }
    }

    // Sort in ascending order by weight (smallest r^2 first).
    std::sort(res_sqs.begin(), res_sqs.end(),
              [](auto a, auto b) { return std::get<0>(a) < std::get<0>(b); });

    // Print the bottom 5 residual entries for debugging.
    // std::cout << "\n\nTop 5 residual determinants (r^2, alpha idx, beta idx):" << std::endl;
    // for (size_t i = res_sqs.size()-1; i > std::max(size_t(5), res_sqs.size()) - 5; --i) {
    //     std::cout << "  (" << std::get<0>(res_sqs[i]) << ", " 
    //               << std::get<1>(res_sqs[i]) << ", " << std::get<2>(res_sqs[i]) << ")" 
    //               << std::endl;
    // }

    // Select residual determinants until the cumulative squared amplitude reaches the threshold.
    double cumulative = 0.0;
    size_t num_keep = 0;
    for (const auto& tup : res_sqs) {
        cumulative += std::get<0>(tup);
        // ++num_keep;
        if (cumulative >= threshold) ++num_keep;
    }
    // std::cout << "Cumulative r^2 threshold reached with " << num_keep 
    //           << " residual determinants (excluding HF)." << std::endl;

    std::vector<std::tuple<double, int, int>> selected_res(res_sqs.end() - num_keep, res_sqs.end());

    // 2. Build the reference indices, screening out those with coefficient magnitude < 1e-6.
    std::vector<std::tuple<int, int>> ref_indices;
    for (size_t I = 0; I < n_alpha_str; ++I) {
        for (size_t J = 0; J < n_beta_str; ++J) {
            std::complex<double> c_ref = reference.get_state().get({I, J});
            if (std::abs(c_ref) < 1e-6)
                continue;
            ref_indices.push_back(std::make_tuple(static_cast<int>(I), static_cast<int>(J)));
        }
    }

    // std::cout << "\nReference state contains " << ref_indices.size() 
    //           << " determinants with |coeff| >= 1e-6." << std::endl;

    // // Print the bottom 5 residual entries for debugging.
    // std::cout << "\nTop 5 (or less) ref determinants (alpha idx, beta idx):" << std::endl;
    // for (size_t i = 0; i < ref_indices.size(); ++i) {
    //     std::cout << "  (" 
    //               << std::get<0>(ref_indices[i]) << ", " << std::get<1>(ref_indices[i]) << ")" 
    //               << std::endl;
    // }

    // 3. Build unique excitation operators as Hermitian combinations.
    // Each operator is represented as a tuple: (cre_alpha, cre_beta, ann_alpha, ann_beta),
    // where the masks are over spatial orbitals.
    std::unordered_set<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>, TupleHash> op_set;

    for (const auto& res_tup : selected_res) {
    // for (auto it = selected_res.rbegin(); it != selected_res.rend(); ++it) {
    //     const auto& res_tup = *it;


        int res_alpha_idx = std::get<1>(res_tup);
        int res_beta_idx  = std::get<2>(res_tup);
        uint64_t res_astr = residual.get_graph().get_astr_at_idx(res_alpha_idx);
        uint64_t res_bstr = residual.get_graph().get_bstr_at_idx(res_beta_idx);

        // Debug: Print the residual bit strings (in hexadecimal).
        // std::cout << "Residual det (" << res_alpha_idx << ", " << res_beta_idx << ") - "
        //           << "alpha: 0x" << std::hex << res_astr << std::dec 
        //           << ", beta: 0x" << std::hex << res_bstr << std::dec << std::endl;

        for (const auto& ref_tup : ref_indices) {
            int ref_alpha_idx = std::get<0>(ref_tup);
            int ref_beta_idx  = std::get<1>(ref_tup);
            uint64_t ref_astr = reference.get_graph().get_astr_at_idx(ref_alpha_idx);
            uint64_t ref_bstr = reference.get_graph().get_bstr_at_idx(ref_beta_idx);

            uint64_t cre_mask_alpha = 0;
            uint64_t ann_mask_alpha = 0;
            uint64_t cre_mask_beta  = 0;
            uint64_t ann_mask_beta  = 0;

            // Loop over spatial orbitals (assume up to 64).
            for (int i = 0; i < 64; ++i) {
                bool ref_bit = (ref_astr >> i) & 1ULL;
                bool res_bit = (res_astr >> i) & 1ULL;
                if (!ref_bit && res_bit) {
                    cre_mask_alpha |= (1ULL << i);
                }
                if (ref_bit && !res_bit) {
                    ann_mask_alpha |= (1ULL << i);
                }
            }
            for (int i = 0; i < 64; ++i) {
                bool ref_bit = (ref_bstr >> i) & 1ULL;
                bool res_bit = (res_bstr >> i) & 1ULL;
                if (!ref_bit && res_bit) {
                    cre_mask_beta |= (1ULL << i);
                }
                if (ref_bit && !res_bit) {
                    ann_mask_beta |= (1ULL << i);
                }
            }

            // Skip operators with repeated indices.
            if ((cre_mask_alpha & ann_mask_alpha) != 0ULL || (cre_mask_beta & ann_mask_beta) != 0ULL)
                continue;

            auto op_term = std::make_tuple(cre_mask_alpha, cre_mask_beta, ann_mask_alpha, ann_mask_beta);
            op_set.insert(op_term);
        }
    }

    // std::cout << "Number of unique excitation operators (pre-spin mapping): " 
    //           << op_set.size() << std::endl;

    // 4. Convert the spatial masks to spin-orbital index lists and form the Hermitian operator T - T†.
    for (const auto& op_term : op_set) {
    // for (auto it = op_set.rbegin(); it != op_set.rend(); ++it) {
        // const auto& op_term = *it;
        uint64_t cre_mask_alpha = std::get<0>(op_term);
        uint64_t cre_mask_beta  = std::get<1>(op_term);
        uint64_t ann_mask_alpha = std::get<2>(op_term);
        uint64_t ann_mask_beta  = std::get<3>(op_term);

        std::vector<std::size_t> cre_idxs;
        std::vector<std::size_t> ann_idxs;

        // Map alpha: spatial orbital i -> spin orbital 2*i (even indices).
        for (int i = 0; i < 64; ++i) {
            if ((cre_mask_alpha >> i) & 1ULL)
                cre_idxs.push_back(2 * i);
            if ((ann_mask_alpha >> i) & 1ULL)
                ann_idxs.push_back(2 * i);
        }
        // Map beta: spatial orbital i -> spin orbital 2*i+1 (odd indices).
        for (int i = 0; i < 64; ++i) {
            if ((cre_mask_beta >> i) & 1ULL)
                cre_idxs.push_back(2 * i + 1);
            if ((ann_mask_beta >> i) & 1ULL)
                ann_idxs.push_back(2 * i + 1);
        }

        // Debug: Print the mapped spin-orbital indices.
        // std::cout << "Excitation operator:" << std::endl;
        // std::cout << "  Creation indices: ";
        // for (auto idx : cre_idxs) std::cout << idx << " ";
        // std::cout << std::endl;
        // std::cout << "  Annihilation indices: ";
        // for (auto idx : ann_idxs) std::cout << idx << " ";
        // std::cout << std::endl;

        // Create the excitation operator T.
        SQOperator T;
        T.add_term(1.0, cre_idxs, ann_idxs);

        // Form the Hermitian combination: T - T†.
        // To obtain T†, reverse the order of the indices.
        std::vector<std::size_t> cre_idxs_rev = cre_idxs;
        std::vector<std::size_t> ann_idxs_rev = ann_idxs;
        std::reverse(cre_idxs_rev.begin(), cre_idxs_rev.end());
        std::reverse(ann_idxs_rev.begin(), ann_idxs_rev.end());
        T.add_term(-1.0, ann_idxs_rev, cre_idxs_rev);
        T.simplify();
        terms_.push_back(std::make_pair(1.0, T));
    }
    
    // std::cout << "Total number of terms added to the pool: " << terms_.size() << std::endl;
}




// from victor
// void SQOpPool::add_connection_pairs(
//       const FCIComputer& residual, 
//       const FCIComputer& reference,
//       const double threshold)
// {
//     // 1. Do the sort of the residual vector and 'keep' residual determinants above threshold

//     // Need to ensure residual and reference have the same shape
//     if (residual.get_state().shape() != reference.get_state().shape()) {  // Condition to throw an error
//         throw std::invalid_argument( "Dimension of residual must have the same shape as reference." );
//     }

//     size_t n_alfa_str = residual.get_state().shape()[0];
//     size_t n_beta_str = residual.get_state().shape()[1];
//     size_t Nfci = n_alfa_str * n_beta_str;

//     // Need a temporary container to store r_mu, I_mu, J_mu that is std::sort(able), size of Nfci
//     std::vector<std::tuple<double, int, int>> res_sqs(Nfci);

//     for (size_t I_mu=0; I_mu < n_alfa_str; ++I_mu) {
//         for (size_t J_mu=0; J_mu < n_beta_str; ++J_mu) {
//             std::complex<double> r_mu = residual.get_state().get({I_mu, J_mu});
//             double r_mu_sq = std::real(r_mu * std::conj(r_mu));

//             size_t IJ_mu = n_beta_str * I_mu + J_mu;

//             res_sqs[IJ_mu] = std::make_tuple(r_mu_sq, I_mu, J_mu);
//         }
//     } 

//     // Sorting the vector
//     std::sort(res_sqs.begin(), res_sqs.end());

//     size_t n_start = 0;
//     double sum = 0.0;

//     for(size_t IJ_mu = 0; IJ_mu < Nfci; ++IJ_mu){
//         sum += std::get<0>(res_sqs[IJ_mu]);
//         ++n_start;

//         if(sum > threshold){
//             break;
//         }

//     }

//     // 2. Initialize (hash?) map of bitstrings, masks will represent alph and beta transitions
//     std::unordered_set<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>, TupleHash> str_set;    

//     // Loop over residual strings
//     for(size_t IJ_mu = n_start; IJ_mu < Nfci; ++IJ_mu){

//         int I_mu = std::get<1>(res_sqs[IJ_mu]);
//         int J_mu = std::get<2>(res_sqs[IJ_mu]);
        
//         uint64_t res_astr = residual.get_graph().get_astr_at_idx(I_mu);
//         uint64_t res_bstr = residual.get_graph().get_bstr_at_idx(J_mu);

//         // Loop over reference strings...
//         for(size_t IJ_mu = n_start; IJ_mu < Nfci; ++IJ_mu){

//             int I_mu = std::get<1>(res_sqs[IJ_mu]);
//             int J_mu = std::get<2>(res_sqs[IJ_mu]);

//             uint64_t ref_astr = reference.get_graph().get_astr_at_idx(I_mu);
//             uint64_t ref_bstr = reference.get_graph().get_bstr_at_idx(J_mu);

//             uint64_t ann_mask_alfa = 0;
//             uint64_t ann_mask_beta = 0;
//             uint64_t cre_mask_alfa = 0;
//             uint64_t cre_mask_beta = 0;

//             // alfa
//             for (int i = 0; i < 64; ++i) {
//                 bool ref = (ref_astr >> i) & 1; 
//                 bool res = (res_astr >> i) & 1;

//                 if (ref == 0 and res == 1) {
//                     cre_mask_alfa ^= (1ULL << i);
//                 }

//                 if (ref == 1 and res == 0) {
//                     ann_mask_alfa ^= (1ULL << i);
//                 }
//             }

//             // beta
//             for (int i = 0; i < 64; ++i) {
//                 bool ref = (ref_bstr >> i) & 1; 
//                 bool res = (res_bstr >> i) & 1;

//                 if (ref == 0 and res == 1) {
//                     cre_mask_beta ^= (1ULL << i);
//                 }

//                 if (ref == 1 and res == 0) {
//                     ann_mask_beta ^= (1ULL << i);
//                 }
//             }

//             std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> op_str = std::make_tuple(cre_mask_alfa, 
//                                                                                         cre_mask_beta, 
//                                                                                         ann_mask_alfa, 
//                                                                                         ann_mask_beta);

//             // add new bitstring pair to our hash map
//             if (str_set.find(op_str) == str_set.end()) {
//                 str_set.insert(op_str);
//             }
//         }
//     }

//     // // 3. add_term(s) based on bitmaks 
//     // SQOperator temp1a;
//     // temp1a.add_term(+1.0, {aa}, {ia});
//     for (const auto& t : str_set) {

//         uint64_t crea = std::get<0>(t);
//         uint64_t creb = std::get<1>(t);
//         uint64_t anna = std::get<2>(t);
//         uint64_t annb = std::get<3>(t);

//         std::vector<std::size_t> ann_idxs;
//         std::vector<std::size_t> cre_idxs;

//         for (int i = 0; i < 32; ++i) {

//             if (bool (crea >> i) & 1) {
//                 cre_idxs.push_back(2*i);
//             }

//             if (bool (creb >> i) & 1) {
//                 cre_idxs.push_back(2*i + 1);
//             }

//             if (bool (anna >> i) & 1) {
//                 ann_idxs.push_back(2*i);
//             }

//             if (bool (annb >> i) & 1) {
//                 ann_idxs.push_back(2*i + 1);
//             }
//         }

//         SQOperator temp;
//         temp.add_term(1.0, cre_idxs, ann_idxs);
//         terms_.push_back(std::make_pair(1.0, temp));
        
//     }
// }

            // // calculate bitmask
            // uint64_t amask = ref_astr ^ res_astr;  // Compute the masks (bits that differ)
            // uint64_t bmask = ref_bstr ^ res_bstr;

            

            // for (size_t i = 0; i < bit_length; ++i) {
            //     uint64_t apos = 1ULL << i;  // Single-bit mask for position i
            //     uint64_t bpos = 1ULL << i;

            //     if (amask & apos) {  // Check if mask has a 1 at this position
            //         if ((res_astr & apos) == 0) {
            //             anna.push_back(apos);  // Add to "anna" (annihilation)
            //         } else {
            //             crea.push_back(apos);  // Add to "crea" (creation)
            //         }
            //     }

            //     if (bmask & bpos) {  // Check if mask has a 1 at this position
            //         if ((res_astr & bpos) == 0) {
            //             anna.push_back(bpos);  // Add to "anna" (annihilation)
            //         } else {
            //             crea.push_back(bpos);  // Add to "crea" (creation)
            //         }
            //     }
            // }
            // ASTR and BSTR from the same index need to be combined into the same SQOP, so need to make unordered map of masks

            // std::tuple<uint64_t, uint64_t> ab_str = std::make_tuple(new_astr, new_bstr);
            
            // // add new bitstring pair to our hash map
            // if (str_set.find(ab_str) == str_set.end()) {
            //     str_set.insert(ab_str);

            // }


void SQOpPool::set_coeffs(const std::vector<std::complex<double>>& new_coeffs){
    if(new_coeffs.size() != terms_.size()){
        throw std::invalid_argument( "Number of new coefficients for quantum operator must equal." );
    }
    for (size_t l = 0; l < new_coeffs.size(); l++){
        terms_[l].first = new_coeffs[l];
    }
}

void SQOpPool::set_coeffs_to_scaler(std::complex<double> new_coeff){
    for (size_t l = 0; l < terms_.size(); l++){
        terms_[l].first = new_coeff;
    }
}

const std::vector<std::pair< std::complex<double>, SQOperator>>& SQOpPool::terms() const{
    return terms_;
}

void SQOpPool::set_orb_spaces(const std::vector<int>& ref){
    int norb = ref.size();
    if (norb%2 == 0){
        norb = static_cast<int>(norb/2);
    } else {
        throw std::invalid_argument("QForte does not yet support systems with an odd number of spin orbitals.");
    }

    nocc_ = 0;
    for (const auto& occupancy : ref){
        nocc_ += occupancy;
    }

    if (nocc_%2 == 0){
        nocc_ = static_cast<int>(nocc_/2);
    } else {
        throw std::invalid_argument("QForte does not yet support systems with an odd number of occupied spin orbitals.");
    }

    nvir_ = static_cast<int>(norb - nocc_);

    // C1 is the default: every spatial orbital has irrep 0.  Non-C1
    // algorithms may override this with set_orb_irreps before fill_pool().
    if (orb_irreps_to_int_.empty() ||
        static_cast<int>(orb_irreps_to_int_.size()) != nocc_ + nvir_) {
        orb_irreps_to_int_ = std::vector<int>(nocc_ + nvir_, 0);
        target_irrep_ = 0;
    }
}

QubitOpPool SQOpPool::get_qubit_op_pool(){
    QubitOpPool A;
    for (auto& term : terms_) {
        // QubitOperator a = term.second.jw_transform();
        // a.mult_coeffs(term.first);
        A.add_term(term.first, term.second.jw_transform());
    }
    return A;
}


QubitOperator SQOpPool::get_qubit_operator(const std::string& order_type, bool combine_like_terms, bool qubit_excitations){
    QubitOperator parent;

    if(order_type=="unique_lex"){
        for (auto& term : terms_) {
            auto child = term.second.jw_transform(qubit_excitations);
            child.mult_coeffs(term.first);
            parent.add_op(child);
        }
        // TODO: analyze ordering here, eliminating simplify will place commuting
        // terms closer together but may introduce redundancy.
        parent.simplify();
        parent.order_terms();
    } else if (order_type=="commuting_grp_lex") {
        for (auto& term : terms_) {
            auto child = term.second.jw_transform(qubit_excitations);
            child.mult_coeffs(term.first);
            child.simplify(combine_like_terms=combine_like_terms);
            child.order_terms();
            parent.add_op(child);

        }
    } else {
        throw std::invalid_argument( "Invalid order_type specified.");
    }
    return parent;
}

void SQOpPool::fill_pool(std::string pool_type){
    if(pool_type=="GSD"){
        size_t norb = nocc_ + nvir_;
        for(size_t i=0; i<norb; i++){
            size_t ia = 2*i;
            size_t ib = 2*i+1;
            for (size_t a=i; a<norb; a++){
                size_t aa = 2*a;
                size_t ab = 2*a+1;

                if( aa != ia ){
                    SQOperator temp1a;
                    temp1a.add_term(+1.0, {aa}, {ia});
                    temp1a.add_term(-1.0, {ia}, {aa});
                    temp1a.simplify();
                    if(temp1a.terms().size() > 0){
                        add_pool_operator(1.0, temp1a);
                    }
                }

                if( ab != ib ){
                    SQOperator temp1b;
                    temp1b.add_term(+1.0, {ab}, {ib});
                    temp1b.add_term(-1.0, {ib}, {ab});
                    temp1b.simplify();
                    if(temp1b.terms().size() > 0){
                        add_pool_operator(1.0, temp1b);
                    }
                }
            }
        }

        std::vector< std::vector<size_t> > uniqe_2bdy;
        std::vector< std::vector<size_t> > adjnt_2bdy;

        for(size_t i=0; i<norb; i++){
            size_t ia = 2*i;
            size_t ib = 2*i+1;
            for(size_t j=i; j<norb; j++){
                size_t ja = 2*j;
                size_t jb = 2*j+1;
                for(size_t a=0; a<norb; a++){
                    size_t aa = 2*a;
                    size_t ab = 2*a+1;
                    for(size_t b=a; b<norb; b++){
                        size_t ba = 2*b;
                        size_t bb = 2*b+1;

                        if((aa != ba) && (ia != ja)){
                            SQOperator temp2aaaa;
                            temp2aaaa.add_term(+1.0, {aa,ba}, {ia,ja});
                            temp2aaaa.add_term(-1.0, {ja,ia}, {ba,aa});
                            temp2aaaa.simplify();
                            if(temp2aaaa.terms().size() > 0){
                                std::vector<size_t> vtemp {std::get<1>(temp2aaaa.terms()[0])[0], std::get<1>(temp2aaaa.terms()[0])[1], std::get<2>(temp2aaaa.terms()[0])[0], std::get<2>(temp2aaaa.terms()[0])[1]};
                                std::vector<size_t> vadjt {std::get<1>(temp2aaaa.terms()[1])[0], std::get<1>(temp2aaaa.terms()[1])[1], std::get<2>(temp2aaaa.terms()[1])[0], std::get<2>(temp2aaaa.terms()[1])[1]};
                                if( (std::find(uniqe_2bdy.begin(), uniqe_2bdy.end(), vtemp) == uniqe_2bdy.end()) ){
                                    if( (std::find(adjnt_2bdy.begin(), adjnt_2bdy.end(), vtemp) == adjnt_2bdy.end()) ){
                                        uniqe_2bdy.push_back(vtemp);
                                        adjnt_2bdy.push_back(vadjt);
                                        add_pool_operator(1.0, temp2aaaa);
                                    }
                                }
                            }
                        }

                        if((ab != bb ) && (ib != jb)){
                            SQOperator temp2bbbb;
                            temp2bbbb.add_term(+1.0, {ab,bb}, {ib,jb});
                            temp2bbbb.add_term(-1.0, {jb,ib}, {bb,ab});
                            temp2bbbb.simplify();
                            if(temp2bbbb.terms().size() > 0){
                                std::vector<size_t> vtemp {std::get<1>(temp2bbbb.terms()[0])[0], std::get<1>(temp2bbbb.terms()[0])[1], std::get<2>(temp2bbbb.terms()[0])[0], std::get<2>(temp2bbbb.terms()[0])[1]};
                                std::vector<size_t> vadjt {std::get<1>(temp2bbbb.terms()[1])[0], std::get<1>(temp2bbbb.terms()[1])[1], std::get<2>(temp2bbbb.terms()[1])[0], std::get<2>(temp2bbbb.terms()[1])[1]};
                                if( (std::find(uniqe_2bdy.begin(), uniqe_2bdy.end(), vtemp) == uniqe_2bdy.end()) ){
                                    if( (std::find(adjnt_2bdy.begin(), adjnt_2bdy.end(), vtemp) == adjnt_2bdy.end()) ){
                                        uniqe_2bdy.push_back(vtemp);
                                        adjnt_2bdy.push_back(vadjt);
                                        add_pool_operator(1.0, temp2bbbb);
                                    }
                                }
                            }
                        }

                        if((aa != bb) && (ia != jb)){
                            SQOperator temp2abab;
                            temp2abab.add_term(+1.0, {aa,bb}, {ia,jb});
                            temp2abab.add_term(-1.0, {jb,ia}, {bb,aa});
                            temp2abab.simplify();
                            if(temp2abab.terms().size() > 0){
                                std::vector<size_t> vtemp {std::get<1>(temp2abab.terms()[0])[0], std::get<1>(temp2abab.terms()[0])[1], std::get<2>(temp2abab.terms()[0])[0], std::get<2>(temp2abab.terms()[0])[1]};
                                std::vector<size_t> vadjt {std::get<1>(temp2abab.terms()[1])[0], std::get<1>(temp2abab.terms()[1])[1], std::get<2>(temp2abab.terms()[1])[0], std::get<2>(temp2abab.terms()[1])[1]};
                                if( (std::find(uniqe_2bdy.begin(), uniqe_2bdy.end(), vtemp) == uniqe_2bdy.end()) ){
                                    if( (std::find(adjnt_2bdy.begin(), adjnt_2bdy.end(), vtemp) == adjnt_2bdy.end()) ){
                                        uniqe_2bdy.push_back(vtemp);
                                        adjnt_2bdy.push_back(vadjt);
                                        add_pool_operator(1.0, temp2abab);
                                    }
                                }
                            }
                        }

                        if((ab != ba) && (ib != ja)){
                            SQOperator temp2baba;
                            temp2baba.add_term(+1.0, {ab,ba}, {ib,ja});
                            temp2baba.add_term(-1.0, {ja,ib}, {ba,ab});
                            temp2baba.simplify();
                            if(temp2baba.terms().size() > 0){
                                std::vector<size_t> vtemp {std::get<1>(temp2baba.terms()[0])[0], std::get<1>(temp2baba.terms()[0])[1], std::get<2>(temp2baba.terms()[0])[0], std::get<2>(temp2baba.terms()[0])[1]};
                                std::vector<size_t> vadjt {std::get<1>(temp2baba.terms()[1])[0], std::get<1>(temp2baba.terms()[1])[1], std::get<2>(temp2baba.terms()[1])[0], std::get<2>(temp2baba.terms()[1])[1]};
                                if( (std::find(uniqe_2bdy.begin(), uniqe_2bdy.end(), vtemp) == uniqe_2bdy.end()) ){
                                    if( (std::find(adjnt_2bdy.begin(), adjnt_2bdy.end(), vtemp) == adjnt_2bdy.end()) ){
                                        uniqe_2bdy.push_back(vtemp);
                                        adjnt_2bdy.push_back(vadjt);
                                        add_pool_operator(1.0, temp2baba);
                                    }
                                }
                            }
                        }

                        if((aa != bb) && (ib != ja)){
                            SQOperator temp2abba;
                            temp2abba.add_term(+1.0, {aa,bb}, {ib,ja});
                            temp2abba.add_term(-1.0, {ja,ib}, {bb,aa});
                            temp2abba.simplify();
                            if(temp2abba.terms().size() > 0){
                                std::vector<size_t> vtemp {std::get<1>(temp2abba.terms()[0])[0], std::get<1>(temp2abba.terms()[0])[1], std::get<2>(temp2abba.terms()[0])[0], std::get<2>(temp2abba.terms()[0])[1]};
                                std::vector<size_t> vadjt {std::get<1>(temp2abba.terms()[1])[0], std::get<1>(temp2abba.terms()[1])[1], std::get<2>(temp2abba.terms()[1])[0], std::get<2>(temp2abba.terms()[1])[1]};
                                if( (std::find(uniqe_2bdy.begin(), uniqe_2bdy.end(), vtemp) == uniqe_2bdy.end()) ){
                                    if( (std::find(adjnt_2bdy.begin(), adjnt_2bdy.end(), vtemp) == adjnt_2bdy.end()) ){
                                        uniqe_2bdy.push_back(vtemp);
                                        adjnt_2bdy.push_back(vadjt);
                                        add_pool_operator(1.0, temp2abba);
                                    }
                                }

                            }
                        }

                        if((ab != ba) && (ia != jb)){
                            SQOperator temp2baab;
                            temp2baab.add_term(+1.0, {ab,ba}, {ia,jb});
                            temp2baab.add_term(-1.0, {jb,ia}, {ba,ab});
                            temp2baab.simplify();
                            if(temp2baab.terms().size() > 0){
                                std::vector<size_t> vtemp {std::get<1>(temp2baab.terms()[0])[0], std::get<1>(temp2baab.terms()[0])[1], std::get<2>(temp2baab.terms()[0])[0], std::get<2>(temp2baab.terms()[0])[1]};
                                std::vector<size_t> vadjt {std::get<1>(temp2baab.terms()[1])[0], std::get<1>(temp2baab.terms()[1])[1], std::get<2>(temp2baab.terms()[1])[0], std::get<2>(temp2baab.terms()[1])[1]};
                                if( (std::find(uniqe_2bdy.begin(), uniqe_2bdy.end(), vtemp) == uniqe_2bdy.end()) ){
                                    if( (std::find(adjnt_2bdy.begin(), adjnt_2bdy.end(), vtemp) == adjnt_2bdy.end()) ){
                                        uniqe_2bdy.push_back(vtemp);
                                        adjnt_2bdy.push_back(vadjt);
                                        add_pool_operator(1.0, temp2baab);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if(pool_type=="GSDx"){
        SQOpPool ph_pool;
        ph_pool.nocc_ = nocc_;
        ph_pool.nvir_ = nvir_;
        ph_pool.orb_irreps_to_int_ = orb_irreps_to_int_;
        ph_pool.target_irrep_ = target_irrep_;
        fill_gsd_particle_hole_terms(ph_pool, nocc_, nvir_);

        SQOpPool gsd_pool;
        gsd_pool.nocc_ = nocc_;
        gsd_pool.nvir_ = nvir_;
        gsd_pool.orb_irreps_to_int_ = orb_irreps_to_int_;
        gsd_pool.target_irrep_ = target_irrep_;
        gsd_pool.fill_pool("GSD");

        std::set<std::string> seen;
        append_unique_pool_terms(*this, ph_pool, seen);
        append_unique_pool_terms(*this, gsd_pool, seen);
    } else if ( (pool_type=="S") || (pool_type=="SD") || (pool_type=="SDT") || (pool_type=="SDTQ") || (pool_type=="SDTQP") || (pool_type=="SDTQPH") || (pool_type=="All") ) {

        int max_nbody = 0;

        if(pool_type=="S") {
            max_nbody = 1;
        } else if(pool_type=="SD") {
            max_nbody = 2;
        }else if(pool_type=="SDT") {
            max_nbody = 3;
        } else if(pool_type=="SDTQ") {
            max_nbody = 4;
        } else if(pool_type=="SDTQP") {
            max_nbody = 5;
        } else if(pool_type=="SDTQPH") {
            max_nbody = 6;
        } else if(pool_type=="All") {
            max_nbody = 2 * nocc_;
        } else {
            throw std::invalid_argument( "Qforte UCC only supports up to Hextuple excitations." );
        }

        // Optimized implementation: directly enumerate valid excitations
        // instead of iterating over all 2^nqb bitstrings
        
        // ============================================
        // Singles (1-body excitations): i -> a
        // ============================================
        if (max_nbody >= 1) {
            for (size_t i = 0; i < static_cast<size_t>(nocc_); i++) {
                size_t ia = 2 * i;      // alpha occupied spin-orbital
                size_t ib = 2 * i + 1;  // beta occupied spin-orbital
                
                for (size_t a = 0; a < static_cast<size_t>(nvir_); a++) {
                    size_t aa = 2 * nocc_ + 2 * a;      // alpha virtual spin-orbital
                    size_t ab = 2 * nocc_ + 2 * a + 1;  // beta virtual spin-orbital
                    
                    // Alpha single: i_α -> a_α (parity: +1 * +1 = +1)
                    SQOperator t_a;
                    t_a.add_term(+1.0, {aa}, {ia});
                    t_a.add_term(-1.0, {ia}, {aa});
                    t_a.simplify();
                    if (t_a.terms().size() > 0) {
                        add_pool_operator(1.0, t_a);
                    }
                    
                    // Beta single: i_β -> a_β (parity: -1 * -1 = +1)
                    SQOperator t_b;
                    t_b.add_term(+1.0, {ab}, {ib});
                    t_b.add_term(-1.0, {ib}, {ab});
                    t_b.simplify();
                    if (t_b.terms().size() > 0) {
                        add_pool_operator(1.0, t_b);
                    }
                }
            }
        }
        
        // ============================================
        // Doubles (2-body excitations): i,j -> a,b
        // ============================================
        if (max_nbody >= 2) {
            for (size_t i = 0; i < static_cast<size_t>(nocc_); i++) {
                size_t ia = 2 * i;
                size_t ib = 2 * i + 1;
                
                for (size_t j = i; j < static_cast<size_t>(nocc_); j++) {
                    size_t ja = 2 * j;
                    size_t jb = 2 * j + 1;
                    
                    for (size_t a = 0; a < static_cast<size_t>(nvir_); a++) {
                        size_t aa = 2 * nocc_ + 2 * a;
                        size_t ab = 2 * nocc_ + 2 * a + 1;
                        
                        for (size_t b = a; b < static_cast<size_t>(nvir_); b++) {
                            size_t ba = 2 * nocc_ + 2 * b;
                            size_t bb = 2 * nocc_ + 2 * b + 1;
                            
                            // αα -> αα: (i_α, j_α) -> (a_α, b_α)
                            // parity: (+1)(+1)(+1)(+1) = +1
                            if (i != j && a != b) {
                                SQOperator t_aaaa;
                                t_aaaa.add_term(+1.0, {aa, ba}, {ia, ja});
                                t_aaaa.add_term(-1.0, {ja, ia}, {ba, aa});
                                t_aaaa.simplify();
                                if (t_aaaa.terms().size() > 0) {
                                    add_pool_operator(1.0, t_aaaa);
                                }
                            }
                            
                            // ββ -> ββ: (i_β, j_β) -> (a_β, b_β)
                            // parity: (-1)(-1)(-1)(-1) = +1
                            if (i != j && a != b) {
                                SQOperator t_bbbb;
                                t_bbbb.add_term(+1.0, {ab, bb}, {ib, jb});
                                t_bbbb.add_term(-1.0, {jb, ib}, {bb, ab});
                                t_bbbb.simplify();
                                if (t_bbbb.terms().size() > 0) {
                                    add_pool_operator(1.0, t_bbbb);
                                }
                            }
                            
                            // αβ -> αβ: (i_α, j_β) -> (a_α, b_β)
                            // parity: (+1)(-1)(+1)(-1) = +1
                            {
                                SQOperator t_abab;
                                t_abab.add_term(+1.0, {aa, bb}, {ia, jb});
                                t_abab.add_term(-1.0, {jb, ia}, {bb, aa});
                                t_abab.simplify();
                                if (t_abab.terms().size() > 0) {
                                    add_pool_operator(1.0, t_abab);
                                }
                            }
                            
                            // βα -> βα: (i_β, j_α) -> (a_β, b_α)
                            // parity: (-1)(+1)(-1)(+1) = +1
                            // Skip if same spatial orbitals to avoid duplicates with αβ->αβ
                            if (i != j || a != b) {
                                SQOperator t_baba;
                                t_baba.add_term(+1.0, {ab, ba}, {ib, ja});
                                t_baba.add_term(-1.0, {ja, ib}, {ba, ab});
                                t_baba.simplify();
                                if (t_baba.terms().size() > 0) {
                                    add_pool_operator(1.0, t_baba);
                                }
                            }

                            // The following two cross mixed-spin cases are distinct
                            // particle-hole doubles when both occupied and virtual
                            // spatial indices differ.  They complete the six spin
                            // patterns for closed-shell doubles:
                            // aaaa, bbbb, abab, baba, abba, baab.
                            if (i != j && a != b) {
                                SQOperator t_abba;
                                t_abba.add_term(+1.0, {aa, bb}, {ib, ja});
                                t_abba.add_term(-1.0, {ja, ib}, {bb, aa});
                                t_abba.simplify();
                                if (t_abba.terms().size() > 0) {
                                    add_pool_operator(1.0, t_abba);
                                }
                            }

                            if (i != j && a != b) {
                                SQOperator t_baab;
                                t_baab.add_term(+1.0, {ab, ba}, {ia, jb});
                                t_baab.add_term(-1.0, {jb, ia}, {ba, ab});
                                t_baab.simplify();
                                if (t_baab.terms().size() > 0) {
                                    add_pool_operator(1.0, t_baab);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // ============================================
        // Triples/quadruples use a generic direct particle-hole builder.
        // Quintuple and higher ranks still use the determinant fallback below.
        // ============================================
        if (max_nbody >= 3) {
            add_particle_hole_rank_terms(*this, nocc_, nvir_, 3);
        }

        if (max_nbody >= 4) {
            add_particle_hole_rank_terms(*this, nocc_, nvir_, 4);
        }

        // ============================================
        // Pentuples and higher (5+ body excitations)
        // ============================================
        if (max_nbody >= 5) {
            int nqb = 2 * (nocc_ + nvir_);
            int nel = 2 * nocc_;
            int na_el = nocc_;
            int nb_el = nocc_;

            for (int I = 0; I < std::pow(2, nqb); I++) {
                QubitBasis basis_I(I);

                if (basis_I.get_num_ones() != na_el + nb_el) {
                    continue;
                }

                int nbody = 0;
                int pn = 0;
                int na_I = 0;
                int nb_I = 0;
                std::vector<size_t> holes;
                std::vector<size_t> particles;
                std::vector<int> parity;

                for (size_t p = 0; p < 2 * static_cast<size_t>(nocc_); p++) {
                    int bit_val = static_cast<int>(basis_I.get_bit(p));
                    nbody += (1 - bit_val);
                    pn += bit_val;
                    if (p % 2 == 0) {
                        na_I += bit_val;
                    } else {
                        nb_I += bit_val;
                    }

                    if (bit_val - 1) {
                        holes.push_back(p);
                        if (p % 2 == 0) {
                            parity.push_back(1);
                        } else {
                            parity.push_back(-1);
                        }
                    }
                }
                for (size_t q = 2 * static_cast<size_t>(nocc_); q < static_cast<size_t>(nqb); q++) {
                    int bit_val = static_cast<int>(basis_I.get_bit(q));
                    pn += bit_val;
                    if (q % 2 == 0) {
                        na_I += bit_val;
                    } else {
                        nb_I += bit_val;
                    }
                    if (bit_val) {
                        particles.push_back(q);
                        if (q % 2 == 0) {
                            parity.push_back(1);
                        } else {
                            parity.push_back(-1);
                        }
                    }
                }

                if (pn == nel && na_I == na_el && nb_I == nb_el) {
                    // Only process ranks 5 and higher here; ranks 1-4 were
                    // already handled by direct occupied/virtual construction.
                    if (nbody >= 5 && nbody <= max_nbody) {
                        int total_parity = 1;
                        for (const auto& z : parity) {
                            total_parity *= z;
                        }

                        if (total_parity == 1) {
                            SQOperator t_temp;
                            t_temp.add_term(+1.0, particles, holes);
                            std::vector<size_t> rparticles(particles.rbegin(), particles.rend());
                            std::vector<size_t> rholes(holes.rbegin(), holes.rend());
                            t_temp.add_term(-1.0, rholes, rparticles);
                            t_temp.simplify();
                            add_pool_operator(1.0, t_temp);
                        }
                    }
                }
            }
        }
    } else if(pool_type=="sa_SD"){
        for(size_t i=0; i<nocc_; i++){
            size_t ia = 2*i;
            size_t ib = 2*i+1;

            for (size_t a=0; a<nvir_; a++){
                size_t aa = 2*nocc_ + 2*a;
                size_t ab = 2*nocc_ + 2*a+1;

                SQOperator temp1;
                temp1.add_term(+1.0/std::sqrt(2), {aa}, {ia});
                temp1.add_term(+1.0/std::sqrt(2), {ab}, {ib});

                temp1.add_term(-1.0/std::sqrt(2), {ia}, {aa});
                temp1.add_term(-1.0/std::sqrt(2), {ib}, {ab});

                temp1.simplify();

                add_pool_operator(1.0, temp1);
            }
        }

        for(size_t i=0; i<nocc_; i++){
            size_t ia = 2*i;
            size_t ib = 2*i+1;

            for(size_t j=i; j<nocc_; j++){
                size_t ja = 2*j;
                size_t jb = 2*j+1;

                for(size_t a=0; a<nvir_; a++){
                    size_t aa = 2*nocc_ + 2*a;
                    size_t ab = 2*nocc_ + 2*a+1;

                    for(size_t b=a; b<nvir_; b++){
                        size_t ba = 2*nocc_ + 2*b;
                        size_t bb = 2*nocc_ + 2*b+1;

                        SQOperator temp2a;
                        if((aa != ba) && (ia != ja)){
                            temp2a.add_term(2.0/std::sqrt(12), {aa,ba}, {ia,ja});
                        }
                        if((ab != bb ) && (ib != jb)){
                            temp2a.add_term(2.0/std::sqrt(12), {ab,bb}, {ib,jb});
                        }
                        if((aa != bb) && (ia != jb)){
                            temp2a.add_term(1.0/std::sqrt(12), {aa,bb}, {ia,jb});
                        }
                        if((ab != ba) && (ib != ja)){
                            temp2a.add_term(1.0/std::sqrt(12), {ab,ba}, {ib,ja});
                        }
                        if((aa != bb) && (ib != ja)){
                            temp2a.add_term(1.0/std::sqrt(12), {aa,bb}, {ib,ja});
                        }
                        if((ab != ba) && (ia != jb)){
                            temp2a.add_term(1.0/std::sqrt(12), {ab,ba}, {ia,jb});
                        }

                        // hermitian conjugate
                        if((ja != ia) && (ba != aa)){
                            temp2a.add_term(-2.0/std::sqrt(12), {ja,ia}, {ba,aa});
                        }
                        if((jb != ib ) && (bb != ab)){
                            temp2a.add_term(-2.0/std::sqrt(12), {jb,ib}, {bb,ab});
                        }
                        if((jb != ia) && (bb != aa)){
                            temp2a.add_term(-1.0/std::sqrt(12), {jb,ia}, {bb,aa});
                        }
                        if((ja != ib) && (ba != ab)){
                            temp2a.add_term(-1.0/std::sqrt(12), {ja,ib}, {ba,ab});
                        }
                        if((ja != ib) && (bb != aa)){
                            temp2a.add_term(-1.0/std::sqrt(12), {ja,ib}, {bb,aa});
                        }
                        if((jb != ia) && (ba != ab)){
                            temp2a.add_term(-1.0/std::sqrt(12), {jb,ia}, {ba,ab});
                        }

                        SQOperator temp2b;
                        if((aa != bb) && (ia != jb)){
                            temp2b.add_term(0.5, {aa,bb}, {ia,jb});
                        }
                        if((ab != ba) && (ib != ja)){
                            temp2b.add_term(0.5, {ab,ba}, {ib,ja});
                        }
                        if((aa != bb) && (ib != ja)){
                            temp2b.add_term(-0.5, {aa,bb}, {ib,ja});
                        }
                        if((ab != ba) && (ia != jb)){
                            temp2b.add_term(-0.5, {ab,ba}, {ia,jb});
                        }

                        // hermetian conjugate
                        if((jb != ia) && (bb != aa)){
                            temp2b.add_term(-0.5, {jb,ia}, {bb,aa});
                        }
                        if((ja != ib) && (ba != ab)){
                            temp2b.add_term(-0.5, {ja,ib}, {ba,ab});
                        }
                        if((ja != ib) && (bb != aa)){
                            temp2b.add_term(0.5, {ja,ib}, {bb,aa});
                        }
                        if((jb != ia) && (ba != ab)){
                            temp2b.add_term(0.5, {jb,ia}, {ba,ab});
                        }

                        temp2a.simplify();
                        temp2b.simplify();

                        std::complex<double> temp2a_norm(0.0, 0.0);
                        std::complex<double> temp2b_norm(0.0, 0.0);
                        for (const auto& term : temp2a.terms()){
                            temp2a_norm += std::norm(std::get<0>(term));
                        }
                        for (const auto& term : temp2b.terms()){
                            temp2b_norm += std::norm(std::get<0>(term));
                        }
                        temp2a.mult_coeffs(std::sqrt(2.0/temp2a_norm));
                        temp2b.mult_coeffs(std::sqrt(2.0/temp2b_norm));

                        if(temp2a.terms().size() > 0){
                            add_pool_operator(1.0, temp2a);
                        }
                        if(temp2b.terms().size() > 0){
                            add_pool_operator(1.0, temp2b);
                        }
                    }
                }
            }
        }
    } else {
        throw std::invalid_argument( "Invalid pool_type specified." );
    }
}

void SQOpPool::fill_pool_kUpCCGSD(int kmax)
{
    for(int k=0; k < kmax; ++k){
        size_t norb = nocc_ + nvir_;
        for(size_t i=0; i<norb; i++){
            size_t ia = 2*i;
            size_t ib = 2*i+1;
            for (size_t a=i; a<norb; a++){
                size_t aa = 2*a;
                size_t ab = 2*a+1;

                if( aa != ia ){
                    SQOperator temp1a;
                    temp1a.add_term(+1.0, {aa}, {ia});
                    temp1a.add_term(-1.0, {ia}, {aa});
                    temp1a.simplify();
                    if(temp1a.terms().size() > 0){
                        add_pool_operator(1.0, temp1a);
                    }
                }

                if( ab != ib ){
                    SQOperator temp1b;
                    temp1b.add_term(+1.0, {ab}, {ib});
                    temp1b.add_term(-1.0, {ib}, {ab});
                    temp1b.simplify();
                    if(temp1b.terms().size() > 0){
                        add_pool_operator(1.0, temp1b);
                    }
                }
            }
        }

        std::vector< std::vector<size_t> > uniqe_2bdy;
        std::vector< std::vector<size_t> > adjnt_2bdy;

        for(size_t p=0; p<norb; ++p){
            size_t pa = 2 * p;
            size_t pb = 2 * p + 1;
            for(size_t q=0; q<norb; ++q){
                size_t qa = 2 * q;
                size_t qb = 2 * q + 1;

                if((pa != qa) && (pb != qb)){
                    SQOperator temp2abab;
                    temp2abab.add_term(-1.0, {pa,pb}, {qa,qb});
                    temp2abab.add_term(+1.0, {qb,qa}, {pb,pa});
                    temp2abab.simplify();
                    if(temp2abab.terms().size() > 0){
                        std::vector<size_t> vtemp {std::get<1>(temp2abab.terms()[0])[0], std::get<1>(temp2abab.terms()[0])[1], std::get<2>(temp2abab.terms()[0])[0], std::get<2>(temp2abab.terms()[0])[1]};
                        std::vector<size_t> vadjt {std::get<1>(temp2abab.terms()[1])[0], std::get<1>(temp2abab.terms()[1])[1], std::get<2>(temp2abab.terms()[1])[0], std::get<2>(temp2abab.terms()[1])[1]};
                        if( (std::find(uniqe_2bdy.begin(), uniqe_2bdy.end(), vtemp) == uniqe_2bdy.end()) ){
                            if( (std::find(adjnt_2bdy.begin(), adjnt_2bdy.end(), vtemp) == adjnt_2bdy.end()) ){
                                uniqe_2bdy.push_back(vtemp);
                                adjnt_2bdy.push_back(vadjt);
                                add_pool_operator(1.0, temp2abab);
                            }
                        }
                    }
                }
            }
        }        
    }
}

void SQOpPool::fill_pool_kUpCCGSDx(int kmax)
{
    for(int k=0; k < kmax; ++k){
        SQOpPool ph_pool;
        ph_pool.nocc_ = nocc_;
        ph_pool.nvir_ = nvir_;
        ph_pool.orb_irreps_to_int_ = orb_irreps_to_int_;
        ph_pool.target_irrep_ = target_irrep_;
        fill_kupccgsd_particle_hole_terms(ph_pool, nocc_, nvir_);

        SQOpPool generalized_pool;
        generalized_pool.nocc_ = nocc_;
        generalized_pool.nvir_ = nvir_;
        generalized_pool.orb_irreps_to_int_ = orb_irreps_to_int_;
        generalized_pool.target_irrep_ = target_irrep_;
        generalized_pool.fill_pool_kUpCCGSD(1);

        std::set<std::string> seen;
        append_unique_pool_terms(*this, ph_pool, seen);
        append_unique_pool_terms(*this, generalized_pool, seen);
    }
}

void SQOpPool::fill_pool_sq_hva(std::complex<double> coeff, const SQOperator& sq_op){
    std::vector<std::pair< std::vector<size_t>, std::vector<size_t>>> h_vec;
    std::vector<std::pair< std::vector<size_t>, std::vector<size_t>>> hd_vec;

    for (size_t l = 0; l < sq_op.terms().size(); l++){
        std::pair< std::vector<size_t>, std::vector<size_t>> h;
        std::pair< std::vector<size_t>, std::vector<size_t>> hd;

        std::complex<double> hl = std::get<0>(sq_op.terms()[l]);
        h.first  = std::get<1>(sq_op.terms()[l]);
        h.second = std::get<2>(sq_op.terms()[l]);

        // skip the scalar term.
        if(h.first.size()==0 and h.second.size()==0){
            continue;
        }

        hd.first = h.second;
        hd.second = h.first;

        std::reverse(hd.first.begin(), hd.first.end());
        std::reverse(hd.second.begin(), hd.second.end());

        std::sort(h.first.begin(), h.first.end());
        std::sort(h.second.begin(), h.second.end());

        std::sort(hd.first.begin(), hd.first.end());
        std::sort(hd.second.begin(), hd.second.end());
        
        // Determine if term is in current set of terms or term adjoints
        // if it isn't found in either then append the vectors
        if (std::find(h_vec.begin(), h_vec.end(), h) == h_vec.end()){
            if (std::find(hd_vec.begin(), hd_vec.end(), h) == hd_vec.end()){
                SQOperator temp;
                if(h == hd or h.first == h.second){
                    // (Nick) Need this checked out for sure
                    temp.add_term(0.5, h.first,  h.second);
                    temp.add_term(0.5, hd.first, hd.second);
                } else {
                    temp.add_term(1.0, h.first,  h.second);
                    temp.add_term(1.0, hd.first, hd.second);
                    temp.simplify();
                }

                terms_.push_back(std::make_pair(coeff, temp));

                h_vec.push_back(h);
                hd_vec.push_back(hd);

            }
        } 
    }
}

void SQOpPool::fill_pool_df_trotter(
    const DFHamiltonian& df_ham,
    const std::complex<double> coeff)
{
    // structure should resembel the apply DFHam funciton in fci_computer.cc
    size_t nleaves = df_ham.get_trotter_basis_change_matrices().size();

    if (nleaves - 1 != df_ham.get_scaled_density_density_matrices().size()){
        throw std::invalid_argument("Incompatiable array lengths.");
    }

    // NOTE(Nick): the first DF givens rotation should ALREADY be time scalled 
    // by the user, otherwise this routine won't match the trotter evolution.
    // As such we anticipate the resulting pool will be evolved by coeff=dt,
    // and so this will correctly scale the coefficents. 
    append_givens_ops_sector(
        df_ham.get_trotter_basis_change_matrices()[0],
        1.0/coeff,
        true
    );

    append_givens_ops_sector(
        df_ham.get_trotter_basis_change_matrices()[0],
        1.0/coeff,
        false
    );

    for (size_t l = 1; l < nleaves; ++l) {
        append_diagonal_ops_all(
            df_ham.get_scaled_density_density_matrices()[l - 1],
            1.0
        );

        // NOTE(Nick): subsequent givens rotations DON'T need time scaling
        append_givens_ops_sector(
            df_ham.get_trotter_basis_change_matrices()[l],
            1.0/coeff,
            true
        );

        append_givens_ops_sector(
            df_ham.get_trotter_basis_change_matrices()[l],
            1.0/coeff,
            false
        );
    }
}

void SQOpPool::append_givens_ops_sector(
    const Tensor& U,
    const std::complex<double> coeff,
    const bool is_alfa)
{
    size_t sigma = 0;
    if (is_alfa){ 
        sigma = 0; 
    } else {
        sigma = 1;
    }

    U.square_error();
    Tensor U2 = U;

    //NOTE(Nick): May be SLOW, or don't need to compute, could just store rots_and_diag
    // in DFHamiltonain class pass directly.
    auto rots_and_diag = DFHamiltonian::givens_decomposition_square(U2);

    auto ivec = std::get<0>(rots_and_diag);
    auto jvec = std::get<1>(rots_and_diag);
    auto thts = std::get<2>(rots_and_diag);
    auto phis = std::get<3>(rots_and_diag);

    auto diags = std::get<4>(rots_and_diag);

    for (size_t k = 0; k < ivec.size(); k++){
        size_t i = ivec[k];
        size_t j = jvec[k];
        double tht = thts[k];
        double phi = phis[k];

        if (std::abs(phi) > 1.0e-12){
            SQOperator num_op1; 
            num_op1.add_term(-phi/2.0, {2 * j + sigma}, {2 * j + sigma});
            num_op1.add_term(-phi/2.0, {2 * j + sigma}, {2 * j + sigma});   
            terms_.push_back(std::make_pair(coeff, num_op1));      
        }

        if (std::abs(tht) > 1.0e-12) {
            std::complex<double> itheta(0.0, tht);
            SQOperator single;
            single.add_term(-itheta, {2 * i + sigma}, {2 * j + sigma});
            single.add_term(+itheta, {2 * j + sigma}, {2 * i + sigma});
            terms_.push_back(std::make_pair(coeff, single));  
        }
    }
        
    for (size_t l = 0; l < diags.size(); l++){
        if (std::abs(diags[l]) > 1.0e-12) {
            double diag_angle = std::atan2(diags[l].imag(), diags[l].real());
            SQOperator num_op2;
            num_op2.add_term(-diag_angle/2.0, {2 * l + sigma}, {2 * l + sigma});
            num_op2.add_term(-diag_angle/2.0, {2 * l + sigma}, {2 * l + sigma});
            terms_.push_back(std::make_pair(coeff, num_op2));  
        }
    }
}

void SQOpPool::append_diagonal_ops_all(
    const Tensor& V, 
    const std::complex<double> coeff)
{
    V.square_error();

    int norbs = V.shape()[0];

    for (size_t p = 0; p < norbs; p++) {
        for (size_t q = 0; q < norbs; q++) {
            for (size_t sig = 0; sig < 2; sig++){
                for (size_t tau = 0; tau < 2; tau++){
                    size_t pq = p*norbs + q;
                    std::complex<double> vpq = V.read_data()[pq];
                    if (std::abs(vpq) > 1.0e-12){
                        SQOperator num_op;
                        if(2 * p + sig == 2 * q + tau && 2 * q + tau == 2 * p + sig && 2 * p + sig == 2 * q + tau){
                            num_op.add_term(0.5 * vpq, {2 * p + sig}, {2 * p + sig});
                            num_op.add_term(0.5 * vpq, {2 * p + sig}, {2 * p + sig});
                        } else {
                            num_op.add_term(-0.5 * vpq, {2 * p + sig, 2 * q + tau}, {2 * p + sig, 2 * q + tau});
                            num_op.add_term(-0.5 * vpq, {2 * p + sig, 2 * q + tau}, {2 * p + sig, 2 * q + tau});
                        }
                        terms_.push_back(std::make_pair(coeff, num_op));    
                    }
                }
            }
        }
    }
}

std::vector<int> SQOpPool::get_count_pauli_terms_ex_deex() const
{
    std::vector<int> counts(size(terms_));
    for (size_t i = 0; i < terms_.size(); ++i) {
        counts[i] = terms_[i].second.count_pauli_terms_ex_deex();
    }
    
    return counts;

}

int SQOpPool::count_cnot_for_jw_exponential(bool qubit_excitations, int trotter_number) const
{
    int total = 0;
    for (size_t i = 0; i < terms_.size(); ++i) {
        total += count_cnot_for_term_jw_exponential(i, qubit_excitations, trotter_number);
    }
    return total;
}

int SQOpPool::count_cnot_for_term_jw_exponential(
    size_t term_index,
    bool qubit_excitations,
    int trotter_number) const
{
    if (term_index >= terms_.size()) {
        throw std::out_of_range("SQOpPool term index is out of range.");
    }
    if (std::abs(terms_[term_index].first) <= 1.0e-12) {
        return 0;
    }
    return terms_[term_index].second.count_cnot_for_jw_exponential(
        qubit_excitations,
        trotter_number);
}

/**
 * @brief Construct the commutativity‐graph tensor for the current operator pool.
 *
 * This function builds and returns an N×N real‐valued Tensor W, where N is the
 * number of second‐quantized operators in the pool (i.e., terms_.size()).  W encodes
 * a simple proxy for the leading Trotter error between any two operators: smaller
 * off‐diagonal entries signal more commuting (or weakly non‐commuting) pairs.
 *
 * In more detail:
 *  - We label each pool operator by an index i∈[0,N).  Each operator is itself a
 *    linear combination of k subterms:
 *      SQOperator op_i = ∑ₖ α_{i,k} · ( creation indices cre_{i,k},
 *                                       annihilation indices ann_{i,k} )
 *  - Each subterm k of operator i is retrieved via op_i.terms(), which returns
 *    tuples (α_{i,k}, cre_{i,k}, ann_{i,k}).
 *  - For any two subterms (α_{i,k},cre_{i,k},ann_{i,k}) and (α_{j,ℓ},cre_{j,ℓ},ann_{j,ℓ}),
 *    define the match‐count M by summing over the four set‐intersections:
 *      M = |ann_{i,k} ∩ cre_{j,ℓ}|
 *        + |ann_{i,k} ∩ ann_{j,ℓ}|
 *        + |cre_{i,k} ∩ cre_{j,ℓ}|
 *        + |cre_{i,k} ∩ ann_{j,ℓ}|
 *  - The pairwise contribution of those two subterms to W(i,j) is
 *      |α_{i,k}| · |α_{j,ℓ}| · M.
 *  - We sum this quantity over all k,ℓ to obtain the symmetric weight W(i,j).
 *
 * The resulting Tensor W can be interpreted as the adjacency‐matrix of an
 * undirected “commutativity graph” whose edge‐weights approximate the Frobenius
 * norm of the commutator between any two pool operators.  Lower weights imply
 * smaller commutators and hence reduced Trotter error when those terms are
 * placed adjacent in a product‐formula ordering.
 *
 * Usage:
 *   Tensor W = my_pool.commutativity_graph();
 *   // then feed W to reorder_terms(W) or to any TSP / graph‐ordering heuristic.
 *
 * @return Tensor  An N×N real matrix of nonnegative weights W(i,j).
 *
 * @throws std::invalid_argument if the Tensor cannot be allocated or if
 *         terms_.size() == 0 (N==0).
 */
Tensor SQOpPool::get_commutativity_graph() const {
    size_t N = terms_.size();
    // construct an N×N tensor filled with zeros
    Tensor W({N, N});
    W.zero();

    // Helper to compute M for two (cre,ann) pairs
    auto count_matches = [&](auto const& cre1, auto const& ann1,
                             auto const& cre2, auto const& ann2) {
        size_t M = 0;
        // |ann1 ∩ cre2|
        for (auto i : ann1)
            for (auto j : cre2)
                if (i == j) ++M;
        // |ann1 ∩ ann2|
        for (auto i : ann1)
            for (auto j : ann2)
                if (i == j) ++M;
        // |cre1 ∩ cre2|
        for (auto i : cre1)
            for (auto j : cre2)
                if (i == j) ++M;
        // |cre1 ∩ ann2|
        for (auto i : cre1)
            for (auto j : ann2)
                if (i == j) ++M;
        return M;
    };

    // Loop over all distinct pairs i<j
    for (size_t i = 0; i < N; ++i) {
        auto const& [coeff_i, op_i] = terms_[i];
        // gather subterms of operator i
        auto const& sub_i = op_i.terms();  
        for (size_t j = i + 1; j < N; ++j) {
            auto const& [coeff_j, op_j] = terms_[j];
            auto const& sub_j = op_j.terms();

            double w_ij = 0.0;
            // sum over all subterms k∈i, ℓ∈j
            for (auto const& s_i : sub_i) {
                double alfa_ik = std::abs(std::get<0>(s_i)) * std::abs(coeff_i);
                auto const& cre_i = std::get<1>(s_i);
                auto const& ann_i = std::get<2>(s_i);
                for (auto const& s_j : sub_j) {
                    double alfa_jl = std::abs(std::get<0>(s_j)) * std::abs(coeff_j);
                    auto const& cre_j = std::get<1>(s_j);
                    auto const& ann_j = std::get<2>(s_j);
                    size_t M = count_matches(cre_i, ann_i, cre_j, ann_j);
                    w_ij += alfa_ik * alfa_jl * static_cast<double>(M);
                }
            }
            W.set({i, j}, w_ij);
            W.set({j, i}, w_ij);  // enforce symmetry
        }
    }
    return W;
}

/**
 * Reorder the pool’s terms to minimize Trotter error by placing
 * weakly non-commuting operators next to one another.
 *
 * This implements a simple greedy “nearest-neighbor” tour on the
 * commutativity graph W: at each step, pick the as-yet-unused operator
 * whose edge-weight to the last-placed operator is minimal.
 *
 * How it works:
 * 1. Verify that W is an N×N matrix matching the number of terms.
 * 2. Maintain a boolean array `used[]` marking which terms have been placed.
 * 3. Seed the ordering at index 0 (or any heuristic choice).
 * 4. For each subsequent position, scan all unused indices `j` and select
 *    the one that minimizes W[current,j], i.e. the operator that commutes
 *    best (smallest commutator norm) with the last one placed.
 * 5. After selecting N operators, overwrite `terms_` with the new sequence.
 */
void SQOpPool::reorder_terms_from_graph(const Tensor &W) {
    size_t N = terms_.size();
    // 1) Sanity-check: W must be N×N
    auto shape = W.shape();
    if (shape.size() != 2 || shape[0] != N || shape[1] != N) {
        throw std::invalid_argument(
            "reorder_terms: W must be an N×N tensor matching terms_.size()");
    }

    // 2) Track which terms have been placed
    std::vector<bool> used(N, false);
    std::vector<std::pair<std::complex<double>, SQOperator>> new_terms;
    new_terms.reserve(N);

    // 3) Seed the tour at index 0 (could also pick the term with smallest total weight)
    size_t current = 0;
    used[current] = true;
    new_terms.push_back(terms_[current]);

    // 4) Greedy nearest-neighbor: build the rest of the tour
    for (size_t step = 1; step < N; ++step) {
        double best_w = std::numeric_limits<double>::infinity();
        size_t best_j = N;  // sentinel for “none found”
        // Scan all unused terms and pick the one minimizing W[current,j]
        for (size_t j = 0; j < N; ++j) {
            if (!used[j]) {
                double w = std::real(W.get({current, j})); 
                if (w < best_w) {
                    best_w = w;
                    best_j = j;
                }
            }
        }
        // Mark and append the chosen operator
        if (best_j >= N) {
            // No unused terms left (should only happen if N==0)
            break;
        }
        used[best_j] = true;
        new_terms.push_back(terms_[best_j]);
        current = best_j;
    }

    // 5) Replace the old term order with the new low-commutator sequence
    terms_ = std::move(new_terms);
}

/**
 * @brief Randomly shuffle the order of the SQOpPool terms.
 *
 * This uses a reproducible pseudo‐random number generator (std::mt19937)
 * initialized with the provided integer seed.  Calling this function with
 * the same seed will always produce the same permutation of `terms_`.
 *
 * @param seed  An integer seed for the random number generator.
 */
void SQOpPool::shuffle_terms_random(int seed) {
    // Initialize a Mersenne Twister RNG with the user‐provided seed
    std::mt19937 rng(seed);

    // Perform an in‐place Fisher–Yates shuffle of the terms_ vector
    std::shuffle(terms_.begin(), terms_.end(), rng);
}

std::string SQOpPool::str() const{
    std::vector<std::string> s;
    s.push_back("");
    int counter = 0;
    for (const auto& term : terms_) {
        s.push_back("----->");
        s.push_back(std::to_string(counter));
        s.push_back("<-----\n");
        s.push_back(to_string(term.first));
        s.push_back("[\n");
        s.push_back(term.second.str());
        s.push_back("]\n\n");
        counter++;
    }
    return join(s, " ");
}
