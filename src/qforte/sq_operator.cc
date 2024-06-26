#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <map>

#include "helpers.h"
#include "gate.h"
#include "circuit.h"
#include "qubit_operator.h"
#include "sq_operator.h"

void SQOperator::add_term(std::complex<double> circ_coeff, const std::vector<size_t>& cre_ops, const std::vector<size_t>& ann_ops) {
    terms_.push_back(std::make_tuple(circ_coeff, cre_ops, ann_ops));
}

void SQOperator::add_op(const SQOperator& qo) {
    terms_.insert(terms_.end(), qo.terms().begin(), qo.terms().end());
}

void SQOperator::set_coeffs(const std::vector<std::complex<double>>& new_coeffs) {
    if(new_coeffs.size() != terms_.size()){
        throw std::invalid_argument( "number of new coefficients for quantum operator must equal " );
    }
    for (auto l = 0; l < new_coeffs.size(); l++) {
        std::get<0>(terms_[l]) = new_coeffs[l];
    }
}

void SQOperator::mult_coeffs(const std::complex<double>& multiplier) {
    for (auto& term : terms_){
        std::get<0>(term) *= multiplier;
    }
}

const std::vector<std::tuple<std::complex<double>, std::vector<size_t>, std::vector<size_t>>>& SQOperator::terms() const {
    return terms_;
}

// TODO(Tyler): Need to expose and need a test case
std::pair<int, int> SQOperator::get_largest_alfa_beta_indices() const {
    int maxodd = -1;
    int maxeven = -1;
    // for term in ops.terms:
    for (auto& term : terms_){
        // loop over creators
        for (auto& cre_idx : std::get<1>(term) ){
            if (cre_idx % 2) maxodd = std::max(static_cast<int>(cre_idx), maxodd);
            else maxeven = std::max(static_cast<int>(cre_idx), maxeven);
        } 
        // loop over anihilators
        for (auto& ann_idx : std::get<2>(term) ){
            if (ann_idx % 2) maxodd = std::max(static_cast<int>(ann_idx), maxodd);
            else maxeven = std::max(static_cast<int>(ann_idx), maxeven);
        }        
    }
    return std::make_pair(maxeven, maxodd);
}

// TODO(Tyler): Need to expose and need a test case
int SQOperator::many_body_order() const {
    int max_rank = -1;
    for (auto& term : terms_){
        int term_rank = 0;
        term_rank += std::get<1>(term).size();
        term_rank += std::get<2>(term).size();
        max_rank = std::max(max_rank, term_rank);        
    }
    return max_rank;
}

// TODO(Tyler): Need to expose and need a test case
std::vector<int> SQOperator::ranks_present() const {
    std::vector<int> ranks_present;
    for (auto& term : terms_){
        int term_rank = 0;
        term_rank += std::get<1>(term).size();
        term_rank += std::get<2>(term).size();
        if ( std::count(ranks_present.begin(), ranks_present.end(), term_rank) == false) {
            ranks_present.push_back(term_rank);
        }     
    }
    std::sort(ranks_present.begin(), ranks_present.end());
    return ranks_present;
}

int SQOperator::canonicalize_helper(std::vector<size_t>& op_list) const {
    auto temp_op = op_list;
    auto length = temp_op.size();
    {
        std::vector<int> temp(length);
        std::iota(std::begin(temp), std::end(temp), 0);
        std::sort(temp.begin(), temp.end(),
            [&](const int& i, const int& j) {
                return (temp_op[i] > temp_op[j]);
            }
        );
        for (int i = 0; i < length; i++) {
            op_list[i] = temp_op[temp[i]];
        }
        return (permutation_phase(temp)) ? -1 : 1;
    }

}

void SQOperator::canonical_order_single_term(std::tuple< std::complex<double>, std::vector<size_t>, std::vector<size_t>>& term ){
    std::get<0>(term) *= canonicalize_helper(std::get<1>(term));
    std::get<0>(term) *= canonicalize_helper(std::get<2>(term));
}

void SQOperator::canonical_order() {
    for (auto& term : terms_) {
        canonical_order_single_term(term);
    }
}

void SQOperator::simplify() {
    canonical_order();
    std::map<std::pair<std::vector<size_t>, std::vector<size_t>>, std::complex<double>> unique_terms;
    for (const auto& term : terms_) {
        auto pair = std::make_pair(std::get<1>(term), std::get<2>(term));
        if (unique_terms.find(pair) == unique_terms.end() ) {
            unique_terms.insert(std::make_pair(pair, std::get<0>(term)));
        } else {
            unique_terms[pair] += std::get<0>(term);
        }
    }
    terms_.clear();
    for (const auto &unique_term : unique_terms){
        if (std::abs(unique_term.second) > 1.0e-12){
            terms_.push_back(std::make_tuple(unique_term.second, unique_term.first.first, unique_term.first.second));
        }
    }
}

bool SQOperator::permutation_phase(std::vector<int> p) const {
    std::vector<int> a(p.size());
    std::iota (std::begin(a), std::end(a), 0);
    size_t cnt = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        while (i != p[i]) {
            ++cnt;
            std::swap (a[i], a[p[i]]);
            std::swap (p[i], p[p[i]]);
        }
    }
    if(cnt % 2 == 0) {
        return false;
    } else {
        return true;
    }
}

void SQOperator::jw_helper(QubitOperator& holder, const std::vector<size_t>& operators, bool creator, bool qubit_excitation) const {
    std::complex<double> halfi(0.0, 0.5);
    if (creator) { halfi *= -1; };

    for (const auto& sq_op : operators) {
        QubitOperator temp;
        Circuit Xcirc;
        Circuit Ycirc;

        if (not qubit_excitation) {
            for (int k = 0; k < sq_op; k++) {
                Xcirc.add_gate(make_gate("Z", k, k));
                Ycirc.add_gate(make_gate("Z", k, k));
            }
        }

        Xcirc.add_gate(make_gate("X", sq_op, sq_op));
        Ycirc.add_gate(make_gate("Y", sq_op, sq_op));
        temp.add_term(0.5, Xcirc);
        temp.add_term(halfi, Ycirc);

        if (holder.terms().size() == 0) {
            holder.add_op(temp);
        } else {
            holder.operator_product(temp);
        }
    }
}

QubitOperator SQOperator::jw_transform(bool qubit_excitation) {
    /// The simplify() function also brings second-quantized operators
    /// to normal order. This also ensures the 1-to-1 mapping between
    /// second-quantized and qubit operators when qubit_excitation=True
    simplify();
    QubitOperator qo;

    for (const auto& fermion_operator : terms_) {
        auto cre_length = std::get<1>(fermion_operator).size();
        auto ann_length = std::get<2>(fermion_operator).size();

        if (cre_length == 0 && ann_length == 0) {
            // Scalars need special logic.
            Circuit scalar_circ;
            QubitOperator scalar_op;
            scalar_op.add_term(std::get<0>(fermion_operator), scalar_circ);
            qo.add_op(scalar_op);
            continue;
        }

        QubitOperator temp1;
        jw_helper(temp1, std::get<1>(fermion_operator), true, qubit_excitation);
        jw_helper(temp1, std::get<2>(fermion_operator), false, qubit_excitation);

        temp1.mult_coeffs(std::get<0>(fermion_operator));
        qo.add_op(temp1);
    }

    qo.simplify();

    return qo;
}

std::vector<SQOperator> SQOperator::split_by_rank(bool simplify){

    if (simplify){
        SQOperator::simplify();
    }

    // What we will be returning
    std::vector<SQOperator> return_vec;

    // list of all the unique ranks
    std::vector<int> rank_list = ranks_present();

    // how many unique ranks there are
    int unique = ranks_present().size();


    std::map<int, SQOperator> group_map;

    // Populate the map with key-value pairs for respective rank
    for (int i = 0; i < unique; i++){
        group_map[rank_list[i]] = SQOperator();
    }

    // Go through each term and add it to its respective SQOperator in our map
    for (int i = 0; i < terms_.size(); i++){

        int rank = (std::get<1>(terms_[i])).size() + (std::get<2>(terms_[i])).size();

        auto it = group_map.find(rank);

        if (it != group_map.end()){

            SQOperator& my_op = it->second;
            
            my_op.add_term(std::get<0>(terms_[i]), std::get<1>(terms_[i]), std::get<2>(terms_[i]));
        }

    }
    // Add all our SQOperators to a vector to return
    for (const auto& entry : group_map){
        SQOperator op = entry.second;
        return_vec.push_back(op);
    }

    return return_vec;

}

std::string SQOperator::str() const {
    std::vector<std::string> s;
    s.push_back("");
    for (const auto& term : terms_) {
        s.push_back(to_string(std::get<0>(term)));
        s.push_back("(");
        for (auto k: std::get<1>(term)) {
            s.push_back(std::to_string(k) + "^");
        }
        for (auto k: std::get<2>(term)) {
            s.push_back(std::to_string(k));
        }
        s.push_back(")\n");
    }
    return join(s, " ");
}
