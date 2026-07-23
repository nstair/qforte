#include "fci_graph.h"
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <iostream>

#include <bitset>

/// Custom construcotr
FCIGraph::FCIGraph(int nalfa, int nbeta, int norb) 
{
    if (norb < 0)
        throw std::invalid_argument("norb needs to be >= 0");
    if (nalfa < 0)
        throw std::invalid_argument("nalfa needs to be >= 0");
    if (nbeta < 0)
        throw std::invalid_argument("nbeta needs to be >= 0");
    if (nalfa > norb)
        throw std::invalid_argument("nalfa needs to be <= norb");
    if (nbeta > norb)
        throw std::invalid_argument("nbeta needs to be <= norb");

    norb_ = norb;
    nalfa_ = nalfa;
    nbeta_ = nbeta;
    lena_ = binom(norb, nalfa); 
    lenb_ = binom(norb, nbeta); 

    std::tie(astr_, aind_) = build_strings(nalfa_, lena_);
    std::tie(bstr_, bind_) = build_strings(nbeta_, lenb_);

    alfa_map_ = build_mapping(astr_, nalfa_, aind_);
    beta_map_ = build_mapping(bstr_, nbeta_, bind_);

    /// NICK: This is an intermediate and likely not needed...
    dexca_ = map_to_deexc(alfa_map_, lena_, norb_, nalfa_);
    dexcb_ = map_to_deexc(beta_map_, lenb_, norb_, nbeta_);

    dexca_vec_ = unroll_from_3d(dexca_);
    dexcb_vec_ = unroll_from_3d(dexcb_);
}

FCIGraph::FCIGraph() : FCIGraph(0, 0, 0) {}

std::pair<std::vector<uint64_t>, std::unordered_map<uint64_t, size_t>> FCIGraph::build_strings(
    int nele, 
    size_t length) 
{
    int norb = norb_;

    std::vector<uint64_t> blist = get_lex_bitstrings(nele, norb); // Assuming get_lex_bitstrings is available

    std::vector<uint64_t> string_list;

    std::unordered_map<uint64_t, size_t> index_list;

    std::vector<std::vector<uint64_t>> Z = get_z_matrix(norb, nele);

    string_list = std::vector<uint64_t>(length, 0);

    for (size_t i = 0; i < length; ++i) {

        uint64_t occ = blist[i];

        size_t address = build_string_address(
            nele, 
            norb, 
            occ,
            Z); 

        string_list[address] = occ;
    }

    for (size_t address = 0; address < string_list.size(); ++address) {
        uint64_t wbit = string_list[address];
        index_list[wbit] = address;
    }

    return std::make_pair(string_list, index_list);
}

Spinmap FCIGraph::build_mapping(
    const std::vector<uint64_t>& strings, 
    int nele, 
    const std::unordered_map<uint64_t, size_t>& index) 
{
    int norb = norb_;
    Spinmap out;

    for (int iorb = 0; iorb < norb; ++iorb) {
        for (int jorb = 0; jorb < norb; ++jorb) {
            std::vector<std::tuple<int, int, int>> value;
            for (uint64_t string : strings) {
                if (get_bit(string, jorb) && !get_bit(string, iorb)) {
                    int parity = count_bits_between(string, iorb, jorb); 
                    int sign = (parity % 2 == 0) ? 1 : -1;
                    value.push_back(
                        std::make_tuple(
                            index.at(string), 
                            index.at(unset_bit(set_bit(string, iorb), jorb)), 
                            sign)
                        );
                } else if (iorb == jorb && get_bit(string, iorb)) {
                    // std::cout << "I get here B" << std::endl;
                    value.push_back(std::make_tuple(index.at(string), index.at(string), 1));
                }
            }
            out[std::make_pair(iorb, jorb)] = value;
        }
    }

    Spinmap result;
    for (const auto& entry : out) {
        const auto& key = entry.first;
        const auto& value = entry.second;
        std::vector<std::tuple<int,int,int>> casted_value;
        for (const auto& tpl : value) {
            casted_value.push_back(
                std::make_tuple(
                    std::get<0>(tpl), 
                    std::get<1>(tpl), 
                    std::get<2>(tpl))
                );
        }
        result[key] = casted_value;
    }
    return result;
}

std::vector<std::vector<std::vector<int>>> FCIGraph::map_to_deexc(
    const Spinmap& mappings, 
    int states, 
    int norbs,
    int nele) 
{
    int lk = nele * (norbs - nele + 1);

    std::vector<std::vector<std::vector<int>>> dexc(
        states, 
        std::vector<std::vector<int>>(lk, std::vector<int>(3, 0)));

    std::vector<int> index(states, 0);
    
    for (const auto& entry : mappings) {
        const auto& key = entry.first;
        const auto& values = entry.second;
        int i = key.first;
        int j = key.second;
        int idx = i * norbs + j;
        
        for (const auto& value : values) {
            int state = std::get<0>(value);
            int target = std::get<1>(value);
            int parity = std::get<2>(value);
            
            dexc[target][index[target]][0] = state;
            dexc[target][index[target]][1] = idx;
            dexc[target][index[target]][2] = parity;
            index[target]++;
        }
    }
    
    return dexc;
}

/// NICK: May be an accelerated version of this funciton, may also be important as it comes up in
// every instance of apply individual op!
std::tuple<int, std::vector<int>, std::vector<int>, std::vector<int>> FCIGraph::make_mapping_each(
    bool alpha, 
    const std::vector<int>& dag, 
    const std::vector<int>& undag) 
{
    timer_.reset();

    std::vector<uint64_t> strings;
    int length;
    
    if (alpha) {
        strings = get_astr();
        length = lena_;
    } else {
        strings = get_bstr();
        length = lenb_;
    }

    std::vector<int> source(length);
    std::vector<int> target(length);
    std::vector<int> parity(length);

    uint64_t dag_mask = 0;
    uint64_t undag_mask = 0;
    int count = 0;

    for (uint64_t i : dag) {
        if (std::find(undag.begin(), undag.end(), i) == undag.end()) {
            dag_mask = set_bit(dag_mask, i);
        }
    }

    for (uint64_t i : undag) { undag_mask = set_bit(undag_mask, i); }

    for (uint64_t index = 0; index < length; index++) {
        uint64_t current = strings[index];
        bool check = ((current & dag_mask) == 0) && ((current & undag_mask ^ undag_mask) == 0);
        
        if (check) {
            uint64_t tmp = current;
            uint64_t parity_value = 0;
            for (size_t i = undag.size(); i > 0; i--) {
                parity_value += count_bits_above(current, undag[i - 1]);
                current = unset_bit(current, undag[i - 1]);
            }
            
            for (size_t i = dag.size(); i > 0; i--) {
                parity_value += count_bits_above(current, dag[i - 1]);
                current = set_bit(current, dag[i - 1]);
            }
            
            source[count] = static_cast<int>(index);
            target[count] = static_cast<int>(current);
            parity[count] = static_cast<int>(parity_value % 2);
            count++;
        }
    }

    timer_.acc_record("make_mapping_each");
    //std::cout << timer_.acc_str_table() << std::endl;

    return std::make_tuple(
                    count,
                    source,
                    target,
                    parity);
}

std::tuple<int, std::vector<int>, std::vector<int>, std::vector<int>> FCIGraph::make_givens_mapping_each(
        bool alpha,
        const std::vector<int>& source_orbs,
        const std::vector<int>& target_orbs)
{

    if (source_orbs.size() != target_orbs.size()){
        throw std::runtime_error("must be same number of alpha annihilators/creators and beta annihilators/creators.");
    }

    if (source_orbs.size() != 1 && source_orbs.size() != 0) {
        throw std::invalid_argument("source_orbs must be a single orbital index or empty");
    }

    if (target_orbs.size() != 1 && target_orbs.size() != 0) {
        throw std::invalid_argument("target_orbs must be a single orbital index or empty");
    }

    timer_.reset();

    std::vector<uint64_t> strings;
    int length;
    
    if (alpha) {
        strings = get_astr();
        length = lena_;
    } else {
        strings = get_bstr();
        length = lenb_;
    }

    if (source_orbs.size() == 0 || target_orbs.size() == 0) {
        std::vector<int> source(length);
        std::vector<int> target(length);
        std::vector<int> parity(length, 1);

        for (int index = 0; index < length; index++){
            source[index] = index;
            target[index] = index;
        }

        return std::make_tuple(
                        length,
                        source,
                        target,
                        parity);
    }

    int source_orb = source_orbs[0];
    int target_orb = target_orbs[0];

    std::vector<int> source(length);
    std::vector<int> target(length);
    std::vector<int> parity(length);

    uint64_t source_mask = set_bit(0, source_orb);
    uint64_t target_mask = set_bit(0, target_orb);

    int low_orb = std::min(source_orb, target_orb);
    int high_orb = std::max(source_orb, target_orb);
    
    int count = 0;

    for (int index = 0; index < length; ++index){
        uint64_t current = strings[index];

        bool source_occupied = (current & source_mask) != 0; 
        bool target_empty = (current & target_mask) == 0;

        if (!source_occupied || !target_empty){
            continue;
        }

        /* Fermionic sign for a_target^dagger a_source:
         *
         * (-1)^(number of occupied same-spin orbitals strictly
         * between source_orb and target_orb).
         */
        int occupied_between = 0;

        for (int orb = low_orb + 1; orb < high_orb; ++orb) {
            occupied_between += static_cast<int>(
                (current >> orb) & uint64_t{1}
            );
        }

        const int parity_value =
            (occupied_between % 2 == 0) ? 1 : -1;

        uint64_t next = current;

        next = unset_bit(next, source_orb);
        next = set_bit(next, target_orb);

        source[count] = index;

        if (alpha){
            target[count] = get_aind_for_str(static_cast<int>(next));
        } else {
            target[count] = get_bind_for_str(static_cast<int>(next));
        }

        parity[count] = parity_value;
        ++count;
    }

    source.resize(count);
    target.resize(count);
    parity.resize(count);

    timer_.acc_record("make_givens_mapping_each");
    
    return std::make_tuple(
                    count,
                    source,
                    target,
                    parity);
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>,
           std::vector<int>, std::vector<int>, std::vector<int>>
FCIGraph::make_givens_matching_each(int q1, int q2)
{
    if (q1 == q2) {
        throw std::invalid_argument("Givens rotation qubits must be different");
    }

    if ((q1 % 2) != (q2 % 2)) {
        throw std::invalid_argument("Givens rotation qubits must be both alpha or both beta");
    }

    timer_.reset();

    const bool alpha_rotation = (q1 % 2 == 0);
    const int source_orb = q1 / 2;
    const int target_orb = q2 / 2;

    if (source_orb < 0 || source_orb >= norb_ || target_orb < 0 || target_orb >= norb_) {
        throw std::invalid_argument("Givens rotation qubit index is inconsistent with graph orbital count");
    }

    const uint64_t source_mask = set_bit(0, source_orb);
    const uint64_t target_mask = set_bit(0, target_orb);

    std::vector<int> sourcea;
    std::vector<int> targeta;
    std::vector<int> phasea;
    std::vector<int> sourceb;
    std::vector<int> targetb;
    std::vector<int> phaseb;

    const int low_orb = std::min(source_orb, target_orb);
    const int high_orb = std::max(source_orb, target_orb);

    auto phase_for_alpha_rotation = [low_orb, high_orb](uint64_t beta_mask) -> int {
        int parity = 0;

        // For an alpha Givens, alpha_source and alpha_target differ only by
        // moving one alpha electron between low_orb and high_orb.  In the
        // FCI spin-blocked <-> interleaved-qubit phase ratio, all common alpha
        // occupations cancel and only beta occupations in [low, high) remain.
        for (int b = low_orb; b < high_orb; ++b) {
            parity ^= static_cast<int>((beta_mask >> b) & uint64_t{1});
        }

        return parity ? -1 : 1;
    };

    auto phase_for_beta_rotation = [low_orb, high_orb](uint64_t alpha_mask) -> int {
        int parity = 0;

        // For a beta Givens, the changing beta electron crosses the occupied
        // alpha creators with spatial indices in (low, high].
        for (int a = low_orb + 1; a <= high_orb; ++a) {
            parity ^= static_cast<int>((alpha_mask >> a) & uint64_t{1});
        }

        return parity ? -1 : 1;
    };

    if (alpha_rotation) {
        sourcea.reserve(static_cast<size_t>(lena_));
        targeta.reserve(static_cast<size_t>(lena_));
        phasea.reserve(static_cast<size_t>(lena_));
        sourceb.reserve(static_cast<size_t>(lenb_));
        targetb.reserve(static_cast<size_t>(lenb_));
        phaseb.reserve(static_cast<size_t>(lenb_));

        for (int ia = 0; ia < lena_; ++ia) {
            const uint64_t alpha_source = astr_[ia];

            if ((alpha_source & source_mask) == 0 || (alpha_source & target_mask) != 0) {
                continue;
            }

            const uint64_t alpha_target =
                set_bit(unset_bit(alpha_source, source_orb), target_orb);
            const int ta = static_cast<int>(aind_.at(alpha_target));

            sourcea.push_back(ia);
            targeta.push_back(ta);
            phasea.push_back(1);
        }

        for (int ib = 0; ib < lenb_; ++ib) {
            const uint64_t beta = bstr_[ib];

            sourceb.push_back(ib);
            targetb.push_back(ib);
            phaseb.push_back(phase_for_alpha_rotation(beta));
        }
    } else {
        sourcea.reserve(static_cast<size_t>(lena_));
        targeta.reserve(static_cast<size_t>(lena_));
        phasea.reserve(static_cast<size_t>(lena_));
        sourceb.reserve(static_cast<size_t>(lenb_));
        targetb.reserve(static_cast<size_t>(lenb_));
        phaseb.reserve(static_cast<size_t>(lenb_));

        for (int ia = 0; ia < lena_; ++ia) {
            const uint64_t alpha = astr_[ia];

            sourcea.push_back(ia);
            targeta.push_back(ia);
            phasea.push_back(phase_for_beta_rotation(alpha));
        }

        for (int ib = 0; ib < lenb_; ++ib) {
            const uint64_t beta_source = bstr_[ib];

            if ((beta_source & source_mask) == 0 || (beta_source & target_mask) != 0) {
                continue;
            }

            const uint64_t beta_target =
                set_bit(unset_bit(beta_source, source_orb), target_orb);
            const int tb = static_cast<int>(bind_.at(beta_target));

            sourceb.push_back(ib);
            targetb.push_back(tb);
            phaseb.push_back(1);
        }
    }

    timer_.acc_record("make_givens_matching_each");

    return std::make_tuple(
                    sourcea,
                    targeta,
                    phasea,
                    sourceb,
                    targetb,
                    phaseb);
}

/// NICK: 1. Consider a faster blas veriosn, 2. consider using qubit basis, 3. rename (too long)
std::vector<uint64_t> FCIGraph::get_lex_bitstrings(int nele, int norb) {

    if (nele > norb) {
        throw std::invalid_argument("can't have more electorns that orbitals");
    }
        
    std::vector<uint64_t> bitstrings;

    // vector of [0,1,2,3,...]
    std::vector<uint64_t> indices(norb);
    for (int i = 0; i < norb; ++i)
        indices[i] = i;

    // vector that is a bitstring of zeros [0,0,0,....]
    std::vector<bool> bitstring(norb, false);
    
    // make hf bitsring [1,1,1,1,0,0,....]
    // esentially state is a bitstring (as a uint_64 for all possible permutations)
    for (int i = 0; i < nele; ++i)
        bitstring[i] = true;

    do {
        uint64_t state = 0;
        // loop over orbital states in bitstring, if there is a particle in that postiong
        // modify state
        for (int i = 0; i < norb; ++i) {
            if (bitstring[i]) { state |= (static_cast<uint64_t>(1) << i);}
        }
        
        bitstrings.push_back(state);
    // use std::prev_permutation to rearrange bitstring into the previous lexicographically ordered permutation.
    /// NICK: std::prev_permutation rearranges the elements in the range [first, last) into the previous lexicographical permutation.
    // It returns true if such a permutation exists (i.e., the sequence was not already in the smallest possible order).
    // It returns false when the sequence reaches its first permutation, and no further previous permutation exists.
    } while (std::prev_permutation(bitstring.begin(), bitstring.end()));

    // sort the bitstrings 
    std::sort(bitstrings.begin(), bitstrings.end());

    return bitstrings;

}

/// NICK: Seems slow..., may want to use qubit basis, convert to size_t maybe??
uint64_t FCIGraph::build_string_address(
    int nele, 
    int norb, 
    uint64_t occ,
    const std::vector<std::vector<uint64_t>>& zmat) 
{

    std::vector<int> occupations;

    for (int i = 0; i < 64; ++i) { // Assuming uint64_t is 64 bits
        if (occ & (1ULL << i)) { occupations.push_back(i); }
    }

    uint64_t address = 0;
    for (int i = 0; i < nele; ++i) {
        address += zmat[i][occupations[i]];
    }

    return address;
}

/// NICK: May want to make faster using blas calls if it becomes a bottleneck
std::vector<std::vector<uint64_t>> FCIGraph::get_z_matrix(int norb, int nele) {
    // Initialize Z matrix with zeros
    std::vector<std::vector<uint64_t>> Z(nele, std::vector<uint64_t>(norb, 0)); 

    if (nele == 0 || norb == 0) { return Z; }

    for (int k = 1; k < nele; ++k) {
        for (int ll = k; ll < norb - nele + k + 1; ++ll) {
            Z[k - 1][ll - 1] = 0;
            for (int m = norb - ll + 1; m < norb - k + 1; ++m) {
                Z[k - 1][ll - 1] += binom(m, nele - k) - binom(m - 1, nele - k - 1);
            }
        }
    }

    int k = nele;
    for (int ll = nele; ll < norb + 1; ++ll) {
        Z[k - 1][ll - 1] = static_cast<uint64_t>(ll - nele);
    }

    return Z;
}
