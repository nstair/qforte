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


/**
 * @brief Builds a mapping from spin orbital pairs to possible state transitions for single-electron excitations.
 *
 * This function constructs a `Spinmap`, which is an unordered map that associates each pair of orbitals `(iorb, jorb)`
 * with a vector of tuples. Each tuple represents a possible transition between quantum states due to an electron
 * being excited from orbital `jorb` to orbital `iorb`. The tuples contain:
 * - The index of the source state (`index.at(string)`).
 * - The index of the target state after the excitation (`index.at(unset_bit(set_bit(string, iorb), jorb))`).
 * - The sign (`+1` or `-1`) determined by the parity of the number of electrons between `iorb` and `jorb`.
 *
 * The function iterates over all possible pairs of orbitals and all provided occupation bitstrings (`strings`),
 * identifying valid excitations where orbital `jorb` is occupied and orbital `iorb` is unoccupied.
 * It calculates the parity to account for the antisymmetric nature of fermionic wavefunctions.
 *
 * @param strings A vector of 64-bit integers representing the occupation number bitstrings of spin configurations.
 * @param nele The number of electrons (not directly used in this function but may provide contextual relevance).
 * @param index An unordered map that maps each bitstring to its corresponding index in the list of configurations.
 * @return A `Spinmap` mapping each orbital pair `(iorb, jorb)` to a vector of tuples `(source_index, target_index, sign)`,
 *         representing all valid single-electron excitations between the orbitals with the associated parity.
 */
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


/**
 * @brief Constructs a mapping from target states to possible de-excitations leading to them.
 *
 * This function processes the provided `Spinmap`, which contains possible single-electron excitations
 * represented as transitions from source states to target states along with parity signs.
 *
 * It creates a nested vector `dexc` where each element corresponds to a target state and contains
 * a list of de-excitation transitions that result in that target state. Each de-excitation is represented
 * by a vector of three integers:
 * - The index of the source state (`state`).
 * - The combined orbital index (`idx`), computed as `i * norbs + j`, representing an electron
 *   being annihilated in orbital `j` and created in orbital `i`.
 * - The parity (`parity`), accounting for the fermionic antisymmetry sign.
 *
 * The structure `dexc` is organized for efficient access, enabling quick lookup of all transitions
 * leading to each target state during computations such as applying the Hamiltonian in FCI simulations.
 *
 * @param mappings A `Spinmap` mapping orbital pairs `(i, j)` to vectors of transitions `(source, target, parity)`.
 * @param states The total number of states (configurations).
 * @param norbs The total number of orbitals.
 * @param nele The number of electrons.
 * @return A nested vector `dexc` of size `states`, where `dexc[target]` contains a list of de-excitation transitions
 *         leading to the target state, with each transition represented by `[source_state, orbital_index, parity]`.
 */
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
    
    // Spinmap key -> std::pair<int, int> => (creator_idx, anihilator_idx)?
    // Spinmap val -> std::vector<std::tuple<int, int, int>> => (source, target, pairity)

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
            
            dexc[target][index[target]][0] = state;   // state index
            dexc[target][index[target]][1] = idx;     // ij shift (matrix index)
            dexc[target][index[target]][2] = parity;  // pairity for i->j transition
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

    return std::make_tuple(
                    count,
                    source,
                    target,
                    parity);
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
