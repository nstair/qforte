#ifndef _fci_graph_h_
#define _fci_graph_h_

#include <vector>
#include <unordered_map>
#include <tuple>
#include <cstdint>
#include <cstddef>

struct PairHash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

using Spinmap = std::unordered_map<std::pair<int, int>, std::vector<std::tuple<int, int, int>>, PairHash>;

class FCIGraph {
public:

    /// Constructor
    FCIGraph(int nalfa, int nbeta, int norb);

    FCIGraph();

    /// Build alfa/beta bitstrings to index the FCI Computer
    std::pair<std::vector<uint64_t>, std::unordered_map<uint64_t, size_t>> build_strings(
        int nele, 
        size_t length); 

    /// Construct the FCI Mapping
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
    Spinmap build_mapping(
        const std::vector<uint64_t>& strings, 
        int nele, 
        const std::unordered_map<uint64_t, size_t>& index);

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
    std::vector<std::vector<std::vector<int>>> map_to_deexc(
        const Spinmap& mappings, 
        int states,
        int norbs,
        int nele);

    std::vector<uint64_t> get_lex_bitstrings(int nele, int norb);

    uint64_t build_string_address(
        int nele, 
        int norb, 
        uint64_t occ,
        const std::vector<std::vector<uint64_t>>& zmat); 

    std::vector<std::vector<uint64_t>> get_z_matrix(int norb, int nele);

    std::tuple<int, std::vector<int>, std::vector<int>, std::vector<int>> make_mapping_each(
        bool alpha, 
        const std::vector<int>& dag, 
        const std::vector<int>& undag); 

    /// ==> Utility Functions for Bit Math (may need to move) <== ///

    /// Combinutorics helper funciton for binomial coefficients
    int binom(int n, int m) {
        if (m < 0 || m > n)
            return 0;

        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));
        for (int i = 0; i <= n; ++i)
            dp[i][0] = 1;

        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= std::min(i, m); ++j) {
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
            }
        }

        return dp[n][m];
    }

    std::vector<int> unroll_from_3d(
        const std::vector<std::vector<std::vector<int>>>& input) 
    {
        std::vector<int> output;
        
        for (const auto& vec2D : input) {
            for (const auto& vec1D : vec2D) {
                output.insert(output.end(), vec1D.begin(), vec1D.end());
            }
        }
        
        return output;
    }
    
    bool get_bit(uint64_t string, size_t pos) { return string & maskbit(pos); }

    constexpr uint64_t maskbit(size_t pos) { return static_cast<uint64_t>(1) << pos; }

    int count_bits_above(uint64_t string, int pos) {
        uint64_t bitmask = (1 << (pos + 1)) - 1;
        uint64_t inverted_mask = ~bitmask;
        uint64_t result = string & inverted_mask;
        int count = 0;
        while (result) {
            result &= (result - 1);
            count++;
        }
        return count;
    }

    int count_bits(uint64_t string) {
        int count = 0;
        while (string) {
            string &= (string - 1);
            count++;
        }
        return count;
    }

    std::vector<int> get_positions(uint64_t string, int nbits) {
        std::vector<int> positions;
        positions.reserve(nbits); // Reserve space for nbits positions

        int pos = 0;
        while (nbits > 0) {
            if (string & 1) {
                positions.push_back(pos);
                --nbits;
            }
            string >>= 1;
            ++pos;
        }

        return positions;
    }

    int count_bits_between(uint64_t string, int pos1, int pos2) {

        uint64_t mask = (((1 << pos1) - 1) ^ ((1 << (pos2 + 1)) - 1)) \
         & (((1 << pos2) - 1) ^ ((1 << (pos1 + 1)) - 1));

        uint64_t masked_string = string & mask;

        int count = 0;
        while (masked_string > 0) {
            count += masked_string & 1;
            masked_string >>= 1;
        }

        return count;
    }

    uint64_t reverse_integer_index(const std::vector<int>& occupations){
        uint64_t string = 0;
        for(int pos : occupations){ string = set_bit(string, pos); }
        return string;
    }

    uint64_t set_bit(uint64_t string, int pos) {
        return string | (1ULL << pos);
    }

    uint64_t unset_bit(uint64_t string, int pos) {
        return string & ~(1ULL << pos);
    }

    /// ==> Setters and Getters <== /// 

    /// return the number of alfa/beta electrons
    size_t get_nalfa() const { return nalfa_; }
    size_t get_nbeta() const { return nbeta_; }

    /// return the number of alfa/beta strings
    size_t get_lena() const { return lena_; }
    size_t get_lenb() const { return lenb_; }

    /// return the alfa/beta bitstrings
    const std::vector<uint64_t>& get_astr() const { return astr_;  }
    const std::vector<uint64_t>& get_bstr() const { return bstr_;  }

    /// return the alfa/beta bitstrings
    int get_astr_at_idx(int idx) const { return static_cast<int>(astr_[idx]);  }
    int get_bstr_at_idx(int idx) const { return static_cast<int>(bstr_[idx]);  }

    int get_aind_for_str(int str) const { return static_cast<int>(aind_.at(static_cast<uint64_t>(str)));  }
    int get_bind_for_str(int str) const { return static_cast<int>(bind_.at(static_cast<uint64_t>(str)));  }

    std::unordered_map<uint64_t, size_t> get_aind() const { return aind_; }
    std::unordered_map<uint64_t, size_t> get_bind() const { return bind_; }

    Spinmap get_alfa_map() const { return alfa_map_; }
    Spinmap get_beta_map() const { return beta_map_; }

    int get_ndexca() const { return dexca_[0].size(); }
    int get_ndexcb() const { return dexcb_[0].size(); }

    std::vector<std::vector<std::vector<int>>> get_dexca() const { return dexca_; }
    std::vector<std::vector<std::vector<int>>> get_dexcb() const { return dexcb_; }

    std::vector<int> get_dexca_vec() const { return dexca_vec_; }
    std::vector<int> get_dexcb_vec() const { return dexcb_vec_; }

    const std::vector<int>& read_dexca_vec() const { return dexca_vec_; }
    const std::vector<int>& read_dexcb_vec() const { return dexcb_vec_; }

private:
    int nalfa_;
    int nbeta_;
    int norb_;
    int lena_;
    int lenb_;

    std::vector<uint64_t> astr_;
    std::vector<uint64_t> bstr_;

    std::unordered_map<uint64_t, size_t> aind_;
    std::unordered_map<uint64_t, size_t> bind_;

    Spinmap alfa_map_;
    Spinmap beta_map_;

    std::vector<std::vector<std::vector<int>>> dexca_;
    std::vector<std::vector<std::vector<int>>> dexcb_;

    std::vector<int> dexca_vec_;
    std::vector<int> dexcb_vec_;
    
};

#endif
