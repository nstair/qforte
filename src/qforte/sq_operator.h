#ifndef _sq_operator_h_
#define _sq_operator_h_

#include <complex>
#include <string>
#include <vector>
#include <numeric>
#include <map>

class Gate;

class QubitOperator;

class SQOperator {
    /* A SQOperator is a linear combination (over C) of vaccuum-normal products of fermionic
     * second quantized operators.
     * The significance of the linear combination is context-dependent, but it should refer
     * to a "basis combination" in some sense, i.e., antihermitian combination for UCC,
     * spin-adapted combination for closed-shell systems, a single operator for traditional CC.
     *
     * All storage, printing, and input of summands in the combination takes tuples of the following form:
     * (1) coefficient
     * (2) vector of orbital-indices of creation operators
     * (3) vector of orbital-indices of annihilation operators,
     * All orbital indices start at zero.
     * Index vectors are lexicographic, i.e., std::tuple<1, {p, q}, {s, r}> means 1 * p^ q^ s r.
     */
  public:
    /// default constructor: creates an empty second quantized operator
    SQOperator() {}

    /// add one product of annihilators and/or creators to this second quantized operator
    /// Input is required in the same format as storage. See terms_ for details.
    void add_term(std::complex<double> coeff, const std::vector<size_t>& cre_ops, const std::vector<size_t>& ann_ops);

    /// add an second quantized operator to the second quantized operator
    void add_op(const SQOperator& sqo);

    /// sets the operator coefficients
    void set_coeffs(const std::vector<std::complex<double>>& new_coeffs);

    /// multiplies the sq operator coefficients by multiplier
    void mult_coeffs(const std::complex<double>& multiplier);

    /// return a vector of terms and their coefficients
    const std::vector<std::tuple< std::complex<double>, std::vector<size_t>, std::vector<size_t>>>& terms() const;

    /// return the largest alpha and beta orbital indicies based on current spin orbitlts 
    /// present in the sq operator
    /// returns -1 if there are no operators present.
    std::pair<int, int> get_largest_alfa_beta_indices() const;

    /// return the largerst n-body order of the sq operator (same as largerst rank)
    int many_body_order() const;

    /// returns a list of ranks present in this SQOperator
    std::vector<int> ranks_present() const;

    /// Put a single term into "canonical" form. Canonical form orders orbital indices
    /// descending.
    void canonical_order_single_term(std::tuple< std::complex<double>, std::vector<size_t>, std::vector<size_t>>& term );

    /// Canonicalize each term. The order of the terms is unaffected.
    void canonical_order();

    /// Combine like terms in terms_. As a side-effect, canonicalizes the order.
    void simplify();

    // Returns the number of unique Pauli operator products resulting from an SQOperator.
    // If the second operator is provided, it computes products between both A and B.
    // If B is nullptr, it computes products for A only.
    size_t count_unique_pauli_products(const SQOperator* B = nullptr) const;

    // Returns the number of CNOT gates required to implement the exponential
    // of a two-term anti-Hermitian SQOperator K of the form:
    //   K = i (g + g^)   or   K = g - g^
    // The circuit implementation is based on a standard decomposition for multi-qubit
    // Pauli rotations assuming linear connectivity, where the number of CNOTs is
    // 2*(r-1), with r being the size of the contiguous qubit support.
    int count_cnot_for_exponential();

    /**
     * @brief Count the number of Pauli‐product terms resulting from the Jordan–Wigner
     *        transform of this two‐term anti‐Hermitian operator K = g – g† or i(g + g†).
     *
     * For a k‐body excitation g (with k creation and k annihilation operators), the
     * JW expansion of g±g† yields exactly 2^(2k-1) Pauli strings.
     *
     * @throws std::invalid_argument if this operator does not consist of exactly two terms
     *         or if the creation/annihilation ranks mismatch.
     * @return The number of distinct Pauli‐product strings in the JW mapping.
     */
    int count_pauli_terms_ex_deex() const;

    /// Return the QubitOperator object corresponding the the Jordan-Wigner
    /// transform of this sq operator. Calls simplify as a side-effect.
    /// If qubit_excitation = true, replace fermionic creation/annihilation
    /// operators by qubit ones.
    /// WARNING: In the current implementation of qubit excitations,
    /// the 1-to-1 mapping between second-quantized operators and their
    /// qubit excitation counterparts is ensured by the normal ordering of
    /// second-quantized operators
    QubitOperator jw_transform(bool qubit_excitation = false);

    std::vector<SQOperator> split_by_rank(bool);

    /// return a vector of string representing this quantum operator
    std::string str() const;

  private:
    /// The linear combination of second quantized operators. Stored as a tuple of
    std::vector<std::tuple< std::complex<double>, std::vector<size_t>, std::vector<size_t>>> terms_;

    /// Calculate the parity of permutation p
    bool permutation_phase(std::vector<int> p) const;

    int canonicalize_helper(std::vector<size_t>& op_list) const;

    /// If operators is a vector of orbital indices, add the corresponding creator
    /// or annihilation qubit operators to holder.
    void jw_helper(QubitOperator& holder, const std::vector<size_t>& operators, bool creator, bool qubit_excitation) const;
};

#endif // _sq_operator_h_
