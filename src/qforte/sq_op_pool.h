#ifndef _sq_op_pool_h_
#define _sq_op_pool_h_

#include <complex>
#include <string>
#include <vector>

class SQOperator;
class QubitOperator;
class QubitOpPool;
class DFHamiltonian;
class Tensor;
class FCIComputer;

// Represents an arbitrary linear combination of second quantized operators.
// May also represent an array of second quantized operators by ignoring
// the coefficients.
class SQOpPool {
  public:
    /// default constructor: creates an empty second quantized operator pool
    SQOpPool() {}

    /// add one set of annihilators and/or creators to the second quantized operator pool
    void add_term(std::complex<double> coeff, const SQOperator& sq_op );

    /// sets the operator pool coefficients
    void set_coeffs(const std::vector<std::complex<double>>& new_coeffs);

    /// sets the operator pool coefficients
    void set_coeffs_to_scaler(std::complex<double> new_coeff);

    /// return a vector of terms and their coeficients
    const std::vector<std::pair< std::complex<double>, SQOperator>>& terms() const;

    /// set the total number of occupied and virtual spatial orbitals from a reference, from the number
    ///     of occupied spin orbitals of each point group symmetry
    void set_orb_spaces(const std::vector<int>& ref);

    /// onous on caller to pass a sq_op that is actually hermitain, should use a hermitian check funciton...
    /// for an operator, splits the operator into hermitan pairs where each pair becomes a term
    /// in the pool vector
    void add_hermitian_pairs(std::complex<double> coeff, const SQOperator& sq_op );

    /// A function that will construct an operator pool that connects (by excitaiton/de-excitation operators) all determinants
    /// in reference to all relevant determinants (based on the cumulative threshold) in residual.
    void add_connection_pairs(
      const FCIComputer& residual, 
      const FCIComputer& reference,
      const double threshold);

    /// returns a QubitOpPool object with one term for each term in terms_
    QubitOpPool get_qubit_op_pool();

    /// returns a single QubitOperator of the JW transformed sq ops
    QubitOperator get_qubit_operator(const std::string& order_type, bool combine_like_terms=true, bool qubit_excitations=false);

    /// builds the sq operator pool
    void fill_pool(std::string pool_type);

    /// builds the sq operator pool using kmax repeats of the 
    /// disentangled unitary paired coupled cluster singleds and
    /// doubles ansatz.
    void fill_pool_kUpCCGSD(int kmax);

    /// builds the sq operator pool based on the operators in the 
    /// second quantized hamiltonain. Similar to add_hermitian pairs
    /// but intitalizees to zero 
    void fill_pool_sq_hva(std::complex<double> coeff, const SQOperator& sq_op);

    /// builds the sq operator pool that, when trotterized, 
    /// reporduces the (trotterized) evolution uder a
    /// DFHamiltonian. 
    void fill_pool_df_trotter(const DFHamiltonian& df_ham, const std::complex<double> coeff);

    /// Append SQ operatrs to the pool corresponding to a Givens rotation defined 
    /// by the matrix U. Used primarily in fill_pool_df_trotter.
    void append_givens_ops_sector(
      const Tensor& U, 
      const std::complex<double> coeff, 
      const bool is_alfa);

    /// Append alpha / beta mixed SQ operatrs to the pool corresponding 
    /// to an exponentiated diagonal operator defined by V.
    /// Used primarily in fill_pool_df_trotter.
    void append_diagonal_ops_all(
      const Tensor& V, 
      const std::complex<double> coeff);

    /// return the number paulit products needed for each term
    std::vector<int> get_count_pauli_terms_ex_deex() const;

    /// return a vector of string representing this sq operator pool
    std::string str() const;

  private:
    /// the number of occupied spatial orbitals
    int nocc_;

    /// the number of virtual spatial orbitals
    int nvir_;

    /// the list of sq operators in the pool
    std::vector<std::pair<std::complex<double>, SQOperator>> terms_;

};

#endif // _sq_op_pool_h_
