#ifndef _fermion_computer_h_
#define _fermion_computer_h_

#include <string>
#include <vector>

#include "torch/torch.h"

#include "qforte-def.h" // double_c

template <class T> std::complex<T> complex_prod(std::complex<T> a, std::complex<T> b) {
    return std::conj<T>(a) * b;
}

template <class T> std::complex<T> add_c(std::complex<T> a, std::complex<T> b) { return a + b; }

class Gate;
class QubitBasis;
class Circuit;
class QubitOperator;
class QubitOpPool;
class SparseMatrix;

class FermionComputer {
  public:
    /// default constructor: create a quantum computer with nqubit qubits
    FermionComputer(int nqubit, double print_threshold = 1.0e-6);

    // set the coefficient vector directly from another coefficient vector
    void set_coeff_vec(const std::vector<double_c> c_vec) { coeff_ = c_vec; }

    /// set the FermionComputer to the state
    /// TODO(Nick): write a better description of the |IaIb> stuff

    void set_state(std::vector<std::pair<QubitBasis, double_c>> state);

    void zero_state();

    /// return the coefficient of a basis state
    std::complex<double> coeff(const QubitBasis& basis);

    /// get timings
    std::vector<std::pair<std::string, double>> get_timings() { return timings_; }

    /// clear the timings
    void clear_timings() { timings_.clear(); }

  private:
    /// the number of qubits
    size_t nqubit_;

    /// the number of basis states (2 ^ nqubit_)
    size_t nbasis_;

    /// the tensor product basis
    std::vector<QubitBasis> basis_;

    /// The coefficients of the starting state in the tensor product basis
    std::vector<std::complex<double>> coeff_;

    // the coefficeient tensor
    // torch::Tensor tensor = torch::zeros({2, 2});
    torch::Tensor tensor = torch::rand({2, 3});

    /// the coefficients of the ending state in the tensor product basis
    std::vector<std::complex<double>> new_coeff_;

    /// timings and descriptions accessable in python
    std::vector<std::pair<std::string, double>> timings_;

    /// print threshold for determinant coefficients    
    double print_threshold_;

    /// the threshold for doing operations with elements of gate matricies
    double compute_threshold_ = 1.0e-16;

};

#endif // _fermion_computer_h_
