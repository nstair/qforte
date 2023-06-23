/// TODO(Nick): Remove unneeded includes

#include <map>
#include <random>
#include <algorithm>
#include <functional>
#include <stdexcept>

// #include <torch>

#include "torch/torch.h"

#include "fmt/format.h"

#include "qubit_basis.h"
#include "circuit.h"
#include "gate.h"
#include "helpers.h"
#include "qubit_operator.h"
#include "qubit_op_pool.h"
#include "timer.h"
#include "sparse_tensor.h"

#include "fermion_computer.h"

// Might need to uncomment?
// #if defined(_OPENMP)
// #include <omp.h>
// extern const bool parallelism_enabled = true;
// #else
// extern const bool parallelism_enabled = false;
// #endif


FermionComputer::FermionComputer(int nqubit, double print_threshold) : nqubit_(nqubit), print_threshold_(print_threshold) {
    nbasis_ = std::pow(2, nqubit_);
    basis_.assign(nbasis_, QubitBasis());
    coeff_.assign(nbasis_, 0.0);
    new_coeff_.assign(nbasis_, 0.0);
    for (size_t i = 0; i < nbasis_; i++) {
        basis_[i] = QubitBasis(i);
    }
    coeff_[0] = 1.;


}

std::complex<double> FermionComputer::coeff(const QubitBasis& basis) {
    return coeff_[basis.add()];
}

void FermionComputer::set_state(std::vector<std::pair<QubitBasis, double_c>> state) {
    std::fill(coeff_.begin(), coeff_.end(), 0.0);
    for (const auto& basis_c : state) {
        coeff_[basis_c.first.add()] = basis_c.second;
    }
}

void FermionComputer::zero_state() { std::fill(coeff_.begin(), coeff_.end(), 0.0); }

