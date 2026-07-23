#include "fmt/format.h"

#include "gate.h"
#include "qubit_basis.h"
#include "sparse_tensor.h"

const std::vector<std::pair<size_t, size_t>> Gate::two_qubits_basis_{
    {0, 0}, {0, 1}, {1, 0}, {1, 1}};
const std::vector<size_t> Gate::index1{0, 1};
const std::vector<size_t> Gate::index2{0, 1, 2, 3};

Gate::Gate(const std::string& label, size_t target, size_t control,
                         std::complex<double> gate[4][4])
    : label_(label),
      target_(target),
      control_(control),
      qubits_(target == control ? std::vector<size_t>{target}
                                : std::vector<size_t>{control, target}),
      gate_{},
      gate4_{} {
    for (const auto& i : index2) {
        for (const auto& j : index2) {
            gate_[i][j] = gate[i][j];
        }
    }
}

Gate::Gate(const std::string& label, const std::vector<size_t>& qubits,
                         std::complex<double> gate[16][16])
    : label_(label),
      target_(qubits.at(0)),
      control_(qubits.at(0)),
      qubits_(qubits),
      gate_{},
      gate4_{} {
    for (size_t i = 0; i < 16; ++i) {
        for (size_t j = 0; j < 16; ++j) {
            gate4_[i][j] = gate[i][j];
        }
    }
}

size_t Gate::target() const { return target_; }

size_t Gate::control() const { return control_; }

const complex_4_4_mat& Gate::gate() const { return gate_; }

const complex_16_16_mat& Gate::gate4() const { return gate4_; }

const std::vector<size_t>& Gate::qubits() const { return qubits_; }

const SparseMatrix Gate::sparse_matrix(size_t nqubit) const {
    size_t nbasis = std::pow(2, nqubit);
    if(nqubits() != 1){
        throw ("Gate must be a Pauli to convert to matrix!");
    }
    else if(target_ >= nqubit){
        throw ("Target index is too large for specified nqbits!");
    }

    SparseMatrix Spmat = SparseMatrix();

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            auto op_i_j = gate_[i][j];
            if (std::abs(op_i_j) > 1.0e-16) {
                for(size_t I = 0; I < nbasis; I++) {
                    QubitBasis basis_I = QubitBasis(I);
                    if (basis_I.get_bit(target_) == j) {
                        QubitBasis basis_J = basis_I;
                        basis_J.set_bit(target_, i);
                        Spmat.set_element(basis_J.add(), basis_I.add(), op_i_j);
                    }
                }
            }
        }
    }
    return Spmat;
}


std::string Gate::gate_id() const { return label_; }

std::string Gate::str() const {
    if (nqubits() == 1) {
        return fmt::format("{}{}", label_, target_);
    }
    if (nqubits() == 4) {
        return fmt::format("{}{}_{}_{}_{}", label_, qubits_[0], qubits_[1],
                           qubits_[2], qubits_[3]);
    }
    return fmt::format("{}{}_{}", label_, target_, control_);
}

std::string Gate::repr() const {
    std::string s;
    if (nqubits() == 4) {
        s = fmt::format("{} gate, qubits:{}, {}, {}, {}\n", label_, qubits_[0],
                        qubits_[1], qubits_[2], qubits_[3]);
        for (size_t i = 0; i < 16; ++i) {
            for (size_t j = 0; j < 16; ++j) {
                s += fmt::format("  {:+f} {:+f} i", std::real(gate4_[i][j]),
                                 std::imag(gate4_[i][j]));
            }
            s += '\n';
        }
        return s;
    }

    s = fmt::format("{} gate, target qubit:{}, contol qubit:{}\n", label_, target_, control_);
    const std::vector<size_t>& index = (nqubits() == 1 ? index1 : index2);
    for (const auto& i : index) {
        for (const auto& j : index) {
            s += fmt::format("  {:+f} {:+f} i", std::real(gate_[i][j]), std::imag(gate_[i][j]));
        }
        s += '\n';
    }
    return s;
}

size_t Gate::nqubits() const { return qubits_.size(); }

Gate Gate::adjoint() const {
    if (nqubits() == 4) {
        std::complex<double> adj_gate[16][16];
        bool self_adjoint = true;
        for (size_t i = 0; i < 16; ++i) {
            for (size_t j = 0; j < 16; ++j) {
                adj_gate[j][i] = std::conj(gate4_[i][j]);
                if (std::norm(adj_gate[j][i] - gate4_[i][j]) > 1.0e-12) {
                    self_adjoint = false;
                }
            }
        }
        if (not self_adjoint) {
            return Gate("adj(" + label_ + ")", qubits_, adj_gate);
        }
        return Gate(label_, qubits_, adj_gate);
    }

    std::complex<double> adj_gate[4][4];
    bool self_adjoint = true;
    for (const auto& i : index2) {
        for (const auto& j : index2) {
            adj_gate[j][i] = std::conj(gate_[i][j]);
            if (std::norm(adj_gate[j][i] - gate_[i][j]) > 1.0e-12) {
                self_adjoint = false;
            }
        }
    }
    if (not self_adjoint) {
        return Gate("adj(" + label_ + ")", target_, control_, adj_gate);
    }
    return Gate(label_, target_, control_, adj_gate);
}

const std::vector<std::pair<size_t, size_t>>& Gate::two_qubits_basis() {
    return two_qubits_basis_;
}
