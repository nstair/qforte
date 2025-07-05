#include <map>
#include <random>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <iterator>

#include "qubit_basis.h"
#include "circuit.h"
#include "gate.h"
#include "helpers.h"
#include "qubit_operator.h"
#include "tensor.h"
#include "tensor_operator.h"
#include "qubit_op_pool.h"
#include "sq_op_pool.h"
#include "timer.h"
#include "sq_operator.h"
#include "blas_math.h"
#include "cuda_runtime.h"

#include "fci_computer_thrust.h"
#include "fci_graph.h"

FCIComputerThrust::FCIComputerThrust(int nel, int sz, int norb, bool on_gpu) : 
    nel_(nel), 
    sz_(sz),
    norb_(norb),
    on_gpu_(on_gpu) {

    if (nel_ < 0) {
        throw std::invalid_argument("Cannot have negative electrons");
    }
    if (nel_ < std::abs(static_cast<double>(sz_))) {
        throw std::invalid_argument("Spin quantum number exceeds physical limits");
    }
    if ((nel_ + sz_) % 2 != 0) {
        throw std::invalid_argument("Parity of spin quantum number and number of electrons is incompatible");
    }

    nalfa_el_ = (nel_ + sz_) / 2;
    nbeta_el_ = nel_ - nalfa_el_;

    nalfa_strs_ = 1;
    for (int i = 1; i <= nalfa_el_; ++i) {
        nalfa_strs_ *= norb_ - i + 1;
        nalfa_strs_ /= i;
    }

    if (nalfa_el_ < 0 || nalfa_el_ > norb_) {
        nalfa_strs_ = 0;
    }

    nbeta_strs_ = 1;
    for (int i = 1; i <= nbeta_el_; ++i) {
        nbeta_strs_ *= norb_ - i + 1;
        nbeta_strs_ /= i;
    }

    if (nbeta_el_ < 0 || nbeta_el_ > norb_) {
        nbeta_strs_ = 0;
    }

    C_.zero_with_shape({nalfa_strs_, nbeta_strs_}, on_gpu_);
    C_.set_name("FCI Computer");

    graph_ = FCIGraph(nalfa_el_, nbeta_el_, norb_);

    timer_ = local_timer();
}

/// Set a particular element of the tensor stored in FCIComputerThrust, specified by idxs
void FCIComputerThrust::set_element(
    const std::vector<size_t>& idxs,
    const std::complex<double> val
        )
{
    C_.set(idxs, val);
}

void FCIComputerThrust::gpu_error() const {

    if (not on_gpu_) {
        throw std::runtime_error("Data not on GPU for FCIComputerThrust" + name_);
    }

}

void FCIComputerThrust::cpu_error() const {

    if (on_gpu_) {
        throw std::runtime_error("Data not on CPU for FCIComputerThrust" + name_);
    }

}

void FCIComputerThrust::to_gpu()
{
    cpu_error();
    C_.to_gpu();
    on_gpu_ = 1;
}

// change to 'to_cpu'
void FCIComputerThrust::to_cpu()
{
    gpu_error();
    C_.to_cpu();
    on_gpu_ = 0;
}

/// apply a TensorOperator to the current state 
// void apply_tensor_operator(const TensorOperator& top);

/// apply a Tensor represending a 1-body spin-orbital indexed operator to the current state 
void FCIComputerThrust::apply_tensor_spin_1bdy(const TensorThrust& h1e, size_t norb) {

    if(h1e.size() != (norb * 2) * (norb * 2)){
        throw std::invalid_argument("Expecting h1e to be nso x nso for apply_tensor_spin_1bdy");
    }

    TensorThrust Cnew({nalfa_strs_, nbeta_strs_}, "Cnew");

    TensorThrust h1e_blk1 = h1e.slice(
        {
            std::make_pair(0, norb_), 
            std::make_pair(0, norb_)
            }
        );

    TensorThrust h1e_blk2 = h1e.slice(
        {
            std::make_pair(norb_, 2*norb_), 
            std::make_pair(norb_, 2*norb_)
            }
        );

    apply_array_1bdy(
        Cnew,
        graph_.read_dexca_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexca(),
        h1e_blk1,
        norb_,
        true);

    apply_array_1bdy(
        Cnew,
        graph_.read_dexcb_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexcb(),
        h1e_blk2,
        norb_,
        false);

    C_ = Cnew;
}

/// apply TensorThrusts represending 1-body and 2-body spatial-orbital indexed operator to the current state 
void FCIComputerThrust::apply_tensor_spat_12bdy(
    const TensorThrust& h1e, 
    const TensorThrust& h2e, 
    const TensorThrust& h2e_einsum, 
    size_t norb) 
{
    if(h1e.size() != norb * norb){
        throw std::invalid_argument("Expecting h1e to be norb x norb for apply_tensor_spat_12bdy");
    }

    if(h2e.size() != norb * norb * norb * norb){
        throw std::invalid_argument("Expecting h2e to be norb x norb x norb x norb for apply_tensor_spat_12bdy");
    }

    if(h2e_einsum.size() != norb * norb * norb * norb){
        throw std::invalid_argument("Expecting h2e_einsum to be norb x norb x norb x norb for apply_tensor_spat_12bdy");
    }

    TensorThrust Cnew({nalfa_strs_, nbeta_strs_}, "Cnew");

    // Apply one-body terms
    apply_array_1bdy(
        Cnew,
        graph_.read_dexca_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexca(),
        h1e,
        norb_,
        true);

    apply_array_1bdy(
        Cnew,
        graph_.read_dexcb_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexcb(),
        h1e,
        norb_,
        false);

    // Apply two-body terms (same spin)
    lm_apply_array12_same_spin_opt(
        Cnew,
        graph_.read_dexca_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexca(),
        h1e,
        h2e,
        norb_,
        true);

    lm_apply_array12_same_spin_opt(
        Cnew,
        graph_.read_dexcb_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexcb(),
        h1e,
        h2e,
        norb_,
        false);

    // Apply two-body terms (different spin)
    lm_apply_array12_diff_spin_opt(
        Cnew,
        graph_.read_dexca_vec(),
        graph_.read_dexcb_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexca(),
        graph_.get_ndexcb(),
        h2e_einsum,
        norb_);

    C_ = Cnew;
}

/// apply TensorThrusts represending 1-body and 2-body spatial-orbital indexed operator
/// as well as a constant to the current state 
void FCIComputerThrust::apply_tensor_spat_012bdy(
    const std::complex<double> h0e,
    const TensorThrust& h1e, 
    const TensorThrust& h2e, 
    const TensorThrust& h2e_einsum, 
    size_t norb) 
{
    TensorThrust Cold = C_;
    
    apply_tensor_spat_12bdy(
        h1e,
        h2e,
        h2e_einsum,
        norb);

    C_.zaxpy(
        Cold,
        h0e,
        1,
        1    
    );
}

/// Set a particular element of this TensorThrust, specified by idxs
void FCIComputerThrust::add_to_element(
    const std::vector<size_t>& idxs,
    const std::complex<double> val
        )
{
    C_.add_to_element(idxs, val);
}

void FCIComputerThrust::apply_tensor_operator(const TensorOperator& top)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("apply_tensor_operator not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::apply_tensor_spin_12bdy(
    const TensorThrust& h1e, 
    const TensorThrust& h2e, 
    size_t norb)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("apply_tensor_spin_12bdy not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::lm_apply_array1(
    const TensorThrust& out,
    const std::vector<int> dexc,
    const int astates,
    const int bstates,
    const int ndexc,
    const TensorThrust& h1e,
    const int norbs,
    const bool is_alpha)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("lm_apply_array1 not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::apply_array_1bdy(
    TensorThrust& out,
    const std::vector<int>& dexc,
    const int astates,
    const int bstates,
    const int ndexc,
    const TensorThrust& h1e,
    const int norbs,
    const bool is_alpha)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("apply_array_1bdy not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::lm_apply_array12_same_spin_opt(
    TensorThrust& out,
    const std::vector<int>& dexc,
    const int alpha_states,
    const int beta_states,
    const int ndexc,
    const TensorThrust& h1e,
    const TensorThrust& h2e,
    const int norbs,
    const bool is_alpha)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("lm_apply_array12_same_spin_opt not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::lm_apply_array12_diff_spin_opt(
    TensorThrust& out,
    const std::vector<int>& adexc,
    const std::vector<int>& bdexc,
    const int alpha_states,
    const int beta_states,
    const int nadexc,
    const int nbdexc,
    const TensorThrust& h2e,
    const int norbs)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("lm_apply_array12_diff_spin_opt not yet implemented for FCIComputerThrust");
}

std::pair<TensorThrust, TensorThrust> FCIComputerThrust::calculate_dvec_spin_with_coeff()
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("calculate_dvec_spin_with_coeff not yet implemented for FCIComputerThrust");
}

TensorThrust FCIComputerThrust::calculate_coeff_spin_with_dvec(std::pair<TensorThrust, TensorThrust>& dvec)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("calculate_coeff_spin_with_dvec not yet implemented for FCIComputerThrust");
}

std::pair<std::vector<int>, std::vector<int>> FCIComputerThrust::evaluate_map_number(
    const std::vector<int>& numa,
    const std::vector<int>& numb)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("evaluate_map_number not yet implemented for FCIComputerThrust");
}

std::pair<std::vector<int>, std::vector<int>> FCIComputerThrust::evaluate_map(
    const std::vector<int>& crea,
    const std::vector<int>& anna,
    const std::vector<int>& creb,
    const std::vector<int>& annb)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("evaluate_map not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::apply_cos_inplace(
    const std::complex<double> time,
    const std::complex<double> coeff,
    const std::vector<int>& crea,
    const std::vector<int>& anna,
    const std::vector<int>& creb,
    const std::vector<int>& annb,
    TensorThrust& Cout)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("apply_cos_inplace not yet implemented for FCIComputerThrust");
}

int FCIComputerThrust::isolate_number_operators(
    const std::vector<int>& cre,
    const std::vector<int>& ann,
    std::vector<int>& crework,
    std::vector<int>& annwork,
    std::vector<int>& number)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("isolate_number_operators not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::evolve_individual_nbody_easy(
    const std::complex<double> time,
    const std::complex<double> coeff,
    const TensorThrust& Cin,
    TensorThrust& Cout,
    const std::vector<int>& crea,
    const std::vector<int>& anna,
    const std::vector<int>& creb,
    const std::vector<int>& annb)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("evolve_individual_nbody_easy not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::evolve_individual_nbody_hard(
    const std::complex<double> time,
    const std::complex<double> coeff,
    const TensorThrust& Cin,
    TensorThrust& Cout,
    const std::vector<int>& crea,
    const std::vector<int>& anna,
    const std::vector<int>& creb,
    const std::vector<int>& annb)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("evolve_individual_nbody_hard not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::evolve_individual_nbody(
    const std::complex<double> time,
    const SQOperator& sqop,
    const TensorThrust& Cin,
    TensorThrust& Cout,
    const bool antiherm,
    const bool adjoint)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("evolve_individual_nbody not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::apply_sqop_evolution(
    const std::complex<double> time,
    const SQOperator& sqop,
    const bool antiherm,
    const bool adjoint)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("apply_sqop_evolution not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::evolve_pool_trotter_basic(
    const SQOpPool& pool,
    const bool antiherm,
    const bool adjoint)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("evolve_pool_trotter_basic not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::evolve_pool_trotter(
    const SQOpPool& pool,
    const double evolution_time,
    const int trotter_steps,
    const int trotter_order,
    const bool antiherm,
    const bool adjoint)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("evolve_pool_trotter not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::evolve_op_taylor(
    const SQOperator& op,
    const double evolution_time,
    const double convergence_thresh,
    const int max_taylor_iter)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("evolve_op_taylor not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::apply_individual_nbody1_accumulate(
    const std::complex<double> coeff, 
    const TensorThrust& Cin,
    TensorThrust& Cout,
    std::vector<int>& targeta,
    std::vector<int>& sourcea,
    std::vector<int>& paritya,
    std::vector<int>& targetb,
    std::vector<int>& sourceb,
    std::vector<int>& parityb)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("apply_individual_nbody1_accumulate not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::apply_individual_nbody_accumulate(
    const std::complex<double> coeff,
    const TensorThrust& Cin,
    TensorThrust& Cout,
    const std::vector<int>& daga,
    const std::vector<int>& undaga, 
    const std::vector<int>& dagb,
    const std::vector<int>& undagb)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("apply_individual_nbody_accumulate not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::apply_individual_sqop_term(
    const std::tuple< std::complex<double>, std::vector<size_t>, std::vector<size_t>>& term,
    const TensorThrust& Cin,
    TensorThrust& Cout)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("apply_individual_sqop_term not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::apply_sqop(const SQOperator& sqop)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("apply_sqop not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::apply_diagonal_of_sqop(
    const SQOperator& sq_op, 
    const bool invert_coeff)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("apply_diagonal_of_sqop not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::apply_sqop_pool(const SQOpPool& sqop_pool)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("apply_sqop_pool not yet implemented for FCIComputerThrust");
}

std::complex<double> FCIComputerThrust::get_exp_val(const SQOperator& sqop)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("get_exp_val not yet implemented for FCIComputerThrust");
}

std::complex<double> FCIComputerThrust::get_exp_val_tensor(
    const std::complex<double> h0e, 
    const TensorThrust& h1e, 
    const TensorThrust& h2e, 
    const TensorThrust& h2e_einsum, 
    size_t norb)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("get_exp_val_tensor not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::scale(const std::complex<double> a)
{
    C_.scale(a);
}

std::vector<double> FCIComputerThrust::direct_expectation_value(const TensorOperator& top)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("direct_expectation_value not yet implemented for FCIComputerThrust");
}

std::complex<double> FCIComputerThrust::coeff(const QubitBasis& abasis, const QubitBasis& bbasis)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("coeff not yet implemented for FCIComputerThrust");
}

void FCIComputerThrust::set_state(const TensorThrust& other_state)
{
    cpu_error();
    other_state.cpu_error();
    C_.copy_in(other_state);
}

void FCIComputerThrust::set_state_gpu(const TensorThrust& other_state)
{
    gpu_error();
    other_state.gpu_error();
    C_.copy_in_gpu(other_state);
}

void FCIComputerThrust::set_state_from_tensor(const Tensor& other_state)
{
    cpu_error();
    C_.copy_in_from_tensor(other_state);
}

void FCIComputerThrust::zero()
{
    cpu_error();
    C_.zero();
}

void FCIComputerThrust::hartree_fock()
{
    cpu_error();
    C_.zero();
    C_.set({0, 0}, 1.0);
}

void FCIComputerThrust::print_vector(const std::vector<int>& vec, const std::string& name)
{
    std::cout << "\n" << name << ": ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << static_cast<int>(vec[i]);
        if (i < vec.size() - 1) {
           std::cout << ", "; 
        }
    }
    std::cout << std::endl;
}

void FCIComputerThrust::print_vector_uint(const std::vector<uint64_t>& vec, const std::string& name)
{
    std::cout << "\n" << name << ": ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) {
            std::cout << ", "; 
        }
    }
    std::cout << std::endl;
}
