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

#include "fci_computer_gpu_kernels.cuh"

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
    // local_timer my_timer = local_timer();
    timer_.reset();
    if ((targetb.size() != sourceb.size()) or (sourceb.size() != parityb.size())) {
        throw std::runtime_error("The sizes of btarget, bsource, and bparity must be the same.");
    }

    if ((targeta.size() != sourcea.size()) or (sourcea.size() != paritya.size())) {
        throw std::runtime_error("The sizes of atarget, asource, and aparity must be the same.");
    }
    // only part that has kernel

    std::vector<cuDoubleComplex> paritya_complex(paritya.size());
    std::vector<cuDoubleComplex> parityb_complex(parityb.size());

    for (size_t i = 0; i < paritya.size(); ++i) {
        paritya_complex[i] = make_cuDoubleComplex(paritya[i], 0.0);
    }

    for (size_t i = 0; i < parityb.size(); ++i) {
        parityb_complex[i] = make_cuDoubleComplex(parityb[i], 0.0);
    }

    // make device pointers out of all the things coming in - use cuda mem copy to a device pointer
    // my_timer.record("error checks");
    // my_timer.reset();
    int* d_sourcea;
    int* d_sourceb;
    int* d_targeta;
    int* d_targetb;
    // int* d_paritya;
    // int* d_parityb;

    cuDoubleComplex* d_paritya;
    cuDoubleComplex* d_parityb;

    // cuDoubleComplex* d_Cin;
    // cuDoubleComplex* d_Cout;

    // cumalloc for these

    int sourcea_mem = sourcea.size() * sizeof(int);
    int sourceb_mem = sourceb.size() * sizeof(int);
    int targetb_mem = targetb.size() * sizeof(int);
    int targeta_mem = targeta.size() * sizeof(int);
    // int paritya_mem = paritya.size() * sizeof(int);
    // int parityb_mem = parityb.size() * sizeof(int);

    int paritya_mem = paritya.size() * sizeof(cuDoubleComplex);
    int parityb_mem = parityb.size() * sizeof(cuDoubleComplex);

    int tensor_mem = Cin.size() * sizeof(cuDoubleComplex);

    timer_.acc_record("initialization in nbody1_acc");
    timer_.reset();

    // my_timer.record("making pointers");
    // my_timer.reset();
    cudaMalloc(&d_sourcea, sourcea_mem);
    cudaMalloc(&d_sourceb, sourceb_mem);
    cudaMalloc(&d_targeta, targeta_mem);
    cudaMalloc(&d_targetb, targetb_mem);

    cudaMalloc(&d_paritya, paritya_mem);
    cudaMalloc(&d_parityb, parityb_mem);

    timer_.acc_record("cuda malloc in nbody1_acc");
    timer_.reset();

    // cudaMalloc(&d_Cin,  tensor_mem);
    // cudaMalloc(&d_Cout, tensor_mem);

    // my_timer.record("cudamallocs");
    // my_timer.reset();

    cudaMemcpy(d_sourcea, sourcea.data(), sourcea_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sourceb, sourceb.data(), sourceb_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(d_targeta, targeta.data(), targeta_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(d_targetb, targetb.data(), targetb_mem, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_paritya, paritya.data(), paritya_mem, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_parityb, parityb.data(), parityb_mem, cudaMemcpyHostToDevice);

    cudaMemcpy(d_paritya, paritya_complex.data(), paritya_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(d_parityb, parityb_complex.data(), parityb_mem, cudaMemcpyHostToDevice);

    timer_.acc_record("cuda memcpy in nbody1_acc");
    timer_.reset();

    // cudaMemcpy(d_Cin,  Cin.read_h_data().data(),  tensor_mem, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_Cout, Cout.read_h_data().data(), tensor_mem, cudaMemcpyHostToDevice);

    // my_timer.record("cudamemcpy");
    // my_timer.reset();

    cuDoubleComplex cu_coeff = make_cuDoubleComplex(coeff.real(), coeff.imag());

    apply_individual_nbody1_accumulate_wrapper(
        cu_coeff, 
        thrust::raw_pointer_cast(Cin.read_d_data().data()), 
        thrust::raw_pointer_cast(Cout.d_data().data()), 
        d_sourcea,
        d_targeta,
        d_paritya,
        d_sourceb,
        d_targetb,
        d_parityb,
        nbeta_strs_,
        targeta.size(),
        targetb.size(),
        tensor_mem);


    timer_.acc_record("calling gpu function");
    timer_.reset();
    // my_timer.record("gpu function");
    // my_timer.reset();


    // cudaMemcpy(Cout.data().data(), d_Cout, tensor_mem, cudaMemcpyDeviceToHost);


    cudaFree(d_sourcea);
    cudaFree(d_sourceb);
    cudaFree(d_targeta);
    cudaFree(d_targetb);
    cudaFree(d_paritya);
    cudaFree(d_parityb);
    // cudaFree(d_Cin);
    // cudaFree(d_Cout);

    timer_.acc_record("cudafree");
    timer_.reset();


    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("Failed to execute the apply_individual_nbody1_accumulate operation on the GPU.");
    }    
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
    if((daga.size() != undaga.size()) or (dagb.size() != undagb.size())){
        throw std::runtime_error("must be same number of alpha anihilators/creators and beta anihilators/creators.");
    }

    local_timer my_timer = local_timer();
    timer_.reset();

    std::tuple<int, std::vector<int>, std::vector<int>, std::vector<int>> ualfamap = graph_.make_mapping_each(
        true,
        daga,
        undaga);

    timer_.acc_record("first 'make_mapping_each' in apply_individual_nbody_accumulate");
    timer_.reset();

    if (std::get<0>(ualfamap) == 0) {
        return;
    }

    std::tuple<int, std::vector<int>, std::vector<int>, std::vector<int>> ubetamap = graph_.make_mapping_each(
        false,
        dagb,
        undagb);

    timer_.acc_record("second 'make_mapping_each' in apply_individual_nbody_accumulate");
    timer_.reset();

    if (std::get<0>(ubetamap) == 0) {
        return;
    }

    std::vector<int> sourcea(std::get<0>(ualfamap));
    std::vector<int> targeta(std::get<0>(ualfamap));
    std::vector<int> paritya(std::get<0>(ualfamap));
    std::vector<int> sourceb(std::get<0>(ubetamap));
    std::vector<int> targetb(std::get<0>(ubetamap));
    std::vector<int> parityb(std::get<0>(ubetamap));

    timer_.acc_record("a lot of initialization in apply_individual_nbody_accumulate");
    timer_.reset();

    /// NICK: All this can be done in the make_mapping_each fucntion.
    /// Maybe try like a make_abbrev_mapping_each

    /// NICK: Might be slow, check this out...
    for (int i = 0; i < std::get<0>(ualfamap); i++) {
        sourcea[i] = std::get<1>(ualfamap)[i];
        targeta[i] = graph_.get_aind_for_str(std::get<2>(ualfamap)[i]);
        paritya[i] = 1.0 - 2.0 * std::get<3>(ualfamap)[i];
    }

    timer_.acc_record("first for loop in apply_individual_nbody_accumulate");
    timer_.reset();

    for (int i = 0; i < std::get<0>(ubetamap); i++) {
        sourceb[i] = std::get<1>(ubetamap)[i];
        targetb[i] = graph_.get_bind_for_str(std::get<2>(ubetamap)[i]);
        parityb[i] = 1.0 - 2.0 * std::get<3>(ubetamap)[i];
    }

    timer_.acc_record("second for loop in apply_individual_nbody_accumulate");
    timer_.reset();
    // std::cout << my_timer.str_table() << std::endl;
    // this is where the if statement goes
    apply_individual_nbody1_accumulate(
        coeff, 
        Cin,
        Cout,
        sourcea,
        targeta,
        paritya,
        sourceb,
        targetb,
        parityb);    
}

void FCIComputerThrust::apply_individual_sqop_term(
    const std::tuple< std::complex<double>, std::vector<size_t>, std::vector<size_t>>& term,
    const TensorThrust& Cin,
    TensorThrust& Cout)
{
    std::vector<int> crea;
    std::vector<int> anna;

    std::vector<int> creb;
    std::vector<int> annb;

    local_timer my_timer = local_timer();
    timer_.reset();

    for(size_t i = 0; i < std::get<1>(term).size(); i++){
        if(std::get<1>(term)[i]%2 == 0){
            crea.push_back(std::floor(std::get<1>(term)[i] / 2));
        } else {
            creb.push_back(std::floor(std::get<1>(term)[i] / 2));
        }
    }

    timer_.acc_record("first loop in apply_individual_sqop_term");
    timer_.reset();

    for(size_t i = 0; i < std::get<2>(term).size(); i++){
        if(std::get<2>(term)[i]%2 == 0){
            anna.push_back(std::floor(std::get<2>(term)[i] / 2));
        } else {
            annb.push_back(std::floor(std::get<2>(term)[i] / 2));
        }
    }

    timer_.acc_record("second loop in apply_individual_sqop_term");
    timer_.reset();

    if (std::get<1>(term).size() != std::get<2>(term).size()) {
        throw std::invalid_argument("Each term must have same number of anihilators and creators");
    }   

    std::vector<size_t> ops1(std::get<1>(term));
    std::vector<size_t> ops2(std::get<2>(term));
    ops1.insert(ops1.end(), ops2.begin(), ops2.end());

    int nswaps = parity_sort(ops1);
    timer_.acc_record("some parity things");
    timer_.reset();
    // std::cout << my_timer.str_table() << std::endl;

    apply_individual_nbody_accumulate(
        pow(-1, nswaps) * std::get<0>(term),
        Cin,
        Cout,
        crea,
        anna, 
        creb,
        annb);
}

void FCIComputerThrust::apply_sqop(const SQOperator& sqop)
{
     C_.gpu_error();
    TensorThrust Cin(C_.shape(), "Cin", true);
    Cin.copy_in_gpu(C_);


    
    local_timer my_timer = local_timer();
    timer_.reset();
    
    // cudaMemcpy(Cin.d_data(), C_.d_data(), Cin.size() * sizeof(cuDoubleComplex))

    C_.zero_gpu();

    timer_.acc_record("making tensor things");
    timer_.reset();

    for (const auto& term : sqop.terms()) {
        if(std::abs(std::get<0>(term)) > compute_threshold_){
        apply_individual_sqop_term(
            term,
            Cin,
            C_);
        }
    }

    timer_.acc_record("first for loop in apply_sqop");
    std::cout << timer_.acc_str_table() << std::endl;
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

/* New methods for copying out data */
void FCIComputerThrust::copy_to_tensor(Tensor& tensor) const
{
    cpu_error();
    C_.copy_to_tensor(tensor);
}

void FCIComputerThrust::copy_to_tensor_thrust_gpu(TensorThrust& tensor) const
{
    gpu_error();
    tensor.copy_in_gpu(C_);
}

void FCIComputerThrust::copy_to_tensor_thrust_cpu(TensorThrust& tensor) const
{
    cpu_error();
    tensor.copy_in(C_);
}