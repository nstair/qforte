#include <map>
#include <random>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <iterator>
#include <limits>

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

#include "fci_computer_gpu.h"
#include "fci_graph_gpu.h"
#include "qforte_globals.h"

#include "cublas_math.cuh"
#include "fci_computer_gpu_kernels.cuh"

namespace {

struct IncomingExcitationTablesGPUV2 {
    thrust::device_vector<long long> offsets;
    thrust::device_vector<int> sources;
    thrust::device_vector<int> pairs;
    thrust::device_vector<int> parities;
};

IncomingExcitationTablesGPUV2 build_incoming_excitation_tables_gpu_v2(
    const std::vector<int>& dexc,
    const int nstates,
    const int ndexc,
    const char* label)
{
    if (nstates < 0) {
        throw std::invalid_argument(std::string(label) + " nstates is negative");
    }
    if (ndexc < 0) {
        throw std::invalid_argument(std::string(label) + " ndexc is negative");
    }

    const size_t expected_size =
        static_cast<size_t>(nstates) * static_cast<size_t>(ndexc) * 3;
    if (dexc.size() != expected_size) {
        throw std::invalid_argument(std::string(label) + " dexc table has unexpected size");
    }

    std::vector<long long> h_counts(static_cast<size_t>(nstates), 0);
    for (int source = 0; source < nstates; ++source) {
        for (int edge = 0; edge < ndexc; ++edge) {
            const int base = 3 * (source * ndexc + edge);
            const int target = dexc[base + 0];
            if (target < 0 || target >= nstates) {
                throw std::invalid_argument(std::string(label) + " dexc target is out of range");
            }
            ++h_counts[static_cast<size_t>(target)];
        }
    }

    std::vector<long long> h_offsets(static_cast<size_t>(nstates) + 1, 0);
    for (int state = 0; state < nstates; ++state) {
        h_offsets[static_cast<size_t>(state) + 1] =
            h_offsets[static_cast<size_t>(state)] + h_counts[static_cast<size_t>(state)];
    }

    const long long total_edges_ll = h_offsets.back();
    if (total_edges_ll < 0 ||
        static_cast<unsigned long long>(total_edges_ll)
            > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error(std::string(label) + " incoming table is too large");
    }
    const size_t total_edges = static_cast<size_t>(total_edges_ll);

    std::vector<int> h_sources(total_edges);
    std::vector<int> h_pairs(total_edges);
    std::vector<int> h_parities(total_edges);
    std::vector<long long> h_cursors = h_offsets;

    for (int source = 0; source < nstates; ++source) {
        for (int edge = 0; edge < ndexc; ++edge) {
            const int base = 3 * (source * ndexc + edge);
            const int target = dexc[base + 0];
            const int pair = dexc[base + 1];
            const int parity = dexc[base + 2];

            const size_t pos = static_cast<size_t>(
                h_cursors[static_cast<size_t>(target)]++);
            h_sources[pos] = source;
            h_pairs[pos] = pair;
            h_parities[pos] = parity;
        }
    }

    IncomingExcitationTablesGPUV2 tables;
    tables.offsets = h_offsets;
    tables.sources = h_sources;
    tables.pairs = h_pairs;
    tables.parities = h_parities;
    return tables;
}

} // namespace

FCIComputerGPU::FCIComputerGPU(int nel, int sz, int norb, bool on_gpu, const std::string& data_type, bool gpu_only) :
    nel_(nel),
    sz_(sz),
    norb_(norb),
    on_gpu_(on_gpu),
    data_type_(data_type),
    gpu_only_(gpu_only){

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

    C_.zero_with_shape(
        {nalfa_strs_, nbeta_strs_},
        on_gpu_,
        data_type_,
        gpu_only_);

    C_.set_name("FCI Computer");

    // Initialize the FCI graph tensors (can get rid of these if only use precomp tuple)
    sourcea_gpu_.reserve(nalfa_strs_);
    targeta_gpu_.reserve(nalfa_strs_);
    paritya_gpu_.reserve(nalfa_strs_);
    paritya_gpu_real_.reserve(nalfa_strs_);

    sourcea_undag_gpu_.reserve(nalfa_strs_);
    targeta_undag_gpu_.reserve(nalfa_strs_);
    paritya_undag_gpu_.reserve(nalfa_strs_);
    paritya_undag_gpu_real_.reserve(nalfa_strs_);

    sourceb_gpu_.reserve(nbeta_strs_);
    targetb_gpu_.reserve(nbeta_strs_);
    parityb_gpu_.reserve(nbeta_strs_);
    parityb_gpu_real_.reserve(nbeta_strs_);

    sourceb_undag_gpu_.reserve(nbeta_strs_);
    targetb_undag_gpu_.reserve(nbeta_strs_);
    parityb_undag_gpu_.reserve(nbeta_strs_);
    parityb_undag_gpu_real_.reserve(nbeta_strs_);

    graph_ = FCIGraphGPU(nalfa_el_, nbeta_el_, norb_);

    // start cublas math
    math_gpu_init();

    // timer_ = local_timer();
}

/// Destructor: properly cleanup GPU resources
FCIComputerGPU::~FCIComputerGPU() {
    try {

        sourcea_gpu_.clear();
        sourcea_gpu_.shrink_to_fit();

        targeta_gpu_.clear();
        targeta_gpu_.shrink_to_fit();

        paritya_gpu_.clear();
        paritya_gpu_.shrink_to_fit();

        paritya_gpu_real_.clear();
        paritya_gpu_real_.shrink_to_fit();

        sourcea_undag_gpu_.clear();
        sourcea_undag_gpu_.shrink_to_fit();

        targeta_undag_gpu_.clear();
        targeta_undag_gpu_.shrink_to_fit();

        paritya_undag_gpu_.clear();
        paritya_undag_gpu_.shrink_to_fit();

        paritya_undag_gpu_real_.clear();
        paritya_undag_gpu_real_.shrink_to_fit();

        sourceb_gpu_.clear();
        sourceb_gpu_.shrink_to_fit();

        targetb_gpu_.clear();
        targetb_gpu_.shrink_to_fit();

        parityb_gpu_.clear();
        parityb_gpu_.shrink_to_fit();

        parityb_gpu_real_.clear();
        parityb_gpu_real_.shrink_to_fit();

        sourceb_undag_gpu_.clear();
        sourceb_undag_gpu_.shrink_to_fit();

        targetb_undag_gpu_.clear();
        targetb_undag_gpu_.shrink_to_fit();

        parityb_undag_gpu_.clear();
        parityb_undag_gpu_.shrink_to_fit();

        parityb_undag_gpu_real_.clear();
        parityb_undag_gpu_real_.shrink_to_fit();

        math_gpu_finalize();

    } catch (const std::exception& e) {
        // std::cerr << "Caught exception in FCIComputerGPU destructor: " << e.what() << std::endl;
    } catch (...) {
        // std::cerr << "Caught unknown exception in FCIComputerGPU destructor." << std::endl;
    }
}

/// Set a particular element of the tensor stored in FCIComputerGPU, specified by idxs
void FCIComputerGPU::set_element(
    const std::vector<size_t>& idxs,
    const std::complex<double> val
        )
{
    C_.set(idxs, val);
}

std::complex<double> FCIComputerGPU::get_element(
    const std::vector<size_t>& idxs
        )
{
    return C_.get(idxs);
}

void FCIComputerGPU::gpu_error() const {

    if (not on_gpu_) {
        throw std::runtime_error("Data not on GPU for FCIComputerGPU " + name_);
    }

}

void FCIComputerGPU::cpu_error() const {

    if (on_gpu_) {
        throw std::runtime_error("Data not on CPU for FCIComputerGPU " + name_);
    }

}

void FCIComputerGPU::to_gpu()
{
    cpu_error();
    C_.to_gpu();
    on_gpu_ = 1;
}

// change to 'to_cpu'
void FCIComputerGPU::to_cpu()
{
    gpu_error();
    C_.to_cpu();
    on_gpu_ = 0;
}

/// apply a TensorOperator to the current state
// void apply_tensor_operator(const TensorOperator& top);

/// apply a Tensor represending a 1-body spin-orbital indexed operator to the current state
void FCIComputerGPU::apply_tensor_spin_1bdy(const TensorGPU& h1e, size_t norb) {

    if(h1e.size() != (norb * 2) * (norb * 2)){
        throw std::invalid_argument("Expecting h1e to be nso x nso for apply_tensor_spin_1bdy");
    }

    TensorGPU Cnew({nalfa_strs_, nbeta_strs_}, "Cnew");

    TensorGPU h1e_blk1 = h1e.slice(
        {
            std::make_pair(0, norb_),
            std::make_pair(0, norb_)
            }
        );

    TensorGPU h1e_blk2 = h1e.slice(
        {
            std::make_pair(norb_, 2*norb_),
            std::make_pair(norb_, 2*norb_)
            }
        );

    apply_array_1bdy_cpu(
        Cnew,
        graph_.read_dexca_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexca(),
        h1e_blk1,
        norb_,
        true);

    apply_array_1bdy_cpu(
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

/// apply TensorGPUs represending 1-body and 2-body spatial-orbital indexed operator to the current state
void FCIComputerGPU::apply_tensor_spat_12bdy_gpu(
    const TensorGPU& h1e,
    const TensorGPU& h2e,
    TensorGPU& h2e_einsum,
    size_t norb) {

    gpu_error();

    if(h1e.size() != (norb) * (norb)){
        throw std::invalid_argument("Expecting h1e to be nmo x nmo for apply_tensor_spat_12bdy_gpu");
    }

    if(h2e.size() != (norb) * (norb) * (norb) * (norb) ){
        throw std::invalid_argument("Expecting h2e to be nso x nso x nso x nso for apply_tensor_spat_12bdy_gpu");
    }

    TensorGPU Cnew({nalfa_strs_, nbeta_strs_}, "Cnew", true, data_type_, true);
    Cnew.zero_gpu();

    timer_.acc_begin("=> same spin alpha outer");
    lm_apply_array12_same_spin_opt_gpu(
        Cnew,
        graph_.read_dexca_vec(), // dexca_tmp
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexca(),
        h1e,
        h2e,
        norb_,
        true);
    timer_.acc_end("=> same spin alpha outer");

    timer_.acc_begin("=> same spin beta outer");

    // Cnew.fineGrainedTranspose();

    lm_apply_array12_same_spin_opt_gpu(
        Cnew,
        graph_.read_dexcb_vec(), // dexcb_tmp - FIXED: was dexca_vec
        nbeta_strs_,             // FIXED: swapped nalfa <-> nbeta after transpose
        nalfa_strs_,             // FIXED: swapped nalfa <-> nbeta after transpose
        graph_.get_ndexcb(),     // FIXED: was get_ndexca()
        h1e,
        h2e,
        norb_,
        false);                  // FIXED: was true, should be false for beta

    // Cnew.fineGrainedTranspose();

    timer_.acc_end("=> same spin beta outer");

    timer_.acc_begin("=> diff spin outer");
    lm_apply_array12_diff_spin_opt_gpu(
        Cnew,
        graph_.read_dexca_vec(),
        graph_.read_dexcb_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexca(),
        graph_.get_ndexca(),
        h2e_einsum,
        norb_);

    timer_.acc_end("=> diff spin outer");

    C_ = Cnew;
}

/// apply TensorGPUs represending 1-body and 2-body spatial-orbital indexed operator
/// as well as a constant to the current state
/// Computes: C_ = h0e * C_ + sigma_12bdy(C_)
/// Memory-optimized: uses 2 state vectors (C_ + Cnew) instead of 3 (C_ + Cold + Cnew)
void FCIComputerGPU::apply_tensor_spat_012bdy_gpu(
    const std::complex<double> h0e,
    const TensorGPU& h1e,
    const TensorGPU& h2e,
    TensorGPU& h2e_einsum,
    size_t norb)
{
    gpu_error();

    if(h1e.size() != (norb) * (norb)){
        throw std::invalid_argument("Expecting h1e to be nmo x nmo for apply_tensor_spat_012bdy_gpu");
    }

    if(h2e.size() != (norb) * (norb) * (norb) * (norb) ){
        throw std::invalid_argument("Expecting h2e to be nso x nso x nso x nso for apply_tensor_spat_012bdy_gpu");
    }

    // Accumulate sigma = H_12bdy * C_ into Cnew while C_ remains untouched
    TensorGPU Cnew({nalfa_strs_, nbeta_strs_}, "Cnew", true, data_type_, true);
    Cnew.zero_gpu();

    timer_.acc_begin("=> same spin alpha outer");
    lm_apply_array12_same_spin_opt_gpu(
        Cnew,
        graph_.read_dexca_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexca(),
        h1e,
        h2e,
        norb_,
        true);
    timer_.acc_end("=> same spin alpha outer");

    timer_.acc_begin("=> same spin beta outer");
    lm_apply_array12_same_spin_opt_gpu(
        Cnew,
        graph_.read_dexcb_vec(),
        nbeta_strs_,
        nalfa_strs_,
        graph_.get_ndexcb(),
        h1e,
        h2e,
        norb_,
        false);
    timer_.acc_end("=> same spin beta outer");

    timer_.acc_begin("=> diff spin outer");
    lm_apply_array12_diff_spin_opt_gpu(
        Cnew,
        graph_.read_dexca_vec(),
        graph_.read_dexcb_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexca(),
        graph_.get_ndexca(),
        h2e_einsum,
        norb_);
    timer_.acc_end("=> diff spin outer");

    // C_ = 1.0 * Cnew + h0e * C_  (no Cold copy needed)
    C_.zaxpby(
        Cnew,
        1.0,
        h0e,
        1,
        1
    );
}

void FCIComputerGPU::apply_tensor_spat_12bdy_gpu_v2(
    const TensorGPU& h1e,
    const TensorGPU& h2e,
    TensorGPU& h2e_einsum,
    size_t norb)
{
    gpu_error();

    if(h1e.size() != (norb) * (norb)){
        throw std::invalid_argument("Expecting h1e to be nmo x nmo for apply_tensor_spat_12bdy_gpu_v2");
    }

    if(h2e.size() != (norb) * (norb) * (norb) * (norb) ){
        throw std::invalid_argument("Expecting h2e to be nso x nso x nso x nso for apply_tensor_spat_12bdy_gpu_v2");
    }

    TensorGPU Cnew({nalfa_strs_, nbeta_strs_}, "Cnew_v2", true, data_type_, true);
    Cnew.zero_gpu();

    timer_.acc_begin("=> same spin alpha outer");
    lm_apply_array12_same_spin_opt_gpu(
        Cnew,
        graph_.read_dexca_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexca(),
        h1e,
        h2e,
        norb_,
        true);
    timer_.acc_end("=> same spin alpha outer");

    timer_.acc_begin("=> same spin beta outer transposed");
    C_.fineGrainedTranspose();
    Cnew.fineGrainedTranspose();

    // Experimental v2 beta same-spin path: after transposing the CI tensors,
    // beta strings are laid out as contiguous rows.  Reuse the same row-major
    // same-spin SpMM path as alpha by passing the beta excitation table with
    // is_alpha=true.  Both tensors are transposed back before mixed-spin.
    lm_apply_array12_same_spin_opt_gpu(
        Cnew,
        graph_.read_dexcb_vec(),
        nbeta_strs_,
        nalfa_strs_,
        graph_.get_ndexcb(),
        h1e,
        h2e,
        norb_,
        true);

    C_.fineGrainedTranspose();
    Cnew.fineGrainedTranspose();
    timer_.acc_end("=> same spin beta outer transposed");

    timer_.acc_begin("=> diff spin outer v2 tiled");
    lm_apply_array12_diff_spin_opt_gpu_v2_tiled(
        Cnew,
        graph_.read_dexca_vec(),
        graph_.read_dexcb_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexca(),
        graph_.get_ndexcb(),
        h2e_einsum,
        norb_);
    timer_.acc_end("=> diff spin outer v2 tiled");

    C_ = Cnew;
}

void FCIComputerGPU::apply_tensor_spat_012bdy_gpu_v2(
    const std::complex<double> h0e,
    const TensorGPU& h1e,
    const TensorGPU& h2e,
    TensorGPU& h2e_einsum,
    size_t norb)
{
    gpu_error();

    if(h1e.size() != (norb) * (norb)){
        throw std::invalid_argument("Expecting h1e to be nmo x nmo for apply_tensor_spat_012bdy_gpu_v2");
    }

    if(h2e.size() != (norb) * (norb) * (norb) * (norb) ){
        throw std::invalid_argument("Expecting h2e to be nso x nso x nso x nso for apply_tensor_spat_012bdy_gpu_v2");
    }

    TensorGPU Cnew({nalfa_strs_, nbeta_strs_}, "Cnew_v2", true, data_type_, true);
    Cnew.zero_gpu();

    timer_.acc_begin("=> same spin alpha outer");
    lm_apply_array12_same_spin_opt_gpu(
        Cnew,
        graph_.read_dexca_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexca(),
        h1e,
        h2e,
        norb_,
        true);
    timer_.acc_end("=> same spin alpha outer");

    timer_.acc_begin("=> same spin beta outer transposed");
    C_.fineGrainedTranspose();
    Cnew.fineGrainedTranspose();

    // Experimental v2 beta same-spin path: after transposing the CI tensors,
    // beta strings are laid out as contiguous rows.  Reuse the same row-major
    // same-spin SpMM path as alpha by passing the beta excitation table with
    // is_alpha=true.  Both tensors are transposed back before mixed-spin.
    lm_apply_array12_same_spin_opt_gpu(
        Cnew,
        graph_.read_dexcb_vec(),
        nbeta_strs_,
        nalfa_strs_,
        graph_.get_ndexcb(),
        h1e,
        h2e,
        norb_,
        true);

    C_.fineGrainedTranspose();
    Cnew.fineGrainedTranspose();
    timer_.acc_end("=> same spin beta outer transposed");

    timer_.acc_begin("=> diff spin outer v2 tiled");
    lm_apply_array12_diff_spin_opt_gpu_v2_tiled(
        Cnew,
        graph_.read_dexca_vec(),
        graph_.read_dexcb_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexca(),
        graph_.get_ndexcb(),
        h2e_einsum,
        norb_);
    timer_.acc_end("=> diff spin outer v2 tiled");

    C_.zaxpby(
        Cnew,
        1.0,
        h0e,
        1,
        1
    );
}

/// Set a particular element of this TensorGPU, specified by idxs
void FCIComputerGPU::add_to_element(
    const std::vector<size_t>& idxs,
    const std::complex<double> val
        )
{
    C_.add_to_element(idxs, val);
}

/// TODO: Not implemented in GPU so skipping
/*
void FCIComputerGPU::apply_tensor_operator(const TensorOperator& top)
{
    // Implementation would be similar to FCIComputerGPU but using TensorGPU
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("");
}*/

/// TODO: this is commented out in FCIComputerGPU, so skipping
/*
void FCIComputerGPU::apply_tensor_spin_12bdy(
    const TensorGPU& h1e,
    const TensorGPU& h2e,
    size_t norb)
{
    // Implementation would be similar to FCIComputerGPU but using TensorGPU
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("");
}*/

/// TODO: in GPU header but not implemented in GPU source
/*
void FCIComputerGPU::lm_apply_array1(
    const TensorGPU& out,
    const std::vector<int> dexc,
    const int astates,
    const int bstates,
    const int ndexc,
    const TensorGPU& h1e,
    const int norbs,
    const bool is_alpha)
{
    // Implementation would be similar to FCIComputerGPU but using TensorGPU
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("");
}*/

void FCIComputerGPU::apply_array_1bdy_cpu(
    TensorGPU& out,
    const std::vector<int>& dexc,
    const int astates,
    const int bstates,
    const int ndexc,
    const TensorGPU& h1e,
    const int norbs,
    const bool is_alpha)
{
    cpu_error();

    const int states1 = is_alpha ? astates : bstates;
    const int states2 = is_alpha ? bstates : astates;
    const int inc1 = is_alpha ? bstates : 1;
    const int inc2 = is_alpha ? 1 : bstates;

    for (int s1 = 0; s1 < states1; ++s1) {
        const int* cdexc = dexc.data() + 3 * s1 * ndexc;
        const int* lim1 = cdexc + 3 * ndexc;
        std::complex<double>* cout = out.data().data() + s1 * inc1;

        for (; cdexc < lim1; cdexc = cdexc + 3) {
            const int target = cdexc[0];
            const int ijshift = cdexc[1];
            const int parity = cdexc[2];

            const std::complex<double> pref = static_cast<double>(parity) * h1e.read_h_data()[ijshift];
            const std::complex<double>* xptr = C_.data().data() + target * inc1;

            math_zaxpy(states2, pref, xptr, inc2, cout, inc2);
        }
    }
}

void FCIComputerGPU::lm_apply_array12_same_spin_opt_gpu(
    TensorGPU& out,
    const std::vector<int>& dexc,
    const int alpha_states,
    const int beta_states,
    const int ndexc,
    const TensorGPU& h1e,
    const TensorGPU& h2e,
    const int norbs,
    const bool is_alpha)
{
    timer_.acc_begin("==> same spin data transfer");

    gpu_error();

    const int states1 = is_alpha ? alpha_states : beta_states;
    const int states2 = is_alpha ? beta_states : alpha_states;
    const int inc1 = is_alpha ? beta_states : 1;
    const int inc2 = is_alpha ? 1 : beta_states;

    // Transfer dexc to device
    thrust::device_vector<int> d_dexc(dexc.begin(), dexc.end());
    const int* d_dexc_ptr = thrust::raw_pointer_cast(d_dexc.data());

    timer_.acc_end("==> same spin data transfer");

    timer_.acc_begin("==> same spin kernel (CSR SpMM)");

    if (data_type_ == "real") {
        // Get device pointers for real data
        double* d_out_real = thrust::raw_pointer_cast(out.d_re_data().data());
        const double* d_C_real = thrust::raw_pointer_cast(C_.d_re_data().data());
        const double* d_h1e_real = thrust::raw_pointer_cast(h1e.read_d_re_data().data());
        const double* d_h2e_real = thrust::raw_pointer_cast(h2e.read_d_re_data().data());

        lm_apply_array12_same_spin_spmm_csr_coalesced_wrapper_real(
            d_out_real,
            d_C_real,
            d_dexc_ptr,
            d_h1e_real,
            d_h2e_real,
            states1,
            states2,
            ndexc,
            norbs,
            inc1,
            inc2);
    } else if (data_type_ == "complex" && h2e.data_type() == "real") {
        // Get device pointers for complex and real data
        cuDoubleComplex* d_out = thrust::raw_pointer_cast(out.d_data().data());
        const cuDoubleComplex* d_C = thrust::raw_pointer_cast(C_.d_data().data());
        const double* d_h1e = thrust::raw_pointer_cast(h1e.read_d_re_data().data());
        const double* d_h2e = thrust::raw_pointer_cast(h2e.read_d_re_data().data());

        lm_apply_array12_same_spin_spmm_csr_coalesced_wrapper_mixed(
            d_out,
            d_C,
            d_dexc_ptr,
            d_h1e,
            d_h2e,
            states1,
            states2,
            ndexc,
            norbs,
            inc1,
            inc2);
    } else if (data_type_ == "complex" && h2e.data_type() == "complex") {
        // Get device pointers for complex data
        cuDoubleComplex* d_out = thrust::raw_pointer_cast(out.d_data().data());
        const cuDoubleComplex* d_C = thrust::raw_pointer_cast(C_.d_data().data());
        const cuDoubleComplex* d_h1e = thrust::raw_pointer_cast(h1e.read_d_data().data());
        const cuDoubleComplex* d_h2e = thrust::raw_pointer_cast(h2e.read_d_data().data());

        lm_apply_array12_same_spin_spmm_csr_coalesced_wrapper(
            d_out,
            d_C,
            d_dexc_ptr,
            d_h1e,
            d_h2e,
            states1,
            states2,
            ndexc,
            norbs,
            inc1,
            inc2);
    } else {
        // Error handling for unsupported data types
        throw std::runtime_error("Unsupported data type combination in lm_apply_array12_same_spin_opt_gpu");
    }

    timer_.acc_end("==> same spin kernel (CSR SpMM)");
}

void FCIComputerGPU::lm_apply_array12_diff_spin_opt_gpu(
    TensorGPU& out,
    const std::vector<int>& adexc,
    const std::vector<int>& bdexc,
    const int alpha_states,
    const int beta_states,
    const int nadexc,
    const int nbdexc,
    TensorGPU& h2e,
    const int norbs)
{
    gpu_error();

    // Copy excitation tables to device
    thrust::device_vector<int> d_adexc(adexc.begin(), adexc.end());
    thrust::device_vector<int> d_bdexc(bdexc.begin(), bdexc.end());

    if (data_type_ == "real") {
        // Get device pointers for real data
        double* d_out_real = thrust::raw_pointer_cast(out.d_re_data().data());
        const double* d_C_real = thrust::raw_pointer_cast(C_.d_re_data().data());
        const double* d_h2e_real = thrust::raw_pointer_cast(h2e.d_re_data().data());

        lm_apply_array12_diff_spin_wrapper_real(
            d_out_real,
            d_C_real,
            thrust::raw_pointer_cast(d_adexc.data()),
            thrust::raw_pointer_cast(d_bdexc.data()),
            d_h2e_real,
            alpha_states,
            beta_states,
            nadexc,
            nbdexc,
            norbs);
    } else if (data_type_ == "complex" && h2e.data_type() == "real") {
        // Get device pointers for complex and real data
        cuDoubleComplex* d_out = thrust::raw_pointer_cast(out.d_data().data());
        const cuDoubleComplex* d_C = thrust::raw_pointer_cast(C_.d_data().data());
        const double* d_h2e = thrust::raw_pointer_cast(h2e.d_re_data().data());

        lm_apply_array12_diff_spin_wrapper_mixed(
            d_out,
            d_C,
            thrust::raw_pointer_cast(d_adexc.data()),
            thrust::raw_pointer_cast(d_bdexc.data()),
            d_h2e,
            alpha_states,
            beta_states,
            nadexc,
            nbdexc,
            norbs);
    } else if (data_type_ == "complex" && h2e.data_type() == "complex") {
        // Get device pointers for complex data
        cuDoubleComplex* d_out = thrust::raw_pointer_cast(out.d_data().data());
        const cuDoubleComplex* d_C = thrust::raw_pointer_cast(C_.d_data().data());
        const cuDoubleComplex* d_h2e = thrust::raw_pointer_cast(h2e.d_data().data());

        lm_apply_array12_diff_spin_wrapper(
            d_out,
            d_C,
            thrust::raw_pointer_cast(d_adexc.data()),
            thrust::raw_pointer_cast(d_bdexc.data()),
            d_h2e,
            alpha_states,
            beta_states,
            nadexc,
            nbdexc,
            norbs);
    } else {
        // Error handling for unsupported data types
        throw std::runtime_error("Unsupported data type combination in lm_apply_array12_diff_spin_opt_gpu");
    }
}

void FCIComputerGPU::lm_apply_array12_diff_spin_opt_gpu_v2_tiled(
    TensorGPU& out,
    const std::vector<int>& adexc,
    const std::vector<int>& bdexc,
    const int alpha_states,
    const int beta_states,
    const int nadexc,
    const int nbdexc,
    TensorGPU& h2e,
    const int norbs)
{
    gpu_error();

    timer_.acc_begin("==> diff spin v2 incoming transfer");
    IncomingExcitationTablesGPUV2 alpha_incoming =
        build_incoming_excitation_tables_gpu_v2(
            adexc,
            alpha_states,
            nadexc,
            "alpha");
    IncomingExcitationTablesGPUV2 beta_incoming =
        build_incoming_excitation_tables_gpu_v2(
            bdexc,
            beta_states,
            nbdexc,
            "beta");
    timer_.acc_end("==> diff spin v2 incoming transfer");

    const long long alpha_states_ll = static_cast<long long>(alpha_states);
    const long long beta_states_ll = static_cast<long long>(beta_states);

    timer_.acc_begin("==> diff spin v2 tiled kernel");
    if (data_type_ == "real") {
        double* d_out_real = thrust::raw_pointer_cast(out.d_re_data().data());
        const double* d_C_real = thrust::raw_pointer_cast(C_.read_d_re_data().data());
        const double* d_h2e_real = thrust::raw_pointer_cast(h2e.read_d_re_data().data());

        lm_apply_array12_diff_spin_v2_tiled_wrapper_real(
            d_out_real,
            d_C_real,
            thrust::raw_pointer_cast(alpha_incoming.offsets.data()),
            thrust::raw_pointer_cast(alpha_incoming.sources.data()),
            thrust::raw_pointer_cast(alpha_incoming.pairs.data()),
            thrust::raw_pointer_cast(alpha_incoming.parities.data()),
            thrust::raw_pointer_cast(beta_incoming.offsets.data()),
            thrust::raw_pointer_cast(beta_incoming.sources.data()),
            thrust::raw_pointer_cast(beta_incoming.pairs.data()),
            thrust::raw_pointer_cast(beta_incoming.parities.data()),
            d_h2e_real,
            alpha_states_ll,
            beta_states_ll,
            norbs);
    } else if (data_type_ == "complex" && h2e.data_type() == "real") {
        cuDoubleComplex* d_out = thrust::raw_pointer_cast(out.d_data().data());
        const cuDoubleComplex* d_C = thrust::raw_pointer_cast(C_.read_d_data().data());
        const double* d_h2e = thrust::raw_pointer_cast(h2e.read_d_re_data().data());

        lm_apply_array12_diff_spin_v2_tiled_wrapper_mixed(
            d_out,
            d_C,
            thrust::raw_pointer_cast(alpha_incoming.offsets.data()),
            thrust::raw_pointer_cast(alpha_incoming.sources.data()),
            thrust::raw_pointer_cast(alpha_incoming.pairs.data()),
            thrust::raw_pointer_cast(alpha_incoming.parities.data()),
            thrust::raw_pointer_cast(beta_incoming.offsets.data()),
            thrust::raw_pointer_cast(beta_incoming.sources.data()),
            thrust::raw_pointer_cast(beta_incoming.pairs.data()),
            thrust::raw_pointer_cast(beta_incoming.parities.data()),
            d_h2e,
            alpha_states_ll,
            beta_states_ll,
            norbs);
    } else if (data_type_ == "complex" && h2e.data_type() == "complex") {
        cuDoubleComplex* d_out = thrust::raw_pointer_cast(out.d_data().data());
        const cuDoubleComplex* d_C = thrust::raw_pointer_cast(C_.read_d_data().data());
        const cuDoubleComplex* d_h2e = thrust::raw_pointer_cast(h2e.read_d_data().data());

        lm_apply_array12_diff_spin_v2_tiled_wrapper(
            d_out,
            d_C,
            thrust::raw_pointer_cast(alpha_incoming.offsets.data()),
            thrust::raw_pointer_cast(alpha_incoming.sources.data()),
            thrust::raw_pointer_cast(alpha_incoming.pairs.data()),
            thrust::raw_pointer_cast(alpha_incoming.parities.data()),
            thrust::raw_pointer_cast(beta_incoming.offsets.data()),
            thrust::raw_pointer_cast(beta_incoming.sources.data()),
            thrust::raw_pointer_cast(beta_incoming.pairs.data()),
            thrust::raw_pointer_cast(beta_incoming.parities.data()),
            d_h2e,
            alpha_states_ll,
            beta_states_ll,
            norbs);
    } else {
        throw std::runtime_error("Unsupported data type combination in lm_apply_array12_diff_spin_opt_gpu_v2_tiled");
    }
    timer_.acc_end("==> diff spin v2 tiled kernel");
}

/// TODO: Not implemented in GPU so skipping
/*
std::pair<TensorGPU, TensorGPU> FCIComputerGPU::calculate_dvec_spin_with_coeff()
{
    // Implementation would be similar to FCIComputerGPU but using TensorGPU
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("");
}*/

TensorGPU FCIComputerGPU::calculate_coeff_spin_with_dvec_cpu(std::pair<TensorGPU, TensorGPU>& dvec)
{
    cpu_error();

    TensorGPU Cnew({nalfa_strs_, nbeta_strs_}, "Cnew");

    for (size_t i = 0; i < norb_; ++i) {
        for (size_t j = 0; j < norb_; ++j) {

            auto alfa_mappings = graph_.get_alfa_map()[std::make_pair(j,i)];
            auto beta_mappings = graph_.get_beta_map()[std::make_pair(j,i)];

            for (const auto& mapping : alfa_mappings) {
                size_t source = std::get<0>(mapping);
                size_t target = std::get<1>(mapping);
                std::complex<double> parity = static_cast<std::complex<double>>(std::get<2>(mapping));
                for (size_t k = 0; k < dvec.first.shape()[3]; ++k) {
                    size_t c_vidxa = k * Cnew.strides()[1] + source * Cnew.strides()[0];
                    size_t d_vidxa = k * dvec.first.strides()[3] + target * dvec.first.strides()[2] + j * dvec.first.strides()[1] + i * dvec.first.strides()[0];
                    Cnew.data()[c_vidxa] += parity * dvec.first.data()[d_vidxa];
                }

            }
            for (const auto& mapping : beta_mappings) {
                size_t source = std::get<0>(mapping);
                size_t target = std::get<1>(mapping);
                std::complex<double> parity = static_cast<std::complex<double>>(std::get<2>(mapping));
                for (size_t k = 0; k < dvec.second.shape()[2]; ++k) {
                    size_t c_vidxb = source * Cnew.strides()[1] + k * Cnew.strides()[0];
                    size_t d_vidxb = target * dvec.second.strides()[3] + k * dvec.second.strides()[2] + j * dvec.second.strides()[1] + i * dvec.second.strides()[0];
                    Cnew.data()[c_vidxb] += parity * dvec.second.data()[d_vidxb];
                }
            }
        }
    }

    return Cnew;
}

std::pair<std::vector<int>, std::vector<int>> FCIComputerGPU::evaluate_map_number_cpu(
    const std::vector<int>& numa,
    const std::vector<int>& numb)
{
    /// TODO: Implement separate CPU and GPU versions of this function
    // cpu_error();

    std::vector<int> amap(nalfa_strs_);
    std::vector<int> bmap(nbeta_strs_);

    uint64_t amask = graph_.reverse_integer_index(numa);
    uint64_t bmask = graph_.reverse_integer_index(numb);

    int acounter = 0;
    for (int index = 0; index < nalfa_strs_; ++index) {
        int current = graph_.get_astr_at_idx(index);
        if (((~current) & amask) == 0) {
            amap[acounter] = index;
            acounter++;
        }
    }

    int bcounter = 0;
    for (int index = 0; index < nbeta_strs_; ++index) {
        int current = graph_.get_bstr_at_idx(index);
        if (((~current) & bmask) == 0) {
            bmap[bcounter] = index;
            bcounter++;
        }
    }

    amap.resize(acounter);
    bmap.resize(bcounter);

    return std::make_pair(amap, bmap);
}

std::pair<std::vector<int>, std::vector<int>> FCIComputerGPU::evaluate_map_cpu(
    const std::vector<int>& crea,
    const std::vector<int>& anna,
    const std::vector<int>& creb,
    const std::vector<int>& annb)
{
    /// TODO: Implement separate CPU and GPU versions of this function
    // cpu_error();

    std::vector<int> amap(nalfa_strs_);
    std::vector<int> bmap(nbeta_strs_);

    uint64_t apmask = graph_.reverse_integer_index(crea);
    uint64_t ahmask = graph_.reverse_integer_index(anna);
    uint64_t bpmask = graph_.reverse_integer_index(creb);
    uint64_t bhmask = graph_.reverse_integer_index(annb);

    int acounter = 0;
    for (int index = 0; index < nalfa_strs_; ++index) {
        int current = graph_.get_astr_at_idx(index);
        if (((~current) & apmask) == 0 && (current & ahmask) == 0) {
            amap[acounter] = index;
            acounter++;
        }
    }

    int bcounter = 0;
    for (int index = 0; index < nbeta_strs_; ++index) {
        int current = graph_.get_bstr_at_idx(index);
        if (((~current) & bpmask) == 0 && (current & bhmask) == 0) {
            bmap[bcounter] = index;
            bcounter++;
        }
    }
    amap.resize(acounter);
    bmap.resize(bcounter);

    return std::make_pair(amap, bmap);
}

void FCIComputerGPU::apply_cos_inplace_cpu(
    const std::complex<double> time,
    const std::complex<double> coeff,
    const std::vector<int>& crea,
    const std::vector<int>& anna,
    const std::vector<int>& creb,
    const std::vector<int>& annb,
    TensorGPU& Cout)
{
    timer_.acc_begin("===>hard: apply cos setup");

    const std::complex<double> cabs = std::abs(coeff);
    const std::complex<double> factor = std::cos(time * cabs);
    cuDoubleComplex factor_gpu = make_cuDoubleComplex(factor.real(), factor.imag());

    std::pair<std::vector<int>, std::vector<int>> maps = evaluate_map_cpu(crea, anna, creb, annb);
    thrust::device_vector<int> d_first(maps.first.begin(), maps.first.end());
    thrust::device_vector<int> d_second(maps.second.begin(), maps.second.end());

    timer_.acc_end("===>hard: apply cos setup");

    timer_.acc_begin("===>hard: apply cos kernal");
    scale_elements_wrapper_complex(
        thrust::raw_pointer_cast(Cout.d_data().data()),
        thrust::raw_pointer_cast(d_first.data()),
        d_first.size(),
        thrust::raw_pointer_cast(d_second.data()),
        d_second.size(),
        nbeta_strs_,
        factor_gpu);
    timer_.acc_end("===>hard: apply cos kernal");
}

int FCIComputerGPU::isolate_number_operators_cpu(
    const std::vector<int>& cre,
    const std::vector<int>& ann,
    std::vector<int>& crework,
    std::vector<int>& annwork,
    std::vector<int>& number)
{
    /// TODO: Implement separate CPU and GPU versions of this function
    // cpu_error();

    int par = 0;
    for (int current : cre) {
        if (std::find(ann.begin(), ann.end(), current) != ann.end()) {
            auto index1 = std::find(crework.begin(), crework.end(), current);
            auto index2 = std::find(annwork.begin(), annwork.end(), current);
            par += static_cast<int>(crework.size()) - (index1 - crework.begin() + 1) + (index2 - annwork.begin());

            crework.erase(index1);
            annwork.erase(index2);
            number.push_back(current);
        }
    }
    return par;
}


/// NOTE: Cin should be const, changing for now
/// NOTE: Cin is actually unused, should not be an arg.
void FCIComputerGPU::evolve_individual_nbody_easy_cpu(
    const std::complex<double> time,
    const std::complex<double> coeff,
    TensorGPU& Cin,
    TensorGPU& Cout,
    const std::vector<int>& crea,
    const std::vector<int>& anna,
    const std::vector<int>& creb,
    const std::vector<int>& annb,
    const PrecompTuple* precomp)
{
    /// TODO: Implement seperate CPU and GPU versions of this function

    std::complex<double> factor = std::exp(-time * std::real(coeff) * std::complex<double>(0.0, 1.0));
    cuDoubleComplex factor_gpu = make_cuDoubleComplex(factor.real(), factor.imag());

    /// Optionally skip the on-the-fly device a/b-target idx formation

    if(precomp){
        timer_.acc_begin("==>easy: scale elements kernel");

        scale_elements_wrapper_complex(
            thrust::raw_pointer_cast(Cout.d_data().data()),
            thrust::raw_pointer_cast(std::get<2>(*precomp).data()),
            std::get<2>(*precomp).size(),
            thrust::raw_pointer_cast(std::get<4>(*precomp).data()),
            std::get<4>(*precomp).size(),
            nbeta_strs_,
            factor_gpu);

        timer_.acc_end("==>easy: scale elements kernel");

    } else {
        timer_.acc_begin("==>easy: setup");

        std::pair<std::vector<int>, std::vector<int>> maps = evaluate_map_number_cpu(anna, annb);

        thrust::device_vector<int> d_first(maps.first.begin(), maps.first.end());
        thrust::device_vector<int> d_second(maps.second.begin(), maps.second.end());

        timer_.acc_end("==>easy: setup");

        timer_.acc_begin("==>easy: scale elements kernel");

        scale_elements_wrapper_complex(
            thrust::raw_pointer_cast(Cout.d_data().data()),
            thrust::raw_pointer_cast(d_first.data()),
            d_first.size(),
            thrust::raw_pointer_cast(d_second.data()),
            d_second.size(),
            nbeta_strs_,
            factor_gpu);

        timer_.acc_end("==>easy: scale elements kernel");
    }
}

/// NOTE: Cin should be const, changing for now
/// NOTE: Cin is actually unused, should not be an arg.
template<class Precomp>
void FCIComputerGPU::evolve_individual_nbody_easy_gpu(
    const std::complex<double> time,
    const std::complex<double> coeff,
    TensorGPU& Cout,
    const std::vector<int>& crea,
    const std::vector<int>& anna,
    const std::vector<int>& creb,
    const std::vector<int>& annb,
    const Precomp* precomp)
{
    /// TODO: Implement seperate CPU and GPU versions of this function

    int n_a = anna.size();
    int n_b = annb.size();

    // std::cout << "\n  ====> n_a <==== " << n_a << std::endl;
    // std::cout << "\n  ====> n_b <==== " << n_b << std::endl;

    int power = n_a * (n_a - 1) / 2 + n_b * (n_b - 1) / 2;

    // std::cout << "\n  ====> power <==== " << power << std::endl;

    std::complex<double> prefactor = coeff * std::pow(-1, power);

    // std::cout << "\n  ====> prefactor <==== " << prefactor << std::endl;

    std::complex<double> factor = std::exp(-time * std::real(prefactor) * std::complex<double>(0.0, 1.0));

    // std::cout << "\n  ====> factor <==== " << factor << std::endl;

    cuDoubleComplex factor_gpu = make_cuDoubleComplex(factor.real(), factor.imag());

    // std::cout << "\n  ====> factor_gpu <==== (" << factor_gpu.x << ", " << factor_gpu.y << ")" << std::endl;

    /// Optionally skip the on-the-fly device a/b-target idx formation

    if(precomp){
        auto runtime_matches_compiletime = [&] {
            if constexpr (std::is_same_v<Precomp, PrecompTuple>) {
                return (data_type_ == std::string("complex"));
            } else if constexpr (std::is_same_v<Precomp, PrecompTupleReal>) {
                return (data_type_ == std::string("real"));
            } else {
                return false;
            }
        };
        if (!runtime_matches_compiletime()) {
            throw std::runtime_error("evolve_individual_nbody_easy_gpu: data_type_/Precomp mismatch.");
        }

        timer_.acc_begin("==>easy: scale elements kernel (precomp)");

        if constexpr (std::is_same_v<Precomp, PrecompTuple>) {
            // Complex path
            auto const& first  = std::get<2>(*precomp);
            auto const& second = std::get<4>(*precomp);

            // Copy device vectors to host for printing
            thrust::host_vector<int> first_host = first;
            thrust::host_vector<int> second_host = second;

            // std::cout << "\n  ====> first (size=" << first_host.size() << ") <====" << std::endl;
            // for (size_t i = 0; i < first_host.size(); ++i) {
            //     std::cout << "  [" << i << "] = " << first_host[i] << std::endl;
            // }

            // std::cout << "\n  ====> second (size=" << second_host.size() << ") <====" << std::endl;
            // for (size_t i = 0; i < second_host.size(); ++i) {
            //     std::cout << "  [" << i << "] = " << second_host[i] << std::endl;
            // }

            scale_elements_wrapper_complex(
                thrust::raw_pointer_cast(Cout.d_data().data()),
                thrust::raw_pointer_cast(first.data()),
                first.size(),
                thrust::raw_pointer_cast(second.data()),
                second.size(),
                nbeta_strs_,
                factor_gpu);

        } else if constexpr (std::is_same_v<Precomp, PrecompTupleReal>) {
            // Real path
            auto const& first  = std::get<2>(*precomp);
            auto const& second = std::get<4>(*precomp);

            scale_elements_wrapper_real(
                thrust::raw_pointer_cast(Cout.d_re_data().data()),
                thrust::raw_pointer_cast(first.data()),
                first.size(),
                thrust::raw_pointer_cast(second.data()),
                second.size(),
                nbeta_strs_,
                factor_gpu.x); // Only real component for real data

        } else {
            throw std::runtime_error("evolve_individual_nbody_easy_gpu: Unsupported Precomp type.");
        }

        timer_.acc_end("==>easy: scale elements kernel (precomp)");

    } else {
        if (data_type_ == "complex") {

            timer_.acc_begin("==>easy: setup");

            std::pair<std::vector<int>, std::vector<int>> maps = evaluate_map_number_cpu(anna, annb);

            thrust::device_vector<int> d_first(maps.first.begin(), maps.first.end());
            thrust::device_vector<int> d_second(maps.second.begin(), maps.second.end());

            timer_.acc_end("==>easy: setup");

            timer_.acc_begin("==>easy: scale elements kernel");

            scale_elements_wrapper_complex(
                thrust::raw_pointer_cast(Cout.d_data().data()),
                thrust::raw_pointer_cast(d_first.data()),
                d_first.size(),
                thrust::raw_pointer_cast(d_second.data()),
                d_second.size(),
                nbeta_strs_,
                factor_gpu);

            timer_.acc_end("==>easy: scale elements kernel");

        } else if (data_type_ == "real") {

            timer_.acc_begin("==>easy: setup");

            std::pair<std::vector<int>, std::vector<int>> maps = evaluate_map_number_cpu(anna, annb);

            thrust::device_vector<int> d_first(maps.first.begin(), maps.first.end());
            thrust::device_vector<int> d_second(maps.second.begin(), maps.second.end());

            timer_.acc_end("==>easy: setup");

            timer_.acc_begin("==>easy: scale elements kernel");

            scale_elements_wrapper_real(
                thrust::raw_pointer_cast(Cout.d_re_data().data()),
                thrust::raw_pointer_cast(d_first.data()),
                d_first.size(),
                thrust::raw_pointer_cast(d_second.data()),
                d_second.size(),
                nbeta_strs_,
                factor_gpu.x); // Only real component for real data

            timer_.acc_end("==>easy: scale elements kernel");

        } else {
            throw std::runtime_error("evolve_individual_nbody_easy_gpu: Unknown data_type_.");
        }
    }
}

/// NOTE: Cin should be const, changing for now
void FCIComputerGPU::evolve_individual_nbody_hard_cpu(
    const std::complex<double> time,
    const std::complex<double> coeff,
    TensorGPU& Cin,
    TensorGPU& Cout,
    const std::vector<int>& crea,
    const std::vector<int>& anna,
    const std::vector<int>& creb,
    const std::vector<int>& annb,
    const PrecompTuple* precomp)
{
    /// TODO: Implement seperate CPU and GPU versions of this function
    // cpu_error();

    timer_.acc_begin("==>hard: setup");
    std::vector<int> dagworka(crea);
    std::vector<int> dagworkb(creb);
    std::vector<int> undagworka(anna);
    std::vector<int> undagworkb(annb);
    std::vector<int> numbera;
    std::vector<int> numberb;

    int parity = 0;
    parity += isolate_number_operators_cpu(
        crea,
        anna,
        dagworka,
        undagworka,
        numbera);

    parity += isolate_number_operators_cpu(
        creb,
        annb,
        dagworkb,
        undagworkb,
        numberb);

    std::complex<double> ncoeff = coeff * std::pow(-1.0, parity);
    std::complex<double> absol = std::abs(ncoeff);
    std::complex<double> sinfactor = std::sin(time * absol) / absol;

    std::vector<int> numbera_dagworka(numbera.begin(), numbera.end());
    numbera_dagworka.insert(numbera_dagworka.end(), dagworka.begin(), dagworka.end());

    std::vector<int> numberb_dagworkb(numberb.begin(), numberb.end());
    numberb_dagworkb.insert(numberb_dagworkb.end(), dagworkb.begin(), dagworkb.end());

    std::vector<int> numbera_undagworka(numbera.begin(), numbera.end());
    numbera_undagworka.insert(numbera_undagworka.end(), undagworka.begin(), undagworka.end());

    std::vector<int> numberb_undagworkb(numberb.begin(), numberb.end());
    numberb_undagworkb.insert(numberb_undagworkb.end(), undagworkb.begin(), undagworkb.end());

    int phase = std::pow(-1, (crea.size() + anna.size()) * (creb.size() + annb.size()));
    std::complex<double> work_cof = std::conj(coeff) * static_cast<double>(phase) * std::complex<double>(0.0, -1.0);

    timer_.acc_end("==>hard: setup");

    if(precomp){

        // timer_.acc_begin("=>copy in Cin <- C_");
        // Cin.copy_in_gpu(C_);
        // timer_.acc_end("=>copy in Cin <- C_");


        timer_.acc_begin("=>gather in Cin <- C_");

        // TODO: Consider custom kernel implementaiton of the gather copy
        // TODO: Try a gather for both sets of indicies (dag+undag sources) at once

        // new funciton that will selectivly copy in
        Cin.gather_in_2D_gpu(
            Cout,
            std::get<2>(*precomp), //sourcea_dag,
            std::get<4>(*precomp) //sourceb_dag,
            );

        Cin.gather_in_2D_gpu(
            Cout,
            std::get<3>(*precomp), //sourcea_undag,
            std::get<5>(*precomp) // sourceb_undag,
            );

        timer_.acc_end("=>gather in Cin <- C_");

        const std::complex<double> cabs = std::abs(ncoeff);
        const std::complex<double> factor = std::cos(time * cabs);
        cuDoubleComplex factor_gpu = make_cuDoubleComplex(factor.real(), factor.imag());

        timer_.acc_begin("===>hard: apply cos kernal");

        scale_elements_wrapper_complex(
            thrust::raw_pointer_cast(Cout.d_data().data()),
            thrust::raw_pointer_cast(std::get<2>(*precomp).data()),
            std::get<2>(*precomp).size(),
            thrust::raw_pointer_cast(std::get<4>(*precomp).data()),
            std::get<4>(*precomp).size(),
            nbeta_strs_,
            factor_gpu);

        scale_elements_wrapper_complex(
            thrust::raw_pointer_cast(Cout.d_data().data()),
            thrust::raw_pointer_cast(std::get<3>(*precomp).data()),
            std::get<3>(*precomp).size(),
            thrust::raw_pointer_cast(std::get<5>(*precomp).data()),
            std::get<5>(*precomp).size(),
            nbeta_strs_,
            factor_gpu);

        timer_.acc_end("===>hard: apply cos kernal");

        timer_.acc_begin("===>hard nbody acc kernel");


        if ((std::get<3>(*precomp).size() != std::get<2>(*precomp).size()) or (std::get<2>(*precomp).size() != std::get<6>(*precomp).size())) {
            throw std::runtime_error("The sizes of atarget, asource, and aparity must be the same.");
        }

        if ((std::get<5>(*precomp).size() != std::get<4>(*precomp).size()) or (std::get<4>(*precomp).size() != std::get<8>(*precomp).size())) {
            throw std::runtime_error("The sizes of atarget, asource, and aparity must be the same.");
        }

        std::complex<double> coeff_dag = work_cof * sinfactor;

        cuDoubleComplex cu_coeff_dag = make_cuDoubleComplex(coeff_dag.real(), coeff_dag.imag());

        if(std::get<2>(*precomp).size() != 0 and std::get<4>(*precomp).size() != 0) {

            apply_individual_nbody1_accumulate_wrapper(
                cu_coeff_dag,
                thrust::raw_pointer_cast(Cin.read_d_data().data()),
                thrust::raw_pointer_cast(Cout.d_data().data()),
                thrust::raw_pointer_cast(std::get<2>(*precomp).data()),
                thrust::raw_pointer_cast(std::get<3>(*precomp).data()),
                thrust::raw_pointer_cast(std::get<6>(*precomp).data()),
                thrust::raw_pointer_cast(std::get<4>(*precomp).data()),
                thrust::raw_pointer_cast(std::get<5>(*precomp).data()),
                thrust::raw_pointer_cast(std::get<8>(*precomp).data()),
                nbeta_strs_,
                std::get<2>(*precomp).size(),
                std::get<4>(*precomp).size(),
                Cin.size() * sizeof(cuDoubleComplex));
        }

        cudaError_t error1 = cudaGetLastError();
        if (error1 != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error1) << std::endl;
            throw std::runtime_error("Failed to execute the apply_individual_nbody1_accumulate operation on the GPU.");
        }

        if ((std::get<2>(*precomp).size() != std::get<3>(*precomp).size()) or (std::get<3>(*precomp).size() != std::get<7>(*precomp).size())) {
            throw std::runtime_error("The sizes of atarget, asource, and aparity must be the same.");
        }

        if ((std::get<4>(*precomp).size() != std::get<5>(*precomp).size()) or (std::get<5>(*precomp).size() != std::get<9>(*precomp).size())) {
            throw std::runtime_error("The sizes of atarget, asource, and aparity must be the same.");
        }

        std::complex<double> coeff_undag = coeff * std::complex<double>(0.0, -1.0) * sinfactor;

        cuDoubleComplex cu_coeff_undag = make_cuDoubleComplex(coeff_undag.real(), coeff_undag.imag());

        if(std::get<3>(*precomp).size() != 0 and std::get<3>(*precomp).size() != 0) {

            apply_individual_nbody1_accumulate_wrapper(
                cu_coeff_undag,
                thrust::raw_pointer_cast(Cin.read_d_data().data()),
                thrust::raw_pointer_cast(Cout.d_data().data()),
                thrust::raw_pointer_cast(std::get<3>(*precomp).data()),
                thrust::raw_pointer_cast(std::get<2>(*precomp).data()),
                thrust::raw_pointer_cast(std::get<7>(*precomp).data()),
                thrust::raw_pointer_cast(std::get<5>(*precomp).data()),
                thrust::raw_pointer_cast(std::get<4>(*precomp).data()),
                thrust::raw_pointer_cast(std::get<9>(*precomp).data()),
                nbeta_strs_,
                std::get<3>(*precomp).size(),
                std::get<5>(*precomp).size(),
                Cin.size() * sizeof(cuDoubleComplex));
        }


        cudaError_t error2 = cudaGetLastError();
        if (error2 != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error2) << std::endl;
            throw std::runtime_error("Failed to execute the apply_individual_nbody1_accumulate operation on the GPU.");
        }

        timer_.acc_end("===>hard nbody acc kernel");


    } else {

        timer_.acc_begin("=>copy in Cin <- C_");
        Cin.copy_in_gpu(C_);
        timer_.acc_end("=>copy in Cin <- C_");

        // std::cout << "\n Cout Before Cos Application Thrust \n" << Cout.str(true, true) << std::endl;

        timer_.acc_begin("==>hard: apply_cos_inplace");
        apply_cos_inplace_cpu(
            time,
            ncoeff,
            numbera_dagworka,
            undagworka,
            numberb_dagworkb,
            undagworkb,
            Cout);

        // std::cout << "\n Cout After 1st Cos Application Thrust \n" << Cout.str(true, true) << std::endl;

        // std::cout << "\n Cout Before 2nd Cos Application Thrust \n" << Cout.str(true, true) << std::endl;

        apply_cos_inplace_cpu(
            time,
            ncoeff,
            numbera_undagworka,
            dagworka,
            numberb_undagworkb,
            dagworkb,
            Cout);

        timer_.acc_end("==>hard: apply_cos_inplace");

        // std::cout << "\n Cout After 2nd Cos Application Thrust \n" << Cout.str(true, true) << std::endl;

        timer_.acc_begin("==>hard: apply_individual_nbody_accumulate_gpu");
        apply_individual_nbody_accumulate_gpu(
            work_cof * sinfactor,
            Cin,
            Cout,
            anna,
            crea,
            annb,
            creb);

        // std::cout << "\n Cout After First Accumulate Application Thrust \n" << Cout.str(true, true) << std::endl;

        apply_individual_nbody_accumulate_gpu(
            coeff * std::complex<double>(0.0, -1.0) * sinfactor,
            Cin,
            Cout,
            crea,
            anna,
            creb,
            annb);

        timer_.acc_end("==>hard: apply_individual_nbody_accumulate_gpu");
    }

    // std::cout << "\n Cout After Second Accumulate Application Thrust \n" << Cout.str(true, true) << std::endl;
}

template<class Precomp>
void FCIComputerGPU::evolve_individual_nbody_hard_gpu(
    const std::complex<double> time,
    const std::complex<double> coeff,
    TensorGPU& Cout,
    const std::vector<int>& crea,
    const std::vector<int>& anna,
    const std::vector<int>& creb,
    const std::vector<int>& annb,
    const Precomp* precomp)
{
    /// TODO: Implement seperate CPU and GPU versions of this function
    // cpu_error();

    timer_.acc_begin("==>hard: setup");
    std::vector<int> dagworka(crea);
    std::vector<int> dagworkb(creb);
    std::vector<int> undagworka(anna);
    std::vector<int> undagworkb(annb);
    std::vector<int> numbera;
    std::vector<int> numberb;

    int parity = 0;
    parity += isolate_number_operators_cpu(
        crea,
        anna,
        dagworka,
        undagworka,
        numbera);

    parity += isolate_number_operators_cpu(
        creb,
        annb,
        dagworkb,
        undagworkb,
        numberb);

    std::complex<double> ncoeff = coeff * std::pow(-1.0, parity);
    std::complex<double> absol = std::abs(ncoeff);
    std::complex<double> sinfactor = std::sin(time * absol) / absol;

    int phase = std::pow(-1, (crea.size() + anna.size()) * (creb.size() + annb.size()));
    std::complex<double> work_cof = std::conj(coeff) * static_cast<double>(phase) * std::complex<double>(0.0, -1.0);

    const std::complex<double> cabs = std::abs(ncoeff);
    const std::complex<double> factor = std::cos(time * cabs);

    std::complex<double> acc_coeff1 = work_cof * sinfactor;
    std::complex<double> acc_coeff2 = coeff * std::complex<double>(0.0, -1.0) * sinfactor;

    timer_.acc_end("==>hard: setup");

    if(precomp){
        auto runtime_matches_compiletime = [&] {
            if constexpr (std::is_same_v<Precomp, PrecompTuple>) {
                return (data_type_ == std::string("complex"));
            } else if constexpr (std::is_same_v<Precomp, PrecompTupleReal>) {
                return (data_type_ == std::string("real"));
            } else {
                return false;
            }
        };
        if (!runtime_matches_compiletime()) {
            throw std::runtime_error("evolve_individual_nbody_hard_gpu: data_type_/Precomp mismatch.");
        }

        // DEBUG: Source Target Parity vectors
        // std::cout << "sourcea1: ";
        //     thrust::copy(std::get<2>(*precomp).begin(), std::get<2>(*precomp).end(), std::ostream_iterator<int>(std::cout, " "));
        // std::cout << "\n";

        // std::cout << "targeta1: ";
        //     thrust::copy(std::get<3>(*precomp).begin(), std::get<3>(*precomp).end(), std::ostream_iterator<int>(std::cout, " "));
        // std::cout << "\n";

        // std::cout << "paritya1: ";
        //     // Note: paritya type depends on Precomp template parameter
        //     if constexpr (std::is_same_v<Precomp, PrecompTuple>) {
        //         // Complex case: cuDoubleComplex
        //         thrust::host_vector<cuDoubleComplex> paritya_host(std::get<6>(*precomp));
        //         for (const auto& val : paritya_host) {
        //             std::cout << "(" << val.x << "," << val.y << ") ";
        //         }
        //     } else if constexpr (std::is_same_v<Precomp, PrecompTupleReal>) {
        //         // Real case: double
        //         thrust::host_vector<double> paritya_host(std::get<6>(*precomp));
        //         for (const auto& val : paritya_host) {
        //             std::cout << val << " ";
        //         }
        //     }
        // std::cout << "\n";

        // std::cout << "State Vec Before: \n" << Cout.str(true, true) << std::endl;

        if constexpr (std::is_same_v<Precomp, PrecompTuple>) {

            timer_.acc_begin("===>hard nbody given kernel");

            // condition here for all row or all col cases
            if (crea.size() == 0 && creb.size() > 0) {
                timer_.acc_begin("===>hard nbody given kernel - col case");

                inplace_givens_update_complex_tiled_wrapper(
                    32,
                    thrust::raw_pointer_cast(Cout.d_data().data()),
                    thrust::raw_pointer_cast(std::get<2>(*precomp).data()), // sourcea1 (dag)
                    thrust::raw_pointer_cast(std::get<3>(*precomp).data()), // targeta1
                    thrust::raw_pointer_cast(std::get<6>(*precomp).data()), // paritya1
                    thrust::raw_pointer_cast(std::get<7>(*precomp).data()), // paritya2
                    thrust::raw_pointer_cast(std::get<4>(*precomp).data()),  // sourceb1 (dag)
                    thrust::raw_pointer_cast(std::get<5>(*precomp).data()), // targetb1
                    thrust::raw_pointer_cast(std::get<8>(*precomp).data()), // parityb1
                    thrust::raw_pointer_cast(std::get<9>(*precomp).data()), // parityb2
                    std::get<2>(*precomp).size(), // nalpha
                    std::get<4>(*precomp).size(), // nb
                    nbeta_strs_,
                    make_cuDoubleComplex(factor.real(), factor.imag()),
                    make_cuDoubleComplex(acc_coeff1.real(), acc_coeff1.imag()),
                    make_cuDoubleComplex(acc_coeff2.real(), acc_coeff2.imag()));

                timer_.acc_end("===>hard nbody given kernel - col case");
                timer_.acc_end("===>hard nbody given kernel");
                return;
            } else if (creb.size() == 0 && crea.size() > 0) {
                timer_.acc_begin("===>hard nbody given kernel - row case");

                inplace_givens_update_complex_rows_wrapper(
                    thrust::raw_pointer_cast(Cout.d_data().data()),  // Cout
                    thrust::raw_pointer_cast(std::get<2>(*precomp).data()),  // const int* sourcea1 (dag)
                    thrust::raw_pointer_cast(std::get<3>(*precomp).data()), // const int* targeta1
                    thrust::raw_pointer_cast(std::get<6>(*precomp).data()), // const cuDoubleComplex* paritya1
                    thrust::raw_pointer_cast(std::get<7>(*precomp).data()), // const cuDoubleComplex* paritya2
                    // thrust::raw_pointer_cast(std::get<8>(*precomp).data()), // const cuDoubleComplex* parityb1
                    // thrust::raw_pointer_cast(std::get<9>(*precomp).data()), // const cuDoubleComplex* parityb2
                    std::get<2>(*precomp).size(), // int na
                    nbeta_strs_,                  // long long nbeta_strs_
                    make_cuDoubleComplex(factor.real(), factor.imag()), // cuDoubleComplex cos_factor
                    make_cuDoubleComplex(acc_coeff1.real(), acc_coeff1.imag()),
                    make_cuDoubleComplex(acc_coeff2.real(), acc_coeff2.imag()));

                timer_.acc_end("===>hard nbody given kernel - row case");
                timer_.acc_end("===>hard nbody given kernel");
                return;
            }

            timer_.acc_begin("===>hard nbody given kernel - general case");

            inplace_givens_update_complex_tiled_wrapper(
                32,
                thrust::raw_pointer_cast(Cout.d_data().data()),
                thrust::raw_pointer_cast(std::get<2>(*precomp).data()), // sourcea1 (dag)
                thrust::raw_pointer_cast(std::get<3>(*precomp).data()), // targeta1
                thrust::raw_pointer_cast(std::get<6>(*precomp).data()), // paritya1
                thrust::raw_pointer_cast(std::get<7>(*precomp).data()), // paritya2
                thrust::raw_pointer_cast(std::get<4>(*precomp).data()),  // sourceb1 (dag)
                thrust::raw_pointer_cast(std::get<5>(*precomp).data()), // targetb1
                thrust::raw_pointer_cast(std::get<8>(*precomp).data()), // parityb1
                thrust::raw_pointer_cast(std::get<9>(*precomp).data()), // parityb2
                std::get<2>(*precomp).size(), // nalpha
                std::get<4>(*precomp).size(), // nb
                nbeta_strs_,
                make_cuDoubleComplex(factor.real(), factor.imag()),
                make_cuDoubleComplex(acc_coeff1.real(), acc_coeff1.imag()),
                make_cuDoubleComplex(acc_coeff2.real(), acc_coeff2.imag()));

            timer_.acc_end("===>hard nbody given kernel - general case");

            cudaError_t error2 = cudaGetLastError();

            if (error2 != cudaSuccess) {
                std::cerr << "CUDA error: " << cudaGetErrorString(error2) << std::endl;
                    throw std::runtime_error("Failed to execute the apply_individual_nbody1_accumulate operation on the GPU.");
                }

            timer_.acc_end("===>hard nbody given kernel");

        } else if constexpr (std::is_same_v<Precomp, PrecompTupleReal>) {
            // Real Case

            timer_.acc_begin("===>hard nbody given kernel");

            // condition here for all row or all col cases
            if (crea.size() == 0 && creb.size() > 0) {
                timer_.acc_begin("===>hard nbody given kernel - beta-only rowmajor");

                // inplace_givens_update_real_tiled_wrapper(
                //     32,
                //     thrust::raw_pointer_cast(Cout.d_re_data().data()),
                //     thrust::raw_pointer_cast(std::get<2>(*precomp).data()), // sourcea1 (dag)
                //     thrust::raw_pointer_cast(std::get<3>(*precomp).data()), // targeta1
                //     thrust::raw_pointer_cast(std::get<6>(*precomp).data()), // paritya1
                //     thrust::raw_pointer_cast(std::get<7>(*precomp).data()), // paritya2
                //     thrust::raw_pointer_cast(std::get<4>(*precomp).data()),  // sourceb1 (dag)
                //     thrust::raw_pointer_cast(std::get<5>(*precomp).data()), // targetb1
                //     thrust::raw_pointer_cast(std::get<8>(*precomp).data()), // parityb1
                //     thrust::raw_pointer_cast(std::get<9>(*precomp).data()), // parityb2
                //     std::get<2>(*precomp).size(), // nalpha
                //     std::get<4>(*precomp).size(), // nb
                //     nbeta_strs_,
                //     factor.real(),
                //     acc_coeff1.real(),
                //     acc_coeff2.real());

                // Beta-only path: no alpha excitations, only beta (column) pairs.
                //
                // We use the row-major tiled beta-only kernel instead of the old
                // column-strided kernel.  In the old kernel each warp stepped down
                // column sb1 with stride nbeta_strs_, which is non-coalesced for a
                // row-major matrix.  The new kernel maps threadIdx.x across beta
                // pairs and threadIdx.y across rows; all threads in a warp stay in
                // the same row, so their accesses to d_Cout[row*nbeta_strs_ + sb1[tx]]
                // are nearly contiguous when sourceb1 is sorted → coalesced.
                //
                // Launch shape: block(32, 8) = 256 threads; grid tiles both axes,
                // giving ~ceil(nb/32)*ceil(na/8) blocks vs. the old nb*ceil(na/1024).
                // For small nb this is a significant increase in block count and
                // hence better SM occupancy / latency hiding.

                inplace_givens_update_real_beta_only_rowmajor_wrapper(
                    thrust::raw_pointer_cast(Cout.d_re_data().data()),
                    thrust::raw_pointer_cast(std::get<4>(*precomp).data()),  // const int* sourceb1
                    thrust::raw_pointer_cast(std::get<5>(*precomp).data()),  // const int* targetb1
                    thrust::raw_pointer_cast(std::get<8>(*precomp).data()),  // const double* parityb1
                    thrust::raw_pointer_cast(std::get<9>(*precomp).data()),  // const double* parityb2
                    static_cast<int>(std::get<4>(*precomp).size()),          // nb
                    static_cast<long long>(nalfa_strs_),                           // nalpha_strs_
                    static_cast<long long>(nbeta_strs_),                           // nbeta_strs_
                    factor.real(),
                    acc_coeff1.real(),
                    acc_coeff2.real());

                timer_.acc_end("===>hard nbody given kernel - beta-only rowmajor");
                timer_.acc_end("===>hard nbody given kernel");
                return;
            } else if (creb.size() == 0 && crea.size() > 0) {
                timer_.acc_begin("===>hard nbody given kernel - row case");

                inplace_givens_update_real_rows_wrapper(
                    thrust::raw_pointer_cast(Cout.d_re_data().data()),
                    thrust::raw_pointer_cast(std::get<2>(*precomp).data()),  // const int* sourcea1 (dag)
                    thrust::raw_pointer_cast(std::get<3>(*precomp).data()), // const int* targeta1
                    thrust::raw_pointer_cast(std::get<6>(*precomp).data()), // const cuDoubleComplex* paritya1
                    thrust::raw_pointer_cast(std::get<7>(*precomp).data()), // const cuDoubleComplex* paritya2
                    std::get<2>(*precomp).size(), // nalpha
                    nbeta_strs_,
                    factor.real(),
                    acc_coeff1.real(),
                    acc_coeff2.real());

                timer_.acc_end("===>hard nbody given kernel - row case");
                timer_.acc_end("===>hard nbody given kernel");
                return;
            }

            timer_.acc_begin("===>hard nbody given kernel - general case");

            inplace_givens_update_real_tiled_wrapper(
                32,
                thrust::raw_pointer_cast(Cout.d_re_data().data()),
                thrust::raw_pointer_cast(std::get<2>(*precomp).data()), // sourcea1 (dag)
                thrust::raw_pointer_cast(std::get<3>(*precomp).data()), // targeta1
                thrust::raw_pointer_cast(std::get<6>(*precomp).data()), // paritya1
                thrust::raw_pointer_cast(std::get<7>(*precomp).data()), // paritya2
                thrust::raw_pointer_cast(std::get<4>(*precomp).data()),  // sourceb1 (dag)
                thrust::raw_pointer_cast(std::get<5>(*precomp).data()), // targetb1
                thrust::raw_pointer_cast(std::get<8>(*precomp).data()), // parityb1
                thrust::raw_pointer_cast(std::get<9>(*precomp).data()), // parityb2
                std::get<2>(*precomp).size(), // nalpha
                std::get<4>(*precomp).size(), // nb
                nbeta_strs_,
                factor.real(),
                acc_coeff1.real(),
                acc_coeff2.real());

            timer_.acc_end("===>hard nbody given kernel - general case");

            cudaError_t error2 = cudaGetLastError();

            if (error2 != cudaSuccess) {
                std::cerr << "CUDA error: " << cudaGetErrorString(error2) << std::endl;
                    throw std::runtime_error("Failed to execute the apply_individual_nbody1_accumulate operation on the GPU.");
                }

            timer_.acc_end("===>hard nbody given kernel");

        } else {
            throw std::runtime_error("evolve_individual_nbody_hard_gpu: Unsupported Precomp type.");
        }

        // std::cout << "State Vec After: \n" << Cout.str(true, true) << std::endl;

    } else {
        if (data_type_ == "complex") {
            // Complex Case

            // No precomp provided: need to compute on the fly
            int counta1 = 0;
            int counta2 = 0;
            int countb1 = 0;
            int countb2 = 0;

            // DAG mapping: annihilators as dag, creators as undag
            graph_.make_mapping_each_otf_gpu_complex(
                true, // const bool is_alpha,
                anna,
                crea,
                &counta1,
                sourcea_gpu_,
                targeta_gpu_,
                paritya_gpu_);

            // std::cout << "sourcea1 " << sourcea_gpu_.str() << std::endl;  // device_vector has no .str() method

            // UNDAG mapping: creators as dag, annihilators as undag
            graph_.make_mapping_each_otf_gpu_complex(
                true, // const bool is_alpha,
                crea,
                anna,
                &counta2,
                sourcea_undag_gpu_,
                targeta_undag_gpu_,
                paritya_undag_gpu_);

            // DAG mapping: annihilators as dag, creators as undag
            graph_.make_mapping_each_otf_gpu_complex(
                false, // const bool is_alpha,
                annb,
                creb,
                &countb1,
                sourceb_gpu_,
                targetb_gpu_,
                parityb_gpu_);

            // UNDAG mapping: creators as dag, annihilators as undag
            graph_.make_mapping_each_otf_gpu_complex(
                false, // const bool is_alpha,
                creb,
                annb,
                &countb2,
                sourceb_undag_gpu_,
                targetb_undag_gpu_,
                parityb_undag_gpu_);

            // DEBUG: Source Target Parity vectors
            // std::cout << "sourcea1: ";
            //     thrust::copy(sourcea_gpu_.begin(), sourcea_gpu_.end(), std::ostream_iterator<int>(std::cout, " "));
            // std::cout << "\n";

            // std::cout << "targeta1: ";
            //     thrust::copy(targeta_gpu_.begin(), targeta_gpu_.end(), std::ostream_iterator<int>(std::cout, " "));
            // std::cout << "\n";

            // std::cout << "paritya1: ";
            //     // Note: paritya contains cuDoubleComplex, copy to host first
            //     thrust::host_vector<cuDoubleComplex> paritya_host(paritya_gpu_);
            //     for (const auto& val : paritya_host) {
            //         std::cout << "(" << val.x << "," << val.y << ") ";
            //     }
            // std::cout << "\n";

            // std::cout << "State Vec Before: \n" << Cout.str(true, true) << std::endl;

            timer_.acc_begin("===>hard nbody given kernel");

            // condition here for all row or all col cases
            if (crea.size() == 0 && creb.size() > 0) {
                timer_.acc_begin("===>hard nbody given kernel - col case");

                inplace_givens_update_complex_tiled_wrapper(
                    32,
                    thrust::raw_pointer_cast(Cout.d_data().data()),
                    thrust::raw_pointer_cast(sourcea_gpu_.data()), // sourcea1 (dag)
                    thrust::raw_pointer_cast(targeta_gpu_.data()), // targeta1
                    thrust::raw_pointer_cast(paritya_gpu_.data()), // paritya1
                    thrust::raw_pointer_cast(paritya_undag_gpu_.data()), // paritya2
                    thrust::raw_pointer_cast(sourceb_gpu_.data()),  // sourceb1 (dag)
                    thrust::raw_pointer_cast(targetb_gpu_.data()), // targetb1
                    thrust::raw_pointer_cast(parityb_gpu_.data()), // parityb1
                    thrust::raw_pointer_cast(parityb_undag_gpu_.data()), // parityb2
                    counta1, // nalpha
                    countb1, // nbeta
                    nbeta_strs_,
                    make_cuDoubleComplex(factor.real(), factor.imag()),
                    make_cuDoubleComplex(acc_coeff1.real(), acc_coeff1.imag()),
                    make_cuDoubleComplex(acc_coeff2.real(), acc_coeff2.imag()));

                timer_.acc_end("===>hard nbody given kernel - col case");
                timer_.acc_end("===>hard nbody given kernel");
                return;
            } else if (creb.size() == 0 && crea.size() > 0) {
                timer_.acc_begin("===>hard nbody given kernel - row case");

                inplace_givens_update_complex_rows_wrapper(
                    thrust::raw_pointer_cast(Cout.d_data().data()),  // Cout
                    thrust::raw_pointer_cast(sourcea_gpu_.data()),  // const int* sourcea1 (dag)
                    thrust::raw_pointer_cast(targeta_gpu_.data()), // const int* targeta1
                    thrust::raw_pointer_cast(paritya_gpu_.data()), // const cuDoubleComplex* paritya1
                    thrust::raw_pointer_cast(paritya_undag_gpu_.data()), // const cuDoubleComplex* paritya2
                    counta1,      // int na
                    nbeta_strs_,  // long long nbeta_strs_
                    make_cuDoubleComplex(factor.real(), factor.imag()),
                    make_cuDoubleComplex(acc_coeff1.real(), acc_coeff1.imag()),
                    make_cuDoubleComplex(acc_coeff2.real(), acc_coeff2.imag()));

                timer_.acc_end("===>hard nbody given kernel - row case");
                timer_.acc_end("===>hard nbody given kernel");
                return;
            }

            timer_.acc_begin("===>hard nbody given kernel - general case");

            inplace_givens_update_complex_tiled_wrapper(
                32,
                thrust::raw_pointer_cast(Cout.d_data().data()),
                thrust::raw_pointer_cast(sourcea_gpu_.data()), // sourcea1 (dag)
                thrust::raw_pointer_cast(targeta_gpu_.data()), // targeta1
                thrust::raw_pointer_cast(paritya_gpu_.data()), // paritya1
                thrust::raw_pointer_cast(paritya_undag_gpu_.data()), // paritya2
                thrust::raw_pointer_cast(sourceb_gpu_.data()),  // sourceb1 (dag)
                thrust::raw_pointer_cast(targetb_gpu_.data()), // targetb1
                thrust::raw_pointer_cast(parityb_gpu_.data()), // parityb1
                thrust::raw_pointer_cast(parityb_undag_gpu_.data()), // parityb2
                counta1, // nalpha
                countb1, // nbeta
                nbeta_strs_,
                make_cuDoubleComplex(factor.real(), factor.imag()),
                make_cuDoubleComplex(acc_coeff1.real(), acc_coeff1.imag()),
                make_cuDoubleComplex(acc_coeff2.real(), acc_coeff2.imag()));

            timer_.acc_end("===>hard nbody given kernel - general case");

            cudaError_t error2 = cudaGetLastError();

            if (error2 != cudaSuccess) {
                std::cerr << "CUDA error: " << cudaGetErrorString(error2) << std::endl;
                    throw std::runtime_error("Failed to execute the apply_individual_nbody1_accumulate operation on the GPU.");
                }

            timer_.acc_end("===>hard nbody given kernel");

            // std::cout << "State Vec After: \n" << Cout.str(true, true) << std::endl;
            // std::cout << "post kernel" << std::endl;
        } else if (data_type_ == "real") {
            // Real Case

            // No precomp provided: need to compute on the fly
            int counta1 = 0;
            int counta2 = 0;
            int countb1 = 0;
            int countb2 = 0;

            // DAG mapping: annihilators as dag, creators as undag
            graph_.make_mapping_each_otf_gpu_real(
                true, // const bool is_alpha,
                anna,
                crea,
                &counta1,
                sourcea_gpu_,
                targeta_gpu_,
                paritya_gpu_real_);

            // std::cout << "sourcea1 " << sourcea_gpu_.str() << std::endl;  // device_vector has no .str() method

            // UNDAG mapping: creators as dag, annihilators as undag
            graph_.make_mapping_each_otf_gpu_real(
                true, // const bool is_alpha,
                crea,
                anna,
                &counta2,
                sourcea_undag_gpu_,
                targeta_undag_gpu_,
                paritya_undag_gpu_real_);

            // DAG mapping: annihilators as dag, creators as undag
            graph_.make_mapping_each_otf_gpu_real(
                false, // const bool is_alpha,
                annb,
                creb,
                &countb1,
                sourceb_gpu_,
                targetb_gpu_,
                parityb_gpu_real_);

            // UNDAG mapping: creators as dag, annihilators as undag
            graph_.make_mapping_each_otf_gpu_real(
                false, // const bool is_alpha,
                creb,
                annb,
                &countb2,
                sourceb_undag_gpu_,
                targetb_undag_gpu_,
                parityb_undag_gpu_real_);

            // DEBUG: Source Target Parity vectors
            // std::cout << "sourcea1: ";
            //     thrust::copy(sourcea_gpu_.begin(), sourcea_gpu_.end(), std::ostream_iterator<int>(std::cout, " "));
            // std::cout << "\n";

            // std::cout << "targeta1: ";
            //     thrust::copy(targeta_gpu_.begin(), targeta_gpu_.end(), std::ostream_iterator<int>(std::cout, " "));
            // std::cout << "\n";

            // std::cout << "State Vec Before: \n" << Cout.str(true, true) << std::endl;

            timer_.acc_begin("===>hard nbody given kernel");

            // condition here for all row or all col cases
            if (crea.size() == 0 && creb.size() > 0) {
                timer_.acc_begin("===>hard nbody given kernel - col case");

                inplace_givens_update_real_tiled_wrapper(
                    32,
                    thrust::raw_pointer_cast(Cout.d_re_data().data()),
                    thrust::raw_pointer_cast(sourcea_gpu_.data()), // sourcea1 (dag)
                    thrust::raw_pointer_cast(targeta_gpu_.data()), // targeta1
                    thrust::raw_pointer_cast(paritya_gpu_real_.data()), // paritya1
                    thrust::raw_pointer_cast(paritya_undag_gpu_real_.data()), // paritya2
                    thrust::raw_pointer_cast(sourceb_gpu_.data()),  // sourceb1 (dag)
                    thrust::raw_pointer_cast(targetb_gpu_.data()), // targetb1
                    thrust::raw_pointer_cast(parityb_gpu_real_.data()), // parityb1
                    thrust::raw_pointer_cast(parityb_undag_gpu_real_.data()), // parityb2
                    counta1, // nalpha
                    countb1, // nbeta
                    nbeta_strs_,
                    factor.real(),
                    acc_coeff1.real(),
                    acc_coeff2.real());

                timer_.acc_end("===>hard nbody given kernel - col case");
                timer_.acc_end("===>hard nbody given kernel");
                return;
            } else if (creb.size() == 0 && crea.size() > 0) {
                timer_.acc_begin("===>hard nbody given kernel - row case");

                inplace_givens_update_real_rows_wrapper(
                    thrust::raw_pointer_cast(Cout.d_re_data().data()),  // Cout
                    thrust::raw_pointer_cast(sourcea_gpu_.data()),  // const int* sourcea1 (dag)
                    thrust::raw_pointer_cast(targeta_gpu_.data()), // const int* targeta1
                    thrust::raw_pointer_cast(paritya_gpu_real_.data()), // const cuDoubleComplex* paritya1
                    thrust::raw_pointer_cast(paritya_undag_gpu_real_.data()), // const cuDoubleComplex* paritya2
                    counta1,      // int nalpha
                    nbeta_strs_,  // long long nbeta_strs_
                    factor.real(),
                    acc_coeff1.real(),
                    acc_coeff2.real());

                timer_.acc_end("===>hard nbody given kernel - row case");
                timer_.acc_end("===>hard nbody given kernel");
                return;
            }

            timer_.acc_begin("===>hard nbody given kernel - general case");

            inplace_givens_update_real_tiled_wrapper(
                32,
                thrust::raw_pointer_cast(Cout.d_re_data().data()),
                thrust::raw_pointer_cast(sourcea_gpu_.data()), // sourcea1 (dag)
                thrust::raw_pointer_cast(targeta_gpu_.data()), // targeta1
                thrust::raw_pointer_cast(paritya_gpu_real_.data()), // paritya1
                thrust::raw_pointer_cast(paritya_undag_gpu_real_.data()), // paritya2
                thrust::raw_pointer_cast(sourceb_gpu_.data()),  // sourceb1 (dag)
                thrust::raw_pointer_cast(targetb_gpu_.data()), // targetb1
                thrust::raw_pointer_cast(parityb_gpu_real_.data()), // parityb1
                thrust::raw_pointer_cast(parityb_undag_gpu_real_.data()), // parityb2
                counta1, // nalpha
                countb1, // nbeta
                nbeta_strs_,
                factor.real(),
                acc_coeff1.real(),
                acc_coeff2.real());

            timer_.acc_end("===>hard nbody given kernel - general case");

            cudaError_t error2 = cudaGetLastError();

            if (error2 != cudaSuccess) {
                std::cerr << "CUDA error: " << cudaGetErrorString(error2) << std::endl;
                    throw std::runtime_error("Failed to execute the apply_individual_nbody1_accumulate operation on the GPU.");
                }

            timer_.acc_end("===>hard nbody given kernel");

        } else {
            throw std::runtime_error("evolve_individual_nbody_hard_gpu: Unsupported data_type_.");
        }
    }
}

/// NOTE: Cin should be const, changing for now
void FCIComputerGPU::evolve_individual_nbody_cpu(
    const std::complex<double> time,
    const SQOperator& sqop,
    TensorGPU& Cin,
    TensorGPU& Cout,
    const bool antiherm,
    const bool adjoint,
    const PrecompTuple* precomp)
{
    /// TODO: Implement seperate CPU and GPU versions of this function
    // cpu_error();

    if (sqop.terms().size() != 2) {
        std::cout << "This sqop has " << sqop.terms().size() << " terms." << std::endl;
        throw std::invalid_argument("Individual n-body code is called with multiple terms");
    }

    /// NICK: TODO, implement a hermitian check, at least for two term SQOperators
    // sqop.hermitian_check();

    timer_.acc_begin("=>evolve_individual_nbody_cpu(setup)");

    auto term = sqop.terms()[0];

    if(std::abs(std::get<0>(term)) < compute_threshold_){
        return;
    }

    if(adjoint){
        std::get<0>(term) *= -1.0;
    }

    if(antiherm){
        std::complex<double> onei(0.0, 1.0);
        std::get<0>(term) *= onei;
    }

    // TODO: Ask Nick about this early return
    if(std::get<1>(term).size()==0 && std::get<2>(term).size()==0){
        std::complex<double> twoi(0.0, -2.0);
        Cout.scale(std::exp(twoi * time * std::get<0>(term)));
        return;
    }

    std::vector<int> crea;
    std::vector<int> anna;
    std::vector<int> creb;
    std::vector<int> annb;

    for(size_t i = 0; i < std::get<1>(term).size(); i++){
        if(std::get<1>(term)[i]%2 == 0){
            crea.push_back(std::floor(std::get<1>(term)[i] / 2));
        } else {
            creb.push_back(std::floor(std::get<1>(term)[i] / 2));
        }
    }

    for(size_t i = 0; i < std::get<2>(term).size(); i++){
        if(std::get<2>(term)[i]%2 == 0){
            anna.push_back(std::floor(std::get<2>(term)[i] / 2));
        } else {
            annb.push_back(std::floor(std::get<2>(term)[i] / 2));
        }
    }

    std::vector<size_t> ops1(std::get<1>(term));
    std::vector<size_t> ops2(std::get<2>(term));
    ops1.insert(ops1.end(), ops2.begin(), ops2.end());

    int nswaps = parity_sort(ops1);

    std::complex<double> parity = std::pow(-1, nswaps);

    timer_.acc_end("=>evolve_individual_nbody_cpu(setup)");

    if (crea == anna && creb == annb) {
        // std::cout << "Made it to easy" << std::endl;

        timer_.acc_begin("=>evolve_individual_nbody_easy_cpu");

        evolve_individual_nbody_easy_cpu(
            time,
            parity * std::get<0>(term),
            Cin,
            Cout,
            crea,
            anna,
            creb,
            annb,
            precomp);

        timer_.acc_end("=>evolve_individual_nbody_easy_cpu");


    } else if (crea.size() == anna.size() && creb.size() == annb.size()) {
        // std::cout << "Made it to hard" << std::endl;

        timer_.acc_begin("=>evolve_individual_nbody_hard_cpu");

        evolve_individual_nbody_hard_cpu(
            time,
            parity * std::get<0>(term),
            Cin,
            Cout,
            crea,
            anna,
            creb,
            annb,
            precomp);

        timer_.acc_end("=>evolve_individual_nbody_hard_cpu");

    } else {
        throw std::invalid_argument("Evolved state must remain in spin and particle-number symmetry sector");
    }
}

/// NOTE: Cin should be const, changing for now
template<class Precomp>
void FCIComputerGPU::evolve_individual_nbody_gpu(
    const std::complex<double> time,
    const SQOperator& sqop,
    TensorGPU& Cout,
    const bool antiherm,
    const bool adjoint,
    const Precomp* precomp)
{

    // std::cout << "Entering evolve_individual_nbody_gpu" << std::endl;

    /// TODO: Implement seperate CPU and GPU versions of this function
    // cpu_error();

    if (sqop.terms().size() != 2) {
        std::cout << "This sqop has " << sqop.terms().size() << " terms." << std::endl;
        throw std::invalid_argument("Individual n-body code is called with multiple terms");
    }

    /// NICK: TODO, implement a hermitian check, at least for two term SQOperators
    // sqop.hermitian_check();

    timer_.acc_begin("=>evolve_individual_nbody_gpu(setup)");

    auto term = sqop.terms()[0];

    if(std::abs(std::get<0>(term)) < compute_threshold_){
        return;
    }

    if(adjoint){
        std::get<0>(term) *= -1.0;
    }

    if(antiherm){
        std::complex<double> onei(0.0, 1.0);
        std::get<0>(term) *= onei;
    }

    // TODO: Ask Nick about this early return
    if(std::get<1>(term).size()==0 && std::get<2>(term).size()==0){
        std::complex<double> twoi(0.0, -2.0);
        Cout.scale(std::exp(twoi * time * std::get<0>(term)));
        return;
    }

    std::vector<int> crea;
    std::vector<int> anna;
    std::vector<int> creb;
    std::vector<int> annb;

    for(size_t i = 0; i < std::get<1>(term).size(); i++){
        if(std::get<1>(term)[i]%2 == 0){
            crea.push_back(std::floor(std::get<1>(term)[i] / 2));
        } else {
            creb.push_back(std::floor(std::get<1>(term)[i] / 2));
        }
    }

    for(size_t i = 0; i < std::get<2>(term).size(); i++){
        if(std::get<2>(term)[i]%2 == 0){
            anna.push_back(std::floor(std::get<2>(term)[i] / 2));
        } else {
            annb.push_back(std::floor(std::get<2>(term)[i] / 2));
        }
    }

    std::vector<size_t> ops1(std::get<1>(term));
    std::vector<size_t> ops2(std::get<2>(term));
    ops1.insert(ops1.end(), ops2.begin(), ops2.end());

    int nswaps = parity_sort(ops1);

    std::complex<double> parity = std::pow(-1, nswaps);

    timer_.acc_end("=>evolve_individual_nbody_cpu(setup)");

    // DEBUG: source and target lists
    // std::cout << "crea: ";
    // for (const auto& val : crea) std::cout << val << " ";
    // std::cout << std::endl;
    // std::cout << "anna: ";
    // for (const auto& val : anna) std::cout << val << " ";
    // std::cout << std::endl;
    // std::cout << "creb: ";
    // for (const auto& val : creb) std::cout << val << " ";
    // std::cout << std::endl;
    // std::cout << "annb: ";
    // for (const auto& val : annb) std::cout << val << " ";
    // std::cout << std::endl;

    if (crea == anna && creb == annb) {
        // std::cout << "Made it to easy" << std::endl;

        // // DEBUG:
        // dump_step_header("[EASY] (scaling)", parity * std::get<0>(term), crea, anna, creb, annb, nswaps, parity);

        // // DEBUG: Optional: dump state just before
        // dump_tensor("  EASY: before", Cout);

        timer_.acc_begin("=>evolve_individual_nbody_easy_gpu");

        // std::cout << "Calling evolve_individual_nbody_easy_gpu" << std::endl;

        evolve_individual_nbody_easy_gpu(
            time,
            parity * 2.0 * std::get<0>(term), // TODO: Ask Nick about (* 2) to coeff?
            Cout,
            crea,
            anna,
            creb,
            annb,
            precomp);

        // // DEBUG: After
        // dump_tensor("  EASY: after", Cout);

        timer_.acc_end("=>evolve_individual_nbody_easy_gpu");

    } else if (crea.size() == anna.size() && creb.size() == annb.size()) {
        // std::cout << "Made it to hard" << std::endl;

        timer_.acc_begin("=>evolve_individual_nbody_hard_gpu");

        // // DEBUG:
        // dump_step_header("[HARD] (Givens)", parity * std::get<0>(term), crea, anna, creb, annb, nswaps, parity);
        // dump_tensor("  HARD: before", Cout);

        // std::cout << "Calling evolve_individual_nbody_hard_gpu" << std::endl;

        evolve_individual_nbody_hard_gpu(
            time,
            parity * std::get<0>(term),
            Cout,
            crea,
            anna,
            creb,
            annb,
            precomp);

        // // DEBUG: After
        // dump_tensor("  HARD: after", Cout);

        timer_.acc_end("=>evolve_individual_nbody_hard_gpu");

    } else {
        throw std::invalid_argument("Evolved state must remain in spin and particle-number symmetry sector");
    }

    // std::cout << "tensor after evolution:\n" << Cout.str(true, true) << std::endl;
}

// NOTE(Nick): The trotter function should directly call evolve_individual_nbody_cpu so we don't
// need to re-initialize Cin for each mu index, only copy, is currently a big
// performace hit!
void FCIComputerGPU::apply_sqop_evolution_gpu(
    const std::complex<double> time,
    const SQOperator& sqop,
    const bool antiherm,
    const bool adjoint)
{
    gpu_error();

    // timer_.acc_begin("=>copy in Cin <- C_");

    // TensorGPU Cin(C_.shape(), "Cin", true);
    // Cin.copy_in_gpu(C_);

    // timer_.acc_end("=>copy in Cin <- C_");

    evolve_individual_nbody_gpu<PrecompTuple>(
        time,
        sqop,
        C_,
        antiherm,
        adjoint,
        nullptr);
}

void FCIComputerGPU::apply_sqop_evolution_from_pool_gpu(
    const std::complex<double> time,
    const SQOpPoolGPU& pool,
    const int mu,
    const bool antiherm,
    const bool adjoint)
{
    gpu_error();

    if (pool.device_vecs_populated()) {
        if (data_type_ == "real") {
            const auto& device_spt_arys = pool.get_mu_tuple_real(mu);
            evolve_individual_nbody_gpu(
                time,
                pool.terms()[mu].second,
                C_,
                antiherm,
                adjoint,
                &device_spt_arys);
        } else {
            const auto& device_spt_arys = pool.get_mu_tuple(mu);
            evolve_individual_nbody_gpu(
                time,
                pool.terms()[mu].second,
                C_,
                antiherm,
                adjoint,
                &device_spt_arys);
        }
    } else {
        evolve_individual_nbody_gpu<PrecompTuple>(
            time,
            pool.terms()[mu].second,
            C_,
            antiherm,
            adjoint,
            nullptr);
    }
}

void FCIComputerGPU::evolve_pool_trotter_basic_gpu(
    const SQOpPoolGPU& pool,
    const bool antiherm,
    const bool adjoint)
{
    gpu_error();

    if (pool.device_vecs_populated()==false){
        // No precomp provided

        if (data_type_ == "complex") {
            if(adjoint){
                for (int i = pool.terms().size() - 1; i >= 0; --i) {
                    evolve_individual_nbody_gpu<PrecompTuple>(
                        pool.terms()[i].first,
                        pool.terms()[i].second,
                        C_,
                        antiherm,
                        adjoint,
                        nullptr);
                }
            } else {
                for (const auto& sqop_term : pool.terms()) {
                    evolve_individual_nbody_gpu<PrecompTuple>(
                        sqop_term.first,
                        sqop_term.second,
                        C_,
                        antiherm,
                        adjoint,
                        nullptr);
                    }
            }
        } else if (data_type_ == "real") {
            if(adjoint){
                for (int i = pool.terms().size() - 1; i >= 0; --i) {
                    evolve_individual_nbody_gpu<PrecompTupleReal>(
                        pool.terms()[i].first,
                        pool.terms()[i].second,
                        C_,
                        antiherm,
                        adjoint,
                        nullptr);
                }
            } else {
                for (const auto& sqop_term : pool.terms()) {
                    evolve_individual_nbody_gpu<PrecompTupleReal>(
                        sqop_term.first,
                        sqop_term.second,
                        C_,
                        antiherm,
                        adjoint,
                        nullptr);
                    }
            }
        } else {
            throw std::runtime_error("Unsupported data type in v5 evolution.");
        }
    } else {
        // Precomp provided

        if (data_type_ == "complex") {
            if(adjoint){
                for (int i = pool.terms().size() - 1; i >= 0; --i) {
                    const auto& device_spt_arys = pool.get_mu_tuple(i);
                    evolve_individual_nbody_gpu(
                        pool.terms()[i].first,
                        pool.terms()[i].second,
                        C_,
                        antiherm,
                        adjoint,
                        &device_spt_arys);
                }
            } else {
                for (int i = 0; i < pool.terms().size(); ++i) {
                    const auto& device_spt_arys = pool.get_mu_tuple(i);
                    evolve_individual_nbody_gpu(
                        pool.terms()[i].first,
                        pool.terms()[i].second,
                        C_,
                        antiherm,
                        adjoint,
                        &device_spt_arys);
                }
            }
        } else if (data_type_ == "real") {
            if(adjoint){
                for (int i = pool.terms().size() - 1; i >= 0; --i) {
                    const auto& device_spt_arys = pool.get_mu_tuple_real(i);
                    evolve_individual_nbody_gpu(
                        pool.terms()[i].first,
                        pool.terms()[i].second,
                        C_,
                        antiherm,
                        adjoint,
                        &device_spt_arys);
                }
            } else {
                for (int i = 0; i < pool.terms().size(); ++i) {
                    const auto& device_spt_arys = pool.get_mu_tuple_real(i);
                    evolve_individual_nbody_gpu(
                        pool.terms()[i].first,
                        pool.terms()[i].second,
                        C_,
                        antiherm,
                        adjoint,
                        &device_spt_arys);
                }
            }
        } else {
            throw std::runtime_error("Unsupported data type in v5 evolution.");
        }
    }
}

void FCIComputerGPU::evolve_pool_trotter_gpu(
    const SQOpPoolGPU& pool,
    double evolution_time,
    int trotter_steps,
    int trotter_order,
    bool antiherm,
    bool adjoint)
{
    gpu_error();

    // Need to support no precomp tuple case later

    // if (pool.device_vecs_populated()==false){
    //     throw std::runtime_error("SQOpPoolGPU STP device arrays must be populated before evolution.");
    // }

    // now checking data type and getting appropriate precomp tuple
    if (pool.data_type() != data_type_) {
        throw std::runtime_error("Data type of SQOpPoolGPU does not match FCIComputerGPU data type.");
    }

    if (!antiherm && (data_type_ == "real" || pool.data_type() == "real")) {
        throw std::runtime_error("v5 Trotter evolution currently only supports antiherm=true.");
    }

    timer_.acc_begin("evolve_pool_trotter_gpu(outer)");

    if(trotter_order == 1){

        std::complex<double> prefactor = evolution_time / static_cast<std::complex<double>>(trotter_steps);

        if(adjoint){
            for( int r = 0; r < trotter_steps; r++) {
                for (int i = pool.terms().size() - 1; i >= 0; --i) {

                    if (pool.device_vecs_populated()==false) {
                        // no precomp provided
                        if (data_type_ == "complex") {
                            evolve_individual_nbody_gpu<PrecompTuple>(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                nullptr);
                        } else if (data_type_ == "real") {
                            evolve_individual_nbody_gpu<PrecompTupleReal>(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                nullptr);
                        } else {
                            throw std::runtime_error("Unsupported data type in v5 evolution.");
                        }
                    } else {
                        if (data_type_ == "complex") {
                            const auto& device_spt_arys = pool.get_mu_tuple(i);
                            evolve_individual_nbody_gpu(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                &device_spt_arys);
                        } else if (data_type_ == "real") {
                            const auto& device_spt_arys = pool.get_mu_tuple_real(i);
                            evolve_individual_nbody_gpu(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                &device_spt_arys);
                        } else {
                            throw std::runtime_error("Unsupported data type in v5 evolution.");
                        }
                    }

                }
            }
        } else {
            for( int r = 0; r < trotter_steps; r++) {
                for (int i = 0; i < pool.terms().size(); ++i) {

                    if (pool.device_vecs_populated()==false) {
                        // no precomp provided
                        if (data_type_ == "complex") {
                            evolve_individual_nbody_gpu<PrecompTuple>(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                nullptr);
                        } else if (data_type_ == "real") {
                            evolve_individual_nbody_gpu<PrecompTupleReal>(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                nullptr);
                        } else {
                            throw std::runtime_error("Unsupported data type in v5 evolution.");
                        }
                    } else {
                        if (data_type_ == "complex") {
                            const auto& device_spt_arys = pool.get_mu_tuple(i);
                            evolve_individual_nbody_gpu(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                &device_spt_arys);
                        } else if (data_type_ == "real") {
                            const auto& device_spt_arys = pool.get_mu_tuple_real(i);
                            evolve_individual_nbody_gpu(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                &device_spt_arys);
                        } else {
                            throw std::runtime_error("Unsupported data type in v5 evolution.");
                        }
                    }

                }
            }
        }

    } else if (trotter_order == 2) {

        std::complex<double> prefactor = 0.5 * evolution_time / static_cast<std::complex<double>>(trotter_steps);

        if(adjoint){
            for( int r = 0; r < trotter_steps; r++) {
                // First pass: backward (reversed order)
                for (int i = pool.terms().size() - 1; i >= 0; --i) {

                    if (pool.device_vecs_populated()==false) {
                        // no precomp provided
                        if (data_type_ == "complex") {
                            evolve_individual_nbody_gpu<PrecompTuple>(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                nullptr);
                        } else if (data_type_ == "real") {
                            evolve_individual_nbody_gpu<PrecompTupleReal>(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                nullptr);
                        } else {
                            throw std::runtime_error("Unsupported data type in v5 evolution.");
                        }
                    } else {
                        if (data_type_ == "complex") {
                            const auto& device_spt_arys = pool.get_mu_tuple(i);
                            evolve_individual_nbody_gpu(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                &device_spt_arys);
                        } else if (data_type_ == "real") {
                            const auto& device_spt_arys = pool.get_mu_tuple_real(i);
                            evolve_individual_nbody_gpu(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                &device_spt_arys);
                        } else {
                            throw std::runtime_error("Unsupported data type in v5 evolution.");
                        }
                    }

                }

                // Second pass: forward
                for (int i = 0; i < pool.terms().size(); ++i) {

                    if (pool.device_vecs_populated()==false) {
                        // no precomp provided
                        if (data_type_ == "complex") {
                            evolve_individual_nbody_gpu<PrecompTuple>(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                nullptr);
                        } else if (data_type_ == "real") {
                            evolve_individual_nbody_gpu<PrecompTupleReal>(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                nullptr);
                        } else {
                            throw std::runtime_error("Unsupported data type in v5 evolution.");
                        }
                    } else {
                        if (data_type_ == "complex") {
                            const auto& device_spt_arys = pool.get_mu_tuple(i);
                            evolve_individual_nbody_gpu(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                &device_spt_arys);
                        } else if (data_type_ == "real") {
                            const auto& device_spt_arys = pool.get_mu_tuple_real(i);
                            evolve_individual_nbody_gpu(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                &device_spt_arys);
                        } else {
                            throw std::runtime_error("Unsupported data type in v5 evolution.");
                        }
                    }

                }
            }
        } else {
            for( int r = 0; r < trotter_steps; r++) {
                // First pass: forward
                for (int i = 0; i < pool.terms().size(); ++i) {

                    if (pool.device_vecs_populated()==false) {
                        // no precomp provided
                        if (data_type_ == "complex") {
                            evolve_individual_nbody_gpu<PrecompTuple>(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                nullptr);
                        } else if (data_type_ == "real") {
                            evolve_individual_nbody_gpu<PrecompTupleReal>(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                nullptr);
                        } else {
                            throw std::runtime_error("Unsupported data type in v5 evolution.");
                        }
                    } else {
                        if (data_type_ == "complex") {
                            const auto& device_spt_arys = pool.get_mu_tuple(i);
                            evolve_individual_nbody_gpu(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                &device_spt_arys);
                        } else if (data_type_ == "real") {
                            const auto& device_spt_arys = pool.get_mu_tuple_real(i);
                            evolve_individual_nbody_gpu(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                &device_spt_arys);
                        } else {
                            throw std::runtime_error("Unsupported data type in v5 evolution.");
                        }
                    }

                }

                // Second pass: backward (reversed order)
                for (int i = pool.terms().size() - 1; i >= 0; --i) {

                    if (pool.device_vecs_populated()==false) {
                        // no precomp provided
                        if (data_type_ == "complex") {
                            evolve_individual_nbody_gpu<PrecompTuple>(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                nullptr);
                        } else if (data_type_ == "real") {
                            evolve_individual_nbody_gpu<PrecompTupleReal>(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                nullptr);
                        } else {
                            throw std::runtime_error("Unsupported data type in v5 evolution.");
                        }
                    } else {
                        if (data_type_ == "complex") {
                            const auto& device_spt_arys = pool.get_mu_tuple(i);
                            evolve_individual_nbody_gpu(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                &device_spt_arys);
                        } else if (data_type_ == "real") {
                            const auto& device_spt_arys = pool.get_mu_tuple_real(i);
                            evolve_individual_nbody_gpu(
                                prefactor * pool.terms()[i].first,
                                pool.terms()[i].second,
                                C_,
                                antiherm,
                                adjoint,
                                &device_spt_arys);
                        } else {
                            throw std::runtime_error("Unsupported data type in v5 evolution.");
                        }
                    }

                }
            }
        }

    } else {
        throw std::runtime_error("Higher than 2nd order trotter not yet implemented");
    }

    timer_.acc_end("evolve_pool_trotter_gpu(outer)");
}

void FCIComputerGPU::evolve_op_taylor_cpu(
    const SQOperator& op,
    const double evolution_time,
    const double convergence_thresh,
    const int max_taylor_iter)
{
    cpu_error();

    TensorGPU Cevol = C_;

    for (int order = 1; order < max_taylor_iter; ++order) {

        // std::cout << "I get here, order: " << order << std::endl;

        // std::cout << "C_: " << C_.str() << std::endl;
        // std::cout << "Cevol: " << Cevol.str() << std::endl;

        std::complex<double> coeff(0.0, -evolution_time);
        apply_sqop_gpu(op);
        scale(coeff);

        Cevol.zaxpy(
            C_,
            1.0 / std::tgamma(order+1),
            1,
            1);

        if (C_.norm() * std::abs(coeff) < convergence_thresh) {
            break;
        }
    }
    C_ = Cevol;
}

/// NOTE: Cin should be const, changing for now
void FCIComputerGPU::apply_individual_nbody1_accumulate_gpu(
    const std::complex<double> coeff,
    TensorGPU& Cin,
    TensorGPU& Cout,
    int counta,
    int countb)
{

    if ((targeta_gpu_.size() != sourcea_gpu_.size()) or (sourcea_gpu_.size() != paritya_gpu_.size())) {
        throw std::runtime_error("The sizes of atarget, asource, and aparity must be the same.");
    }

    if ((targetb_gpu_.size() != sourceb_gpu_.size()) or (sourceb_gpu_.size() != parityb_gpu_.size())) {
        throw std::runtime_error("The sizes of btarget, bsource, and bparity must be the same.");
    }

    cuDoubleComplex cu_coeff = make_cuDoubleComplex(coeff.real(), coeff.imag());

    // Call the GPU kernel using thrust raw pointers directly
    apply_individual_nbody1_accumulate_wrapper(
        cu_coeff,
        thrust::raw_pointer_cast(Cin.read_d_data().data()),
        thrust::raw_pointer_cast(Cout.d_data().data()),
        thrust::raw_pointer_cast(sourcea_gpu_.data()),
        thrust::raw_pointer_cast(targeta_gpu_.data()),
        thrust::raw_pointer_cast(paritya_gpu_.data()),
        thrust::raw_pointer_cast(sourceb_gpu_.data()),
        thrust::raw_pointer_cast(targetb_gpu_.data()),
        thrust::raw_pointer_cast(parityb_gpu_.data()),
        nbeta_strs_,
        counta,
        countb,
        Cin.size() * sizeof(cuDoubleComplex));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("Failed to execute the apply_individual_nbody1_accumulate operation on the GPU.");
    }
}

/// NOTE: Cin should be const, changing for now
void FCIComputerGPU::apply_individual_nbody_accumulate_gpu(
    const std::complex<double> coeff,
    TensorGPU& Cin,
    TensorGPU& Cout,
    const std::vector<int>& daga,
    const std::vector<int>& undaga,
    const std::vector<int>& dagb,
    const std::vector<int>& undagb)
{
    gpu_error();

    if((daga.size() != undaga.size()) or (dagb.size() != undagb.size())){
        throw std::runtime_error("must be same number of alpha anihilators/creators and beta anihilators/creators.");
    }

    timer_.acc_begin("===>hard nbody acc setup");

    int counta = 0;
    int countb = 0;

    graph_.make_mapping_each_otf_gpu_complex(
        true,
        daga,
        undaga,
        &counta,
        sourcea_gpu_,
        targeta_gpu_,
        paritya_gpu_);

    if (counta == 0) {
        return;
    }

    graph_.make_mapping_each_otf_gpu_complex(
        false,
        dagb,
        undagb,
        &countb,
        sourceb_gpu_,
        targetb_gpu_,
        parityb_gpu_);

    if (countb == 0) {
        return;
    }

    timer_.acc_end("===>hard nbody acc setup");


    timer_.acc_begin("===>hard nbody acc kernel");
    /// TODO: changing this function to use private members of FCIComputerGPU
    apply_individual_nbody1_accumulate_gpu(
        coeff,
        Cin,
        Cout,
        counta,
        countb);

    timer_.acc_end("===>hard nbody acc kernel");
}

/// NOTE: Cin should be const, changing for now
void FCIComputerGPU::apply_individual_sqop_term_gpu(
    const std::tuple< std::complex<double>, std::vector<size_t>, std::vector<size_t>>& term,
    TensorGPU& Cin,
    TensorGPU& Cout)
{
    std::vector<int> crea;
    std::vector<int> anna;

    std::vector<int> creb;
    std::vector<int> annb;

    local_timer my_timer = local_timer();

    for(size_t i = 0; i < std::get<1>(term).size(); i++){
        if(std::get<1>(term)[i]%2 == 0){
            crea.push_back(std::floor(std::get<1>(term)[i] / 2));
        } else {
            creb.push_back(std::floor(std::get<1>(term)[i] / 2));
        }
    }

    for(size_t i = 0; i < std::get<2>(term).size(); i++){
        if(std::get<2>(term)[i]%2 == 0){
            anna.push_back(std::floor(std::get<2>(term)[i] / 2));
        } else {
            annb.push_back(std::floor(std::get<2>(term)[i] / 2));
        }
    }

    if (std::get<1>(term).size() != std::get<2>(term).size()) {
        throw std::invalid_argument("Each term must have same number of anihilators and creators");
    }

    std::vector<size_t> ops1(std::get<1>(term));
    std::vector<size_t> ops2(std::get<2>(term));
    ops1.insert(ops1.end(), ops2.begin(), ops2.end());

    int nswaps = parity_sort(ops1);

    apply_individual_nbody_accumulate_gpu(
        pow(-1, nswaps) * std::get<0>(term),
        Cin,
        Cout,
        crea,
        anna,
        creb,
        annb);
}

void FCIComputerGPU::apply_sqop_gpu(const SQOperator& sqop)
{
    C_.gpu_error();
    TensorGPU Cin(C_.shape(), "Cin", true, C_.data_type(), true);
    Cin.copy_in_gpu(C_);

    local_timer my_timer = local_timer();

    C_.zero_gpu();

    for (const auto& term : sqop.terms()) {
        if(std::abs(std::get<0>(term)) > compute_threshold_){
        apply_individual_sqop_term_gpu(
            term,
            Cin,
            C_);
        }
    }
}

void FCIComputerGPU::apply_diagonal_of_sqop_cpu(
    const SQOperator& sq_op,
    const bool invert_coeff)
{
    cpu_error();

    TensorGPU Cin = C_;
    C_.zero();

    for(const auto& term : sq_op.terms()){
        std::tuple< std::complex<double>, std::vector<size_t>, std::vector<size_t>> temp_term;
        std::vector<size_t> ann;
        std::vector<size_t> cre;
        cre = std::get<1>(term);
        ann = std::get<2>(term);

        std::sort(cre.begin(), cre.end());
        std::sort(ann.begin(), ann.end());

        if(std::equal(cre.begin(), cre.end(), ann.begin(), ann.end()) && std::abs(std::get<0>(term)) > compute_threshold_){
            std::get<1>(temp_term) = cre;
            std::get<2>(temp_term) = ann;

            if(invert_coeff){
                std::get<0>(temp_term) = 1.0 / std::get<0>(term);
            } else {
                std::get<0>(temp_term) = std::get<0>(term);
            }

            apply_individual_sqop_term_gpu(
                temp_term,
                Cin,
                C_);
        }
    }
}

void FCIComputerGPU::apply_sqop_pool_cpu(const SQOpPool& sqop_pool)
{
    cpu_error();

    TensorGPU Cin = C_;
    C_.zero();

    for (const auto& sqop : sqop_pool.terms()) {
        std::complex<double> outer_coeff = sqop.first;
        for (const auto& term : sqop.second.terms()) {
            std::tuple< std::complex<double>, std::vector<size_t>, std::vector<size_t>> temp_term = term;

            std::get<0>(temp_term) *= outer_coeff;

            if(std::abs(std::get<0>(temp_term)) > compute_threshold_){
                apply_individual_sqop_term_gpu(
                    temp_term,
                    Cin,
                    C_);
            }
        }
    }
}

std::complex<double> FCIComputerGPU::get_exp_val_cpu(const SQOperator& sqop)
{
    TensorGPU Cin = C_;
    C_.zero();
    for (const auto& term : sqop.terms()) {
        if(std::abs(std::get<0>(term)) > compute_threshold_){
        apply_individual_sqop_term_gpu(
            term,
            Cin,
            C_);
        }
    }
    std::complex<double> val = C_.vector_dot(Cin);
    C_ = Cin;
    return val;
}

std::complex<double> FCIComputerGPU::get_exp_val(const SQOperator& sqop)
{
    gpu_error();

    TensorGPU Cin(
        {nalfa_strs_, nbeta_strs_},
        "Cin",
        true,
        data_type_,
        true
    );

    Cin.copy_in_gpu(C_);

    C_.zero_gpu();

    for (const auto& term : sqop.terms()) {
        if(std::abs(std::get<0>(term)) > compute_threshold_){
        apply_individual_sqop_term_gpu(
            term,
            Cin,
            C_);
        }
    }

    std::complex<double> val = C_.vector_dot(Cin);
    C_.copy_in_gpu(Cin);

    return val;
}

std::complex<double> FCIComputerGPU::get_exp_val_tensor_gpu(
    const std::complex<double> h0e,
    const TensorGPU& h1e,
    const TensorGPU& h2e,
    TensorGPU& h2e_einsum,
    size_t norb)
{
    // Save C_ into a gpu_only temporary to avoid host memory overhead
    TensorGPU Cin({nalfa_strs_, nbeta_strs_}, "Cin", true, data_type_, true);
    Cin.copy_in_gpu(C_);

    apply_tensor_spat_012bdy_gpu_v2(
        h0e,
        h1e,
        h2e,
        h2e_einsum,
        norb
    );

    std::complex<double> val = C_.vector_dot(Cin);

    /// TODO: change to move opperation not deep copy
    C_.copy_in_gpu(Cin);
    return val;
}

void FCIComputerGPU::scale(const std::complex<double> a)
{
    C_.scale(a);
}

std::complex<double> FCIComputerGPU::state_vector_dot_gpu(FCIComputerGPU& other) const {
    gpu_error();
    other.gpu_error();
    return C_.vector_dot(other.C_);
}

void FCIComputerGPU::dot_individual_sqop_term_gpu(
    const std::tuple< std::complex<double>, std::vector<size_t>, std::vector<size_t>>& term,
    const TensorGPU& psi,
    const TensorGPU& sigma,
    cuDoubleComplex* d_accum)
{
    // Identical alpha/beta index splitting as apply_individual_sqop_term_gpu.
    std::vector<int> crea, anna, creb, annb;

    for (size_t i = 0; i < std::get<1>(term).size(); i++) {
        if (std::get<1>(term)[i] % 2 == 0)
            crea.push_back(static_cast<int>(std::floor(std::get<1>(term)[i] / 2)));
        else
            creb.push_back(static_cast<int>(std::floor(std::get<1>(term)[i] / 2)));
    }
    for (size_t i = 0; i < std::get<2>(term).size(); i++) {
        if (std::get<2>(term)[i] % 2 == 0)
            anna.push_back(static_cast<int>(std::floor(std::get<2>(term)[i] / 2)));
        else
            annb.push_back(static_cast<int>(std::floor(std::get<2>(term)[i] / 2)));
    }

    if (std::get<1>(term).size() != std::get<2>(term).size())
        throw std::invalid_argument("Each term must have same number of annihilators and creators");

    std::vector<size_t> ops1(std::get<1>(term));
    std::vector<size_t> ops2(std::get<2>(term));
    ops1.insert(ops1.end(), ops2.begin(), ops2.end());
    int nswaps = parity_sort(ops1);

    std::complex<double> coeff = std::pow(-1, nswaps) * std::get<0>(term);

    // Build alpha-spin mapping into the existing member buffers.
    int counta = 0, countb = 0;
    graph_.make_mapping_each_otf_gpu_complex(
        true, crea, anna, &counta,
        sourcea_gpu_, targeta_gpu_, paritya_gpu_);
    if (counta == 0) return;

    graph_.make_mapping_each_otf_gpu_complex(
        false, creb, annb, &countb,
        sourceb_gpu_, targetb_gpu_, parityb_gpu_);
    if (countb == 0) return;

    cuDoubleComplex cu_coeff = make_cuDoubleComplex(coeff.real(), coeff.imag());

    // Launch the fused dot kernel — no output tensor, accumulates into d_accum.
    dot_individual_nbody1_wrapper(
        cu_coeff,
        thrust::raw_pointer_cast(psi.read_d_data().data()),
        thrust::raw_pointer_cast(sigma.read_d_data().data()),
        thrust::raw_pointer_cast(sourcea_gpu_.data()),
        thrust::raw_pointer_cast(targeta_gpu_.data()),
        thrust::raw_pointer_cast(paritya_gpu_.data()),
        thrust::raw_pointer_cast(sourceb_gpu_.data()),
        thrust::raw_pointer_cast(targetb_gpu_.data()),
        thrust::raw_pointer_cast(parityb_gpu_.data()),
        nbeta_strs_,
        counta,
        countb,
        d_accum);
}

/// Compute <sigma | sqop | (*this)> without modifying either state vector.
/// A single device scalar is allocated (zeroed) once; all SQOp terms accumulate
/// into it via the fused dot kernel.  One sync at the end brings the result back.
std::complex<double> FCIComputerGPU::dot_sqop_gpu(FCIComputerGPU& sigma, const SQOperator& sqop)
{
    gpu_error();
    sigma.gpu_error();

    // Single device accumulator, zeroed once for the entire sqop sum.
    thrust::device_vector<cuDoubleComplex> d_accum_vec(1, make_cuDoubleComplex(0.0, 0.0));
    cuDoubleComplex* d_accum = thrust::raw_pointer_cast(d_accum_vec.data());

    for (const auto& term : sqop.terms()) {
        if (std::abs(std::get<0>(term)) > compute_threshold_) {
            // C_ is the ket; sigma.C_ is the bra.  Neither is touched.
            dot_individual_sqop_term_gpu(term, C_, sigma.C_, d_accum);
        }
    }

    // Final sync is a no-op (wrapper already synced), kept for safety.
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_sqop_gpu sync failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    cuDoubleComplex res = d_accum_vec[0];
    return std::complex<double>(res.x, res.y);
}

void FCIComputerGPU::dot_individual_sqop_term_gpu_real(
    const std::tuple< std::complex<double>, std::vector<size_t>, std::vector<size_t>>& term,
    const TensorGPU& psi,
    const TensorGPU& sigma,
    double* d_accum)
{
    std::vector<int> crea, anna, creb, annb;

    for (size_t i = 0; i < std::get<1>(term).size(); i++) {
        if (std::get<1>(term)[i] % 2 == 0)
            crea.push_back(static_cast<int>(std::floor(std::get<1>(term)[i] / 2)));
        else
            creb.push_back(static_cast<int>(std::floor(std::get<1>(term)[i] / 2)));
    }
    for (size_t i = 0; i < std::get<2>(term).size(); i++) {
        if (std::get<2>(term)[i] % 2 == 0)
            anna.push_back(static_cast<int>(std::floor(std::get<2>(term)[i] / 2)));
        else
            annb.push_back(static_cast<int>(std::floor(std::get<2>(term)[i] / 2)));
    }

    if (std::get<1>(term).size() != std::get<2>(term).size())
        throw std::invalid_argument("Each term must have same number of annihilators and creators");

    std::vector<size_t> ops1(std::get<1>(term));
    std::vector<size_t> ops2(std::get<2>(term));
    ops1.insert(ops1.end(), ops2.begin(), ops2.end());
    int nswaps = parity_sort(ops1);

    // For a real sqop the coefficient must be real; take real part after parity.
    double coeff = std::real(std::pow(-1, nswaps) * std::get<0>(term));

    int counta = 0, countb = 0;
    graph_.make_mapping_each_otf_gpu_real(
        true, crea, anna, &counta,
        sourcea_gpu_, targeta_gpu_, paritya_gpu_real_);
    if (counta == 0) return;

    graph_.make_mapping_each_otf_gpu_real(
        false, creb, annb, &countb,
        sourceb_gpu_, targetb_gpu_, parityb_gpu_real_);
    if (countb == 0) return;

    dot_individual_nbody1_real_wrapper(
        coeff,
        thrust::raw_pointer_cast(psi.read_d_re_data().data()),
        thrust::raw_pointer_cast(sigma.read_d_re_data().data()),
        thrust::raw_pointer_cast(sourcea_gpu_.data()),
        thrust::raw_pointer_cast(targeta_gpu_.data()),
        thrust::raw_pointer_cast(paritya_gpu_real_.data()),
        thrust::raw_pointer_cast(sourceb_gpu_.data()),
        thrust::raw_pointer_cast(targetb_gpu_.data()),
        thrust::raw_pointer_cast(parityb_gpu_real_.data()),
        nbeta_strs_,
        counta,
        countb,
        d_accum);
}

double FCIComputerGPU::dot_sqop_gpu_real(FCIComputerGPU& sigma, const SQOperator& sqop)
{
    gpu_error();
    sigma.gpu_error();

    thrust::device_vector<double> d_accum_vec(1, 0.0);
    double* d_accum = thrust::raw_pointer_cast(d_accum_vec.data());

    for (const auto& term : sqop.terms()) {
        if (std::abs(std::get<0>(term)) > compute_threshold_) {
            dot_individual_sqop_term_gpu_real(term, C_, sigma.C_, d_accum);
        }
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_sqop_gpu_real sync failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    return d_accum_vec[0];
}

/// Compute <sigma | K_mu | this> using precomputed pool data.
/// Reuses source/target/parity already stored by the evolution precomputation:
///   dag  term (excitation):   source=scale_inds_undag, target=scale_inds_dag, parity=parity_undag
///   undag term (de-excit.):   source=scale_inds_dag,   target=scale_inds_undag, parity=parity_dag
/// Only the two scalar coefficients (parity_sort * c_mu) are uniquely precomputed for the dot.
double FCIComputerGPU::dot_sqop_from_pool_gpu_real(
    FCIComputerGPU& sigma,
    const SQOpPoolGPU& pool,
    int mu)
{
    gpu_error();
    sigma.gpu_error();

    if (!pool.device_vecs_populated()) {
        return dot_sqop_gpu_real(sigma, pool.terms()[mu].second);
    }

    thrust::device_vector<double> d_accum_vec(1, 0.0);
    double* d_accum = thrust::raw_pointer_cast(d_accum_vec.data());

    const double* psi_data   = thrust::raw_pointer_cast(C_.read_d_re_data().data());
    const double* sigma_data = thrust::raw_pointer_cast(sigma.C_.read_d_re_data().data());

    // dag term (excitation): source=undag, target=dag, parity=undag
    int counta_dag = static_cast<int>(pool.terms_scale_indsa_undag_gpu()[mu].size());
    int countb_dag = static_cast<int>(pool.terms_scale_indsb_undag_gpu()[mu].size());

    // undag term (de-excitation): source=dag, target=undag, parity=dag
    int counta_undag = static_cast<int>(pool.terms_scale_indsa_dag_gpu()[mu].size());
    int countb_undag = static_cast<int>(pool.terms_scale_indsb_dag_gpu()[mu].size());

    bool dag_valid   = (counta_dag > 0 && countb_dag > 0);
    bool undag_valid = (counta_undag > 0 && countb_undag > 0);

    DotKernelKind kind = pool.dot_kernel_kinds()[mu];

    if (dag_valid && undag_valid && kind != DotKernelKind::Easy) {
        // Use specialized paired-dual kernels based on operator classification
        if (kind == DotKernelKind::AlphaOnly) {
            // Alpha-only: beta is identity, run coalesced row-traversal kernel
            // source_a=undag, target_a=dag; parity_uv=parity_undag, parity_vu=parity_dag
            dot_alpha_only_dual_real_wrapper(
                psi_data, sigma_data,
                thrust::raw_pointer_cast(pool.terms_scale_indsa_undag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_scale_indsa_dag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_paritya_undag_re_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_paritya_dag_re_gpu()[mu].data()),
                counta_dag, nbeta_strs_,
                pool.dot_coeff_dag_re()[mu],
                pool.dot_coeff_undag_re()[mu],
                d_accum);
        } else if (kind == DotKernelKind::BetaOnly) {
            // Beta-only: alpha is identity, run beta-fast row-major kernel
            dot_beta_only_dual_real_wrapper(
                psi_data, sigma_data,
                thrust::raw_pointer_cast(pool.terms_scale_indsb_undag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_scale_indsb_dag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_parityb_undag_re_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_parityb_dag_re_gpu()[mu].data()),
                countb_dag, nalfa_strs_, nbeta_strs_,
                pool.dot_coeff_dag_re()[mu],
                pool.dot_coeff_undag_re()[mu],
                d_accum);
        } else {
            // Mixed: general tiled kernel with beta in fast dimension
            dot_mixed_dual_real_wrapper(
                psi_data, sigma_data,
                thrust::raw_pointer_cast(pool.terms_scale_indsa_undag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_scale_indsa_dag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_paritya_undag_re_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_paritya_dag_re_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_scale_indsb_undag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_scale_indsb_dag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_parityb_undag_re_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_parityb_dag_re_gpu()[mu].data()),
                counta_dag, countb_dag, nbeta_strs_,
                pool.dot_coeff_dag_re()[mu],
                pool.dot_coeff_undag_re()[mu],
                d_accum);
        }
    } else if (dag_valid && undag_valid) {
        // Generic fused dual launch (fallback for Easy or unclassified)
        dot_individual_nbody1_real_dual_wrapper(
            pool.dot_coeff_dag_re()[mu],
            psi_data, sigma_data,
            thrust::raw_pointer_cast(pool.terms_scale_indsa_undag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsa_dag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_paritya_undag_re_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsb_undag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsb_dag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_parityb_undag_re_gpu()[mu].data()),
            counta_dag, countb_dag,
            pool.dot_coeff_undag_re()[mu],
            thrust::raw_pointer_cast(pool.terms_scale_indsa_dag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsa_undag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_paritya_dag_re_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsb_dag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsb_undag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_parityb_dag_re_gpu()[mu].data()),
            counta_undag, countb_undag,
            nbeta_strs_, d_accum);
    } else if (dag_valid) {
        dot_individual_nbody1_real_wrapper(
            pool.dot_coeff_dag_re()[mu],
            psi_data, sigma_data,
            thrust::raw_pointer_cast(pool.terms_scale_indsa_undag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsa_dag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_paritya_undag_re_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsb_undag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsb_dag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_parityb_undag_re_gpu()[mu].data()),
            nbeta_strs_, counta_dag, countb_dag, d_accum);
    } else if (undag_valid) {
        dot_individual_nbody1_real_wrapper(
            pool.dot_coeff_undag_re()[mu],
            psi_data, sigma_data,
            thrust::raw_pointer_cast(pool.terms_scale_indsa_dag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsa_undag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_paritya_dag_re_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsb_dag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsb_undag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_parityb_dag_re_gpu()[mu].data()),
            nbeta_strs_, counta_undag, countb_undag, d_accum);
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_sqop_from_pool_gpu_real sync failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    return d_accum_vec[0];
}

/// Compute <sigma | K_mu | this> (complex path) using precomputed pool arrays.
std::complex<double> FCIComputerGPU::dot_sqop_from_pool_gpu(
    FCIComputerGPU& sigma,
    const SQOpPoolGPU& pool,
    int mu)
{
    gpu_error();
    sigma.gpu_error();

    if (!pool.device_vecs_populated()) {
        return dot_sqop_gpu(sigma, pool.terms()[mu].second);
    }

    thrust::device_vector<cuDoubleComplex> d_accum_vec(1, make_cuDoubleComplex(0.0, 0.0));
    cuDoubleComplex* d_accum = thrust::raw_pointer_cast(d_accum_vec.data());

    const cuDoubleComplex* psi_data   = thrust::raw_pointer_cast(C_.read_d_data().data());
    const cuDoubleComplex* sigma_data = thrust::raw_pointer_cast(sigma.C_.read_d_data().data());

    // dag term (excitation): source=undag, target=dag, parity=undag
    int counta_dag = static_cast<int>(pool.terms_scale_indsa_undag_gpu()[mu].size());
    int countb_dag = static_cast<int>(pool.terms_scale_indsb_undag_gpu()[mu].size());

    // undag term (de-excitation): source=dag, target=undag, parity=dag
    int counta_undag = static_cast<int>(pool.terms_scale_indsa_dag_gpu()[mu].size());
    int countb_undag = static_cast<int>(pool.terms_scale_indsb_dag_gpu()[mu].size());

    bool dag_valid   = (counta_dag > 0 && countb_dag > 0);
    bool undag_valid = (counta_undag > 0 && countb_undag > 0);

    DotKernelKind kind = pool.dot_kernel_kinds()[mu];

    if (dag_valid && undag_valid && kind != DotKernelKind::Easy) {
        if (kind == DotKernelKind::AlphaOnly) {
            dot_alpha_only_dual_wrapper(
                psi_data, sigma_data,
                thrust::raw_pointer_cast(pool.terms_scale_indsa_undag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_scale_indsa_dag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_paritya_undag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_paritya_dag_gpu()[mu].data()),
                counta_dag, nbeta_strs_,
                pool.dot_coeff_dag()[mu],
                pool.dot_coeff_undag()[mu],
                d_accum);
        } else if (kind == DotKernelKind::BetaOnly) {
            dot_beta_only_dual_wrapper(
                psi_data, sigma_data,
                thrust::raw_pointer_cast(pool.terms_scale_indsb_undag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_scale_indsb_dag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_parityb_undag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_parityb_dag_gpu()[mu].data()),
                countb_dag, nalfa_strs_, nbeta_strs_,
                pool.dot_coeff_dag()[mu],
                pool.dot_coeff_undag()[mu],
                d_accum);
        } else {
            dot_mixed_dual_wrapper(
                psi_data, sigma_data,
                thrust::raw_pointer_cast(pool.terms_scale_indsa_undag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_scale_indsa_dag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_paritya_undag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_paritya_dag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_scale_indsb_undag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_scale_indsb_dag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_parityb_undag_gpu()[mu].data()),
                thrust::raw_pointer_cast(pool.terms_parityb_dag_gpu()[mu].data()),
                counta_dag, countb_dag, nbeta_strs_,
                pool.dot_coeff_dag()[mu],
                pool.dot_coeff_undag()[mu],
                d_accum);
        }
    } else if (dag_valid && undag_valid) {
        dot_individual_nbody1_dual_wrapper(
            pool.dot_coeff_dag()[mu],
            psi_data, sigma_data,
            thrust::raw_pointer_cast(pool.terms_scale_indsa_undag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsa_dag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_paritya_undag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsb_undag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsb_dag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_parityb_undag_gpu()[mu].data()),
            counta_dag, countb_dag,
            pool.dot_coeff_undag()[mu],
            thrust::raw_pointer_cast(pool.terms_scale_indsa_dag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsa_undag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_paritya_dag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsb_dag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsb_undag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_parityb_dag_gpu()[mu].data()),
            counta_undag, countb_undag,
            nbeta_strs_, d_accum);
    } else if (dag_valid) {
        dot_individual_nbody1_wrapper(
            pool.dot_coeff_dag()[mu],
            psi_data, sigma_data,
            thrust::raw_pointer_cast(pool.terms_scale_indsa_undag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsa_dag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_paritya_undag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsb_undag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsb_dag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_parityb_undag_gpu()[mu].data()),
            nbeta_strs_, counta_dag, countb_dag, d_accum);
    } else if (undag_valid) {
        dot_individual_nbody1_wrapper(
            pool.dot_coeff_undag()[mu],
            psi_data, sigma_data,
            thrust::raw_pointer_cast(pool.terms_scale_indsa_dag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsa_undag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_paritya_dag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsb_dag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_scale_indsb_undag_gpu()[mu].data()),
            thrust::raw_pointer_cast(pool.terms_parityb_dag_gpu()[mu].data()),
            nbeta_strs_, counta_undag, countb_undag, d_accum);
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_sqop_from_pool_gpu sync failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    cuDoubleComplex res = d_accum_vec[0];
    return std::complex<double>(res.x, res.y);
}

void FCIComputerGPU::copy_state_into(TensorGPU& tensor) const {
    tensor.shape_error(C_.shape());
    gpu_error();
    tensor = C_;
}


/// TODO: This is commented out in TensorGPU
/*
std::vector<double> FCIComputerGPU::direct_expectation_value(const TensorOperator& top)
{
    // Implementation would be similar to FCIComputerGPU but using TensorGPU
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("");
}*/

/// TODO: Not implemented in TensorGPU
/*
std::complex<double> FCIComputerGPU::coeff(const QubitBasis& abasis, const QubitBasis& bbasis)
{
    // Implementation would be similar to FCIComputerGPU but using TensorGPU
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("");
}*/

void FCIComputerGPU::set_state_cpu(const TensorGPU& other_state)
{
    cpu_error();
    other_state.cpu_error();
    C_.copy_in(other_state);
}

void FCIComputerGPU::set_state_gpu(const TensorGPU& other_state)
{
    gpu_error();
    other_state.gpu_error();
    C_.copy_in_gpu(other_state);
}

void FCIComputerGPU::set_state_from_other_cpu(const FCIComputerGPU& other)
{
    cpu_error();
    other.cpu_error();
    C_.copy_in(other.C_);
}

void FCIComputerGPU::set_state_from_other_gpu(const FCIComputerGPU& other)
{
    gpu_error();
    other.gpu_error();
    C_.copy_in_gpu(other.C_);
}

void FCIComputerGPU::set_state_from_tensor_cpu(const Tensor& other_state)
{
    cpu_error();
    C_.copy_in_from_tensor(other_state);
}

void FCIComputerGPU::zero_cpu()
{
    cpu_error();
    C_.zero();
}

void FCIComputerGPU::hartree_fock_cpu()
{
    cpu_error();
    C_.zero();
    C_.set({0, 0}, 1.0);
}

void FCIComputerGPU::hartree_fock_gpu()
{
    gpu_error();
    C_.zero_gpu();
    C_.set_gpu({0, 0}, 1.0);
}

void FCIComputerGPU::print_vector(const std::vector<int>& vec, const std::string& name)
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

void FCIComputerGPU::print_vector_thrust(const thrust::host_vector<int>& vec, const std::string& name)
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

void FCIComputerGPU::print_vector_uint(const std::vector<uint64_t>& vec, const std::string& name)
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

void FCIComputerGPU::print_vector_thrust_cuDoubleComplex(const thrust::host_vector<cuDoubleComplex>& vec, const std::string& name)
{
    std::cout << "\n" << name << ": ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::complex<double> tmp = {vec[i].x, vec[i].y};
        std::cout << tmp;
        if (i < vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;
}

void FCIComputerGPU::populate_index_arrays_for_pool_evo(SQOpPoolGPU& pool){

    if(pool.device_vecs_populated()){
        return;
    }

    // Ensure data_type consistency between this computer and the pool
    const std::string comp_dt = data_type_;
    const std::string pool_dt = pool.data_type();
    if (pool_dt != comp_dt) {
        throw std::runtime_error("Data type mismatch between FCIComputerGPU and SQOpPoolGPU.");
    }

    // Determine which parity storage formats we need to populate
    const bool need_complex = (pool_dt == "complex");
    const bool real_only    = (pool_dt == "real");

    for (int i=0; i<static_cast<int>(pool.terms().size()); ++i){

        auto sqop = pool.terms()[i].second;

        if (sqop.terms().size() != 2) {
            std::cout << "This sqop has " << sqop.terms().size() << " terms." << std::endl;
            throw std::invalid_argument("Individual n-body code is called with multiple terms");
        }

        // append h_mu
        pool.outer_coeffs().push_back(pool.terms()[i].first);

        auto term = sqop.terms()[0];

        /// append c_mu
        pool.inner_coeffs().push_back(std::get<0>(term));

        std::vector<int> crea;
        std::vector<int> anna;
        std::vector<int> creb;
        std::vector<int> annb;

        for(size_t k = 0; k < std::get<1>(term).size(); k++){
            if(std::get<1>(term)[k]%2 == 0){
                crea.push_back(static_cast<int>(std::get<1>(term)[k] / 2));
            } else {
                creb.push_back(static_cast<int>(std::get<1>(term)[k] / 2));
            }
        }

        for(size_t k = 0; k < std::get<2>(term).size(); k++){
            if(std::get<2>(term)[k]%2 == 0){
                anna.push_back(static_cast<int>(std::get<2>(term)[k] / 2));
            } else {
                annb.push_back(static_cast<int>(std::get<2>(term)[k] / 2));
            }
        }

        std::vector<size_t> ops1(std::get<1>(term));
        std::vector<size_t> ops2(std::get<2>(term));
        ops1.insert(ops1.end(), ops2.begin(), ops2.end());

        int nswaps = parity_sort(ops1);
        std::complex<double> parity = std::pow(-1, nswaps);

        if (crea == anna && creb == annb) {

            // EASY BRANCH
            std::pair<std::vector<int>, std::vector<int>> maps = evaluate_map_number_cpu(anna, annb);
            thrust::device_vector<int> d_alfa(maps.first.begin(), maps.first.end());
            thrust::device_vector<int> d_beta(maps.second.begin(), maps.second.end());

            pool.terms_scale_indsa_dag_gpu().emplace_back(d_alfa);
            pool.terms_scale_indsa_undag_gpu().emplace_back();
            pool.terms_scale_indsb_dag_gpu().emplace_back(d_beta);
            pool.terms_scale_indsb_undag_gpu().emplace_back();

            if (need_complex) {
                pool.terms_paritya_dag_gpu().emplace_back();
                pool.terms_paritya_undag_gpu().emplace_back();
                pool.terms_parityb_dag_gpu().emplace_back();
                pool.terms_parityb_undag_gpu().emplace_back();
            } else if (real_only) {
                pool.terms_paritya_dag_re_gpu().emplace_back();
                pool.terms_paritya_undag_re_gpu().emplace_back();
                pool.terms_parityb_dag_re_gpu().emplace_back();
                pool.terms_parityb_undag_re_gpu().emplace_back();
            }

            pool.dot_kernel_kinds().push_back(DotKernelKind::Easy);

        } else if (crea.size() == anna.size() && creb.size() == annb.size()) {

            // HARD BRANCH
            std::vector<int> dagworka(crea);
            std::vector<int> dagworkb(creb);
            std::vector<int> undagworka(anna);
            std::vector<int> undagworkb(annb);
            std::vector<int> numbera;
            std::vector<int> numberb;

            int parity_shift = 0;
            parity_shift += isolate_number_operators_cpu(
                crea,
                anna,
                dagworka,
                undagworka,
                numbera);

            parity_shift += isolate_number_operators_cpu(
                creb,
                annb,
                dagworkb,
                undagworkb,
                numberb);

            std::vector<int> numbera_dagworka(numbera.begin(), numbera.end());
            numbera_dagworka.insert(numbera_dagworka.end(), dagworka.begin(), dagworka.end());

            std::vector<int> numberb_dagworkb(numberb.begin(), numberb.end());
            numberb_dagworkb.insert(numberb_dagworkb.end(), dagworkb.begin(), dagworkb.end());

            std::vector<int> numbera_undagworka(numbera.begin(), numbera.end());
            numbera_undagworka.insert(numbera_undagworka.end(), undagworka.begin(), undagworka.end());

            std::vector<int> numberb_undagworkb(numberb.begin(), numberb.end());
            numberb_undagworkb.insert(numberb_undagworkb.end(), undagworkb.begin(), undagworkb.end());

            std::pair<std::vector<int>, std::vector<int>> maps1 = evaluate_map_cpu(
                numbera_dagworka,
                undagworka,
                numberb_dagworkb,
                undagworkb);

            thrust::device_vector<int> d_alfa_1st(maps1.first.begin(), maps1.first.end());
            thrust::device_vector<int> d_beta_1st(maps1.second.begin(), maps1.second.end());

            std::pair<std::vector<int>, std::vector<int>> maps2 = evaluate_map_cpu(
                numbera_undagworka,
                dagworka,
                numberb_undagworkb,
                dagworkb);

            thrust::device_vector<int> d_alfa_2nd(maps2.first.begin(), maps2.first.end());
            thrust::device_vector<int> d_beta_2nd(maps2.second.begin(), maps2.second.end());

            pool.terms_scale_indsa_dag_gpu().emplace_back(d_alfa_1st);
            pool.terms_scale_indsa_undag_gpu().emplace_back(d_alfa_2nd);
            pool.terms_scale_indsb_dag_gpu().emplace_back(d_beta_1st);
            pool.terms_scale_indsb_undag_gpu().emplace_back(d_beta_2nd);

            if((anna.size() != crea.size()) or (annb.size() != creb.size())){
                throw std::runtime_error("must be same number of alpha anihilators/creators and beta anihilators/creators.");
            }

            int counta = 0;
            int countb = 0;

            // DAG mappings
            if (need_complex) {
                graph_.make_mapping_each_pre_gpu_complex(
                    true,
                    anna,
                    crea,
                    &counta,
                    pool.terms_paritya_dag_gpu());

                // std::cout << "sourcea1 " << pool.terms_sourcea_dag_gpu().str() << std::endl;  // device_vector has no .str() method
            } else if (real_only) {
                graph_.make_mapping_each_pre_gpu_real(
                    true,
                    anna,
                    crea,
                    &counta,
                    pool.terms_paritya_dag_re_gpu());
            }

            if (need_complex) {
                graph_.make_mapping_each_pre_gpu_complex(
                    false,
                    annb,
                    creb,
                    &countb,
                    pool.terms_parityb_dag_gpu());
            } else if (real_only) {
                graph_.make_mapping_each_pre_gpu_real(
                    false,
                    annb,
                    creb,
                    &countb,
                    pool.terms_parityb_dag_re_gpu());
            }

            if((anna.size() != crea.size()) or (annb.size() != creb.size())){
                throw std::runtime_error("must be same number of alpha anihilators/creators and beta anihilators/creators.");
            }

            // UNDAG mappings
            counta = 0;
            countb = 0;

            if (need_complex) {
                graph_.make_mapping_each_pre_gpu_complex(
                    true,
                    crea,
                    anna,
                    &counta,
                    pool.terms_paritya_undag_gpu());
            } else if (real_only) {
                graph_.make_mapping_each_pre_gpu_real(
                    true,
                    crea,
                    anna,
                    &counta,
                    pool.terms_paritya_undag_re_gpu());
            }

            if (need_complex) {
                graph_.make_mapping_each_pre_gpu_complex(
                    false,
                    creb,
                    annb,
                    &countb,
                    pool.terms_parityb_undag_gpu());
            } else if (real_only) {
                graph_.make_mapping_each_pre_gpu_real(
                    false,
                    creb,
                    annb,
                    &countb,
                    pool.terms_parityb_undag_re_gpu());
            }

            // Classify kernel kind for specialized dot dispatch
            if (creb.empty() && annb.empty()) {
                pool.dot_kernel_kinds().push_back(DotKernelKind::AlphaOnly);
            } else if (crea.empty() && anna.empty()) {
                pool.dot_kernel_kinds().push_back(DotKernelKind::BetaOnly);
            } else {
                pool.dot_kernel_kinds().push_back(DotKernelKind::Mixed);
            }

        } else {
            throw std::invalid_argument("Evolved state must remain in spin and particle-number symmetry sector");
        }

        // === Precompute scalar coefficients for dot_sqop_from_pool_gpu ===
        // Source, target, and parity are reused from the evolution arrays above:
        //   dag  term (excitation):   source = scale_inds_undag, target = scale_inds_dag,
        //                             parity = parity_undag  (built with make_mapping_each_pre(crea,anna))
        //   undag term (de-excit.):   source = scale_inds_dag,   target = scale_inds_undag,
        //                             parity = parity_dag    (built with make_mapping_each_pre(anna,crea))
        // Only the parity_sort * inner_coeff scalars are genuinely new per operator.
        {
            // Term 0 (dag / excitation)
            auto term0 = sqop.terms()[0];
            std::vector<size_t> ops0a(std::get<1>(term0));
            std::vector<size_t> ops0b(std::get<2>(term0));
            ops0a.insert(ops0a.end(), ops0b.begin(), ops0b.end());
            int nswaps0 = parity_sort(ops0a);
            std::complex<double> c0 = std::pow(-1, nswaps0) * std::get<0>(term0);

            // Term 1 (undag / de-excitation)
            auto term1 = sqop.terms()[1];
            std::vector<size_t> ops1a(std::get<1>(term1));
            std::vector<size_t> ops1b(std::get<2>(term1));
            ops1a.insert(ops1a.end(), ops1b.begin(), ops1b.end());
            int nswaps1 = parity_sort(ops1a);
            std::complex<double> c1 = std::pow(-1, nswaps1) * std::get<0>(term1);

            if (need_complex) {
                pool.dot_coeff_dag().push_back(make_cuDoubleComplex(c0.real(), c0.imag()));
                pool.dot_coeff_undag().push_back(make_cuDoubleComplex(c1.real(), c1.imag()));
            } else {
                pool.dot_coeff_dag_re().push_back(std::real(c0));
                pool.dot_coeff_undag_re().push_back(std::real(c1));
            }
        }

    }

    pool.set_device_vecs_populated(true);
}

/* New methods for copying out data */
void FCIComputerGPU::copy_to_tensor_cpu(Tensor& tensor) const
{
    cpu_error();
    C_.copy_to_tensor(tensor);
}

void FCIComputerGPU::copy_to_tensor_thrust_gpu(TensorGPU& tensor) const
{
    gpu_error();
    tensor.copy_in_gpu(C_);
}

void FCIComputerGPU::copy_to_tensor_thrust_cpu(TensorGPU& tensor) const
{
    cpu_error();
    tensor.copy_in(C_);
}
