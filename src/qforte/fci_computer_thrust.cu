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
#include "fci_graph_thrust.h"

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

    // Initialize the FCI graph tensors
    sourcea_gpu_ = thrust::device_vector<int>(nalfa_strs_);
    targeta_gpu_ = thrust::device_vector<int>(nalfa_strs_);
    paritya_gpu_ = thrust::device_vector<cuDoubleComplex>(nalfa_strs_);
    sourceb_gpu_ = thrust::device_vector<int>(nbeta_strs_);
    targetb_gpu_ = thrust::device_vector<int>(nbeta_strs_);
    parityb_gpu_ = thrust::device_vector<cuDoubleComplex>(nbeta_strs_);

    graph_ = FCIGraphThrust(nalfa_el_, nbeta_el_, norb_);

    // timer_ = local_timer();
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
    apply_array_1bdy_cpu(
        Cnew,
        graph_.read_dexca_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexca(),
        h1e,
        norb_,
        true);

    apply_array_1bdy_cpu(
        Cnew,
        graph_.read_dexcb_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexcb(),
        h1e,
        norb_,
        false);

    // Apply two-body terms (same spin)
    lm_apply_array12_same_spin_opt_cpu(
        Cnew,
        graph_.read_dexca_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexca(),
        h1e,
        h2e,
        norb_,
        true);

    lm_apply_array12_same_spin_opt_cpu(
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
    lm_apply_array12_diff_spin_opt_cpu(
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

/// TODO: Not implemented in GPU so skipping
/*
void FCIComputerThrust::apply_tensor_operator(const TensorOperator& top)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("");
}*/

/// TODO: this is commented out in FCIComputerGPU, so skipping
/*
void FCIComputerThrust::apply_tensor_spin_12bdy(
    const TensorThrust& h1e, 
    const TensorThrust& h2e, 
    size_t norb)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("");
}*/

/// TODO: in GPU header but not implemented in GPU source
/*
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
    throw std::runtime_error("");
}*/

void FCIComputerThrust::apply_array_1bdy_cpu(
    TensorThrust& out,
    const std::vector<int>& dexc,
    const int astates,
    const int bstates,
    const int ndexc,
    const TensorThrust& h1e,
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

void FCIComputerThrust::lm_apply_array12_same_spin_opt_cpu(
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
    cpu_error();

    const int states1 = is_alpha ? alpha_states : beta_states;
    const int states2 = is_alpha ? beta_states : alpha_states;
    const int inc1 = is_alpha ? beta_states : 1;
    const int inc2 = is_alpha ? 1 : beta_states;

    std::vector<std::complex<double>> temp(states1, 0.0);

    for (int s1 = 0; s1 < states1; ++s1) {
        std::fill(temp.begin(), temp.end(), 0.0);
        const int *cdexc = dexc.data() + 3 * s1 * ndexc;
        const int *lim1 = cdexc + 3 * ndexc;
        std::complex<double> *cout = out.data().data() + s1 * inc1;

        for (; cdexc < lim1; cdexc = cdexc + 3) {
            const int s2 = cdexc[0];
            const int ijshift = cdexc[1];
            const int parity1 = cdexc[2];
            const int *cdexc2 = dexc.data() + 3 * s2 * ndexc;
            const int *lim2 = cdexc2 + 3 * ndexc;
            const int h2e_id = ijshift * norbs * norbs;
            const std::complex<double> *h2etmp = h2e.read_h_data().data() + h2e_id;
            temp[s2] += static_cast<double>(parity1) * h1e.read_h_data()[ijshift];

            for (; cdexc2 < lim2; cdexc2 += 3) {
                const int target = cdexc2[0];
                const int klshift = cdexc2[1];
                const int parity = cdexc2[2] * parity1;
                const std::complex<double> pref = static_cast<double>(parity) * h2etmp[klshift];
                temp[target] += pref;
            }
        }
        const std::complex<double> *xptr = C_.data().data();
        for (int ii = 0; ii < states1; ii++) {
            const std::complex<double> ttt = temp[ii];
            math_zaxpy(states2, ttt, xptr, inc2, cout, inc2);
            xptr += inc1;
        }
    }
}

void FCIComputerThrust::lm_apply_array12_diff_spin_opt_cpu(
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
    cpu_error();

    const int nadexc_tot = alpha_states * nadexc;
    const int norbs2 = norbs * norbs;
    const int one = 1;

    std::vector<int> signs(nadexc_tot);
    std::vector<int> coff(nadexc_tot);
    std::vector<int> boff(nadexc_tot);

    int nest = 0;
    for (int s1 = 0; s1 < alpha_states; ++s1) {
        for (int i = 0; i < nadexc; ++i) {
            const int orbij = adexc[3 * (s1 * nadexc + i) + 1];
            if (orbij == 0) ++nest;
        }
    }

    std::vector<std::complex<double>> vtemp(nest);
    std::vector<std::complex<double>> ctemp(nest * alpha_states);

    for (int orbid = 0; orbid < norbs2; ++orbid) {
        int nsig = 0;
        for (int s1 = 0; s1 < alpha_states; ++s1) {
            for (int i = 0; i < nbdexc; ++i) {
                const int orbij = adexc[3 * (s1 * nadexc + i) + 1];
                if (orbij == orbid) {
                    signs[nsig] = adexc[3 * (s1 * nadexc + i) + 2];
                    coff[nsig] = adexc[3 * (s1 * nadexc + i)];
                    boff[nsig] = s1;
                    ++nsig;
                }
            }
        }

        std::fill(ctemp.begin(), ctemp.end(), std::complex<double>(0.0));

        for (int isig = 0; isig < nsig; ++isig) {
            const int offset = coff[isig];
            const std::complex<double> *cptr = C_.data().data() + offset * beta_states;
            std::complex<double> *tptr = ctemp.data() + isig;
            const std::complex<double> zsign = signs[isig];
            math_zaxpy(beta_states, zsign, cptr, one, tptr, nsig);
        }

        const std::complex<double> *tmperi = h2e.read_h_data().data() + orbid * norbs2;

        for (int s2 = 0; s2 < beta_states; ++s2) {
            
            // TODO(Tyler): need for open mp
            // const int ithrd = 0;
            // const std::complex<double> *vpt = vtemp.data() + ithrd * nsig;
            // for (int kk = 0; kk < nsig; ++kk) vpt[kk] = 0.0;

            std::fill(vtemp.begin(), vtemp.begin() + nsig, std::complex<double>(0.0));
            

            for (int j = 0; j < nbdexc; ++j) {
                int idx2 = bdexc[3 * (s2 * nbdexc + j)];
                const int parity = bdexc[3 * (s2 * nbdexc + j) + 2];
                const int orbkl = bdexc[3 * (s2 * nbdexc + j) + 1];
                const std::complex<double> ttt = std::complex<double>(parity, 0.0) * tmperi[orbkl];
                const std::complex<double> *cctmp = ctemp.data() + idx2 * nsig;
                math_zaxpy(nsig, ttt, cctmp, one, vtemp.data(), one);
            }

            std::complex<double> *tmpout = out.data().data() + s2;
            for (int isig = 0; isig < nsig; ++isig) {
                tmpout[beta_states * boff[isig]] += vtemp[isig];
            }
        }
    }
}

/// TODO: Not implemented in GPU so skipping
/*
std::pair<TensorThrust, TensorThrust> FCIComputerThrust::calculate_dvec_spin_with_coeff()
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("");
}*/

TensorThrust FCIComputerThrust::calculate_coeff_spin_with_dvec_cpu(std::pair<TensorThrust, TensorThrust>& dvec)
{
    cpu_error();

    TensorThrust Cnew({nalfa_strs_, nbeta_strs_}, "Cnew");

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

std::pair<std::vector<int>, std::vector<int>> FCIComputerThrust::evaluate_map_number_cpu(
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

std::pair<std::vector<int>, std::vector<int>> FCIComputerThrust::evaluate_map_cpu(
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

void FCIComputerThrust::apply_cos_inplace_cpu(
    const std::complex<double> time,
    const std::complex<double> coeff,
    const std::vector<int>& crea,
    const std::vector<int>& anna,
    const std::vector<int>& creb,
    const std::vector<int>& annb,
    TensorThrust& Cout)
{
    /// TODO: Implement separate CPU and GPU versions of this function
    // cpu_error();

    // bool reset = false;
    // if (Cout.on_gpu()) {
    //     reset = true;
    //     Cout.to_cpu();
    // }

    timer_.acc_begin("===>hard: apply cos setup");

    const std::complex<double> cabs = std::abs(coeff);
    const std::complex<double> factor = std::cos(time * cabs);
    cuDoubleComplex factor_gpu = make_cuDoubleComplex(factor.real(), factor.imag());

    std::pair<std::vector<int>, std::vector<int>> maps = evaluate_map_cpu(crea, anna, creb, annb);
    thrust::device_vector<int> d_first(maps.first.begin(), maps.first.end());
    thrust::device_vector<int> d_second(maps.second.begin(), maps.second.end());

    timer_.acc_end("===>hard: apply cos setup");

    timer_.acc_begin("===>hard: apply cos kernal");
    scale_elements_wrapper(
        thrust::raw_pointer_cast(Cout.d_data().data()),
        thrust::raw_pointer_cast(d_first.data()), 
        d_first.size(),
        thrust::raw_pointer_cast(d_second.data()), 
        d_second.size(),
        nbeta_strs_,
        factor_gpu);
    timer_.acc_end("===>hard: apply cos kernal");

    // if (reset) {
    //     Cout.to_gpu();
    // }
}

int FCIComputerThrust::isolate_number_operators_cpu(
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
void FCIComputerThrust::evolve_individual_nbody_easy_cpu(
    const std::complex<double> time,
    const std::complex<double> coeff,
    TensorThrust& Cin,
    TensorThrust& Cout,
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
        scale_elements_wrapper(
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
        scale_elements_wrapper(
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
void FCIComputerThrust::evolve_individual_nbody_hard_cpu(
    const std::complex<double> time,
    const std::complex<double> coeff,
    TensorThrust& Cin,
    TensorThrust& Cout,
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

        const std::complex<double> cabs = std::abs(coeff);
        const std::complex<double> factor = std::cos(time * cabs);
        cuDoubleComplex factor_gpu = make_cuDoubleComplex(factor.real(), factor.imag());

        timer_.acc_begin("===>hard: apply cos kernal");
        
        scale_elements_wrapper(
            thrust::raw_pointer_cast(Cout.d_data().data()),
            thrust::raw_pointer_cast(std::get<2>(*precomp).data()), 
            std::get<2>(*precomp).size(),
            thrust::raw_pointer_cast(std::get<4>(*precomp).data()), 
            std::get<4>(*precomp).size(),
            nbeta_strs_,
            factor_gpu);

        scale_elements_wrapper(
            thrust::raw_pointer_cast(Cout.d_data().data()),
            thrust::raw_pointer_cast(std::get<3>(*precomp).data()), 
            std::get<3>(*precomp).size(),
            thrust::raw_pointer_cast(std::get<5>(*precomp).data()), 
            std::get<5>(*precomp).size(),
            nbeta_strs_,
            factor_gpu);

        timer_.acc_end("===>hard: apply cos kernal");


        if ((std::get<10>(*precomp).size() != std::get<6>(*precomp).size()) or (std::get<6>(*precomp).size() != std::get<14>(*precomp).size())) {
            throw std::runtime_error("The sizes of atarget, asource, and aparity must be the same.");
        }

        if ((std::get<12>(*precomp).size() != std::get<8>(*precomp).size()) or (std::get<8>(*precomp).size() != std::get<16>(*precomp).size())) {
            throw std::runtime_error("The sizes of atarget, asource, and aparity must be the same.");
        }

        std::complex<double> coeff_dag = work_cof * sinfactor;

        cuDoubleComplex cu_coeff_dag = make_cuDoubleComplex(coeff_dag.real(), coeff_dag.imag());

        // Call the GPU kernel using thrust raw pointers directly
        apply_individual_nbody1_accumulate_wrapper(
            cu_coeff_dag, 
            thrust::raw_pointer_cast(Cin.read_d_data().data()), 
            thrust::raw_pointer_cast(Cout.d_data().data()), 
            thrust::raw_pointer_cast(std::get<6>(*precomp).data()),
            thrust::raw_pointer_cast(std::get<10>(*precomp).data()),
            thrust::raw_pointer_cast(std::get<14>(*precomp).data()),
            thrust::raw_pointer_cast(std::get<8>(*precomp).data()),
            thrust::raw_pointer_cast(std::get<12>(*precomp).data()),
            thrust::raw_pointer_cast(std::get<16>(*precomp).data()),
            nbeta_strs_,
            std::get<6>(*precomp).size(),
            std::get<8>(*precomp).size(),
            Cin.size() * sizeof(cuDoubleComplex));

        cudaError_t error1 = cudaGetLastError();
        if (error1 != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error1) << std::endl;
            throw std::runtime_error("Failed to execute the apply_individual_nbody1_accumulate operation on the GPU.");
        }

        if ((std::get<11>(*precomp).size() != std::get<7>(*precomp).size()) or (std::get<7>(*precomp).size() != std::get<15>(*precomp).size())) {
            throw std::runtime_error("The sizes of atarget, asource, and aparity must be the same.");
        }

        if ((std::get<13>(*precomp).size() != std::get<9>(*precomp).size()) or (std::get<9>(*precomp).size() != std::get<17>(*precomp).size())) {
            throw std::runtime_error("The sizes of atarget, asource, and aparity must be the same.");
        }

        std::complex<double> coeff_undag = coeff * std::complex<double>(0.0, -1.0) * sinfactor;

        cuDoubleComplex cu_coeff_undag = make_cuDoubleComplex(coeff_undag.real(), coeff_undag.imag());

        // Call the GPU kernel using thrust raw pointers directly
        apply_individual_nbody1_accumulate_wrapper(
            cu_coeff_undag, 
            thrust::raw_pointer_cast(Cin.read_d_data().data()), 
            thrust::raw_pointer_cast(Cout.d_data().data()), 
            thrust::raw_pointer_cast(std::get<7>(*precomp).data()),
            thrust::raw_pointer_cast(std::get<11>(*precomp).data()),
            thrust::raw_pointer_cast(std::get<15>(*precomp).data()),
            thrust::raw_pointer_cast(std::get<9>(*precomp).data()),
            thrust::raw_pointer_cast(std::get<13>(*precomp).data()),
            thrust::raw_pointer_cast(std::get<17>(*precomp).data()),
            nbeta_strs_,
            std::get<7>(*precomp).size(),
            std::get<9>(*precomp).size(),
            Cin.size() * sizeof(cuDoubleComplex));

        cudaError_t error2 = cudaGetLastError();
        if (error2 != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error2) << std::endl;
            throw std::runtime_error("Failed to execute the apply_individual_nbody1_accumulate operation on the GPU.");
        }

        
    } else {
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

/// NOTE: Cin should be const, changing for now
void FCIComputerThrust::evolve_individual_nbody_cpu(
    const std::complex<double> time,
    const SQOperator& sqop,
    TensorThrust& Cin,
    TensorThrust& Cout,
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

// NOTE(Nick): The trotter function should directly call evolve_individual_nbody_cpu so we don't 
// need to re-initialize Cin for each mu index, only copy, is currently a big 
// performace hit!
void FCIComputerThrust::apply_sqop_evolution_gpu(
    const std::complex<double> time,
    const SQOperator& sqop,
    const bool antiherm,
    const bool adjoint)
{
    gpu_error();

    timer_.acc_begin("=>copy in Cin <- C_");

    TensorThrust Cin(C_.shape(), "Cin", true);
    Cin.copy_in_gpu(C_);

    timer_.acc_end("=>copy in Cin <- C_");

    // NOTE(Nick): needs gpu treatment
    evolve_individual_nbody_cpu(
        time,
        sqop,
        Cin,
        C_,
        antiherm,
        adjoint); 
}

void FCIComputerThrust::evolve_pool_trotter_basic_gpu(
    const SQOpPool& pool,
    const bool antiherm,
    const bool adjoint)
{
    gpu_error();

    if(adjoint){
        for (int i = pool.terms().size() - 1; i >= 0; --i) {
            apply_sqop_evolution_gpu(
                pool.terms()[i].first, 
                pool.terms()[i].second,
                antiherm,
                adjoint);
        }
    } else {
        for (const auto& sqop_term : pool.terms()) {
            apply_sqop_evolution_gpu(
                sqop_term.first, 
                sqop_term.second,
                antiherm,
                adjoint);
            }
    }
}

void FCIComputerThrust::evolve_pool_trotter_gpu(
    const SQOpPool& pool,
    const double evolution_time,
    const int trotter_steps,
    const int trotter_order,
    const bool antiherm,
    const bool adjoint)
{
    gpu_error();

    timer_.acc_begin("evolve_pool_trotter_gpu(outer)");

    if(trotter_order == 1){

        std::complex<double> prefactor = evolution_time / static_cast<std::complex<double>>(trotter_steps);

        if(adjoint){
            for( int r = 0; r < trotter_steps; r++) {
                for (int i = pool.terms().size() - 1; i >= 0; --i) {
                    apply_sqop_evolution_gpu(
                        prefactor * pool.terms()[i].first, 
                        pool.terms()[i].second,
                        antiherm,
                        adjoint);
                }
            }
                

        } else {
            for( int r = 0; r < trotter_steps; r++) {
                for (const auto& sqop_term : pool.terms()) {
                    apply_sqop_evolution_gpu(
                        prefactor * sqop_term.first, 
                        sqop_term.second,
                        antiherm,
                        adjoint);
                }
            }
        }

    } else if (trotter_order == 2 ) {
        std::complex<double> prefactor = 0.5 * evolution_time / static_cast<std::complex<double>>(trotter_steps);

        if(adjoint){
            for( int r = 0; r < trotter_steps; r++) {
                for (int i = pool.terms().size() - 1; i >= 0; --i) {
                    (
                        prefactor * pool.terms()[i].first, 
                        pool.terms()[i].second,
                        antiherm,
                        adjoint);
                }

                for (const auto& sqop_term : pool.terms()) {
                    (
                        prefactor * sqop_term.first, 
                        sqop_term.second,
                        antiherm,
                        adjoint);
                }
            }
                

        } else {
            for( int r = 0; r < trotter_steps; r++) {
                for (const auto& sqop_term : pool.terms()) {
                    (
                        prefactor * sqop_term.first, 
                        sqop_term.second,
                        antiherm,
                        adjoint);
                }

                for (int i = pool.terms().size() - 1; i >= 0; --i) {
                    (
                        prefactor * pool.terms()[i].first, 
                        pool.terms()[i].second,
                        antiherm,
                        adjoint);
                }
            }
        }
    } else {
        throw std::runtime_error("Higher than 2nd order trotter not yet implemented"); 
    }

    timer_.acc_end("evolve_pool_trotter_gpu(outer)");
}

void FCIComputerThrust::evolve_pool_trotter_gpu_v2(
    const SQOpPool& pool,
    const double evolution_time,
    const int trotter_steps,
    const int trotter_order,
    const bool antiherm,
    const bool adjoint)
{
    gpu_error();

    timer_.acc_begin("evolve_pool_trotter_gpu(outer)");

    TensorThrust Cin(C_.shape(), "Cin", true);

    if(trotter_order == 1){

        std::complex<double> prefactor = evolution_time / static_cast<std::complex<double>>(trotter_steps);

        if(adjoint){
            for( int r = 0; r < trotter_steps; r++) {
                for (int i = pool.terms().size() - 1; i >= 0; --i) {
                    
                    timer_.acc_begin("=>copy in Cin <- C_");
                    Cin.copy_in_gpu(C_);
                    timer_.acc_end("=>copy in Cin <- C_");

                    evolve_individual_nbody_cpu(
                        prefactor * pool.terms()[i].first,
                        pool.terms()[i].second,
                        Cin,
                        C_,
                        antiherm,
                        adjoint); 

                }
            }
                

        } else {
            for( int r = 0; r < trotter_steps; r++) {
                for (const auto& sqop_term : pool.terms()) {

                    timer_.acc_begin("=>copy in Cin <- C_");
                    Cin.copy_in_gpu(C_);
                    timer_.acc_end("=>copy in Cin <- C_");

                    evolve_individual_nbody_cpu(
                        prefactor * sqop_term.first,
                        sqop_term.second,
                        Cin,
                        C_,
                        antiherm,
                        adjoint); 

                }
            }
        }

    }  else {
        throw std::runtime_error("Higher than 1st order trotter not yet implemented"); 
    }

    timer_.acc_end("evolve_pool_trotter_gpu(outer)");
}

// adds useage of pre-computed stp device arrays
void FCIComputerThrust::evolve_pool_trotter_gpu_v3(
    const SQOpPoolThrust& pool,
    double evolution_time,
    int trotter_steps,
    int trotter_order,
    bool antiherm,
    bool adjoint)
{
    gpu_error();

    timer_.acc_begin("evolve_pool_trotter_gpu(outer)");

    TensorThrust Cin(C_.shape(), "Cin", true);

    if(trotter_order == 1){

        std::complex<double> prefactor = evolution_time / static_cast<std::complex<double>>(trotter_steps);

        if(adjoint){
            for( int r = 0; r < trotter_steps; r++) {
                for (int i = pool.terms().size() - 1; i >= 0; --i) {
                    
                    timer_.acc_begin("=>copy in Cin <- C_");
                    Cin.copy_in_gpu(C_);
                    timer_.acc_end("=>copy in Cin <- C_");

                    const auto& device_spt_arys = pool.get_mu_tuple(i);

                    evolve_individual_nbody_cpu(
                        prefactor * pool.terms()[i].first,
                        pool.terms()[i].second,
                        Cin,
                        C_,
                        antiherm,
                        adjoint,
                        &device_spt_arys); 

                }
            }
                

        } else {
            for( int r = 0; r < trotter_steps; r++) {
                for (int i = 0; i < pool.terms().size(); ++i) {

                    timer_.acc_begin("=>copy in Cin <- C_");
                    Cin.copy_in_gpu(C_);
                    timer_.acc_end("=>copy in Cin <- C_");

                    const auto& device_spt_arys = pool.get_mu_tuple(i);

                    evolve_individual_nbody_cpu(
                        prefactor * pool.terms()[i].first,
                        pool.terms()[i].second,
                        Cin,
                        C_,
                        antiherm,
                        adjoint,
                        &device_spt_arys); 

                }
            }
        }

    }  else {
        throw std::runtime_error("Higher than 1st order trotter not yet implemented"); 
    }

    timer_.acc_end("evolve_pool_trotter_gpu(outer)");
}

void FCIComputerThrust::evolve_op_taylor_cpu(
    const SQOperator& op,
    const double evolution_time,
    const double convergence_thresh,
    const int max_taylor_iter)
{
    cpu_error();

    TensorThrust Cevol = C_;

    for (int order = 1; order < max_taylor_iter; ++order) {

        // std::cout << "I get here, order: " << order << std::endl;

        // std::cout << "C_: " << C_.str() << std::endl;
        // std::cout << "Cevol: " << Cevol.str() << std::endl;

        std::complex<double> coeff(0.0, -evolution_time);
        apply_sqop_gpu(op);
        scale_cpu(coeff);

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
void FCIComputerThrust::apply_individual_nbody1_accumulate_gpu(
    const std::complex<double> coeff, 
    TensorThrust& Cin,
    TensorThrust& Cout,
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
void FCIComputerThrust::apply_individual_nbody_accumulate_gpu(
    const std::complex<double> coeff,
    TensorThrust& Cin,
    TensorThrust& Cout,
    const std::vector<int>& daga,
    const std::vector<int>& undaga, 
    const std::vector<int>& dagb,
    const std::vector<int>& undagb)
{
    gpu_error();

    if((daga.size() != undaga.size()) or (dagb.size() != undagb.size())){
        throw std::runtime_error("must be same number of alpha anihilators/creators and beta anihilators/creators.");
    }

    // local_timer my_timer = local_timer();
    // timer_.reset();

    timer_.acc_begin("===>hard nbody acc setup");

    int counta = 0;
    int countb = 0;

    graph_.make_mapping_each_gpu_v2(
        true,
        daga,
        undaga,
        &counta,
        sourcea_gpu_,
        targeta_gpu_,
        paritya_gpu_);

    // timer_.acc_record("first 'make_mapping_each' in apply_individual_nbody_accumulate");
    // timer_.reset();

    if (counta == 0) {
        return;
    }

    graph_.make_mapping_each_gpu_v2(
        false,
        dagb,
        undagb,
        &countb,
        sourceb_gpu_,
        targetb_gpu_,
        parityb_gpu_);

    // timer_.acc_record("second 'make_mapping_each' in apply_individual_nbody_accumulate");
    // timer_.reset();

    if (countb == 0) {
        return;
    }

    // timer_.acc_record("second for loop in apply_individual_nbody_accumulate");
    // timer_.reset();

    // thrust::host_vector<int> sourcea_cpu(counta);
    // thrust::host_vector<int> targeta_cpu(counta);
    // thrust::host_vector<cuDoubleComplex> paritya_cpu(counta);

    // thrust::host_vector<int> sourceb_cpu(countb);
    // thrust::host_vector<int> targetb_cpu(countb);
    // thrust::host_vector<cuDoubleComplex> parityb_cpu(countb);

    // thrust::copy(sourcea_gpu_.begin(), sourcea_gpu_.begin() + counta, sourcea_cpu.begin());
    // thrust::copy(targeta_gpu_.begin(), targeta_gpu_.begin() + counta, targeta_cpu.begin());
    // thrust::copy(paritya_gpu_.begin(), paritya_gpu_.begin() + counta, paritya_cpu.begin());

    // thrust::copy(sourceb_gpu_.begin(), sourceb_gpu_.begin() + countb, sourceb_cpu.begin());
    // thrust::copy(targetb_gpu_.begin(), targetb_gpu_.begin() + countb, targetb_cpu.begin());
    // thrust::copy(parityb_gpu_.begin(), parityb_gpu_.begin() + countb, parityb_cpu.begin());

    // print_vector_thrust(sourcea_cpu, "sourcea");
    // print_vector_thrust(targeta_cpu, "targeta");
    // print_vector_thrust_cuDoubleComplex(paritya_cpu, "paritya");
    // print_vector_thrust(sourceb_cpu, "sourceb");
    // print_vector_thrust(targetb_cpu, "targetb");
    // print_vector_thrust_cuDoubleComplex(parityb_cpu, "parityb");

    timer_.acc_end("===>hard nbody acc setup");


    timer_.acc_begin("===>hard nbody acc kernel");
    /// TODO: changing this function to use private members of FCIComputerThrust
    apply_individual_nbody1_accumulate_gpu(
        coeff, 
        Cin,
        Cout,
        counta,
        countb);

    timer_.acc_end("===>hard nbody acc kernel");
}

/// NOTE: Cin should be const, changing for now
void FCIComputerThrust::apply_individual_sqop_term_gpu(
    const std::tuple< std::complex<double>, std::vector<size_t>, std::vector<size_t>>& term,
    TensorThrust& Cin,
    TensorThrust& Cout)
{
    std::vector<int> crea;
    std::vector<int> anna;

    std::vector<int> creb;
    std::vector<int> annb;

    local_timer my_timer = local_timer();
    // timer_.reset();

    for(size_t i = 0; i < std::get<1>(term).size(); i++){
        if(std::get<1>(term)[i]%2 == 0){
            crea.push_back(std::floor(std::get<1>(term)[i] / 2));
        } else {
            creb.push_back(std::floor(std::get<1>(term)[i] / 2));
        }
    }

    // timer_.acc_record("first loop in apply_individual_sqop_term");
    // timer_.reset();

    for(size_t i = 0; i < std::get<2>(term).size(); i++){
        if(std::get<2>(term)[i]%2 == 0){
            anna.push_back(std::floor(std::get<2>(term)[i] / 2));
        } else {
            annb.push_back(std::floor(std::get<2>(term)[i] / 2));
        }
    }

    // timer_.acc_record("second loop in apply_individual_sqop_term");
    // timer_.reset();

    if (std::get<1>(term).size() != std::get<2>(term).size()) {
        throw std::invalid_argument("Each term must have same number of anihilators and creators");
    }   

    std::vector<size_t> ops1(std::get<1>(term));
    std::vector<size_t> ops2(std::get<2>(term));
    ops1.insert(ops1.end(), ops2.begin(), ops2.end());

    int nswaps = parity_sort(ops1);
    // timer_.acc_record("some parity things");
    // timer_.reset();
    // std::cout << my_timer.str_table() << std::endl;

    apply_individual_nbody_accumulate_gpu(
        pow(-1, nswaps) * std::get<0>(term),
        Cin,
        Cout,
        crea,
        anna, 
        creb,
        annb);
}

void FCIComputerThrust::apply_sqop_gpu(const SQOperator& sqop)
{
     C_.gpu_error();
    TensorThrust Cin(C_.shape(), "Cin", true);
    Cin.copy_in_gpu(C_);


    
    local_timer my_timer = local_timer();
    // timer_.reset();
    
    // cudaMemcpy(Cin.d_data(), C_.d_data(), Cin.size() * sizeof(cuDoubleComplex))

    C_.zero_gpu();

    // timer_.acc_record("making tensor things");
    // timer_.reset();

    for (const auto& term : sqop.terms()) {
        if(std::abs(std::get<0>(term)) > compute_threshold_){
        apply_individual_sqop_term_gpu(
            term,
            Cin,
            C_);
        }
    }

    // timer_.acc_record("first for loop in apply_sqop");
    // std::cout << // timer_.acc_str_table() << std::endl;
}

void FCIComputerThrust::apply_diagonal_of_sqop_cpu(
    const SQOperator& sq_op, 
    const bool invert_coeff)
{
    cpu_error();

    TensorThrust Cin = C_;
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

void FCIComputerThrust::apply_sqop_pool_cpu(const SQOpPool& sqop_pool)
{
    cpu_error();

    TensorThrust Cin = C_;
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

std::complex<double> FCIComputerThrust::get_exp_val_cpu(const SQOperator& sqop)
{
    TensorThrust Cin = C_;
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

std::complex<double> FCIComputerThrust::get_exp_val_tensor_cpu(
    const std::complex<double> h0e, 
    const TensorThrust& h1e, 
    const TensorThrust& h2e, 
    const TensorThrust& h2e_einsum, 
    size_t norb)
{
    TensorThrust Cin = C_;

    apply_tensor_spat_012bdy(
        h0e,
        h1e, 
        h2e, 
        h2e_einsum, 
        norb
    );

    std::complex<double> val = C_.vector_dot(Cin);

    C_ = Cin;
    return val;
}

void FCIComputerThrust::scale_cpu(const std::complex<double> a)
{
    C_.scale(a);
}

/// TODO: This is commented out in TensorGPU
/*
std::vector<double> FCIComputerThrust::direct_expectation_value(const TensorOperator& top)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("");
}*/

/// TODO: Not implemented in TensorGPU
/*
std::complex<double> FCIComputerThrust::coeff(const QubitBasis& abasis, const QubitBasis& bbasis)
{
    // Implementation would be similar to FCIComputerGPU but using TensorThrust
    // This is a placeholder - full implementation would need to be added
    throw std::runtime_error("");
}*/

void FCIComputerThrust::set_state_cpu(const TensorThrust& other_state)
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

void FCIComputerThrust::set_state_from_tensor_cpu(const Tensor& other_state)
{
    cpu_error();
    C_.copy_in_from_tensor(other_state);
}

void FCIComputerThrust::zero_cpu()
{
    cpu_error();
    C_.zero();
}

void FCIComputerThrust::hartree_fock_cpu()
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

void FCIComputerThrust::print_vector_thrust(const thrust::host_vector<int>& vec, const std::string& name)
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

void FCIComputerThrust::print_vector_thrust_cuDoubleComplex(const thrust::host_vector<cuDoubleComplex>& vec, const std::string& name)
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

void FCIComputerThrust::populate_index_arrays_for_pool_evo(SQOpPoolThrust& pool){
    
    if(pool.device_vecs_populated()){
        return;
    }

    for (int i=0; i<pool.terms().size(); ++i){

        auto sqop = pool.terms()[i].second;

        if (sqop.terms().size() != 2) {
            std::cout << "This sqop has " << sqop.terms().size() << " terms." << std::endl;
            throw std::invalid_argument("Individual n-body code is called with multiple terms");
        }

        // append h_mu
        pool.outer_coeffs().push_back(pool.terms()[i].first);

        /// NICK: TODO, implement a hermitian check, at least for two term SQOperators
        // sqop.hermitian_check();

        auto term = sqop.terms()[0];

        if(std::abs(std::get<0>(term)) < compute_threshold_){
            continue;
        }

        /// append c_mu
        pool.inner_coeffs().push_back(std::get<0>(term));

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

        if (crea == anna && creb == annb) {
            // std::cout << "Made it to easy" << std::endl;

            // evolve_individual_nbody_easy_cpu(
            //     time,
            //     parity * std::get<0>(term), 
            //     Cin,
            //     Cout,
            //     crea,
            //     anna, 
            //     creb,
            //     annb);

            std::pair<std::vector<int>, std::vector<int>> maps = evaluate_map_number_cpu(anna, annb);
            thrust::device_vector<int> d_alfa(maps.first.begin(), maps.first.end());
            thrust::device_vector<int> d_beta(maps.second.begin(), maps.second.end());

            // Indicies for scale inplace
            pool.terms_scale_indsa_dag_gpu().emplace_back(d_alfa);
            pool.terms_scale_indsa_undag_gpu().emplace_back();
            pool.terms_scale_indsb_dag_gpu().emplace_back(d_beta);
            pool.terms_scale_indsb_undag_gpu().emplace_back();

            // Accumulation (source) index maps
            pool.terms_sourcea_dag_gpu().emplace_back();
            pool.terms_sourcea_undag_gpu().emplace_back();
            pool.terms_sourceb_dag_gpu().emplace_back();
            pool.terms_sourceb_undag_gpu().emplace_back();

            // Accumulation (target) index maps
            pool.terms_targeta_dag_gpu().emplace_back();
            pool.terms_targeta_undag_gpu().emplace_back();
            pool.terms_targetb_dag_gpu().emplace_back();
            pool.terms_targetb_undag_gpu().emplace_back();

            // Parity/phase maps
            pool.terms_paritya_dag_gpu().emplace_back();
            pool.terms_paritya_undag_gpu().emplace_back();
            pool.terms_parityb_dag_gpu().emplace_back();
            pool.terms_parityb_undag_gpu().emplace_back();


        } else if (crea.size() == anna.size() && creb.size() == annb.size()) {
            // std::cout << "Made it to hard" << std::endl;

            // evolve_individual_nbody_hard_cpu(
            //     time,
            //     parity * std::get<0>(term),
            //     Cin,
            //     Cout,
            //     crea,
            //     anna, 
            //     creb,
            //     annb);

            std::vector<int> dagworka(crea);
            std::vector<int> dagworkb(creb);
            std::vector<int> undagworka(anna);
            std::vector<int> undagworkb(annb);
            std::vector<int> numbera;
            std::vector<int> numberb;
            
            // int parity = 0;
            // parity += isolate_number_operators_cpu(
            //     crea,
            //     anna,
            //     dagworka,
            //     undagworka,
            //     numbera);

            // parity += isolate_number_operators_cpu(
            //     creb,
            //     annb,
            //     dagworkb,
            //     undagworkb,
            //     numberb);

            // std::complex<double> ncoeff = coeff * std::pow(-1.0, parity);
            // std::complex<double> absol = std::abs(ncoeff);
            // std::complex<double> sinfactor = std::sin(time * absol) / absol;

            std::vector<int> numbera_dagworka(numbera.begin(), numbera.end());
            numbera_dagworka.insert(numbera_dagworka.end(), dagworka.begin(), dagworka.end());

            std::vector<int> numberb_dagworkb(numberb.begin(), numberb.end());
            numberb_dagworkb.insert(numberb_dagworkb.end(), dagworkb.begin(), dagworkb.end());

            std::vector<int> numbera_undagworka(numbera.begin(), numbera.end());
            numbera_undagworka.insert(numbera_undagworka.end(), undagworka.begin(), undagworka.end());

            std::vector<int> numberb_undagworkb(numberb.begin(), numberb.end());
            numberb_undagworkb.insert(numberb_undagworkb.end(), undagworkb.begin(), undagworkb.end());

            // int phase = std::pow(-1, (crea.size() + anna.size()) * (creb.size() + annb.size()));
            // std::complex<double> work_cof = std::conj(coeff) * static_cast<double>(phase) * std::complex<double>(0.0, -1.0);

            // void FCIComputerThrust::apply_cos_inplace_cpu(
            //     const std::complex<double> time,
            //     const std::complex<double> coeff,
            //     const std::vector<int>& crea,
            //     const std::vector<int>& anna,
            //     const std::vector<int>& creb,
            //     const std::vector<int>& annb,
            //     TensorThrust& Cout)

            // apply_cos_inplace_cpu(
            //     time,
            //     ncoeff,
            //     numbera_dagworka,
            //     undagworka,
            //     numberb_dagworkb,
            //     undagworkb,
            //     Cout);

            std::pair<std::vector<int>, std::vector<int>> maps1 = evaluate_map_cpu(
                numbera_dagworka, 
                undagworka, 
                numberb_dagworkb, 
                undagworkb);

            thrust::device_vector<int> d_alfa_1st(maps1.first.begin(), maps1.first.end());
            thrust::device_vector<int> d_beta_1st(maps1.second.begin(), maps1.second.end());

            // apply_cos_inplace_cpu(
            //     time,
            //     ncoeff,
            //     numbera_undagworka,
            //     dagworka,
            //     numberb_undagworkb,
            //     dagworkb,
            //     Cout);

            std::pair<std::vector<int>, std::vector<int>> maps2 = evaluate_map_cpu(
                numbera_undagworka, 
                dagworka, 
                numberb_undagworkb, 
                dagworkb);

            thrust::device_vector<int> d_alfa_2nd(maps2.first.begin(), maps2.first.end());
            thrust::device_vector<int> d_beta_2nd(maps2.second.begin(), maps2.second.end());

            // Indicies for scale inplace
            pool.terms_scale_indsa_dag_gpu().emplace_back(d_alfa_1st);
            pool.terms_scale_indsa_undag_gpu().emplace_back(d_alfa_2nd);
            pool.terms_scale_indsb_dag_gpu().emplace_back(d_beta_1st);
            pool.terms_scale_indsb_undag_gpu().emplace_back(d_beta_2nd);

      
            // apply_individual_nbody_accumulate_gpu(
            //     work_cof * sinfactor,
            //     Cin,
            //     Cout, 
            //     anna,
            //     crea,
            //     annb,
            //     creb);

            // void FCIComputerThrust::apply_individual_nbody_accumulate_gpu(
            // const std::complex<double> coeff,
            // TensorThrust& Cin,
            // TensorThrust& Cout,
            // const std::vector<int>& daga,
            // const std::vector<int>& undaga, 
            // const std::vector<int>& dagb,
            // const std::vector<int>& undagb)
        
            // gpu_error();

            // YOU ARE HERE!!! BELOW NEEDS TO BE REVISED!

            if((anna.size() != crea.size()) or (annb.size() != creb.size())){
                throw std::runtime_error("must be same number of alpha anihilators/creators and beta anihilators/creators.");
            }

            int counta = 0;
            int countb = 0;

            // TODO:Nick need a v3 that resizes stp_gpu and populates it...
            graph_.make_mapping_each_gpu_v3(
                true,
                anna,
                crea,
                &counta,
                pool.terms_sourcea_dag_gpu(), // sourcea_gpu_,  
                pool.terms_targeta_dag_gpu(), // targeta_gpu_,  
                pool.terms_paritya_dag_gpu()); // paritya_gpu_); 

            if (counta == 0) {
                continue;
            }

            graph_.make_mapping_each_gpu_v3(
                false,
                annb,
                creb,
                &countb,
                pool.terms_sourceb_dag_gpu(), // sourceb_gpu_, 
                pool.terms_sourceb_dag_gpu(), // targetb_gpu_, 
                pool.terms_parityb_dag_gpu()); // parityb_gpu_); 

            if (countb == 0) {
                continue;
            }

            // void FCIComputerThrust::apply_individual_nbody_accumulate_gpu(
            // const std::complex<double> coeff,
            // TensorThrust& Cin,
            // TensorThrust& Cout,
            // const std::vector<int>& daga,
            // const std::vector<int>& undaga, 
            // const std::vector<int>& dagb,
            // const std::vector<int>& undagb)

            // apply_individual_nbody_accumulate_gpu(
            //     coeff * std::complex<double>(0.0, -1.0) * sinfactor,
            //     Cin,
            //     Cout, 
            //     crea,
            //     anna,
            //     creb,
            //     annb);

            if((anna.size() != crea.size()) or (annb.size() != creb.size())){
                throw std::runtime_error("must be same number of alpha anihilators/creators and beta anihilators/creators.");
            }

            counta = 0;
            countb = 0;

            // TODO:Nick need a v3 that resizes stp_gpu and populates it...
            graph_.make_mapping_each_gpu_v3(
                true,
                anna,
                crea,
                &counta,
                pool.terms_sourcea_undag_gpu(), // sourcea_gpu_,  
                pool.terms_targeta_undag_gpu(), // targeta_gpu_,  
                pool.terms_paritya_undag_gpu()); // paritya_gpu_); 

            if (counta == 0) {
                continue;
            }

            graph_.make_mapping_each_gpu_v3(
                false,
                annb,
                creb,
                &countb,
                pool.terms_sourceb_undag_gpu(), // sourceb_gpu_, 
                pool.terms_sourceb_undag_gpu(), // targetb_gpu_, 
                pool.terms_parityb_undag_gpu()); // parityb_gpu_); 

            if (countb == 0) {
                continue;
            }

        } else {
            throw std::invalid_argument("Evolved state must remain in spin and particle-number symmetry sector");
        }

    }

    pool.set_device_vecs_populated(true);

}

/* New methods for copying out data */
void FCIComputerThrust::copy_to_tensor_cpu(Tensor& tensor) const
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