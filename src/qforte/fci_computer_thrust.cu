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

    const std::complex<double> cabs = std::abs(coeff);
    const std::complex<double> factor = std::cos(time * cabs);
    cuDoubleComplex factor_gpu = make_cuDoubleComplex(factor.real(), factor.imag());

    std::pair<std::vector<int>, std::vector<int>> maps = evaluate_map_cpu(crea, anna, creb, annb);
    thrust::device_vector<int> d_first(maps.first.begin(), maps.first.end());
    thrust::device_vector<int> d_second(maps.second.begin(), maps.second.end());

    scale_elements_wrapper(
        thrust::raw_pointer_cast(Cout.d_data().data()),
        thrust::raw_pointer_cast(d_first.data()), 
        d_first.size(),
        thrust::raw_pointer_cast(d_second.data()), 
        d_second.size(),
        nbeta_strs_,
        factor_gpu);

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
    const std::vector<int>& annb)
{
    /// TODO: Implement seperate CPU and GPU versions of this function
    // cpu_error();

    // bool reset1 = false;
    // bool reset2 = false;

    // if (Cin.on_gpu()) {
    //     Cin.to_cpu();
    //     reset1 = true;
    // }

    // if (Cout.on_gpu()) {
    //     Cout.to_cpu();
    //     reset2 = true;
    // }

    // if (reset1 != reset2) {
    //     throw std::runtime_error("Both Cin and Cout must be on the same device (CPU or GPU)");
    // }

    std::complex<double> factor = std::exp(-time * std::real(coeff) * std::complex<double>(0.0, 1.0));
    cuDoubleComplex factor_gpu = make_cuDoubleComplex(factor.real(), factor.imag());

    std::pair<std::vector<int>, std::vector<int>> maps = evaluate_map_number_cpu(anna, annb);
    thrust::device_vector<int> d_first(maps.first.begin(), maps.first.end());
    thrust::device_vector<int> d_second(maps.second.begin(), maps.second.end());

    scale_elements_wrapper(
        thrust::raw_pointer_cast(Cout.d_data().data()),
        thrust::raw_pointer_cast(d_first.data()), 
        d_first.size(),
        thrust::raw_pointer_cast(d_second.data()), 
        d_second.size(),
        nbeta_strs_,
        factor_gpu);

    // if (reset1) {
    //     Cin.to_gpu();
    // }

    // if (reset2) {
    //     Cout.to_gpu();
    // }
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
    const std::vector<int>& annb)
{
    /// TODO: Implement seperate CPU and GPU versions of this function
    // cpu_error();

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

    // std::cout << "\n Cout Before Cos Application Thrust \n" << Cout.str(true, true) << std::endl;

    apply_cos_inplace_cpu(
        time,
        ncoeff,
        numbera_dagworka,
        undagworka,
        numberb_dagworkb,
        undagworkb,
        Cout);

    // std::cout << "\n Cout After 1st Cos Application Thrust \n" << Cout.str(true, true) << std::endl;

    std::vector<int> numbera_undagworka(numbera.begin(), numbera.end());
    numbera_undagworka.insert(numbera_undagworka.end(), undagworka.begin(), undagworka.end());

    std::vector<int> numberb_undagworkb(numberb.begin(), numberb.end());
    numberb_undagworkb.insert(numberb_undagworkb.end(), undagworkb.begin(), undagworkb.end());

    // std::cout << "\n Cout Before 2nd Cos Application Thrust \n" << Cout.str(true, true) << std::endl;

    apply_cos_inplace_cpu(
        time,
        ncoeff,
        numbera_undagworka,
        dagworka,
        numberb_undagworkb,
        dagworkb,
        Cout);

    // std::cout << "\n Cout After 2nd Cos Application Thrust \n" << Cout.str(true, true) << std::endl;

    int phase = std::pow(-1, (crea.size() + anna.size()) * (creb.size() + annb.size()));
    std::complex<double> work_cof = std::conj(coeff) * static_cast<double>(phase) * std::complex<double>(0.0, -1.0);

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

    // std::cout << "\n Cout After Second Accumulate Application Thrust \n" << Cout.str(true, true) << std::endl;
}

/// NOTE: Cin should be const, changing for now
void FCIComputerThrust::evolve_individual_nbody_cpu(
    const std::complex<double> time,
    const SQOperator& sqop,
    TensorThrust& Cin,
    TensorThrust& Cout,
    const bool antiherm,
    const bool adjoint)
{
    /// TODO: Implement seperate CPU and GPU versions of this function
    // cpu_error();

    if (sqop.terms().size() != 2) {
        std::cout << "This sqop has " << sqop.terms().size() << " terms." << std::endl;
        throw std::invalid_argument("Individual n-body code is called with multiple terms");
    }

    /// NICK: TODO, implement a hermitian check, at least for two term SQOperators
    // sqop.hermitian_check();

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

    if (crea == anna && creb == annb) {
        // std::cout << "Made it to easy" << std::endl;

        evolve_individual_nbody_easy_cpu(
            time,
            parity * std::get<0>(term), 
            Cin,
            Cout,
            crea,
            anna, 
            creb,
            annb);
    } else if (crea.size() == anna.size() && creb.size() == annb.size()) {
        // std::cout << "Made it to hard" << std::endl;

        evolve_individual_nbody_hard_cpu(
            time,
            parity * std::get<0>(term),
            Cin,
            Cout,
            crea,
            anna, 
            creb,
            annb);

    } else {
        throw std::invalid_argument("Evolved state must remain in spin and particle-number symmetry sector");
    }
}

void FCIComputerThrust::apply_sqop_evolution_gpu(
    const std::complex<double> time,
    const SQOperator& sqop,
    const bool antiherm,
    const bool adjoint)
{
    gpu_error();

    TensorThrust Cin(C_.shape(), "Cin", true);
    Cin.copy_in_gpu(C_);

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
    timer_.reset();
    
    if ((targeta_gpu_.size() != sourcea_gpu_.size()) or (sourcea_gpu_.size() != paritya_gpu_.size())) {
        throw std::runtime_error("The sizes of atarget, asource, and aparity must be the same.");
    }

    if ((targetb_gpu_.size() != sourceb_gpu_.size()) or (sourceb_gpu_.size() != parityb_gpu_.size())) {
        throw std::runtime_error("The sizes of btarget, bsource, and bparity must be the same.");
    }

    timer_.acc_record("error checks in nbody1_acc");
    timer_.reset();

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

    // std::cout << "cout: \n" << Cout.str() << std::endl;

    timer_.acc_record("calling gpu function");
    timer_.reset();

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

    local_timer my_timer = local_timer();
    timer_.reset();

    int counta = 0;
    int countb = 0;

    // Want to check if I can just check counta and countb
    // std::cout << "daga size: " << daga.size() << std::endl;
    // std::cout << "undaga size: " << undaga.size() << std::endl;
    // std::cout << "counta: " << counta << std::endl;

    // std::cout << "dagb size: " << dagb.size() << std::endl;
    // std::cout << "undagb size: " << undagb.size() << std::endl;
    // std::cout << "countb: " << countb << std::endl;

    // If one side is empty, take optimized path before building the other side unnecessarily
    if (dagb.empty() && undagb.empty()) {
        apply_individual_nbody_accumulate_gpu_row_only(coeff, Cin, Cout, daga, undaga);
        return;
    }
    if (daga.empty() && undaga.empty()) {
        apply_individual_nbody_accumulate_gpu_col_only(coeff, Cin, Cout, dagb, undagb);
        return;
    }

    graph_.make_mapping_each_gpu_v2(
        true,
        daga,
        undaga,
        &counta,
        sourcea_gpu_,
        targeta_gpu_,
        paritya_gpu_);

    timer_.acc_record("first 'make_mapping_each' in apply_individual_nbody_accumulate");
    timer_.reset();

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

    timer_.acc_record("second 'make_mapping_each' in apply_individual_nbody_accumulate");
    timer_.reset();

    if (countb == 0) {
        return;
    }

    timer_.acc_record("second for loop in apply_individual_nbody_accumulate");
    timer_.reset();

    thrust::host_vector<int> sourcea_cpu(counta);
    thrust::host_vector<int> targeta_cpu(counta);
    thrust::host_vector<cuDoubleComplex> paritya_cpu(counta);

    thrust::host_vector<int> sourceb_cpu(countb);
    thrust::host_vector<int> targetb_cpu(countb);
    thrust::host_vector<cuDoubleComplex> parityb_cpu(countb);

    thrust::copy(sourcea_gpu_.begin(), sourcea_gpu_.begin() + counta, sourcea_cpu.begin());
    thrust::copy(targeta_gpu_.begin(), targeta_gpu_.begin() + counta, targeta_cpu.begin());
    thrust::copy(paritya_gpu_.begin(), paritya_gpu_.begin() + counta, paritya_cpu.begin());

    thrust::copy(sourceb_gpu_.begin(), sourceb_gpu_.begin() + countb, sourceb_cpu.begin());
    thrust::copy(targetb_gpu_.begin(), targetb_gpu_.begin() + countb, targetb_cpu.begin());
    thrust::copy(parityb_gpu_.begin(), parityb_gpu_.begin() + countb, parityb_cpu.begin());

    // std::cout << "\n\n----------------------------------------------------------\n\n" << std::endl;
    
    // print_vector_thrust(sourcea_cpu, "sourcea");
    // print_vector_thrust(targeta_cpu, "targeta");
    // print_vector_thrust_cuDoubleComplex(paritya_cpu, "paritya");
    // print_vector_thrust(sourceb_cpu, "sourceb");
    // print_vector_thrust(targetb_cpu, "targetb");
    // print_vector_thrust_cuDoubleComplex(parityb_cpu, "parityb");

    // std::cout << "\n\n----------------------------------------------------------\n\n" << std::endl;

    /// TODO: changing this function to use private members of FCIComputerThrust
    apply_individual_nbody1_accumulate_gpu(
        coeff, 
        Cin,
        Cout,
        counta,
        countb);
}

// Optimized path when beta-side is empty: perform row-wise accumulate (no column derefs in kernel)
void FCIComputerThrust::apply_individual_nbody_accumulate_gpu_row_only(
    const std::complex<double> coeff,
    TensorThrust& Cin,
    TensorThrust& Cout,
    const std::vector<int>& daga,
    const std::vector<int>& undaga)
{
    int counta = 0;

    graph_.make_mapping_each_gpu_v2(
        true,
        daga,
        undaga,
        &counta,
        sourcea_gpu_,
        targeta_gpu_,
        paritya_gpu_);

    if (counta == 0) return;

    cuDoubleComplex cu_coeff = make_cuDoubleComplex(coeff.real(), coeff.imag());

    // Launch a kernel specialized for full-row axpy-like accumulate over all beta strings
    apply_row_accumulate_wrapper(
        cu_coeff,
        thrust::raw_pointer_cast(Cin.read_d_data().data()),
        thrust::raw_pointer_cast(Cout.d_data().data()),
        thrust::raw_pointer_cast(sourcea_gpu_.data()),
        thrust::raw_pointer_cast(targeta_gpu_.data()),
        thrust::raw_pointer_cast(paritya_gpu_.data()),
        static_cast<int>(nbeta_strs_),
        counta,
        static_cast<int>(Cin.size() * sizeof(cuDoubleComplex)));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error (row_only): " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("Failed in row-only accumulate on the GPU.");
    }
}

// Optimized path when alpha-side is empty: perform column-wise accumulate (no row derefs in kernel)
void FCIComputerThrust::apply_individual_nbody_accumulate_gpu_col_only(
    const std::complex<double> coeff,
    TensorThrust& Cin,
    TensorThrust& Cout,
    const std::vector<int>& dagb,
    const std::vector<int>& undagb)
{
    int countb = 0;

    graph_.make_mapping_each_gpu_v2(
        false,
        dagb,
        undagb,
        &countb,
        sourceb_gpu_,
        targetb_gpu_,
        parityb_gpu_);

    if (countb == 0) return;

    cuDoubleComplex cu_coeff = make_cuDoubleComplex(coeff.real(), coeff.imag());

    apply_col_accumulate_wrapper(
        cu_coeff,
        thrust::raw_pointer_cast(Cin.read_d_data().data()),
        thrust::raw_pointer_cast(Cout.d_data().data()),
        thrust::raw_pointer_cast(sourceb_gpu_.data()),
        thrust::raw_pointer_cast(targetb_gpu_.data()),
        thrust::raw_pointer_cast(parityb_gpu_.data()),
        static_cast<int>(nbeta_strs_),
        static_cast<int>(nalfa_strs_),
        countb,
        static_cast<int>(Cin.size() * sizeof(cuDoubleComplex)));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error (col_only): " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("Failed in col-only accumulate on the GPU.");
    }
}

// Implement applying a full SQOperator by accumulating all its terms using the GPU accumulate path.
// This works even when the class is currently in CPU mode by staging temporary GPU tensors.
void FCIComputerThrust::apply_sqop_gpu(const SQOperator& sqop)
{
    // Prepare input (Cin) and output (Cout) tensors on GPU without changing class device state
    const bool was_gpu = on_gpu_;

    // Build GPU copies
    TensorThrust Cin_gpu(C_.shape(), "Cin_gpu", true);
    TensorThrust Cout_gpu(C_.shape(), "Cout_gpu", true);
    Cout_gpu.zero_gpu();

    // Copy C_ into Cin_gpu.d_data()
    if (was_gpu) {
        // Device-to-device copy
        thrust::copy(
            thrust::device,
            C_.read_d_data().begin(),
            C_.read_d_data().end(),
            Cin_gpu.d_data().begin());
    } else {
        // Host-to-device copy with conversion
        std::vector<cuDoubleComplex> tmp(C_.size());
        const auto& hsrc = C_.read_h_data();
        for (size_t i = 0; i < C_.size(); ++i) {
            tmp[i] = make_cuDoubleComplex(hsrc[i].real(), hsrc[i].imag());
        }
        thrust::copy(tmp.begin(), tmp.end(), Cin_gpu.d_data().begin());
    }

    // Accumulate each term into Cout_gpu
    for (const auto& term : sqop.terms()) {
        std::complex<double> c = std::get<0>(term);
        if (std::abs(c) <= compute_threshold_) continue;

        // Split creators/annihilators by spin
        std::vector<int> crea, creb, anna, annb;
        const auto& creators = std::get<1>(term);
        const auto& annihils = std::get<2>(term);

        if (creators.size() != annihils.size()) {
            throw std::invalid_argument("Each term must have same number of annihilators and creators");
        }

        for (size_t i = 0; i < creators.size(); ++i) {
            if (creators[i] % 2 == 0) {
                crea.push_back(static_cast<int>(creators[i] / 2));
            } else {
                creb.push_back(static_cast<int>(creators[i] / 2));
            }
        }
        for (size_t i = 0; i < annihils.size(); ++i) {
            if (annihils[i] % 2 == 0) {
                anna.push_back(static_cast<int>(annihils[i] / 2));
            } else {
                annb.push_back(static_cast<int>(annihils[i] / 2));
            }
        }

        // Parity from sorting combined ops
        std::vector<size_t> ops1 = creators;
        std::vector<size_t> ops2 = annihils;
        ops1.insert(ops1.end(), ops2.begin(), ops2.end());
        int nswaps = parity_sort(ops1);
        std::complex<double> coeff = std::pow(-1, nswaps) * c;

        // Use the optimized accumulate path. Temporarily override class on_gpu_ so gpu_error() passes.
        bool saved = on_gpu_;
        on_gpu_ = true;
        try {
            apply_individual_nbody_accumulate_gpu(
                coeff,
                Cin_gpu,
                Cout_gpu,
                crea,
                anna,
                creb,
                annb);
        } catch (...) {
            on_gpu_ = saved;
            throw;
        }
        on_gpu_ = saved;
    }

    // Move Cout back into C_
    if (was_gpu) {
        // Device-to-device copy into C_.d_data()
        thrust::copy(
            thrust::device,
            Cout_gpu.read_d_data().begin(),
            Cout_gpu.read_d_data().end(),
            C_.d_data().begin());
    } else {
        // Device-to-host with conversion into C_.h_data()
        std::vector<cuDoubleComplex> tmp(C_.size());
        thrust::copy(Cout_gpu.read_d_data().begin(), Cout_gpu.read_d_data().end(), tmp.begin());
        auto& hdst = C_.h_data();
        for (size_t i = 0; i < C_.size(); ++i) {
            hdst[i] = std::complex<double>(cuCreal(tmp[i]), cuCimag(tmp[i]));
        }
    }
}

void FCIComputerThrust::scale_cpu(const std::complex<double> a)
{
    cpu_error();
    C_.scale(a);
}