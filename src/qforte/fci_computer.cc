#include <map>
#include <random>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <iterator>

// #include "fmt/format.h"

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

#include "fci_computer.h"
#include "fci_graph.h"




FCIComputer::FCIComputer(int nel, int sz, int norb) : 
    nel_(nel), 
    sz_(sz),
    norb_(norb) {

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

    C_.zero_with_shape({nalfa_strs_, nbeta_strs_});
    C_.set_name("FCI Computer");

    graph_ = FCIGraph(nalfa_el_, nbeta_el_, norb_);

    timer_ = local_timer();
}

/// Set a particular element of the tensor stored in FCIComputer, specified by idxs
void FCIComputer::set_element(
    const std::vector<size_t>& idxs,
    const std::complex<double> val
        )
{
    C_.set(idxs, val);
}

void FCIComputer::do_on_gpu() {
    use_gpu_operations_ = true;
}

void FCIComputer::do_on_cpu() {
    use_gpu_operations_ = false;
}

/// apply a TensorOperator to the current state 
void apply_tensor_operator(const TensorOperator& top);

/// apply a Tensor represending a 1-body spin-orbital indexed operator to the current state 
void FCIComputer::apply_tensor_spin_1bdy(const Tensor& h1e, size_t norb) {

    if(h1e.size() != (norb * 2) * (norb * 2)){
        throw std::invalid_argument("Expecting h1e to be nso x nso for apply_tensor_spin_1bdy");
    }

    Tensor Cnew({nalfa_strs_, nbeta_strs_}, "Cnew");

    Tensor h1e_blk1 = h1e.slice(
        {
            std::make_pair(0, norb_), 
            std::make_pair(0, norb_)
            }
        );

    Tensor h1e_blk2 = h1e.slice(
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

/// apply a Tensor represending a 1-body spatial-orbital indexed operator to the current state 
void FCIComputer::apply_tensor_spat_1bdy(const Tensor& h1e, size_t norb) {

    if(h1e.size() != (norb) * (norb)){
        throw std::invalid_argument("Expecting h1e to be nso x nso for apply_tensor_spin_1bdy");
    }

    Tensor Cnew({nalfa_strs_, nbeta_strs_}, "Cnew");

    apply_array_1bdy(
        Cnew,
        graph_.read_dexca_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexca(),
        h1e,
        norb_,
        true);

    C_.transpose();

    apply_array_1bdy(
        Cnew,
        graph_.read_dexcb_vec(),
        nbeta_strs_,
        nalfa_strs_,
        graph_.get_ndexcb(),
        h1e,
        norb_,
        false);

    C_ = Cnew;
}

/// apply Tensors represending 1-body and 2-body spin-orbital indexed operator to the current state 
/// A LOT of wasted memory here, will want to improve...
void FCIComputer::apply_tensor_spin_12bdy(
    const Tensor& h1e, 
    const Tensor& h2e, 
    size_t norb) 
{
    if(norb > 10 or norb_ > 10){
        throw std::invalid_argument("Don't use this function with more that 10 orbitals, too much memory");
    }

    if(h1e.size() != (norb * 2) * (norb * 2)){
        throw std::invalid_argument("Expecting h1e to be nso x nso for apply_tensor_spin_12bdy");
    }

    if(h2e.size() != (norb * 2) * (norb * 2) * (norb * 2) * (norb * 2) ){
        throw std::invalid_argument("Expecting h2e to be nso x nso x nso nso for apply_tensor_spin_12bdy");
    }

    Tensor Cnew({nalfa_strs_, nbeta_strs_}, "Cnew");
    Tensor h1e_new = h1e;

    Tensor h2e_new(h2e.shape(), "A2");
    Tensor::permute(
        {"i", "j", "k", "l"}, 
        {"i", "k", "j", "l"}, 
        h2e, 
        h2e_new); 

    h2e_new.scale(-1.0);

    for(int k = 0; k < 2 * norb_; k++){
        Tensor h2e_k = h2e.slice(
        {
            std::make_pair(0, 2 * norb_), 
            std::make_pair(k, k+1),
            std::make_pair(k, k+1), 
            std::make_pair(0, 2 * norb_)
            }
        );

        h2e_k.scale(-1.0);

        h1e_new.zaxpy(
            h2e_k,
            1.0,
            1,
            1);
    }

    /// NICK: Keeping this here in case future debuggin is needed
    // 1 std::cout << "\n\n  ====> h1e <====" << h1e_new.print_nonzero() << std::endl;
    // 1 std::cout << "\n\n  ====> h2e <====" << h2e_new.print_nonzero() << std::endl;

    Tensor h1e_blk1 = h1e_new.slice(
        {
            std::make_pair(0, norb_), 
            std::make_pair(0, norb_)
            }
        );

    Tensor h1e_blk2 = h1e_new.slice(
        {
            std::make_pair(norb_, 2*norb_), 
            std::make_pair(norb_, 2*norb_)
            }
        );

    Tensor h2e_blk11 = h2e_new.slice(
        {
            std::make_pair(0, norb_), 
            std::make_pair(0, norb_),
            std::make_pair(0, norb_), 
            std::make_pair(0, norb_)
            }
        );

    Tensor h2e_blk12 = h2e_new.slice(
        {
            std::make_pair(0, norb_), 
            std::make_pair(0, norb_),
            std::make_pair(norb_, 2*norb_), 
            std::make_pair(norb_, 2*norb_)
            }
        );

    Tensor h2e_blk21 = h2e_new.slice(
        {
            std::make_pair(norb_, 2*norb_), 
            std::make_pair(norb_, 2*norb_),
            std::make_pair(0, norb_), 
            std::make_pair(0, norb_)
            }
        );

    Tensor h2e_blk22 = h2e_new.slice(
        {
            std::make_pair(norb_, 2*norb_), 
            std::make_pair(norb_, 2*norb_),
            std::make_pair(norb_, 2*norb_), 
            std::make_pair(norb_, 2*norb_)
            }
        );

    std::pair<Tensor, Tensor> dvec = calculate_dvec_spin_with_coeff();
    
    Tensor dveca_new(dvec.first.shape(),  "dveca_new");
    Tensor dvecb_new(dvec.second.shape(), "dvecb_new");

    Tensor::einsum(
        {"i", "j"},
        {"i", "j", "k", "l"},
        {"k", "l"},
        h1e_blk1,
        dvec.first,
        Cnew, 
        1.0,
        0.0
    );

    Tensor::einsum(
        {"i", "j"},
        {"i", "j", "k", "l"},
        {"k", "l"},
        h1e_blk2,
        dvec.second,
        Cnew, 
        1.0,
        0.0
    );

    Tensor::einsum(
        {"i", "j", "k", "l"},
        {"k", "l", "m", "n"},
        {"i", "j", "m", "n"},
        h2e_blk11,
        dvec.first,
        dveca_new, 
        1.0,
        0.0
    );

    Tensor::einsum(
        {"i", "j", "k", "l"},
        {"k", "l", "m", "n"},
        {"i", "j", "m", "n"},
        h2e_blk12,
        dvec.second,
        dveca_new, 
        1.0,
        0.0
    );

    Tensor::einsum(
        {"i", "j", "k", "l"},
        {"k", "l", "m", "n"},
        {"i", "j", "m", "n"},
        h2e_blk21,
        dvec.first,
        dvecb_new, 
        1.0,
        0.0
    );

    Tensor::einsum(
        {"i", "j", "k", "l"},
        {"k", "l", "m", "n"},
        {"i", "j", "m", "n"},
        h2e_blk22,
        dvec.second,
        dvecb_new, 
        1.0,
        0.0
    );

    std::pair<Tensor, Tensor> dvec_new = std::make_pair(dveca_new, dvecb_new);

    Cnew.zaxpy(
        calculate_coeff_spin_with_dvec(dvec_new),
        1.0,
        1,
        1    
    );

    C_ = Cnew;
}

/// apply Tensors represending 1-body and 2-body spin-orbital indexed operator to the current state 
void FCIComputer::apply_tensor_spin_012bdy(
    const Tensor& h0e, 
    const Tensor& h1e, 
    const Tensor& h2e, 
    size_t norb) 
{
    h0e.shape_error({1});
    Tensor Cold = C_;
    
    apply_tensor_spin_12bdy(
        h1e,
        h2e,
        norb);

    C_.zaxpy(
        Cold,
        h0e.get({0}),
        1,
        1    
    );
}

/// apply Tensors represending 1-body and 2-body spatial-orbital indexed operator to the current state 
void FCIComputer::apply_tensor_spat_12bdy(
    const Tensor& h1e, 
    const Tensor& h2e, 
    const Tensor& h2e_einsum, 
    size_t norb) 
{
    if(h1e.size() != (norb) * (norb)){
        throw std::invalid_argument("Expecting h1e to be nmo x nmo for apply_tensor_spat_12bdy");
    }

    if(h2e.size() != (norb) * (norb) * (norb) * (norb) ){
        throw std::invalid_argument("Expecting h2e to be nso x nso x nso nso for apply_tensor_spin_12bdy");
    }

    Tensor Cnew({nalfa_strs_, nbeta_strs_}, "Cnew");
    Cnew.zero();

    lm_apply_array12_same_spin_opt(
        Cnew, 
        graph_.read_dexca_vec(), // dexca_tmp
        nalfa_strs_,
        nbeta_strs_, 
        graph_.get_ndexca(),
        h1e, 
        h2e,
        norb_,
        true);

    Cnew.transpose();
        
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

    Cnew.transpose();

    lm_apply_array12_diff_spin_opt(
        Cnew,
        graph_.read_dexca_vec(),
        graph_.read_dexcb_vec(),
        nalfa_strs_,
        nbeta_strs_, 
        graph_.get_ndexca(),
        graph_.get_ndexca(),
        h2e_einsum, 
        norb_); 

    C_ = Cnew;
}

/// apply Tensors represending 1-body and 2-body spatial-orbital indexed operator
/// as well as a constant to the current state 
void FCIComputer::apply_tensor_spat_012bdy(
    const std::complex<double> h0e,
    const Tensor& h1e, 
    const Tensor& h2e, 
    const Tensor& h2e_einsum, 
    size_t norb) 
{
    
    Tensor Ctemp = C_;
    
    apply_tensor_spat_12bdy(
        h1e,
        h2e,
        h2e_einsum,
        norb);

    C_.zaxpy(
        Ctemp,
        h0e,
        1,
        1    
    );
}

// NICK: VERY VERY Slow, will want even a better c++ implementation!
// Try with einsum once working or perhaps someting like the above...?
std::pair<Tensor, Tensor> FCIComputer::calculate_dvec_spin_with_coeff() {

    Tensor dveca({norb_, norb_, nalfa_strs_, nbeta_strs_}, "dveca");
    Tensor dvecb({norb_, norb_, nalfa_strs_, nbeta_strs_}, "dvecb");

    for (size_t i = 0; i < norb_; ++i) {
        for (size_t j = 0; j < norb_; ++j) {
            auto alfa_mappings = graph_.get_alfa_map()[std::make_pair(i,j)];
            auto beta_mappings = graph_.get_beta_map()[std::make_pair(i,j)];

            for (const auto& mapping : alfa_mappings) {
                size_t source = std::get<0>(mapping);
                size_t target = std::get<1>(mapping);
                std::complex<double> parity = static_cast<std::complex<double>>(std::get<2>(mapping));
                for (size_t k = 0; k < dveca.shape()[3]; ++k) {
                    size_t c_vidxa = k * C_.strides()[1] + source * C_.strides()[0];
                    size_t d_vidxa = k * dveca.strides()[3] + target * dveca.strides()[2] + j * dveca.strides()[1] + i * dveca.strides()[0];
                    dveca.data()[d_vidxa] += parity * C_.data()[c_vidxa];
                }
            }

            for (const auto& mapping : beta_mappings) {
                size_t source = std::get<0>(mapping);
                size_t target = std::get<1>(mapping);
                std::complex<double> parity = static_cast<std::complex<double>>(std::get<2>(mapping));
                for (size_t k = 0; k < dvecb.shape()[2]; ++k) {
                    size_t c_vidxb = source * C_.strides()[1] + k * C_.strides()[0];
                    size_t d_vidxb = target * dvecb.strides()[3] + k * dvecb.strides()[2] + j * dvecb.strides()[1] + i * dvecb.strides()[0];
                    dvecb.data()[d_vidxb] += parity * C_.data()[c_vidxb];
                }
            }
        }
    }
    return std::make_pair(dveca, dvecb);
}

// ALSO SLOW
Tensor FCIComputer::calculate_coeff_spin_with_dvec(std::pair<Tensor, Tensor>& dvec) {
    Tensor Cnew({nalfa_strs_, nbeta_strs_}, "Cnew");

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

/// NICK: make this more c++ style, not a fan of the raw pointers :(
void FCIComputer::apply_array_1bdy(
    Tensor& out,
    const std::vector<int>& dexc,
    const int astates,
    const int bstates,
    const int ndexc,
    const Tensor& h1e,
    const int norbs,
    const bool is_alpha)
{
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

            const std::complex<double> pref = static_cast<double>(parity) * h1e.read_data()[ijshift];
            const std::complex<double>* xptr = C_.data().data() + target * inc1;

            math_zaxpy(states2, pref, xptr, inc2, cout, inc2);
        }
    }
}

void FCIComputer::lm_apply_array12_same_spin_opt(
    Tensor& out,
    const std::vector<int>& dexc,
    const int alpha_states,
    const int beta_states,
    const int ndexc,
    const Tensor& h1e,
    const Tensor& h2e,
    const int norbs,
    const bool is_alpha)
{
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
            const std::complex<double> *h2etmp = h2e.read_data().data() + h2e_id;
            temp[s2] += static_cast<double>(parity1) * h1e.read_data()[ijshift];

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

void FCIComputer::lm_apply_array12_diff_spin_opt(
    Tensor& out,
    const std::vector<int>& adexc,
    const std::vector<int>& bdexc,
    const int alpha_states,
    const int beta_states,
    const int nadexc,
    const int nbdexc,
    const Tensor& h2e,
    const int norbs) 
{
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

        const std::complex<double> *tmperi = h2e.read_data().data() + orbid * norbs2;

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

/// apply a 1-body and 2-body TensorOperator to the current state 
void apply_tensor_spin_12_body(const TensorOperator& top){
    // Stuff
}

std::pair<std::vector<int>, std::vector<int>> FCIComputer::evaluate_map_number(
    const std::vector<int>& numa, 
    const std::vector<int>& numb) 
{
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

std::pair<std::vector<int>, std::vector<int>> FCIComputer::evaluate_map(
    const std::vector<int>& crea,
    const std::vector<int>& anna,
    const std::vector<int>& creb,
    const std::vector<int>& annb) 
{
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

void FCIComputer::apply_cos_inplace(
    const std::complex<double> time,
    const std::complex<double> coeff,
    const std::vector<int>& crea,
    const std::vector<int>& anna,
    const std::vector<int>& creb,
    const std::vector<int>& annb,
    Tensor& Cout)
{
    const std::complex<double> cabs = std::abs(coeff);
    const std::complex<double> factor = std::cos(time * cabs);

    std::pair<std::vector<int>, std::vector<int>> maps = evaluate_map(crea, anna, creb, annb);

    if (maps.first.size() != 0 and maps.second.size() != 0){
        for (size_t i = 0; i < maps.first.size(); i++){
            for (size_t j = 0; j < maps.second.size(); j++){
                Cout.data()[maps.first[i] * nbeta_strs_ +  maps.second[j]] *= factor;
            }
        }       
    }
}

int FCIComputer::isolate_number_operators(
    const std::vector<int>& cre,
    const std::vector<int>& ann,
    std::vector<int>& crework,
    std::vector<int>& annwork,
    std::vector<int>& number) 
{

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

void FCIComputer::evolve_individual_nbody_easy(
    const std::complex<double> time,
    const std::complex<double> coeff,
    const Tensor& Cin,  
    Tensor& Cout,       
    const std::vector<int>& crea,
    const std::vector<int>& anna,
    const std::vector<int>& creb,
    const std::vector<int>& annb) 
{
    int n_a = anna.size();
    int n_b = annb.size();

    int power = n_a * (n_a - 1) / 2 + n_b * (n_b - 1) / 2;
    std::complex<double> prefactor = coeff * std::pow(-1, power);
    std::complex<double> factor = std::exp(-time * std::real(prefactor) * std::complex<double>(0.0, 1.0));
    std::pair<std::vector<int>, std::vector<int>> maps = evaluate_map_number(anna, annb);

    if (maps.first.size() != 0 and maps.second.size() != 0){
        for (size_t i = 0; i < maps.first.size(); i++){
            for (size_t j = 0; j < maps.second.size(); j++){
                Cout.data()[maps.first[i] * nbeta_strs_ +  maps.second[j]] *= factor;
            }
        }       
    }
}

void FCIComputer::evolve_individual_nbody_easy_v2(
    const std::complex<double> time,
    const std::complex<double> coeff, 
    Tensor& Cout,       
    const std::vector<int>& crea,
    const std::vector<int>& anna,
    const std::vector<int>& creb,
    const std::vector<int>& annb) 
{
    int n_a = anna.size();
    int n_b = annb.size();

    int power = n_a * (n_a - 1) / 2 + n_b * (n_b - 1) / 2;
    std::complex<double> prefactor = coeff * std::pow(-1, power);
    std::complex<double> factor = std::exp(-time * std::real(prefactor) * std::complex<double>(0.0, 1.0));
    std::pair<std::vector<int>, std::vector<int>> maps = evaluate_map_number(anna, annb);

    if (maps.first.size() != 0 and maps.second.size() != 0){
        for (size_t i = 0; i < maps.first.size(); i++){
            for (size_t j = 0; j < maps.second.size(); j++){
                Cout.data()[maps.first[i] * nbeta_strs_ +  maps.second[j]] *= factor;
            }
        }       
    }
}

void FCIComputer::evolve_individual_nbody_hard(
    const std::complex<double> time,
    const std::complex<double> coeff,
    const Tensor& Cin,  
    Tensor& Cout,       
    const std::vector<int>& crea,
    const std::vector<int>& anna,
    const std::vector<int>& creb,
    const std::vector<int>& annb) 
{
    std::vector<int> dagworka(crea);
    std::vector<int> dagworkb(creb);
    std::vector<int> undagworka(anna);
    std::vector<int> undagworkb(annb);
    std::vector<int> numbera;
    std::vector<int> numberb;

    int parity = 0;
    parity += isolate_number_operators(
        crea,
        anna,
        dagworka,
        undagworka,
        numbera);

    parity += isolate_number_operators(
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

    apply_cos_inplace(
        time,
        ncoeff,
        numbera_dagworka,
        undagworka,
        numberb_dagworkb,
        undagworkb,
        Cout);

    std::vector<int> numbera_undagworka(numbera.begin(), numbera.end());
    numbera_undagworka.insert(numbera_undagworka.end(), undagworka.begin(), undagworka.end());

    std::vector<int> numberb_undagworkb(numberb.begin(), numberb.end());
    numberb_undagworkb.insert(numberb_undagworkb.end(), undagworkb.begin(), undagworkb.end());

    apply_cos_inplace(
        time,
        ncoeff,
        numbera_undagworka,
        dagworka,
        numberb_undagworkb,
        dagworkb,
        Cout);

    int phase = std::pow(-1, (crea.size() + anna.size()) * (creb.size() + annb.size()));
    std::complex<double> work_cof = std::conj(coeff) * static_cast<double>(phase) * std::complex<double>(0.0, -1.0);

    apply_individual_nbody_accumulate(
        work_cof * sinfactor,
        Cin,
        Cout, 
        anna,
        crea,
        annb,
        creb);

    apply_individual_nbody_accumulate(
        coeff * std::complex<double>(0.0, -1.0) * sinfactor,
        Cin,
        Cout, 
        crea,
        anna,
        creb,
        annb);
}

void FCIComputer::evolve_individual_nbody_hard_v2(
    const std::complex<double> time,
    const std::complex<double> coeff,
    Tensor& Cout,       
    const std::vector<int>& crea,
    const std::vector<int>& anna,
    const std::vector<int>& creb,
    const std::vector<int>& annb) 
{
    std::vector<int> dagworka(crea);
    std::vector<int> dagworkb(creb);
    std::vector<int> undagworka(anna);
    std::vector<int> undagworkb(annb);
    std::vector<int> numbera;
    std::vector<int> numberb;

    int parity = 0;
    parity += isolate_number_operators(
        crea,
        anna,
        dagworka,
        undagworka,
        numbera);

    parity += isolate_number_operators(
        creb,
        annb,
        dagworkb,
        undagworkb,
        numberb);

    std::vector<int> numbera_dagworka(numbera.begin(), numbera.end());
    numbera_dagworka.insert(numbera_dagworka.end(), dagworka.begin(), dagworka.end());

    std::vector<int> numberb_dagworkb(numberb.begin(), numberb.end());
    numberb_dagworkb.insert(numberb_dagworkb.end(), dagworkb.begin(), dagworkb.end());

    std::pair<std::vector<int>, std::vector<int>> maps1 = evaluate_map(
        numbera_dagworka,
        undagworka,
        numberb_dagworkb,
        undagworkb);

    std::vector<int> numbera_undagworka(numbera.begin(), numbera.end());
    numbera_undagworka.insert(numbera_undagworka.end(), undagworka.begin(), undagworka.end());

    std::vector<int> numberb_undagworkb(numberb.begin(), numberb.end());
    numberb_undagworkb.insert(numberb_undagworkb.end(), undagworkb.begin(), undagworkb.end());

    std::pair<std::vector<int>, std::vector<int>> maps2 = evaluate_map(
        numbera_undagworka, 
        dagworka, 
        numberb_undagworkb, 
        dagworkb);

    // ===> Begin 1st Accumulate index generation <=== //

    int phase = std::pow(-1, (crea.size() + anna.size()) * (creb.size() + annb.size()));
    std::complex<double> work_cof = std::conj(coeff) * static_cast<double>(phase) * std::complex<double>(0.0, -1.0);

    if((anna.size() != crea.size()) or (annb.size() != creb.size())){
        throw std::runtime_error("must be same number of alpha anihilators/creators and beta anihilators/creators.");
    }

    std::tuple<int, std::vector<int>, std::vector<int>, std::vector<int>> ualfamap1 = graph_.make_mapping_each(
        true,
        anna,
        crea);

    if (std::get<0>(ualfamap1) == 0) {
        return;
    }

    std::tuple<int, std::vector<int>, std::vector<int>, std::vector<int>> ubetamap1 = graph_.make_mapping_each(
        false,
        annb,
        creb);

    if (std::get<0>(ubetamap1) == 0) {
        return;
    }

    std::vector<int> sourcea1(std::get<0>(ualfamap1));
    std::vector<int> targeta1(std::get<0>(ualfamap1));
    std::vector<int> paritya1(std::get<0>(ualfamap1));
    std::vector<int> sourceb1(std::get<0>(ubetamap1));
    std::vector<int> targetb1(std::get<0>(ubetamap1));
    std::vector<int> parityb1(std::get<0>(ubetamap1));

    /// NICK: All this can be done in the make_mapping_each fucntion.
    /// Maybe try like a make_abbrev_mapping_each

    /// NICK: Might be slow, check this out...
    for (int i = 0; i < std::get<0>(ualfamap1); i++) {
        sourcea1[i] = std::get<1>(ualfamap1)[i];
        targeta1[i] = graph_.get_aind_for_str(std::get<2>(ualfamap1)[i]);
        paritya1[i] = 1.0 - 2.0 * std::get<3>(ualfamap1)[i];
    }

    for (int i = 0; i < std::get<0>(ubetamap1); i++) {
        sourceb1[i] = std::get<1>(ubetamap1)[i];
        targetb1[i] = graph_.get_bind_for_str(std::get<2>(ubetamap1)[i]);
        parityb1[i] = 1.0 - 2.0 * std::get<3>(ubetamap1)[i];
    }

    // ===> Begin 2nd Accumulate index generation <=== //             

    std::tuple<int, std::vector<int>, std::vector<int>, std::vector<int>> ualfamap2 = graph_.make_mapping_each(
        true,
        crea,
        anna);

    if (std::get<0>(ualfamap2) == 0) {
        return;
    }

    std::tuple<int, std::vector<int>, std::vector<int>, std::vector<int>> ubetamap2 = graph_.make_mapping_each(
        false,
        creb,
        annb);

    if (std::get<0>(ubetamap2) == 0) {
        return;
    }

    std::vector<int> sourcea2(std::get<0>(ualfamap2));
    std::vector<int> targeta2(std::get<0>(ualfamap2));
    std::vector<int> paritya2(std::get<0>(ualfamap2));
    std::vector<int> sourceb2(std::get<0>(ubetamap2));
    std::vector<int> targetb2(std::get<0>(ubetamap2));
    std::vector<int> parityb2(std::get<0>(ubetamap2));

    /// NICK: All this can be done in the make_mapping_each fucntion.
    /// Maybe try like a make_abbrev_mapping_each

    /// NICK: Might be slow, check this out...
    for (int i = 0; i < std::get<0>(ualfamap2); i++) {
        sourcea2[i] = std::get<1>(ualfamap2)[i];
        targeta2[i] = graph_.get_aind_for_str(std::get<2>(ualfamap2)[i]);
        paritya2[i] = 1.0 - 2.0 * std::get<3>(ualfamap2)[i];
    }

    for (int i = 0; i < std::get<0>(ubetamap2); i++) {
        sourceb2[i] = std::get<1>(ubetamap2)[i];
        targetb2[i] = graph_.get_bind_for_str(std::get<2>(ubetamap2)[i]);
        parityb2[i] = 1.0 - 2.0 * std::get<3>(ubetamap2)[i];
    }

    // re-work coefficeints (dt * h_mu or t_mu )

    std::complex<double> ncoeff = coeff * std::pow(-1.0, parity);
    std::complex<double> absol = std::abs(ncoeff);
    std::complex<double> sinfactor = std::sin(time * absol) / absol;

    const std::complex<double> cabs = std::abs(ncoeff);
    const std::complex<double> factor = std::cos(time * cabs);

    std::complex<double>  acc_coeff1 = work_cof * sinfactor;

    std::complex<double>  acc_coeff2 = coeff * std::complex<double>(0.0, -1.0) * sinfactor;

    // ===> In-place 2×2 Givens-like update (replaces the 4 loops used previously) <===
    //
    // This assumes the pairing relationships hold:
    //   sourcea1 == targeta2,  sourceb1 == targetb2
    //   sourcea2 == targeta1,  sourceb2 == targetb1
    // so that each coupled pair of determinants can be updated with a 2×2 block:
    //
    //   [ u' ] = [ factor                  acc_coeff2 * p2 ] [ u0 ]
    //   [ v' ]   [ acc_coeff1 * p1         factor          ] [ v0 ]
    //
    // where
    //   u ≡ Cout[sourcea1[i], sourceb1[j]]
    //   v ≡ Cout[targeta1[i], targetb1[j]]
    //   p1 = paritya1[i] * parityb1[j]  (for the g† leg)
    //   p2 = paritya2[i] * parityb2[j]  (for the g   leg)
    //
    // Notes:
    //   - factor  = cos(time * |ncoeff|)
    //   - acc_coeff1 = conj(coeff) * phase * (-i) * sin(time*|ncoeff|)/|ncoeff|
    //   - acc_coeff2 = coeff        *       (-i) * sin(time*|ncoeff|)/|ncoeff|
    //   - We snapshot u0,v0 before writing so the update is safely in-place.
    //

    // (Optional) quick sanity guard to avoid out-of-bounds if any leg is empty
    if (!sourcea1.empty() && !sourceb1.empty() &&
        sourcea1.size() == targeta1.size() &&
        sourceb1.size() == targetb1.size() &&
        sourcea1.size() == sourcea2.size() && targeta1.size() == sourcea2.size() &&
        sourceb1.size() == sourceb2.size() && targetb1.size() == sourceb2.size()) {

        // (Optional) check the expected pairings; if you want hard asserts, replace with throws.
        const bool pairA_ok = std::equal(sourcea1.begin(), sourcea1.end(), targeta2.begin())
                        && std::equal(sourcea2.begin(), sourcea2.end(), targeta1.begin());
        const bool pairB_ok = std::equal(sourceb1.begin(), sourceb1.end(), targetb2.begin())
                        && std::equal(sourceb2.begin(), sourceb2.end(), targetb1.begin());

        if (pairA_ok && pairB_ok) {
            std::complex<double>* __restrict data = Cout.data().data();

            for (std::size_t ia = 0; ia < sourcea1.size(); ++ia) {
                const int sa = sourcea1[ia];
                const int ta = targeta1[ia];

                // Precompute row offsets
                const std::size_t sa_row = static_cast<std::size_t>(sa) * static_cast<std::size_t>(nbeta_strs_);
                const std::size_t ta_row = static_cast<std::size_t>(ta) * static_cast<std::size_t>(nbeta_strs_);

                // Per-row parity for the two legs
                const int pa1 = paritya1[ia]; // ±1
                const int pa2 = paritya2[ia]; // ±1

                for (std::size_t ib = 0; ib < sourceb1.size(); ++ib) {
                    const int sb = sourceb1[ib];
                    const int tb = targetb1[ib];

                    // Column indices
                    const std::size_t u_idx = sa_row + static_cast<std::size_t>(sb); // u ≡ (sa,sb)
                    const std::size_t v_idx = ta_row + static_cast<std::size_t>(tb); // v ≡ (ta,tb)

                    // Snapshot old values before writing (crucial for in-place correctness)
                    const std::complex<double> u0 = data[u_idx];
                    const std::complex<double> v0 = data[v_idx];

                    // Per-column parity for the two legs
                    const int pb1 = parityb1[ib]; // ±1
                    const int pb2 = parityb2[ib]; // ±1

                    // Combined parity prefactors for off-diagonal couplings
                    const std::complex<double> p1 = static_cast<double>(pa1 * pb1); // for g† leg
                    const std::complex<double> p2 = static_cast<double>(pa2 * pb2); // for g  leg

                    // 2×2 update
                    const std::complex<double> u_new = factor * u0 + acc_coeff2 * p2 * v0;
                    const std::complex<double> v_new = factor * v0 + acc_coeff1 * p1 * u0;

                    data[u_idx] = u_new;
                    data[v_idx] = v_new;
                }
            }
        } else {
            // Fallback: if pairings don't hold, keep the original four-loop path (or throw).
            // throw std::logic_error("Expected (A1,B1)<->(A2,B2) pairing does not hold; cannot use in-place 2x2 update.");
        }
    }

    // ---- ultra-compact debug print ------------------------------------------
    // {
    //     std::cout << "\n ==> New Op <== \n" << std::endl;

    //     auto print_vec = [](const char* label, const std::vector<int>& v) {
    //         std::cout << label << " ";
    //         for (std::size_t i = 0; i < v.size(); ++i) {
    //             if (i) std::cout << ' ';
    //             std::cout << v[i];
    //         }
    //         std::cout << "\n";
    //     };
    //     auto eq = [](const std::vector<int>& a, const std::vector<int>& b) {
    //         return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
    //     };

    //     // maps (4 lines)
    //     // print_vec("M1.A:", maps1.first);
    //     // print_vec("M1.B:", maps1.second);
    //     // print_vec("M2.A:", maps2.first);
    //     // print_vec("M2.B:", maps2.second);

    //     // Six lines: values only
    //     print_vec("A1.src:", sourcea1);
    //     print_vec("A2.tgt:", targeta2);
    //     print_vec("M1.A:  ", maps1.first);
    //     std::cout << "\n" << std::endl;

    //     print_vec("A1.tgt:", targeta1);
    //     print_vec("A2.src:", sourcea2);
    //     print_vec("M2.A:  ", maps2.first);
    //     std::cout << "\n" << std::endl;

    //     print_vec("B1.src:", sourceb1);
    //     print_vec("B2.tgt:", targetb2);
    //     print_vec("M1.B:  ", maps1.second);
    //     std::cout << "\n" << std::endl;

    //     print_vec("B1.tgt:", targetb1);
    //     print_vec("B2.src:", sourceb2);
    //     print_vec("M2.B:  ", maps2.second);
    //     std::cout << "\n" << std::endl;

    //     // True/false equivalence checks (two directions)
    //     std::cout << "eq(A1.src, A2.tgt)=" << (eq(sourcea1, targeta2) ? "true" : "false")
    //             << "  eq(B1.src, B2.tgt)=" << (eq(sourceb1, targetb2) ? "true" : "false")
    //             << "  eq(A2.src, A1.tgt)=" << (eq(sourcea2, targeta1) ? "true" : "false")
    //             << "  eq(B2.src, B1.tgt)=" << (eq(sourceb2, targetb1) ? "true" : "false")
    //             << "\n";

    //     std::cout << "\n\n" << std::endl;
    // }
    // --------------------------------------------------------------------------


}

void FCIComputer::evolve_individual_nbody(
    const std::complex<double> time,
    const SQOperator& sqop,
    const Tensor& Cin,
    Tensor& Cout,
    const bool antiherm,
    const bool adjoint) 
{

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

    if (crea == anna && creb == annb) {
        
        std::complex<double> factor;
        evolve_individual_nbody_easy(
            time,
            parity * 2.0 * std::get<0>(term), 
            Cin,
            Cout,
            crea,
            anna, 
            creb,
            annb);
    } else if (crea.size() == anna.size() && creb.size() == annb.size()) {
        
        evolve_individual_nbody_hard(
            time,
            parity * std::get<0>(term),
            Cin,
            Cout,
            crea,
            anna, 
            creb,
            annb);

    } else {
        print_vector(crea, "crea");
        print_vector(anna, "anna");
        print_vector(creb, "creb");
        print_vector(annb, "annb");
        throw std::invalid_argument(
            "Evolved state must remain in spin and particle-number symmetry sector, bad op above"
        );
    }
}

/// Same as above but all inplace
void FCIComputer::evolve_individual_nbody_v2(
    const std::complex<double> time,
    const SQOperator& sqop,
    Tensor& Cout,
    const bool antiherm,
    const bool adjoint) 
{

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

    if (crea == anna && creb == annb) {
        
        std::complex<double> factor;
        evolve_individual_nbody_easy_v2(
            time,
            parity * 2.0 * std::get<0>(term), 
            Cout,
            crea,
            anna, 
            creb,
            annb);

    } else if (crea.size() == anna.size() && creb.size() == annb.size()) {

        evolve_individual_nbody_hard_v2(
            time,
            parity * std::get<0>(term),
            Cout,
            crea,
            anna, 
            creb,
            annb);

    } else {
        print_vector(crea, "crea");
        print_vector(anna, "anna");
        print_vector(creb, "creb");
        print_vector(annb, "annb");
        throw std::invalid_argument(
            "Evolved state must remain in spin and particle-number symmetry sector, bad op above"
        );
    }
}

void FCIComputer::evolve_op_taylor(
      const SQOperator& op,
      const double evolution_time,
      const double convergence_thresh,
      const int max_taylor_iter,
      const bool real_evolution)

{
    Tensor Cevol = C_;

    for (int order = 1; order < max_taylor_iter; ++order) {

        // std::cout << "I get here, order: " << order << std::endl;

        // std::cout << "C_: " << C_.str() << std::endl;
        // std::cout << "Cevol: " << Cevol.str() << std::endl;
        std::complex<double> coeff;

        if (real_evolution) {
            coeff = std::complex<double>(-evolution_time, 0.0);
        } else {
            coeff = std::complex<double>(0.0, -evolution_time);
        }

        apply_sqop(op);

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

void FCIComputer::evolve_op2_taylor(
      const SQOperator& op,
      const double evolution_time,
      const double convergence_thresh,
      const int max_taylor_iter,
      const bool real_evolution)

{
    Tensor Cevol = C_;

    for (int order = 1; order < max_taylor_iter; ++order) {

        // std::cout << "I get here, order: " << order << std::endl;

        // std::cout << "C_: " << C_.str() << std::endl;
        // std::cout << "Cevol: " << Cevol.str() << std::endl;
        std::complex<double> coeff;

        if (real_evolution) {
            coeff = std::complex<double>(-evolution_time, 0.0);
        } else {
            coeff = std::complex<double>(0.0, -evolution_time);
        }

        apply_sqop(op);
        apply_sqop(op);

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

void FCIComputer::evolve_tensor_taylor(
      const std::complex<double> h0e,
      const Tensor& h1e, 
      const Tensor& h2e, 
      const Tensor& h2e_einsum, 
      size_t norb,
      const double evolution_time,
      const double convergence_thresh,
      const int max_taylor_iter,
      const bool real_evolution)

{
    Tensor Cevol = C_;

    for (int order = 1; order < max_taylor_iter; ++order) {

        // std::cout << "I get here, order: " << order << std::endl;

        // std::cout << "C_: " << C_.str() << std::endl;
        // std::cout << "Cevol: " << Cevol.str() << std::endl;
        std::complex<double> coeff;

        if (real_evolution) {
            coeff = std::complex<double>(-evolution_time, 0.0);
        } else {
            coeff = std::complex<double>(0.0, -evolution_time);
        }

        apply_tensor_spat_012bdy(
            h0e,
            h1e,
            h2e,
            h2e_einsum,
            norb);

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

void FCIComputer::evolve_tensor2_taylor(
      const std::complex<double> h0e,
      const Tensor& h1e, 
      const Tensor& h2e, 
      const Tensor& h2e_einsum, 
      size_t norb,
      const double evolution_time,
      const double convergence_thresh,
      const int max_taylor_iter,
      const bool real_evolution)

{
    Tensor Cevol = C_;

    for (int order = 1; order < max_taylor_iter; ++order) {

        // std::cout << "I get here, order: " << order << std::endl;

        // std::cout << "C_: " << C_.str() << std::endl;
        // std::cout << "Cevol: " << Cevol.str() << std::endl;
        std::complex<double> coeff;

        if (real_evolution) {
            coeff = std::complex<double>(-evolution_time, 0.0);
        } else {
            coeff = std::complex<double>(0.0, -evolution_time);
        }

        apply_tensor_spat_012bdy(
            h0e,
            h1e,
            h2e,
            h2e_einsum,
            norb);

        // Causes discrepancy!!!
        // Maybe bcause this does Cnew = HCold + Cold
        // And NOT Cnew = HCold as would be expected?
        apply_tensor_spat_012bdy(
            h0e,
            h1e,
            h2e,
            h2e_einsum,
            norb);

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

void FCIComputer::evolve_pool_trotter_basic(
      const SQOpPool& pool,
      const bool antiherm,
      const bool adjoint)

{
    if(adjoint){
        for (int i = pool.terms().size() - 1; i >= 0; --i) {
            apply_sqop_evolution(
                pool.terms()[i].first, 
                pool.terms()[i].second,
                antiherm,
                adjoint);
        }
    } else {
        for (const auto& sqop_term : pool.terms()) {
            apply_sqop_evolution(
                sqop_term.first, 
                sqop_term.second,
                antiherm,
                adjoint);
            }
    }
}

void FCIComputer::evolve_pool_trotter_basic_v2(
      const SQOpPool& pool,
      const bool antiherm,
      const bool adjoint)

{
    if(adjoint){
        for (int i = pool.terms().size() - 1; i >= 0; --i) {
            apply_sqop_evolution_v2(
                pool.terms()[i].first, 
                pool.terms()[i].second,
                antiherm,
                adjoint);
        }
    } else {
        for (const auto& sqop_term : pool.terms()) {
            apply_sqop_evolution_v2(
                sqop_term.first, 
                sqop_term.second,
                antiherm,
                adjoint);
            }
    }
}

void FCIComputer::evolve_pool_trotter(
      const SQOpPool& pool,
      const double evolution_time,
      const int trotter_steps,
      const int trotter_order,
      const bool antiherm,
      const bool adjoint)

{
    if(trotter_order == 1){

        std::complex<double> prefactor = evolution_time / static_cast<std::complex<double>>(trotter_steps);

        if(adjoint){
            for( int r = 0; r < trotter_steps; r++) {
                for (int i = pool.terms().size() - 1; i >= 0; --i) {
                    apply_sqop_evolution(
                        prefactor * pool.terms()[i].first, 
                        pool.terms()[i].second,
                        antiherm,
                        adjoint);
                }
            }
                

        } else {
            for( int r = 0; r < trotter_steps; r++) {
                for (const auto& sqop_term : pool.terms()) {
                    apply_sqop_evolution(
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
                    apply_sqop_evolution(
                        prefactor * pool.terms()[i].first, 
                        pool.terms()[i].second,
                        antiherm,
                        adjoint);
                }

                for (const auto& sqop_term : pool.terms()) {
                    apply_sqop_evolution(
                        prefactor * sqop_term.first, 
                        sqop_term.second,
                        antiherm,
                        adjoint);
                }
            }
                

        } else {
            for( int r = 0; r < trotter_steps; r++) {
                for (const auto& sqop_term : pool.terms()) {
                    apply_sqop_evolution(
                        prefactor * sqop_term.first, 
                        sqop_term.second,
                        antiherm,
                        adjoint);
                }

                for (int i = pool.terms().size() - 1; i >= 0; --i) {
                    apply_sqop_evolution(
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

void FCIComputer::evolve_pool_trotter_v2(
      const SQOpPool& pool,
      const double evolution_time,
      const int trotter_steps,
      const int trotter_order,
      const bool antiherm,
      const bool adjoint)

{
    if(trotter_order == 1){

        std::complex<double> prefactor = evolution_time / static_cast<std::complex<double>>(trotter_steps);

        if(adjoint){
            for( int r = 0; r < trotter_steps; r++) {
                for (int i = pool.terms().size() - 1; i >= 0; --i) {
                    apply_sqop_evolution_v2(
                        prefactor * pool.terms()[i].first, 
                        pool.terms()[i].second,
                        antiherm,
                        adjoint);
                }
            }
                

        } else {
            for( int r = 0; r < trotter_steps; r++) {
                for (const auto& sqop_term : pool.terms()) {
                    apply_sqop_evolution_v2(
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
                    apply_sqop_evolution_v2(
                        prefactor * pool.terms()[i].first, 
                        pool.terms()[i].second,
                        antiherm,
                        adjoint);
                }

                for (const auto& sqop_term : pool.terms()) {
                    apply_sqop_evolution_v2(
                        prefactor * sqop_term.first, 
                        sqop_term.second,
                        antiherm,
                        adjoint);
                }
            }
                

        } else {
            for( int r = 0; r < trotter_steps; r++) {
                for (const auto& sqop_term : pool.terms()) {
                    apply_sqop_evolution_v2(
                        prefactor * sqop_term.first, 
                        sqop_term.second,
                        antiherm,
                        adjoint);
                }

                for (int i = pool.terms().size() - 1; i >= 0; --i) {
                    apply_sqop_evolution_v2(
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

void FCIComputer::apply_sqop_evolution(
      const std::complex<double> time,
      const SQOperator& sqop,
      const bool antiherm,
      const bool adjoint)
{
    Tensor Cin = C_;
    // NOTE(Nick): needs gpu treatment
    evolve_individual_nbody(
        time,
        sqop,
        Cin,
        C_,
        antiherm,
        adjoint); 
}

// void FCIComputer::apply_individual_nbody1_accumulate_gpu(
//     const std::complex<double> coeff, 
//     const Tensor& Cin,
//     Tensor& Cout,
//     std::vector<int>& sourcea,
//     std::vector<int>& targeta,
//     std::vector<int>& paritya,
//     std::vector<int>& sourceb,
//     std::vector<int>& targetb,
//     std::vector<int>& parityb)
// {
    
//     local_timer my_timer = local_timer();
//     my_timer.reset();
//     if ((targetb.size() != sourceb.size()) or (sourceb.size() != parityb.size())) {
//         throw std::runtime_error("The sizes of btarget, bsource, and bparity must be the same.");
//     }

//     if ((targeta.size() != sourcea.size()) or (sourcea.size() != paritya.size())) {
//         throw std::runtime_error("The sizes of atarget, asource, and aparity must be the same.");
//     }
//     // only part that has kernel

//     // make device pointers out of all the things coming in - use cuda mem copy to a device pointer
//     my_timer.record("error checks");
//     my_timer.reset();
//     int* d_sourcea;
//     int* d_sourceb;
//     int* d_targeta;
//     int* d_targetb;
//     int* d_paritya;
//     int* d_parityb;

//     cuDoubleComplex* d_Cin;
//     cuDoubleComplex* d_Cout;

//     // cumalloc for these

//     int sourcea_mem = sourcea.size() * sizeof(int);
//     int sourceb_mem = sourceb.size() * sizeof(int);
//     int targetb_mem = targetb.size() * sizeof(int);
//     int targeta_mem = targeta.size() * sizeof(int);
//     int paritya_mem = paritya.size() * sizeof(int);
//     int parityb_mem = parityb.size() * sizeof(int);

//     int tensor_mem = Cin.size() * sizeof(std::complex<double>);

//     my_timer.record("making pointers");
//     my_timer.reset();
//     cudaMalloc(&d_sourcea, sourcea_mem);
//     cudaMalloc(&d_sourceb, sourceb_mem);
//     cudaMalloc(&d_targeta, targeta_mem);
//     cudaMalloc(&d_targetb, targetb_mem);
//     cudaMalloc(&d_paritya, paritya_mem);
//     cudaMalloc(&d_parityb, parityb_mem);

//     cudaMalloc(&d_Cin,  tensor_mem);
//     cudaMalloc(&d_Cout, tensor_mem);

//     my_timer.record("cudamallocs");
//     my_timer.reset();

//     cudaMemcpy(d_sourcea, sourcea.data(), sourcea_mem, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_sourceb, sourceb.data(), sourceb_mem, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_targeta, targeta.data(), targeta_mem, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_targetb, targetb.data(), targetb_mem, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_paritya, paritya.data(), paritya_mem, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_parityb, parityb.data(), parityb_mem, cudaMemcpyHostToDevice);

//     cudaMemcpy(d_Cin,  Cin.read_data().data(),  tensor_mem, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_Cout, Cout.read_data().data(), tensor_mem, cudaMemcpyHostToDevice);

//     my_timer.record("cudamemcpy");
//     my_timer.reset();

//     cuDoubleComplex cu_coeff = make_cuDoubleComplex(coeff.real(), coeff.imag());

//     apply_individual_nbody1_accumulate_wrapper(
//         cu_coeff, 
//         d_Cin, 
//         d_Cout, 
//         d_sourcea,
//         d_targeta,
//         d_paritya,
//         d_sourceb,
//         d_targetb,
//         d_parityb,
//         nbeta_strs_,
//         targeta.size(),
//         targetb.size(),
//         tensor_mem);
//     my_timer.record("gpu function");
//     my_timer.reset();


//     cudaMemcpy(Cout.data().data(), d_Cout, tensor_mem, cudaMemcpyDeviceToHost);


//     cudaFree(d_sourcea);
//     cudaFree(d_sourceb);
//     cudaFree(d_targeta);
//     cudaFree(d_targetb);
//     cudaFree(d_paritya);
//     cudaFree(d_parityb);
//     cudaFree(d_Cin);
//     cudaFree(d_Cout);


//     cudaError_t error = cudaGetLastError();
//     if (error != cudaSuccess) {
//         std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
//         throw std::runtime_error("Failed to execute the apply_individual_nbody1_accumulate operation on the GPU.");
//     }

//     my_timer.record("cuda free and transfer");
//     // std::cout << my_timer.str_table() << std::endl;

//     // do same thing for cin cout

//     // call the add wrapper
//     // do this in cu file with grid stride loop
//     // for (int i = 0; i < targeta.size(); i++) {
//     //     int ta_idx = targeta[i] * nbeta_strs_;
//     //     int sa_idx = sourcea[i] * nbeta_strs_;
//     //     std::complex<double> pref = coeff * static_cast<std::complex<double>>(paritya[i]);
//     //     for (int j = 0; j < targetb.size(); j++) {
//     //         Cout.data()[ta_idx + targetb[j]] += pref * static_cast<std::complex<double>>(parityb[j]) * Cin.read_data()[sa_idx + sourceb[j]];
//     //     }
//     // }



//     // free it
// }


void FCIComputer::apply_individual_nbody1_accumulate_cpu(
    const std::complex<double> coeff, 
    const Tensor& Cin,
    Tensor& Cout,
    std::vector<int>& sourcea,
    std::vector<int>& targeta,
    std::vector<int>& paritya,
    std::vector<int>& sourceb,
    std::vector<int>& targetb,
    std::vector<int>& parityb)
{
    if ((targetb.size() != sourceb.size()) or (sourceb.size() != parityb.size())) {
        throw std::runtime_error("The sizes of btarget, bsource, and bparity must be the same.");
    }

    if ((targeta.size() != sourcea.size()) or (sourcea.size() != paritya.size())) {
        throw std::runtime_error("The sizes of atarget, asource, and aparity must be the same.");
    }
    // local_timer my_timer = local_timer();
    timer_.reset();
    // only part that has kernel
    for (int i = 0; i < targeta.size(); i++) {
        int ta_idx = targeta[i] * nbeta_strs_;
        int sa_idx = sourcea[i] * nbeta_strs_;
        std::complex<double> pref = coeff * static_cast<std::complex<double>>(paritya[i]);
        for (int j = 0; j < targetb.size(); j++) {
            Cout.data()[ta_idx + targetb[j]] += pref * static_cast<std::complex<double>>(parityb[j]) * Cin.read_data()[sa_idx + sourceb[j]];
        }
    }
    
    timer_.acc_record("cpu function");
    // std::cout << my_timer.str_table() << std::endl;
}


void FCIComputer::apply_individual_nbody1_accumulate(
    const std::complex<double> coeff, 
    const Tensor& Cin,
    Tensor& Cout,
    std::vector<int>& sourcea,
    std::vector<int>& targeta,
    std::vector<int>& paritya,
    std::vector<int>& sourceb,
    std::vector<int>& targetb,
    std::vector<int>& parityb)
{
       apply_individual_nbody1_accumulate_cpu(
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

void FCIComputer::apply_individual_nbody_accumulate(
    const std::complex<double> coeff,
    const Tensor& Cin,
    Tensor& Cout,
    const std::vector<int>& daga,
    const std::vector<int>& undaga, 
    const std::vector<int>& dagb,
    const std::vector<int>& undagb)
{

    if((daga.size() != undaga.size()) or (dagb.size() != undagb.size())){
        throw std::runtime_error("must be same number of alpha anihilators/creators and beta anihilators/creators.");
    }

    local_timer my_timer = local_timer();
    my_timer.reset();

    std::tuple<int, std::vector<int>, std::vector<int>, std::vector<int>> ualfamap = graph_.make_mapping_each(
        true,
        daga,
        undaga);

    my_timer.record("first 'make_mapping_each' in apply_individual_nbody_accumulate");
    my_timer.reset();

    if (std::get<0>(ualfamap) == 0) {
        return;
    }

    std::tuple<int, std::vector<int>, std::vector<int>, std::vector<int>> ubetamap = graph_.make_mapping_each(
        false,
        dagb,
        undagb);

    my_timer.record("second 'make_mapping_each' in apply_individual_nbody_accumulate");
    my_timer.reset();

    if (std::get<0>(ubetamap) == 0) {
        return;
    }

    std::vector<int> sourcea(std::get<0>(ualfamap));
    std::vector<int> targeta(std::get<0>(ualfamap));
    std::vector<int> paritya(std::get<0>(ualfamap));
    std::vector<int> sourceb(std::get<0>(ubetamap));
    std::vector<int> targetb(std::get<0>(ubetamap));
    std::vector<int> parityb(std::get<0>(ubetamap));

    my_timer.record("a lot of initialization in apply_individual_nbody_accumulate");
    my_timer.reset();

    /// NICK: All this can be done in the make_mapping_each fucntion.
    /// Maybe try like a make_abbrev_mapping_each

    /// NICK: Might be slow, check this out...
    for (int i = 0; i < std::get<0>(ualfamap); i++) {
        sourcea[i] = std::get<1>(ualfamap)[i];
        targeta[i] = graph_.get_aind_for_str(std::get<2>(ualfamap)[i]);
        paritya[i] = 1.0 - 2.0 * std::get<3>(ualfamap)[i];
    }

    my_timer.record("first for loop in apply_individual_nbody_accumulate");
    my_timer.reset();

    for (int i = 0; i < std::get<0>(ubetamap); i++) {
        sourceb[i] = std::get<1>(ubetamap)[i];
        targetb[i] = graph_.get_bind_for_str(std::get<2>(ubetamap)[i]);
        parityb[i] = 1.0 - 2.0 * std::get<3>(ubetamap)[i];
    }

    print_vector(sourcea, "sourcea");
    print_vector(targeta, "targeta");
    print_vector(paritya, "paritya");
    print_vector(sourceb, "sourceb");
    print_vector(targetb, "targetb");
    print_vector(parityb, "parityb");

    my_timer.record("second for loop in apply_individual_nbody_accumulate");
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

void FCIComputer::apply_individual_sqop_term(
    const std::tuple< std::complex<double>, std::vector<size_t>, std::vector<size_t>>& term,
    const Tensor& Cin,
    Tensor& Cout)
{

    std::vector<int> crea;
    std::vector<int> anna;

    std::vector<int> creb;
    std::vector<int> annb;

    local_timer my_timer = local_timer();
    my_timer.reset();

    for(size_t i = 0; i < std::get<1>(term).size(); i++){
        if(std::get<1>(term)[i]%2 == 0){
            crea.push_back(std::floor(std::get<1>(term)[i] / 2));
        } else {
            creb.push_back(std::floor(std::get<1>(term)[i] / 2));
        }
    }

    my_timer.record("first loop in apply_individual_sqop_term");
    my_timer.reset();

    for(size_t i = 0; i < std::get<2>(term).size(); i++){
        if(std::get<2>(term)[i]%2 == 0){
            anna.push_back(std::floor(std::get<2>(term)[i] / 2));
        } else {
            annb.push_back(std::floor(std::get<2>(term)[i] / 2));
        }
    }

    my_timer.record("second loop in apply_individual_sqop_term");
    my_timer.reset();

    if (std::get<1>(term).size() != std::get<2>(term).size()) {
        throw std::invalid_argument("Each term must have same number of anihilators and creators");
    }   

    std::vector<size_t> ops1(std::get<1>(term));
    std::vector<size_t> ops2(std::get<2>(term));
    ops1.insert(ops1.end(), ops2.begin(), ops2.end());

    int nswaps = parity_sort(ops1);
    my_timer.record("some parity things");
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

/// NICK: Check out  accumulation, don't need to do it this way..
void FCIComputer::apply_sqop(const SQOperator& sqop){
    
    local_timer my_timer = local_timer();
    my_timer.reset();
    Tensor Cin = C_;
    C_.zero();

    my_timer.record("making tensor things");
    my_timer.reset();

    for (const auto& term : sqop.terms()) {
        if(std::abs(std::get<0>(term)) > compute_threshold_){
        apply_individual_sqop_term(
            term,
            Cin,
            C_);
        }
    }
    my_timer.record("first for loop in apply_sqop");
    // std::cout << my_timer.str_table() << std::endl;
}

/// diagonal only 
void FCIComputer::apply_diagonal_of_sqop(const SQOperator& sqop, const bool invert_coeff){
    Tensor Cin = C_;
    C_.zero();

    for(const auto& term : sqop.terms()){
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

            apply_individual_sqop_term(
                temp_term,
                Cin,
                C_);
        }
    }
}

/// NICK: Check out  accumulation, don't need to do it this way..
void FCIComputer::apply_sqop_pool(const SQOpPool& sqop_pool){
    Tensor Cin = C_;
    C_.zero();

    for (const auto& sqop : sqop_pool.terms()) {
        std::complex<double> outer_coeff = sqop.first;
        for (const auto& term : sqop.second.terms()) {
            std::tuple< std::complex<double>, std::vector<size_t>, std::vector<size_t>> temp_term = term;

            std::get<0>(temp_term) *= outer_coeff;

            if(std::abs(std::get<0>(temp_term)) > compute_threshold_){
                apply_individual_sqop_term(
                    temp_term,
                    Cin,
                    C_);
            }
        }
    }
}

/// NICK: Check out  accumulation, don't need to do it this way..
std::complex<double> FCIComputer::get_exp_val(const SQOperator& sqop) {
    Tensor Cin = C_;
    C_.zero();
    for (const auto& term : sqop.terms()) {
        if(std::abs(std::get<0>(term)) > compute_threshold_){
        apply_individual_sqop_term(
            term,
            Cin,
            C_);
        }
    }
    std::complex<double> val = C_.vector_dot(Cin);
    C_ = Cin;
    return val;
}

std::complex<double> FCIComputer::get_exp_val_tensor(
    const std::complex<double> h0e, 
    const Tensor& h1e, 
    const Tensor& h2e, 
    const Tensor& h2e_einsum, 
    size_t norb)  
{
    Tensor Cin = C_;

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

// ====> Double Factorization functions and such <=== //

// NOTE(Nick): This funciton may be SLOW, will want to reduce the amount of 
// tensory copying to accelerate.
void FCIComputer::apply_df_ham(
      const DFHamiltonian& df_ham,
      const double nuc_rep_en)
{
    size_t nleaves = df_ham.get_basis_change_matrices().size();

    if (nleaves != df_ham.get_scaled_density_density_matrices().size()){
        throw std::invalid_argument("Incompatiable array lengths.");
    }

    Tensor Cin = C_;
    Tensor Cnew({nalfa_strs_, nbeta_strs_}, "Cnew");
    Tensor H1 = df_ham.get_one_body_ints();
    H1.add(df_ham.get_one_body_correction());

    // first apply the modified one body term
    apply_array_1bdy(
        Cnew,
        graph_.read_dexca_vec(),
        nalfa_strs_,
        nbeta_strs_,
        graph_.get_ndexca(),
        H1,
        norb_,
        true);

    C_.transpose();

    apply_array_1bdy(
        Cnew,
        graph_.read_dexcb_vec(),
        nbeta_strs_,
        nalfa_strs_,
        graph_.get_ndexcb(),
        H1,
        norb_,
        false);

    for (size_t l = 0; l < nleaves; ++l) {

        // reset to the initial state
        C_ = Cin;

        SQOpPool glpool;
        SQOpPool dlpool;
        
        // get pool for Gl_alfa
        glpool.append_givens_ops_sector(
            df_ham.get_basis_change_matrices()[l],
            1.0,
            true
        );

        // get pool for Gl_beta
        glpool.append_givens_ops_sector(
            df_ham.get_basis_change_matrices()[l],
            1.0,
            false
        );

        // get pool for dl_beta
        dlpool.append_diagonal_ops_all(
            df_ham.get_scaled_density_density_matrices()[l],
            1.0
        );

        // apply Gl
        evolve_pool_trotter_basic(
            glpool,
            false,
            false
        );

        // NOTE(Nick): SLOW, we want the below funciton for fast diagonal applicaion
        // apply_diagonal_from_mat(
        //     df_ham.get_scaled_density_density_matrices()[l]
        // );

        apply_sqop_pool(dlpool);

        // apply Gl^dag
        evolve_pool_trotter_basic(
            glpool,
            false,
            true
        );

        // accumulate
        Cnew.zaxpy(
            C_,
            1.0,
            1,
            1
        );
    }

    // account for zero body energy
    Cnew.zaxpy(
            Cin,
            nuc_rep_en,
            1,
            1
        );

    C_ = Cnew;
}

void FCIComputer::evolve_df_ham_trotter(
      const DFHamiltonian& df_ham,
      const double evolution_time)
{
    size_t nleaves = df_ham.get_trotter_basis_change_matrices().size();

    if (nleaves - 1 != df_ham.get_scaled_density_density_matrices().size()){
        throw std::invalid_argument("Incompatiable array lengths.");
    }


    evolve_givens(
        df_ham.get_trotter_basis_change_matrices()[0],
        true);

    evolve_givens(
        df_ham.get_trotter_basis_change_matrices()[0],
        false);

    

    for (size_t l = 1; l < nleaves; ++l) {

        evolve_diagonal_from_mat(
            df_ham.get_scaled_density_density_matrices()[l - 1],
            evolution_time
        );
        
        evolve_givens(
            df_ham.get_trotter_basis_change_matrices()[l],
            true
        );

        evolve_givens(
            df_ham.get_trotter_basis_change_matrices()[l],
            false
        );
    }

}

// NOTE(Nick): could be made faster if this becomes a popular subroutine
void FCIComputer::apply_two_determinant_rotations(
      const std::vector<std::vector<size_t>> IJ_source,
      const std::vector<std::vector<size_t>> IJ_target,
      const std::vector<double> angles,
      const bool adjoint
    )
{
    if(IJ_source.size() != IJ_target.size() or IJ_source.size() != angles.size()){
        throw std::invalid_argument("IJ_source, IJ_target, and angles to all have the same size");
    }

    size_t N = IJ_source.size();

    // Tensor Cin = C_;

    if(adjoint){
        for(int n = N-1; n >= 0; --n){
            std::complex<double> C_source = C_.get(IJ_source[n]);
            std::complex<double> C_target = C_.get(IJ_target[n]);
            double theta = angles[n];

            std::complex<double> C_source_prime =  std::cos(theta)*C_source + std::sin(theta)*C_target;
            std::complex<double> C_target_prime = -std::sin(theta)*C_source + std::cos(theta)*C_target;

            C_.set(IJ_source[n], C_source_prime);
            C_.set(IJ_target[n], C_target_prime);
    }
    } else {
        for(int n = 0; n < N; ++n){
            std::complex<double> C_source = C_.get(IJ_source[n]);
            std::complex<double> C_target = C_.get(IJ_target[n]);
            double theta = angles[n];

            std::complex<double> C_source_prime = std::cos(theta)*C_source - std::sin(theta)*C_target;
            std::complex<double> C_target_prime = std::sin(theta)*C_source + std::cos(theta)*C_target;

            C_.set(IJ_source[n], C_source_prime);
            C_.set(IJ_target[n], C_target_prime);
        }
    }

    

}

// NOTE(Nick): If this proves exceedingly slow,
// it is possible to apply blocks of these givens
// rotations in parallel, will look into this if
// it is a probelm.
void FCIComputer::evolve_givens(
    const Tensor& U,
    const bool is_alfa)
{

    size_t sigma = 0;
    if (is_alfa){ 
        sigma = 0; 
    } else {
        sigma = 1;
    }

    U.square_error();

    if (U.shape()[0] != norb_) {
        throw std::invalid_argument("U must be a square norb x norb matrix.");
    }

    Tensor U2 = U;

    //NOTE(Nick): May be SLOW, or don't need to compute, could just store rots_and_diag
    // in DFHamiltonain class pass directly.
    auto rots_and_diag = DFHamiltonian::givens_decomposition_square(U2);

    auto ivec = std::get<0>(rots_and_diag);
    auto jvec = std::get<1>(rots_and_diag);
    auto thts = std::get<2>(rots_and_diag);
    auto phis = std::get<3>(rots_and_diag);

    auto diags = std::get<4>(rots_and_diag);

    // Iterate through each layer and time evolve by the appropriate
    // SQOperators
    for (size_t k = 0; k < ivec.size(); k++){
        size_t i = ivec[k];
        size_t j = jvec[k];
        double tht = thts[k];
        double phi = phis[k];

        if (std::abs(phi) > compute_threshold_){
            SQOperator num_op1; 
            num_op1.add_term(-phi/2.0, {2 * j + sigma}, {2 * j + sigma});
            num_op1.add_term(-phi/2.0, {2 * j + sigma}, {2 * j + sigma});            
            apply_sqop_evolution(1.0, num_op1);
        }

        if (std::abs(tht) > compute_threshold_) {
            std::complex<double> itheta(0.0, tht);
            SQOperator single;
            single.add_term(-itheta, {2 * i + sigma}, {2 * j + sigma});
            single.add_term(+itheta, {2 * j + sigma}, {2 * i + sigma});
            apply_sqop_evolution(1.0, single);
        }
    }
        
    // Evolve the last diagonal phases
    for (size_t l = 0; l < diags.size(); l++){
        if (std::abs(diags[l]) > 1.0e-13) {
            double diag_angle = std::atan2(diags[l].imag(), diags[l].real());
            SQOperator num_op2;
            num_op2.add_term(-diag_angle/2.0, {2 * l + sigma}, {2 * l + sigma});
            num_op2.add_term(-diag_angle/2.0, {2 * l + sigma}, {2 * l + sigma});
            apply_sqop_evolution(1.0, num_op2);
        }
    }
}

void FCIComputer::evolve_diagonal_from_mat(
    const Tensor& V,
    const double evolution_time)
{

    // NOTE(Nick): time sclae the arrays, maybe make V non const and just scale by 1/t after?
    Tensor V2 = V;
    Tensor D({V.shape()[0]}, "D2");

    std::complex<double> idt(0.0, -evolution_time); 
    V2.scale(idt);

    apply_diagonal_array(
        C_, 
        graph_.get_astr(),
        graph_.get_bstr(),
        D,
        V2,
        nalfa_strs_,
        nbeta_strs_,
        nalfa_el_,
        nbeta_el_,
        norb_);

}


void FCIComputer::apply_diagonal_from_mat(const Tensor& V)
{

    throw std::invalid_argument("apply_diagonal_from_mat is not yet funcitonal");

    Tensor D({V.shape()[0]}, "D2");

    apply_diagonal_array(
        C_, 
        graph_.get_astr(),
        graph_.get_bstr(),
        D,
        V,
        nalfa_strs_,
        nbeta_strs_,
        nalfa_el_,
        nbeta_el_,
        norb_,
        false);
}

void FCIComputer::apply_diagonal_array(
        Tensor& C, // Just try in-place for now...
        const std::vector<uint64_t>& astrs,
        const std::vector<uint64_t>& bstrs,
        const Tensor& D,
        const Tensor& V,
        const size_t nalfa_strs,
        const size_t nbeta_strs,
        const size_t nalfa_el,
        const size_t nbeta_el,
        const size_t norb,
        const bool exponentiate)
{
    D.shape_error({norb});
    V.shape_error({norb, norb});

    std::vector<std::complex<double>> adiag(nalfa_strs);
    std::vector<std::complex<double>> bdiag(nbeta_strs);
    std::vector<int> aocc(nalfa_strs * nalfa_el);
    std::vector<int> bocc(nbeta_strs * nbeta_el);

    for (int as = 0; as < nalfa_strs; ++as) {
        std::vector<int> astr_occ = graph_.get_positions(astrs[as], nalfa_el);
        if (astr_occ.size() != nalfa_el) {
            std::cerr << "Error: astr_occ size mismatch at as=" << as << std::endl;
            return;
        }
        std::copy(astr_occ.begin(), astr_occ.end(), aocc.begin() + nalfa_el * as);
    }

    for (int bs = 0; bs < nbeta_strs; ++bs) {
        std::vector<int> bstr_occ = graph_.get_positions(bstrs[bs], nbeta_el);
        if (bstr_occ.size() != nbeta_el) {
            std::cerr << "Error: bstr_occ size mismatch at bs=" << bs << std::endl;
            return;
        }
        std::copy(bstr_occ.begin(), bstr_occ.end(), bocc.begin() + nbeta_el * bs);
    }

    std::vector<std::complex<double>> diagexp(norb);
    std::vector<std::complex<double>> arrayexp(norb * norb);


    // NOTE(Nick), we don't need to do this every time...,
    // can pre-compute the exponentias and store them
    // in DFHamiltonian

    // NOTE(Nick): Also, the non exponeintal version of this funciton is not yet working...

    if(exponentiate){
        for (int i = 0; i < norb; ++i) {
            diagexp[i] = std::exp(D.read_data()[i]);
        }

        for (int i = 0; i < norb * norb; ++i) {
            arrayexp[i] = std::exp(V.read_data()[i]);
        }
    } else {
        for (int i = 0; i < norb; ++i) {
            diagexp[i] = D.read_data()[i];
        }

        for (int i = 0; i < norb * norb; ++i) {
            arrayexp[i] = V.read_data()[i];
        }
    }
    

    apply_diagonal_array_part(
        adiag, 
        aocc, 
        diagexp, 
        arrayexp, 
        nalfa_strs, 
        nalfa_el, 
        norb);

    apply_diagonal_array_part(
        bdiag,
        bocc, 
        diagexp, 
        arrayexp, 
        nbeta_strs, 
        nbeta_el, 
        norb);

    std::vector<std::complex<double>> aarrays(norb);

    for (int as = 0; as < nalfa_strs; ++as) {
        const int* caocc = aocc.data() + (as * nalfa_el);
        std::fill(aarrays.begin(), aarrays.end(), std::complex<double>(1.0));

        for (int ela = 0; ela < nalfa_el; ++ela) {
            const std::complex<double>* carray = arrayexp.data() + (caocc[ela] * norb);
            for (int i = 0; i < norb; ++i) {
                aarrays[i] *= carray[i];
            }
        }

        for (int bs = 0; bs < nbeta_strs; ++bs) {
            std::complex<double> diag_ele = 1.0;
            const int* cbocc = bocc.data() + (bs * nbeta_el);
            for (int elb = 0; elb < nbeta_el; ++elb) {
                diag_ele *= aarrays[cbocc[elb]];
            }
            C.data()[as * nbeta_strs + bs] *= diag_ele * diag_ele * bdiag[bs] * adiag[as];
        }
    // }
    }
}

void FCIComputer::apply_diagonal_array_part(
        std::vector<std::complex<double>>& out, // is the output?
        const std::vector<int>& occ, 
        const std::vector<std::complex<double>>& diag, 
        const std::vector<std::complex<double>>& array, 
        const size_t nstrs, 
        const size_t nel, 
        const size_t norb)
{
    for (int i = 0; i < nstrs; ++i) {
        std::complex<double> p_adiag = 1.0;
        const int *c_occ = occ.data() + (i * nel);

        for (int el = 0; el < nel; ++el) {
            p_adiag *= diag[c_occ[el]];

            const std::complex<double>* c_array = array.data() + norb * c_occ[el];
            for (int el2 = 0; el2 < nel; ++el2) {
                p_adiag *= c_array[c_occ[el2]];
            }
        }
        out[i] = p_adiag;
    }
}

// ====> Helper and other basic functions below <=== //

/// apply a constant to the FCI quantum computer.
void FCIComputer::scale(const std::complex<double> a){
    C_.scale(a);
}


// std::vector<std::complex<double>> FCIComputer::direct_expectation_value(const TensorOperator& top){
//     // Stuff
// }

void FCIComputer::set_state(const Tensor& other_state) {
    C_.copy_in(other_state);
}

/// Sets all coefficeints fo the FCI Computer to Zero except the HF Determinant (set to 1).
void FCIComputer::hartree_fock() {
    C_.zero();
    C_.set({0, 0}, 1.0);
}

/// Sets all coefficeints fo the FCI Computer to Zero
void FCIComputer::zero_state() {
    C_.zero();
}

/// return the indexes of non-zero elements
std::vector<std::vector<size_t>> FCIComputer::get_nonzero_idxs() const 
{
    std::vector<std::vector<size_t>> temp;

    // Think about how to add the non-zero indicies to temp by looping over the matrix..
    // for i, j matrix indicies

    for(size_t i = 0; i < nalfa_strs_; i++) {
        for (size_t j = 0; j < nbeta_strs_; j++) {
            if(std::abs(C_.read_data()[i*nbeta_strs_ + j]) > compute_threshold_){
                std::vector<size_t> idxs = {i, j};
                temp.push_back(idxs);
            }
        }
    }
    return temp;
}

void FCIComputer::print_vector(const std::vector<int>& vec, const std::string& name) {
    std::cout << "\n" << name << ": ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << static_cast<int>(vec[i]);
        if (i < vec.size() - 1) {
           std::cout << ", "; 
        }
    }
    std::cout << std::endl;
}

void FCIComputer::print_vector_z(const std::vector<std::complex<double>>& vec, const std::string& name) {
    std::cout << "\n" << name << ": ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << static_cast<std::complex<double>>(vec[i]);
        if (i < vec.size() - 1) {
           std::cout << ", "; 
        }
    }
    std::cout << std::endl;
}

void FCIComputer::print_vector_uint(const std::vector<uint64_t>& vec, const std::string& name) {
    std::cout << "\n" << name << ": ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) {
            std::cout << ", "; 
        }
    }
    std::cout << std::endl;
}



