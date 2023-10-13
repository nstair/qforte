#include <map>
#include <random>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <cmath>

// #include "fmt/format.h"

#include "qubit_basis.h"
#include "circuit.h"
#include "gate.h"
#include "helpers.h"
#include "qubit_operator.h"
#include "tensor.h"
#include "tensor_operator.h"
#include "qubit_op_pool.h"
#include "timer.h"
#include "sq_operator.h"
#include "blas_math.h"

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


/// apply a 1-body and 2-body TensorOperator to the current state 
void apply_tensor_spin_12_body(const TensorOperator& top){
    // Stuff
}

void apply_sqop(const SQOperator& sqop){
    // Stuff
}


// dont know the dimensions of arguments? since theyre being indexed multiple times

//coeff (c_ variable, the state, memebr variable), ocoeff, icoeff are all tensors
void FCIComputer::apply_individual_nbody1_accumulate(
    Tensor& coeff, 
    Tensor& ocoeff,
    Tensor& icoeff,
    std::vector<std::tuple<size_t, size_t, double>>& amap, 
    std::vector<size_t>& btarget,
    std::vector<size_t>& bsource,
    std::vector<double>& bparity)
{
    // Check size validity (in a real application, more robust checks might be necessary)
    if (btarget.size() != bsource.size() || bsource.size() != bparity.size()) {
        throw std::runtime_error("The sizes of btarget, bsource, and bparity must be the same.");
    }

    for (const auto& entry : amap) {
        size_t sourcea = std::get<0>(entry);
        size_t targeta = std::get<1>(entry);
        double paritya = std::get<2>(entry);
        
        for (size_t i = 0; i < btarget.size(); i++) {
            // Indices for the tensors
            std::vector<size_t> idx_target = {targeta, btarget[i]};
            std::vector<size_t> idx_source = {sourcea, bsource[i]};

            // Update ocoeff based on the formula
            ocoeff.set(idx_target, ocoeff.get(idx_target) + coeff.get({}) * paritya * icoeff.get(idx_source) * bparity[i]);
        }
    }

}

// do i even need idata as an argument?
void FCIComputer::apply_individual_nbody_accumulate(
    const std::complex<double>& coeff,
    const std::vector<int>& daga,
    const std::vector<int>& undaga, 
    const std::vector<int>& dagb,
    const std::vector<int>& undagb)
{

    assert(daga.shape()[0] == undaga.shape()[0] && dagb.shape()[0] == undagb.shape()[0]);

    Tensor ualphamap({lena(), 3}, "ualphamap");
    Tensor ubetamap({lenb(), 3}, "ubetamap");

    int acount = _core.make_mapping_each(ualphamap, true, daga, undaga);
    if (acount == 0) {
        return;
    }
    int bcount = _core.make_mapping_each(ubetamap, false, dagb, undagb);
    if (bcount == 0) {
        return;
    }

    ualphamap = ualphamap.slice({{0, acount}, {0, 3}});
    ubetamap = ubetamap.slice({{0, bcount}, {0, 3}});

    Spinmap alphamap({acount, 3}, "alphamap"); // Not going to be a tensor, it is a spin map 
    Tensor sourceb_vec({bcount}, "sourceb_vec");
    Tensor targetb_vec({bcount}, "targetb_vec");
    Tensor parityb_vec({bcount}, "parityb_vec");

    for (int i = 0; i < acount; ++i) {
        alphamap.set({i, 0}, ualphamap.get({i, 0}));
        alphamap.set({i, 1}, _core.index_alpha(ualphamap.get({i, 1})));
        alphamap.set({i, 2}, 1.0 - 2.0 * ualphamap.get({i, 2}));
    }

    for (int i = 0; i < bcount; ++i) {
        sourceb_vec.set({i}, ubetamap.get({i, 0}));
        targetb_vec.set({i}, _core.index_beta(ubetamap.get({i, 1})));
        parityb_vec.set({i}, 1.0 - 2.0 * ubetamap.get({i, 2}));
    }

    if (/*fqe.settings.use_accelerated_code*/) { 
        // _apply_individual_nbody1_accumulate(...); 
    } else {
        //_apply_individual_nbody1_accumulate_python(...); 
    }
}


/// apply a constant to the FCI quantum computer.
void scale(const std::complex<double> a);


// std::vector<std::complex<double>> FCIComputer::direct_expectation_value(const TensorOperator& top){
//     // Stuff
// }

void FCIComputer::set_state(const Tensor other_state) {
    // Stuff
}

void FCIComputer::zero() {
    // Stuff
}

/// Sets all coefficeints fo the FCI Computer to Zero except the HF Determinant (set to 1).
void FCIComputer::hartree_fock() {
    C_.zero();
    C_.set({0, 0}, 1.0);
}



