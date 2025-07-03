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
#include "fci_graph.h"

#include "fci_computer_thrust.h"


FCIComputerThrust::FCIComputerThrust(int nel, int sz, int norb, bool on_gpu) : 
    nel_(nel), sz_(sz), norb_(norb), on_gpu_(on_gpu) {
    
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

/// Get a particular element of the tensor stored in FCIComputerThrust, specified by idxs
std::complex<double> FCIComputerThrust::get_element(
    const std::vector<size_t>& idxs
        ) const 
{    
    return C_.get(idxs);
}

void FCIComputerThrust::cpu_error() const {
    if (on_gpu_) {
        throw std::runtime_error("Data not on CPU for FCIComputerThrust" + name_);
    }
}

void FCIComputerThrust::gpu_error() const {
    if (!on_gpu_) {
        throw std::runtime_error("Data not on GPU for FCIComputerThrust" + name_);
    }
}

void FCIComputerThrust::to_gpu() {
    cpu_error();
    C_.to_gpu();
    on_gpu_ = 1;
}

void FCIComputerThrust::to_cpu() {
    gpu_error();
    C_.to_cpu();
    on_gpu_ = 0;
}

void FCIComputerThrust::apply_tensor_spin_1bdy(
    const TensorGPUThrust& h1e, 
    size_t norb) {
    
    if(h1e.size() != (norb * 2) * (norb * 2)){
        throw std::invalid_argument("Expecting h1e to be nso x nso for apply_tensor_spin_1bdy");
    }

    TensorGPUThrust Cnew({nalfa_strs_, nbeta_strs_}, "Cnew");
    TensorGPUThrust h1e_blk1 = h1e.slice(
        {
            std::make_pair(0, norb_), 
            std::make_pair(0, norb_)
            }
        );

    TensorGPUThrust h1e_blk2 = h1e.slice(
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


/// TODO: ask nick if this makes sense or if a new solution is need so don't
/// have to copy data around in this function 
/// NICK: make this more c++ style, not a fan of the raw pointers :(
void FCIComputerThrust::apply_array_1bdy(
    TensorGPUThrust& out,
    const std::vector<int>& dexc,
    const int astates,
    const int bstates,
    const int ndexc,
    const TensorGPUThrust& h1e,
    const int norbs,
    const bool is_alpha)
{
    const int states1 = is_alpha ? astates : bstates;
    const int states2 = is_alpha ? bstates : astates;
    const int inc1 = is_alpha ? bstates : 1;
    const int inc2 = is_alpha ? 1 : bstates;

    // Get copies of the data that we'll manipulate
    std::vector<std::complex<double>> out_data = out.read_data();
    std::vector<std::complex<double>> h1e_data = h1e.read_data();
    std::vector<std::complex<double>> c_data = C_.read_data();
    
    // Create temporary workspace for math_zaxpy to operate on
    std::vector<std::complex<double>> work_data = out_data;
    
    for (int s1 = 0; s1 < states1; ++s1) {
        const int* cdexc = dexc.data() + 3 * s1 * ndexc;
        const int* lim1 = cdexc + 3 * ndexc;
        std::complex<double>* cout = work_data.data() + s1 * inc1;

        for (; cdexc < lim1; cdexc = cdexc + 3) {
            const int target = cdexc[0];
            const int ijshift = cdexc[1];
            const int parity = cdexc[2];

            const std::complex<double> pref = static_cast<double>(parity) * h1e_data[ijshift];
            const std::complex<double>* xptr = c_data.data() + target * inc1;

            math_zaxpy(states2, pref, xptr, inc2, cout, inc2);
        }
    }
    
    // Update the tensor with our manipulated data
    out.fill_from_nparray(work_data, out.shape());
}