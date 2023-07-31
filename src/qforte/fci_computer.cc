#include <map>
#include <random>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <string>

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

#include "fci_computer.h"

// #if defined(_OPENMP)
// #include <omp.h>
// extern const bool parallelism_enabled = true;
// #else
// extern const bool parallelism_enabled = false;
// #endif

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

    state_.zero_with_shape({nalfa_strs_, nbeta_strs_});
    state_.set_name("FCI Computer");
}

/// apply a TensorOperator to the current state 
void apply_tensor_operator(const TensorOperator& top);

/// apply a 1-body TensorOperator to the current state 
void apply_tensor_spin_1bdy(const TensorOperator& top);

// TODO(Nick): Uncomment, will need someting like the following:
// std::vector<double> apply_array_spin1(const std::vector<double>& h1e, int norb) {
// assert(h1e.size() == (norb * 2) * (norb * 2));

// int ncol = 0;
// int jorb = 0;
// for (int j = 0; j < norb * 2; ++j) {
// bool anyNonZero = false;
// for (int i = 0; i < norb * 2; ++i) {
//     if (h1e[i + j * (norb * 2)] != 0) {
//         anyNonZero = true;
//         break;
//     }
// }
// if (anyNonZero) {
//     ncol += 1;
//     jorb = j;
// }
// if (ncol > 1) {
//     break;
// }
// }

// std::vector<double> out;
// if (ncol > 1) {
// // Implementation of dense_apply_array_spin1_lm
// out = lm_apply_array1(coeff, {h1e.begin(), h1e.begin() + norb * norb},
//                         core._dexca, lena(), lenb(), norb, true);
// lm_apply_array1(coeff, {h1e.begin() + norb * norb, h1e.end()},
//                 core._dexcb, lena(), lenb(), norb, false, out);
// } else {
// if (jorb < norb) {
//     std::vector<double> dvec = calculate_dvec_spin_fixed_j(jorb);
//     std::vector<double> h1eview(norb);
//     for (int i = 0; i < norb; ++i) {
//         h1eview[i] = h1e[i + jorb * (norb * 2)];
//     }
//     out = tensordot(h1eview, dvec, 1);
// } else {
//     std::vector<double> dvec = calculate_dvec_spin_fixed_j(jorb);
//     std::vector<double> h1eview(norb);
//     for (int i = 0; i < norb; ++i) {
//         h1eview[i] = h1e[i + jorb * (norb * 2)];
//     }
//     out = tensordot(h1eview, dvec, 1);
// }
// }

// return out;
// }

/// apply a 1-body and 2-body TensorOperator to the current state 
void apply_tensor_spin_12_body(const TensorOperator& top){
    // Stuff
}

void apply_sqop(const SQOperator& sqop){
    // Stuff
}

void FCIComputer::apply_sqop_single_term(std::complex<double> coeff, 
                            std::vector<int> daga,
                            std::vector<int> undaga, 
                            std::vector<int> dagb, 
                            std::vector<int> undagb)
{

    if (daga.size() + dagb.size() != undaga.size() + undagb.size()){
        throw std::runtime_error("dag/undag lists aren't the same size.\n");
    }

    int nda = daga.size() - undaga.size();

    // this might be unnecessary
    size_t nalpha =  nalfa_el_;
    size_t nbeta = nbeta_el_;
    std::vector<size_t> source = {nalpha, nbeta, norb_};

    // this line should say "if the nalpha and nbeta are in the sector we are looking at"
    // but since we are only looking at one, i don't think it is necessary.
    if (true){

        auto& target = source; //FqeData object which is basically nalpha, nbeta, norb

        // lenb and lena is the length of beta/alpha configuration space??
        std::vector<std::vector<uint64_t>> ualphamap(source[0], std::vector<uint64_t>(3, 0));
        std::vector<std::vector<uint64_t>> ubetamap(source[1], std::vector<uint64_t>(3, 0));

        int acount = source._core.make_mapping_each(ualphamap, true, daga, undaga);
        int bcount = source._core.make_mapping_each(ubetamap, false, dagb, undagb);
        ualphamap.resize(acount);
        ubetamap.resize(bcount);

    }

/*
class FqeData:
    """This is a basic data structure for use in the FQE.
    """

    def __init__(self,
                 nalpha: int,
                 nbeta: int,
                 norb: int,

*/



 

}




/*

    void apply_individual_nbody_accumulate(std::complex<double> coeff,
                                           FqeDataSet& idata,
                                           const std::vector<int>& daga,
                                           const std::vector<int>& undaga,
                                           const std::vector<int>& dagb,
                                           const std::vector<int>& undagb)
    {
        assert(daga.size() + dagb.size() == undaga.size() + undagb.size());
        int nda = daga.size() - undaga.size();

        for (const auto& entry : idata._data)
        {
            int nalpha = entry.first.first;
            int nbeta = entry.first.second;
            const auto& source = entry.second;

            if (idata._data.count({nalpha + nda, nbeta - nda}))
            {
                auto& target = idata._data[{nalpha + nda, nbeta - nda}];

                std::vector<std::vector<uint64_t>> ualphamap(source.lena(), std::vector<uint64_t>(3, 0));
                std::vector<std::vector<uint64_t>> ubetamap(source.lenb(), std::vector<uint64_t>(3, 0));

                int acount = source._core.make_mapping_each(ualphamap, true, daga, undaga);
                int bcount = source._core.make_mapping_each(ubetamap, false, dagb, undagb);
                ualphamap.resize(acount);
                ubetamap.resize(bcount);

                std::vector<std::vector<int64_t>> alphamap(acount, std::vector<int64_t>(3, 0));
                std::vector<std::vector<int64_t>> betamap(bcount, std::vector<int64_t>(3, 0));

                for (int i = 0; i < acount; ++i)
                {
                    alphamap[i][0] = ualphamap[i][0];
                    alphamap[i][1] = target._core.index_alpha(ualphamap[i][1]);
                    alphamap[i][2] = 1 - 2 * ualphamap[i][2];
                }

                for (int i = 0; i < bcount; ++i)
                {
                    betamap[i][0] = ubetamap[i][0];
                    betamap[i][1] = target._core.index_beta(ubetamap[i][1]);
                    betamap[i][2] = 1 - 2 * ubetamap[i][2];
                }

                if (fqe::settings.use_accelerated_code)
                {
                    if (!alphamap.empty() && !betamap.empty())
                    {
                        double pfac = pow(-1.0, (dagb.size() + undagb.size()) * nalpha);
                        std::vector<int> sourceb_vec;
                        std::vector<int> targetb_vec;
                        std::vector<double> parityb_vec;
                        for (const auto& entry : betamap)
                        {
                            sourceb_vec.push_back(entry[0]);
                            targetb_vec.push_back(entry[1]);
                            parityb_vec.push_back(entry[2] * pfac);
                        }
                        _apply_individual_nbody1_accumulate(coeff, target.coeff, source.coeff, alphamap, targetb_vec, sourceb_vec, parityb_vec);
                    }
                }
                else
                {
                    for (const auto& entrya : alphamap)
                    {
                        int sourcea = entrya[0];
                        int targeta = entrya[1];
                        double paritya = entrya[2];
                        paritya *= pow(-1.0, (dagb.size() + undagb.size()) * nalpha);
                        for (const auto& entryb : betamap)
                        {
                            int sourceb = entryb[0];
                            int targetb = entryb[1];
                            double parityb = entryb[2];
                            double work = coeff * source.coeff[sourcea][sourceb];
                            target.coeff[targeta][targetb] += work * paritya * parityb;
                        }
                    }
                }
            }
        }
    }

private:
    // Define other private member functions if needed.
};

*/


/*
def apply_individual_nbody_accumulate(self, coeff: complex,
                                        idata: 'FqeDataSet', daga: List[int],
                                        undaga: List[int], dagb: List[int],
                                        undagb: List[int]) -> None:
    """
    Apply function with an individual operator represented in arrays,
    which can handle spin-nonconserving operators

    Args:
        coeff (complex): scalar coefficient to be multiplied to the result

        idata (FqeDataSet): input FqeDataSet to which the operators are applied

        daga (List[int]): indices corresponding to the alpha creation \
            operators in the Hamiltonian

        undaga (List[int]): indices corresponding to the alpha annihilation \
            operators in the Hamiltonian

        dagb (List[int]): indices corresponding to the beta creation \
            operators in the Hamiltonian

        undagb (List[int]): indices corresponding to the beta annihilation \
            operators in the Hamiltonian
    """
    assert len(daga) + len(dagb) == len(undaga) + len(undagb)
    nda = len(daga) - len(undaga)

    for (nalpha, nbeta), source in idata._data.items():
        if (nalpha + nda, nbeta - nda) in self._data.keys():
            target = self._data[(nalpha + nda, nbeta - nda)]

            ualphamap = numpy.zeros((source.lena(), 3), dtype=numpy.uint64)
            ubetamap = numpy.zeros((source.lenb(), 3), dtype=numpy.uint64)

            acount = source._core.make_mapping_each(ualphamap, True, daga,
                                                    undaga)
            bcount = source._core.make_mapping_each(ubetamap, False, dagb,
                                                    undagb)
            ualphamap = ualphamap[:acount, :]
            ubetamap = ubetamap[:bcount, :]

            alphamap = numpy.zeros((acount, 3), dtype=numpy.int64)
            betamap = numpy.zeros((bcount, 3), dtype=numpy.int64)

            alphamap[:, 0] = ualphamap[:, 0]
            for i in range(acount):
                alphamap[i, 1] = target._core.index_alpha(ualphamap[i, 1])
            alphamap[:, 2] = 1 - 2 * ualphamap[:, 2]

            betamap[:, 0] = ubetamap[:, 0]
            for i in range(bcount):
                betamap[i, 1] = target._core.index_beta(ubetamap[i, 1])
            betamap[:, 2] = 1 - 2 * ubetamap[:, 2]

            if fqe.settings.use_accelerated_code:
                if alphamap.size != 0 and betamap.size != 0:
                    pfac = (-1)**((len(dagb) + len(undagb)) * nalpha)
                    sourceb_vec = numpy.array(betamap[:, 0])
                    targetb_vec = numpy.array(betamap[:, 1])
                    parityb_vec = numpy.array(betamap[:, 2]) * pfac
                    _apply_individual_nbody1_accumulate(
                        coeff, target.coeff, source.coeff, alphamap,
                        targetb_vec, sourceb_vec, parityb_vec)
            else:
                for sourcea, targeta, paritya in alphamap:
                    paritya *= (-1)**((len(dagb) + len(undagb)) * nalpha)
                    for sourceb, targetb, parityb in betamap:
                        work = coeff * source.coeff[sourcea, sourceb]
                        target.coeff[targeta,
                                        targetb] += work * paritya * parityb
*/




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
    // Stuff
}



