#ifndef _sq_op_pool_thrust_h_
#define _sq_op_pool_thrust_h_

#include <complex>
#include <string>
#include <vector>
#include <tuple>
#include <iostream>
#include <iomanip>

#include <cuda_runtime.h>
#include <cuComplex.h>
// #include <thrust/host_vector.h>
#include <thrust/device_vector.h>
// #include <thrust/transform.h>
// #include <thrust/functional.h>
// #include <thrust/inner_product.h>
// #include <thrust/fill.h>
// #include <thrust/copy.h>


#include <thrust/host_vector.h>

// #include "qforte-def.h"

class SQOperator;
class QubitOperator;
class QubitOpPool;

// Represents an arbitrary linear combination of second quantized operators.
// May also represent an array of second quantized operators by ignoring
// the coefficients.
class SQOpPoolThrust {
  public:
    /// default constructor: creates an empty second quantized operator pool
    SQOpPoolThrust() {}

    /// add one set of annihilators and/or creators to the second quantized operator pool
    void add_term(std::complex<double> coeff, const SQOperator& sq_op );

    /// sets the operator pool coefficients
    void set_coeffs(const std::vector<std::complex<double>>& new_coeffs);

    /// sets the operator pool coefficients
    void set_coeffs_to_scaler(std::complex<double> new_coeff);

    /// return a vector of terms and their coeficients
    const std::vector<std::pair< std::complex<double>, SQOperator>>& terms() const;

    /// ===> Getters/Setters for GPU term maps (mutable) <===

    /// return a bool indicating whether the device stp arrays have been populated
    bool device_vecs_populated() const { return device_vecs_populated_; } 

    /// set whether the divice vecs have already been populated or not
    void set_device_vecs_populated(bool val) { device_vecs_populated_ = val; }

    /// return a mutable vector of term coeficients h_mu
    std::vector<std::complex<double>>& outer_coeffs() {return outer_coeffs_;}

    /// return a mutable vector of term coeficients c_mu
    std::vector<std::complex<double>>& inner_coeffs() {return inner_coeffs_;}

    // Scale-inplace index maps
    std::vector<thrust::device_vector<int>>& terms_scale_indsa_dag_gpu() { return terms_scale_indsa_dag_gpu_; }
    std::vector<thrust::device_vector<int>>& terms_scale_indsa_undag_gpu() { return terms_scale_indsa_undag_gpu_; }
    std::vector<thrust::device_vector<int>>& terms_scale_indsb_dag_gpu() { return terms_scale_indsb_dag_gpu_; }
    std::vector<thrust::device_vector<int>>& terms_scale_indsb_undag_gpu() { return terms_scale_indsb_undag_gpu_; }

    // Accumulation (source) index maps
    std::vector<thrust::device_vector<int>>& terms_sourcea_dag_gpu() { return terms_sourcea_dag_gpu_; }
    std::vector<thrust::device_vector<int>>& terms_sourcea_undag_gpu() { return terms_sourcea_undag_gpu_; }
    std::vector<thrust::device_vector<int>>& terms_sourceb_dag_gpu() { return terms_sourceb_dag_gpu_; }
    std::vector<thrust::device_vector<int>>& terms_sourceb_undag_gpu() { return terms_sourceb_undag_gpu_; }

    // Accumulation (target) index maps
    std::vector<thrust::device_vector<int>>& terms_targeta_dag_gpu() { return terms_targeta_dag_gpu_; }
    std::vector<thrust::device_vector<int>>& terms_targeta_undag_gpu() { return terms_targeta_undag_gpu_; }
    std::vector<thrust::device_vector<int>>& terms_targetb_dag_gpu() { return terms_targetb_dag_gpu_; }
    std::vector<thrust::device_vector<int>>& terms_targetb_undag_gpu() { return terms_targetb_undag_gpu_; }

    // Parity/phase maps
    std::vector<thrust::device_vector<cuDoubleComplex>>& terms_paritya_dag_gpu() { return terms_paritya_dag_gpu_; }
    std::vector<thrust::device_vector<cuDoubleComplex>>& terms_paritya_undag_gpu() { return terms_paritya_undag_gpu_; }
    std::vector<thrust::device_vector<cuDoubleComplex>>& terms_parityb_dag_gpu() { return terms_parityb_dag_gpu_; }
    std::vector<thrust::device_vector<cuDoubleComplex>>& terms_parityb_undag_gpu() { return terms_parityb_undag_gpu_; }

    // Read-only tuple view of the mu-th entries:
    // (inner_coeffs_[mu], outer_coeffs_[mu],
    //  terms_scale_indsa_dag_gpu_[mu],    terms_scale_indsa_undag_gpu_[mu],
    //  terms_scale_indsb_dag_gpu_[mu],    terms_scale_indsb_undag_gpu_[mu],
    //  terms_sourcea_dag_gpu_[mu],        terms_sourcea_undag_gpu_[mu],
    //  terms_sourceb_dag_gpu_[mu],        terms_sourceb_undag_gpu_[mu],
    //  terms_targeta_dag_gpu_[mu],        terms_targeta_undag_gpu_[mu],
    //  terms_targetb_dag_gpu_[mu],        terms_targetb_undag_gpu_[mu],
    //  terms_paritya_dag_gpu_[mu],        terms_paritya_undag_gpu_[mu],
    //  terms_parityb_dag_gpu_[mu],        terms_parityb_undag_gpu_[mu])
    std::tuple<
        const std::complex<double>&, const std::complex<double>&,
        const thrust::device_vector<int>&, const thrust::device_vector<int>&,
        const thrust::device_vector<int>&, const thrust::device_vector<int>&,
        const thrust::device_vector<int>&, const thrust::device_vector<int>&,
        const thrust::device_vector<int>&, const thrust::device_vector<int>&,
        const thrust::device_vector<int>&, const thrust::device_vector<int>&,
        const thrust::device_vector<int>&, const thrust::device_vector<int>&,
        const thrust::device_vector<cuDoubleComplex>&, const thrust::device_vector<cuDoubleComplex>&,
        const thrust::device_vector<cuDoubleComplex>&, const thrust::device_vector<cuDoubleComplex>&
    >
    get_mu_tuple(size_t mu) const
    {
        return std::tuple<
            const std::complex<double>&, const std::complex<double>&,
            const thrust::device_vector<int>&, const thrust::device_vector<int>&,
            const thrust::device_vector<int>&, const thrust::device_vector<int>&,
            const thrust::device_vector<int>&, const thrust::device_vector<int>&,
            const thrust::device_vector<int>&, const thrust::device_vector<int>&,
            const thrust::device_vector<int>&, const thrust::device_vector<int>&,
            const thrust::device_vector<int>&, const thrust::device_vector<int>&,
            const thrust::device_vector<cuDoubleComplex>&, const thrust::device_vector<cuDoubleComplex>&,
            const thrust::device_vector<cuDoubleComplex>&, const thrust::device_vector<cuDoubleComplex>&
        >(
            inner_coeffs_[mu],                // 0
            outer_coeffs_[mu],                // 1
            terms_scale_indsa_dag_gpu_[mu],   // 2 
            terms_scale_indsa_undag_gpu_[mu], // 3
            terms_scale_indsb_dag_gpu_[mu],   // 4 
            terms_scale_indsb_undag_gpu_[mu], // 5
            terms_sourcea_dag_gpu_[mu],       // 6
            terms_sourcea_undag_gpu_[mu],     // 7
            terms_sourceb_dag_gpu_[mu],       // 8
            terms_sourceb_undag_gpu_[mu],     // 9
            terms_targeta_dag_gpu_[mu],       // 10
            terms_targeta_undag_gpu_[mu],     // 11
            terms_targetb_dag_gpu_[mu],       // 12
            terms_targetb_undag_gpu_[mu],     // 13
            terms_paritya_dag_gpu_[mu],       // 14 
            terms_paritya_undag_gpu_[mu],     // 15
            terms_parityb_dag_gpu_[mu],       // 16
            terms_parityb_undag_gpu_[mu]      // 17
        );
    }

    // Verifies that inner_coeffs_, outer_coeffs_, and all 16 term arrays
    // have the same length (same number of μ-terms). Returns the common
    // size on success; returns 0 and prints a report on mismatch.
    std::size_t check_mu_tuple_container_sizes() const;

    // Print sizes of all arrays referenced by get_mu_tuple(mu)
    void print_mu_tuple_dims(std::size_t mu) const;

    // Print actual elements of all arrays referenced by get_mu_tuple(mu)
    void print_mu_tuple_elements(std::size_t mu) const;


    /// set the total number of occupied and virtual spatial orbitals from a reference, from the number
    ///     of occupied spin orbitals of each point group symmetry
    void set_orb_spaces(const std::vector<int>& ref);

    /// onous on caller to pass a sq_op that is actually hermitain, should use a hermitian check funciton...
    /// for an operator, splits the operator into hermitan pairs where each pair becomes a term
    /// in the pool vector
    void add_hermitian_pairs(std::complex<double> coeff, const SQOperator& sq_op );

    /// returns a QubitOpPool object with one term for each term in terms_
    QubitOpPool get_qubit_op_pool();

    /// returns a single QubitOperator of the JW transformed sq ops
    QubitOperator get_qubit_operator(const std::string& order_type, bool combine_like_terms=true, bool qubit_excitations=false);

    /// builds the sq operator pool
    void fill_pool(std::string pool_type);

    /// return a vector of string representing this sq operator pool
    std::string str() const;

  private:
    /// the number of occupied spatial orbitals
    int nocc_;

    /// the number of virtual spatial orbitals
    int nvir_;

    /// the list of sq operators in the pool
    std::vector<std::pair<std::complex<double>, SQOperator>> terms_;


    /// ===> Below are Objects used only for Trotterized Time Evolution or dUCC <===

    bool device_vecs_populated_ = false;

    /// the list of just the outer coefficients h_mu in h_mu( c_mu g_mu - c_mu g_mu^) (or similar hermitain case)
    /// often just a list of ones in time evolution
    std::vector<std::complex<double>> outer_coeffs_;

    /// the list of just the inner coefficients c_mu in h_mu( c_mu g_mu - c_mu g_mu^) (or similar hermitain case)
    /// c_mu assumed to be the same for both terms
    std::vector<std::complex<double>> inner_coeffs_;


    /// ===> For Sclae inplace

    /// the list of alfa/beta indicies for inplace ops for FCIComputerGPU
    std::vector<thrust::device_vector<int>> terms_scale_indsa_dag_gpu_; 
    std::vector<thrust::device_vector<int>> terms_scale_indsa_undag_gpu_; //note: not used in easy case
    std::vector<thrust::device_vector<int>> terms_scale_indsb_dag_gpu_; 
    std::vector<thrust::device_vector<int>> terms_scale_indsb_undag_gpu_; //note: not used in easy case


    /// ===> For accumulation (gather->scale->transfer)

    /// the list of alfa/beta source indicies for FCIComputerGPU
    std::vector<thrust::device_vector<int>> terms_sourcea_dag_gpu_;
    std::vector<thrust::device_vector<int>> terms_sourcea_undag_gpu_;
    std::vector<thrust::device_vector<int>> terms_sourceb_dag_gpu_;
    std::vector<thrust::device_vector<int>> terms_sourceb_undag_gpu_; 

    /// the list of alfa/beta target indicies for FCIComputerGPU
    std::vector<thrust::device_vector<int>> terms_targeta_dag_gpu_;
    std::vector<thrust::device_vector<int>> terms_targeta_undag_gpu_;
    std::vector<thrust::device_vector<int>> terms_targetb_dag_gpu_;
    std::vector<thrust::device_vector<int>> terms_targetb_undag_gpu_;
    
    /// the list of alfa/beta parities for FCIComputerGPU
    std::vector<thrust::device_vector<cuDoubleComplex>> terms_paritya_dag_gpu_;
    std::vector<thrust::device_vector<cuDoubleComplex>> terms_paritya_undag_gpu_;
    std::vector<thrust::device_vector<cuDoubleComplex>> terms_parityb_dag_gpu_;
    std::vector<thrust::device_vector<cuDoubleComplex>> terms_parityb_undag_gpu_;



};

#endif // _sq_op_pool_thrust_h_
