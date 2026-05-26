#ifndef _sq_op_pool_gpu_h_
#define _sq_op_pool_gpu_h_

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

/// Classification of pool operator terms for specialized dot kernel dispatch.
enum class DotKernelKind : int {
    AlphaOnly = 0,   ///< Only alpha spin excitations (beta=identity)
    BetaOnly  = 1,   ///< Only beta spin excitations (alpha=identity)
    Mixed     = 2,   ///< Both alpha and beta excitations
    Easy      = 3    ///< Number-operator-like (no excitation, only scaling)
};

class SQOperator;
class QubitOperator;
class QubitOpPool;

// Represents an arbitrary linear combination of second quantized operators.
// May also represent an array of second quantized operators by ignoring
// the coefficients.
class SQOpPoolGPU {
  public:
    /// default constructor: creates an empty second quantized operator pool
    SQOpPoolGPU() {}

    /// construct with a data_type ("complex" or "real"). Defaults to "complex".
    explicit SQOpPoolGPU(const std::string& data_type)
        : data_type_(data_type) {
        validate_data_type_();
    }

    /// destructor: safely cleans up device vectors
    ~SQOpPoolGPU();

    /// add one set of annihilators and/or creators to the second quantized operator pool
    void add_term(std::complex<double> coeff, const SQOperator& sq_op );

    /// sets the operator pool coefficients
    void set_coeffs(const std::vector<std::complex<double>>& new_coeffs);

    /// updates the operator pool terms & outer coefficients with new_coeffs
    void update_evolution_coeffs(const std::vector<std::complex<double>>& new_coeffs);

    /// sets the operator pool coefficients
    void set_coeffs_to_scaler(std::complex<double> new_coeff);

    /// return a vector of terms and their coeficients
    const std::vector<std::pair< std::complex<double>, SQOperator>>& terms() const;

    /// ===> Getters/Setters for GPU term maps (mutable) <===

    /// return a bool indicating whether the device stp arrays have been populated
    bool device_vecs_populated() const { return device_vecs_populated_; } 

    /// set whether the divice vecs have already been populated or not
    void set_device_vecs_populated(bool val) { device_vecs_populated_ = val; }

    /// data type accessor
    const std::string& data_type() const { return data_type_; }

    /// set data type (validates value)
    void set_data_type(const std::string& dt) {
        data_type_ = dt;
        validate_data_type_();
    }

    /// return a mutable vector of term coeficients h_mu
    std::vector<std::complex<double>>& outer_coeffs() {return outer_coeffs_;}

    /// return a mutable vector of term coeficients c_mu
    std::vector<std::complex<double>>& inner_coeffs() {return inner_coeffs_;}

    // Scale-inplace index maps
    std::vector<thrust::device_vector<int>>& terms_scale_indsa_dag_gpu() { return terms_scale_indsa_dag_gpu_; }
    const std::vector<thrust::device_vector<int>>& terms_scale_indsa_dag_gpu() const { return terms_scale_indsa_dag_gpu_; }
    std::vector<thrust::device_vector<int>>& terms_scale_indsa_undag_gpu() { return terms_scale_indsa_undag_gpu_; }
    const std::vector<thrust::device_vector<int>>& terms_scale_indsa_undag_gpu() const { return terms_scale_indsa_undag_gpu_; }
    std::vector<thrust::device_vector<int>>& terms_scale_indsb_dag_gpu() { return terms_scale_indsb_dag_gpu_; }
    const std::vector<thrust::device_vector<int>>& terms_scale_indsb_dag_gpu() const { return terms_scale_indsb_dag_gpu_; }
    std::vector<thrust::device_vector<int>>& terms_scale_indsb_undag_gpu() { return terms_scale_indsb_undag_gpu_; }
    const std::vector<thrust::device_vector<int>>& terms_scale_indsb_undag_gpu() const { return terms_scale_indsb_undag_gpu_; }

    // Parity/phase maps (complex)
    std::vector<thrust::device_vector<cuDoubleComplex>>& terms_paritya_dag_gpu() { return terms_paritya_dag_gpu_; }
    const std::vector<thrust::device_vector<cuDoubleComplex>>& terms_paritya_dag_gpu() const { return terms_paritya_dag_gpu_; }
    std::vector<thrust::device_vector<cuDoubleComplex>>& terms_paritya_undag_gpu() { return terms_paritya_undag_gpu_; }
    const std::vector<thrust::device_vector<cuDoubleComplex>>& terms_paritya_undag_gpu() const { return terms_paritya_undag_gpu_; }
    std::vector<thrust::device_vector<cuDoubleComplex>>& terms_parityb_dag_gpu() { return terms_parityb_dag_gpu_; }
    const std::vector<thrust::device_vector<cuDoubleComplex>>& terms_parityb_dag_gpu() const { return terms_parityb_dag_gpu_; }
    std::vector<thrust::device_vector<cuDoubleComplex>>& terms_parityb_undag_gpu() { return terms_parityb_undag_gpu_; }
    const std::vector<thrust::device_vector<cuDoubleComplex>>& terms_parityb_undag_gpu() const { return terms_parityb_undag_gpu_; }

    // Parity/phase maps (real)
    std::vector<thrust::device_vector<double>>& terms_paritya_dag_re_gpu() { return terms_paritya_dag_re_gpu_; }
    const std::vector<thrust::device_vector<double>>& terms_paritya_dag_re_gpu() const { return terms_paritya_dag_re_gpu_; }
    std::vector<thrust::device_vector<double>>& terms_paritya_undag_re_gpu() { return terms_paritya_undag_re_gpu_; }
    const std::vector<thrust::device_vector<double>>& terms_paritya_undag_re_gpu() const { return terms_paritya_undag_re_gpu_; }
    std::vector<thrust::device_vector<double>>& terms_parityb_dag_re_gpu() { return terms_parityb_dag_re_gpu_; }
    const std::vector<thrust::device_vector<double>>& terms_parityb_dag_re_gpu() const { return terms_parityb_dag_re_gpu_; }
    std::vector<thrust::device_vector<double>>& terms_parityb_undag_re_gpu() { return terms_parityb_undag_re_gpu_; }
    const std::vector<thrust::device_vector<double>>& terms_parityb_undag_re_gpu() const { return terms_parityb_undag_re_gpu_; }
    // std::vector<thrust::device_vector<double>>& terms_parityb_dag_im_gpu() { return terms_parityb_dag_im_gpu_; }
    // std::vector<thrust::device_vector<double>>& terms_parityb_undag_im_gpu() { return terms_parityb_undag_im_gpu_; }
    // std::vector<thrust::device_vector<double>>& terms_paritya_undag_im_gpu() { return terms_paritya_undag_im_gpu_; }
    // std::vector<thrust::device_vector<double>>& terms_paritya_dag_im_gpu() { return terms_paritya_dag_im_gpu_; }

    // === Dot scalar coefficient accessors (parity_sort * inner_coeff, one per mu) ===
    /// Coefficient for term 0 (excitation/dag direction): parity_sort(crea+anna) * c_mu_t0
    std::vector<cuDoubleComplex>& dot_coeff_dag() { return dot_coeff_dag_; }
    const std::vector<cuDoubleComplex>& dot_coeff_dag() const { return dot_coeff_dag_; }
    /// Coefficient for term 1 (de-excitation/undag direction): parity_sort(anna+crea) * c_mu_t1
    std::vector<cuDoubleComplex>& dot_coeff_undag() { return dot_coeff_undag_; }
    const std::vector<cuDoubleComplex>& dot_coeff_undag() const { return dot_coeff_undag_; }
    /// Real-path equivalents
    std::vector<double>& dot_coeff_dag_re() { return dot_coeff_dag_re_; }
    const std::vector<double>& dot_coeff_dag_re() const { return dot_coeff_dag_re_; }
    std::vector<double>& dot_coeff_undag_re() { return dot_coeff_undag_re_; }
    const std::vector<double>& dot_coeff_undag_re() const { return dot_coeff_undag_re_; }

    /// Per-operator kernel classification for specialized dot dispatch
    std::vector<DotKernelKind>& dot_kernel_kinds() { return dot_kernel_kinds_; }
    const std::vector<DotKernelKind>& dot_kernel_kinds() const { return dot_kernel_kinds_; }

    // Read-only tuple view of the mu-th entries:
    // (inner_coeffs_[mu], outer_coeffs_[mu],
    //  terms_scale_indsa_dag_gpu_[mu],    terms_scale_indsa_undag_gpu_[mu],
    //  terms_scale_indsb_dag_gpu_[mu],    terms_scale_indsb_undag_gpu_[mu],
    //  terms_paritya_dag_gpu_[mu],        terms_paritya_undag_gpu_[mu],
    //  terms_parityb_dag_gpu_[mu],        terms_parityb_undag_gpu_[mu])
    std::tuple<
        const std::complex<double>&, const std::complex<double>&,
        const thrust::device_vector<int>&, // fwd alpha: (sourcea_dag, targeta_undag, terms_scale_indsa_dag)
        const thrust::device_vector<int>&, // bwd alpha: (sourcea_undag, targeta_dag, terms_scale_indsa_undag)
        const thrust::device_vector<int>&, // fwd beta: (sourceb_dag, targetb_undag, terms_scale_indsb_dag)
        const thrust::device_vector<int>&, // bwd beta: (sourceb_undag, targetb_dag, terms_scale_indsb_undag)
        const thrust::device_vector<cuDoubleComplex>&, const thrust::device_vector<cuDoubleComplex>&,
        const thrust::device_vector<cuDoubleComplex>&, const thrust::device_vector<cuDoubleComplex>&
    >
    get_mu_tuple(size_t mu) const
    {
        return std::tuple<
            const std::complex<double>&, const std::complex<double>&,
            const thrust::device_vector<int>&, 
            const thrust::device_vector<int>&,
            const thrust::device_vector<int>&, 
            const thrust::device_vector<int>&,
            const thrust::device_vector<cuDoubleComplex>&, const thrust::device_vector<cuDoubleComplex>&,
            const thrust::device_vector<cuDoubleComplex>&, const thrust::device_vector<cuDoubleComplex>&
        >(
            inner_coeffs_[mu],                // 0
            outer_coeffs_[mu],                // 1
            terms_scale_indsa_dag_gpu_[mu],   // 2 
            terms_scale_indsa_undag_gpu_[mu], // 3
            terms_scale_indsb_dag_gpu_[mu],   // 4 
            terms_scale_indsb_undag_gpu_[mu], // 5
            terms_paritya_dag_gpu_[mu],       // 6
            terms_paritya_undag_gpu_[mu],     // 7
            terms_parityb_dag_gpu_[mu],       // 8
            terms_parityb_undag_gpu_[mu]      // 9
        );
    }

    // Optional read-only tuple for real-only parity storage (no imaginary parts)
    // Tuple indices:
    //  0: inner_coeffs_[mu]
    //  1: outer_coeffs_[mu]
    //  2-5: the 12 integer index arrays (scale/source/target)
    // 14: terms_paritya_dag_re_gpu_[mu]
    // 15: terms_paritya_undag_re_gpu_[mu]
    // 16: terms_parityb_dag_re_gpu_[mu]
    // 17: terms_parityb_undag_re_gpu_[mu]
    std::tuple<
        const std::complex<double>&, const std::complex<double>&,
        const thrust::device_vector<int>&, // fwd alpha: (sourcea_dag, targeta_undag, terms_scale_indsa_dag)
        const thrust::device_vector<int>&, // bwd alpha: (sourcea_undag, targeta_dag, terms_scale_indsa_undag)
        const thrust::device_vector<int>&, // fwd beta: (sourceb_dag, targetb_undag, terms_scale_indsb_dag)
        const thrust::device_vector<int>&, // bwd beta: (sourceb_undag, targetb_dag, terms_scale_indsb_undag)
        const thrust::device_vector<double>&, const thrust::device_vector<double>&,
        const thrust::device_vector<double>&, const thrust::device_vector<double>&
    >
    get_mu_tuple_real(size_t mu) const {
        return std::tuple<
            const std::complex<double>&, const std::complex<double>&,
            const thrust::device_vector<int>&, 
            const thrust::device_vector<int>&,
            const thrust::device_vector<int>&, 
            const thrust::device_vector<int>&,
            const thrust::device_vector<double>&, const thrust::device_vector<double>&,
            const thrust::device_vector<double>&, const thrust::device_vector<double>&
        >(
            inner_coeffs_[mu],                // 0
            outer_coeffs_[mu],                // 1
            terms_scale_indsa_dag_gpu_[mu],   // 2
            terms_scale_indsa_undag_gpu_[mu], // 3
            terms_scale_indsb_dag_gpu_[mu],   // 4
            terms_scale_indsb_undag_gpu_[mu], // 5
            terms_paritya_dag_re_gpu_[mu],    // 6 (real parity a dag)
            terms_paritya_undag_re_gpu_[mu],  // 7 (real parity a undag)
            terms_parityb_dag_re_gpu_[mu],    // 8 (real parity b dag)
            terms_parityb_undag_re_gpu_[mu]   // 9 (real parity b undag)
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

    /// builds the sq operator pool kUpG style
    void fill_pool_kUpCCGSD(int kmax);

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


    /// ===> For Scale inplace

    /// the list of alfa/beta indicies for inplace ops for FCIComputerGPU
    std::vector<thrust::device_vector<int>> terms_scale_indsa_dag_gpu_; 
    std::vector<thrust::device_vector<int>> terms_scale_indsa_undag_gpu_; //note: not used in easy case
    std::vector<thrust::device_vector<int>> terms_scale_indsb_dag_gpu_; 
    std::vector<thrust::device_vector<int>> terms_scale_indsb_undag_gpu_; //note: not used in easy case
    
    /// the list of alfa/beta parities for FCIComputerGPU (complex)
    std::vector<thrust::device_vector<cuDoubleComplex>> terms_paritya_dag_gpu_;
    std::vector<thrust::device_vector<cuDoubleComplex>> terms_paritya_undag_gpu_;
    std::vector<thrust::device_vector<cuDoubleComplex>> terms_parityb_dag_gpu_;
    std::vector<thrust::device_vector<cuDoubleComplex>> terms_parityb_undag_gpu_;

    /// the list of alfa/beta parities in real form
    std::vector<thrust::device_vector<double>> terms_paritya_dag_re_gpu_;
    std::vector<thrust::device_vector<double>> terms_paritya_undag_re_gpu_;
    std::vector<thrust::device_vector<double>> terms_parityb_dag_re_gpu_;
    std::vector<thrust::device_vector<double>> terms_parityb_undag_re_gpu_;

    /// ===> Scalar coefficients for dot_sqop_from_pool_gpu (parity_sort * inner_coeff) <===
    /// These are the only genuinely new data for the dot path.
    /// All source/target/parity arrays are reused from the evolution precomputed data above:
    ///   dag  term (excitation):   source = terms_scale_inds*_undag, target = terms_scale_inds*_dag,
    ///                             parity = terms_parity*_undag (built with make_mapping_each_pre(crea,anna))
    ///   undag term (de-excit.):   source = terms_scale_inds*_dag,   target = terms_scale_inds*_undag,
    ///                             parity = terms_parity*_dag  (built with make_mapping_each_pre(anna,crea))
    std::vector<cuDoubleComplex> dot_coeff_dag_;    ///< parity_sort(crea+anna)*c_mu for term 0
    std::vector<cuDoubleComplex> dot_coeff_undag_;  ///< parity_sort(anna+crea)*c_mu for term 1
    std::vector<double> dot_coeff_dag_re_;          ///< real-path equivalent
    std::vector<double> dot_coeff_undag_re_;        ///< real-path equivalent

    /// Per-operator kernel classification for specialized dot dispatch
    std::vector<DotKernelKind> dot_kernel_kinds_;

    /// How to store parity data for kernels using this pool: "complex" or "real"
    std::string data_type_ = "complex";

    /// ensure data_type_ is valid
    void validate_data_type_() const {
        if (data_type_ != "complex" && data_type_ != "real") {
            throw std::invalid_argument("SQOpPoolGPU: unsupported data_type. Must be one of {complex, real}.");
        }
    }
};

#endif // _sq_op_pool_gpu_h_
