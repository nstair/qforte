#pragma once
#ifndef _fci_computer_thrust_h_
#define _fci_computer_thrust_h_

#include <string>
#include <vector>

#include "qforte-def.h" 
#include "tensor.h" 
#include "tensor_thrust.h"
#include "fci_graph.h"
#include "fci_graph_thrust.h"
#include "timer.h"
#include "sq_op_pool_thrust.h"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

class local_timer;
class Gate;
class QubitBasis;
class SQOperator;
class TensorOperator;
class TensorThrust;
class FCIGraph;
class FCIGraphThrust;
class SQOpPool;
class SQOpPoolThrust;

class FCIComputerThrust {
  public:
    /// default constructor: create a 'FCI' quantum computer 
    /// the computer represends a restricted hilbert space for 'chemistry'
    /// nel: the number of electrons
    /// sz: the z componant spin
    /// norb: the number of spatial orbitals 
    /// Implementation will be reminicient of modenrn determinant CI codes
    /// Implementation also borrows HEAVILY from the fermionic quantum emulator wfn class
    /// see (https://quantumai.google/openfermion/fqe) and related article
    FCIComputerThrust(int nel, int sz, int norb, bool on_gpu = false);

    /// apply a SQOperator to the current state.
    /// (this operation is generally not a physical quantum computing operation).
    /// Only works if the sqo conservs particle number and multiplicity.
    /// TODO(Tyler?): implement this...
    // void apply_sq_operator(const QubitOperator& qo);

    /// Set a particular element of this TensorThrust, specified by idxs
    void set_element(const std::vector<size_t>& idxs,
            const std::complex<double> val
            );

    void cpu_error() const;

    void gpu_error() const;

    void to_gpu();

    void to_cpu();

    /// Set a particular element of this TensorThrust, specified by idxs
    void add_to_element(const std::vector<size_t>& idxs,
            const std::complex<double> val
            );

    /// apply a TensorOperator to the current state 
    void apply_tensor_operator(const TensorOperator& top);

    /// apply a Tensor represending a 1-body spin-orbital indexed operator to the current state 
    void apply_tensor_spin_1bdy(
      const TensorThrust& h1e, 
      size_t norb);

    /// apply TensorThrusts represending 1-body and 2-body spin-orbital indexed operator to the current state 
    void apply_tensor_spin_12bdy(
      const TensorThrust& h1e, 
      const TensorThrust& h2e, 
      size_t norb);

    /// apply TensorThrusts represending 1-body and 2-body spatial-orbital indexed operator to the current state 
    void apply_tensor_spat_12bdy(
      const TensorThrust& h1e, 
      const TensorThrust& h2e, 
      const TensorThrust& h2e_einsum, 
      size_t norb);

    /// apply TensorThrusts represending 1-body and 2-body spatial-orbital indexed operator
    /// as well as a constant to the current state 
    void apply_tensor_spat_012bdy(
      const std::complex<double> h0e,
      const TensorThrust& h1e, 
      const TensorThrust& h2e, 
      const TensorThrust& h2e_einsum, 
      size_t norb);

    void lm_apply_array1(
      const TensorThrust& out,
      const std::vector<int> dexc,
      const int astates,
      const int bstates,
      const int ndexc,
      const TensorThrust& h1e,
      const int norbs,
      const bool is_alpha);

    void apply_array_1bdy_cpu(
      TensorThrust& out,
      const std::vector<int>& dexc,
      const int astates,
      const int bstates,
      const int ndexc,
      const TensorThrust& h1e,
      const int norbs,
      const bool is_alpha);

    void lm_apply_array12_same_spin_opt_cpu(
      TensorThrust& out,
      const std::vector<int>& dexc,
      const int alpha_states,
      const int beta_states,
      const int ndexc,
      const TensorThrust& h1e,
      const TensorThrust& h2e,
      const int norbs,
      const bool is_alpha); 

    void lm_apply_array12_diff_spin_opt_cpu(
      TensorThrust& out,
      const std::vector<int>& adexc,
      const std::vector<int>& bdexc,
      const int alpha_states,
      const int beta_states,
      const int nadexc,
      const int nbdexc,
      const TensorThrust& h2e,
      const int norbs); 

    std::pair<TensorThrust, TensorThrust> calculate_dvec_spin_with_coeff();

    TensorThrust calculate_coeff_spin_with_dvec_cpu(std::pair<TensorThrust, TensorThrust>& dvec);

    std::pair<std::vector<int>, std::vector<int>> evaluate_map_number_cpu(
      const std::vector<int>& numa,
      const std::vector<int>& numb); 

    std::pair<std::vector<int>, std::vector<int>> evaluate_map_cpu(
      const std::vector<int>& crea,
      const std::vector<int>& anna,
      const std::vector<int>& creb,
      const std::vector<int>& annb); 

    void apply_cos_inplace_cpu(
      const std::complex<double> time,
      const std::complex<double> coeff,
      const std::vector<int>& crea,
      const std::vector<int>& anna,
      const std::vector<int>& creb,
      const std::vector<int>& annb,
      TensorThrust& Cout);

    int isolate_number_operators_cpu(
      const std::vector<int>& cre,
      const std::vector<int>& ann,
      std::vector<int>& crework,
      std::vector<int>& annwork,
      std::vector<int>& number); 

    void evolve_individual_nbody_easy_cpu(
      const std::complex<double> time,
      const std::complex<double> coeff,
      TensorThrust& Cin,
      TensorThrust& Cout,
      const std::vector<int>& crea,
      const std::vector<int>& anna,
      const std::vector<int>& creb,
      const std::vector<int>& annb); 

    void evolve_individual_nbody_hard_cpu(
      const std::complex<double> time,
      const std::complex<double> coeff,
      TensorThrust& Cin,
      TensorThrust& Cout,
      const std::vector<int>& crea,
      const std::vector<int>& anna,
      const std::vector<int>& creb,
      const std::vector<int>& annb); 

    void evolve_individual_nbody_cpu(
      const std::complex<double> time,
      const SQOperator& sqop,
      TensorThrust& Cin,
      TensorThrust& Cout,
      const bool antiherm = false,
      const bool adjoint = false);

    void apply_sqop_evolution_gpu(
      const std::complex<double> time,
      const SQOperator& sqop,
      const bool antiherm = false,
      const bool adjoint = false);

    void evolve_pool_trotter_basic_gpu(
      const SQOpPool& pool,
      const bool antiherm = false,
      const bool adjoint = false);

    void evolve_pool_trotter_gpu(
      const SQOpPool& pool,
      const double evolution_time,
      const int trotter_steps,
      const int trotter_order,
      const bool antiherm = false,
      const bool adjoint = false);

    void evolve_pool_trotter_gpu_v2(
      const SQOpPool& pool,
      const double evolution_time,
      const int trotter_steps,
      const int trotter_order,
      const bool antiherm = false,
      const bool adjoint = false);

    void evolve_op_taylor_cpu(
      const SQOperator& op,
      const double evolution_time,
      const double convergence_thresh,
      const int max_taylor_iter);

    void apply_individual_nbody1_accumulate_gpu(
      const std::complex<double> coeff, 
      TensorThrust& Cin,
      TensorThrust& Cout,
      int counta,
      int countb);

    void apply_individual_nbody_accumulate_gpu(
      const std::complex<double> coeff,
      TensorThrust& Cin,
      TensorThrust& Cout,
      const std::vector<int>& daga,
      const std::vector<int>& undaga, 
      const std::vector<int>& dagb,
      const std::vector<int>& undagb);

    void apply_individual_sqop_term_gpu(
      const std::tuple< std::complex<double>, std::vector<size_t>, std::vector<size_t>>& term,
      TensorThrust& Cin,
      TensorThrust& Cout);

    void apply_sqop_gpu(const SQOperator& sqop);

    void apply_diagonal_of_sqop_cpu(
      const SQOperator& sq_op, 
      const bool invert_coeff = true);

    void apply_sqop_pool_cpu(const SQOpPool& sqop_pool);

    std::complex<double> get_exp_val_cpu(const SQOperator& sqop);

    std::complex<double> get_exp_val_tensor_cpu(
      const std::complex<double> h0e, 
      const TensorThrust& h1e, 
      const TensorThrust& h2e, 
      const TensorThrust& h2e_einsum, 
      size_t norb);  

    void scale_cpu(const std::complex<double> a);

    std::vector<double> direct_expectation_value(const TensorOperator& top);

    /// return a string representing the state of the computer
    /// TODO(Nick) Implement (this will be a pain :/)
    std::string str(
      bool print_data,
      bool print_complex
      ) 
    {
      return C_.str(print_data, print_complex); 
    }

    /// return a tensor of the coeficients
    TensorThrust get_state() const { return C_; }

    /// return a tensor of the coeficients
    TensorThrust get_state_deep() const { 
      TensorThrust Cprime = C_; 
      return Cprime; 
    }

    std::complex<double> coeff(const QubitBasis& abasis, const QubitBasis& bbasis);

    /// return the dot product of the current FCIComputerThrust state (as the ket) and the HF state (i.e. <HF|C_>)
    std::complex<double> get_hf_dot() const {
      cpu_error(); 
      return C_.get({0,0}); 
    }

    /// return the number of electrons
    size_t get_nel() const { return nel_; }

    /// return the z-componant spin
    size_t get_sz() const { return sz_; }
    
    /// return the number of spatial orbitals
    size_t none_ops() const { return norb_; }

    /// get timings
    std::vector<std::pair<std::string, double>> get_timings() { return timings_; }

    /// clear the timings
    void clear_timings() { timings_.clear(); }

    local_timer get_acc_timer() { return timer_; }

    void set_state_cpu(const TensorThrust& other_state);

    void set_state_gpu(const TensorThrust& other_state);

    void set_state_from_tensor_cpu(const Tensor& other_state);

    void zero_cpu();

    void hartree_fock_cpu();

    void print_vector(const std::vector<int>& vec, const std::string& name);

    void print_vector_thrust(const thrust::host_vector<int>& vec, const std::string& name);

    void print_vector_uint(const std::vector<uint64_t>& vec, const std::string& name);

    void print_vector_thrust_cuDoubleComplex(const thrust::host_vector<cuDoubleComplex>& vec, const std::string& name);


    /// ===> Helpers for populating device index/parity arrays for a particular SQOpPool
    
    /// uses the graph to populate the src/target/parity device vectors, keeps data on device for re-use
    void populate_index_arrays_for_pool_evo(SQOpPoolThrust& pool);

    /* Below are new methods for getting tensor data out of the computer */

    void copy_to_tensor_cpu(Tensor& tensor) const;
    void copy_to_tensor_thrust_gpu(TensorThrust& tensor) const;
    void copy_to_tensor_thrust_cpu(TensorThrust& tensor) const;

    const std::vector<size_t>& get_shape() const { return C_.shape(); }

  private:

    bool on_gpu_;
    size_t nel_;
    size_t nalfa_el_;
    size_t nbeta_el_;
    size_t nalfa_strs_;
    size_t nbeta_strs_;
    size_t sz_;
    size_t norb_;
    size_t nabasis_;
    size_t nbbasis_;
    const std::string name_ = "FCIComputerThrust State";
    TensorThrust C_;
    thrust::device_vector<int> sourcea_gpu_;
    thrust::device_vector<int> targeta_gpu_;
    thrust::device_vector<cuDoubleComplex> paritya_gpu_;
    thrust::device_vector<int> sourceb_gpu_;
    thrust::device_vector<int> targetb_gpu_;
    thrust::device_vector<cuDoubleComplex> parityb_gpu_;
    FCIGraphThrust graph_;

    local_timer timer_;
    std::vector<std::pair<std::string, double>> timings_;
    double compute_threshold_ = 1.0e-12;
};

#endif // _fci_computer_thrust_h_
