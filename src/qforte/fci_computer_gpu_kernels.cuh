#pragma once
#include <complex>
#include <cuComplex.h>

// ==============================================
// Original Implementation:
// Only keeping to support non - precomp version
// Should either be updated to use Givens or removed
// ==============================================

__device__ double atomicAdd_double(double* address, double val);

__global__ void apply_individual_nbody1_accumulate_kernel_atomic(
    const cuDoubleComplex coeff, 
    const cuDoubleComplex* d_Cin, 
    cuDoubleComplex* d_Cout, 
    const int* d_sourcea,
    const int* d_targeta,
    const cuDoubleComplex* d_paritya,
    const int* d_sourceb,
    const int* d_targetb,
    const cuDoubleComplex* d_parityb,
    int nbeta_strs_,
    int targeta_size,
    int targetb_size,
    int tensor_size);

extern "C" void apply_individual_nbody1_accumulate_wrapper(
    const cuDoubleComplex coeff, 
    const cuDoubleComplex* d_Cin, 
    cuDoubleComplex* d_Cout, 
    const int* d_sourcea,
    const int* d_targeta,
    const cuDoubleComplex* d_paritya,
    const int* d_sourceb,
    const int* d_targetb,
    const cuDoubleComplex* d_parityb,
    int nbeta_strs_,
    int targeta_size,
    int targetb_size,
    int tensor_size);

// ==============================================
// Fused apply+dot kernel and wrapper (Complex)
// Computes <sigma | K | psi> as a scalar reduction.
// Neither d_psi nor d_sigma is modified.
// d_accum must be a device pointer to a zeroed cuDoubleComplex.
// ==============================================

__global__ void dot_individual_nbody1_kernel(
    cuDoubleComplex coeff,
    const cuDoubleComplex* d_psi,
    const cuDoubleComplex* d_sigma,
    const int* d_sourcea,
    const int* d_targeta,
    const cuDoubleComplex* d_paritya,
    const int* d_sourceb,
    const int* d_targetb,
    const cuDoubleComplex* d_parityb,
    int nbeta_strs_,
    int targeta_size,
    int targetb_size,
    cuDoubleComplex* d_accum);

extern "C" void dot_individual_nbody1_wrapper(
    cuDoubleComplex coeff,
    const cuDoubleComplex* d_psi,
    const cuDoubleComplex* d_sigma,
    const int* d_sourcea,
    const int* d_targeta,
    const cuDoubleComplex* d_paritya,
    const int* d_sourceb,
    const int* d_targetb,
    const cuDoubleComplex* d_parityb,
    int nbeta_strs_,
    int targeta_size,
    int targetb_size,
    cuDoubleComplex* d_accum);

// ==============================================
// Scale elements kernel and wrapper (Complex)
// ==============================================

__global__ void scale_elements_kernel(
    cuDoubleComplex* d_Cout,
    const int* d_first, 
    int first_size,
    const int* d_second, 
    int second_size,
    int nbeta_strs_,
    cuDoubleComplex factor);

extern "C" void scale_elements_wrapper_complex(
    cuDoubleComplex* d_Cout,
    const int* d_first, 
    int first_size,
    const int* d_second, 
    int second_size,
    int nbeta_strs_,
    cuDoubleComplex factor);

// ==============================================
// Scale elements kernel and wrapper (Real)
// ==============================================

__global__ void scale_elements_kernel_real(
    double* __restrict__ d_Cout,
    const int* __restrict__ d_first,
    int first_size,
    const int* __restrict__ d_second,
    int second_size,
    int nbeta_strs_,
    double factor);

extern "C" void scale_elements_wrapper_real(
    double* d_Cout,
    const int* d_first,
    int first_size,
    const int* d_second,
    int second_size,
    int nbeta_strs_,
    double factor);

// ==============================================
// In-place Givens update kernels and wrappers (Complex)
// ==============================================

__global__ void inplace_givens_update_rows_kernel(
    cuDoubleComplex* __restrict__ d_Cout,
    const int* __restrict__ sourcea1,
    const int* __restrict__ targeta1,
    const cuDoubleComplex* __restrict__ paritya1,
    const cuDoubleComplex* __restrict__ paritya2,
    int na,
    int nbeta_strs_,
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2);

extern "C" void inplace_givens_update_complex_rows_wrapper(
    cuDoubleComplex* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const cuDoubleComplex* paritya1,
    const cuDoubleComplex* paritya2,
    int na,
    int nbeta_strs_,
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2);

__global__ void inplace_givens_update_cols_kernel(
    cuDoubleComplex* __restrict__ d_Cout,
    const int* __restrict__ sourcea1,
    const int* __restrict__ targeta1,
    const cuDoubleComplex* __restrict__ paritya1,
    const cuDoubleComplex* __restrict__ paritya2,
    const int* __restrict__ sourceb1,
    const int* __restrict__ targetb1,
    const cuDoubleComplex* __restrict__ parityb1,
    const cuDoubleComplex* __restrict__ parityb2,
    int nalpha, 
    int nb,
    int nbeta_strs_,
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2);

__global__ void inplace_givens_update_complex_tiled(
    cuDoubleComplex* __restrict__ d_Cout,
    const int* __restrict__ sourcea1,
    const int* __restrict__ targeta1,
    const cuDoubleComplex* __restrict__ paritya1,
    const cuDoubleComplex* __restrict__ paritya2,
    const int* __restrict__ sourceb1,
    const int* __restrict__ targetb1,
    const cuDoubleComplex* __restrict__ parityb1,
    const cuDoubleComplex* __restrict__ parityb2,
    int nalpha,          // rows
    int nb,              // number of column-pairs
    int nbeta_strs_,
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2);

extern "C" void inplace_givens_update_complex_tiled_wrapper(
    int BX_runtime,                      // pick 32 or 64 (must divide warp multiples)
    cuDoubleComplex* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const cuDoubleComplex* paritya1,
    const cuDoubleComplex* paritya2,
    const int* sourceb1,
    const int* targetb1,
    const cuDoubleComplex* parityb1,
    const cuDoubleComplex* parityb2,
    int nalpha,          // rows
    int nb,              // number of column-pairs
    int nbeta_strs_,     // leading dimension (num columns)
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2);

// ==============================================
// In-place Givens update kernels and wrappers (Real)
// ==============================================

__global__ void inplace_givens_update_rows_kernel_real(
    double* __restrict__ d_Cout,
    const int* __restrict__ sourcea1,      // [na]
    const int* __restrict__ targeta1,      // [na]
    const double* __restrict__ paritya1,   // [na]  (g† leg, row)
    const double* __restrict__ paritya2,   // [na]  (g  leg, row)
    int na,
    int nbeta_strs_,                        // number of columns
    double factor,
    double acc_coeff1,
    double acc_coeff2);

extern "C" void inplace_givens_update_real_rows_wrapper(
    double* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const double* paritya1,
    const double* paritya2,
    int na,
    int nbeta_strs_,
    double factor,
    double acc_coeff1,
    double acc_coeff2);

template <int ROWS_PER_THREAD>
__global__ void inplace_givens_update_cols_kernel_real(
    double* __restrict__ d_Cout,
    const int* __restrict__ sourceb1,      // [nb]
    const int* __restrict__ targetb1,      // [nb]
    const double* __restrict__ parityb1,   // [nb]  (g† leg, col)
    const double* __restrict__ parityb2,   // [nb]  (g  leg, col)
    int nb,
    long long nalpha_strs_,                // number of rows
    long long nbeta_strs_,                 // number of columns
    double factor,
    double acc_coeff1,
    double acc_coeff2);

extern "C" void inplace_givens_update_real_cols_wrapper(
    double* d_Cout,
    const int* sourceb1,
    const int* targetb1,
    const double* parityb1,
    const double* parityb2,
    int nb,
    long long nalpha_strs_,
    long long nbeta_strs_,
    double factor,
    double acc_coeff1,
    double acc_coeff2);

// ==============================================
// Beta-only row-major tiled Givens kernel (Real)
// Keeps warps row-local: threadIdx.x walks beta pairs, threadIdx.y walks rows.
// This gives coalesced (unit-stride) loads/stores within each warp and produces
// more blocks than the column kernel when nb is small.
// ==============================================

template<int BX, int AY>
__global__ void inplace_givens_update_beta_only_rowmajor_real(
    double* __restrict__ d_Cout,
    const int* __restrict__ sourceb1,      // [nb]  beta source indices
    const int* __restrict__ targetb1,      // [nb]  beta target indices
    const double* __restrict__ parityb1,   // [nb]  parity for source row (g† leg)
    const double* __restrict__ parityb2,   // [nb]  parity for target row (g  leg)
    int nb,                                // number of beta pairs
    int nalpha_strs_,                      // number of alpha strings (rows)
    int nbeta_strs_,                       // leading dimension (columns)
    double factor,
    double acc_coeff1,
    double acc_coeff2);

// Internal template launcher for beta-only row-major kernel
template<int BX, int AY>
static void launch_inplace_givens_update_beta_only_rowmajor_real(
    double* d_Cout,
    const int* sourceb1,
    const int* targetb1,
    const double* parityb1,
    const double* parityb2,
    int nb,
    int nalpha_strs_,
    int nbeta_strs_,
    double factor,
    double acc_coeff1,
    double acc_coeff2);

extern "C" void inplace_givens_update_real_beta_only_rowmajor_wrapper(
    double* d_Cout,
    const int* sourceb1,
    const int* targetb1,
    const double* parityb1,
    const double* parityb2,
    int nb,
    int nalpha_strs_,
    int nbeta_strs_,
    double factor,
    double acc_coeff1,
    double acc_coeff2);

__global__ void inplace_givens_update_real_tiled(
    double* __restrict__ d_Cout,
    const int* __restrict__ sourcea1,
    const int* __restrict__ targeta1,
    const double* __restrict__ paritya1,
    const double* __restrict__ paritya2,
    const int* __restrict__ sourceb1,
    const int* __restrict__ targetb1,
    const double* __restrict__ parityb1,
    const double* __restrict__ parityb2,
    int nalpha,          // rows
    int nb,              // number of column-pairs
    int nbeta_strs_,     // leading dimension (num columns)
    double factor,
    double acc_coeff1,
    double acc_coeff2);

static void launch_inplace_givens_update_real_tiled(
    double* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const double* paritya1,
    const double* paritya2,
    const int* sourceb1,
    const int* targetb1,
    const double* parityb1,
    const double* parityb2,
    int nalpha,
    int nb,
    int nbeta_strs_,
    double factor,
    double acc_coeff1,
    double acc_coeff2);

extern "C" void inplace_givens_update_real_tiled_wrapper(
    int BX_runtime,                      // pick 32 or 64 (must divide warp multiples)
    double* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const double* paritya1,
    const double* paritya2,
    const int* sourceb1,
    const int* targetb1,
    const double* parityb1,
    const double* parityb2,
    int nalpha,          // rows
    int nb,              // number of column-pairs
    int nbeta_strs_,     // leading dimension (num columns)
    double factor,
    double acc_coeff1,
    double acc_coeff2);

// ==============================================
// Same Spin Implementation with cuSPARSE
// ==============================================

extern "C" void lm_apply_array12_same_spin_spmm_csr_coalesced_wrapper(
    cuDoubleComplex* d_out,
    const cuDoubleComplex* d_C,
    const int* d_dexc,
    const cuDoubleComplex* d_h1e,
    const cuDoubleComplex* d_h2e,
    int states1,
    int states2,
    int ndexc,
    int norbs,
    int inc1,
    int inc2);

extern "C" void lm_apply_array12_same_spin_spmm_csr_coalesced_wrapper_real(
    double* d_out,
    const double* d_C,
    const int* d_dexc,
    const double* d_h1e,
    const double* d_h2e,
    int states1,
    int states2,
    int ndexc,
    int norbs,
    int inc1,
    int inc2);

extern "C" void lm_apply_array12_same_spin_spmm_csr_coalesced_wrapper_mixed(
    cuDoubleComplex* d_out,
    const cuDoubleComplex* d_C,
    const int* d_dexc,
    const double* d_h1e,
    const double* d_h2e,
    int states1,
    int states2,
    int ndexc,
    int norbs,
    int inc1,
    int inc2);

// ===============================================
// Diff Spin Implementation
// ===============================================

extern "C" void lm_apply_array12_diff_spin_wrapper(
    cuDoubleComplex* d_out,
    const cuDoubleComplex* d_C,
    const int* d_adexc,
    const int* d_bdexc,
    const cuDoubleComplex* d_h2e,
    int alpha_states,
    int beta_states,
    int nadexc,
    int nbdexc,
    int norbs);

extern "C" void lm_apply_array12_diff_spin_wrapper_real(
    double* d_out,
    const double* d_C,
    const int* d_adexc,
    const int* d_bdexc,
    const double* d_h2e,
    int alpha_states,
    int beta_states,
    int nadexc,
    int nbdexc,
    int norbs);

extern "C" void lm_apply_array12_diff_spin_wrapper_mixed(
    cuDoubleComplex* d_out,
    const cuDoubleComplex* d_C,
    const int* d_adexc,
    const int* d_bdexc,
    const double* d_h2e,
    int alpha_states,
    int beta_states,
    int nadexc,
    int nbdexc,
    int norbs);