#pragma once
#include <complex>
#include <cuComplex.h>
// CUDA kernel for element-wise addition of two tensors
// __global__ void apply_individual_nbody1_accumulate_kernel(
//     const cuDoubleComplex coeff, 
//     const cuDoubleComplex* d_Cin, 
//     cuDoubleComplex* d_Cout, 
//     const int* d_sourcea,
//     const int* d_targeta,
//     const int* d_paritya,
//     const int* d_sourceb,
//     const int* d_targetb,
//     const int* d_parityb,
//     int nbeta_strs_,
//     int targeta_size,
//     int targetb_size,
//     int tensor_size);

__global__ void apply_individual_nbody1_accumulate_kernel(
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

extern "C" void apply_individual_nbody1_accumulate_wrapper_v2(
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

    /*
extern "C" void apply_individual_nbody1_accumulate_wrapper_shared(
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
    */

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

// === New in-place 2x2 Givens-like update kernel (hard n-body, v4) ===
__global__ void inplace_givens_update_kernel(
    cuDoubleComplex* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const cuDoubleComplex* paritya1,
    const cuDoubleComplex* paritya2,
    const int* sourceb1,
    const int* targetb1,
    const cuDoubleComplex* parityb1,
    const cuDoubleComplex* parityb2,
    int na,
    int nb,
    int nbeta_strs_,
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2);

extern "C" void inplace_givens_update_wrapper(
    cuDoubleComplex* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const cuDoubleComplex* paritya1,
    const cuDoubleComplex* paritya2,
    const int* sourceb1,
    const int* targetb1,
    const cuDoubleComplex* parityb1,
    const cuDoubleComplex* parityb2,
    int na,
    int nb,
    int nbeta_strs_,
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2);

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

template<int BX>
__global__ void givens_update_soa_tiled_factor_real(
    double* __restrict__ dCr, double* __restrict__ dCi, // C (in-place)
    const int* __restrict__ sourcea1, // row metadata
    const int* __restrict__ targeta1, // row metadata
    const double* __restrict__ pa1r, const double* __restrict__ pa1i,
    const double* __restrict__ pa2r, const double* __restrict__ pa2i,
    const int* __restrict__ sourceb1, // column metadata
    const int* __restrict__ targetb1, // column metadata
    const double* __restrict__ pb1r, const double* __restrict__ pb1i,
    const double* __restrict__ pb2r, const double* __restrict__ pb2i,
    int nalpha, int nb, int nbeta_strs_, // sizes
    double factor_real,                 // scalar: purely real
    double acc1r, double acc1i,         // scalar: complex
    double acc2r, double acc2i);        // scalar: complex

template<int BX>
__global__ void givens_update_soa_tiled_factor_imag(
    double* __restrict__ dCr, double* __restrict__ dCi, // C (in-place)
    const int* __restrict__ sourcea1, // row metadata
    const int* __restrict__ targeta1, // row metadata
    const double* __restrict__ pa1r, const double* __restrict__ pa1i,
    const double* __restrict__ pa2r, const double* __restrict__ pa2i,
    const int* __restrict__ sourceb1, // column metadata
    const int* __restrict__ targetb1, // column metadata
    const double* __restrict__ pb1r, const double* __restrict__ pb1i,
    const double* __restrict__ pb2r, const double* __restrict__ pb2i,
    int nalpha, int nb, int nbeta_strs_, // sizes
    double factor_imag,                 // scalar: purely imaginary: i*factor_imag
    double acc1r, double acc1i,         // scalar: complex
    double acc2r, double acc2i);        // scalar: complex

template<int BX>
static void launch_givens_soa_real(
    double* dCr, double* dCi,
    const double* pa1r, const double* pa1i,
    const double* pa2r, const double* pa2i,
    const double* pb1r, const double* pb1i,
    const double* pb2r, const double* pb2i,
    const int* sourcea1, const int* targeta1,
    const int* sourceb1, const int* targetb1,
    int nalpha, int nb, int nbeta_strs_,
    double factor_real,
    double acc1r, double acc1i,
    double acc2r, double acc2i);

template<int BX>
static void launch_givens_soa_imag(
    double* dCr, double* dCi,
    const double* pa1r, const double* pa1i,
    const double* pa2r, const double* pa2i,
    const double* pb1r, const double* pb1i,
    const double* pb2r, const double* pb2i,
    const int* sourcea1, const int* targeta1,
    const int* sourceb1, const int* targetb1,
    int nalpha, int nb, int nbeta_strs_,
    double factor_imag,
    double acc1r, double acc1i,
    double acc2r, double acc2i);

extern "C" void givens_update_tiled_wrapper_soa(
    int BX_runtime,
    double* dCr, double* dCi, // C (SoA)
    const double* paritya1_r, const double* paritya1_i, // row metadata
    const double* paritya2_r, const double* paritya2_i, // row metadata
    const int* sourcea1, const int* targeta1, // row metadata
    const double* parityb1_r, const double* parityb1_i, // column metadata
    const double* parityb2_r, const double* parityb2_i, // column metadata
    const int* sourceb1, const int* targetb1, // column metadata
    int nalpha, int nb, int nbeta_strs_, // sizes
    int factor_is_imag,         // 0 => real, 1 => imaginary
    double factor_val,          // if real: factor = factor_val; if imag: factor = i*factor_val
    double acc1_r, double acc1_i,   // accumulator (complex)
    double acc2_r, double acc2_i);  // accumulator (complex)

// __global__ void scale_elements_kernel_soa(
//     double* __restrict__ d_Cout,
//     const int* __restrict__ d_first, 
//     int first_size,
//     const int* __restrict__ d_second, 
//     int second_size,
//     int nbeta_strs_,
//     double factor);

// extern "C" void scale_elements_wrapper_soa(
//     double* d_Cout,
//     const int* d_first, 
//     int first_size,
//     const int* d_second, 
//     int second_size,
//     int nbeta_strs_,
//     double factor);

// __global__ void scale_elements_kernel_soa_factor_real(
//     double* __restrict__ dCr, double* __restrict__ dCi,
//     const int* __restrict__ d_first, int first_size,
//     const int* __restrict__ d_second, int second_size,
//     int nbeta_strs_, double fr);  // factor = fr

// __global__ void scale_elements_kernel_soa_factor_imag(
//     double* __restrict__ dCr, double* __restrict__ dCi,
//     const int* __restrict__ d_first, int first_size,
//     const int* __restrict__ d_second, int second_size,
//     int nbeta_strs_, double fi);  // factor = i*fi

// extern "C" void scale_elements_wrapper_soa_factor_real(
//     double* dCr, double* dCi,
//     const int* d_first, int first_size,
//     const int* d_second, int second_size,
//     int nbeta_strs_, double fr);

// extern "C" void scale_elements_wrapper_soa_factor_imag(
//     double* dCr, double* dCi,
//     const int* d_first, int first_size,
//     const int* d_second, int second_size,
//     int nbeta_strs_, double fi);

__global__ void scale_elements_kernel_soa(
    double* __restrict__ dCr, double* __restrict__ dCi,
    const int* __restrict__ d_first, int first_size,
    const int* __restrict__ d_second, int second_size,
    int nbeta_strs_,
    double fr, double fi);

extern "C" void scale_elements_wrapper_soa(
    double* dCr, double* dCi,
    const int* d_first, int first_size,
    const int* d_second, int second_size,
    int nbeta_strs_, double fr, double fi);