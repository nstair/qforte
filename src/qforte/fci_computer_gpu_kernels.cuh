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

extern "C" void scale_elements_wrapper(
    cuDoubleComplex* d_Cout,
    const int* d_first, 
    int first_size,
    const int* d_second, 
    int second_size,
    int nbeta_strs_,
    cuDoubleComplex factor);