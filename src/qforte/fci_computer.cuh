#pragma once
#include <complex>
#include <cuComplex.h>
// CUDA kernel for element-wise addition of two tensors
__global__ void apply_individual_nbody1_accumulate_kernel(
    const cuDoubleComplex* coeff, 
    const cuDoubleComplex* y, 
    const int* d_sourcea,
    const int* d_targeta,
    const int* d_paritya,
    const int* d_sourceb,
    const int* d_targetb,
    const int* d_parityb,
    int nbeta_strs_,
    int targeta_size,
    int targetb_size);

extern "C" void apply_individual_nbody1_accumulate_kernel(
    const cuDoubleComplex* coeff, 
    const cuDoubleComplex* y, 
    const int* d_sourcea,
    const int* d_targeta,
    const int* d_paritya,
    const int* d_sourceb,
    const int* d_targetb,
    const int* d_parityb,
    int nbeta_strs_,
    int targeta_size,
    int targetb_size);

