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
    int targetb_size);


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

__global__ void evolve_individual_nbody_easy_kernel(
    cuDoubleComplex* Cout_data,
    cuDoubleComplex factor,
    const int* map_first,
    const int* map_second,
    size_t map_first_size,
    size_t map_second_size,
    size_t nbeta_strs);

extern "C" void evolve_individual_nbody_easy_wrapper(
    cuDoubleComplex* Cout_data,
    const cuDoubleComplex factor,
    const int* map_first,
    const int* map_second,
    int map_first_size,
    int map_second_size,
    int nbeta_strs);


