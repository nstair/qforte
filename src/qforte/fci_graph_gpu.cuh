#include <stdint.h>

// #pragma once
#include <complex>
#include <cuComplex.h>



// Function to set a specific bit in a bitmask
__device__ uint64_t set_bit(uint64_t mask, int pos);

// Function to unset a specific bit in a bitmask
__device__ uint64_t unset_bit(uint64_t mask, int pos);

// Function to count bits above a certain position
__device__ int count_bits_above(uint64_t number, int pos);

// Function to check if a value is in an array
__device__ bool contains(const int* array, int size, int value);

__global__ void make_mapping_each_kernel(
    const int* dag,
    const int dag_size,
    const int* undag,
    const int undag_size,
    const uint64_t* strings,
    const int length,
    int* source,
    int* target,
    int* parity,
    int* count);

extern "C" void make_mapping_each_wrapper(
    const int* dag,
    const int dag_size,
    const int* undag,
    const int undag_size,
    const uint64_t* strings,
    const int length,
    int* source,
    int* target,
    int* parity,
    int* count) 