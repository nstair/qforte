#pragma once
#include <complex>
#include <cuComplex.h>

__global__ void make_mapping_each_kernel(
    bool alpha,
    const int* dag,
    int dag_size,
    const int* undag,
    int undag_size,
    const uint64_t* strings,
    int length,
    int* source,
    int* target,
    int* parity,
    int* count);

extern "C" void make_mapping_each_wrapper(
    bool alpha,
    const int* dag,
    int dag_size,
    const int* undag,
    int undag_size,
    const uint64_t* strings,
    int length,
    int* source,
    int* target,
    int* parity,
    int* count) 