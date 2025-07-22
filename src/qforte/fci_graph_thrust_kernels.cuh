#pragma once
#include <complex>
#include <cuComplex.h>

__global__ void make_mapping_each_kernel(
    const uint64_t* strings,
    int length,
    const int* dag, int dag_size,
    const int* undag, int undag_size,
    uint64_t dag_mask, uint64_t undag_mask,
    int* source, int* target, int* parity, int* count);

extern "C" void make_mapping_each_cuda(
    const uint64_t* d_strings,
    int length,
    const int* d_dag, int dag_size,
    const int* d_undag, int undag_size,
    int* d_source,
    int* d_target,
    int* d_parity,
    int* d_count,
    uint64_t dag_mask, uint64_t undag_mask);