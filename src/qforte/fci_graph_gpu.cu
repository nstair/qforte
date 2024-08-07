#include "fci_graph_gpu.cuh"
#include <cuda_runtime.h>
#include <iostream>


__device__ uint64_t set_bit(uint64_t mask, int pos) {
    return mask | (1ULL << pos);
}

__device__ uint64_t unset_bit(uint64_t mask, int pos) {
    return mask & ~(1ULL << pos);
}

__device__ int count_bits_above(uint64_t number, int pos) {
    return __popc(number & ~((1ULL << (pos + 1)) - 1));
}

__device__ bool contains(const int* array, int size, int value) {
    for (int i = 0; i < size; ++i) {
        if (array[i] == value) {
            return true;
        }
    }
    return false;
}

__global__ void make_mapping_each_kernel(
    const int* dag,
    const int dag_size,
    const int* undag,
    const int undag_size,
    const uint64_t* strings,
    int length,
    int* source,
    int* target,
    int* parity,
    int* count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length) return;

    uint64_t dag_mask = 0;
    uint64_t undag_mask = 0;

    // Build dag_mask
    for (int i = 0; i < dag_size; i++) {
        if (!contains(undag, undag_size, dag[i])) {
            dag_mask = set_bit(dag_mask, dag[i]);
        }
    }

    // Build undag_mask
    for (int i = 0; i < undag_size; i++) {
        undag_mask = set_bit(undag_mask, undag[i]);
    }

    uint64_t current = strings[index];
    bool check = ((current & dag_mask) == 0) && ((current & (undag_mask ^ undag_mask)) == 0);

    if (check) {
        uint64_t tmp = current;
        uint64_t parity_value = 0;

        // Compute parity_value for undag
        for (int i = undag_size - 1; i >= 0; i--) {
            parity_value += count_bits_above(current, undag[i]);
            current = unset_bit(current, undag[i]);
        }

        // Compute parity_value for dag
        for (int i = dag_size - 1; i >= 0; i--) {
            parity_value += count_bits_above(current, dag[i]);
            current = set_bit(current, dag[i]);
        }

        int local_index = atomicAdd(count, 1);
        source[local_index] = index;
        target[local_index] = static_cast<int>(current);
        parity[local_index] = static_cast<int>(parity_value % 2);
    }
}

void make_mapping_each_wrapper(
    const int* dag,
    const int dag_size,
    const int* undag,
    const int undag_size,
    const uint64_t* strings,
    int length,
    int* source,
    int* target,
    int* parity,
    int* count) 
{

    int blockSize = 256; 
    int numBlocks = (length + blockSize - 1) / blockSize;

    make_mapping_each_kernel<<<numBlocks, blockSize>>>(
        d_dag, 
        dag.size(),
        d_undag, 
        undag.size(),
        d_strings, 
        length,
        d_source, 
        d_target, 
        d_parity, 
        d_count);
}