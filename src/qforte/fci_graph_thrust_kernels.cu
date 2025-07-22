#include <cuda_runtime.h>
#include <stdint.h>

__device__ inline uint64_t set_bit_gpu(uint64_t n, int idx) {
    return n | (1ULL << idx);
}

__device__ inline uint64_t unset_bit_gpu(uint64_t n, int idx) {
    return n & ~(1ULL << idx);
}

__device__ inline int count_bits_above_gpu(uint64_t n, int idx) {
    uint64_t mask = (1ULL << idx) - 1;
    return __popcll(n & ~mask);
}

__global__ void make_mapping_each_kernel(
    const uint64_t* strings,
    int length,
    const int* dag, int dag_size,
    const int* undag, int undag_size,
    uint64_t dag_mask, uint64_t undag_mask,
    int* source, int* target, int* parity, int* count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length) return;

    uint64_t current = strings[index];
    bool check = ((current & dag_mask) == 0) && ((current & undag_mask ^ undag_mask) == 0);

    if (check) {
        uint64_t parity_value = 0;
        for (int i = undag_size - 1; i >= 0; i--) {
            parity_value += count_bits_above_gpu(current, undag[i]);
            current = unset_bit_gpu(current, undag[i]);
        }
        for (int i = dag_size - 1; i >= 0; i--) {
            parity_value += count_bits_above_gpu(current, dag[i]);
            current = set_bit_gpu(current, dag[i]);
        }

        int pos = atomicAdd(count, 1);
        source[pos] = index;
        target[pos] = static_cast<int>(current);
        parity[pos] = static_cast<int>(parity_value % 2);
    }
}

void make_mapping_each_cuda(
    const uint64_t* d_strings,
    int length,
    const int* d_dag, int dag_size,
    const int* d_undag, int undag_size,
    int* d_source,
    int* d_target,
    int* d_parity,
    int* d_count,
    uint64_t dag_mask, uint64_t undag_mask)
{
    int blockSize = 256;
    int numBlocks = (length + blockSize - 1) / blockSize;

    make_mapping_each_kernel<<<numBlocks, blockSize>>>(
        d_strings, length,
        d_dag, dag_size,
        d_undag, undag_size,
        dag_mask, undag_mask,
        d_source,
        d_target,
        d_parity,
        d_count);

    cudaDeviceSynchronize();
}
