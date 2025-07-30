#include <cuda_runtime.h>
#include <stdint.h>
#include <cuComplex.h>
#include <cstdio>

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

// GPU device function to perform binary search for index mapping
__device__ int binary_search_index_map(const uint64_t* keys, const int* values, int size, uint64_t target) {
    int left = 0, right = size - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (keys[mid] == target) {
            return values[mid];
        } else if (keys[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1; // Not found
}

__global__ void make_mapping_each_gpu_kernel(
    const uint64_t* strings,
    const uint64_t* map_keys,
    const int* map_values,
    int map_size,
    const int* dag,
    const int* undag,
    int dag_size,
    int undag_size,
    uint64_t dag_mask,
    uint64_t undag_mask,
    int length,
    int* source,
    int* target,
    cuDoubleComplex* parity,
    int* count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length) return;

    uint64_t current = strings[index];
    
    // Check if the operator can act on this determinant
    bool check = ((current & dag_mask) == 0) && ((current & undag_mask) == undag_mask);
    
    if (check) {
        uint64_t parity_value = 0;
        uint64_t result_string = current;
        
        // Apply annihilation operators (undag) - process in reverse order
        for (int i = undag_size - 1; i >= 0; i--) {
            parity_value += count_bits_above_gpu(result_string, undag[i]);
            result_string = unset_bit_gpu(result_string, undag[i]);
        }
        
        // Apply creation operators (dag) - process in reverse order
        for (int i = dag_size - 1; i >= 0; i--) {
            parity_value += count_bits_above_gpu(result_string, dag[i]);
            result_string = set_bit_gpu(result_string, dag[i]);
        }
        
        // Find the target index using binary search
        int target_index = binary_search_index_map(map_keys, map_values, map_size, result_string);
        
        if (target_index >= 0) {
            // Atomically increment count and get insertion position
            int pos = atomicAdd(count, 1);
            
            // Store results
            source[pos] = index;
            target[pos] = target_index;
            
            // Convert parity from 0/1 to +1/-1 format
            int parity_int = 1 - 2 * static_cast<int>(parity_value % 2);
            parity[pos].x = static_cast<double>(parity_int);
            parity[pos].y = 0.0;
        }
    }
}

extern "C" void make_mapping_each_kernel_wrapper(
    const uint64_t* d_strings,
    const uint64_t* d_map_keys,
    const int* d_map_values,
    int map_size,
    const int* d_dag,
    const int* d_undag,
    int dag_size,
    int undag_size,
    uint64_t dag_mask,
    uint64_t undag_mask,
    int length,
    int* d_source,
    int* d_target,
    cuDoubleComplex* d_parity,
    int* d_count)
{
    // Calculate grid and block sizes
    int block_size = 256;
    int grid_size = (length + block_size - 1) / block_size;
    
    // Launch kernel
    make_mapping_each_gpu_kernel<<<grid_size, block_size>>>(
        d_strings, d_map_keys, d_map_values, map_size,
        d_dag, d_undag, dag_size, undag_size,
        dag_mask, undag_mask, length,
        d_source, d_target, d_parity, d_count
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
}
