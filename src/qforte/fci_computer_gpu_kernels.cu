#include "fci_computer_gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>

// Helper function for atomic add with double precision
__device__ double atomicAdd_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


// Helper function for atomic add with cuDoubleComplex
__device__ void atomicAdd_complex(cuDoubleComplex* addr, cuDoubleComplex val) {
    atomicAdd_double(&(addr->x), val.x);
    atomicAdd_double(&(addr->y), val.y);
}


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
//     int tensor_size) 
// {
//     int index1 = blockIdx.x * blockDim.x + threadIdx.x;
//     // int index2 = blockIdx.d_Cout * blockDim.d_Cout + threadIdx.d_Cout;
    
//     if (index1 < targeta_size) {

//         int ta_idx = d_targeta[index1] * nbeta_strs_;
//         int sa_idx = d_sourcea[index1] * nbeta_strs_;

//         cuDoubleComplex pref = cuCmul(coeff, make_cuDoubleComplex(d_paritya[index1], 0.0));

//         for (int j = 0; j < targetb_size; j++) {

//             cuDoubleComplex term = cuCmul(pref, make_cuDoubleComplex(d_parityb[j], 0.0));
//             term = cuCmul(term, d_Cin[sa_idx + d_sourceb[j]]);
//             // atomicAdd(&d_Cout[ta_idx + d_targetb[j]].x, term.x);
//             // atomicAdd(&d_Cout[ta_idx + d_targetb[j]].y, term.y);
//             d_Cout[ta_idx + d_targetb[j]].x += term.x;
//             // d_Cout[ta_idx + d_targetb[j]].y += term.y;

//         }


//     }
// }

// CUDA kernel
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
//     int tensor_size) 
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx < targeta_size) {
//         int ta_idx = d_targeta[idx] * nbeta_strs_;
//         int sa_idx = d_sourcea[idx] * nbeta_strs_;
//         cuDoubleComplex pref = cuCmul(coeff, make_cuDoubleComplex(d_paritya[idx], 0.0));

//         #pragma unroll
//         for (int j = 0; j < targetb_size; ++j) {
//             cuDoubleComplex term = cuCmul(pref, make_cuDoubleComplex(d_parityb[j], 0.0));
//             term = cuCmul(term, d_Cin[sa_idx + d_sourceb[j]]);
//             d_Cout[ta_idx + d_targetb[j]] = cuCadd(term,  d_Cout[ta_idx + d_targetb[j]]);
//         }
//     }
// }

// V2 about 2x faster
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
    int tensor_size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < targeta_size) {
        int ta_idx = d_targeta[idx] * nbeta_strs_;
        int sa_idx = d_sourcea[idx] * nbeta_strs_;
        
        // cuDoubleComplex pref = cuCmul(coeff, make_cuDoubleComplex(d_paritya[idx], 0.0));
        cuDoubleComplex pref = cuCmul(coeff, d_paritya[idx]);

         if (idy < targetb_size)  {
            cuDoubleComplex term = cuCmul(pref, d_parityb[idy]);

            term = cuCmul(term, d_Cin[sa_idx + d_sourceb[idy]]);

            d_Cout[ta_idx + d_targetb[idy]] = cuCadd(term,  d_Cout[ta_idx + d_targetb[idy]]);
        }
    }
}


// V3 about same as V2
// __global__ void apply_individual_nbody1_accumulate_kernel(
//     const cuDoubleComplex coeff, 
//     const cuDoubleComplex* __restrict__ d_Cin, 
//     cuDoubleComplex* __restrict__ d_Cout, 
//     const int* __restrict__ d_sourcea,
//     const int* __restrict__ d_targeta,
//     const int* __restrict__ d_paritya,
//     const int* __restrict__ d_sourceb,
//     const int* __restrict__ d_targetb,
//     const int* __restrict__ d_parityb,
//     int nbeta_strs_,
//     int targeta_size,
//     int targetb_size,
//     int tensor_size) 
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;

//     if (idx < targetb_size) {
//         int ta_idx = d_targeta[idx] * nbeta_strs_;
//         int sa_idx = d_sourcea[idx] * nbeta_strs_;

//         cuDoubleComplex pref = cuCmul(coeff, make_cuDoubleComplex(d_paritya[idx], 0.0));

//         if (idy < targetb_size)  {
//             cuDoubleComplex term = cuCmul(pref, make_cuDoubleComplex(d_parityb[idy], 0.0));
//             term = cuCmul(term, d_Cin[sa_idx + d_sourceb[idy]]);
            
//             d_Cout[ta_idx + d_targetb[idy]] = cuCadd(term,  d_Cout[ta_idx + d_targetb[idy]]);
//         }
//     }
// }

// __global__ void apply_individual_nbody1_accumulate_kernel(
//     const cuDoubleComplex coeff, 
//     const cuDoubleComplex* __restrict__ d_Cin, 
//     cuDoubleComplex* __restrict__ d_Cout, 
//     const int* __restrict__ d_sourcea,
//     const int* __restrict__ d_targeta,
//     const int* __restrict__ d_paritya,
//     const int* __restrict__ d_sourceb,
//     const int* __restrict__ d_targetb,
//     const int* __restrict__ d_parityb,
//     int nbeta_strs_,
//     int targeta_size,
//     int targetb_size,
//     int tensor_size) 
// {
//     // // Allocate shared memory for source, target, and parity arrays
//     // extern __shared__ int shared_mem[];
//     // int* shared_sourcea = shared_mem;
//     // int* shared_targeta = shared_mem + targeta_size;
//     // int* shared_paritya = shared_mem + 2 * targeta_size;
//     // int* shared_sourceb = shared_mem + 3 * targeta_size;
//     // int* shared_targetb = shared_mem + 3 * targeta_size + targetb_size;
//     // int* shared_parityb = shared_mem + 3 * targeta_size + 2 * targetb_size;

//     // int tid = threadIdx.x + threadIdx.y * blockDim.x;
    
//     // // Load data into shared memory
//     // for (int i = tid; i < targeta_size; i += blockDim.x * blockDim.y) {
//     //     shared_sourcea[i] = d_sourcea[i];
//     //     shared_targeta[i] = d_targeta[i];
//     //     shared_paritya[i] = d_paritya[i];
//     // }
//     // for (int i = tid; i < targetb_size; i += blockDim.x * blockDim.y) {
//     //     shared_sourceb[i] = d_sourceb[i];
//     //     shared_targetb[i] = d_targetb[i];
//     //     shared_parityb[i] = d_parityb[i];
//     // }

//     // // Synchronize to ensure all threads have loaded their data into shared memory
//     // __syncthreads();

//     // Grid-stride loop for processing elements
//     for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < targeta_size; idx += blockDim.x * gridDim.x) {
//         int ta_idx = d_targeta[idx] * nbeta_strs_;
//         int sa_idx = d_sourcea[idx] * nbeta_strs_;
        
//         cuDoubleComplex pref = cuCmul(coeff, make_cuDoubleComplex(d_paritya[idx], 0.0));

//         for (int idy = blockIdx.y * blockDim.y + threadIdx.y; idy < targetb_size; idy += blockDim.y * gridDim.y) {
//             cuDoubleComplex term = cuCmul(pref, make_cuDoubleComplex(d_parityb[idy], 0.0));
//             term = cuCmul(term, d_Cin[sa_idx + d_sourceb[idy]]);

//             d_Cout[ta_idx + d_targetb[idy]] = cuCadd(term, d_Cout[ta_idx + d_targetb[idy]]);
//         }
//     }
// }

// V2_atomic - thread-safe version using atomicAdd to prevent race conditions
__global__ void apply_individual_nbody1_accumulate_kernel_atomic(
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
    int tensor_size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < targeta_size) {
        int ta_idx = d_targeta[idx] * nbeta_strs_;
        int sa_idx = d_sourcea[idx] * nbeta_strs_;
        
        cuDoubleComplex pref = cuCmul(coeff, d_paritya[idx]);

        if (idy < targetb_size) {
            cuDoubleComplex term = cuCmul(pref, d_parityb[idy]);
            term = cuCmul(term, d_Cin[sa_idx + d_sourceb[idy]]);

            // Thread-safe atomic accumulation
            int output_idx = ta_idx + d_targetb[idy];
            atomicAdd_double(&d_Cout[output_idx].x, term.x);
            atomicAdd_double(&d_Cout[output_idx].y, term.y);
        }
    }
}

void apply_individual_nbody1_accumulate_wrapper(
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
    int tensor_size) 
{
    // 2D grid configuration for the atomic kernel
    dim3 blockSize(16, 16);  // 16x16 = 256 threads per block
    dim3 gridSize((targeta_size + blockSize.x - 1) / blockSize.x,
                  (targetb_size + blockSize.y - 1) / blockSize.y);
    
    apply_individual_nbody1_accumulate_kernel_atomic<<<gridSize, blockSize>>>(
        coeff, d_Cin, d_Cout, d_sourcea, d_targeta, d_paritya, 
        d_sourceb, d_targetb, d_parityb, nbeta_strs_, 
        targeta_size, targetb_size, tensor_size);
   

    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch apply_individual_nbody1_accumulate_kernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        throw std::runtime_error("Kernel launch failed");
    }

    // Wait for the kernel to complete and check for errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution failed (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        throw std::runtime_error("Kernel execution failed");
    }
}


__global__ void apply_individual_nbody1_accumulate_kernel_atomic_v2(
    const cuDoubleComplex coeff, 
    const cuDoubleComplex* __restrict__ d_Cin,        // NEW: __restrict__
    cuDoubleComplex* __restrict__ d_Cout,             // NEW: __restrict__
    const int* __restrict__ d_sourcea,                // NEW: __restrict__
    const int* __restrict__ d_targeta,
    const cuDoubleComplex* __restrict__ d_paritya,
    const int* __restrict__ d_sourceb,
    const int* __restrict__ d_targetb,
    const cuDoubleComplex* __restrict__ d_parityb,
    int nbeta_strs_,
    int targeta_size,
    int targetb_size,
    int tensor_size) 
{
    int total = targeta_size * targetb_size;           // NEW: Flatten 2D grid
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < total) {
        int idx = index / targetb_size;                // Recover original indices
        int idy = index % targetb_size;

        int ta_idx = d_targeta[idx] * nbeta_strs_;
        int sa_idx = d_sourcea[idx] * nbeta_strs_;

        cuDoubleComplex pref = cuCmul(coeff, d_paritya[idx]);
        cuDoubleComplex term = cuCmul(pref, d_parityb[idy]);
        term = cuCmul(term, d_Cin[sa_idx + d_sourceb[idy]]);

        int output_idx = ta_idx + d_targetb[idy];
        atomicAdd_double(&d_Cout[output_idx].x, term.x);
        atomicAdd_double(&d_Cout[output_idx].y, term.y);
    }
}

void apply_individual_nbody1_accumulate_wrapper_v2(
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
    int tensor_size) 
{
    int total = targeta_size * targetb_size;
    int blockSize = 256;
    int numBlocks = (total + blockSize - 1) / blockSize;
    
    apply_individual_nbody1_accumulate_kernel_atomic<<<numBlocks, blockSize>>>(
        coeff, d_Cin, d_Cout, d_sourcea, d_targeta, d_paritya, 
        d_sourceb, d_targetb, d_parityb, nbeta_strs_, 
        targeta_size, targetb_size, tensor_size);
    
    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch apply_individual_nbody1_accumulate_kernel_atomic_v2 (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        throw std::runtime_error("Kernel launch failed");
    }

    // Wait for the kernel to complete and check for errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution failed (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        throw std::runtime_error("Kernel execution failed");
    }
}


/* This is only better if the colision rate is very high */

/*
__global__ void apply_individual_nbody1_accumulate_kernel_shared(
    const cuDoubleComplex coeff, 
    const cuDoubleComplex* __restrict__ d_Cin,
    cuDoubleComplex* __restrict__ d_Cout,
    const int* __restrict__ d_sourcea,
    const int* __restrict__ d_targeta,
    const cuDoubleComplex* __restrict__ d_paritya,
    const int* __restrict__ d_sourceb,
    const int* __restrict__ d_targetb,
    const cuDoubleComplex* __restrict__ d_parityb,
    int nbeta_strs_,
    int targeta_size,
    int targetb_size) 
{
    // Flatten the 2D grid to 1D, as in previous example
    int total = targeta_size * targetb_size;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x;
    int tid = threadIdx.x;

    // Allocate shared memory for reduction:
    extern __shared__ int shared[]; // Dynamic shared mem: int + complex per thread
    int* s_idx = shared; // [blockDim.x] -- each thread's output_idx
    cuDoubleComplex* s_val = (cuDoubleComplex*)&s_idx[blockSize]; // [blockDim.x] -- each thread's value

    // Each thread computes its term and output_idx
    int output_idx = -1;
    cuDoubleComplex term = make_cuDoubleComplex(0.0, 0.0);

    if (index < total) {
        int idx = index / targetb_size;
        int idy = index % targetb_size;

        int ta_idx = d_targeta[idx] * nbeta_strs_;
        int sa_idx = d_sourcea[idx] * nbeta_strs_;

        cuDoubleComplex pref = cuCmul(coeff, d_paritya[idx]);
        term = cuCmul(pref, d_parityb[idy]);
        term = cuCmul(term, d_Cin[sa_idx + d_sourceb[idy]]);

        output_idx = ta_idx + d_targetb[idy];
    }

    // Store each thread's output index and value in shared memory
    s_idx[tid] = output_idx;
    s_val[tid] = term;

    __syncthreads();

    // **Block-wise reduction by output_idx**
    // Each thread checks if it's the first occurrence of its output_idx in this block
    // If so, it sums all contributions in the block with that output_idx
    if (output_idx >= 0) {
        cuDoubleComplex block_sum = s_val[tid];

        // Only the first occurrence of this output_idx in the block performs the atomicAdd
        bool is_first = true;
        for (int t = 0; t < tid; ++t) {
            if (s_idx[t] == output_idx) {
                is_first = false;
                break;
            }
        }
        if (is_first) {
            // Sum all other threads in the block with the same output_idx
            for (int t = tid + 1; t < blockSize; ++t) {
                if (s_idx[t] == output_idx) {
                    block_sum.x += s_val[t].x;
                    block_sum.y += s_val[t].y;
                }
            }
            // One atomic add per unique output_idx per block
            atomicAdd_complex(&d_Cout[output_idx], block_sum);
        }
    }
}


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
    int tensor_size) 
{
    int blockSize = 256;
    int numBlocks = (targeta_size * targetb_size + blockSize - 1) / blockSize;

    // Allocate shared memory for reduction
    size_t sharedMemSize = blockSize * (sizeof(int) + sizeof(cuDoubleComplex));

    apply_individual_nbody1_accumulate_kernel_shared<<<numBlocks, blockSize, sharedMemSize>>>(
        coeff, d_Cin, d_Cout, d_sourcea, d_targeta, d_paritya, 
        d_sourceb, d_targetb, d_parityb, nbeta_strs_, 
        targeta_size, targetb_size);
    
    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch apply_individual_nbody1_accumulate_kernel_shared (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        throw std::runtime_error("Kernel launch failed");
    }

    // Wait for the kernel to complete and check for errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution failed (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        throw std::runtime_error("Kernel execution failed");
    }
}
    */

__global__ void scale_elements_kernel(
    cuDoubleComplex* d_Cout,
    const int* d_first, 
    int first_size,
    const int* d_second, 
    int second_size,
    int nbeta_strs_,
    cuDoubleComplex factor) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < first_size && j < second_size) {
        int idx = d_first[i] * nbeta_strs_ + d_second[j];
        d_Cout[idx] = cuCmul(d_Cout[idx], factor);
    }
}

extern "C" void scale_elements_wrapper(
    cuDoubleComplex* d_Cout,
    const int* d_first, 
    int first_size,
    const int* d_second, 
    int second_size,
    int nbeta_strs_,
    cuDoubleComplex factor) 
{
    dim3 blockSize(16, 16);
    dim3 gridSize((first_size + blockSize.x - 1) / blockSize.x, 
                  (second_size + blockSize.y - 1) / blockSize.y);

    scale_elements_kernel<<<gridSize, blockSize>>>(d_Cout, d_first, first_size, d_second, second_size, nbeta_strs_, factor);

    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch scale_elements_kernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        throw std::runtime_error("Kernel launch failed");
    }

    // Wait for the kernel to complete and check for errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution failed (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        throw std::runtime_error("Kernel execution failed");
    }
}