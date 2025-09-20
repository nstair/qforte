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

extern "C" void scale_elements_wrapper_complex(
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

__global__ void scale_elements_kernel_soa(
    double* __restrict__ d_Cout,
    const int* __restrict__ d_first, 
    int first_size,
    const int* __restrict__ d_second, 
    int second_size,
    int nbeta_strs_,
    double factor) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < first_size && j < second_size) {
        int idx = d_first[i] * nbeta_strs_ + d_second[j];
        d_Cout[idx] *= factor;
    }
}


extern "C" void scale_elements_wrapper_soa(
    double* d_Cout,
    const int* d_first, 
    int first_size,
    const int* d_second, 
    int second_size,
    int nbeta_strs_,
    double factor) 
{
    dim3 blockSize(16, 16);
    dim3 gridSize((first_size  + blockSize.x - 1) / blockSize.x, 
                  (second_size + blockSize.y - 1) / blockSize.y);

    // Real-valued kernel launch (provide this kernel or a templated alias)
    scale_elements_kernel_soa<<<gridSize, blockSize>>>(
        d_Cout, d_first, first_size, d_second, second_size, nbeta_strs_, factor);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch scale_elements_kernel_soa ("
                  << cudaGetErrorString(err) << ")!" << std::endl;
        throw std::runtime_error("Kernel launch failed");
    }

    // Sync and check for runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution failed ("
                  << cudaGetErrorString(err) << ")!" << std::endl;
        throw std::runtime_error("Kernel execution failed");
    }
}

// === New in-place 2x2 Givens-like update kernel implementation ===
__global__ void inplace_givens_update_kernel(
    cuDoubleComplex* __restrict__ d_Cout,
    const int* __restrict__ sourcea1,
    const int* __restrict__ targeta1,
    const cuDoubleComplex* __restrict__ paritya1,
    const cuDoubleComplex* __restrict__ paritya2,
    const int* __restrict__ sourceb1,
    const int* __restrict__ targetb1,
    const cuDoubleComplex* __restrict__ parityb1,
    const cuDoubleComplex* __restrict__ parityb2,
    int na,
    int nb,
    int nbeta_strs_,
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2)
{
    int ta = threadIdx.x;
    int tb = threadIdx.y;

    int ia = blockIdx.x * blockDim.x + ta;
    int ib = blockIdx.y * blockDim.y + tb;

    if (ia >= na || ib >= nb) return;

    __shared__ int s_sourcea1[16];
    __shared__ int s_targeta1[16];
    __shared__ cuDoubleComplex s_paritya1[16];
    __shared__ cuDoubleComplex s_paritya2[16];
    __shared__ int s_sourceb1[16];
    __shared__ int s_targetb1[16];
    __shared__ cuDoubleComplex s_parityb1[16];
    __shared__ cuDoubleComplex s_parityb2[16];

    // Load data into shared memory
    s_sourcea1[ta] = sourcea1[ia];
    s_targeta1[ta] = targeta1[ia];
    s_paritya1[ta] = paritya1[ia];
    s_paritya2[ta] = paritya2[ia];
    s_sourceb1[tb] = sourceb1[ib];
    s_targetb1[tb] = targetb1[ib];
    s_parityb1[tb] = parityb1[ib];
    s_parityb2[tb] = parityb2[ib];

    /* do everything except final write with shared mem */

    // Indices for leg1
    int sa1 = s_sourcea1[ta];
    int ta1 = s_targeta1[ta];
    int sb1 = s_sourceb1[tb];
    int tb1 = s_targetb1[tb];

    // Row offsets
    int sa1_row = sa1 * nbeta_strs_;
    int ta1_row = ta1 * nbeta_strs_;

    // Column indices
    int idx_u = sa1_row + sb1; // u ≡ (sa1,sb1)
    int idx_v = ta1_row + tb1; // v ≡ (ta1,tb1)

    // Snapshot u0, v0 (in-place safety)
    cuDoubleComplex u0 = d_Cout[idx_u];
    cuDoubleComplex v0 = d_Cout[idx_v];

    // Combined parity factors
    cuDoubleComplex p1 = cuCmul(s_paritya1[ta], s_parityb1[tb]); // g† leg
    cuDoubleComplex p2 = cuCmul(s_paritya2[ta], s_parityb2[tb]); // g  leg

    // u' = factor * u0 + acc_coeff2 * p2 * v0
    cuDoubleComplex term_u = cuCmul(acc_coeff2, cuCmul(p2, v0));
    cuDoubleComplex u_new = cuCadd(cuCmul(factor, u0), term_u);

    // v' = factor * v0 + acc_coeff1 * p1 * u0
    cuDoubleComplex term_v = cuCmul(acc_coeff1, cuCmul(p1, u0));
    cuDoubleComplex v_new = cuCadd(cuCmul(factor, v0), term_v);

    // Sync threads before writing back
    __syncthreads();

    d_Cout[idx_u] = u_new;
    d_Cout[idx_v] = v_new;
}

extern "C" void inplace_givens_update_wrapper(
    cuDoubleComplex* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const cuDoubleComplex* paritya1,
    const cuDoubleComplex* paritya2,
    const int* sourceb1,
    const int* targetb1,
    const cuDoubleComplex* parityb1,
    const cuDoubleComplex* parityb2,
    int na,
    int nb,
    int nbeta_strs_,
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2)
{
    if (na == 0 || nb == 0) return;

    dim3 block(16, 16);
    dim3 grid((na + block.x - 1) / block.x, (nb + block.y - 1) / block.y);

    inplace_givens_update_kernel<<<grid, block>>>(
        d_Cout,
        sourcea1, targeta1, paritya1,
        paritya2,
        sourceb1, targetb1, parityb1,
        parityb2,
        na, nb, nbeta_strs_,
        factor, acc_coeff1, acc_coeff2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch inplace_givens_update_kernel (" << cudaGetErrorString(err) << ")" << std::endl;
        throw std::runtime_error("inplace_givens_update_kernel launch failed");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "inplace_givens_update_kernel execution failed (" << cudaGetErrorString(err) << ")" << std::endl;
        throw std::runtime_error("inplace_givens_update_kernel execution failed");
    }
}

// Rows-only, coalesced across columns.
// One block processes one (sa1, ta1) pair; threads iterate j across nbeta_strs_.
/// TODO: shared memory
__global__ void inplace_givens_update_rows_kernel(
    cuDoubleComplex* __restrict__ d_Cout,
    const int* __restrict__ sourcea1,      // [na]
    const int* __restrict__ targeta1,      // [na]
    const cuDoubleComplex* __restrict__ paritya1, // [na]  (g† leg, row)
    const cuDoubleComplex* __restrict__ paritya2, // [na]  (g  leg, row)
    int na,
    int nbeta_strs_,                        // number of columns
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2)
{
    int ia = blockIdx.x;                          // one block per (sa1, ta1) pair
    if (ia >= na) return;

    // Broadcast row-scoped values once per block
    __shared__ int s_sa1, s_ta1;
    __shared__ cuDoubleComplex s_pa1, s_pa2;
    if (threadIdx.x == 0) {
        s_sa1 = sourcea1[ia];
        s_ta1 = targeta1[ia];
        s_pa1 = paritya1[ia];
        s_pa2 = paritya2[ia];
    }
    __syncthreads();

    const int sa1 = s_sa1, ta1 = s_ta1;
    const cuDoubleComplex pa1 = s_pa1, pa2 = s_pa2;
    const int base_u = sa1 * nbeta_strs_;
    const int base_v = ta1 * nbeta_strs_;

    for (int col = threadIdx.x; col < nbeta_strs_; col += blockDim.x) {
        const int idx_u = base_u + col;   // (sa1, col)
        const int idx_v = base_v + col;   // (ta1, col)

        const cuDoubleComplex u0 = d_Cout[idx_u];
        const cuDoubleComplex v0 = d_Cout[idx_v];

        const cuDoubleComplex u_new = cuCadd(cuCmul(factor, u0), cuCmul(acc_coeff2, cuCmul(pa2, v0)));
        const cuDoubleComplex v_new = cuCadd(cuCmul(factor, v0), cuCmul(acc_coeff1, cuCmul(pa1, u0)));

        d_Cout[idx_u] = u_new;
        d_Cout[idx_v] = v_new;
    }
}


extern "C" void inplace_givens_update_complex_rows_wrapper(
    cuDoubleComplex* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const cuDoubleComplex* paritya1,
    const cuDoubleComplex* paritya2,
    int na,
    int nbeta_strs_,
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2)
{
    if (na == 0 || nbeta_strs_ == 0) return;

    // Choose threads per block: cover columns with good occupancy.
    // Clamp to device limits if you prefer; 256 is a good default.
    int threads = std::min(256, nbeta_strs_);
    // Keep at least one warp
    if (threads < 32) threads = 32;

    dim3 block(threads);
    dim3 grid(na);  // one block per (sa1, ta1) pair

    inplace_givens_update_rows_kernel<<<grid, block>>>(
        d_Cout,
        sourcea1, targeta1, paritya1, paritya2,
        na, nbeta_strs_,
        factor, acc_coeff1, acc_coeff2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch inplace_givens_update_rows_kernel ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_rows_kernel launch failed");
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "inplace_givens_update_rows_kernel execution failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_rows_kernel execution failed");
    }
}

template<int BX>  // number of column-pairs handled per block (e.g., 32)
__global__ void inplace_givens_update_complex_tiled(
    cuDoubleComplex* __restrict__ d_Cout,
    const int* __restrict__ sourcea1,
    const int* __restrict__ targeta1,
    const cuDoubleComplex* __restrict__ paritya1,
    const cuDoubleComplex* __restrict__ paritya2,
    const int* __restrict__ sourceb1,
    const int* __restrict__ targetb1,
    const cuDoubleComplex* __restrict__ parityb1,
    const cuDoubleComplex* __restrict__ parityb2,
    int nalpha,          // rows
    int nb,              // number of column-pairs
    int nbeta_strs_,
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2)
{
    // Block covers BX consecutive column-pairs starting at ib0
    const int ib0 = blockIdx.x * BX;
    if (ib0 >= nb) return;

    // Thread layout: x = column within the tile, y = row lane inside a small row strip
    const int tx = threadIdx.x;             // [0, BX)
    const int ty = threadIdx.y;             // [0, AY)
    constexpr int AY = 8;                   // small row strip per block
    static_assert(BX % 32 == 0, "Pick BX multiple of warp width for coalescing");

    // Shared: BX col-pair metadata + AY row metadata
    __shared__ int s_sb1[BX], s_tb1[BX];
    __shared__ cuDoubleComplex s_pb1[BX], s_pb2[BX];

    __shared__ int s_sa1[AY], s_ta1[AY];
    __shared__ cuDoubleComplex s_pa1[AY], s_pa2[AY];

    // Load the BX column-pairs (one per tx lane; replicate across ty)
    if (tx + ib0 < nb && ty == 0) {
        const int ib = ib0 + tx;
        s_sb1[tx] = sourceb1[ib];
        s_tb1[tx] = targetb1[ib];
        s_pb1[tx] = parityb1[ib];
        s_pb2[tx] = parityb2[ib];
    }
    __syncthreads();

    // Sweep rows in strips of AY
    for (int ia0 = blockIdx.y * AY; ia0 < nalpha; ia0 += gridDim.y * AY)
    {
        // Cache AY row metadata once
        if (ty < AY && tx == 0) {
            const int ia = ia0 + ty;
            if (ia < nalpha) {
                s_sa1[ty] = sourcea1[ia];
                s_ta1[ty] = targeta1[ia];
                s_pa1[ty] = paritya1[ia];
                s_pa2[ty] = paritya2[ia];
            }
        }
        __syncthreads();

        const int ia = ia0 + ty;
        if (ia < nalpha && tx + ib0 < nb) {
            // Registers for the row
            const int sa1 = s_sa1[ty];
            const int ta1 = s_ta1[ty];
            const cuDoubleComplex pa1 = s_pa1[ty];
            const cuDoubleComplex pa2 = s_pa2[ty];

            // Registers for this column-pair
            // const int ib   = ib0 + tx;
            const int sb1  = s_sb1[tx];
            const int tb1  = s_tb1[tx];
            const cuDoubleComplex pb1 = s_pb1[tx];
            const cuDoubleComplex pb2 = s_pb2[tx];

            const int base_u = sa1 * nbeta_strs_;
            const int base_v = ta1 * nbeta_strs_;

            const int idx_u  = base_u + sb1;  // (sa1, sb1)
            const int idx_v  = base_v + tb1;  // (ta1, tb1)

            // Within a warp, tx varies ⇒ idx_* vary by +1 (contiguous) if sb1/tb1 are consecutive.
            // To ensure that, store column-pairs for a tile as consecutive sb1/tb1 (typical).
            const cuDoubleComplex u0 = d_Cout[idx_u];
            const cuDoubleComplex v0 = d_Cout[idx_v];

            const cuDoubleComplex p1 = cuCmul(pa1, pb1);
            const cuDoubleComplex p2 = cuCmul(pa2, pb2);

            const cuDoubleComplex u_new = cuCadd(cuCmul(factor, u0), cuCmul(acc_coeff2, cuCmul(p2, v0)));
            const cuDoubleComplex v_new = cuCadd(cuCmul(factor, v0), cuCmul(acc_coeff1, cuCmul(p1, u0)));

            d_Cout[idx_u] = u_new;
            d_Cout[idx_v] = v_new;
        }
        __syncthreads();
    }
}

// Internal helper to launch a particular BX specialization
template<int BX>
static void launch_inplace_givens_update_complex_tiled(
    cuDoubleComplex* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const cuDoubleComplex* paritya1,
    const cuDoubleComplex* paritya2,
    const int* sourceb1,
    const int* targetb1,
    const cuDoubleComplex* parityb1,
    const cuDoubleComplex* parityb2,
    int nalpha,
    int nb,
    int nbeta_strs_,
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2)
{
    if (nalpha == 0 || nb == 0 || nbeta_strs_ == 0) return;

    // Must match the kernel's constexpr AY
    constexpr int AY = 8;

    // Each block covers BX consecutive column-pairs and AY rows (as a strip).
    const int grid_x = (nb + BX - 1) / BX;
    const int grid_y = std::max(1, (nalpha + AY - 1) / AY);

    // Block has BX threads along x (columns in the tile) and AY along y (rows in the strip).
    dim3 block(BX, AY);
    dim3 grid(grid_x, grid_y);

    // Sanity: make sure block size is legal (BX*AY <= 1024 on most GPUs)
    if (block.x * block.y > 1024) {
        throw std::invalid_argument("Block size BX*AY exceeds device limit");
    }

    inplace_givens_update_complex_tiled<BX><<<grid, block>>>(
        d_Cout,
        sourcea1, targeta1, paritya1, paritya2,
        sourceb1, targetb1, parityb1, parityb2,
        nalpha, nb, nbeta_strs_,
        factor, acc_coeff1, acc_coeff2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch inplace_givens_update_complex_tiled<"
                  << BX << "> (" << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_complex_tiled launch failed");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "inplace_givens_update_complex_tiled<" << BX
                  << "> execution failed (" << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_complex_tiled execution failed");
    }
}

// Extern "C" wrapper with runtime BX selection.
// Supported BX values are 32 and 64 by default (add more cases as you like).
extern "C" void inplace_givens_update_complex_tiled_wrapper(
    int BX_runtime,                      // pick 32 or 64 (must divide warp multiples)
    cuDoubleComplex* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const cuDoubleComplex* paritya1,
    const cuDoubleComplex* paritya2,
    const int* sourceb1,
    const int* targetb1,
    const cuDoubleComplex* parityb1,
    const cuDoubleComplex* parityb2,
    int nalpha,          // rows
    int nb,              // number of column-pairs
    int nbeta_strs_,     // leading dimension (num columns)
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2)
{
    if (nalpha == 0 || nb == 0 || nbeta_strs_ == 0) return;

    switch (BX_runtime) {
        case 64:
            launch_inplace_givens_update_complex_tiled<64>(
                d_Cout, sourcea1, targeta1, paritya1, paritya2,
                sourceb1, targetb1, parityb1, parityb2,
                nalpha, nb, nbeta_strs_,
                factor, acc_coeff1, acc_coeff2);
            break;
        case 32:
            launch_inplace_givens_update_complex_tiled<32>(
                d_Cout, sourcea1, targeta1, paritya1, paritya2,
                sourceb1, targetb1, parityb1, parityb2,
                nalpha, nb, nbeta_strs_,
                factor, acc_coeff1, acc_coeff2);
            break;
        default:
            // Fallback or throw—here we fallback to 32 for convenience.
            std::cerr << "Warning: unsupported BX=" << BX_runtime
                      << " — defaulting to BX=32.\n";
            launch_inplace_givens_update_complex_tiled<32>(
                d_Cout, sourcea1, targeta1, paritya1, paritya2,
                sourceb1, targetb1, parityb1, parityb2,
                nalpha, nb, nbeta_strs_,
                factor, acc_coeff1, acc_coeff2);
            break;
    }
}

// =============================
// All / SoA kernel
// u' = factor * u + acc2 * ((pa2*pb2) * v)
// v' = factor * v + acc1 * ((pa1*pb1) * u)
// Everything here is either real / imaginary scalars.
// =============================
template<int BX>
__global__ void givens_update_soa_tiled(
    double* __restrict__ dC,          // C_data
    const int* __restrict__ sourcea1,
    const int* __restrict__ targeta1,
    const double* __restrict__ paritya1,  // pa1
    const double* __restrict__ paritya2,  // pa2
    const int* __restrict__ sourceb1,
    const int* __restrict__ targetb1,
    const double* __restrict__ parityb1,  // pb1
    const double* __restrict__ parityb2,  // pb2
    int nalpha,
    int nb,
    int nbeta_strs_,
    double factor,
    double acc1,
    double acc2)
{
    constexpr int AY = 8;
    static_assert(BX % 32 == 0, "BX must be a multiple of 32");

    const int ib0 = blockIdx.x * BX;
    if (ib0 >= nb) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ int    s_sb1[BX], s_tb1[BX];
    __shared__ double s_pb1[BX], s_pb2[BX];

    __shared__ int    s_sa1[AY], s_ta1[AY];
    __shared__ double s_pa1[AY], s_pa2[AY];

    // Load column-pair metadata (once per block along y)
    if (tx + ib0 < nb && ty == 0) {
        const int ib = ib0 + tx;
        s_sb1[tx] = sourceb1[ib];
        s_tb1[tx] = targetb1[ib];
        s_pb1[tx] = parityb1[ib];
        s_pb2[tx] = parityb2[ib];
    }
    __syncthreads();

    for (int ia0 = blockIdx.y * AY; ia0 < nalpha; ia0 += gridDim.y * AY) {
        // Load row metadata (once per block along x)
        if (ty < AY && tx == 0) {
            const int ia = ia0 + ty;
            if (ia < nalpha) {
                s_sa1[ty] = sourcea1[ia];
                s_ta1[ty] = targeta1[ia];
                s_pa1[ty] = paritya1[ia];
                s_pa2[ty] = paritya2[ia];
            }
        }
        __syncthreads();

        const int ia = ia0 + ty;
        if (ia < nalpha && tx + ib0 < nb) {
            const int sa1 = s_sa1[ty];
            const int ta1 = s_ta1[ty];
            const int sb1 = s_sb1[tx];
            const int tb1 = s_tb1[tx];

            const double pa1 = s_pa1[ty];
            const double pa2 = s_pa2[ty];
            const double pb1 = s_pb1[tx];
            const double pb2 = s_pb2[tx];

            const int idx_u = sa1 * nbeta_strs_ + sb1;
            const int idx_v = ta1 * nbeta_strs_ + tb1;

            const double u0 = dC[idx_u];
            const double v0 = dC[idx_v];

            const double p1 = pa1 * pb1;
            const double p2 = pa2 * pb2;

            const double u_new = factor * u0 + acc2 * (p2 * v0);
            const double v_new = factor * v0 + acc1 * (p1 * u0);

            dC[idx_u] = u_new;
            dC[idx_v] = v_new;
        }
        __syncthreads();
    }
}

// =============================
// Launchers
// =============================
template<int BX>
static void launch_givens_soa(
    double* dC,
    const double* paritya1,
    const double* paritya2,
    const double* parityb1,
    const double* parityb2,
    const int* sourcea1,
    const int* targeta1,
    const int* sourceb1,
    const int* targetb1,
    int nalpha,
    int nb,
    int nbeta_strs_,
    double factor,
    double acc1,
    double acc2)
{
    if (nalpha == 0 || nb == 0 || nbeta_strs_ == 0) return;

    constexpr int AY = 8;
    const int grid_x = (nb + BX - 1) / BX;
    const int grid_y = std::max(1, (nalpha + AY - 1) / AY);
    dim3 block(BX, AY), grid(grid_x, grid_y);

    if (block.x * block.y > 1024)
        throw std::invalid_argument("Block size BX*AY exceeds device limit");

    givens_update_soa_tiled<BX><<<grid, block>>>(
        dC, sourcea1, targeta1, paritya1, paritya2,
        sourceb1, targetb1, parityb1, parityb2,
        nalpha, nb, nbeta_strs_,
        factor, acc1, acc2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "givens_update_soa_tiled<" << BX << "> launch failed: "
                  << cudaGetErrorString(err) << "\n";
        throw std::runtime_error("kernel launch failed");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "givens_update_soa_tiled<" << BX << "> exec failed: "
                  << cudaGetErrorString(err) << "\n";
        throw std::runtime_error("kernel exec failed");
    }
}

// =============================
// Single extern "C" wrapper
// =============================
extern "C" void givens_update_tiled_wrapper_soa(
    int BX_runtime,
    double* dC,
    const double* paritya1,
    const double* paritya2,
    const double* parityb1,
    const double* parityb2,
    const int* sourcea1,
    const int* targeta1,
    const int* sourceb1,
    const int* targetb1,
    int nalpha,
    int nb,
    int nbeta_strs_,
    double factor,
    double acc1,
    double acc2)
{
    if (nalpha == 0 || nb == 0 || nbeta_strs_ == 0) return;

    switch (BX_runtime) {
        case 64:
            launch_givens_soa<64>(
                dC, paritya1, paritya2, parityb1, parityb2,
                sourcea1, targeta1, sourceb1, targetb1,
                nalpha, nb, nbeta_strs_, factor, acc1, acc2
            );
            break;
        case 32: default:
            launch_givens_soa<32>(
                dC, paritya1, paritya2, parityb1, parityb2,
                sourcea1, targeta1, sourceb1, targetb1,
                nalpha, nb, nbeta_strs_, factor, acc1, acc2
            );
            break;
    }
}