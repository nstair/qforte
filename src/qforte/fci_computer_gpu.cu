#include "fci_computer_gpu.cuh"
#include <cuda_runtime.h>
#include <iostream>



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
    int targetb_size) 
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
    
    int blocksPerGrid = (tensor_size + 256 - 1) / 256;
    apply_individual_nbody1_accumulate_kernel<<<blocksPerGrid, 256>>>(
        coeff, 
        d_Cin, 
        d_Cout, 
        d_sourcea, 
        d_targeta, 
        d_paritya, 
        d_sourceb, 
        d_targetb, 
        d_parityb, 
        nbeta_strs_, 
        targeta_size, 
        targetb_size);
   

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
