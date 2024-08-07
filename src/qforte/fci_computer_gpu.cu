#include "fci_computer_gpu.cuh"
#include <cuda_runtime.h>
#include <iostream>


// __device__ double2 cuDoubleComplexToDouble2(cuDoubleComplex z) {
//     double2 d;
//     d.x = cuCreal(z);
//     d.y = cuCimag(z);
//     return d;
// }

// __device__ cuDoubleComplex double2ToCuDoubleComplex(double2 d) {
//     return make_cuDoubleComplex(d.x, d.y);
// }

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

//         for (int j = 0; j < targetb_size; ++j) {
//             cuDoubleComplex term = cuCmul(pref, make_cuDoubleComplex(d_parityb[j], 0.0));
//             term = cuCmul(term, d_Cin[sa_idx + d_sourceb[j]]);

//             double2 term_double2 = cuDoubleComplexToDouble2(term);
//             double2* d_Cout_double2 = reinterpret_cast<double2*>(&d_Cout[ta_idx + d_targetb[j]]);

//             atomicAdd(&d_Cout_double2->x, term_double2.x);
//             atomicAdd(&d_Cout_double2->y, term_double2.y);
//         }
//     }
// }


// CUDA kernel
__global__ void apply_individual_nbody1_accumulate_kernel(
    const cuDoubleComplex coeff, 
    const cuDoubleComplex* d_Cin, 
    cuDoubleComplex* d_Cout, 
    const int* d_sourcea,
    const int* d_targeta,
    const int* d_paritya,
    const int* d_sourceb,
    const int* d_targetb,
    const int* d_parityb,
    int nbeta_strs,
    int targeta_size,
    int targetb_size,
    int tensor_size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < targeta_size) {

        int ta_idx = d_targeta[idx] * nbeta_strs;

        int sa_idx = d_sourcea[idx] * nbeta_strs;

        cuDoubleComplex pref = cuCmul(coeff, make_cuDoubleComplex(d_paritya[idx], 0.0));

        for (int j = 0; j < targetb_size; ++j) {

            cuDoubleComplex term = cuCmul(pref, make_cuDoubleComplex(d_parityb[j], 0.0));

            term = cuCmul(term, d_Cin[sa_idx + d_sourceb[j]]);

            // atomicAdd(&d_Cout[ta_idx + d_targetb[j]], term);
            d_Cout[ta_idx + d_targetb[j]] = cuCadd(term,  d_Cout[ta_idx + d_targetb[j]]);

        }
    }
}

void apply_individual_nbody1_accumulate_wrapper(
    const cuDoubleComplex coeff, 
    const cuDoubleComplex* d_Cin, 
    cuDoubleComplex* d_Cout, 
    const int* d_sourcea,
    const int* d_targeta,
    const int* d_paritya,
    const int* d_sourceb,
    const int* d_targetb,
    const int* d_parityb,
    int nbeta_strs,
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
        nbeta_strs, 
        targeta_size, 
        targetb_size, 
        tensor_size);
   

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
