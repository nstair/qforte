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
//     int targetb_size) 
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx < targeta_size) {
//         int ta_idx = d_targeta[idx] * nbeta_strs_;
//         int sa_idx = d_sourcea[idx] * nbeta_strs_;
//         cuDoubleComplex pref = cuCmul(coeff, make_cuDoubleComplex(d_paritya[idx], 0.0));

//         // #pragma unroll
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
    

    // cudaPointerAttributes attributes;
    // cudaError_t err = cudaPointerGetAttributes(&attributes, d_Cin);
    // if (err == cudaSuccess) {
    //     std::cout << "Succcess for Cin\n" << std::endl;
    // } else {
    //     std::cerr << "Invalid pointer or unrecognized memory: Cin" << cudaGetErrorString(err) << std::endl;
    // }

    // err = cudaPointerGetAttributes(&attributes, d_Cout);
    // if (err == cudaSuccess) {
    //     std::cout << "Succcess for Cout\n" << std::endl;
    // } else {
    //     std::cerr << "Invalid pointer or unrecognized memory: Cout" << cudaGetErrorString(err) << std::endl;
    // }

    // err = cudaPointerGetAttributes(&attributes, d_sourcea);
    // if (err == cudaSuccess) {
    //     std::cout << "Succcess for source a\n" << std::endl;
    // } else {
    //     std::cerr << "Invalid pointer or unrecognized memory: source a" << cudaGetErrorString(err) << std::endl;
    // }

    // err = cudaPointerGetAttributes(&attributes, d_sourceb);
    // if (err == cudaSuccess) {
    //     std::cout << "Succcess for source b\n" << std::endl;
    // } else {
    //     std::cerr << "Invalid pointer or unrecognized memory: source b" << cudaGetErrorString(err) << std::endl;
    // }

    // err = cudaPointerGetAttributes(&attributes, d_targeta);
    // if (err == cudaSuccess) {
    //     std::cout << "Succcess for target a\n" << std::endl;
    // } else {
    //     std::cerr << "Invalid pointer or unrecognized memory: target a" << cudaGetErrorString(err) << std::endl;
    // }

    // err = cudaPointerGetAttributes(&attributes, d_targetb);
    // if (err == cudaSuccess) {
    //     std::cout << "Succcess for target b\n" << std::endl;
    // } else {
    //     std::cerr << "Invalid pointer or unrecognized memory: target b" << cudaGetErrorString(err) << std::endl;
    // }

    // err = cudaPointerGetAttributes(&attributes, d_paritya);
    // if (err == cudaSuccess) {
    //     std::cout << "Succcess for parity a\n" << std::endl;
    // } else {
    //     std::cerr << "Invalid pointer or unrecognized memory: parity a" << cudaGetErrorString(err) << std::endl;
    // }

    // err = cudaPointerGetAttributes(&attributes, d_parityb);
    // if (err == cudaSuccess) {
    //     std::cout << "Succcess for parity b\n" << std::endl;
    // } else {
    //     std::cerr << "Invalid pointer or unrecognized memory: parity b" << cudaGetErrorString(err) << std::endl;
    // }

    // int maxThreadsPerBlock;
    // cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    // std::cout << "Max threads per block: " << maxThreadsPerBlock << std::endl;

    
    // int blocksPerGrid = (tensor_size + 256 - 1) / 256;
    int total_threads = targeta_size * targetb_size;
    int blocksPerGrid = (total_threads + 256 - 1) / 256;
    // std::cout << "bpg: " << blocksPerGrid << std::endl;
    // std::cout << "threads: " << total_threads << std::endl;

    // std::cout << "coeff: " << coeff << std::endl;

    // std::cout << "cin: " << d_Cin << std::endl;
    // std::cout << "cout: " << d_Cout << std::endl;
    // std::cout << "src a: " << d_sourcea << std::endl;
    // std::cout << "src b: " << d_sourceb << std::endl;
    // std::cout << "target a: " << d_targeta << std::endl;
    // std::cout << "target b: " << d_targetb << std::endl;
    // std::cout << "parity a: " << d_paritya << std::endl;
    // std::cout << "parity b: " << d_parityb << std::endl;
    // std::cout << "nbeta_strs: " << nbeta_strs_ << std::endl;
    // std::cout << "target a size: " << targeta_size << std::endl;
    // std::cout << "target b size: " << targetb_size << std::endl;




    apply_individual_nbody1_accumulate_kernel<<<256, 256>>>(
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

__global__ void evolve_individual_nbody_easy_kernel(
    cuDoubleComplex* Cout_data,
    cuDoubleComplex factor,
    const int* map_first,
    const int* map_second,
    size_t map_first_size,
    size_t map_second_size,
    size_t nbeta_strs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < map_first_size) {
        if (idy < map_second_size) {
            int index = map_first[idx] * nbeta_strs + map_second[idy];
            Cout_data[index] = cuCmul(Cout_data[index], factor);
        }
    }

}

void evolve_individual_nbody_easy_wrapper(
    cuDoubleComplex* Cout_data,
    const cuDoubleComplex factor,
    const int* map_first,
    const int* map_second,
    int map_first_size,
    int map_second_size,
    int nbeta_strs)
{

    evolve_individual_nbody_easy_kernel<<<256, 256>>>(
        Cout_data,
        factor,
        map_first,
        map_second,
        map_first_size,
        map_second_size,
        nbeta_strs
    );

}

// starting over

// __global__ void apply_individual_nbody1_accumulate_fresh_kernel(
//     const cuDoubleComplex coeff,
//     const cuDoubleComplex* Cin,
//     cuDoubleComplex* Cout,
//     const int* sourcea,
//     const int* targeta,
//     const int* paritya,
//     const int* sourceb,
//     const int* targetb,
//     const int* parityb,
//     int nbeta_strs,
//     int targeta_size,
//     int targetb_size)
// {

//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;

//     // apparently don't need a nested if statement
//     if (idx < targeta_size && idy < targetb_size){
//         int ta_idx = targeta[idx] * nbeta_strs;
//         int sa_idx = sourcea[idx] * nbeta_strs;

//         cuDoubleComplex pref = cuCmul(coeff, make_cuDoubleComplex((double)paritya[idx], 0.0));

//         // atomicAdd(
//         //     &Cout[ta_idx + targetb[idy]].x,
//         //     pref.x * parityb[idy] * Cin[sa_idx + sourceb[idy]].x - 
//         //     pref.y * parityb[idy] * Cin[sa_idx + sourceb[idy]].y
//         // );

//             term = cuCmul(term, d_Cin[sa_idx + d_sourceb[idy]]);


//         Cout[ta_idx + targetb[idy]].x = cuCadd(Cout[ta_idx + targetb[idy]].x, )

//         atomicAdd(
//             &Cout[ta_idx + targetb[idy]].y,
//             pref.x * parityb[idy] * Cin[sa_idx + sourceb[idy]].y + 
//             pref.y * parityb[idy] * Cin[sa_idx + sourceb[idy]].x
//         );

//     }

// }

// void apply_individual_nbody1_accumulate_fresh_wrapper(
//     const cuDoubleComplex coeff,
//     const cuDoubleComplex* Cin,
//     cuDoubleComplex* Cout,
//     const int* sourcea,
//     const int* targeta,
//     const int* paritya,
//     const int* sourceb,
//     const int* targetb,
//     const int* parityb,
//     int nbeta_strs,
//     int targeta_size,
//     int targetb_size)
// {

//     apply_individual_nbody1_accumulate_fresh_kernel<<<256, 256>>>(
//         coeff,
//         Cin,
//         Cout,
//         sourcea,
//         targeta,
//         paritya,
//         sourceb,
//         targetb,
//         parityb,
//         nbeta_strs,
//         targeta_size,
//         targetb_size
//     );

// }