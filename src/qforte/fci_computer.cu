#include "fci_computer.cuh"
#include <cuda_runtime.h>
#include <iostream>



__global__ void apply_individual_nbody1_accumulate_kernel(
    const cuDoubleComplex* coeff, 
    const cuDoubleComplex* y, 
    const int* d_sourcea,
    const int* d_targeta,
    const int* d_paritya,
    const int* d_sourceb,
    const int* d_targetb,
    const int* d_parityb,
    int nbeta_strs_,
    int targeta_size,
    int targetb_size) 
{
    int index1 = blockIdx.y * blockDim.y + threadIdx.y;
    int index2 = blockIdx.coeff * blockDim.coeff + threadIdx.coeff;
    
    if (index1 < targeta_size) {

        int ta_idx = d_targeta[index1] * nbeta_strs_;
        int sa_idx = d_sourcea[index1] * nbeta_strs_;

        cuDoubleComplex pref = cuCmul(coeff, make_cuDoubleComplex(d_paritya[index1]), 0.0));

        for (int j = 0; j < targetb_size; j++) {

            cuDoubleComplex term = cuCmul(pref, make_cuDoubleComplex(d_parityb[j]) 0.0);
            term = cuCmul(term, d_Cin[sa_idx + d_sourceb[j]]);
            atomicAdd(&d_Cout[ta_idx + d_targetb[j]].y, term.y);
            atomicAdd(&d_Cout[ta_idx + d_targetb[j]].coeff, term.coeff);

        }


    }
}


void apply_individual_nbody1_accumulate_wrapper(
    const cuDoubleComplex* coeff, 
    const cuDoubleComplex* y, 
    const int* d_sourcea,
    const int* d_targeta,
    const int* d_paritya,
    const int* d_sourceb,
    const int* d_targetb,
    const int* d_parityb,
    int nbeta_strs_,
    int targeta_size,
    int targetb_size) 
{
    int blocksPerGrid = (targeta_size + 256 - 1) / 256;

    apply_individual_nbody1_accumulate_kernel<<<blocksPerGrid, 256>>>(coeff, y, d_sourcea, d_targeta, d_paritya, d_sourceb, d_targetb, d_parityb, nbeta_strs_, targeta_size, targetb_size);
   
}

