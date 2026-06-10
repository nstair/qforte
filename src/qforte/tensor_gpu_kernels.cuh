#ifndef _tensor_gpu_kernels_cuh_
#define _tensor_gpu_kernels_cuh_

#include <cuda_runtime.h>
#include <cuComplex.h>

// Tile dimensions for transpose kernel
#define TILE_DIM 32
#define BLOCK_ROWS 8

// Kernel declarations
__global__ void transposeCoalescedDouble(double *odata, const double *idata, long long width, long long height);
__global__ void transposeCoalescedComplex(cuDoubleComplex *odata, const cuDoubleComplex *idata, long long width, long long height);

// Wrapper functions
void launchTransposeDouble(double *d_out, const double *d_in, long long width, long long height);
void launchTransposeComplex(cuDoubleComplex *d_out, const cuDoubleComplex *d_in, long long width, long long height);

#endif // _tensor_gpu_kernels_cuh_
