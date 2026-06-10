#include "tensor_gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdexcept>

// Optimized transpose kernel for double precision
__global__ void transposeCoalescedDouble(double *odata, const double *idata, long long width, long long height)
{
    __shared__ double tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts

    long long x = (long long)blockIdx.x * TILE_DIM + threadIdx.x;
    long long y = (long long)blockIdx.y * TILE_DIM + threadIdx.y;

    // Read from input with coalesced access
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
        }
    }

    __syncthreads();

    // Transpose block offset
    x = (long long)blockIdx.y * TILE_DIM + threadIdx.x;
    y = (long long)blockIdx.x * TILE_DIM + threadIdx.y;

    // Write to output with coalesced access
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Optimized transpose kernel for complex double precision
__global__ void transposeCoalescedComplex(cuDoubleComplex *odata, const cuDoubleComplex *idata, long long width, long long height)
{
    __shared__ cuDoubleComplex tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts

    long long x = (long long)blockIdx.x * TILE_DIM + threadIdx.x;
    long long y = (long long)blockIdx.y * TILE_DIM + threadIdx.y;

    // Read from input with coalesced access
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = cuConj(idata[(y + j) * width + x]);
        }
    }

    __syncthreads();

    // Transpose block offset
    x = (long long)blockIdx.y * TILE_DIM + threadIdx.x;
    y = (long long)blockIdx.x * TILE_DIM + threadIdx.y;

    // Write to output with coalesced access
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Wrapper function for double transpose
void launchTransposeDouble(double *d_out, const double *d_in, long long width, long long height)
{
    dim3 dimGrid((unsigned int)((width + TILE_DIM - 1) / TILE_DIM),
                (unsigned int)((height + TILE_DIM - 1) / TILE_DIM), 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    transposeCoalescedDouble<<<dimGrid, dimBlock>>>(d_out, d_in, width, height);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

// Wrapper function for complex transpose
void launchTransposeComplex(cuDoubleComplex *d_out, const cuDoubleComplex *d_in, long long width, long long height)
{
    dim3 dimGrid((unsigned int)((width + TILE_DIM - 1) / TILE_DIM),
                (unsigned int)((height + TILE_DIM - 1) / TILE_DIM), 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    transposeCoalescedComplex<<<dimGrid, dimBlock>>>(d_out, d_in, width, height);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}
