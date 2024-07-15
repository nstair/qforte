#include "tensor_gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>


// Kernel for adding two tensors
__global__ void add_kernel(cuDoubleComplex* x, const cuDoubleComplex* y, size_t n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        //x[index] = x[index] + y[index];
        x[index] = cuCadd(x[index], y[index]);
    }
}

__global__ void add_kernel2(cuDoubleComplex* x, const cuDoubleComplex* y, size_t n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        //x[index] = x[index] + y[index];
        x[index] = cuCadd(x[index], y[index]);

    }
}

void add_wrapper(std::complex<double>* x, const std::complex<double>* y, size_t n, int threadsPerBlock) {
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cuDoubleComplex *d_x, *d_y;
    cudaMalloc(&d_x, n * sizeof(cuDoubleComplex));
    cudaMalloc(&d_y, n * sizeof(cuDoubleComplex));

    cuDoubleComplex *h_x = new cuDoubleComplex[n];
    cuDoubleComplex *h_y = new cuDoubleComplex[n];
    for (size_t i = 0; i < n; ++i) {
        h_x[i] = make_cuDoubleComplex(x[i].real(), x[i].imag());
        h_y[i] = make_cuDoubleComplex(y[i].real(), y[i].imag());
    }

    cudaMemcpy(d_x, h_x, n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, n);

    cudaMemcpy(h_x, d_x, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::complex<double>(cuCreal(h_x[i]), cuCimag(h_x[i]));
    }

    cudaFree(d_x);
    cudaFree(d_y);
    delete[] h_x;
    delete[] h_y;
}


void add_wrapper2(cuDoubleComplex* d_x, const cuDoubleComplex* d_y, int n, int threadsPerBlock) {
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // cuDoubleComplex val1 = make_cuDoubleComplex(d_x.real(), d_x.imag());
    // cuDoubleComplex val2 = make_cuDoubleComplex(d_y.real(), d_y.imag());

    // add_kernel2<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, n);

    add_kernel2<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, n);
   
}

