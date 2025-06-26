#pragma once
#include <complex>
#include <cuComplex.h>
// CUDA kernel for element-wise addition of two tensors
__global__ void add_kernel(std::complex<double>* x, const std::complex<double>* y, size_t n);

__global__ void add_kernel2(cuDoubleComplex* x, const cuDoubleComplex* y, size_t n);

extern "C" void add_wrapper(std::complex<double>* x, const std::complex<double>* y, size_t n, int threadsPerBlock);

extern "C" void add_wrapper2(cuDoubleComplex* x, const cuDoubleComplex* y, int n, int threadsPerBlock);

extern "C" void add_wrapper_thrust(cuDoubleComplex* d_x, const cuDoubleComplex* d_y, size_t n);
