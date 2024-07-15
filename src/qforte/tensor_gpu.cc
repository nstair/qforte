#include "tensor_gpu.h"
#include "blas_math.h"
#include "cuda_runtime.h"
#include "tensor_gpu_kernels.cuh"


// May need an analog these eventually
// #include "../util/string.hpp"
// #include <lightspeed/math.hpp>

#include <iostream>
#include <string>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <utility>
#include <algorithm>

// namespace lightspeed { 

size_t TensorGPU::total_memory__ = 0;

/// Constructor
TensorGPU::TensorGPU(
    const std::vector<size_t>& shape,
    const std::string& name
    ) :
    shape_(shape),
    name_(name)
{
    strides_.resize(shape_.size());
    size_ = 1L;
    for (int i = shape_.size() - 1; i >= 0; i--) {
        strides_[i] = size_;
        size_ *= shape_[i];
    }  
    h_data_.resize(size_,0.0);

    initialized_ = 1;

    // Ed's special memory thing
    total_memory__ += h_data_.size() * sizeof(std::complex<double>);

    // allocate device memory
    cudaMalloc((void**) & d_data_, size_ * sizeof(std::complex<double>));
}

TensorGPU::TensorGPU()
{
    shape_.assign(1, 1);
    strides_.resize(1);
    size_ = 1L;
    h_data_.resize(size_, 0.0);
    total_memory__ += h_data_.size() * sizeof(std::complex<double>);
}

/// Destructor
TensorGPU::~TensorGPU()
{
    // Ed's special memory thing
    total_memory__ -= h_data_.size() * sizeof(std::complex<double>);

    // free the device memory
    cudaFree(d_data_);
}

void TensorGPU::to_gpu()
{

    if (initialized_ == 0) {
        std::cerr << "Tensor not initialized" << std::endl;
        return;
    }

    on_gpu_ = 1;

    cudaError_t error_status = cudaMemcpy(d_data_, h_data_.data(), size_ * sizeof(std::complex<double>), cudaMemcpyHostToDevice);

    if (error_status != cudaSuccess) {
        std::cerr << "Failed to transfer data to GPU. Error msg: " << cudaGetErrorString(error_status) << std::endl;
    }

}

// change to 'to_cpu'
void TensorGPU::from_gpu()
{
    
    if (initialized_ == 0) {
        std::cerr << "Tensor not initialized" << std::endl;
        return;
    }

    on_gpu_ = 0;

    cudaError_t error_status = cudaMemcpy(h_data_.data(), d_data_, size_ * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);

    if (error_status != cudaSuccess) {
        std::cerr << "Failed to transfer data to CPU. Error msg: " << cudaGetErrorString(error_status) << std::endl;
    }

}

void TensorGPU::add(const TensorGPU& other) {

    if (shape_ != other.shape_) {
        throw std::runtime_error("Tensor shapes are not compatible for addition.");
    }

    // add_wrapper(d_data_, other.get_d_data(), size_, 256);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("Failed to execute the add operation on the GPU.");
    }
}

void TensorGPU::add2(const TensorGPU& other) {

    if (shape_ != other.shape_) {
        throw std::runtime_error("Tensor shapes are not compatible for addition.");
    }

    gpu_error();
    other.gpu_error();
    add_wrapper2(d_data_, other.get_d_data(), size_, 256);

    

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("Failed to execute the add operation on the GPU.");
    }
}

void TensorGPU::gpu_error() const {

    if (not on_gpu_) {
        throw std::runtime_error("Data not on GPU for " + name_);
    }

}

void TensorGPU::zero()
{
    memset(h_data_.data(), '\0', sizeof(std::complex<double>) * size_);
}

void TensorGPU::set(
    const std::vector<size_t>& idxs,
    const std::complex<double> val
        )
{
    ndim_error(idxs.size());

    if( idxs.size() == 1 ) {
        h_data_[idxs[0]] = val;
    } else if (idxs.size() == 2) {
        h_data_[shape()[1]*idxs[0] + idxs[1]] = val;
    } else {
        for (int i = 0; i < ndim(); i++) {
            if (idxs[i] < 0 || idxs[i] >= shape()[i]) {
                std::cerr << "Index out of bounds for dimension " << i << std::endl;
            }
        }      
        size_t vidx = 0;
        size_t stride = 1;
        
        for (int i = ndim() - 1; i >= 0; i--) {
            vidx += idxs[i] * stride;
            stride *= shape()[i];
        }
        h_data_[vidx] = val;
    } 
}

void TensorGPU::ndim_error(size_t ndims) const
{
    if (!(ndim() == ndims)) {
        std::stringstream ss;
        ss << "Tensor should be " << ndims << " ndim, but is " << ndim() << " ndim.";
        throw std::runtime_error(ss.str());
    }
}



/*

add private member variable that checks to see if it is on the gpu
functions change this variable

change the name of from_gpu

gpu_error checks that variable

*/