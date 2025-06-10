// #include "tensor_gpu.h"
#include "blas_math.h"
#include "cuda_runtime.h"
// #include "tensor.h"
#include "tensor_gpu.cuh"
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
    const std::string& name,
    bool on_gpu) :
    shape_(shape),
    name_(name),
    on_gpu_(on_gpu)
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

    // size_t freeMem, totalMem;
    // cudaMemGetInfo(&freeMem, &totalMem);
    // std::cout << "Free = " << freeMem << " / " << totalMem << std::endl;



    // allocate device memory
    // cudaError_t err = cudaMalloc((void**) & d_data_, size_ * sizeof(std::complex<double>));
    cudaError_t err = cudaMalloc(&d_data_, size_ * sizeof(cuDoubleComplex));
    // std::cout << "created " << name_ << " at " << this << " with size of " << size_ * sizeof(cuDoubleComplex) << " and " << &d_data_ << std::endl;

    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed\n" << std::endl;
    }
    

}

TensorGPU::TensorGPU()
{
    shape_.assign(1, 1);
    strides_.resize(1);
    size_ = 1L;
    h_data_.resize(size_, 0.0);
    total_memory__ += h_data_.size() * sizeof(std::complex<double>);
    on_gpu_ = false;

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    // std::cout << "Free = " << freeMem << " / " << totalMem << std::endl;


    // allocate device memory
    // cudaError_t err = cudaMalloc((void**) & d_data_, size_ * sizeof(std::complex<double>));
    cudaError_t err = cudaMalloc(&d_data_, size_ * sizeof(cuDoubleComplex));
    // std::cout << "created " << name_ << " at " << this << " with size of " << size_ * sizeof(cuDoubleComplex) << " and " << d_data_ << std::endl;
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed\n" << std::endl;
    }
}

/// Destructor
TensorGPU::~TensorGPU()
{
    // std::cout << "destroying: " << name_ << " at " << this << " and " << d_data_ << std::endl;
    cudaDeviceSynchronize();
    // Ed's special memory thing
    total_memory__ -= h_data_.size() * sizeof(std::complex<double>);

    // free the device memory

    cudaFree(d_data_);
}


void TensorGPU::to_gpu()
{
    cpu_error();
    if (initialized_ == 0) {
        std::cerr << "Tensor not initialized" << std::endl;
        return;
    }

    on_gpu_ = 1;

    cudaError_t error_status = cudaMemcpy(d_data_, h_data_.data(), size_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    if (error_status != cudaSuccess) {
        std::cerr << "Failed to transfer data to GPU. Error msg: " << cudaGetErrorString(error_status) << std::endl;
    }

}

// change to 'to_cpu'
void TensorGPU::to_cpu()
{
    gpu_error();
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
    add_wrapper2(d_data_, other.read_d_data(), size_, 256);

    

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("Failed to execute the add operation on the GPU.");
    }
}

void TensorGPU::gpu_error() const {

    if (not on_gpu_) {
        throw std::runtime_error("Data not on GPU for Tensor" + name_);
    }

}

void TensorGPU::cpu_error() const {

    if (on_gpu_) {
        throw std::runtime_error("Data not on CPU for Tensor" + name_);
    }

}

void TensorGPU::zero()
{
    cpu_error();
    memset(h_data_.data(), '\0', sizeof(std::complex<double>) * size_);
}

void TensorGPU::zero_gpu()
{
    gpu_error();
    cudaError_t err = cudaMemset(d_data_, '\0', sizeof(cuDoubleComplex) * size_);

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(err));
    }
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

void TensorGPU::fill_from_nparray(std::vector<std::complex<double>> arr, std::vector<size_t> shape)
{

    cpu_error();

    if (shape_ != shape){
        throw std::runtime_error("The shapes are not the same.");
    }

    std::memcpy(h_data_.data(), arr.data(), sizeof(std::complex<double>) * size_);

}

double TensorGPU::norm(){

    double result = 0;

    for (int i = 0; i < size_; i++){

        result += std::real(h_data_[i]) * std::real(h_data_[i]) + std::imag(h_data_[i]) * std::imag(h_data_[i]);

    }

    result = std::sqrt(result);

    return result;

}



/*

add private member variable that checks to see if it is on the gpu
functions change this variable

change the name of from_gpu

gpu_error checks that variable

*/


void TensorGPU::add_to_element(
    const std::vector<size_t>& idxs,
    const std::complex<double> val
        )
{
    ndim_error(idxs.size());

    if( idxs.size() == 1 ) {
        h_data_[idxs[0]] += val;
    } else if (idxs.size() == 2) {
        h_data_[shape()[1]*idxs[0] + idxs[1]] += val;
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
        h_data_[vidx] = +val;
    } 
}

void TensorGPU::fill_from_np(std::vector<std::complex<double>> arr, std::vector<size_t> shape){
    if (shape_ != shape){
        throw std::runtime_error("The Shapes are not the same.");
    }
    std::memcpy(h_data_.data(), arr.data(), sizeof(std::complex<double>)*size_);
}

void TensorGPU::zero_with_shape(const std::vector<size_t>& shape, bool on_gpu)
{
    std::vector<size_t> strides;
    strides.resize(shape.size());
    size_t size = 1L;

    for (int i = shape.size() - 1; i >= 0; i--) {
        strides[i] = size;
        size *= shape[i];
    }  

    shape_ = shape;
    strides_ = strides;
    size_ = size;
    h_data_.resize(size_, 0.0);
    memset(h_data_.data(),'\0',sizeof(std::complex<double>)*size_);

    initialized_ = 1;

    // Ed's special memory thing
    total_memory__ = h_data_.size() * sizeof(std::complex<double>);

    cudaError_t err1 = cudaFree(d_data_);
    if (err1 != cudaSuccess) {
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err1));
        return;
    }

    // Allocate new memory
    cudaError_t err2 = cudaMalloc((void**) & d_data_, size_ * sizeof(cuDoubleComplex));
    if (err2 != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err2));
        return;
    }

    cudaError_t err3 = cudaMemset(d_data_, '\0', sizeof(cuDoubleComplex) * size_);

    if (err3 != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(err3));
    }

    on_gpu_ = on_gpu;
}



/// Get the vector index for this tensor based on the tensor index
size_t TensorGPU::tidx_to_vidx(const std::vector<size_t>& tidx) const
{   
    size_t vidx = 0;
    for (int i = ndim() - 1; i >= 0; i--) {
        vidx += tidx[i] * strides_[i];
    }
    return vidx;
}

size_t TensorGPU::tidx_to_trans_vidx(const std::vector<size_t>& tidx, const std::vector<size_t>& axes) const
{   
    size_t vidx = 0;
    for (int i = ndim() - 1; i >= 0; i--) {
        vidx += tidx[i] * strides_[axes[i]];
    }
    return vidx;
}

/// Get the tensor index for this tensor based on the vector index
std::vector<size_t> TensorGPU::vidx_to_tidx(size_t vidx) const
{
    std::vector<size_t> tidx(ndim());
    size_t vidx_tmp = vidx;

    for (int i = ndim() - 1; i >= 0; i--) {
        tidx[i] = vidx_tmp % shape()[i];
        vidx_tmp /= shape()[i];
    }
    return tidx;
}

/// Get a particular element of tis TensorGPU, specified by idxs
// TODO(Nick/Tyler) use strides_
std::complex<double> TensorGPU::get(
    const std::vector<size_t>& idxs
    ) const
{
    ndim_error(idxs.size());

    if( idxs.size() == 1 ) {
        return h_data_[idxs[0]];
    } else if (idxs.size() == 2) {
        return h_data_[shape()[1]*idxs[0] + idxs[1]];
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

        return h_data_[vidx];
    }
}

void TensorGPU::shape_error(const std::vector<size_t>& shape) const
{
    ndim_error(shape.size());
    for (size_t i = 0; i < ndim(); i++) {
        if (shape_[i] != shape[i]) {
            std::stringstream ss;
            ss << "TensorGPU should be (";
            for (size_t j = 0; j < ndim(); j++) {
                ss << shape[j];
                if (j < ndim() - 1) {
                    ss << ",";
                }
            }
            ss << ") shape, but is (";
            for (size_t j = 0; j < ndim(); j++) {
                ss << shape_[j];
                if (j < ndim() - 1) {
                    ss << ",";
                }
            }
            ss << ") shape.";
            throw std::runtime_error(ss.str());
        }
    }
}

void TensorGPU::square_error() const 
{
    ndim_error(2);
    if (shape_[0] != shape_[1]) {
        std::stringstream ss;
        ss << "TensorGPU should be square, but is ";
        ss << "(" << shape_[0] << "," << shape_[1] << ") shape.";
        throw std::runtime_error(ss.str());
    }
}

std::shared_ptr<TensorGPU> TensorGPU::clone()
{
    return std::shared_ptr<TensorGPU>(new TensorGPU(*this)); 
}

void TensorGPU::identity()
{
    square_error();
    zero();
    for (size_t i = 0; i < shape_[0]; i++) {
        h_data_[i * shape_[1] + i] = 1.0;
    }
}

void TensorGPU::symmetrize()
{
    square_error();
    for (size_t i = 0; i < shape_[0]; i++) {
        for (size_t j = 0; j < shape_[0]; j++) {
            h_data_[i * shape_[1] + j] =
            h_data_[j * shape_[1] + i] = 0.5 * (
            h_data_[i * shape_[1] + j] +
            h_data_[j * shape_[1] + i]);
        }
    }
}

void TensorGPU::antisymmetrize()
{
    square_error();
    for (size_t i = 0; i < shape_[0]; i++) {
        for (size_t j = 0; j < shape_[0]; j++) {
            std::complex<double> val = 0.5 * (
            h_data_[i * shape_[1] + j] -
            h_data_[j * shape_[1] + i]);
            h_data_[i * shape_[1] + j] = val;
            h_data_[j * shape_[1] + i] = - val;
        }
    }
}

// TODO(NICK:) reimplement Scal
// void TensorGPU::scale(std::complex<double> a)
// {
//     // C_DSCAL(size_,a,h_data_.data(),1);
//     for(size_t i = 0; i < size_; i++){
//         h_data_[i] *= a;
//     }
// }

void TensorGPU::scale(std::complex<double> a)
{
    math_zscale(size_, a, h_data_.data(), 1);
}

void TensorGPU::copy_in(
    const TensorGPU& other
    )
{
    cpu_error();
    other.cpu_error();
    shape_error(other.shape());
    std::memcpy(h_data_.data(), other.read_h_data().data(), sizeof(cuDoubleComplex)*size_);
}

void TensorGPU::copy_in_gpu(const TensorGPU& other)
{
    gpu_error();
    other.gpu_error();
    shape_error(other.shape());
    cudaError_t err = cudaMemcpy(d_data_, other.read_d_data(), size_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    cudaMemset(d_data_, 0, size_ * sizeof(std::complex<double>));
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed\n" << std::endl;
    }

}

void TensorGPU::copy_in_from_tensor(const Tensor& other)
{
    cpu_error();
    shape_error(other.shape());
    std::memcpy(h_data_.data(), other.read_data().data(), sizeof(cuDoubleComplex)*size_);    
}

void TensorGPU::subtract(const TensorGPU& other){

    shape_error(other.shape());
    for (size_t i = 0; i < size_; i++){
        h_data_[i] -= other.read_h_data()[i];
    }
}

// void TensorGPU::axpby(

// void TensorGPU::zaxpby(

//     const std::shared_ptr<Tensor>& other,
//     std::complex<double> a,
//     std::complex<double> b
//     )
// {
//     shape_error(other->shape());
    
//     C_DSCAL(size_,b,h_data_.data(),1);
//     C_DAXPY(size_,a,other->data().data(),1,h_data_.data(),1); 
// } OLD

void TensorGPU::zaxpby(
    const TensorGPU& x,
    std::complex<double> a,
    std::complex<double> b,
    const int incx,
    const int incy
    )
{
    shape_error(x.shape());
    math_zscale(size_, b, h_data_.data(),1);
    math_zaxpy(size_, a, x.read_h_data().data(), incx, h_data_.data(), incy); 
}

void TensorGPU::zaxpy(
    const TensorGPU& x, 
    const std::complex<double> alpha, 
    const int incx, 
    const int incy)
{
    // Check if the two tensors have compatible shapes
    if (shape_ != x.shape()) {
        throw std::runtime_error("TensorGPU::zaxpy: Incompatible tensor shapes for axpy operation.");
    }

    // Get the raw data pointers for both tensors
    const std::complex<double>* x_data = x.read_h_data().data();
    std::complex<double>* y_data = h_data_.data();

    // Call the zaxpy function from blas_math.h to perform the operation
    math_zaxpy(size_, alpha, x_data, incx, y_data, incy);
}


void TensorGPU::gemm(
    const TensorGPU& B,
    const char transa,
    const char transb,
    const std::complex<double> alpha,
    const std::complex<double> beta,
    const bool mult_B_on_right)
{

    if ((shape_.size() != 2) || (shape_ != B.shape())){
        throw std::runtime_error("Incompatible shape(s)/dimension(s).");
    }

  const int M = (transa == 'N') ? shape_[0] : shape_[1];
  const int N = (transb == 'N') ? B.shape()[1] : B.shape()[0];
  const int K = (transa == 'N') ? shape_[1] : shape_[0];
  
  std::complex<double>* A_data = h_data_.data();
  const std::complex<double>* B_data = B.read_h_data().data();
  
  // Since TensorGPU C is 'this' TensorGPU
  std::complex<double>* C_data = h_data_.data();

  if(mult_B_on_right) {
    math_zgemm(transa, transb, M, N, K, alpha, A_data, shape_[1], B_data, B.shape()[1], beta, C_data, shape_[1]);
  } 
  else {
    math_zgemm(transb, transa, N, M, K, alpha, B_data, B.shape()[1], A_data, shape_[1], beta, C_data, shape_[1]);
  }

}


std::complex<double> TensorGPU::vector_dot(
    const TensorGPU& other
    ) const
{
    shape_error(other.shape());

    // return math_zdot(
    //     size_, 
    //     const_cast<std::complex<double>*>(h_data_.data()), 
    //     1, 
    //     other.read_h_data().data(), 
    //     1);

    std::complex<double> result = 0.0;

    for (int i = 0; i < size_; i++){
        result += std::conj(h_data_[i]) * other.read_h_data()[i];
    }

    return result;

    // return math_zdot(
    //     size_, 
    //     h_data_.data(), 
    //     1, 
    //     other.read_h_data().data(), 
    //     1);
    
}

// NOTE(Nick) we maywant to return sharred pointer to a tensor instead...
// std::shared_pointer<TensorGPU> TensorGPU::transpose() const
TensorGPU TensorGPU::transpose() const
{
    ndim_error(2);
    // std::shared_ptr<TensorGPU> T(new TensorGPU({shape_[1], shape_[0]}));
    TensorGPU T({shape_[1], shape_[0]});
    std::complex<double>* Tp = T.data().data();
    const std::complex<double>* Ap = h_data_.data();
    for (size_t ind1 = 0; ind1 < shape_[0]; ind1++) {
        for (size_t ind2 = 0; ind2 < shape_[1]; ind2++) {
            Tp[ind2 * shape_[0] + ind1] = Ap[ind1 * shape_[1] + ind2];
        }
    }
    return T;
}

// NOTE(Nick) we maywant to return sharred pointer to a tensor instead...
TensorGPU TensorGPU::general_transpose(const std::vector<size_t>& axes) const 
{
    if (axes.size() != ndim()) {
        throw std::invalid_argument("Invalid axes permutation");
    }

    std::vector<size_t> transposed_shape(ndim());
    for (size_t i = 0; i < ndim(); ++i) {
        transposed_shape[i] = shape_[axes[i]];
    }

    // std::shared_ptr<TensorGPU> transposed_tensor(new TensorGPU(transposed_shape));
    TensorGPU transposed_tensor(transposed_shape);

    std::complex<double>* transposed_data = transposed_tensor.data().data();
    // const std::complex<double>* original_data = h_data_.data();

    // This works but probably can be made more efficient.
    // Fix if it turns out to be a bottleneck
    for (size_t i = 0; i < size_; i++){
        std::vector<size_t> tidx_trans = vidx_to_tidx(i);
        size_t t_vidx = transposed_tensor.tidx_to_trans_vidx(tidx_trans, axes);
        transposed_data[t_vidx] = h_data_[i];
    }

    return transposed_tensor;  
}


TensorGPU TensorGPU::slice(std::vector<std::pair<size_t, size_t>> idxs)const{

    std::vector<size_t> new_shape(idxs.size());
    std::vector<size_t> new_shape2;
    // std::vector<size_t> new_strides(idxs.size());
    size_t new_size = 1;

    if (idxs.size() != ndim()){
        throw std::invalid_argument("Number of slices should match the number of dimensions.");
    }


    for (int i = idxs.size() - 1; i >= 0; i--){
        if (idxs[i].first >= idxs[i].second || idxs[i].second > shape_[i]){
            // throw std::invalid_argument("Invalid slice index.");
            std::cout << " maybe invalid slice index? " << std::endl;
        }

        new_shape[i] = idxs[i].second - idxs[i].first;
        // new_strides[i] = strides_[i];
        new_size *= new_shape[i];

    }

    for(size_t dim : new_shape){ 
        if(dim != 1) { new_shape2.push_back(dim); }
    }

    TensorGPU new_tensor(new_shape2, name_ + "_sliced");

    std::vector<size_t> old_tidx(ndim());
    std::fill(old_tidx.begin(), old_tidx.end(), 0);
    std::vector<size_t> new_tidx(new_shape.size(), 0);

    for (size_t vidx = 0; vidx < size_; vidx++){
        old_tidx = vidx_to_tidx(vidx);
        bool is_in_slice = true;

        for (size_t i = 0; i < old_tidx.size(); i++){
            if (old_tidx[i] < idxs[i].first || old_tidx[i] >= idxs[i].second){
                is_in_slice = false;
                break;
            }
            else{
                new_tidx[i] = old_tidx[i] - idxs[i].first;
            }
        }

        if (is_in_slice){
            size_t new_vidx = new_tensor.tidx_to_vidx(new_tidx);
            new_tensor.data()[new_vidx] = h_data_[vidx];
        }

    }


    return new_tensor;
}

std::vector<std::vector<size_t>> TensorGPU::get_nonzero_tidxs() const 
{   
    std::vector<std::vector<size_t>> nonzero_tidxs;
    for(size_t i = 0; i < size_; i++) {
        if(h_data_[i] != 0.0){
            std::vector<size_t> tidxs = vidx_to_tidx(i);
            nonzero_tidxs.push_back(tidxs);
        }
    }
    return nonzero_tidxs;
}

// TODO(Tyler?): Column printing is a little clunky for complex
// need to fix
std::string TensorGPU::str(
    bool print_data, 
    bool print_complex, 
    int maxcols,
    const std::string& data_format,
    const std::string& header_format
    ) const
{
    cpu_error();
    std::string str = "";
    str += std::printf( "TensorGPU: %s\n", name_.c_str());
    str += std::printf( "  Ndim  = %zu\n", ndim());
    str += std::printf( "  Size  = %zu\n", size());
    str += std::printf( "  Shape = (");
    for (size_t dim = 0; dim < ndim(); dim++) {
        str += std::printf( "%zu", shape_[dim]);
        if (dim < ndim() - 1) {
            str += std::printf( ",");
        }
    }
    str += std::printf(")\n");

    if (print_data) {

        std::string data_format2 = data_format;

        if(print_complex){ data_format2 = "%f%+fi";}

        str += std::printf("\n");
            
        std::string order0str1 = "  " + data_format2 + "\n";
        std::string order1str1 = "  %5zu " + data_format2 + "\n";
        std::string order2str1 = " " + header_format;
        std::string order2str2 = " " + data_format2;

        int order = ndim();
        size_t nelem = size();

        size_t page_size = 1L;
        size_t rows = 1;
        size_t cols = 1;
        if (order >= 1) {
            page_size *= shape_[order - 1];
            rows = shape_[order - 1];
        }
        if (order >= 2) {
            page_size *= shape_[order - 2];
            rows = shape_[order - 2];
            cols = shape_[order - 1];
        }

        str += std::printf( "  Data:\n\n");

        if (nelem > 0){
            size_t pages = nelem / page_size;
            for (size_t page = 0L; page < pages; page++) {

                if (order > 2) {
                    str += std::printf( "  Page (");
                    size_t num = page;
                    size_t den = pages;
                    size_t val;
                    for (int k = 0; k < order - 2; k++) {
                        den /= shape_[k];
                        val = num / den;
                        num -= val * den;
                        str += std::printf("%zu,",val);
                    }
                    str += std::printf( "*,*):\n\n");
                }

                const std::complex<double>* vp = h_data_.data() + page * page_size;
                if (order == 0) {
                    str += std::printf( order0str1.c_str(), *(vp));
                } else if(order == 1) {
                    for (size_t i=0; i<page_size; ++i) {
                        str += std::printf( order1str1.c_str(), i, *(vp + i));
                    }
                } else {
                    for (size_t j = 0; j < cols; j += maxcols) {
                        size_t ncols = (j + maxcols >= cols ? cols - j : maxcols);
                
                        // Column Header
                        str += std::printf("  %5s", "");
                        for (size_t jj = j; jj < j+ncols; jj++) {
                            str += std::printf(order2str1.c_str(), jj);
                        }
                        str += std::printf("\n");

                        // Data
                        for (size_t i = 0; i < rows; i++) {
                            str += std::printf("  %5zu", i);
                            for (size_t jj = j; jj < j+ncols; jj++) {
                                str += std::printf(order2str2.c_str(), *(vp + i * cols + jj));
                            }
                            str += std::printf("\n");
                        }

                        // Block separator
                        if (page < pages - 1 || j + maxcols < cols - 1) str += std::printf("\n");
                    }
                }
            }
        }
    }
    return str;
}

std::string TensorGPU::print_nonzero() const 
{   
    std::string str = "\n Nonzero indices and elements of TensorGPU: \n";
    str += " ========================================== ";

    for(size_t i = 0; i < size_; i++) {
        if(h_data_[i] != 0.0){
            std::vector<size_t> tidxs = vidx_to_tidx(i);
            std::stringstream tidxs_stream;
            std::copy(
                tidxs.begin(), 
                tidxs.end(), 
                std::ostream_iterator<size_t>(tidxs_stream, " "));
            
            std::string tidxs_string = tidxs_stream.str();
    
            str += "\n  ( ";
            str += tidxs_string;
            str += ")  ";
            str += std::to_string(h_data_[i].real());
            str += " + ";
            str += std::to_string(h_data_[i].imag());
        }
    }
    return str;
}
// py::array_t<double> array
// std::vector<std::complex<double>>


// TODO(Nick): Re-Implement
// void Tensor::print() const
// {
//     std::cout << string() << std::endl;
// }

// void Tensor::print(const std::string& name)
// {
//     std::string bak_name = name_;
//     set_name(name);
//     print();
//     set_name(bak_name);
// }

// } // namespace lightspeed