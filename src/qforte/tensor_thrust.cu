#include "tensor_thrust.h"
#include "blas_math.h"
#include "cuda_runtime.h"
#include "tensor.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <string>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <utility>
#include <algorithm>

size_t TensorThrust::total_memory__ = 0;

// Custom functors for thrust operations
struct complex_add {
    __host__ __device__
    cuDoubleComplex operator()(const cuDoubleComplex& a, const cuDoubleComplex& b) const {
        return cuCadd(a, b);
    }
};

struct complex_subtract {
    __host__ __device__
    cuDoubleComplex operator()(const cuDoubleComplex& a, const cuDoubleComplex& b) const {
        return cuCsub(a, b);
    }
};

struct complex_scale {
    cuDoubleComplex scalar;
    
    complex_scale(cuDoubleComplex _scalar) : scalar(_scalar) {}
    
    __host__ __device__
    cuDoubleComplex operator()(const cuDoubleComplex& x) const {
        return cuCmul(scalar, x);
    }
};

struct complex_axpy {
    cuDoubleComplex alpha;
    
    complex_axpy(cuDoubleComplex _alpha) : alpha(_alpha) {}
    
    __host__ __device__
    cuDoubleComplex operator()(const cuDoubleComplex& x, const cuDoubleComplex& y) const {
        return cuCadd(cuCmul(alpha, x), y);
    }
};

struct complex_axpby {
    cuDoubleComplex alpha;
    cuDoubleComplex beta;
    
    complex_axpby(cuDoubleComplex _alpha, cuDoubleComplex _beta) : alpha(_alpha), beta(_beta) {}
    
    __host__ __device__
    cuDoubleComplex operator()(const cuDoubleComplex& x, const cuDoubleComplex& y) const {
        return cuCadd(cuCmul(alpha, x), cuCmul(beta, y));
    }
};

struct complex_norm_squared {
    __host__ __device__
    double operator()(const cuDoubleComplex& x) const {
        return cuCreal(x) * cuCreal(x) + cuCimag(x) * cuCimag(x);
    }
};

struct complex_dot_product {
    __host__ __device__
    cuDoubleComplex operator()(const cuDoubleComplex& a, const cuDoubleComplex& b) const {
        return cuCmul(cuConj(a), b);
    }
};

// Device functors for type conversion - Host side only
struct host_to_device_complex {
    __host__
    cuDoubleComplex operator()(const std::complex<double>& z) const {
        return make_cuDoubleComplex(z.real(), z.imag());
    }
};

struct device_to_host_complex {
    __host__
    std::complex<double> operator()(const cuDoubleComplex& z) const {
        return std::complex<double>(cuCreal(z), cuCimag(z));
    }
};

// Helper function to convert std::complex<double> to cuDoubleComplex
__host__ cuDoubleComplex to_cuDoubleComplex(const std::complex<double>& z) {
    return make_cuDoubleComplex(z.real(), z.imag());
}

// Helper function to convert cuDoubleComplex to std::complex<double>
__host__ std::complex<double> from_cuDoubleComplex(const cuDoubleComplex& z) {
    return std::complex<double>(cuCreal(z), cuCimag(z));
}

/// Constructor
TensorThrust::TensorThrust(
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
    
    // Initialize host vector
    h_data_.resize(size_, std::complex<double>(0.0, 0.0));
    
    // Initialize device vector
    d_data_.resize(size_, make_cuDoubleComplex(0.0, 0.0));
    
    initialized_ = true;
    
    // Ed's special memory thing
    total_memory__ += h_data_.size() * sizeof(std::complex<double>);
}

TensorThrust::TensorThrust()
{
    shape_.assign(1, 1);
    strides_.resize(1);
    size_ = 1L;
    
    // Initialize host vector
    h_data_.resize(size_, std::complex<double>(0.0, 0.0));
    
    // Initialize device vector
    d_data_.resize(size_, make_cuDoubleComplex(0.0, 0.0));
    
    total_memory__ += h_data_.size() * sizeof(std::complex<double>);
    on_gpu_ = false;
    initialized_ = true;
}

/// Destructor
TensorThrust::~TensorThrust()
{
    // Ed's special memory thing
    total_memory__ -= h_data_.size() * sizeof(std::complex<double>);
    
    // Thrust vectors automatically handle cleanup
}

void TensorThrust::to_gpu()
{
    cpu_error();
    if (initialized_ == 0) {
        std::cerr << "Tensor not initialized" << std::endl;
        return;
    }
    
    on_gpu_ = true;
    
    // Manual conversion on CPU then copy to GPU
    std::vector<cuDoubleComplex> temp_gpu_data(size_);
    for (size_t i = 0; i < size_; i++) {
        temp_gpu_data[i] = make_cuDoubleComplex(h_data_[i].real(), h_data_[i].imag());
    }
    
    // Copy converted data to device
    thrust::copy(temp_gpu_data.begin(), temp_gpu_data.end(), d_data_.begin());
}

void TensorThrust::to_cpu()
{
    gpu_error();
    if (initialized_ == 0) {
        std::cerr << "Tensor not initialized" << std::endl;
        return;
    }
    
    on_gpu_ = false;
    
    // Copy from device to temporary CPU vector
    std::vector<cuDoubleComplex> temp_cpu_data(size_);
    thrust::copy(d_data_.begin(), d_data_.end(), temp_cpu_data.begin());
    
    // Manual conversion on CPU
    for (size_t i = 0; i < size_; i++) {
        h_data_[i] = std::complex<double>(cuCreal(temp_cpu_data[i]), cuCimag(temp_cpu_data[i]));
    }
}

void TensorThrust::add(const TensorThrust& other) {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Tensor shapes are not compatible for addition.");
    }
    
    if (on_gpu_) {
        gpu_error();
        other.gpu_error();
        thrust::transform(thrust::device, d_data_.begin(), d_data_.end(), 
                         other.d_data_.begin(), d_data_.begin(), complex_add());
    } else {
        cpu_error();
        other.cpu_error();
        for (size_t i = 0; i < size_; i++) {
            h_data_[i] += other.h_data_[i];
        }
    }
}

void TensorThrust::add_thrust(const TensorThrust& other) {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Tensor shapes are not compatible for addition.");
    }
    
    gpu_error();
    other.gpu_error();
    
    thrust::transform(thrust::device, d_data_.begin(), d_data_.end(), 
                     other.d_data_.begin(), d_data_.begin(), complex_add());
}

void TensorThrust::gpu_error() const {
    if (not on_gpu_) {
        throw std::runtime_error("Tensor is not on GPU but GPU operation was requested.");
    }
}

void TensorThrust::cpu_error() const {
    if (on_gpu_) {
        throw std::runtime_error("Tensor is on GPU but CPU operation was requested.");
    }
}

void TensorThrust::zero()
{
    cpu_error();
    thrust::fill(h_data_.begin(), h_data_.end(), std::complex<double>(0.0, 0.0));
}

void TensorThrust::zero_gpu()
{
    gpu_error();
    thrust::fill(thrust::device, d_data_.begin(), d_data_.end(), 
                 make_cuDoubleComplex(0.0, 0.0));
}

void TensorThrust::set(
    const std::vector<size_t>& idxs,
    const std::complex<double> val
        )
{
    cpu_error();
    ndim_error(idxs.size());
    
    if (idxs.size() == 1) {
        h_data_[idxs[0]] = val;
    } else {
        size_t vidx = tidx_to_vidx(idxs);
        h_data_[vidx] = val;
    }
}

void TensorThrust::ndim_error(size_t ndims) const
{
    if (!(ndim() == ndims)) {
        throw std::runtime_error("Tensor ndim mismatch.");
    }
}

void TensorThrust::fill_from_nparray(std::vector<std::complex<double>> arr, std::vector<size_t> shape)
{
    cpu_error();
    
    if (shape_ != shape) {
        throw std::runtime_error("Shape mismatch in fill_from_nparray.");
    }
    
    if (arr.size() != size_) {
        throw std::runtime_error("Array size mismatch in fill_from_nparray.");
    }
    
    thrust::copy(arr.begin(), arr.end(), h_data_.begin());
}

double TensorThrust::norm() {
    if (on_gpu_) {
        double result = thrust::transform_reduce(thrust::device, d_data_.begin(), d_data_.end(),
                                                complex_norm_squared(), 0.0, thrust::plus<double>());
        return std::sqrt(result);
    } else {
        double result = 0.0;
        for (size_t i = 0; i < size_; i++) {
            result += std::norm(h_data_[i]);
        }
        return std::sqrt(result);
    }
}

void TensorThrust::add_to_element(
    const std::vector<size_t>& idxs,
    const std::complex<double> val
        )
{
    cpu_error();
    ndim_error(idxs.size());
    
    if (idxs.size() == 1) {
        h_data_[idxs[0]] += val;
    } else {
        size_t vidx = tidx_to_vidx(idxs);
        h_data_[vidx] += val;
    }
}

void TensorThrust::fill_from_np(std::vector<std::complex<double>> arr, std::vector<size_t> shape) {
    cpu_error();
    
    if (shape_ != shape) {
        throw std::runtime_error("Shape mismatch in fill_from_np.");
    }
    
    if (arr.size() != size_) {
        throw std::runtime_error("Array size mismatch in fill_from_np.");
    }
    
    thrust::copy(arr.begin(), arr.end(), h_data_.begin());
}

void TensorThrust::zero_with_shape(const std::vector<size_t>& shape, bool on_gpu)
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
    
    // Resize and zero the vectors
    h_data_.resize(size_, std::complex<double>(0.0, 0.0));
    d_data_.resize(size_, make_cuDoubleComplex(0.0, 0.0));
    
    initialized_ = true;
    
    // Ed's special memory thing
    total_memory__ = h_data_.size() * sizeof(std::complex<double>);
    
    on_gpu_ = on_gpu;
}

/// Get the vector index for this tensor based on the tensor index
size_t TensorThrust::tidx_to_vidx(const std::vector<size_t>& tidx) const
{
    size_t vidx = 0;
    for (int i = ndim() - 1; i >= 0; i--) {
        vidx += tidx[i] * strides_[i];
    }
    return vidx;
}

size_t TensorThrust::tidx_to_trans_vidx(const std::vector<size_t>& tidx, const std::vector<size_t>& axes) const
{
    size_t vidx = 0;
    for (int i = ndim() - 1; i >= 0; i--) {
        vidx += tidx[axes[i]] * strides_[i];
    }
    return vidx;
}

/// Get the tensor index for this tensor based on the vector index
std::vector<size_t> TensorThrust::vidx_to_tidx(size_t vidx) const
{
    std::vector<size_t> tidx(ndim());
    size_t vidx_tmp = vidx;
    
    for (int i = ndim() - 1; i >= 0; i--) {
        tidx[i] = vidx_tmp / strides_[i];
        vidx_tmp -= tidx[i] * strides_[i];
    }
    return tidx;
}

/// Get a particular element of this TensorThrust, specified by idxs
std::complex<double> TensorThrust::get(
    const std::vector<size_t>& idxs
    ) const
{
    cpu_error();
    ndim_error(idxs.size());
    
    if (idxs.size() == 1) {
        return h_data_[idxs[0]];
    } else {
        size_t vidx = tidx_to_vidx(idxs);
        return h_data_[vidx];
    }
}

void TensorThrust::shape_error(const std::vector<size_t>& shape) const
{
    ndim_error(shape.size());
    for (size_t i = 0; i < ndim(); i++) {
        if (shape_[i] != shape[i]) {
            throw std::runtime_error("Tensor shape mismatch.");
        }
    }
}

void TensorThrust::square_error() const
{
    ndim_error(2);
    if (shape_[0] != shape_[1]) {
        throw std::runtime_error("Tensor is not square.");
    }
}

std::shared_ptr<TensorThrust> TensorThrust::clone()
{
    return std::shared_ptr<TensorThrust>(new TensorThrust(*this));
}

void TensorThrust::identity()
{
    cpu_error();
    square_error();
    zero();
    for (size_t i = 0; i < shape_[0]; i++) {
        h_data_[i * shape_[1] + i] = std::complex<double>(1.0, 0.0);
    }
}

void TensorThrust::symmetrize()
{
    cpu_error();
    square_error();
    for (size_t i = 0; i < shape_[0]; i++) {
        for (size_t j = 0; j < shape_[1]; j++) {
            std::complex<double> val = 0.5 * (h_data_[i * shape_[1] + j] + h_data_[j * shape_[1] + i]);
            h_data_[i * shape_[1] + j] = val;
            h_data_[j * shape_[1] + i] = val;
        }
    }
}

void TensorThrust::antisymmetrize()
{
    cpu_error();
    square_error();
    for (size_t i = 0; i < shape_[0]; i++) {
        for (size_t j = 0; j < shape_[1]; j++) {
            std::complex<double> val = 0.5 * (h_data_[i * shape_[1] + j] - h_data_[j * shape_[1] + i]);
            h_data_[i * shape_[1] + j] = val;
            h_data_[j * shape_[1] + i] = -val;
        }
    }
}

void TensorThrust::scale(std::complex<double> a)
{
    if (on_gpu_) {
        gpu_error();
        cuDoubleComplex alpha = make_cuDoubleComplex(a.real(), a.imag());
        thrust::transform(thrust::device, d_data_.begin(), d_data_.end(), 
                         d_data_.begin(), complex_scale(alpha));
    } else {
        cpu_error();
        for (size_t i = 0; i < size_; i++) {
            h_data_[i] *= a;
        }
    }
}

void TensorThrust::copy_in(const TensorThrust& other)
{
    cpu_error();
    other.cpu_error();
    shape_error(other.shape());
    thrust::copy(other.h_data_.begin(), other.h_data_.end(), h_data_.begin());
}

void TensorThrust::copy_in_gpu(const TensorThrust& other)
{
    gpu_error();
    other.gpu_error();
    shape_error(other.shape());
    thrust::copy(thrust::device, other.d_data_.begin(), other.d_data_.end(), d_data_.begin());
}

void TensorThrust::copy_in_from_tensor(const Tensor& other)
{
    cpu_error();
    shape_error(other.shape());
    thrust::copy(other.read_data().begin(), other.read_data().end(), h_data_.begin());
}

void TensorThrust::copy_to_tensor(Tensor& dest) const
{
    cpu_error();
    dest.shape_error(shape_);
    
    // Copy data from TensorThrust's host vector to the Tensor
    std::copy(h_data_.begin(), h_data_.end(), dest.data().begin());
}

void TensorThrust::subtract(const TensorThrust& other) {
    shape_error(other.shape());
    
    if (on_gpu_) {
        gpu_error();
        other.gpu_error();
        thrust::transform(thrust::device, d_data_.begin(), d_data_.end(), 
                         other.d_data_.begin(), d_data_.begin(), complex_subtract());
    } else {
        cpu_error();
        other.cpu_error();
        for (size_t i = 0; i < size_; i++) {
            h_data_[i] -= other.h_data_[i];
        }
    }
}

void TensorThrust::zaxpby(
    const TensorThrust& x,
    std::complex<double> a,
    std::complex<double> b,
    const int incx,
    const int incy
    )
{
    shape_error(x.shape());
    
    if (on_gpu_) {
        gpu_error();
        x.gpu_error();
        cuDoubleComplex alpha = make_cuDoubleComplex(a.real(), a.imag());
        cuDoubleComplex beta = make_cuDoubleComplex(b.real(), b.imag());
        thrust::transform(thrust::device, x.d_data_.begin(), x.d_data_.end(), 
                         d_data_.begin(), d_data_.begin(), complex_axpby(alpha, beta));
    } else {
        cpu_error();
        x.cpu_error();
        math_zscale(size_, b, h_data_.data(), 1);
        math_zaxpy(size_, a, x.read_h_data().data(), incx, h_data_.data(), incy);
    }
}

void TensorThrust::zaxpy(
    const TensorThrust& x,
    const std::complex<double> alpha,
    const int incx,
    const int incy)
{
    if (shape_ != x.shape()) {
        throw std::runtime_error("Tensor shapes are not compatible for zaxpy.");
    }
    
    if (on_gpu_) {
        gpu_error();
        x.gpu_error();
        cuDoubleComplex alpha_cu = make_cuDoubleComplex(alpha.real(), alpha.imag());
        thrust::transform(thrust::device, x.d_data_.begin(), x.d_data_.end(), 
                         d_data_.begin(), d_data_.begin(), complex_axpy(alpha_cu));
    } else {
        cpu_error();
        x.cpu_error();
        math_zaxpy(size_, alpha, x.read_h_data().data(), incx, h_data_.data(), incy);
    }
}

void TensorThrust::gemm(
    const TensorThrust& B,
    const char transa,
    const char transb,
    const std::complex<double> alpha,
    const std::complex<double> beta,
    const bool mult_B_on_right)
{
    cpu_error();
    B.cpu_error();
    
    if ((shape_.size() != 2) || (shape_ != B.shape())) {
        throw std::runtime_error("Invalid tensor shapes for gemm.");
    }
    
    const int M = (transa == 'N') ? shape_[0] : shape_[1];
    const int N = (transb == 'N') ? B.shape()[1] : B.shape()[0];
    const int K = (transa == 'N') ? shape_[1] : shape_[0];
    
    std::complex<double>* A_data = h_data_.data();
    const std::complex<double>* B_data = B.read_h_data().data();
    
    std::complex<double>* C_data = h_data_.data();
    
    if (mult_B_on_right) {
        math_zgemm(transa, transb, M, N, K, alpha, A_data, shape_[0], B_data, B.shape()[0], beta, C_data, shape_[0]);
    } else {
        math_zgemm(transb, transa, N, M, K, alpha, B_data, B.shape()[0], A_data, shape_[0], beta, C_data, shape_[0]);
    }
}

std::complex<double> TensorThrust::vector_dot(const TensorThrust& other) const
{
    shape_error(other.shape());
    
    if (on_gpu_) {
        gpu_error();
        other.gpu_error();
        cuDoubleComplex result = thrust::inner_product(thrust::device, 
                                                      d_data_.begin(), d_data_.end(),
                                                      other.d_data_.begin(),
                                                      make_cuDoubleComplex(0.0, 0.0),
                                                      complex_add(),
                                                      complex_dot_product());
        return std::complex<double>(cuCreal(result), cuCimag(result));
    } else {
        cpu_error();
        other.cpu_error();
        std::complex<double> result = 0.0;
        for (size_t i = 0; i < size_; i++) {
            result += std::conj(h_data_[i]) * other.h_data_[i];
        }
        return result;
    }
}

TensorThrust TensorThrust::transpose() const
{
    cpu_error();
    ndim_error(2);
    
    TensorThrust T({shape_[1], shape_[0]});
    std::complex<double>* Tp = T.data().data();
    const std::complex<double>* Ap = h_data_.data();
    
    for (size_t ind1 = 0; ind1 < shape_[0]; ind1++) {
        for (size_t ind2 = 0; ind2 < shape_[1]; ind2++) {
            Tp[ind2 * shape_[0] + ind1] = Ap[ind1 * shape_[1] + ind2];
        }
    }
    return T;
}

TensorThrust TensorThrust::general_transpose(const std::vector<size_t>& axes) const
{
    cpu_error();
    
    if (axes.size() != ndim()) {
        throw std::runtime_error("Axes size must match tensor dimensions.");
    }
    
    std::vector<size_t> transposed_shape(ndim());
    for (size_t i = 0; i < ndim(); ++i) {
        transposed_shape[i] = shape_[axes[i]];
    }
    
    TensorThrust transposed_tensor(transposed_shape);
    
    std::complex<double>* transposed_data = transposed_tensor.data().data();
    
    for (size_t i = 0; i < size_; i++) {
        std::vector<size_t> old_tidx = vidx_to_tidx(i);
        std::vector<size_t> new_tidx(ndim());
        
        for (size_t j = 0; j < ndim(); j++) {
            new_tidx[j] = old_tidx[axes[j]];
        }
        
        size_t new_vidx = transposed_tensor.tidx_to_vidx(new_tidx);
        transposed_data[new_vidx] = h_data_[i];
    }
    
    return transposed_tensor;
}

TensorThrust TensorThrust::slice(std::vector<std::pair<size_t, size_t>> idxs) const
{
    cpu_error();
    
    std::vector<size_t> new_shape(idxs.size());
    std::vector<size_t> new_shape2;
    
    if (idxs.size() != ndim()) {
        throw std::runtime_error("Number of slice indices must match tensor dimensions.");
    }
    
    for (size_t i = 0; i < idxs.size(); i++) {
        new_shape[i] = idxs[i].second - idxs[i].first;
        if (new_shape[i] > 0) {
            new_shape2.push_back(new_shape[i]);
        }
    }
    
    TensorThrust new_tensor(new_shape2, name_ + "_sliced");
    
    std::vector<size_t> old_tidx(ndim());
    std::fill(old_tidx.begin(), old_tidx.end(), 0);
    
    size_t new_vidx = 0;
    for (size_t vidx = 0; vidx < size_; vidx++) {
        std::vector<size_t> tidx = vidx_to_tidx(vidx);
        
        bool in_slice = true;
        for (size_t i = 0; i < ndim(); i++) {
            if (tidx[i] < idxs[i].first || tidx[i] >= idxs[i].second) {
                in_slice = false;
                break;
            }
        }
        
        if (in_slice) {
            new_tensor.h_data_[new_vidx] = h_data_[vidx];
            new_vidx++;
        }
    }
    
    return new_tensor;
}

std::vector<std::vector<size_t>> TensorThrust::get_nonzero_tidxs() const
{
    cpu_error();
    
    std::vector<std::vector<size_t>> nonzero_tidxs;
    for (size_t i = 0; i < size_; i++) {
        if (std::abs(h_data_[i]) > 1e-12) {
            nonzero_tidxs.push_back(vidx_to_tidx(i));
        }
    }
    return nonzero_tidxs;
}

std::string TensorThrust::str(
    bool print_data,
    bool print_complex,
    int maxcols,
    const std::string& data_format,
    const std::string& header_format
    ) const
{
    std::string str = "";
    str += std::printf( "TensorThrust: %s\n", name_.c_str());
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

std::string TensorThrust::print_nonzero() const
{
    cpu_error();
    
    std::string str = "\n Nonzero indices and elements of TensorThrust: \n";
    str += " ========================================== \n";
    
    for (size_t i = 0; i < size_; i++) {
        if (std::abs(h_data_[i]) > 1e-12) {
            std::vector<size_t> tidx = vidx_to_tidx(i);
            str += " (";
            for (size_t j = 0; j < tidx.size(); j++) {
                str += std::to_string(tidx[j]);
                if (j < tidx.size() - 1) str += ", ";
            }
            str += ") = " + std::to_string(h_data_[i].real()) + " + " + std::to_string(h_data_[i].imag()) + "i\n";
        }
    }
    return str;
}

// Static methods implementation would go here
// These require more complex implementation that depends on the specific use case
// For now, providing basic structure

TensorThrust TensorThrust::chain(
    const std::vector<TensorThrust>& As,
    const std::vector<bool>& trans,
    std::complex<double> alpha,
    std::complex<double> beta)
{
    // This would need to be implemented based on specific requirements
    throw std::runtime_error("TensorThrust::chain not yet implemented");
}

void TensorThrust::permute(
    const std::vector<std::string>& Ainds,
    const std::vector<std::string>& Cinds,
    const TensorThrust& A,
    TensorThrust& C2,
    std::complex<double> alpha,
    std::complex<double> beta)
{
    // This would need to be implemented based on specific requirements
    throw std::runtime_error("TensorThrust::permute not yet implemented");
}

void TensorThrust::einsum(
    const std::vector<std::string>& Ainds,
    const std::vector<std::string>& Binds,
    const std::vector<std::string>& Cinds,
    const TensorThrust& A,
    const TensorThrust& B,
    TensorThrust& C3,
    std::complex<double> alpha,
    std::complex<double> beta)
{
    // This would need to be implemented based on specific requirements
    throw std::runtime_error("TensorThrust::einsum not yet implemented");
}