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


#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/extrema.h>                       // minmax_element
#include <thrust/device_ptr.h>                    // raw_pointer_cast
#include <thrust/system/cuda/execution_policy.h>  // thrust::cuda::par

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

struct make_complex {
    __host__ __device__
    cuDoubleComplex operator()(const double re, const double im) const {
        return make_cuDoubleComplex(re, im);
    }
};

// Device structs for extracting real and imaginary parts
struct real_part {
        __host__ __device__
        double operator()(const cuDoubleComplex& z) const { return z.x; }  // z.x == real
    };
    
struct imag_part {
    __host__ __device__
    double operator()(const cuDoubleComplex& z) const { return z.y; }  // z.y == imag
};

// Host structs for type conversion
struct real_part_cpu {
    double operator()(const std::complex<double>& z) const { return z.real(); }
};

struct imag_part_cpu {
    double operator()(const std::complex<double>& z) const { return z.imag(); }
};

struct make_complex_cpu {
    std::complex<double> operator()(double re, double im) const { return {re, im}; }
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
    const bool on_gpu,
    const std::string& data_type
    ) :
    shape_(shape),
    name_(name),
    on_gpu_(on_gpu),
    data_type_(data_type)
{
    
    strides_.resize(shape_.size());
    size_ = 1L;
    for (int i = shape_.size() - 1; i >= 0; i--) {
        strides_[i] = size_;
        size_ *= shape_[i];
    }
    

    if (data_type_ == "real"){

        h_re_data_.resize(size_, 0.0);
        d_re_data_.resize(size_, 0.0);

    } else if (data_type_ == "soa"){

        h_re_data_.resize(size_, 0.0);
        d_re_data_.resize(size_, 0.0);
        h_im_data_.resize(size_, 0.0);
        d_im_data_.resize(size_, 0.0);
        
    } else if (data_type_ == "complex"){
        
        // was the origional option...
        h_data_.resize(size_, std::complex<double>(0.0, 0.0));
        d_data_.resize(size_, make_cuDoubleComplex(0.0, 0.0));

        // If complex, on_complex_ is true ALWYAYS
        on_complex_ = true;

    }  else if (data_type_ == "all"){

        h_re_data_.resize(size_, 0.0);
        d_re_data_.resize(size_, 0.0);
        h_im_data_.resize(size_, 0.0);
        d_im_data_.resize(size_, 0.0);
  
        h_data_.resize(size_, std::complex<double>(0.0, 0.0));
        d_data_.resize(size_, make_cuDoubleComplex(0.0, 0.0));

        // If all, on_complex_ can be true or false, start with true
        // used when moving to/from complex to soa
        on_complex_ = true;

    } else {
        throw std::runtime_error("Invalid data_type specified for TensorThrust. Must be 'complex', 'real', 'soa' or 'all'.");
    }
    
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

    // Initialize SoA vectors, trivial overhaed for now...
    h_re_data_.resize(size_, 0.0);
    d_re_data_.resize(size_, 0.0);
    h_im_data_.resize(size_, 0.0);
    d_im_data_.resize(size_, 0.0);
    
    total_memory__ += h_data_.size() * sizeof(std::complex<double>);
    on_gpu_ = false;
    on_complex_ = true;
    initialized_ = true;
    name_ = "T";
    data_type_ = "complex";
}

/// Destructor
TensorThrust::~TensorThrust()
{
    // Ed's special memory thing
    total_memory__ -= h_data_.size() * sizeof(std::complex<double>);
    
    // Thrust vectors automatically handle cleanup
}

// TDDO: Get rid of temporary conversions and copy directly between host and device
void TensorThrust::to_gpu()
{
    cpu_error();
    if (initialized_ == 0) {
        std::cerr << "Tensor not initialized" << std::endl;
        return;
    }
    
    if(data_type_ == "complex"){
        // Manual conversion on CPU then copy to GPU
        std::vector<cuDoubleComplex> temp_gpu_data(size_);
        for (size_t i = 0; i < size_; i++) {
            temp_gpu_data[i] = make_cuDoubleComplex(h_data_[i].real(), h_data_[i].imag());
        }
        
        // Copy converted data to device
        thrust::copy(temp_gpu_data.begin(), temp_gpu_data.end(), d_data_.begin());

    } else if (data_type_ == "real"){
        // Copy real data to device
        thrust::copy(h_re_data_.begin(), h_re_data_.end(), d_re_data_.begin());

    } else if (data_type_ == "soa"){
        // Copy SoA data to device
        thrust::copy(h_re_data_.begin(), h_re_data_.end(), d_re_data_.begin());
        thrust::copy(h_im_data_.begin(), h_im_data_.end(), d_im_data_.begin());

    } else if (data_type_ == "all"){
        
        // Manual conversion on CPU then copy to GPU
        std::vector<cuDoubleComplex> temp_gpu_data(size_);
        for (size_t i = 0; i < size_; i++) {
            temp_gpu_data[i] = make_cuDoubleComplex(h_data_[i].real(), h_data_[i].imag());
        }
        
        // Copy converted data to device
        thrust::copy(temp_gpu_data.begin(), temp_gpu_data.end(), d_data_.begin());

        thrust::copy(h_re_data_.begin(), h_re_data_.end(), d_re_data_.begin());
        thrust::copy(h_im_data_.begin(), h_im_data_.end(), d_im_data_.begin());
    } else {
        throw std::runtime_error("Invalid data_type specified for TensorThrust. Must be 'complex', 'real', 'soa' or 'all'.");
    }

    on_gpu_ = true;
}

// TDDO: Get rid of temporary conversions and copy directly between host and device
void TensorThrust::to_cpu()
{
    gpu_error();
    if (initialized_ == 0) {
        std::cerr << "Tensor not initialized" << std::endl;
        return;
    }
    
    // Copy from device to temporary CPU vector
    std::vector<cuDoubleComplex> temp_cpu_data(size_);
    thrust::copy(d_data_.begin(), d_data_.end(), temp_cpu_data.begin());
    
    // Manual conversion on CPU
    for (size_t i = 0; i < size_; i++) {
        h_data_[i] = std::complex<double>(cuCreal(temp_cpu_data[i]), cuCimag(temp_cpu_data[i]));
    }


    if(data_type_ == "complex"){
        // Copy from device to temporary CPU vector
        std::vector<cuDoubleComplex> temp_cpu_data(size_);
        thrust::copy(d_data_.begin(), d_data_.end(), temp_cpu_data.begin());
        
        // Manual conversion on CPU
        for (size_t i = 0; i < size_; i++) {
            h_data_[i] = std::complex<double>(cuCreal(temp_cpu_data[i]), cuCimag(temp_cpu_data[i]));
        }
        

    } else if (data_type_ == "real"){
        // Copy real data to device
        thrust::copy(d_re_data_.begin(), d_re_data_.end(), h_re_data_.begin());
        

    } else if (data_type_ == "soa"){
        // Copy SoA data to device
        thrust::copy(d_re_data_.begin(), d_re_data_.end(), h_re_data_.begin());
        thrust::copy(d_im_data_.begin(), d_im_data_.end(), h_im_data_.begin());
        

    } else if (data_type_ == "all"){
        thrust::copy(d_re_data_.begin(), d_re_data_.end(), h_re_data_.begin());
        thrust::copy(d_im_data_.begin(), d_im_data_.end(), h_im_data_.begin());

        // Copy from device to temporary CPU vector
        std::vector<cuDoubleComplex> temp_cpu_data(size_);
        thrust::copy(d_data_.begin(), d_data_.end(), temp_cpu_data.begin());
        
        // Manual conversion on CPU
        for (size_t i = 0; i < size_; i++) {
            h_data_[i] = std::complex<double>(cuCreal(temp_cpu_data[i]), cuCimag(temp_cpu_data[i]));
        }
        
    } else {
        throw std::runtime_error("Invalid data_type specified for TensorThrust. Must be 'complex', 'real', 'soa' or 'all'.");
    }

    on_gpu_ = false;
}

void TensorThrust::complex_to_soa_gpu()
{
    gpu_error();
    all_error();
    on_complex_error();

    thrust::transform(thrust::device, d_data_.begin(), d_data_.end(), d_re_data_.begin(), real_part{});
    thrust::transform(thrust::device, d_data_.begin(), d_data_.end(), d_im_data_.begin(), imag_part{});

    on_complex_ = false;
}

void TensorThrust::soa_to_complex_gpu()
{
    gpu_error();
    all_error();
    on_soa_error();

    // Combine SoA (double,double) -> interleaved cuDoubleComplex on device
    thrust::transform(thrust::device, d_re_data_.begin(), d_re_data_.end(), d_im_data_.begin(), d_data_.begin(), make_complex{});

    on_complex_ = true;
}

void TensorThrust::complex_to_soa_cpu()
{
    all_error();
    on_complex_error();  // ensures we're currently in complex mode

    thrust::transform(thrust::host, h_data_.begin(), h_data_.end(), h_re_data_.begin(), real_part_cpu{});
    thrust::transform(thrust::host, h_data_.begin(), h_data_.end(), h_im_data_.begin(), imag_part_cpu{});

    on_complex_ = false; // now SoA-active ("complex") on host
}

void TensorThrust::soa_to_complex_cpu()
{
    all_error();
    on_soa_error();  // ensures we're currently in soa mode

    thrust::transform(thrust::host, h_re_data_.begin(), h_re_data_.end(), h_im_data_.begin(), h_data_.begin(), make_complex_cpu{});

    on_complex_ = true; // now AoS complex-active on host
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

void TensorThrust::complex_error() const {
    if (data_type_ == "real" or data_type_ == "soa") {
        throw std::runtime_error("Tensor data type is not 'complex' but only complex operation is supported currently.");
    }

    // ok if data_type_ is "all" if data is "on complex"
    if (data_type_ == "all" && on_complex_ != true) {
        throw std::runtime_error("Tensor data is not 'on complex' but only complex operation is supported currently.");
    }
}

void TensorThrust::on_complex_error() const {
    if (on_complex_ != true) {
        throw std::runtime_error("Tensor data is not 'on complex' but needs to be for this operation.");
    }
}

void TensorThrust::on_soa_error() const {
    if (on_complex_ == true) {
        throw std::runtime_error("Tensor data is not 'on soa' but needs to be for this operation.");
    }
}

void TensorThrust::real_error() const {
    if (data_type_ != "real") {
        throw std::runtime_error("Tensor data type is not 'real' but only real operation is supported currently.");
    }
}

void TensorThrust::soa_error() const {
    if (data_type_ != "soa") {
        throw std::runtime_error("Tensor data type is not 'soa' but only soa operation is supported currently.");
    }
}

void TensorThrust::all_error() const {
    if (data_type_ != "all") {
        throw std::runtime_error("Tensor data type is not 'all' but only all operation is supported currently.");
    }
}

void TensorThrust::data_type_error(const std::string& other_data_type) const
{
    if (data_type_ != other_data_type) {
        throw std::runtime_error("Tensor data type mismatch.");
    }
}

void TensorThrust::zero()
{
    cpu_error();
    complex_error();
    thrust::fill(h_data_.begin(), h_data_.end(), std::complex<double>(0.0, 0.0));
}

void TensorThrust::zero_gpu()
{
    gpu_error();
    complex_error();
    thrust::fill(thrust::device, d_data_.begin(), d_data_.end(), 
                 make_cuDoubleComplex(0.0, 0.0));
}

// ==========================================
// These now should throw error depending on 
// data_type_ and on_complex_
// ==========================================
thrust::host_vector<std::complex<double>>& TensorThrust::h_data() 
{
    cpu_error();
    on_complex_error();
    return h_data_;
}

const thrust::host_vector<double>& TensorThrust::read_h_re_data() const 
{
    cpu_error();
    on_soa_error();
    return h_re_data_;
}

const thrust::host_vector<double>& TensorThrust::read_h_im_data() const 
{
    cpu_error();
    on_soa_error();
    return h_im_data_;
}

const thrust::host_vector<std::complex<double>>& TensorThrust::read_h_data() const 
{
    cpu_error();
    on_complex_error();
    return h_data_; 
}

thrust::device_vector<cuDoubleComplex>& TensorThrust::d_data() 
{
    gpu_error();
    on_complex_error();
    return d_data_; 
}

thrust::device_vector<double>& TensorThrust::d_re_data() 
{
    gpu_error();
    on_soa_error();
    return d_re_data_;
}

thrust::device_vector<double>& TensorThrust::d_im_data() 
{
    gpu_error();
    on_soa_error();
    return d_im_data_;
}

const thrust::device_vector<cuDoubleComplex>& TensorThrust::read_d_data() const 
{
    gpu_error();
    on_complex_error();
    return d_data_; 
}

const thrust::device_vector<double>& TensorThrust::read_d_re_data() const 
{
    gpu_error();
    on_soa_error();
    return d_re_data_;
}

const thrust::device_vector<double>& TensorThrust::read_d_im_data() const 
{
    gpu_error();
    on_soa_error();
    return d_im_data_;
}

void TensorThrust::set(
    const std::vector<size_t>& idxs,
    const std::complex<double> val
        )
{
    cpu_error();
    complex_error();
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
    complex_error();
    
    if (shape_ != shape) {
        throw std::runtime_error("Shape mismatch in fill_from_nparray.");
    }
    
    if (arr.size() != size_) {
        throw std::runtime_error("Array size mismatch in fill_from_nparray.");
    }
    
    thrust::copy(arr.begin(), arr.end(), h_data_.begin());
}

double TensorThrust::norm() {

    complex_error();

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
    complex_error();
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
    complex_error();
    
    if (shape_ != shape) {
        throw std::runtime_error("Shape mismatch in fill_from_np.");
    }
    
    if (arr.size() != size_) {
        throw std::runtime_error("Array size mismatch in fill_from_np.");
    }
    
    thrust::copy(arr.begin(), arr.end(), h_data_.begin());
}

void TensorThrust::zero_with_shape(
    const std::vector<size_t>& shape, 
    bool on_gpu, 
    const std::string& data_type)
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
    data_type_ = data_type;

    if(data_type_ == "real"){

        h_re_data_.resize(size_, 0.0);
        d_re_data_.resize(size_, 0.0);

    } else if (data_type_ == "soa"){

        h_re_data_.resize(size_, 0.0);
        d_re_data_.resize(size_, 0.0);
        h_im_data_.resize(size_, 0.0);
        d_im_data_.resize(size_, 0.0);
        
    } else if (data_type_ == "complex"){
        
        // was the origional option...
        h_data_.resize(size_, std::complex<double>(0.0, 0.0));
        d_data_.resize(size_, make_cuDoubleComplex(0.0, 0.0));

        on_complex_ = true;

    }  else if (data_type_ == "all"){

        h_re_data_.resize(size_, 0.0);
        d_re_data_.resize(size_, 0.0);
        h_im_data_.resize(size_, 0.0);
        d_im_data_.resize(size_, 0.0);
  
        h_data_.resize(size_, std::complex<double>(0.0, 0.0));
        d_data_.resize(size_, make_cuDoubleComplex(0.0, 0.0));

        on_complex_ = true;

    } else {
        throw std::runtime_error("Invalid data_type specified for TensorThrust. Must be 'complex', 'real', 'soa' or 'all'.");
    }
    
    // Resize and zero the vectors
    // h_data_.resize(size_, std::complex<double>(0.0, 0.0));
    // d_data_.resize(size_, make_cuDoubleComplex(0.0, 0.0));
    
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
    complex_error();
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
    complex_error();
    square_error();
    zero();
    for (size_t i = 0; i < shape_[0]; i++) {
        h_data_[i * shape_[1] + i] = std::complex<double>(1.0, 0.0);
    }
}

void TensorThrust::symmetrize()
{
    cpu_error();
    complex_error();
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
    complex_error();
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
    complex_error();
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

void TensorThrust::gather_in_2D_gpu(
    const TensorThrust& other,
    const thrust::device_vector<int>& i_inds,
    const thrust::device_vector<int>& j_inds
)
{
    complex_error();
    other.complex_error();

    gpu_error();
    other.gpu_error();
    ndim_error(2);
    shape_error(other.shape());


    const int nrows = static_cast<int>(i_inds.size());
    const int ncols = static_cast<int>(j_inds.size());
    if (nrows == 0 || ncols == 0) {
        return; 
    }

    const int H = static_cast<int>(shape_[0]);
    const int W = static_cast<int>(shape_[1]);

    // Raw device pointers to data and index arrays
    cuDoubleComplex* __restrict__ dst = thrust::raw_pointer_cast(d_data_.data());
    
    const cuDoubleComplex* __restrict__ src = thrust::raw_pointer_cast(other.d_data_.data());

    const int* __restrict__ d_rows = thrust::raw_pointer_cast(i_inds.data());
    const int* __restrict__ d_cols = thrust::raw_pointer_cast(j_inds.data());

    const size_t N = static_cast<size_t>(nrows) * static_cast<size_t>(ncols);

    // Execute: each k maps to (rsel, m) -> (i,j) -> single copy
    // attach a stream with .on(stream_) if you track one
    auto exec = thrust::cuda::par; 

    thrust::for_each(
        exec,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(N),
        [=] __device__ (size_t k)
        {
            const int rsel = static_cast<int>(k / ncols);
            const int m    = static_cast<int>(k % ncols);

            const int i = d_rows[rsel];
            const int j = d_cols[m];

            // Prevalidated: 0 <= i < H, 0 <= j < W
            const size_t off = static_cast<size_t>(i) * static_cast<size_t>(W) + static_cast<size_t>(j);
            dst[off] = src[off];
        });
}

void TensorThrust::copy_in(const TensorThrust& other)
{
    cpu_error();
    other.cpu_error();
    shape_error(other.shape());
    data_type_error(other.data_type());

    if(data_type_ == "complex"){
        thrust::copy(other.h_data_.begin(), other.h_data_.end(), h_data_.begin()); 

    } else if (data_type_ == "real"){
        thrust::copy(other.h_re_data_.begin(), other.h_re_data_.end(), h_re_data_.begin());

    } else if (data_type_ == "soa"){
        thrust::copy(other.h_re_data_.begin(), other.h_re_data_.end(), h_re_data_.begin());
        thrust::copy(other.h_im_data_.begin(), other.h_im_data_.end(), h_im_data_.begin());    

    } else if (data_type_ == "all"){
        thrust::copy(other.h_data_.begin(), other.h_data_.end(), h_data_.begin()); 
        thrust::copy(other.h_re_data_.begin(), other.h_re_data_.end(), h_re_data_.begin());
        thrust::copy(other.h_im_data_.begin(), other.h_im_data_.end(), h_im_data_.begin());
        
    } else {
        throw std::runtime_error("Invalid data_type specified for TensorThrust. Must be 'complex', 'real', 'soa' or 'all'.");
    }

}

void TensorThrust::copy_in_gpu(const TensorThrust& other)
{
    gpu_error();
    other.gpu_error();
    shape_error(other.shape());
    data_type_error(other.data_type());

    if(data_type_ == "complex"){
        thrust::copy(thrust::device, other.d_data_.begin(), other.d_data_.end(), d_data_.begin()); 

    } else if (data_type_ == "real"){
        thrust::copy(thrust::device, other.d_re_data_.begin(), other.d_re_data_.end(), d_re_data_.begin());

    } else if (data_type_ == "soa"){
        thrust::copy(thrust::device, other.d_re_data_.begin(), other.d_re_data_.end(), d_re_data_.begin());
        thrust::copy(thrust::device, other.d_im_data_.begin(), other.d_im_data_.end(), d_im_data_.begin());    

    } else if (data_type_ == "all"){
        thrust::copy(thrust::device, other.d_data_.begin(), other.d_data_.end(), d_data_.begin()); 
        thrust::copy(thrust::device, other.d_re_data_.begin(), other.d_re_data_.end(), d_re_data_.begin());
        thrust::copy(thrust::device, other.d_im_data_.begin(), other.d_im_data_.end(), d_im_data_.begin());
        
    } else {
        throw std::runtime_error("Invalid data_type specified for TensorThrust. Must be 'complex', 'real', 'soa' or 'all'.");
    }

    // thrust::copy(thrust::device, other.d_data_.begin(), other.d_data_.end(), d_data_.begin());
}

void TensorThrust::copy_in_from_tensor(const Tensor& other)
{
    cpu_error();
    complex_error();
    shape_error(other.shape());
    thrust::copy(other.read_data().begin(), other.read_data().end(), h_data_.begin());
}

void TensorThrust::copy_to_tensor(Tensor& dest) const
{
    cpu_error();
    complex_error();
    dest.shape_error(shape_);
    
    // Copy data from TensorThrust's host vector to the Tensor
    std::copy(h_data_.begin(), h_data_.end(), dest.data().begin());
}

void TensorThrust::subtract(const TensorThrust& other) {
    shape_error(other.shape());
    complex_error();
    
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
    complex_error();
    
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
    shape_error(x.shape());
    complex_error();
    
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
    complex_error();
    
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
    complex_error();
    
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
    complex_error();
    
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
    complex_error();
    
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
    complex_error();
    
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
    complex_error();
    
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
    )
{
    bool reset = false;
    if (on_gpu_) {
        reset = true;
        to_cpu();
    }

    cpu_error();
    complex_error();

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

    if (reset) {
        to_gpu();
    }
    return str;
}

std::string TensorThrust::print_nonzero() const
{
    cpu_error();
    complex_error();
    
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