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

        on_complex_ = false;
        
    } else if (data_type_ == "complex"){
        
        // was the origional option...
        h_data_.resize(size_, std::complex<double>(0.0, 0.0));
        d_data_.resize(size_, make_cuDoubleComplex(0.0, 0.0));

        // If complex, on_complex_ is true ALWYAYS
        on_complex_ = true;

    } else {
        throw std::runtime_error("Invalid data_type specified for TensorThrust. Must be 'complex' or 'real'.");
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

    // Initialize real vectors, trivial overhaed for now...
    h_re_data_.resize(size_, 0.0);
    d_re_data_.resize(size_, 0.0);
    
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

    } else {
        throw std::runtime_error("Invalid data_type specified for TensorThrust. Must be 'complex' or 'real'.");
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
        
    } else {
        throw std::runtime_error("Invalid data_type specified for TensorThrust. Must be 'complex' or 'real'.");
    }

    on_gpu_ = false;
}

void TensorThrust::add(const TensorThrust& other) {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Tensor shapes are not compatible for addition.");
    }

    if (data_type_ != other.data_type_) {
        throw std::runtime_error("Data type mismatch in add.");
    }

    if (on_gpu_) {
        gpu_error();
        other.gpu_error();

        if (data_type_ == "complex") {
            thrust::transform(thrust::device, d_data_.begin(), d_data_.end(), other.d_data_.begin(), d_data_.begin(), complex_add());
        } else if (data_type_ == "real") {
            thrust::transform(thrust::device, d_re_data_.begin(), d_re_data_.end(), other.d_re_data_.begin(), d_re_data_.begin(), thrust::plus<double>());
        } else {
            throw std::runtime_error("Unsupported data type in add (GPU).");
        }

    } else {
        cpu_error();
        other.cpu_error();

        if (data_type_ == "complex") {
            for (size_t i = 0; i < size_; i++) h_data_[i] += other.h_data_[i];
        } else if (data_type_ == "real") {
            for (size_t i = 0; i < size_; i++) h_re_data_[i] += other.h_re_data_[i];
        } else {
            throw std::runtime_error("Unsupported data type in add (CPU).");
        }

    }
}

void TensorThrust::add_thrust(const TensorThrust& other) {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Tensor shapes are not compatible for addition.");
    }
    gpu_error();
    other.gpu_error();

    if (data_type_ != other.data_type_) {
        throw std::runtime_error("Data type mismatch in add_thrust.");
    }

    if (data_type_ == "complex") {
        thrust::transform(thrust::device, d_data_.begin(), d_data_.end(), other.d_data_.begin(), d_data_.begin(), complex_add());
    } else if (data_type_ == "real") {
        thrust::transform(thrust::device, d_re_data_.begin(), d_re_data_.end(), other.d_re_data_.begin(), d_re_data_.begin(), thrust::plus<double>());
    } else {
        throw std::runtime_error("Unsupported data type in add_thrust.");
    }
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
    if (data_type_ == "real") {
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

void TensorThrust::on_real_error() const {
    if (on_complex_ != false) {
        throw std::runtime_error("Tensor data is not 'on real' but needs to be for this operation.");
    }
}

void TensorThrust::real_error() const {
    if (data_type_ != "real") {
        throw std::runtime_error("Tensor data type is not 'real' but only real operation is supported currently.");
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
    if (data_type_ == "complex") {
        thrust::fill(h_data_.begin(), h_data_.end(), std::complex<double>(0.0, 0.0));
    } else if (data_type_ == "real") {
        thrust::fill(h_re_data_.begin(), h_re_data_.end(), 0.0);
    } else {
        throw std::runtime_error("Unsupported data type in zero().");
    }
}

void TensorThrust::zero_gpu()
{
    gpu_error();
    if (data_type_ == "complex") {
        thrust::fill(thrust::device, d_data_.begin(), d_data_.end(), make_cuDoubleComplex(0.0, 0.0));
    } else if (data_type_ == "real") {
        thrust::fill(thrust::device, d_re_data_.begin(), d_re_data_.end(), 0.0);
    } else {
        throw std::runtime_error("Unsupported data type in zero_gpu().");
    }
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
    on_real_error();
    return h_re_data_;
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
    on_real_error();
    return d_re_data_;
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
    on_real_error();
    return d_re_data_;
}

void TensorThrust::set(
    const std::vector<size_t>& idxs,
    const std::complex<double> val
        )
{
    cpu_error();
    ndim_error(idxs.size());

    size_t vidx = (idxs.size()==1)? idxs[0] : tidx_to_vidx(idxs);
    if (data_type_ == "complex") {
        h_data_[vidx] = val;
    } else if (data_type_ == "real") {
        if (std::abs(val.imag()) > 1e-14) throw std::runtime_error("Attempt to assign complex value to real tensor.");
        h_re_data_[vidx] = val.real();
    } else {
        throw std::runtime_error("Unsupported data type in set().");
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
    if (shape_ != shape) throw std::runtime_error("Shape mismatch in fill_from_nparray.");
    if (arr.size() != size_) throw std::runtime_error("Array size mismatch in fill_from_nparray.");

    if (data_type_ == "complex") {
        thrust::copy(arr.begin(), arr.end(), h_data_.begin());
    } else if (data_type_ == "real") {
        for (size_t i=0;i<size_;++i) {
            if (std::abs(arr[i].imag()) > 1e-14) throw std::runtime_error("Imag component present loading into real tensor.");
            h_re_data_[i] = arr[i].real();
        }
    } else {
        throw std::runtime_error("Unsupported data type in fill_from_nparray().");
    }
}

double TensorThrust::norm() {
    if (on_gpu_) {
        if (data_type_ == "complex") {
            double result = thrust::transform_reduce(thrust::device, d_data_.begin(), d_data_.end(), complex_norm_squared(), 0.0, thrust::plus<double>());
            return std::sqrt(result);
        } else if (data_type_ == "real") {
            double result = thrust::transform_reduce(thrust::device, d_re_data_.begin(), d_re_data_.end(), thrust::square<double>(), 0.0, thrust::plus<double>());
            return std::sqrt(result);
        } else {
            throw std::runtime_error("Unsupported data type in norm() on GPU.");
        }
    } else {
        if (data_type_ == "complex") {
            double result = 0.0; for (size_t i=0;i<size_;++i) result += std::norm(h_data_[i]); return std::sqrt(result);
        } else if (data_type_ == "real") {
            double result=0.0; for (size_t i=0;i<size_;++i) { double v=h_re_data_[i]; result += v*v;} return std::sqrt(result);
        } else {
            throw std::runtime_error("Unsupported data type in norm() on CPU.");
        }
    }
}

void TensorThrust::add_to_element(
    const std::vector<size_t>& idxs,
    const std::complex<double> val
        ) {
    cpu_error();
    ndim_error(idxs.size());

    size_t vidx = (idxs.size()==1)? idxs[0] : tidx_to_vidx(idxs);
    if (data_type_ == "complex") {
        h_data_[vidx] += val;
    } else if (data_type_ == "real") {
        if (std::abs(val.imag())>1e-14) throw std::runtime_error("Imag component in add_to_element for real tensor.");
        h_re_data_[vidx] += val.real();
    } else {
        throw std::runtime_error("Unsupported data type in add_to_element().");
    }
}

void TensorThrust::fill_from_np(std::vector<std::complex<double>> arr, std::vector<size_t> shape) {
    cpu_error();
    if (shape_ != shape) throw std::runtime_error("Shape mismatch in fill_from_np.");
    if (arr.size() != size_) throw std::runtime_error("Array size mismatch in fill_from_np.");
    
    if (data_type_ == "complex") {
        thrust::copy(arr.begin(), arr.end(), h_data_.begin());
    } else if (data_type_ == "real") {
        for (size_t i=0;i<size_;++i){ if (std::abs(arr[i].imag())>1e-14) throw std::runtime_error("Imag component present loading into real tensor."); h_re_data_[i]=arr[i].real(); }
    } else {
        throw std::runtime_error("Unsupported data type in fill_from_np().");
    }
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

        on_complex_ = false;
        
    } else if (data_type_ == "complex"){
        
        // was the origional option...
        h_data_.resize(size_, std::complex<double>(0.0, 0.0));
        d_data_.resize(size_, make_cuDoubleComplex(0.0, 0.0));

        on_complex_ = true;

    } else {
        throw std::runtime_error("Invalid data_type specified for TensorThrust. Must be 'complex' or 'real'.");
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
    ndim_error(idxs.size());

    size_t vidx = (idxs.size()==1)? idxs[0] : tidx_to_vidx(idxs);
    if (data_type_ == "complex") {
        return h_data_[vidx];
    } else if (data_type_ == "real") {
        return std::complex<double>(h_re_data_[vidx], 0.0);
    } else {
        throw std::runtime_error("Unsupported data type in get().");
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
    if (data_type_ == "complex") {
        for (size_t i=0;i<shape_[0];++i) h_data_[i*shape_[1]+i] = {1.0,0.0};
    } else if (data_type_ == "real") {
        for (size_t i=0;i<shape_[0];++i) h_re_data_[i*shape_[1]+i] = 1.0;
    } else {
        throw std::runtime_error("Unsupported data type in identity().");
    }
}

void TensorThrust::symmetrize()
{
    cpu_error();
    square_error();

    if (data_type_ == "complex") {
        for (size_t i=0;i<shape_[0];++i) for (size_t j=0;j<shape_[1];++j){ auto val = 0.5*(h_data_[i*shape_[1]+j] + h_data_[j*shape_[1]+i]); h_data_[i*shape_[1]+j]=val; h_data_[j*shape_[1]+i]=val; }
    } else if (data_type_ == "real") {
        for (size_t i=0;i<shape_[0];++i) for (size_t j=0;j<shape_[1];++j){ double val = 0.5*(h_re_data_[i*shape_[1]+j] + h_re_data_[j*shape_[1]+i]); h_re_data_[i*shape_[1]+j]=val; h_re_data_[j*shape_[1]+i]=val; }
    } else {
        throw std::runtime_error("Unsupported data type in symmetrize().");
    }
}

void TensorThrust::antisymmetrize()
{
    cpu_error();
    square_error();

    if (data_type_ == "complex") {
        for (size_t i=0;i<shape_[0];++i) for (size_t j=0;j<shape_[1];++j){ auto val = 0.5*(h_data_[i*shape_[1]+j] - h_data_[j*shape_[1]+i]); h_data_[i*shape_[1]+j]=val; h_data_[j*shape_[1]+i]=-val; }
    } else if (data_type_ == "real") {
        for (size_t i=0;i<shape_[0];++i) for (size_t j=0;j<shape_[1];++j){ double val = 0.5*(h_re_data_[i*shape_[1]+j] - h_re_data_[j*shape_[1]+i]); h_re_data_[i*shape_[1]+j]=val; h_re_data_[j*shape_[1]+i]=-val; }
    } else {
        throw std::runtime_error("Unsupported data type in antisymmetrize().");
    }
}

void TensorThrust::scale(std::complex<double> a)
{
    if (data_type_ == "real" && std::abs(a.imag())>1e-14) throw std::runtime_error("Complex scale on real tensor.");
    
    if (on_gpu_) {
        gpu_error();
        if (data_type_ == "complex") {
            cuDoubleComplex alpha = make_cuDoubleComplex(a.real(), a.imag());
            thrust::transform(thrust::device, d_data_.begin(), d_data_.end(), d_data_.begin(), complex_scale(alpha));
        } else if (data_type_ == "real") {
            double ar = a.real();
            thrust::transform(thrust::device, d_re_data_.begin(), d_re_data_.end(), d_re_data_.begin(), [=] __device__ (double x){ return ar * x; });
        } else {
            throw std::runtime_error("Unsupported data type in scale (GPU).");
        }
    } else{
        cpu_error();
        if (data_type_ == "complex") {
            for (size_t i=0;i<size_;++i) h_data_[i] *= a;
        } else if (data_type_ == "real") {
            double ar = a.real();
            for (size_t i=0;i<size_;++i) h_re_data_[i] *= ar;
        } else {
            throw std::runtime_error("Unsupported data type in scale (CPU).");
        }
    }
}

void TensorThrust::gather_in_2D_gpu(
    const TensorThrust& other,
    const thrust::device_vector<int>& i_inds,
    const thrust::device_vector<int>& j_inds
)
{
    if (data_type_ != other.data_type_) throw std::runtime_error("Data type mismatch in gather_in_2D_gpu.");
    gpu_error(); 
    other.gpu_error(); 
    ndim_error(2); 
    shape_error(other.shape());

    const int nrows = (int)i_inds.size(); 
    const int ncols = (int)j_inds.size(); 
    if (nrows==0 || ncols==0) return; 
    const int H=(int)shape_[0]; 
    const int W=(int)shape_[1];
    
    if (data_type_ == "complex") {
        cuDoubleComplex* __restrict__ dst = thrust::raw_pointer_cast(d_data_.data());
        const cuDoubleComplex* __restrict__ src = thrust::raw_pointer_cast(other.d_data_.data());
        const int* __restrict__ d_rows = thrust::raw_pointer_cast(i_inds.data());
        const int* __restrict__ d_cols = thrust::raw_pointer_cast(j_inds.data());
        size_t N = (size_t)nrows * (size_t)ncols; auto exec = thrust::cuda::par;
        thrust::for_each(exec, thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(N), [=] __device__ (size_t k){ int rsel = (int)(k / ncols); int m = (int)(k % ncols); int i = d_rows[rsel]; int j = d_cols[m]; size_t off = (size_t)i * (size_t)W + (size_t)j; dst[off] = src[off]; });
    } else if (data_type_ == "real") {
        double* __restrict__ dst = thrust::raw_pointer_cast(d_re_data_.data());
        const double* __restrict__ src = thrust::raw_pointer_cast(other.d_re_data_.data());
        const int* __restrict__ d_rows = thrust::raw_pointer_cast(i_inds.data());
        const int* __restrict__ d_cols = thrust::raw_pointer_cast(j_inds.data());
        size_t N = (size_t)nrows * (size_t)ncols; auto exec = thrust::cuda::par;
        thrust::for_each(exec, thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(N), [=] __device__ (size_t k){ int rsel = (int)(k / ncols); int m = (int)(k % ncols); int i = d_rows[rsel]; int j = d_cols[m]; size_t off = (size_t)i * (size_t)W + (size_t)j; dst[off] = src[off]; });
    } else {
        throw std::runtime_error("Unsupported data type in gather_in_2D_gpu.");
    }
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
        
    } else {
        throw std::runtime_error("Invalid data_type specified for TensorThrust. Must be 'complex' or 'real'.");
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
        
    } else {
        throw std::runtime_error("Invalid data_type specified for TensorThrust. Must be 'complex' or 'real'.");
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
    dest.shape_error(shape_);

    // real & complex version:
    if (data_type_ == "complex") {
        thrust::copy(h_data_.begin(), h_data_.end(), dest.data().begin());
    } else if (data_type_ == "real") {
        thrust::transform(
            h_re_data_.begin(), h_re_data_.end(),
            dest.data().begin(),
            [] __host__ (double x) { return std::complex<double>(x, 0.0); }
        );
    }
}

void TensorThrust::subtract(const TensorThrust& other) {
    shape_error(other.shape());
    if (data_type_ != other.data_type_) throw std::runtime_error("Data type mismatch in subtract.");

    if (on_gpu_) {
        gpu_error(); other.gpu_error();
        if (data_type_ == "complex") {
            thrust::transform(thrust::device, d_data_.begin(), d_data_.end(), other.d_data_.begin(), d_data_.begin(), complex_subtract());
        } else if (data_type_ == "real") {
            thrust::transform(thrust::device, d_re_data_.begin(), d_re_data_.end(), other.d_re_data_.begin(), d_re_data_.begin(), thrust::minus<double>());
        } else {
            throw std::runtime_error("Unsupported data type in subtract() GPU path.");
        }
    } else {
        cpu_error(); other.cpu_error();
        if (data_type_ == "complex") { 
            for (size_t i=0;i<size_;++i) h_data_[i] -= other.h_data_[i]; 
        } else if (data_type_ == "real") { 
            for (size_t i=0;i<size_;++i) h_re_data_[i] -= other.h_re_data_[i]; 
        } else { 
            throw std::runtime_error("Unsupported data type in subtract() CPU path."); 
        }
    }
}

void TensorThrust::zaxpby(
    const TensorThrust& x,
    std::complex<double> a,
    std::complex<double> b,
    const int incx,
    const int incy)
{
    shape_error(x.shape());
    if (data_type_ != x.data_type_) throw std::runtime_error("Data type mismatch in zaxpby.");
    if (data_type_ == "real" && (std::abs(a.imag())>1e-14 || std::abs(b.imag())>1e-14)) throw std::runtime_error("Complex coefficients on real tensor in zaxpby.");
    if (on_gpu_) {
        gpu_error(); x.gpu_error();
        if (data_type_ == "complex") {
            cuDoubleComplex alpha = make_cuDoubleComplex(a.real(), a.imag());
            cuDoubleComplex beta = make_cuDoubleComplex(b.real(), b.imag());
            thrust::transform(thrust::device, x.d_data_.begin(), x.d_data_.end(), d_data_.begin(), d_data_.begin(), complex_axpby(alpha, beta));
        } else if (data_type_ == "real") {
            double ar=a.real(), br=b.real();
            double* __restrict__ y_ptr = thrust::raw_pointer_cast(d_re_data_.data());
            const double* __restrict__ x_ptr = thrust::raw_pointer_cast(x.d_re_data_.data());
            thrust::for_each(
                thrust::device,
                thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator<size_t>(size_),
                [=] __device__ (size_t i){ y_ptr[i] = ar * x_ptr[i] + br * y_ptr[i]; });
        } else {
            throw std::runtime_error("Unsupported data type in zaxpby (GPU).");
        }
    } else {
        cpu_error(); x.cpu_error();
        if (data_type_ == "complex") {
            math_zscale(size_, b, h_data_.data(), 1);
            math_zaxpy(size_, a, x.read_h_data().data(), incx, h_data_.data(), incy);
        } else if (data_type_ == "real") {
            // y = a*x + b*y  -> scale y by b then axpy with a
            math_dscal(size_, b.real(), h_re_data_.data(), 1);
            math_daxpy(size_, a.real(), x.h_re_data_.data(), incx, h_re_data_.data(), incy);
        } else {
            throw std::runtime_error("Unsupported data type in zaxpby (CPU).");
        }
    }
}

void TensorThrust::zaxpy(
    const TensorThrust& x,
    const std::complex<double> alpha,
    const int incx,
    const int incy)
{
    shape_error(x.shape());
    if (data_type_ != x.data_type_) throw std::runtime_error("Data type mismatch in zaxpy.");
    if (data_type_ == "real" && std::abs(alpha.imag())>1e-14) throw std::runtime_error("Complex alpha on real tensor in zaxpy.");
    if (on_gpu_) {
        gpu_error(); x.gpu_error();
        if (data_type_ == "complex") {
            cuDoubleComplex alpha_cu = make_cuDoubleComplex(alpha.real(), alpha.imag());
            thrust::transform(thrust::device, x.d_data_.begin(), x.d_data_.end(), d_data_.begin(), d_data_.begin(), complex_axpy(alpha_cu));
        } else if (data_type_ == "real") {
            double ar = alpha.real();
            double* __restrict__ y_ptr = thrust::raw_pointer_cast(d_re_data_.data());
            const double* __restrict__ x_ptr = thrust::raw_pointer_cast(x.d_re_data_.data());
            thrust::for_each(
                thrust::device,
                thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator<size_t>(size_),
                [=] __device__ (size_t i){ y_ptr[i] += ar * x_ptr[i]; });
        } else {
            throw std::runtime_error("Unsupported data type in zaxpy (GPU).");
        }
    } else {
        cpu_error(); x.cpu_error();
        if (data_type_ == "complex") {
            math_zaxpy(size_, alpha, x.read_h_data().data(), incx, h_data_.data(), incy);
        } else if (data_type_ == "real") {
            math_daxpy(size_, alpha.real(), x.h_re_data_.data(), incx, h_re_data_.data(), incy);
        } else {
            throw std::runtime_error("Unsupported data type in zaxpy (CPU).");
        }
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
    cpu_error(); B.cpu_error();
    if (data_type_ != B.data_type_) throw std::runtime_error("Data type mismatch in gemm.");
    if ((shape_.size() != 2) || (shape_ != B.shape())) throw std::runtime_error("Invalid tensor shapes for gemm.");
    const int M = (transa=='N') ? (int)shape_[0] : (int)shape_[1];
    const int K = (transa=='N') ? (int)shape_[1] : (int)shape_[0];
    const int N = (transb=='N') ? (int)B.shape_[1] : (int)B.shape_[0];
    const int lda = (transa=='N') ? (int)shape_[1] : (int)shape_[0];
    const int ldb = (transb=='N') ? (int)B.shape_[1] : (int)B.shape_[0];
    const int ldc = N; // Row-major storage (consistent with existing zgemm wrapper)
    if (data_type_ == "complex") {
        math_zgemm(transa, transb, M, N, K, alpha, h_data_.data(), lda, B.read_h_data().data(), ldb, beta, h_data_.data(), ldc);
    } else if (data_type_ == "real") {
        if (std::abs(alpha.imag())>1e-14 || std::abs(beta.imag())>1e-14) throw std::runtime_error("Complex alpha/beta on real gemm.");
        math_dgemm(transa, transb, M, N, K, alpha.real(), h_re_data_.data(), lda, B.h_re_data_.data(), ldb, beta.real(), h_re_data_.data(), ldc);
    } else {
        throw std::runtime_error("Unsupported data type in gemm().");
    }
}

std::complex<double> TensorThrust::vector_dot(const TensorThrust& other) const
{
    shape_error(other.shape());
    if (data_type_ != other.data_type_) throw std::runtime_error("Data type mismatch in vector_dot.");

    if (on_gpu_) {
        gpu_error(); other.gpu_error();
        if (data_type_ == "complex") {
            cuDoubleComplex result = thrust::inner_product(thrust::device, d_data_.begin(), d_data_.end(), other.d_data_.begin(), make_cuDoubleComplex(0.0,0.0), complex_add(), complex_dot_product());
            return {cuCreal(result), cuCimag(result)};
        } else if (data_type_ == "real") {
            double result = thrust::inner_product(thrust::device, d_re_data_.begin(), d_re_data_.end(), other.d_re_data_.begin(), 0.0, thrust::plus<double>(), thrust::multiplies<double>());
            return {result, 0.0};
        } else {
            throw std::runtime_error("Unsupported data type in vector_dot (GPU).");
        }
    } else {
        cpu_error(); other.cpu_error();
        if (data_type_ == "complex") {
            std::complex<double> result = 0.0; for (size_t i=0;i<size_;++i) result += std::conj(h_data_[i]) * other.h_data_[i]; return result;
        } else if (data_type_ == "real") {
            double result = 0.0; for (size_t i=0;i<size_;++i) result += h_re_data_[i] * other.h_re_data_[i]; return {result,0.0};
        } else {
            throw std::runtime_error("Unsupported data type in vector_dot (CPU).");
        }
    }
}

TensorThrust TensorThrust::transpose() const
{
    cpu_error(); 
    ndim_error(2);

    TensorThrust T({shape_[1], shape_[0]}, name_ + "_T", false, data_type_);
    if (data_type_ == "complex") {
        for (size_t i=0;i<shape_[0];++i) for (size_t j=0;j<shape_[1];++j) T.h_data_[j*shape_[0]+i] = h_data_[i*shape_[1]+j];
    } else if (data_type_ == "real") {
        for (size_t i=0;i<shape_[0];++i) for (size_t j=0;j<shape_[1];++j) T.h_re_data_[j*shape_[0]+i] = h_re_data_[i*shape_[1]+j];
    } else {
        throw std::runtime_error("Unsupported data type in transpose.");
    }
    return T;
}

TensorThrust TensorThrust::general_transpose(const std::vector<size_t>& axes) const
{
    cpu_error();
    if (axes.size() != ndim()) throw std::runtime_error("Axes size must match tensor dimensions.");
    std::vector<size_t> transposed_shape(ndim()); 

    for (size_t i = 0; i < ndim(); ++i) {
        transposed_shape[i] = shape_[axes[i]];
    }

    TensorThrust transposed_tensor(transposed_shape, name_+"_gt", false, data_type_);
    for (size_t i=0 ; i < size_ ; ++i) { 
        std::vector<size_t> old_tidx = vidx_to_tidx(i); 
        std::vector<size_t> new_tidx(ndim()); 
        
        for (size_t j=0;j<ndim();++j) {
            new_tidx[j]=old_tidx[axes[j]];
        }

        size_t new_vidx = transposed_tensor.tidx_to_vidx(new_tidx); 
        if (data_type_=="complex") {
            transposed_tensor.h_data_[new_vidx]=h_data_[i]; 
        } else if (data_type_=="real") {
            transposed_tensor.h_re_data_[new_vidx]=h_re_data_[i]; 
        } else {
            throw std::runtime_error("Unsupported data type in general_transpose.");
        }
    }
    
    return transposed_tensor;
}

TensorThrust TensorThrust::slice(std::vector<std::pair<size_t, size_t>> idxs) const
{
    cpu_error();
    if (idxs.size() != ndim()) throw std::runtime_error("Number of slice indices must match tensor dimensions.");

    std::vector<size_t> new_shape(idxs.size());
    std::vector<size_t> new_shape2;
    for (size_t i = 0; i < idxs.size(); ++i) {
        new_shape[i] = idxs[i].second - idxs[i].first;
        if (new_shape[i] > 0) new_shape2.push_back(new_shape[i]);
    }
    
    TensorThrust new_tensor(new_shape2, name_+"_sliced", false, data_type_);
    size_t new_vidx = 0;
    for (size_t vidx = 0; vidx < size_; ++vidx) {
        auto tidx = vidx_to_tidx(vidx);
        bool in_slice = true;
        
        for (size_t i = 0; i < ndim(); ++i) {
            if (tidx[i] < idxs[i].first || tidx[i] >= idxs[i].second) {
                in_slice = false;
                break;
            }
        }
        
        if (in_slice) {
            if (data_type_ == "complex") {
                new_tensor.h_data_[new_vidx] = h_data_[vidx];
            } else if (data_type_ == "real") {
                new_tensor.h_re_data_[new_vidx] = h_re_data_[vidx];
            } else {
                throw std::runtime_error("Unsupported data type in slice.");
            }
            new_vidx++;
        }
    }
    return new_tensor;
}

std::vector<std::vector<size_t>> TensorThrust::get_nonzero_tidxs() const
{
    cpu_error();

    std::vector<std::vector<size_t>> nonzero_tidxs;
    if (data_type_ == "complex") {
        for (size_t i = 0; i < size_; ++i) {
            if (std::abs(h_data_[i]) > 1e-12) {
                nonzero_tidxs.push_back(vidx_to_tidx(i));
            }
        }
    } else if (data_type_ == "real") {
        for (size_t i = 0; i < size_; ++i) {
            if (std::abs(h_re_data_[i]) > 1e-12) {
                nonzero_tidxs.push_back(vidx_to_tidx(i));
            }
        }
    } else {
        throw std::runtime_error("Unsupported data type in get_nonzero_tidxs.");
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
    bool reset=false; 
    if (on_gpu_) {
        reset=true;
        to_cpu();
    }
    
    cpu_error();
    std::ostringstream oss; 
    oss << "TensorThrust: " << name_ << "\n"; 
    oss << "  Ndim  = " << ndim() << "\n"; 
    oss << "  Size  = " << size() << "\n"; 
    oss << "  Shape = ("; 
    
    for(size_t dim = 0; dim < ndim(); ++dim) { 
        oss << shape_[dim]; if (dim<ndim()-1) oss << ","; 
    } 
    oss << ")\n"; 
    if (print_data) { 
        oss << "\n  Data:\n\n"; 
        if (size_ > 0) { 
            if (ndim() == 1) { 
                for(size_t i = 0; i < size_; ++i) { 
                    if (data_type_ == "complex") {
                        oss << "  [" << i << "]=" << h_data_[i] << "\n"; 
                    } else if (data_type_ == "real") {
                        oss << "  [" << i << "]=" << h_re_data_[i] << "\n"; 
                    } else {
                        throw std::runtime_error("Unsupported data_type_ in print.");
                    }
                } 
            } else { // simplified print for >1 dims
                for(size_t i = 0; i < size_; ++i) { 
                    if (data_type_ == "complex") {
                        oss << h_data_[i] << " "; 
                    } else if (data_type_ == "real") {
                        oss << h_re_data_[i] << " "; 
                    } else {
                        throw std::runtime_error("Unsupported data_type_ in print.");
                    }

                    if ((i + 1) % shape_.back() == 0) {
                        oss << "\n"; 
                    }
                }
            } 
        } 
    }
    if (reset) to_gpu(); 
    return oss.str();
}

std::string TensorThrust::print_nonzero() const
{
    cpu_error();
    std::ostringstream oss; 
    oss << "\n Nonzero indices and elements of TensorThrust: \n"; 
    oss << " ========================================== \n"; 
    
    if (data_type_=="complex") { 
        for (size_t i = 0; i < size_; ++i) { 
            if (std::abs(h_data_[i]) > 1e-12) { 
                auto tidx = vidx_to_tidx(i); 
                oss << " ("; 

                for (size_t j = 0; j < tidx.size(); ++j) { 
                    oss << tidx[j]; 
                    if (j < tidx.size() - 1) oss << ", "; 
                } 
                oss << ") = " << h_data_[i] << "\n"; 
            } 
        } 
    } else if (data_type_=="real") { 
        for (size_t i = 0; i < size_; ++i) { 
            if (std::abs(h_re_data_[i]) > 1e-12) { 
                auto tidx = vidx_to_tidx(i); 
                oss << " ("; 
                for (size_t j = 0; j < tidx.size(); ++j) { 
                    oss << tidx[j]; 
                    if (j < tidx.size() - 1) oss << ", "; 
                } 
                oss << ") = " << h_re_data_[i] << "\n"; 
            } 
        } 
    } else { 
        throw std::runtime_error("Unsupported data_type_ in print_nonzero()."); 
    }
    return oss.str();
}

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