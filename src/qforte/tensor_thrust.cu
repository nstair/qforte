#include "tensor_thrust.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <sstream>  // For std::ostringstream


// Constructor for TensorGPUThrust
// Initializes an empty tensor with the given shape and name.
TensorGPUThrust::TensorGPUThrust()
    : name_("T"), shape_({0}), size_(0), on_gpu_(false)
{
    // No data to allocate
    h_data_.clear();
    d_data_.clear();
}

// Constructor for TensorGPUThrust with shape and name
// Initializes the tensor with the given shape and name, allocating memory.
// The tensor is initialized with complex zeros.
TensorGPUThrust::TensorGPUThrust(const std::vector<size_t>& shape,
                                 const std::string& name,
                                 bool on_gpu)
    : name_(name), shape_(shape), on_gpu_(on_gpu) {
    size_ = 1;
    for (auto s : shape) size_ *= s;
    h_data_.resize(size_);
    d_data_.resize(size_);
    thrust::fill(h_data_.begin(), h_data_.end(), thrust::complex<double>(0, 0));
    initialized_ = true;

    // set strides based on shape
    update_strides();
}


void TensorGPUThrust::to_gpu() {
    if (!on_gpu_) {
        // Copy data from host to device
        d_data_ = h_data_;
        on_gpu_ = true;
    }
}


void TensorGPUThrust::to_cpu() {
    if (on_gpu_) {
        // Copy data from device to host
        h_data_ = d_data_;
        on_gpu_ = false;
    }
}


void TensorGPUThrust::resize(const std::vector<size_t>& new_shape) {
    // Compute the new size
    size_t new_size = 1;
    for (auto s : new_shape) new_size *= s;

    // Resize the host and device vectors
    h_data_.resize(new_size);
    d_data_.resize(new_size);

    // Update the shape and size
    shape_ = new_shape;
    size_ = new_size;

    // Update the strides
    update_strides();
}

void TensorGPUThrust::zero() {
    // Fill the host vector with complex zeros
    thrust::fill(h_data_.begin(), h_data_.end(), thrust::complex<double>(0, 0));
}

// Set the data in the tensor given list of indices and a value for those indices.
void TensorGPUThrust::set(const std::vector<size_t>& idxs,
                          const std::complex<double> value) {
    // Check for ndim error
    if (idxs.size() != shape_.size()) {
        throw std::runtime_error("ndim error: indices size does not match tensor shape.");
    }
    thrust::complex<double> val = thrust::complex<double>(value.real(), value.imag());

    if( idxs.size() == 1 ) {
        h_data_[idxs[0]] = val;
    } else if (idxs.size() == 2) {
        h_data_[shape_[1]*idxs[0] + idxs[1]] = val;
    } else {
        for (int i = 0; i < shape_.size(); i++) {
            if (idxs[i] >= shape_[i]) {
                std::cerr << "Index out of bounds for dimension " << i << std::endl;
            }
        }      
        size_t vidx = 0;
        size_t stride = 1;
        
        for (int i = shape_.size() - 1; i >= 0; i--) {
            vidx += idxs[i] * stride;
            stride *= shape_[i];
        }
        h_data_[vidx] = val;
        h_data_[vidx] = val;
    }
}

// Fill the tensor with a numpy array.
void TensorGPUThrust::fill_from_nparray(std::vector<std::complex<double>> data, std::vector<size_t> shape) {
    if (shape_ != shape){
        throw std::runtime_error("The Shapes are not the same.");
    }

    for (size_t i = 0; i < data.size(); ++i) {
        h_data_[i] = thrust::complex<double>(data[i].real(), data[i].imag());
    }
}

void TensorGPUThrust::add(const TensorGPUThrust& other) {
    if (size_ != other.size_) {
        throw std::runtime_error("Tensor sizes do not match for addition.");
    }
    thrust::transform(
        d_data_.begin(), d_data_.end(),
        other.d_data_.begin(),
        d_data_.begin(),
        thrust::plus<thrust::complex<double>>());
}

std::vector<std::complex<double>> TensorGPUThrust::get_data() const {
    std::vector<std::complex<double>> out_data(size_);

    for (size_t i = 0; i < out_data.size(); ++i) {
        out_data[i] = std::complex<double>(h_data_[i].real(), h_data_[i].imag());
    }
    return out_data;
}

void TensorGPUThrust::ndim_error(const std::vector<size_t>& idxs) const {
    if (idxs.size() != shape_.size()) {
        throw std::runtime_error("ndim error: indices size does not match tensor shape.");
    }
}

void TensorGPUThrust::update_strides() {
    strides_.resize(shape_.size());
    strides_[shape_.size() - 1] = 1;
    for (int i = shape_.size() - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

std::string TensorGPUThrust::str() const {
    std::ostringstream oss;
    oss << "TensorGPUThrust: " << name_ << "\n";
    oss << "Shape: ";
    for (size_t s : shape_) {
        oss << s << " ";
    }
    oss << "\n";
    oss << "Data: ";
    for (size_t i = 0; i < h_data_.size(); ++i) {
        oss << "(" << h_data_[i].real() << ", " << h_data_[i].imag() << ") ";
    }
    return oss.str();
}