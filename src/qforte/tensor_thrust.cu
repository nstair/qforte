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
    if (on_gpu_) {
        thrust::fill(d_data_.begin(), d_data_.end(), thrust::complex<double>(0, 0));
    } else {
        thrust::fill(h_data_.begin(), h_data_.end(), thrust::complex<double>(0, 0));
    }
}

// Zero the tensor with a specific shape.
void TensorGPUThrust::zero_with_shape(const std::vector<size_t>& shape, bool on_gpu) {
    update_strides(); // Update strides based on the new shape
    resize(shape); // Resize the tensor to the new shape
    zero(); // Fill the tensor with zeros
    on_gpu_ = on_gpu; // Set the on_gpu flag
    initialized_ = true; // Mark the tensor as initialized
}

// Set the data in the tensor at the given indices to the specified complex value.
void TensorGPUThrust::set(const std::vector<size_t>& idxs,
                          const std::complex<double> value) {
    // Check for ndim error
    if (idxs.size() != shape_.size()) {
        throw std::runtime_error("ndim error: indices size does not match tensor shape.");
    }
    
    // Check for out of bounds indices
    for (int i = 0; i < shape_.size(); i++) {
        if (idxs[i] >= shape_[i]) {
            throw std::runtime_error("Index out of bounds for dimension " + std::to_string(i));
        }
    }
    
    // Convert std::complex to thrust::complex
    thrust::complex<double> val = thrust::complex<double>(value.real(), value.imag());
    
    // Calculate linear index using strides_
    size_t vidx = 0;
    for (int i = 0; i < idxs.size(); i++) {
        vidx += idxs[i] * strides_[i];
    }
    
    // Set the value
    h_data_[vidx] = val;
}

// Get the value at the specified indices in the tensor.
std::complex<double> TensorGPUThrust::get(const std::vector<size_t>& idxs) const {
    // Check for ndim error
    if (idxs.size() != shape_.size()) {
        throw std::runtime_error("ndim error: indices size does not match tensor shape.");
    }

    // Check for out of bounds indices
    for (int i = 0; i < shape_.size(); i++) {
        if (idxs[i] >= shape_[i]) {
            throw std::runtime_error("Index out of bounds for dimension " + std::to_string(i));
        }
    }

    // Calculate linear index using strides_
    size_t vidx = 0;
    for (int i = 0; i < idxs.size(); i++) {
        vidx += idxs[i] * strides_[i];
    }

    // Get the value
    return std::complex<double>(h_data_[vidx].real(), h_data_[vidx].imag());
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

TensorGPUThrust TensorGPUThrust::slice(std::vector<std::pair<size_t, size_t>> idxs) const {
    // Check if the number of slice specifications matches the tensor dimensions
    if (idxs.size() != shape_.size()) {
        throw std::runtime_error("slice error: number of slice ranges does not match tensor dimensions");
    }

    // Calculate the shape of the resulting slice
    std::vector<size_t> new_shape;
    for (const auto& range : idxs) {
        // Validate range
        if (range.first >= range.second) {
            throw std::runtime_error("slice error: invalid range (start >= end)");
        }
        if (range.second > shape_[new_shape.size()]) {
            throw std::runtime_error("slice error: range exceeds dimension size");
        }
        
        // Calculate size of this dimension in the result
        new_shape.push_back(range.second - range.first);
    }
    
    // Create result tensor with the new shape
    TensorGPUThrust result(new_shape, name_ + "_slice", on_gpu_);
    
    // For each element in the result tensor...
    std::vector<size_t> result_idx(new_shape.size(), 0);
    bool done = false;
    
    while (!done) {
        // Convert result indices to source tensor indices
        std::vector<size_t> src_idx(shape_.size());
        for (size_t i = 0; i < shape_.size(); i++) {
            src_idx[i] = result_idx[i] + idxs[i].first;
        }
        
        // Copy the value
        result.set(result_idx, this->get(src_idx));
        
        // Increment indices
        for (int i = new_shape.size() - 1; i >= 0; i--) {
            result_idx[i]++;
            if (result_idx[i] < new_shape[i]) {
                break;
            }
            result_idx[i] = 0;
            if (i == 0) {
                done = true;
            }
        }
    }
    
    return result;
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

std::vector<std::complex<double>> TensorGPUThrust::data() {
    if (on_gpu_) {
        throw std::runtime_error("Data is on GPU, cannot return host data.");
    }

    std::vector<std::complex<double>> result(h_data_.size());
    for (size_t i = 0; i < h_data_.size(); ++i) {
        result[i] = std::complex<double>(h_data_[i].real(), h_data_[i].imag());
    }
    return result;
}

std::vector<std::complex<double>> TensorGPUThrust::read_data() const {
    if (on_gpu_) {
        throw std::runtime_error("Data is on GPU, cannot return host data.");
    }

    std::vector<std::complex<double>> result(h_data_.size());
    for (size_t i = 0; i < h_data_.size(); ++i) {
        result[i] = std::complex<double>(h_data_[i].real(), h_data_[i].imag());
    }
    return result;
}