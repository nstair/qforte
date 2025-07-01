#include "tensor_thrust.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>

std::vector<float> thrust_square(const std::vector<float>& input) {
    thrust::device_vector<float> d_input(input);
    thrust::device_vector<float> d_output(input.size());
    thrust::transform(d_input.begin(), d_input.end(), d_output.begin(),
                      [] __device__ (float x) { return x * x; });
    std::vector<float> output(input.size());
    thrust::copy(d_output.begin(), d_output.end(), output.begin());
    return output;
}


// Constructor for TensorGPUThrust
// Initializes an empty tensor with the given shape and name.
TensorGPUThrust::TensorGPUThrust()
    : name_("T"), shape_({0}), size_(0), on_gpu_(false)
{
    // No data to allocate
    h_data_.clear();
    d_vec_.clear();
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
    d_vec_.resize(size_);
    thrust::fill(d_vec_.begin(), d_vec_.end(), thrust::complex<double>(0, 0));
    initialized_ = true;

    // set strides based on shape
    update_strides();
}

// Set the data in the tensor given list of indices and a value for those indices.
void TensorGPUThrust::set_data(const std::vector<size_t>& idxs,
                               const std::complex<double>& values) {
    // Check for ndim error
    ndim_error(idxs.size());

    size_t flat_index = 0;
    for (size_t d = 0; d < idxs.size(); ++d) {
        if (idxs[d] >= shape_[d]) {
            throw std::runtime_error("Index out of bounds in set_data.");
        }
        flat_index += idxs[d] * strides_[d];
    }

    data_[flat_index] = value;
}

void TensorGPUThrust::add(const TensorGPUThrust& other) {
    if (size_ != other.size_) {
        throw std::runtime_error("Tensor sizes do not match for addition.");
    }
    thrust::transform(
        d_vec_.begin(), d_vec_.end(),
        other.d_vec_.begin(),
        d_vec_.begin(),
        thrust::plus<thrust::complex<double>>());
}

std::vector<thrust::complex<double>> TensorGPUThrust::get_host_data() const {
    std::vector<thrust::complex<double>> out(size_);
    thrust::copy(d_vec_.begin(), d_vec_.end(), out.begin());
    return out;
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
