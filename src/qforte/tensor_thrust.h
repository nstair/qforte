#pragma once
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>
std::vector<float> thrust_square(const std::vector<float>& input);

class TensorGPUThrust {
public:
    TensorGPUThrust(const std::vector<size_t>& shape,
                    const std::string& name = "T",
                    bool on_gpu = false);
    TensorGPUThrust();
    ~TensorGPUThrust() = default;

    void to_gpu();
    void to_cpu();

    void add(const TensorGPUThrust&);

    void zero();

    std::vector<std::complex<double>> get_data() const;     // Returns a standard representation of the data on the host
    std::string get_name() const { return name_; }            // Returns the name of the tensor
    size_t ndim() const { return shape_.size(); }            // Returns the number of dimensions of the tensor
    size_t size() const { return size_; }                    // Returns the total number of elements in the tensor
    const std::vector<size_t>& shape() const { return shape_; }  // Returns size in each dimension
    bool initialized() const { return initialized_; }         // Returns true if the tensor has been initialized
    void update_strides();                 // Updates the strides based on the shape of the tensor

    void resize(const std::vector<size_t>& new_shape); // Resizes the tensor to the new shape, preserving data if possible

    void set(const std::vector<size_t>&, 
             const std::complex<double>
             ); // Sets the value at the given indexes
    
    void fill_from_nparray(std::vector<std::complex<double>>, std::vector<size_t>);

    void ndim_error(const std::vector<size_t>&) const; // Checks if the number of indices matches the number of dimensions

    std::string str() const; // Returns a string representation of the tensor



private:
    std::string name_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t size_;
    bool initialized_ = 0;
    bool on_gpu_;

    thrust::host_vector<thrust::complex<double>> h_data_;
    thrust::device_vector<thrust::complex<double>> d_data_;
};