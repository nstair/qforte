#pragma once
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>

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

    void set_name(const std::string& name) { name_ = name; } // Sets the name of the tensor
    std::string get_name() const { return name_; }            // Returns the name of the tensor
    size_t ndim() const { return shape_.size(); }            // Returns the number of dimensions of the tensor
    size_t size() const { return size_; }                    // Returns the total number of elements in the tensor
    const std::vector<size_t>& shape() const { return shape_; }  // Returns size in each dimension
    bool initialized() const { return initialized_; }         // Returns true if the tensor has been initialized
    void update_strides();                 // Updates the strides based on the shape of the tensor

    std::vector<std::complex<double>> data();              // Returns a std copy of the data (host)
    std::vector<std::complex<double>> read_data() const;    // Returns a std copy of the data (host)

    void resize(const std::vector<size_t>& new_shape); // Resizes the tensor to the new shape, preserving data if possible

    void set(const std::vector<size_t>&, 
             const std::complex<double>
             ); // Sets the value at the given index

    std::complex<double> get(const std::vector<size_t>&) const; // Gets the value at the given index

    void fill_from_nparray(std::vector<std::complex<double>>, std::vector<size_t>);

    void zero_with_shape(const std::vector<size_t>& shape, bool on_gpu);

    TensorGPUThrust slice(std::vector<std::pair<size_t, size_t>> idxs) const;

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