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

    std::vector<std::complex<double>>& get_host_data();     // Returns a standard representation of the data on the host
    std::string get_name();                                 // Returns the name of the tensor
    size_t get_ndim();                                      // Returns the number of dimensions of the tensor
    size_t get_size();                                      // Returns the total number of elements in the tensor
    double get_norm();                                      // Returns the norm of the tensor
    const std::vector<size_t>& shape() const;               // Returns size in each dimension
    const bool initialized() const;                         // Returns true if the tensor has been initialized
    void TensorGPUThrust::update_strides();                 // Updates the strides based on the shape of the tensor

    void set(const std::vector<size_t>&, 
             const std::complex<double>
             ); // Sets the value at the given indexes
    
    void fill_from_nparray(std::vector<std::complex<double>>, std::vector<size_t>);

    void TensorGPUThrust::ndim_error(const std::vector<size_t>&) const; // Checks if the number of indices matches the number of dimensions

    std::string str(
        bool print_data = true,
        bool print_complex = false,
        int maxcols = 6,
        const std::string& data_format = "%12.7f",
        const std::string& header_format = "%12zu"
    ) const;



private:
    std::string name_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t size_;
    bool initialized_ = 0;
    bool on_gpu_;

    thrust::host_vector<thrust::complex<double>> h_data_;
    thrust::device_vector<thrust::complex<double>> d_vec_;
};