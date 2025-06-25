#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <complex>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/complex.h>

namespace py = pybind11;

// Convert std::complex to thrust::complex
struct to_thrust_complex {
    __host__ __device__
    thrust::complex<double> operator()(const std::complex<double>& c) const {
        return thrust::complex<double>(c.real(), c.imag());
    }
};

// Convert thrust::complex to std::complex
struct to_std_complex {
    __host__ __device__
    std::complex<double> operator()(const thrust::complex<double>& c) const {
        return std::complex<double>(c.real(), c.imag());
    }
};

// Add two complex vectors using Thrust
std::vector<std::complex<double>> add_complex_thrust(
    const std::vector<std::complex<double>>& a, 
    const std::vector<std::complex<double>>& b) 
{
    int n = a.size();
    std::vector<std::complex<double>> result(n);
    
    // Create thrust device vectors
    thrust::device_vector<thrust::complex<double>> d_a(n);
    thrust::device_vector<thrust::complex<double>> d_b(n);
    thrust::device_vector<thrust::complex<double>> d_result(n);
    
    // Copy data to device vectors and convert to thrust::complex
    for (int i = 0; i < n; i++) {
        d_a[i] = thrust::complex<double>(a[i].real(), a[i].imag());
        d_b[i] = thrust::complex<double>(b[i].real(), b[i].imag());
    }
    
    // Perform addition
    thrust::transform(
        d_a.begin(), d_a.end(), 
        d_b.begin(), 
        d_result.begin(), 
        thrust::plus<thrust::complex<double>>()
    );
    
    // Copy result back to host
    for (int i = 0; i < n; i++) {
        thrust::complex<double> val = d_result[i];
        result[i] = std::complex<double>(val.real(), val.imag());
    }
    
    return result;
}

// Multiply two complex vectors using Thrust
std::vector<std::complex<double>> multiply_complex_thrust(
    const std::vector<std::complex<double>>& a, 
    const std::vector<std::complex<double>>& b) 
{
    int n = a.size();
    std::vector<std::complex<double>> result(n);
    
    // Create thrust device vectors
    thrust::device_vector<thrust::complex<double>> d_a(n);
    thrust::device_vector<thrust::complex<double>> d_b(n);
    thrust::device_vector<thrust::complex<double>> d_result(n);
    
    // Copy data to device vectors and convert to thrust::complex
    for (int i = 0; i < n; i++) {
        d_a[i] = thrust::complex<double>(a[i].real(), a[i].imag());
        d_b[i] = thrust::complex<double>(b[i].real(), b[i].imag());
    }
    
    // Perform multiplication
    thrust::transform(
        d_a.begin(), d_a.end(), 
        d_b.begin(), 
        d_result.begin(), 
        thrust::multiplies<thrust::complex<double>>()
    );
    
    // Copy result back to host
    for (int i = 0; i < n; i++) {
        thrust::complex<double> val = d_result[i];
        result[i] = std::complex<double>(val.real(), val.imag());
    }
    
    return result;
}

// Scale a complex vector using Thrust
std::vector<std::complex<double>> scale_complex_thrust(
    const std::vector<std::complex<double>>& a, 
    const std::complex<double>& scalar) 
{
    int n = a.size();
    std::vector<std::complex<double>> result(n);
    
    // Create thrust device vectors
    thrust::device_vector<thrust::complex<double>> d_a(n);
    thrust::device_vector<thrust::complex<double>> d_result(n);
    
    // Copy data to device vectors and convert to thrust::complex
    for (int i = 0; i < n; i++) {
        d_a[i] = thrust::complex<double>(a[i].real(), a[i].imag());
    }
    
    // Create scalar in thrust format
    thrust::complex<double> t_scalar(scalar.real(), scalar.imag());
    
    // Create scaling functor
    auto scale_functor = [t_scalar] __device__ (const thrust::complex<double>& x) {
        return x * t_scalar;
    };
    
    // Perform scaling
    thrust::transform(
        d_a.begin(), d_a.end(), 
        d_result.begin(), 
        scale_functor
    );
    
    // Copy result back to host
    for (int i = 0; i < n; i++) {
        thrust::complex<double> val = d_result[i];
        result[i] = std::complex<double>(val.real(), val.imag());
    }
    
    return result;
}

// Version with numpy arrays
py::array_t<std::complex<double>> add_complex_numpy(
    py::array_t<std::complex<double>> a, 
    py::array_t<std::complex<double>> b) 
{
    // Check input dimensions
    if (a.ndim() != 1 || b.ndim() != 1)
        throw std::runtime_error("Input arrays must be 1-dimensional");
    if (a.size() != b.size())
        throw std::runtime_error("Input arrays must have the same size");
    
    // Create output array
    auto result = py::array_t<std::complex<double>>(a.size());
    
    // Convert to std::vector for easier handling
    std::vector<std::complex<double>> vec_a(a.size());
    std::vector<std::complex<double>> vec_b(b.size());
    
    // Get array buffers
    auto buf_a = a.request();
    auto buf_b = b.request();
    auto buf_result = result.request();
    
    // Copy data to vectors
    std::complex<double>* ptr_a = static_cast<std::complex<double>*>(buf_a.ptr);
    std::complex<double>* ptr_b = static_cast<std::complex<double>*>(buf_b.ptr);
    
    for (size_t i = 0; i < a.size(); i++) {
        vec_a[i] = ptr_a[i];
        vec_b[i] = ptr_b[i];
    }
    
    // Perform computation using Thrust
    auto vec_result = add_complex_thrust(vec_a, vec_b);
    
    // Copy result back to numpy array
    std::complex<double>* ptr_result = static_cast<std::complex<double>*>(buf_result.ptr);
    for (size_t i = 0; i < vec_result.size(); i++) {
        ptr_result[i] = vec_result[i];
    }
    
    return result;
}

PYBIND11_MODULE(thrust_test_py, m) {
    m.doc() = "Test module for Thrust/CUDA in Python";
    
    // Add functions
    m.def("add_complex", &add_complex_thrust, "Add two complex vectors using Thrust/CUDA");
    m.def("multiply_complex", &multiply_complex_thrust, "Multiply two complex vectors using Thrust/CUDA");
    m.def("scale_complex", &scale_complex_thrust, "Scale a complex vector using Thrust/CUDA");
    
    // NumPy array versions
    m.def("add_complex_numpy", &add_complex_numpy, "Add two complex NumPy arrays using Thrust/CUDA");
    
    // Version info
    m.attr("__version__") = "0.1.0";
}
