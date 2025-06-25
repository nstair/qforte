#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/copy.h>

// Simple multiply-by-two functor
struct multiply_by_two {
    __host__ __device__
    float operator()(const float& x) const {
        return x * 2.0f;
    }
};
int main() {
    // Create a host vector with 5 elements
    thrust::host_vector<float> h_vec(5);
    
    // Initialize the vector
    h_vec[0] = 1.0f;
    h_vec[1] = 2.0f;
    h_vec[2] = 3.0f;
    h_vec[3] = 4.0f;
    h_vec[4] = 5.0f;
    
    // Print the host vector
    std::cout << "Host vector: ";
    for (size_t i = 0; i < h_vec.size(); i++) {
        std::cout << h_vec[i] << " ";
    }
    std::cout << std::endl;
    
    // Copy data to device
    thrust::device_vector<float> d_vec = h_vec;
    
    // Multiply each value by 2 on the device
    thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(),
                     multiply_by_two());
    
    // Copy result back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
    
    // Print the result
    std::cout << "After multiplying by 2: ";
    for (size_t i = 0; i < h_vec.size(); i++) {
        std::cout << h_vec[i] << " ";
    }
    std::cout << std::endl;
    
    // Add values of two vectors
    thrust::device_vector<float> d_vec2(5, 10.0f);  // Initialize with 10.0
    thrust::device_vector<float> d_result(5);
    
    thrust::transform(d_vec.begin(), d_vec.end(), d_vec2.begin(), d_result.begin(),
                     thrust::plus<float>());
    
    // Copy the result back to host
    thrust::host_vector<float> h_result = d_result;
    
    // Print the result
    std::cout << "After adding 10 to each element: ";
    for (size_t i = 0; i < h_result.size(); i++) {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Thrust test completed successfully!" << std::endl;
    return 0;
}
