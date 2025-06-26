# GPU Acceleration in QForte

## Overview

QForte is implementing GPU acceleration to enhance computational performance for quantum simulation tasks. The implementation focuses on offloading computationally intensive operations to NVIDIA GPUs using CUDA. This document outlines the current GPU implementation in QForte, its architecture, and how it accelerates quantum simulations.

## Key Components

### 1. TensorGPU Class

The `TensorGPU` class is a central component for GPU acceleration in QForte. It extends the functionality of the standard `Tensor` class by providing GPU-specific operations:

```cpp
class TensorGPU {
public:
    // Constructors
    TensorGPU(const std::vector<size_t>& shape, const std::string& name = "T", bool on_gpu = false);
    TensorGPU();
    ~TensorGPU();

    // GPU data transfer methods
    void to_gpu();  // Transfer data from CPU to GPU
    void to_cpu();  // Transfer data from GPU to CPU
    
    // GPU operations
    void add(const TensorGPU&);  // Add another tensor (CPU implementation)
    void add2(const TensorGPU&); // Add another tensor (GPU implementation)
    void zero();                 // Set all elements to zero (CPU)
    void zero_gpu();             // Set all elements to zero (GPU)
    
    // Other operations
    void scale(std::complex<double> a);
    void copy_in(const TensorGPU& other);
    void copy_in_gpu(const TensorGPU& other);
    void copy_in_from_tensor(const Tensor& other);
    void subtract(const TensorGPU& other);
    // ... many more mathematical operations
};
```

The `TensorGPU` class maintains both host-side data (`h_data_`) and device-side data (`d_data_`), with methods to transfer data between them. It tracks whether the tensor's data currently resides on the GPU with the `on_gpu_` flag.

### 2. FCIComputerGPU Class

The `FCIComputerGPU` class implements a GPU-accelerated version of the Full Configuration Interaction (FCI) quantum computer:

```cpp
class FCIComputerGPU {
public:
    // Constructor
    FCIComputerGPU(int nel, int sz, int norb, bool on_gpu = false);
    
    // Data transfer methods
    void to_gpu();
    void to_cpu();
    
    // Error checking
    void gpu_error() const;
    void cpu_error() const;
    
    // Apply operators
    void apply_sqop(const SQOperator& sqop);
    void apply_tensor_spin_1bdy(const TensorGPU& h1e, size_t norb);
    void apply_tensor_spat_12bdy(const TensorGPU& h1e, const TensorGPU& h2e, const TensorGPU& h2e_einsum, size_t norb);
    // ... and more
};
```

This class implements GPU-accelerated versions of operations like applying quantum operators, evolving quantum states, and calculating expectation values.

### 3. CUDA Kernels

The implementation uses several CUDA kernels to accelerate operations:

1. **Tensor Addition Kernel**: Implemented in `tensor_gpu_kernels.cuh` for element-wise addition of tensors.

    ```cpp
    __global__ void add_kernel2(cuDoubleComplex* x, const cuDoubleComplex* y, size_t n);
    ```

2. **FCI Computer Operations**: Implemented in `fci_computer_gpu.cuh` for applying operators to quantum states.

    ```cpp
    __global__ void apply_individual_nbody1_accumulate_kernel(
        const cuDoubleComplex coeff, 
        const cuDoubleComplex* d_Cin, 
        cuDoubleComplex* d_Cout, 
        const int* d_sourcea,
        const int* d_targeta,
        const cuDoubleComplex* d_paritya,
        const int* d_sourceb,
        const int* d_targetb,
        const cuDoubleComplex* d_parityb,
        int nbeta_strs_,
        int targeta_size,
        int targetb_size,
        int tensor_size);
    ```

## Implementation Details

### Memory Management

The GPU implementation carefully manages memory transfers between CPU and GPU:

1. **Allocation**: CUDA memory is allocated during object construction:
   ```cpp
   cudaMalloc((void**) & d_data_, size_ * sizeof(std::complex<double>));
   ```

2. **Transfer to GPU**:
   ```cpp
   cudaMemcpy(d_data_, h_data_.data(), size_ * sizeof(std::complex<double>), cudaMemcpyHostToDevice);
   ```

3. **Transfer from GPU**:
   ```cpp
   cudaMemcpy(h_data_.data(), d_data_, size_ * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
   ```

4. **Deallocation**: Memory is freed in the destructor:
   ```cpp
   cudaFree(d_data_);
   ```

### Performance Monitoring

The implementation includes performance monitoring with the `local_timer` class:

```cpp
local_timer my_timer = local_timer();
timer_.reset();
// GPU operation
timer_.acc_record("operation_name");
```

This allows developers to track the performance of different operations and identify bottlenecks.

## Core GPU-Accelerated Operations

### 1. Tensor Operations

Basic tensor operations accelerated on the GPU include:

- Element-wise addition (`add2`)
- Scaling (`scale`)
- Matrix multiplication (`gemm`)
- Vector dot product (`vector_dot`)
- Tensor transposition (`transpose`)

### 2. Quantum Operations

The most computationally intensive quantum operations now accelerated on GPU include:

- Applying second-quantized operators (`apply_sqop`)
- Time evolution under Hamiltonians (`evolve_pool_trotter`)
- Calculating expectation values (`get_exp_val`)
- Applying one-body and two-body operators (`apply_tensor_spat_12bdy`)

## Performance Impact

The GPU acceleration provides significant performance improvements for large quantum simulations, particularly for:

1. Systems with many qubits/orbitals
2. Operations involving dense matrices
3. Time evolution simulations with many trotter steps

## Development Status

The GPU acceleration in QForte is currently under active development. The implementation focuses on:

1. Porting more operations to run on the GPU
2. Optimizing memory transfers between CPU and GPU
3. Improving the parallelization of CUDA kernels
4. Adding support for more complex quantum operations

## Future Work

Planned improvements to the GPU acceleration include:

1. Support for multiple GPUs
2. Optimization for specific GPU architectures
3. Implementation of specialized quantum algorithms on the GPU
4. Better integration with other quantum simulation libraries

## Thrust Library Integration

The Thrust library provides a high-level, template-based interface for CUDA programming that can significantly simplify GPU code development. Replacing raw CUDA code with Thrust equivalents can lead to more maintainable, efficient, and concise code. The following files should be targeted for Thrust integration:

### Core CUDA Implementation Files

1. `/home/zach_gonzales/qforte/src/qforte/tensor_gpu.cu`
   - Contains raw CUDA kernels for tensor operations
   - Candidates for Thrust replacement: memory allocation, vector operations, reductions

2. `/home/zach_gonzales/qforte/src/qforte/tensor_gpu_kernels.cuh`
   - Header defining CUDA kernel interfaces
   - Should be updated to use Thrust algorithms where applicable

3. `/home/zach_gonzales/qforte/src/qforte/fci_computer_gpu.cu`
   - Implements FCI computer operations on GPU
   - Key targets: vector operations, reductions, transforms

4. `/home/zach_gonzales/qforte/src/qforte/fci_computer_gpu.cuh`
   - Header file for FCI GPU operations
   - Update to reflect Thrust-based implementations

### Supporting C++ Files with CUDA Calls

5. `/home/zach_gonzales/qforte/src/qforte/tensor_gpu.cc`
   - Contains raw CUDA memory operations (cudaMalloc, cudaMemcpy)
   - Replace with Thrust device_vector/host_vector for automatic memory management

6. `/home/zach_gonzales/qforte/src/qforte/fci_computer_gpu.cc`
   - Contains multiple raw CUDA memory allocations and kernel launches
   - Can leverage Thrust for more concise and safer operations

### Key Thrust Replacements

Here are examples of raw CUDA code that can be replaced with Thrust equivalents:

1. **Memory Management**
   - Replace: `cudaMalloc`, `cudaFree`, `cudaMemcpy`
   - With: `thrust::device_vector`, `thrust::host_vector`, and their copy constructors

2. **Vector Operations**
   - Replace: Custom CUDA kernels for element-wise operations
   - With: `thrust::transform`, `thrust::for_each`

3. **Reductions**
   - Replace: Custom reduction kernels
   - With: `thrust::reduce`, `thrust::transform_reduce`

4. **Sorting and Searching**
   - Replace: Custom sorting implementations
   - With: `thrust::sort`, `thrust::lower_bound`, `thrust::binary_search`

### Example Transformation

Current raw CUDA implementation for tensor addition:

```cpp
__global__ void add_kernel(cuDoubleComplex* x, const cuDoubleComplex* y, size_t n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        x[index] = cuCadd(x[index], y[index]);
    }
}

// Host-side wrapper
void add_wrapper(cuDoubleComplex* x, const cuDoubleComplex* y, size_t n, int threadsPerBlock) {
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    add_kernel<<<blocksPerGrid, threadsPerBlock>>>(x, y, n);
}
```

Equivalent Thrust implementation:

```cpp
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

// Using Thrust's transform function with plus operator
void add_wrapper_thrust(thrust::device_vector<cuDoubleComplex>& x, 
                        const thrust::device_vector<cuDoubleComplex>& y) {
    thrust::transform(x.begin(), x.end(), y.begin(), x.begin(), 
                     thrust::plus<cuDoubleComplex>());
}
```

## Usage Example

Here's an example of using the GPU-accelerated features:

```python
import qforte as qf
import numpy as np

# Create an FCI Computer with GPU support
fci_comp_gpu = qf.FCIComputerGPU(nel=14, sz=0, norb=14)

# Initialize with a random state
random_array = np.random.rand(fci_comp_gpu.get_state().shape()[0], fci_comp_gpu.get_state().shape()[1])
random = np.array(random_array, dtype=np.dtype(np.complex128))
Crand = qf.Tensor(fci_comp_gpu.get_state().shape(), "Crand")
Crand.fill_from_nparray(random.ravel(), Crand.shape())
rand_nrm = Crand.norm()
Crand.scale(1/rand_nrm)
fci_comp_gpu.set_state_from_tensor(Crand)

# Transfer to GPU
fci_comp_gpu.to_gpu()

# Create a second-quantized operator
sqop = qf.SQOperator()
sqop.add_term(5.5, [4, 5], [1, 0])

# Apply the operator (runs on GPU)
fci_comp_gpu.apply_sqop(sqop)

# Transfer back to CPU
fci_comp_gpu.to_cpu()
```

## Conclusion

The GPU acceleration in QForte significantly enhances the performance of quantum simulations, allowing for the simulation of larger and more complex quantum systems. As the implementation matures, it will enable researchers to explore quantum phenomena that were previously computationally prohibitive.
