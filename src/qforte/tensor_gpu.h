#ifndef _tensor_gpu_h_
#define _tensor_gpu_h_

#include <memory>
#include <cstddef>
#include <string>
#include <vector>
#include <iterator>

#include <cuda_runtime.h>
#include <cuComplex.h>


#include "qforte-def.h"
#include "tensor.h"

class TensorGPU {

public:

// => Constructors <= //

/**
 * Constructor: Builds and initializes the Tensor to all zeros.
 *
 * @param shape the shape of the tensor
 * @param name name of the tensor (for printing/filename use)
 **/
TensorGPU(
    const std::vector<size_t>& shape,
    const std::string& name = "T"
    );

TensorGPU();

~TensorGPU();

void to_gpu();
void to_cpu();
void add(const TensorGPU&);
void zero();
std::vector<std::complex<double>>& read_data() { return h_data_; }
// std::complex<double>* get_d_data() const { return d_data_; }
cuDoubleComplex* get_d_data() const { return d_data_; }

void add2(const TensorGPU& other);

void gpu_error() const;

void cpu_error() const;

std::string name() const { return name_; }

/// The number of dimensions of this Tensor, inferred from shape
size_t ndim() const { return shape_.size(); }

/// The total number of elements of this Tensor (the product of shape)
size_t size() const { return size_; }

double norm();

/// The size in each dimension of this Tensor
const std::vector<size_t>& shape() const { return shape_; }

void set(const std::vector<size_t>& idxs,
         const std::complex<double> val
         );

void ndim_error(size_t) const;

void fill_from_nparray(std::vector<std::complex<double>>, std::vector<size_t>);

const std::vector<std::complex<double>>& read_h_data() const { return h_data_; }


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

// The host side data
std::vector<std::complex<double>> h_data_;

// The device side data pointer
// std::complex<double>* d_data_; 

cuDoubleComplex* d_data_;

int on_gpu_ = 0;


// => Ed's special total memory thing <= //

private: 

static size_t total_memory__;

public:

/**
 * Current total global memory usage of Tensor in bytes. 
 * Computes as t.size() * sizeof(double) for all tensors t that are currently
 * in scope.
 **/
static size_t total_memory() { return total_memory__; }

};

// } // namespace lightspeed

#endif // _tensor_gpu_h_