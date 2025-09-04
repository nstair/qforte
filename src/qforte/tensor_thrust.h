#ifndef _tensor_thrust_h_
#define _tensor_thrust_h_

#include <memory>
#include <cstddef>
#include <string>
#include <vector>
#include <iterator>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

#include "qforte-def.h"
#include "tensor.h"

class Tensor;

class TensorThrust {

public:

// => Constructors <= //

/**
 * Constructor: Builds and initializes the Tensor to all zeros.
 *
 * @param shape the shape of the tensor
 * @param name name of the tensor (for printing/filename use)
 **/
TensorThrust(
    const std::vector<size_t>& shape,
    const std::string& name = "T",
    const bool on_gpu = false,
    const std::string& data_type = "complex"
    );

TensorThrust();

~TensorThrust();

void to_gpu();

void to_cpu();

// The next four funcitons are just for testing purposes
void complex_to_soa_gpu();

void soa_to_complex_gpu();

void complex_to_soa_cpu();

void soa_to_complex_cpu();

bool on_gpu() const { return on_gpu_; }

void add(const TensorThrust&);

void zero();

void zero_gpu();

thrust::host_vector<std::complex<double>>& h_data() { return h_data_; }

const thrust::host_vector<std::complex<double>>& read_h_data() const { return h_data_; }

thrust::device_vector<cuDoubleComplex>& d_data() { return d_data_; }

const thrust::device_vector<cuDoubleComplex>& read_d_data() const { return d_data_; }

void add_thrust(const TensorThrust& other);

// Throw if not on GPU
void gpu_error() const;

// Throw if not on CPU
void cpu_error() const;

// Throw if not "complex" data type OR
// Throw if "all" data type but on_complex_ is false
void complex_error() const;

// Throw if on_complex_ is false
void on_complex_error() const;

// Throw if on_complex_ is true
void on_soa_error() const;

// Throw if not "real" data type
void real_error() const;

// Throw if not "soa" data type
void soa_error() const;

// Throw if not "all" data type
void all_error() const;

// Throw if this data type is not the same as other's
void data_type_error(const std::string&) const;

std::string name() const { return name_; }

std::string data_type() const { return data_type_; }

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

std::string str(
    bool print_data = true,
    bool print_complex = false,
    int maxcols = 6,
    const std::string& data_format = "%12.7f",
    const std::string& header_format = "%12zu"
    ); 

/// The offset between consecutive indices within each dimension
const std::vector<size_t>& strides() const { return strides_; }

/// Whether the tensor has been initilized or not 
const bool initialized() const { return initialized_; }

// => Data Accessors <= //

/**
 * The data of this Tensor, using C-style compound indexing. Modifying the
 * elements of this vector will modify the data of this Tensor
 *
 * @return a reference to the vector data of this tensor
 **/
thrust::host_vector<std::complex<double>>& data() { return h_data_; }

// => Setters <= //

/// Set this Tensor's name to @param name
void set_name(const std::string& name) { name_ = name; } 

/// Set this Tensor's strides to @param strides
void set_strides(const std::vector<size_t> strides) { strides_ = strides; } 

/// Set this Tensor to all zeros with @param shape 
void zero_with_shape(
    const std::vector<size_t>& shape, 
    bool on_gpu,
    const std::string& data_type = "complex"
    );

// => Clone Actions <= //

/// Create a new copy of this Tensor (same size and data)
std::shared_ptr<TensorThrust> clone();

void fill_from_np(std::vector<std::complex<double>>, std::vector<size_t>);

/// Set a particular element of this Tensor, specified by idxs
void add_to_element(const std::vector<size_t>& idxs,
         const std::complex<double> val
         );

/// Get a particular element of this Tensor, specified by idxs
std::complex<double> get(const std::vector<size_t>& idxs) const;

/// Get the vector index for this tensor based on the tensor index
size_t tidx_to_vidx(const std::vector<size_t>& tidx) const;

/// Get the vector index for this tensor based on the tensor index, and axes
size_t tidx_to_trans_vidx(const std::vector<size_t>& tidx, const std::vector<size_t>& axes) const;

/// Get the tensor index for this tensor based on the vector index
std::vector<size_t> vidx_to_tidx(size_t vidx) const;

// => Simple Core Actions <= //

/**
 * Set this 2D square Tensor to the identity matrix
 * Throw if not 2D square
 **/
void identity();

/**
 * Set this 2D square Tensor T to 0.5 * (T + T')
 * Throw if not 2D square
 **/
void symmetrize();

/**
 * Set this 2D square Tensor T to 0.5 * (T - T')
 * Throw if not 2D square
 **/
void antisymmetrize();

/**
 * Scale this Tensor by param a
 * @param a the scalar multiplier
 **/
void scale(std::complex<double> a);


/**
 * Gather elements from another 2D TensorThrust into this Tensor using provided index vectors.
 * The indices specify which elements to copy from the source tensor.
 * Throw if shapes or indices are incompatible.
 *
 * @param other Source TensorThrust to gather from
 * @param i_inds Device vector of row indices
 * @param j_inds Device vector of column indices
 **/
void gather_in_2D_gpu(
    const TensorThrust& other,
    const thrust::device_vector<int>& i_inds, //sourcea_dag,
    const thrust::device_vector<int>& j_inds //sourceb_dag,
    );


/**
 * Copy the data of Tensor other to this Tensor
 * @param other Tensor to copy data from
 * Throw if other is not same shape 
 **/
void copy_in(const TensorThrust& other); 

void copy_in_gpu(const TensorThrust& other);

void copy_in_from_tensor(const Tensor& other);

void copy_to_tensor(Tensor& dest) const;

/**
 * Update this Tensor (y) to be y = a * x + b * y
 * Throw if x is not same shape 
 **/
void axpby(const std::shared_ptr<TensorThrust>& x, double a, double b);

/**
 * Subtract one tensor from another
 * Throw if x is not same shape 
 **/
 void subtract(const TensorThrust& other);

/**
 * Compute the dot product between this and other Tensors,
 * by unrolling this and other Tensor and adding sum of products of
 * elements
 *
 * @param other Tensor to take dot product with
 * @return the dot product
 * Throw if other is not same shape 
 **/
std::complex<double> vector_dot(const TensorThrust& other) const;

/**
 * Compute a new copy of this Tensor which is a transpose of this. Works only
 * for matrices. 
 *
 * @return a transposed copy of this
 * Throw if not 2 ndim
 **/
TensorThrust transpose() const;

/**
 * Compute a new copy of this Tensor which is a transpose of this.
 *
 * @return a transposed copy of this according to axes
 **/
TensorThrust general_transpose(const std::vector<size_t>& axes) const;

/**
 * Create a new tensor based off the given sliced indexes.
 * 
 * @param idxs A vector of pairs with the indexes for the respective dimension.
 * @return a new tensor with new shape, size, and data
 * Throw if given too many indexes for the dimensions or if given invalid syntax for indexes.
 **/
TensorThrust slice(std::vector<std::pair<size_t, size_t>> idxs) const;

std::vector<std::vector<size_t>> get_nonzero_tidxs() const;

// => Printing <= //

std::string print_nonzero() const;

// => Error Throwers <= //

/**
 * Throw std::runtime_error if shape != shape()
 * First calls ndim_error(shape_.size())
 **/
void shape_error(const std::vector<size_t>& shape) const;

/**
 * Throw std::runtime_error if not square matrix
 * First calls ndim_error(2)
 **/
void square_error() const;

/// ===============> MATH <===================== ///

void zaxpy(
    const TensorThrust& x, 
    const std::complex<double> alpha,
    const int incx,
    const int incy);

void zaxpby(
    const TensorThrust& x,
    std::complex<double> a,
    std::complex<double> b,
    const int incx,
    const int incy);

void gemm(
    const TensorThrust& B,
    const char transa,
    const char transb,
    const std::complex<double> alpha,
    const std::complex<double> beta,
    const bool multOnRight);

// => Tensor Multiplication/Permutation <= //

/**
 * Performed the chained matrix multiplication:
 *      
 *  C = alpha * As[0]^trans[0] * As[1]^trans[1] * ... + beta * C
 *      
 *  @param As the list of A core Tensors
 *  @param trans the list of transpose arguments
 *  @param C the resultant matrix - if this argument is not provided, C is
 *      allocated and set to zero in the routine
 *  @param alpha the prefactor of the chained multiply
 *  @param beta the prefactor of the register tensor C
 *  @return C - the resultant tensor (for chaining and new allocation)
 **/
static TensorThrust chain(
    const std::vector<TensorThrust>& As,
    const std::vector<bool>& trans,
    std::complex<double> alpha,
    std::complex<double> beta);

static void permute(
    const std::vector<std::string>& Ainds,
    const std::vector<std::string>& Cinds,
    const TensorThrust& A,
    TensorThrust& C2,
    std::complex<double> alpha = 1.0,
    std::complex<double> beta = 0.0);

static void einsum(
    const std::vector<std::string>& Ainds,
    const std::vector<std::string>& Binds,
    const std::vector<std::string>& Cinds,
    const TensorThrust& A,
    const TensorThrust& B,
    TensorThrust& C3,
    std::complex<double> alpha = 1.0,
    std::complex<double> beta = 0.0);

private:

std::string name_;

// Whether the data is real, complex or stored in structure of arrays (SoA) format
// can be "complex" or "real" or "soa" or "all"
// "all" should only be used for testing purposes
std::string data_type_;

std::vector<size_t> shape_;

std::vector<size_t> strides_;

size_t size_;

bool initialized_ = 0;

// The host side data using thrust
thrust::host_vector<std::complex<double>> h_data_;

// The device side data using thrust
thrust::device_vector<cuDoubleComplex> d_data_;

// The real host side data using thrust
thrust::host_vector<double> h_re_data_;

// The real device side data using thrust
thrust::device_vector<double> d_re_data_;

// The imaginary host side data using thrust
thrust::host_vector<double> h_im_data_;

// The imaginary device side data using thrust
thrust::device_vector<double> d_im_data_;

// Whether the data is currently on the GPU
bool on_gpu_;

// If using all, is data complex or soa?
bool on_complex_;

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

#endif // _tensor_thrust_h_
