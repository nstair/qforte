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

class Tensor;

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
    const std::string& name = "T",
    bool on_gpu = false
    );

TensorGPU();

~TensorGPU();

void to_gpu();
void to_cpu();
void add(const TensorGPU&);

void zero();

void zero_gpu();

std::vector<std::complex<double>>& h_data() { return h_data_; }

const std::vector<std::complex<double>>& read_h_data() const { return h_data_; }

cuDoubleComplex* d_data() { return d_data_; }

const cuDoubleComplex* read_d_data() const { return d_data_; }

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


std::string str(
    bool print_data = true,
    bool print_complex = false,
    int maxcols = 6,
    const std::string& data_format = "%12.7f",
    const std::string& header_format = "%12zu"
    ) const; 



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
std::vector<std::complex<double>>& data() { return h_data_; }

// => Setters <= //

/// Set this Tensor's name to @param name
void set_name(const std::string& name) { name_ = name; } 

/// Set this Tensor's strides to @param strides
void set_strides(const std::vector<size_t> strides) { strides_ = strides; } 

/// Set this Tensor to all seros with @param shape 
void zero_with_shape(const std::vector<size_t>& shape, bool on_gpu);

// => Clone Actions <= //

/// Create a new copy of this Tensor (same size and data)
std::shared_ptr<TensorGPU> clone();

void fill_from_np(std::vector<std::complex<double>>, std::vector<size_t>);

/// Set a particular element of tis Tensor, specified by idxs
void add_to_element(const std::vector<size_t>& idxs,
         const std::complex<double> val
         );

/// Get a particular element of tis Tensor, specified by idxs
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
 * Copy the data of Tensor other to this Tensor
 * @param other Tensor to copy data from
 * Throw if other is not same shape 
 * TODO: This is covered by a static Python method, deprecate and remove this function.
 **/
void copy_in(const TensorGPU& other); 

void copy_in_gpu(const TensorGPU& other);

void copy_in_from_tensor(const Tensor& other);

/**
 * Update this Tensor (y) to be y = a * x + b * y
 * Throw if x is not same shape 
 **/
void axpby(const std::shared_ptr<TensorGPU>& x, double a, double b);

/**
 * Subtract one tensor from another
 * Throw if x is not same shape 
 **/
 void subtract(const TensorGPU& other);

/**
 * Compute the dot product between this and other Tensors,
 * by unrolling this and other Tensor and adding sum of products of
 * elements
 *
 * @param other Tensor to take dot product with
 * @return the dot product
 * Throw if other is not same shape 
 **/
// double vector_dot(const std::shared_ptr<Tensor>& other) const;
std::complex<double> vector_dot(const TensorGPU& other) const;

/**
 * Compute a new copy of this Tensor which is a transpose of this. Works only
 * for matrices. 
 *
 * @return a transposed copy of this
 * Throw if not 2 ndim
 **/
TensorGPU transpose() const;

/**
 * Compute a new copy of this Tensor which is a transpose of this.
 *
 * @return a transposed copy of this acording to axes
 **/
TensorGPU general_transpose(const std::vector<size_t>& axes) const;

/**
 * Create a new tensor based off the given sliced indexes.
 * 
 * @param idxs A vector of pairs with the indexes for the respective dimension.
 * @return a new tensor with new shape, size, and data
 * Throw if given too many indexes for the dimensions or if given invalid syntax for indexes.
 **/
TensorGPU slice(std::vector<std::pair<size_t, size_t>> idxs) const;

std::vector<std::vector<size_t>> get_nonzero_tidxs() const;

// => Printing <= //


std::string print_nonzero() const;


/**
 * Print string representation of this Tensor
 **/
// void print() const;

/**
 * Print string representation of this Tensor with name
 **/
// void print(const std::string& name);

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
    const TensorGPU& x, 
    const std::complex<double> alpha,
    const int incx,
    const int incy);

void zaxpby(
    const TensorGPU& x,
    std::complex<double> a,
    std::complex<double> b,
    const int incx,
    const int incy);

void gemm(
    const TensorGPU& B,
    const char transa,
    const char transb,
    const std::complex<double> alpha,
    const std::complex<double> beta,
    const bool multOnRight);

/// NICK: Comment out the functions below for now, will need external lib
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
static TensorGPU chain(
    const std::vector<TensorGPU>& As,
    const std::vector<bool>& trans,
    // const Tensor& C = Tensor(),
    std::complex<double> alpha,
    std::complex<double> beta);

// static Tensor permute(
static void permute(
    const std::vector<std::string>& Ainds,
    const std::vector<std::string>& Cinds,
    const TensorGPU& A,
    // const Tensor& C2 = Tensor(), // This again, ability to have uninitialized tensor
    TensorGPU& C2,
    std::complex<double> alpha = 1.0,
    std::complex<double> beta = 0.0);

static void einsum(
    const std::vector<std::string>& Ainds,
    const std::vector<std::string>& Binds,
    const std::vector<std::string>& Cinds,
    const TensorGPU& A,
    const TensorGPU& B,
    // const Tensor& C3 = Tensor(),
    TensorGPU& C3,
    std::complex<double> alpha = 1.0,
    std::complex<double> beta = 0.0);


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

bool on_gpu_;


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