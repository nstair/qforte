#include "fci_computer_gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <algorithm>
#include <limits>
#include <stdexcept>

// ==============================================
// Error checking macros
// =============================================

#define CHECK_CUDA(call) do {                                  \
  cudaError_t err_ = (call);                                   \
  if (err_ != cudaSuccess) {                                   \
    std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__  \
              << " : " << cudaGetErrorString(err_) << "\n";    \
    throw std::runtime_error("CUDA failure");                  \
  }                                                            \
} while(0)

#define CHECK_CUSPARSE(call) do {                              \
  cusparseStatus_t st_ = (call);                               \
  if (st_ != CUSPARSE_STATUS_SUCCESS) {                        \
    std::cerr << "cuSPARSE error " << __FILE__ << ":"          \
              << __LINE__ << " : " << (int)st_ << "\n";        \
    throw std::runtime_error("cuSPARSE failure");              \
  }                                                            \
} while(0)

// ==============================================
// Functor for cuDoubleComplex addition - used with thrust
// ==============================================

struct cuCadd_op {
  __host__ __device__
  cuDoubleComplex operator()(const cuDoubleComplex& a,
                             const cuDoubleComplex& b) const {
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
  }
};

// ==============================================
// Original Implementation:
// Only keeping to support non - precomp version
// Should either be updated to use Givens or removed
// ==============================================

// Helper function for atomic add with double precision
__device__ double atomicAdd_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// V2_atomic - thread-safe version using atomicAdd to prevent race conditions
__global__ void apply_individual_nbody1_accumulate_kernel_atomic(
    const cuDoubleComplex coeff, 
    const cuDoubleComplex* d_Cin, 
    cuDoubleComplex* d_Cout, 
    const int* d_sourcea,
    const int* d_targeta,
    const cuDoubleComplex* d_paritya,
    const int* d_sourceb,
    const int* d_targetb,
    const cuDoubleComplex* d_parityb,
    long long nbeta_strs_,
    int targeta_size,
    int targetb_size,
    long long tensor_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < targeta_size) {
        long long ta_idx = (long long)d_targeta[idx] * nbeta_strs_;
        long long sa_idx = (long long)d_sourcea[idx] * nbeta_strs_;
        
        cuDoubleComplex pref = cuCmul(coeff, d_paritya[idx]);

        if (idy < targetb_size) {
            cuDoubleComplex term = cuCmul(pref, d_parityb[idy]);
            term = cuCmul(term, d_Cin[sa_idx + d_sourceb[idy]]);

            // Thread-safe atomic accumulation
            long long output_idx = ta_idx + d_targetb[idy];
            atomicAdd_double(&d_Cout[output_idx].x, term.x);
            atomicAdd_double(&d_Cout[output_idx].y, term.y);
        }
    }
}

void apply_individual_nbody1_accumulate_wrapper(
    const cuDoubleComplex coeff, 
    const cuDoubleComplex* d_Cin, 
    cuDoubleComplex* d_Cout, 
    const int* d_sourcea,
    const int* d_targeta,
    const cuDoubleComplex* d_paritya,
    const int* d_sourceb,
    const int* d_targetb,
    const cuDoubleComplex* d_parityb,
    long long nbeta_strs_,
    int targeta_size,
    int targetb_size,
    long long tensor_size)
{
    // 2D grid configuration for the atomic kernel
    dim3 blockSize(16, 16);  // 16x16 = 256 threads per block
    dim3 gridSize((targeta_size + blockSize.x - 1) / blockSize.x,
                  (targetb_size + blockSize.y - 1) / blockSize.y);
    
    apply_individual_nbody1_accumulate_kernel_atomic<<<gridSize, blockSize>>>(
        coeff, d_Cin, d_Cout, d_sourcea, d_targeta, d_paritya, 
        d_sourceb, d_targetb, d_parityb, nbeta_strs_, 
        targeta_size, targetb_size, tensor_size);
   

    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch apply_individual_nbody1_accumulate_kernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        throw std::runtime_error("Kernel launch failed");
    }

    // Wait for the kernel to complete and check for errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution failed (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        throw std::runtime_error("Kernel execution failed");
    }
}

// ==============================================
// Fused apply+dot kernel and wrapper (Complex)
// Computes <sigma | K | psi> as a scalar reduction into d_accum.
// Neither d_psi nor d_sigma is ever modified.
// ==============================================

/// Per-thread contribution: conj(sigma[target]) * coeff * parity_a * parity_b * psi[source]
/// Shared-memory block reduction followed by one atomicAdd per block to d_accum.
__global__ void dot_individual_nbody1_kernel(
    cuDoubleComplex coeff,
    const cuDoubleComplex* d_psi,
    const cuDoubleComplex* d_sigma,
    const int* d_sourcea,
    const int* d_targeta,
    const cuDoubleComplex* d_paritya,
    const int* d_sourceb,
    const int* d_targetb,
    const cuDoubleComplex* d_parityb,
    long long nbeta_strs_,
    int targeta_size,
    int targetb_size,
    cuDoubleComplex* d_accum)
{
    // Shared memory: first half = real parts, second half = imaginary parts.
    // Allocated as 2 * blockDim.x * blockDim.y doubles by the host.
    extern __shared__ double sh[];
    int block_threads = blockDim.x * blockDim.y;
    double* sh_real = sh;
    double* sh_imag = sh + block_threads;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // alpha-mapping index
    int idy = blockIdx.y * blockDim.y + threadIdx.y;  // beta-mapping index
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    double acc_re = 0.0, acc_im = 0.0;

    if (idx < targeta_size && idy < targetb_size) {
        // term = coeff * parity_a[idx] * parity_b[idy] * psi[source_a*nb + source_b]
        cuDoubleComplex pref = cuCmul(coeff, d_paritya[idx]);
        cuDoubleComplex term = cuCmul(pref, d_parityb[idy]);
        term = cuCmul(term, d_psi[(long long)d_sourcea[idx] * nbeta_strs_ + d_sourceb[idy]]);

        // contribution = conj(sigma[target_a*nb + target_b]) * term
        long long target_idx = (long long)d_targeta[idx] * nbeta_strs_ + d_targetb[idy];
        cuDoubleComplex val = cuCmul(cuConj(d_sigma[target_idx]), term);
        acc_re = val.x;
        acc_im = val.y;
    }

    sh_real[tid] = acc_re;
    sh_imag[tid] = acc_im;
    __syncthreads();

    // Parallel reduction within the block.
    for (int s = block_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_real[tid] += sh_real[tid + s];
            sh_imag[tid] += sh_imag[tid + s];
        }
        __syncthreads();
    }

    // One atomic write per block into the global accumulator.
    if (tid == 0) {
        atomicAdd_double(&d_accum->x, sh_real[0]);
        atomicAdd_double(&d_accum->y, sh_imag[0]);
    }
}

extern "C" void dot_individual_nbody1_wrapper(
    cuDoubleComplex coeff,
    const cuDoubleComplex* d_psi,
    const cuDoubleComplex* d_sigma,
    const int* d_sourcea,
    const int* d_targeta,
    const cuDoubleComplex* d_paritya,
    const int* d_sourceb,
    const int* d_targetb,
    const cuDoubleComplex* d_parityb,
    long long nbeta_strs_,
    int targeta_size,
    int targetb_size,
    cuDoubleComplex* d_accum)
{
    dim3 blockSize(16, 16);  // 256 threads per block
    dim3 gridSize(
        (targeta_size + blockSize.x - 1) / blockSize.x,
        (targetb_size + blockSize.y - 1) / blockSize.y);

    // Two arrays of 256 doubles (real + imag) in shared memory.
    size_t sharedMemSize = 2 * blockSize.x * blockSize.y * sizeof(double);

    dot_individual_nbody1_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        coeff, d_psi, d_sigma,
        d_sourcea, d_targeta, d_paritya,
        d_sourceb, d_targetb, d_parityb,
        nbeta_strs_, targeta_size, targetb_size, d_accum);

    // Kernels across SQOp terms are queued in the default stream and execute
    // in-order; one cudaDeviceSynchronize in the caller (dot_sqop_gpu) suffices.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_individual_nbody1_kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_individual_nbody1_kernel sync failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

// ==============================================
// Dot kernel and wrapper (Real)
// ==============================================

/// Real-path per-thread contribution: sigma[target] * coeff * parity_a * parity_b * psi[source]
/// State vectors are stored as double (d_re_data_); parities are double.
__global__ void dot_individual_nbody1_real_kernel(
    double coeff,
    const double* d_psi,
    const double* d_sigma,
    const int* d_sourcea,
    const int* d_targeta,
    const double* d_paritya,
    const int* d_sourceb,
    const int* d_targetb,
    const double* d_parityb,
    long long nbeta_strs_,
    int targeta_size,
    int targetb_size,
    double* d_accum)
{
    extern __shared__ double sh_real[];

    int block_threads = blockDim.x * blockDim.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    double acc = 0.0;

    if (idx < targeta_size && idy < targetb_size) {
        double pref = coeff * d_paritya[idx] * d_parityb[idy];
        double val  = pref * d_psi[(long long)d_sourcea[idx] * nbeta_strs_ + d_sourceb[idy]];
        // Real bra: no conjugation needed
        acc = d_sigma[(long long)d_targeta[idx] * nbeta_strs_ + d_targetb[idy]] * val;
    }

    sh_real[tid] = acc;
    __syncthreads();

    for (int s = block_threads / 2; s > 0; s >>= 1) {
        if (tid < s) sh_real[tid] += sh_real[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_accum, sh_real[0]);
    }
}

extern "C" void dot_individual_nbody1_real_wrapper(
    double coeff,
    const double* d_psi,
    const double* d_sigma,
    const int* d_sourcea,
    const int* d_targeta,
    const double* d_paritya,
    const int* d_sourceb,
    const int* d_targetb,
    const double* d_parityb,
    long long nbeta_strs_,
    int targeta_size,
    int targetb_size,
    double* d_accum)
{
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (targeta_size + blockSize.x - 1) / blockSize.x,
        (targetb_size + blockSize.y - 1) / blockSize.y);

    size_t sharedMemSize = blockSize.x * blockSize.y * sizeof(double);

    dot_individual_nbody1_real_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        coeff, d_psi, d_sigma,
        d_sourcea, d_targeta, d_paritya,
        d_sourceb, d_targetb, d_parityb,
        nbeta_strs_, targeta_size, targetb_size, d_accum);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_individual_nbody1_real_kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_individual_nbody1_real_kernel sync failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

// ==============================================
// Fused dual-term dot kernel and wrapper (Complex)
// Processes both dag and undag terms in a single launch using blockIdx.z.
// ==============================================

__global__ void dot_individual_nbody1_dual_kernel(
    cuDoubleComplex coeff0,
    const int* d_sourcea0,
    const int* d_targeta0,
    const cuDoubleComplex* d_paritya0,
    const int* d_sourceb0,
    const int* d_targetb0,
    const cuDoubleComplex* d_parityb0,
    int targeta_size0,
    int targetb_size0,
    cuDoubleComplex coeff1,
    const int* d_sourcea1,
    const int* d_targeta1,
    const cuDoubleComplex* d_paritya1,
    const int* d_sourceb1,
    const int* d_targetb1,
    const cuDoubleComplex* d_parityb1,
    int targeta_size1,
    int targetb_size1,
    const cuDoubleComplex* d_psi,
    const cuDoubleComplex* d_sigma,
    long long nbeta_strs_,
    cuDoubleComplex* d_accum)
{
    extern __shared__ double sh[];
    int block_threads = blockDim.x * blockDim.y;
    double* sh_real = sh;
    double* sh_imag = sh + block_threads;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int term_id = blockIdx.z;  // 0 = dag term, 1 = undag term

    // Select parameters based on which term this z-slice handles
    cuDoubleComplex coeff       = (term_id == 0) ? coeff0 : coeff1;
    const int* d_sourcea        = (term_id == 0) ? d_sourcea0 : d_sourcea1;
    const int* d_targeta        = (term_id == 0) ? d_targeta0 : d_targeta1;
    const cuDoubleComplex* d_pa = (term_id == 0) ? d_paritya0 : d_paritya1;
    const int* d_sourceb        = (term_id == 0) ? d_sourceb0 : d_sourceb1;
    const int* d_targetb        = (term_id == 0) ? d_targetb0 : d_targetb1;
    const cuDoubleComplex* d_pb = (term_id == 0) ? d_parityb0 : d_parityb1;
    int ta_size                 = (term_id == 0) ? targeta_size0 : targeta_size1;
    int tb_size                 = (term_id == 0) ? targetb_size0 : targetb_size1;

    double acc_re = 0.0, acc_im = 0.0;

    if (idx < ta_size && idy < tb_size) {
        cuDoubleComplex pref = cuCmul(coeff, d_pa[idx]);
        cuDoubleComplex term = cuCmul(pref, d_pb[idy]);
        term = cuCmul(term, d_psi[(long long)d_sourcea[idx] * nbeta_strs_ + d_sourceb[idy]]);

        long long target_idx = (long long)d_targeta[idx] * nbeta_strs_ + d_targetb[idy];
        cuDoubleComplex val = cuCmul(cuConj(d_sigma[target_idx]), term);
        acc_re = val.x;
        acc_im = val.y;
    }

    sh_real[tid] = acc_re;
    sh_imag[tid] = acc_im;
    __syncthreads();

    for (int s = block_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_real[tid] += sh_real[tid + s];
            sh_imag[tid] += sh_imag[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd_double(&d_accum->x, sh_real[0]);
        atomicAdd_double(&d_accum->y, sh_imag[0]);
    }
}

extern "C" void dot_individual_nbody1_dual_wrapper(
    cuDoubleComplex coeff0,
    const cuDoubleComplex* d_psi,
    const cuDoubleComplex* d_sigma,
    const int* d_sourcea0,
    const int* d_targeta0,
    const cuDoubleComplex* d_paritya0,
    const int* d_sourceb0,
    const int* d_targetb0,
    const cuDoubleComplex* d_parityb0,
    int targeta_size0,
    int targetb_size0,
    cuDoubleComplex coeff1,
    const int* d_sourcea1,
    const int* d_targeta1,
    const cuDoubleComplex* d_paritya1,
    const int* d_sourceb1,
    const int* d_targetb1,
    const cuDoubleComplex* d_parityb1,
    int targeta_size1,
    int targetb_size1,
    long long nbeta_strs_,
    cuDoubleComplex* d_accum)
{
    dim3 blockSize(16, 16);
    int max_ta = (targeta_size0 > targeta_size1) ? targeta_size0 : targeta_size1;
    int max_tb = (targetb_size0 > targetb_size1) ? targetb_size0 : targetb_size1;
    dim3 gridSize(
        (max_ta + blockSize.x - 1) / blockSize.x,
        (max_tb + blockSize.y - 1) / blockSize.y,
        2);  // z=0 for dag, z=1 for undag

    size_t sharedMemSize = 2 * blockSize.x * blockSize.y * sizeof(double);

    dot_individual_nbody1_dual_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        coeff0,
        d_sourcea0, d_targeta0, d_paritya0,
        d_sourceb0, d_targetb0, d_parityb0,
        targeta_size0, targetb_size0,
        coeff1,
        d_sourcea1, d_targeta1, d_paritya1,
        d_sourceb1, d_targetb1, d_parityb1,
        targeta_size1, targetb_size1,
        d_psi, d_sigma,
        nbeta_strs_, d_accum);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_individual_nbody1_dual_kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_individual_nbody1_dual_kernel sync failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

// ==============================================
// Fused dual-term dot kernel and wrapper (Real)
// Processes both dag and undag terms in a single launch using blockIdx.z.
// ==============================================

__global__ void dot_individual_nbody1_real_dual_kernel(
    double coeff0,
    const int* d_sourcea0,
    const int* d_targeta0,
    const double* d_paritya0,
    const int* d_sourceb0,
    const int* d_targetb0,
    const double* d_parityb0,
    int targeta_size0,
    int targetb_size0,
    double coeff1,
    const int* d_sourcea1,
    const int* d_targeta1,
    const double* d_paritya1,
    const int* d_sourceb1,
    const int* d_targetb1,
    const double* d_parityb1,
    int targeta_size1,
    int targetb_size1,
    const double* d_psi,
    const double* d_sigma,
    long long nbeta_strs_,
    double* d_accum)
{
    extern __shared__ double sh_real[];

    int block_threads = blockDim.x * blockDim.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int term_id = blockIdx.z;  // 0 = dag term, 1 = undag term

    // Select parameters based on which term this z-slice handles
    double coeff         = (term_id == 0) ? coeff0 : coeff1;
    const int* d_sourcea = (term_id == 0) ? d_sourcea0 : d_sourcea1;
    const int* d_targeta = (term_id == 0) ? d_targeta0 : d_targeta1;
    const double* d_pa   = (term_id == 0) ? d_paritya0 : d_paritya1;
    const int* d_sourceb = (term_id == 0) ? d_sourceb0 : d_sourceb1;
    const int* d_targetb = (term_id == 0) ? d_targetb0 : d_targetb1;
    const double* d_pb   = (term_id == 0) ? d_parityb0 : d_parityb1;
    int ta_size          = (term_id == 0) ? targeta_size0 : targeta_size1;
    int tb_size          = (term_id == 0) ? targetb_size0 : targetb_size1;

    double acc = 0.0;

    if (idx < ta_size && idy < tb_size) {
        double pref = coeff * d_pa[idx] * d_pb[idy];
        double val  = pref * d_psi[(long long)d_sourcea[idx] * nbeta_strs_ + d_sourceb[idy]];
        acc = d_sigma[(long long)d_targeta[idx] * nbeta_strs_ + d_targetb[idy]] * val;
    }

    sh_real[tid] = acc;
    __syncthreads();

    for (int s = block_threads / 2; s > 0; s >>= 1) {
        if (tid < s) sh_real[tid] += sh_real[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_accum, sh_real[0]);
    }
}

extern "C" void dot_individual_nbody1_real_dual_wrapper(
    double coeff0,
    const double* d_psi,
    const double* d_sigma,
    const int* d_sourcea0,
    const int* d_targeta0,
    const double* d_paritya0,
    const int* d_sourceb0,
    const int* d_targetb0,
    const double* d_parityb0,
    int targeta_size0,
    int targetb_size0,
    double coeff1,
    const int* d_sourcea1,
    const int* d_targeta1,
    const double* d_paritya1,
    const int* d_sourceb1,
    const int* d_targetb1,
    const double* d_parityb1,
    int targeta_size1,
    int targetb_size1,
    long long nbeta_strs_,
    double* d_accum)
{
    dim3 blockSize(16, 16);
    int max_ta = (targeta_size0 > targeta_size1) ? targeta_size0 : targeta_size1;
    int max_tb = (targetb_size0 > targetb_size1) ? targetb_size0 : targetb_size1;
    dim3 gridSize(
        (max_ta + blockSize.x - 1) / blockSize.x,
        (max_tb + blockSize.y - 1) / blockSize.y,
        2);  // z=0 for dag, z=1 for undag

    size_t sharedMemSize = blockSize.x * blockSize.y * sizeof(double);

    dot_individual_nbody1_real_dual_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        coeff0,
        d_sourcea0, d_targeta0, d_paritya0,
        d_sourceb0, d_targetb0, d_parityb0,
        targeta_size0, targetb_size0,
        coeff1,
        d_sourcea1, d_targeta1, d_paritya1,
        d_sourceb1, d_targetb1, d_parityb1,
        targeta_size1, targetb_size1,
        d_psi, d_sigma,
        nbeta_strs_, d_accum);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_individual_nbody1_real_dual_kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_individual_nbody1_real_dual_kernel sync failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

// ==============================================
// Givens-style tiled dot kernels (Real)
// Beta-fast warp layout for coalesced row-major access.
// Paired dual: each thread computes both dag and undag contributions.
// ==============================================

// ---- Alpha-only paired dual dot kernel (Real) ----
// For operators that only excite alpha spin (beta is identity-like).
// Each alpha map entry connects source_row -> target_row; thread walks beta columns.
// block: (BX=128, 1), grid: (ceil(nbeta/BX), na_pairs)
template <int BX>
__global__ void dot_alpha_only_dual_real_kernel(
    const double* __restrict__ d_psi,
    const double* __restrict__ d_sigma,
    const int* __restrict__ d_sourcea,    // source for dag direction (= scale_inds_undag)
    const int* __restrict__ d_targeta,    // target for dag direction (= scale_inds_dag)
    const double* __restrict__ d_paritya_uv,  // parity for dag direction
    const double* __restrict__ d_paritya_vu,  // parity for undag direction
    int na_pairs,
    long long nbeta_strs_,
    double coeff_uv,    // dot_coeff_dag
    double coeff_vu,    // dot_coeff_undag
    double* __restrict__ d_accum)
{
    extern __shared__ double sh[];

    int b  = blockIdx.x * BX + threadIdx.x;
    int ia = blockIdx.y;
    int tid = threadIdx.x;

    double acc = 0.0;

    if (ia < na_pairs && b < nbeta_strs_) {
        int sa = d_sourcea[ia];
        int ta = d_targeta[ia];

        long long u_idx = (long long)sa * nbeta_strs_ + b;
        long long v_idx = (long long)ta * nbeta_strs_ + b;

        double psi_u  = d_psi[u_idx];
        double psi_v  = d_psi[v_idx];
        double sig_u  = d_sigma[u_idx];
        double sig_v  = d_sigma[v_idx];

        double puv = coeff_uv * d_paritya_uv[ia];
        double pvu = coeff_vu * d_paritya_vu[ia];

        // dag:   sigma[target] * psi[source] = sig_v * psi_u
        // undag: sigma[source] * psi[target] = sig_u * psi_v
        acc = puv * sig_v * psi_u + pvu * sig_u * psi_v;
    }

    sh[tid] = acc;
    __syncthreads();

    for (int s = BX / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_accum, sh[0]);
    }
}

extern "C" void dot_alpha_only_dual_real_wrapper(
    const double* d_psi,
    const double* d_sigma,
    const int* d_sourcea,
    const int* d_targeta,
    const double* d_paritya_uv,
    const double* d_paritya_vu,
    int na_pairs,
    long long nbeta_strs_,
    double coeff_uv,
    double coeff_vu,
    double* d_accum)
{
    constexpr int BX = 128;
    dim3 blockSize(BX);
    dim3 gridSize(
        (static_cast<int>(nbeta_strs_) + BX - 1) / BX,
        na_pairs);

    size_t sharedMemSize = BX * sizeof(double);

    dot_alpha_only_dual_real_kernel<BX><<<gridSize, blockSize, sharedMemSize>>>(
        d_psi, d_sigma,
        d_sourcea, d_targeta, d_paritya_uv, d_paritya_vu,
        na_pairs, nbeta_strs_, coeff_uv, coeff_vu, d_accum);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_alpha_only_dual_real_kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_alpha_only_dual_real_kernel sync failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

// ---- Beta-only paired dual dot kernel (Real) ----
// For operators that only excite beta spin (alpha is identity-like).
// All alpha rows contribute; each beta map entry connects source_col -> target_col.
// block: (BX=32, AY=8), grid: (ceil(nb_pairs/BX), ceil(nalpha/AY))
template <int BX, int AY>
__global__ void dot_beta_only_dual_real_kernel(
    const double* __restrict__ d_psi,
    const double* __restrict__ d_sigma,
    const int* __restrict__ d_sourceb,    // source for dag direction
    const int* __restrict__ d_targetb,    // target for dag direction
    const double* __restrict__ d_parityb_uv,  // parity for dag direction
    const double* __restrict__ d_parityb_vu,  // parity for undag direction
    int nb_pairs,
    long long nalpha_strs_,
    long long nbeta_strs_,
    double coeff_uv,
    double coeff_vu,
    double* __restrict__ d_accum)
{
    __shared__ int    s_sb[BX], s_tb[BX];
    __shared__ double s_cuv[BX], s_cvu[BX];
    // Reduction shared memory
    __shared__ double s_red[BX * AY];

    int tx = threadIdx.x;  // beta pair lane
    int ty = threadIdx.y;  // alpha row lane
    int tid = ty * BX + tx;

    int ib = blockIdx.x * BX + tx;
    long long a = (long long)blockIdx.y * AY + ty;

    // Load beta metadata into shared memory (one row of threads does it)
    if (ty == 0) {
        if (ib < nb_pairs) {
            s_sb[tx]  = d_sourceb[ib];
            s_tb[tx]  = d_targetb[ib];
            s_cuv[tx] = coeff_uv * d_parityb_uv[ib];
            s_cvu[tx] = coeff_vu * d_parityb_vu[ib];
        } else {
            s_sb[tx] = 0;
            s_tb[tx] = 0;
            s_cuv[tx] = 0.0;
            s_cvu[tx] = 0.0;
        }
    }
    __syncthreads();

    double acc = 0.0;

    if (a < nalpha_strs_ && ib < nb_pairs) {
        long long base = a * nbeta_strs_;

        long long u_idx = base + s_sb[tx];
        long long v_idx = base + s_tb[tx];

        double psi_u = d_psi[u_idx];
        double psi_v = d_psi[v_idx];
        double sig_u = d_sigma[u_idx];
        double sig_v = d_sigma[v_idx];

        acc = s_cuv[tx] * sig_v * psi_u + s_cvu[tx] * sig_u * psi_v;
    }

    // Block reduction
    s_red[tid] = acc;
    __syncthreads();

    int block_threads = BX * AY;
    for (int s = block_threads / 2; s > 0; s >>= 1) {
        if (tid < s) s_red[tid] += s_red[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_accum, s_red[0]);
    }
}

extern "C" void dot_beta_only_dual_real_wrapper(
    const double* d_psi,
    const double* d_sigma,
    const int* d_sourceb,
    const int* d_targetb,
    const double* d_parityb_uv,
    const double* d_parityb_vu,
    int nb_pairs,
    long long nalpha_strs_,
    long long nbeta_strs_,
    double coeff_uv,
    double coeff_vu,
    double* d_accum)
{
    constexpr int BX = 32;
    constexpr int AY = 8;
    dim3 blockSize(BX, AY);  // 256 threads/block
    dim3 gridSize(
        (nb_pairs + BX - 1) / BX,
        (static_cast<int>(nalpha_strs_) + AY - 1) / AY);

    dot_beta_only_dual_real_kernel<BX, AY><<<gridSize, blockSize>>>(
        d_psi, d_sigma,
        d_sourceb, d_targetb, d_parityb_uv, d_parityb_vu,
        nb_pairs, nalpha_strs_, nbeta_strs_,
        coeff_uv, coeff_vu, d_accum);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_beta_only_dual_real_kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_beta_only_dual_real_kernel sync failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

// ---- Mixed alpha×beta paired dual tiled dot kernel (Real) ----
// For operators exciting both alpha and beta spin.
// block: (BX=32, AY=8), grid: (ceil(nb_pairs/BX), ceil(na_pairs/AY))
// threadIdx.x walks beta pairs (fast/coalesced), threadIdx.y walks alpha pairs.
// Metadata is loaded into shared memory for reuse across the opposing dimension.
template <int BX, int AY>
__global__ void dot_mixed_dual_real_kernel(
    const double* __restrict__ d_psi,
    const double* __restrict__ d_sigma,
    const int* __restrict__ d_sourcea,
    const int* __restrict__ d_targeta,
    const double* __restrict__ d_paritya_uv,
    const double* __restrict__ d_paritya_vu,
    const int* __restrict__ d_sourceb,
    const int* __restrict__ d_targetb,
    const double* __restrict__ d_parityb_uv,
    const double* __restrict__ d_parityb_vu,
    int na_pairs,
    int nb_pairs,
    long long nbeta_strs_,
    double coeff_uv,
    double coeff_vu,
    double* __restrict__ d_accum)
{
    // Shared memory for metadata
    __shared__ int    s_sa[AY], s_ta[AY];
    __shared__ double s_pauv[AY], s_pavu[AY];

    __shared__ int    s_sb[BX], s_tb[BX];
    __shared__ double s_pbuv[BX], s_pbvu[BX];

    // Reduction shared memory
    __shared__ double s_red[BX * AY];

    int tx = threadIdx.x;  // beta-pair lane
    int ty = threadIdx.y;  // alpha-pair lane
    int tid = ty * BX + tx;

    int ia = blockIdx.y * AY + ty;
    int ib = blockIdx.x * BX + tx;

    // Load alpha metadata (one column of threads)
    if (tx == 0) {
        if (ia < na_pairs) {
            s_sa[ty]   = d_sourcea[ia];
            s_ta[ty]   = d_targeta[ia];
            s_pauv[ty] = d_paritya_uv[ia];
            s_pavu[ty] = d_paritya_vu[ia];
        } else {
            s_sa[ty] = 0; s_ta[ty] = 0;
            s_pauv[ty] = 0.0; s_pavu[ty] = 0.0;
        }
    }

    // Load beta metadata (one row of threads)
    if (ty == 0) {
        if (ib < nb_pairs) {
            s_sb[tx]   = d_sourceb[ib];
            s_tb[tx]   = d_targetb[ib];
            s_pbuv[tx] = d_parityb_uv[ib];
            s_pbvu[tx] = d_parityb_vu[ib];
        } else {
            s_sb[tx] = 0; s_tb[tx] = 0;
            s_pbuv[tx] = 0.0; s_pbvu[tx] = 0.0;
        }
    }
    __syncthreads();

    double acc = 0.0;

    if (ia < na_pairs && ib < nb_pairs) {
        long long u_idx = (long long)s_sa[ty] * nbeta_strs_ + s_sb[tx];
        long long v_idx = (long long)s_ta[ty] * nbeta_strs_ + s_tb[tx];

        double psi_u = d_psi[u_idx];
        double psi_v = d_psi[v_idx];
        double sig_u = d_sigma[u_idx];
        double sig_v = d_sigma[v_idx];

        double cuv = coeff_uv * s_pauv[ty] * s_pbuv[tx];
        double cvu = coeff_vu * s_pavu[ty] * s_pbvu[tx];

        acc = cuv * sig_v * psi_u + cvu * sig_u * psi_v;
    }

    // Block reduction
    s_red[tid] = acc;
    __syncthreads();

    int block_threads = BX * AY;
    for (int s = block_threads / 2; s > 0; s >>= 1) {
        if (tid < s) s_red[tid] += s_red[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_accum, s_red[0]);
    }
}

extern "C" void dot_mixed_dual_real_wrapper(
    const double* d_psi,
    const double* d_sigma,
    const int* d_sourcea,
    const int* d_targeta,
    const double* d_paritya_uv,
    const double* d_paritya_vu,
    const int* d_sourceb,
    const int* d_targetb,
    const double* d_parityb_uv,
    const double* d_parityb_vu,
    int na_pairs,
    int nb_pairs,
    long long nbeta_strs_,
    double coeff_uv,
    double coeff_vu,
    double* d_accum)
{
    constexpr int BX = 32;
    constexpr int AY = 8;
    dim3 blockSize(BX, AY);  // 256 threads/block
    dim3 gridSize(
        (nb_pairs + BX - 1) / BX,
        (na_pairs + AY - 1) / AY);

    dot_mixed_dual_real_kernel<BX, AY><<<gridSize, blockSize>>>(
        d_psi, d_sigma,
        d_sourcea, d_targeta, d_paritya_uv, d_paritya_vu,
        d_sourceb, d_targetb, d_parityb_uv, d_parityb_vu,
        na_pairs, nb_pairs, nbeta_strs_,
        coeff_uv, coeff_vu, d_accum);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_mixed_dual_real_kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_mixed_dual_real_kernel sync failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

// ---- Alpha-only paired dual dot kernel (Complex) ----
template <int BX>
__global__ void dot_alpha_only_dual_kernel(
    const cuDoubleComplex* __restrict__ d_psi,
    const cuDoubleComplex* __restrict__ d_sigma,
    const int* __restrict__ d_sourcea,
    const int* __restrict__ d_targeta,
    const cuDoubleComplex* __restrict__ d_paritya_uv,
    const cuDoubleComplex* __restrict__ d_paritya_vu,
    int na_pairs,
    long long nbeta_strs_,
    cuDoubleComplex coeff_uv,
    cuDoubleComplex coeff_vu,
    cuDoubleComplex* __restrict__ d_accum)
{
    extern __shared__ double sh[];
    double* sh_real = sh;
    double* sh_imag = sh + BX;

    int b  = blockIdx.x * BX + threadIdx.x;
    int ia = blockIdx.y;
    int tid = threadIdx.x;

    double acc_re = 0.0, acc_im = 0.0;

    if (ia < na_pairs && b < nbeta_strs_) {
        int sa = d_sourcea[ia];
        int ta = d_targeta[ia];

        long long u_idx = (long long)sa * nbeta_strs_ + b;
        long long v_idx = (long long)ta * nbeta_strs_ + b;

        cuDoubleComplex psi_u = d_psi[u_idx];
        cuDoubleComplex psi_v = d_psi[v_idx];
        cuDoubleComplex sig_u = d_sigma[u_idx];
        cuDoubleComplex sig_v = d_sigma[v_idx];

        // dag: conj(sigma[target]) * coeff * parity * psi[source]
        cuDoubleComplex puv = cuCmul(coeff_uv, d_paritya_uv[ia]);
        cuDoubleComplex val_dag = cuCmul(cuCmul(puv, cuConj(sig_v)), psi_u);

        // undag: conj(sigma[source]) * coeff * parity * psi[target]
        cuDoubleComplex pvu = cuCmul(coeff_vu, d_paritya_vu[ia]);
        cuDoubleComplex val_undag = cuCmul(cuCmul(pvu, cuConj(sig_u)), psi_v);

        acc_re = val_dag.x + val_undag.x;
        acc_im = val_dag.y + val_undag.y;
    }

    sh_real[tid] = acc_re;
    sh_imag[tid] = acc_im;
    __syncthreads();

    for (int s = BX / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_real[tid] += sh_real[tid + s];
            sh_imag[tid] += sh_imag[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd_double(&d_accum->x, sh_real[0]);
        atomicAdd_double(&d_accum->y, sh_imag[0]);
    }
}

extern "C" void dot_alpha_only_dual_wrapper(
    const cuDoubleComplex* d_psi,
    const cuDoubleComplex* d_sigma,
    const int* d_sourcea,
    const int* d_targeta,
    const cuDoubleComplex* d_paritya_uv,
    const cuDoubleComplex* d_paritya_vu,
    int na_pairs,
    long long nbeta_strs_,
    cuDoubleComplex coeff_uv,
    cuDoubleComplex coeff_vu,
    cuDoubleComplex* d_accum)
{
    constexpr int BX = 128;
    dim3 blockSize(BX);
    dim3 gridSize(
        (static_cast<int>(nbeta_strs_) + BX - 1) / BX,
        na_pairs);

    size_t sharedMemSize = 2 * BX * sizeof(double);

    dot_alpha_only_dual_kernel<BX><<<gridSize, blockSize, sharedMemSize>>>(
        d_psi, d_sigma,
        d_sourcea, d_targeta, d_paritya_uv, d_paritya_vu,
        na_pairs, nbeta_strs_, coeff_uv, coeff_vu, d_accum);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_alpha_only_dual_kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_alpha_only_dual_kernel sync failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

// ---- Beta-only paired dual dot kernel (Complex) ----
template <int BX, int AY>
__global__ void dot_beta_only_dual_kernel(
    const cuDoubleComplex* __restrict__ d_psi,
    const cuDoubleComplex* __restrict__ d_sigma,
    const int* __restrict__ d_sourceb,
    const int* __restrict__ d_targetb,
    const cuDoubleComplex* __restrict__ d_parityb_uv,
    const cuDoubleComplex* __restrict__ d_parityb_vu,
    int nb_pairs,
    long long nalpha_strs_,
    long long nbeta_strs_,
    cuDoubleComplex coeff_uv,
    cuDoubleComplex coeff_vu,
    cuDoubleComplex* __restrict__ d_accum)
{
    __shared__ int    s_sb[BX], s_tb[BX];
    __shared__ double s_cuv_re[BX], s_cuv_im[BX];
    __shared__ double s_cvu_re[BX], s_cvu_im[BX];
    __shared__ double s_red_re[BX * AY];
    __shared__ double s_red_im[BX * AY];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * BX + tx;

    int ib = blockIdx.x * BX + tx;
    long long a = (long long)blockIdx.y * AY + ty;

    if (ty == 0) {
        if (ib < nb_pairs) {
            s_sb[tx] = d_sourceb[ib];
            s_tb[tx] = d_targetb[ib];
            cuDoubleComplex cuv = cuCmul(coeff_uv, d_parityb_uv[ib]);
            cuDoubleComplex cvu = cuCmul(coeff_vu, d_parityb_vu[ib]);
            s_cuv_re[tx] = cuv.x; s_cuv_im[tx] = cuv.y;
            s_cvu_re[tx] = cvu.x; s_cvu_im[tx] = cvu.y;
        } else {
            s_sb[tx] = 0; s_tb[tx] = 0;
            s_cuv_re[tx] = 0.0; s_cuv_im[tx] = 0.0;
            s_cvu_re[tx] = 0.0; s_cvu_im[tx] = 0.0;
        }
    }
    __syncthreads();

    double acc_re = 0.0, acc_im = 0.0;

    if (a < nalpha_strs_ && ib < nb_pairs) {
        long long base = a * nbeta_strs_;
        long long u_idx = base + s_sb[tx];
        long long v_idx = base + s_tb[tx];

        cuDoubleComplex psi_u = d_psi[u_idx];
        cuDoubleComplex psi_v = d_psi[v_idx];
        cuDoubleComplex sig_u = d_sigma[u_idx];
        cuDoubleComplex sig_v = d_sigma[v_idx];

        cuDoubleComplex cuv_coeff = make_cuDoubleComplex(s_cuv_re[tx], s_cuv_im[tx]);
        cuDoubleComplex cvu_coeff = make_cuDoubleComplex(s_cvu_re[tx], s_cvu_im[tx]);

        cuDoubleComplex val_dag   = cuCmul(cuCmul(cuv_coeff, cuConj(sig_v)), psi_u);
        cuDoubleComplex val_undag = cuCmul(cuCmul(cvu_coeff, cuConj(sig_u)), psi_v);

        acc_re = val_dag.x + val_undag.x;
        acc_im = val_dag.y + val_undag.y;
    }

    s_red_re[tid] = acc_re;
    s_red_im[tid] = acc_im;
    __syncthreads();

    int block_threads = BX * AY;
    for (int s = block_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_red_re[tid] += s_red_re[tid + s];
            s_red_im[tid] += s_red_im[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd_double(&d_accum->x, s_red_re[0]);
        atomicAdd_double(&d_accum->y, s_red_im[0]);
    }
}

extern "C" void dot_beta_only_dual_wrapper(
    const cuDoubleComplex* d_psi,
    const cuDoubleComplex* d_sigma,
    const int* d_sourceb,
    const int* d_targetb,
    const cuDoubleComplex* d_parityb_uv,
    const cuDoubleComplex* d_parityb_vu,
    int nb_pairs,
    long long nalpha_strs_,
    long long nbeta_strs_,
    cuDoubleComplex coeff_uv,
    cuDoubleComplex coeff_vu,
    cuDoubleComplex* d_accum)
{
    constexpr int BX = 32;
    constexpr int AY = 8;
    dim3 blockSize(BX, AY);
    dim3 gridSize(
        (nb_pairs + BX - 1) / BX,
        (static_cast<int>(nalpha_strs_) + AY - 1) / AY);

    dot_beta_only_dual_kernel<BX, AY><<<gridSize, blockSize>>>(
        d_psi, d_sigma,
        d_sourceb, d_targetb, d_parityb_uv, d_parityb_vu,
        nb_pairs, nalpha_strs_, nbeta_strs_,
        coeff_uv, coeff_vu, d_accum);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_beta_only_dual_kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_beta_only_dual_kernel sync failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

// ---- Mixed alpha×beta paired dual tiled dot kernel (Complex) ----
template <int BX, int AY>
__global__ void dot_mixed_dual_kernel(
    const cuDoubleComplex* __restrict__ d_psi,
    const cuDoubleComplex* __restrict__ d_sigma,
    const int* __restrict__ d_sourcea,
    const int* __restrict__ d_targeta,
    const cuDoubleComplex* __restrict__ d_paritya_uv,
    const cuDoubleComplex* __restrict__ d_paritya_vu,
    const int* __restrict__ d_sourceb,
    const int* __restrict__ d_targetb,
    const cuDoubleComplex* __restrict__ d_parityb_uv,
    const cuDoubleComplex* __restrict__ d_parityb_vu,
    int na_pairs,
    int nb_pairs,
    long long nbeta_strs_,
    cuDoubleComplex coeff_uv,
    cuDoubleComplex coeff_vu,
    cuDoubleComplex* __restrict__ d_accum)
{
    __shared__ int    s_sa[AY], s_ta[AY];
    __shared__ double s_pauv_re[AY], s_pauv_im[AY];
    __shared__ double s_pavu_re[AY], s_pavu_im[AY];

    __shared__ int    s_sb[BX], s_tb[BX];
    __shared__ double s_pbuv_re[BX], s_pbuv_im[BX];
    __shared__ double s_pbvu_re[BX], s_pbvu_im[BX];

    __shared__ double s_red_re[BX * AY];
    __shared__ double s_red_im[BX * AY];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * BX + tx;

    int ia = blockIdx.y * AY + ty;
    int ib = blockIdx.x * BX + tx;

    if (tx == 0) {
        if (ia < na_pairs) {
            s_sa[ty] = d_sourcea[ia];
            s_ta[ty] = d_targeta[ia];
            cuDoubleComplex pa_uv = d_paritya_uv[ia];
            cuDoubleComplex pa_vu = d_paritya_vu[ia];
            s_pauv_re[ty] = pa_uv.x; s_pauv_im[ty] = pa_uv.y;
            s_pavu_re[ty] = pa_vu.x; s_pavu_im[ty] = pa_vu.y;
        } else {
            s_sa[ty] = 0; s_ta[ty] = 0;
            s_pauv_re[ty] = 0.0; s_pauv_im[ty] = 0.0;
            s_pavu_re[ty] = 0.0; s_pavu_im[ty] = 0.0;
        }
    }

    if (ty == 0) {
        if (ib < nb_pairs) {
            s_sb[tx] = d_sourceb[ib];
            s_tb[tx] = d_targetb[ib];
            cuDoubleComplex pb_uv = d_parityb_uv[ib];
            cuDoubleComplex pb_vu = d_parityb_vu[ib];
            s_pbuv_re[tx] = pb_uv.x; s_pbuv_im[tx] = pb_uv.y;
            s_pbvu_re[tx] = pb_vu.x; s_pbvu_im[tx] = pb_vu.y;
        } else {
            s_sb[tx] = 0; s_tb[tx] = 0;
            s_pbuv_re[tx] = 0.0; s_pbuv_im[tx] = 0.0;
            s_pbvu_re[tx] = 0.0; s_pbvu_im[tx] = 0.0;
        }
    }
    __syncthreads();

    double acc_re = 0.0, acc_im = 0.0;

    if (ia < na_pairs && ib < nb_pairs) {
        long long u_idx = (long long)s_sa[ty] * nbeta_strs_ + s_sb[tx];
        long long v_idx = (long long)s_ta[ty] * nbeta_strs_ + s_tb[tx];

        cuDoubleComplex psi_u = d_psi[u_idx];
        cuDoubleComplex psi_v = d_psi[v_idx];
        cuDoubleComplex sig_u = d_sigma[u_idx];
        cuDoubleComplex sig_v = d_sigma[v_idx];

        // cuv = coeff_uv * parity_a_uv * parity_b_uv
        cuDoubleComplex pa_uv = make_cuDoubleComplex(s_pauv_re[ty], s_pauv_im[ty]);
        cuDoubleComplex pb_uv = make_cuDoubleComplex(s_pbuv_re[tx], s_pbuv_im[tx]);
        cuDoubleComplex cuv = cuCmul(coeff_uv, cuCmul(pa_uv, pb_uv));

        // cvu = coeff_vu * parity_a_vu * parity_b_vu
        cuDoubleComplex pa_vu = make_cuDoubleComplex(s_pavu_re[ty], s_pavu_im[ty]);
        cuDoubleComplex pb_vu = make_cuDoubleComplex(s_pbvu_re[tx], s_pbvu_im[tx]);
        cuDoubleComplex cvu = cuCmul(coeff_vu, cuCmul(pa_vu, pb_vu));

        cuDoubleComplex val_dag   = cuCmul(cuv, cuCmul(cuConj(sig_v), psi_u));
        cuDoubleComplex val_undag = cuCmul(cvu, cuCmul(cuConj(sig_u), psi_v));

        acc_re = val_dag.x + val_undag.x;
        acc_im = val_dag.y + val_undag.y;
    }

    s_red_re[tid] = acc_re;
    s_red_im[tid] = acc_im;
    __syncthreads();

    int block_threads = BX * AY;
    for (int s = block_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_red_re[tid] += s_red_re[tid + s];
            s_red_im[tid] += s_red_im[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd_double(&d_accum->x, s_red_re[0]);
        atomicAdd_double(&d_accum->y, s_red_im[0]);
    }
}

extern "C" void dot_mixed_dual_wrapper(
    const cuDoubleComplex* d_psi,
    const cuDoubleComplex* d_sigma,
    const int* d_sourcea,
    const int* d_targeta,
    const cuDoubleComplex* d_paritya_uv,
    const cuDoubleComplex* d_paritya_vu,
    const int* d_sourceb,
    const int* d_targetb,
    const cuDoubleComplex* d_parityb_uv,
    const cuDoubleComplex* d_parityb_vu,
    int na_pairs,
    int nb_pairs,
    long long nbeta_strs_,
    cuDoubleComplex coeff_uv,
    cuDoubleComplex coeff_vu,
    cuDoubleComplex* d_accum)
{
    constexpr int BX = 32;
    constexpr int AY = 8;
    dim3 blockSize(BX, AY);
    dim3 gridSize(
        (nb_pairs + BX - 1) / BX,
        (na_pairs + AY - 1) / AY);

    dot_mixed_dual_kernel<BX, AY><<<gridSize, blockSize>>>(
        d_psi, d_sigma,
        d_sourcea, d_targeta, d_paritya_uv, d_paritya_vu,
        d_sourceb, d_targetb, d_parityb_uv, d_parityb_vu,
        na_pairs, nb_pairs, nbeta_strs_,
        coeff_uv, coeff_vu, d_accum);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_mixed_dual_kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("dot_mixed_dual_kernel sync failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

// ==============================================
// Scale elements kernel and wrapper (Complex)
// ==============================================

__global__ void scale_elements_kernel(
    cuDoubleComplex* d_Cout,
    const int* d_first, 
    int first_size,
    const int* d_second, 
    int second_size,
    long long nbeta_strs_,
    cuDoubleComplex factor) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < first_size && j < second_size) {
        long long idx = (long long)d_first[i] * nbeta_strs_ + d_second[j];
        d_Cout[idx] = cuCmul(d_Cout[idx], factor);
    }
}

extern "C" void scale_elements_wrapper_complex(
    cuDoubleComplex* d_Cout,
    const int* d_first, 
    int first_size,
    const int* d_second, 
    int second_size,
    long long nbeta_strs_,
    cuDoubleComplex factor) 
{
    if (first_size <= 0 || second_size <= 0 || nbeta_strs_ <= 0) return;
    // Fast path for identity scaling
    if (cuCreal(factor) == 1.0 && cuCimag(factor) == 0.0) return;

    dim3 blockSize(16, 16);
    dim3 gridSize((first_size + blockSize.x - 1) / blockSize.x, 
                  (second_size + blockSize.y - 1) / blockSize.y);

    scale_elements_kernel<<<gridSize, blockSize>>>(d_Cout, d_first, first_size, d_second, second_size, nbeta_strs_, factor);

    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch scale_elements_kernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        throw std::runtime_error("Kernel launch failed");
    }

    // Wait for the kernel to complete and check for errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution failed (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        throw std::runtime_error("Kernel execution failed");
    }
}

// ==============================================
// Scale elements kernel and wrapper (Real)
// ==============================================

__global__ void scale_elements_kernel_real(
    double* __restrict__ d_Cout,
    const int* __restrict__ d_first,
    int first_size,
    const int* __restrict__ d_second,
    int second_size,
    long long nbeta_strs_,
    double factor)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < first_size && j < second_size) {
        const long long idx = (long long)d_first[i] * nbeta_strs_ + d_second[j];
        d_Cout[idx] *= factor;
    }
}

extern "C" void scale_elements_wrapper_real(
    double* d_Cout,
    const int* d_first,
    int first_size,
    const int* d_second,
    int second_size,
    long long nbeta_strs_,
    double factor)
{
    if (first_size <= 0 || second_size <= 0 || nbeta_strs_ <= 0) return;
    if (factor == 1.0) return; // noop fast-path

    dim3 blockSize(16, 16);
    dim3 gridSize((first_size + blockSize.x - 1) / blockSize.x,
                  (second_size + blockSize.y - 1) / blockSize.y);

    scale_elements_kernel_real<<<gridSize, blockSize>>>(
        d_Cout, d_first, first_size, d_second, second_size, nbeta_strs_, factor);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch scale_elements_kernel_real ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("scale_elements_kernel_real launch failed");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "scale_elements_kernel_real execution failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("scale_elements_kernel_real execution failed");
    }
}

// ==============================================
// In-place Givens update kernels and wrappers (Complex)
// ==============================================

// Rows-only, coalesced across columns.
// One block processes one (sa1, ta1) pair; threads iterate j across nbeta_strs_.
__global__ void inplace_givens_update_rows_kernel(
    cuDoubleComplex* __restrict__ d_Cout,
    const int* __restrict__ sourcea1,      // [na]
    const int* __restrict__ targeta1,      // [na]
    const cuDoubleComplex* __restrict__ paritya1, // [na]  (g† leg, row)
    const cuDoubleComplex* __restrict__ paritya2, // [na]  (g  leg, row)
    int na,
    long long nbeta_strs_,                        // number of columns
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2)
{
    int ia = blockIdx.x;                          // one block per (sa1, ta1) pair
    if (ia >= na) return;

    // Broadcast row-scoped values once per block
    __shared__ int s_sa1, s_ta1;
    __shared__ cuDoubleComplex s_pa1, s_pa2;
    if (threadIdx.x == 0) {
        s_sa1 = sourcea1[ia];
        s_ta1 = targeta1[ia];
        s_pa1 = paritya1[ia];
        s_pa2 = paritya2[ia];
    }
    __syncthreads();

    const int sa1 = s_sa1, ta1 = s_ta1;
    const cuDoubleComplex pa1 = s_pa1, pa2 = s_pa2;
    const long long base_u = (long long)sa1 * nbeta_strs_;
    const long long base_v = (long long)ta1 * nbeta_strs_;

    for (long long col = threadIdx.x; col < nbeta_strs_; col += blockDim.x) {
        const long long idx_u = base_u + col;   // (sa1, col)
        const long long idx_v = base_v + col;   // (ta1, col)

        const cuDoubleComplex u0 = d_Cout[idx_u];
        const cuDoubleComplex v0 = d_Cout[idx_v];

        const cuDoubleComplex u_new = cuCadd(cuCmul(factor, u0), cuCmul(acc_coeff2, cuCmul(pa2, v0)));
        const cuDoubleComplex v_new = cuCadd(cuCmul(factor, v0), cuCmul(acc_coeff1, cuCmul(pa1, u0)));

        d_Cout[idx_u] = u_new;
        d_Cout[idx_v] = v_new;
    }
}


extern "C" void inplace_givens_update_complex_rows_wrapper(
    cuDoubleComplex* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const cuDoubleComplex* paritya1,
    const cuDoubleComplex* paritya2,
    int na,
    long long nbeta_strs_,
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2)
{
    if (na == 0 || nbeta_strs_ == 0) return;

    // Choose threads per block: cover columns with good occupancy.
    // Clamp to device limits if you prefer; 256 is a good default.
    int threads = static_cast<int>(std::min<long long>(256, nbeta_strs_));
    // Keep at least one warp
    if (threads < 32) threads = 32;

    dim3 block(threads);
    dim3 grid(na);  // one block per (sa1, ta1) pair

    inplace_givens_update_rows_kernel<<<grid, block>>>(
        d_Cout,
        sourcea1, targeta1, paritya1, paritya2,
        na, nbeta_strs_,
        factor, acc_coeff1, acc_coeff2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch inplace_givens_update_rows_kernel ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_rows_kernel launch failed");
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "inplace_givens_update_rows_kernel execution failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_rows_kernel execution failed");
    }
}

template<int BX>  // number of column-pairs handled per block (e.g., 32)
__global__ void inplace_givens_update_complex_tiled(
    cuDoubleComplex* __restrict__ d_Cout,
    const int* __restrict__ sourcea1,
    const int* __restrict__ targeta1,
    const cuDoubleComplex* __restrict__ paritya1,
    const cuDoubleComplex* __restrict__ paritya2,
    const int* __restrict__ sourceb1,
    const int* __restrict__ targetb1,
    const cuDoubleComplex* __restrict__ parityb1,
    const cuDoubleComplex* __restrict__ parityb2,
    int nalpha,          // rows
    int nb,              // number of column-pairs
    long long nbeta_strs_,
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2)
{
    // Block covers BX consecutive column-pairs starting at ib0
    const int ib0 = blockIdx.x * BX;
    if (ib0 >= nb) return;

    // Thread layout: x = column within the tile, y = row lane inside a small row strip
    const int tx = threadIdx.x;             // [0, BX)
    const int ty = threadIdx.y;             // [0, AY)
    constexpr int AY = 8;                   // small row strip per block
    static_assert(BX % 32 == 0, "Pick BX multiple of warp width for coalescing");

    // Shared: BX col-pair metadata + AY row metadata
    __shared__ int s_sb1[BX], s_tb1[BX];
    __shared__ cuDoubleComplex s_pb1[BX], s_pb2[BX];

    __shared__ int s_sa1[AY], s_ta1[AY];
    __shared__ cuDoubleComplex s_pa1[AY], s_pa2[AY];

    // Load the BX column-pairs (one per tx lane; replicate across ty)
    if (tx + ib0 < nb && ty == 0) {
        const int ib = ib0 + tx;
        s_sb1[tx] = sourceb1[ib];
        s_tb1[tx] = targetb1[ib];
        s_pb1[tx] = parityb1[ib];
        s_pb2[tx] = parityb2[ib];
    }
    __syncthreads();

    // Sweep rows in strips of AY
    for (int ia0 = blockIdx.y * AY; ia0 < nalpha; ia0 += gridDim.y * AY)
    {
        // Cache AY row metadata once
        if (ty < AY && tx == 0) {
            const int ia = ia0 + ty;
            if (ia < nalpha) {
                s_sa1[ty] = sourcea1[ia];
                s_ta1[ty] = targeta1[ia];
                s_pa1[ty] = paritya1[ia];
                s_pa2[ty] = paritya2[ia];
            }
        }
        __syncthreads();

        const int ia = ia0 + ty;
        if (ia < nalpha && tx + ib0 < nb) {
            // Registers for the row
            const int sa1 = s_sa1[ty];
            const int ta1 = s_ta1[ty];
            const cuDoubleComplex pa1 = s_pa1[ty];
            const cuDoubleComplex pa2 = s_pa2[ty];

            // Registers for this column-pair
            // const int ib   = ib0 + tx;
            const int sb1  = s_sb1[tx];
            const int tb1  = s_tb1[tx];
            const cuDoubleComplex pb1 = s_pb1[tx];
            const cuDoubleComplex pb2 = s_pb2[tx];

            const long long base_u = (long long)sa1 * nbeta_strs_;
            const long long base_v = (long long)ta1 * nbeta_strs_;

            const long long idx_u  = base_u + sb1;  // (sa1, sb1)
            const long long idx_v  = base_v + tb1;  // (ta1, tb1)

            // Within a warp, tx varies ⇒ idx_* vary by +1 (contiguous) if sb1/tb1 are consecutive.
            // To ensure that, store column-pairs for a tile as consecutive sb1/tb1 (typical).
            const cuDoubleComplex u0 = d_Cout[idx_u];
            const cuDoubleComplex v0 = d_Cout[idx_v];

            const cuDoubleComplex p1 = cuCmul(pa1, pb1);
            const cuDoubleComplex p2 = cuCmul(pa2, pb2);

            const cuDoubleComplex u_new = cuCadd(cuCmul(factor, u0), cuCmul(acc_coeff2, cuCmul(p2, v0)));
            const cuDoubleComplex v_new = cuCadd(cuCmul(factor, v0), cuCmul(acc_coeff1, cuCmul(p1, u0)));

            d_Cout[idx_u] = u_new;
            d_Cout[idx_v] = v_new;
        }
        __syncthreads();
    }
}

// Internal helper to launch a particular BX specialization
template<int BX>
static void launch_inplace_givens_update_complex_tiled(
    cuDoubleComplex* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const cuDoubleComplex* paritya1,
    const cuDoubleComplex* paritya2,
    const int* sourceb1,
    const int* targetb1,
    const cuDoubleComplex* parityb1,
    const cuDoubleComplex* parityb2,
    int nalpha,
    int nb,
    long long nbeta_strs_,
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2)
{
    if (nalpha == 0 || nb == 0 || nbeta_strs_ == 0) return;

    // Must match the kernel's constexpr AY
    constexpr int AY = 8;

    // Each block covers BX consecutive column-pairs and AY rows (as a strip).
    const int grid_x = (nb + BX - 1) / BX;
    const int grid_y = std::max(1, (nalpha + AY - 1) / AY);

    // Block has BX threads along x (columns in the tile) and AY along y (rows in the strip).
    dim3 block(BX, AY);
    dim3 grid(grid_x, grid_y);

    // Sanity: make sure block size is legal (BX*AY <= 1024 on most GPUs)
    if (block.x * block.y > 1024) {
        throw std::invalid_argument("Block size BX*AY exceeds device limit");
    }

    inplace_givens_update_complex_tiled<BX><<<grid, block>>>(
        d_Cout,
        sourcea1, targeta1, paritya1, paritya2,
        sourceb1, targetb1, parityb1, parityb2,
        nalpha, nb, nbeta_strs_,
        factor, acc_coeff1, acc_coeff2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch inplace_givens_update_complex_tiled<"
                  << BX << "> (" << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_complex_tiled launch failed");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "inplace_givens_update_complex_tiled<" << BX
                  << "> execution failed (" << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_complex_tiled execution failed");
    }
}

// Extern "C" wrapper with runtime BX selection.
// Supported BX values are 32 and 64 by default (add more cases as you like).
extern "C" void inplace_givens_update_complex_tiled_wrapper(
    int BX_runtime,                      // pick 32 or 64 (must divide warp multiples)
    cuDoubleComplex* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const cuDoubleComplex* paritya1,
    const cuDoubleComplex* paritya2,
    const int* sourceb1,
    const int* targetb1,
    const cuDoubleComplex* parityb1,
    const cuDoubleComplex* parityb2,
    int nalpha,          // rows
    int nb,              // number of column-pairs
    long long nbeta_strs_,     // leading dimension (num columns)
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2)
{
    if (nalpha == 0 || nb == 0 || nbeta_strs_ == 0) return;

    switch (BX_runtime) {
        case 64:
            launch_inplace_givens_update_complex_tiled<64>(
                d_Cout, sourcea1, targeta1, paritya1, paritya2,
                sourceb1, targetb1, parityb1, parityb2,
                nalpha, nb, nbeta_strs_,
                factor, acc_coeff1, acc_coeff2);
            break;
        case 32:
            launch_inplace_givens_update_complex_tiled<32>(
                d_Cout, sourcea1, targeta1, paritya1, paritya2,
                sourceb1, targetb1, parityb1, parityb2,
                nalpha, nb, nbeta_strs_,
                factor, acc_coeff1, acc_coeff2);
            break;
        default:
            // Fallback or throw—here we fallback to 32 for convenience.
            std::cerr << "Warning: unsupported BX=" << BX_runtime
                      << " — defaulting to BX=32.\n";
            launch_inplace_givens_update_complex_tiled<32>(
                d_Cout, sourcea1, targeta1, paritya1, paritya2,
                sourceb1, targetb1, parityb1, parityb2,
                nalpha, nb, nbeta_strs_,
                factor, acc_coeff1, acc_coeff2);
            break;
    }
}

// ==============================================
// In-place Givens update kernels and wrappers (Real)
// ==============================================

/// One block processes one (sa1, ta1) pair; threads iterate j across nbeta_strs_.
/// pa1, pa2 are row-scoped real parities (±1).
__global__ void inplace_givens_update_rows_kernel_real(
    double* __restrict__ d_Cout,
    const int* __restrict__ sourcea1,      // [na]
    const int* __restrict__ targeta1,      // [na]
    const double* __restrict__ paritya1,   // [na]  (g† leg, row)
    const double* __restrict__ paritya2,   // [na]  (g  leg, row)
    int na,
    long long nbeta_strs_,                        // number of columns
    double factor,
    double acc_coeff1,
    double acc_coeff2)
{
    const int ia = blockIdx.x;  // one block per (sa1, ta1) pair
    if (ia >= na) return;

    // Broadcast row-scoped values once per block
    __shared__ int s_sa1, s_ta1;
    __shared__ double s_pa1, s_pa2;
    if (threadIdx.x == 0) {
        s_sa1 = sourcea1[ia];
        s_ta1 = targeta1[ia];
        s_pa1 = paritya1[ia];
        s_pa2 = paritya2[ia];
    }
    __syncthreads();

    const int sa1 = s_sa1, ta1 = s_ta1;
    const double pa1 = s_pa1, pa2 = s_pa2;

    const long long base_u = (long long)sa1 * nbeta_strs_;
    const long long base_v = (long long)ta1 * nbeta_strs_;

    // Precompute per-row scalings to save a couple MULs in the loop
    const double a_row = acc_coeff2 * pa2;
    const double b_row = acc_coeff1 * pa1;

    // Each thread walks columns with stride blockDim.x (coalesced)
    for (long long col = threadIdx.x; col < nbeta_strs_; col += blockDim.x) {
        const long long idx_u = base_u + col;   // (sa1, col)
        const long long idx_v = base_v + col;   // (ta1, col)

        const double u0 = d_Cout[idx_u];
        const double v0 = d_Cout[idx_v];

        const double u_new = factor * u0 + a_row * v0;
        const double v_new = factor * v0 + b_row * u0;

        d_Cout[idx_u] = u_new;
        d_Cout[idx_v] = v_new;
    }
}

extern "C" void inplace_givens_update_real_rows_wrapper(
    double* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const double* paritya1,
    const double* paritya2,
    int na,
    long long nbeta_strs_,
    double factor,
    double acc_coeff1,
    double acc_coeff2)
{
    if (na == 0 || nbeta_strs_ == 0) return;

    // Choose threads per block: cover columns with good occupancy.
    int threads = static_cast<int>(std::min<long long>(256, nbeta_strs_));
    if (threads < 32) threads = 32;  // keep at least one warp

    dim3 block(threads);
    dim3 grid(na);  // one block per (sa1, ta1) pair

    inplace_givens_update_rows_kernel_real<<<grid, block>>>(
        d_Cout,
        sourcea1, targeta1, paritya1, paritya2,
        na, nbeta_strs_,
        factor, acc_coeff1, acc_coeff2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch inplace_givens_update_rows_kernel_real ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_rows_kernel_real launch failed");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "inplace_givens_update_rows_kernel_real execution failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_rows_kernel_real execution failed");
    }
}

template <int ROWS_PER_THREAD>
__global__ void inplace_givens_update_cols_kernel_real(
    double* __restrict__ d_Cout,
    const int* __restrict__ sourceb1,      // [nb]
    const int* __restrict__ targetb1,      // [nb]
    const double* __restrict__ parityb1,   // [nb]
    const double* __restrict__ parityb2,   // [nb]
    int nb,
    long long nalpha_strs_,
    long long nbeta_strs_,
    double factor,
    double acc_coeff1,
    double acc_coeff2)
{
    const int ib = blockIdx.y;   // beta-pair index
    if (ib >= nb) return;

    // Pair-scoped data: one read per block
    __shared__ int s_sb1, s_tb1;
    __shared__ double s_pb1, s_pb2;

    if (threadIdx.x == 0) {
        s_sb1 = sourceb1[ib];
        s_tb1 = targetb1[ib];
        s_pb1 = parityb1[ib];
        s_pb2 = parityb2[ib];
    }
    __syncthreads();

    const int sb1 = s_sb1;
    const int tb1 = s_tb1;

    const double a_col = acc_coeff2 * s_pb2;
    const double b_col = acc_coeff1 * s_pb1;

    // Tile rows across grid.x, and give each thread multiple rows
    long long row0 =
        static_cast<long long>(blockIdx.x) * (blockDim.x * ROWS_PER_THREAD)
        + threadIdx.x;

    const long long grid_stride =
        static_cast<long long>(gridDim.x) * (blockDim.x * ROWS_PER_THREAD);

    for (long long row = row0; row < nalpha_strs_; row += grid_stride) {

        #pragma unroll
        for (int k = 0; k < ROWS_PER_THREAD; ++k) {
            const long long r = row + static_cast<long long>(k) * blockDim.x;
            if (r < nalpha_strs_) {
                const long long base  = r * nbeta_strs_;
                const long long idx_u = base + sb1;
                const long long idx_v = base + tb1;

                const double u0 = d_Cout[idx_u];
                const double v0 = d_Cout[idx_v];
                
                d_Cout[idx_u] = factor * u0 + a_col * v0;
                d_Cout[idx_v] = factor * v0 + b_col * u0;
            }
        }
    }
}

extern "C" void inplace_givens_update_real_cols_wrapper(
    double* d_Cout,
    const int* sourceb1,
    const int* targetb1,
    const double* parityb1,
    const double* parityb2,
    int nb,
    long long nalpha_strs_,
    long long nbeta_strs_,
    double factor,
    double acc_coeff1,
    double acc_coeff2)
{
    if (nb == 0 || nalpha_strs_ == 0 || nbeta_strs_ == 0) return;

    // Good default for a latency-bound strided kernel.
    constexpr int threads = 256;
    constexpr int rows_per_thread = 4;

    const long long rows_per_block =
        static_cast<long long>(threads) * rows_per_thread;

    long long grid_x_ll =
        (nalpha_strs_ + rows_per_block - 1) / rows_per_block;

    // Clamp to CUDA's 1D grid-x limit for ordinary launches.
    int grid_x = static_cast<int>(std::min<long long>(grid_x_ll, 65535));

    dim3 block(threads);
    dim3 grid(grid_x, nb);

    inplace_givens_update_cols_kernel_real<rows_per_thread><<<grid, block>>>(
        d_Cout,
        sourceb1,
        targetb1,
        parityb1,
        parityb2,
        nb,
        nalpha_strs_,
        nbeta_strs_,
        factor,
        acc_coeff1,
        acc_coeff2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch inplace_givens_update_cols_kernel_real ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_cols_kernel_real launch failed");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "inplace_givens_update_cols_kernel_real execution failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_cols_kernel_real execution failed");
    }
}


template<int BX>  // number of column-pairs handled per block (e.g., 32 or 64)
__global__ void inplace_givens_update_real_tiled(
    double* __restrict__ d_Cout,
    const int* __restrict__ sourcea1,
    const int* __restrict__ targeta1,
    const double* __restrict__ paritya1,
    const double* __restrict__ paritya2,
    const int* __restrict__ sourceb1,
    const int* __restrict__ targetb1,
    const double* __restrict__ parityb1,
    const double* __restrict__ parityb2,
    int nalpha,          // rows
    int nb,              // number of column-pairs
    long long nbeta_strs_,     // leading dimension (num columns)
    double factor,
    double acc_coeff1,
    double acc_coeff2)
{
    // Block covers BX consecutive column-pairs starting at ib0
    const int ib0 = blockIdx.x * BX;
    if (ib0 >= nb) return;

    // Thread layout: x = column within the tile, y = row lane inside a small row strip
    const int tx = threadIdx.x;             // [0, BX)
    const int ty = threadIdx.y;             // [0, AY)
    constexpr int AY = 8;                   // small row strip per block
    static_assert(BX % 32 == 0, "Pick BX multiple of warp width for coalescing");

    // Shared: BX col-pair metadata + AY row metadata
    __shared__ int s_sb1[BX], s_tb1[BX];
    __shared__ double s_pb1[BX], s_pb2[BX];

    __shared__ int s_sa1[AY], s_ta1[AY];
    __shared__ double s_pa1[AY], s_pa2[AY];

    // Load the BX column-pairs (one per tx lane; replicate across ty)
    if (tx + ib0 < nb && ty == 0) {
        const int ib = ib0 + tx;
        s_sb1[tx] = sourceb1[ib];
        s_tb1[tx] = targetb1[ib];
        s_pb1[tx] = parityb1[ib];
        s_pb2[tx] = parityb2[ib];
    }
    __syncthreads();

    // Sweep rows in strips of AY
    for (int ia0 = blockIdx.y * AY; ia0 < nalpha; ia0 += gridDim.y * AY)
    {
        // Cache AY row metadata once
        if (ty < AY && tx == 0) {
            const int ia = ia0 + ty;
            if (ia < nalpha) {
                s_sa1[ty] = sourcea1[ia];
                s_ta1[ty] = targeta1[ia];
                s_pa1[ty] = paritya1[ia];
                s_pa2[ty] = paritya2[ia];
            }
        }
        __syncthreads();

        const int ia = ia0 + ty;
        if (ia < nalpha && tx + ib0 < nb) {
            // Registers for the row
            const int sa1 = s_sa1[ty];
            const int ta1 = s_ta1[ty];
            const double pa1 = s_pa1[ty];
            const double pa2 = s_pa2[ty];

            // Registers for this column-pair
            const int sb1  = s_sb1[tx];
            const int tb1  = s_tb1[tx];
            const double pb1 = s_pb1[tx];
            const double pb2 = s_pb2[tx];

            const long long base_u = (long long)sa1 * nbeta_strs_;
            const long long base_v = (long long)ta1 * nbeta_strs_;

            const long long idx_u  = base_u + sb1;  // (sa1, sb1)
            const long long idx_v  = base_v + tb1;  // (ta1, tb1)

            const double u0 = d_Cout[idx_u];
            const double v0 = d_Cout[idx_v];

            // Real "parity" products
            const double p1 = pa1 * pb1;
            const double p2 = pa2 * pb2;

            // Givens-like coupled update (real)
            const double u_new = factor * u0 + acc_coeff2 * (p2 * v0);
            const double v_new = factor * v0 + acc_coeff1 * (p1 * u0);

            d_Cout[idx_u] = u_new;
            d_Cout[idx_v] = v_new;
        }
        __syncthreads();
    }
}

// Internal helper to launch a particular BX specialization
template<int BX>
static void launch_inplace_givens_update_real_tiled(
    double* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const double* paritya1,
    const double* paritya2,
    const int* sourceb1,
    const int* targetb1,
    const double* parityb1,
    const double* parityb2,
    int nalpha,
    int nb,
    long long nbeta_strs_,
    double factor,
    double acc_coeff1,
    double acc_coeff2)
{
    if (nalpha == 0 || nb == 0 || nbeta_strs_ == 0) return;

    // Must match the kernel's constexpr AY
    constexpr int AY = 8;

    // Each block covers BX consecutive column-pairs and AY rows (as a strip).
    const int grid_x = (nb + BX - 1) / BX;
    const int grid_y = std::max(1, (nalpha + AY - 1) / AY);

    // Block has BX threads along x (columns in the tile) and AY along y (rows in the strip).
    dim3 block(BX, AY);
    dim3 grid(grid_x, grid_y);

    // Sanity: make sure block size is legal (BX*AY <= 1024 on most GPUs)
    if (block.x * block.y > 1024) {
        throw std::invalid_argument("Block size BX*AY exceeds device limit");
    }

    inplace_givens_update_real_tiled<BX><<<grid, block>>>(
        d_Cout,
        sourcea1, targeta1, paritya1, paritya2,
        sourceb1, targetb1, parityb1, parityb2,
        nalpha, nb, nbeta_strs_,
        factor, acc_coeff1, acc_coeff2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch inplace_givens_update_real_tiled<"
                  << BX << "> (" << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_real_tiled launch failed");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "inplace_givens_update_real_tiled<" << BX
                  << "> execution failed (" << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_real_tiled execution failed");
    }
}

// Extern "C" wrapper with runtime BX selection.
// Supported BX values are 32 and 64 by default (add more cases as you like).
extern "C" void inplace_givens_update_real_tiled_wrapper(
    int BX_runtime,                      // pick 32 or 64 (must divide warp multiples)
    double* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const double* paritya1,
    const double* paritya2,
    const int* sourceb1,
    const int* targetb1,
    const double* parityb1,
    const double* parityb2,
    int nalpha,          // rows
    int nb,              // number of column-pairs
    long long nbeta_strs_,     // leading dimension (num columns)
    double factor,
    double acc_coeff1,
    double acc_coeff2)
{
    if (nalpha == 0 || nb == 0 || nbeta_strs_ == 0) return;

    switch (BX_runtime) {
        case 64:
            launch_inplace_givens_update_real_tiled<64>(
                d_Cout, sourcea1, targeta1, paritya1, paritya2,
                sourceb1, targetb1, parityb1, parityb2,
                nalpha, nb, nbeta_strs_,
                factor, acc_coeff1, acc_coeff2);
            break;
        case 32:
            launch_inplace_givens_update_real_tiled<32>(
                d_Cout, sourcea1, targeta1, paritya1, paritya2,
                sourceb1, targetb1, parityb1, parityb2,
                nalpha, nb, nbeta_strs_,
                factor, acc_coeff1, acc_coeff2);
            break;
        default:
            std::cerr << "Warning: unsupported BX=" << BX_runtime
                      << " — defaulting to BX=32.\n";
            launch_inplace_givens_update_real_tiled<32>(
                d_Cout, sourcea1, targeta1, paritya1, paritya2,
                sourceb1, targetb1, parityb1, parityb2,
                nalpha, nb, nbeta_strs_,
                factor, acc_coeff1, acc_coeff2);
            break;
    }
}

// ==============================================
// Beta-only row-major tiled Givens kernel (Real)
//
// Memory layout rationale
// -----------------------
// d_Cout is row-major: element (row, col) lives at row * nbeta_strs_ + col.
//
// OLD column kernel: one block owns one beta pair; threads vary over rows.
//   Lane l touches row_base + l*nbeta_strs_ + sb1  → stride = nbeta_strs_ across lanes.
//   On a row-major matrix this is the worst possible access pattern for coalescing.
//
// THIS kernel: threadIdx.x selects a beta column-pair; threadIdx.y selects a row.
//   All threads in a warp share the same row and touch adjacent columns:
//     base = row * nbeta_strs_;
//     lane l accesses base + sourceb1[ib0 + l]
//   When the sourceb1 array is sorted, neighbouring lanes access nearly-contiguous
//   columns → close to unit-stride, coalesced global loads/stores.
//
// Launch shape: block(BX, AY), grid((nb+BX-1)/BX, (na+AY-1)/AY)
//   BX = 32  → one warp in x (ideal for row-major coalescing)
//   AY = 8   → 256 threads per block (good occupancy target)
//   Grid produces ceil(nb/32) * ceil(na/8) blocks, which is much more than
//   the column kernel's nb * ceil(na/1024) blocks when nb is small.
// ==============================================

template<int BX, int AY>
__global__ void inplace_givens_update_beta_only_rowmajor_real(
    double* __restrict__ d_Cout,
    const int* __restrict__ sourceb1,   // [nb]
    const int* __restrict__ targetb1,   // [nb]
    const double* __restrict__ parityb1, // [nb]  g† parity
    const double* __restrict__ parityb2, // [nb]  g  parity
    int nb,
    long long nalpha_strs_,
    long long nbeta_strs_,
    double factor,
    double acc_coeff1,
    double acc_coeff2)
{
    static_assert(BX % 32 == 0, "BX must be a warp multiple for coalescing");

    const int tx  = threadIdx.x;          // beta-pair lane within block
    const int ty  = threadIdx.y;          // row lane within block

    const int ib  = blockIdx.x * BX + tx; // global beta-pair index
    const long long row = (long long)blockIdx.y * AY + ty; // global alpha-row index

    // ---- Shared memory: BX beta-pair metadata --------------------------------
    // Only row ty==0 loads; all rows reuse it.  One __syncthreads() is enough.
    __shared__ int    s_sb1[BX], s_tb1[BX];
    __shared__ double s_a[BX],   s_b[BX];  // pre-scaled parity products

    if (ty == 0) {
        if (ib < nb) {
            s_sb1[tx] = sourceb1[ib];
            s_tb1[tx] = targetb1[ib];
            // absorbed scalars: avoids two multiplies per thread in the hot path
            s_a[tx]   = acc_coeff2 * parityb2[ib];  // coefficient on v0 → u
            s_b[tx]   = acc_coeff1 * parityb1[ib];  // coefficient on u0 → v
        } else {
            // Out-of-range lane — init to harmless sentinel (will be guarded below)
            s_sb1[tx] = 0;
            s_tb1[tx] = 0;
            s_a[tx]   = 0.0;
            s_b[tx]   = 0.0;
        }
    }
    __syncthreads();

    // ---- Hot path: row-local read-modify-write -------------------------------
    // All active threads in a warp share the same `row`, so
    //   base + s_sb1[tx]  for consecutive tx values  → adjacent column accesses.
    // When sourceb1 is sorted this collapses to one or two 32-byte cache lines.
    if (row < nalpha_strs_ && ib < nb) {
        const long long base = (long long)row * nbeta_strs_;

        const double u0 = d_Cout[base + s_sb1[tx]];
        const double v0 = d_Cout[base + s_tb1[tx]];

        d_Cout[base + s_sb1[tx]] = fma(s_a[tx], v0, factor * u0);
        d_Cout[base + s_tb1[tx]] = fma(s_b[tx], u0, factor * v0);
    }
}

// Internal template launcher
template<int BX, int AY>
static void launch_inplace_givens_update_beta_only_rowmajor_real(
    double* d_Cout,
    const int* sourceb1,
    const int* targetb1,
    const double* parityb1,
    const double* parityb2,
    int nb,
    long long nalpha_strs_,
    long long nbeta_strs_,
    double factor,
    double acc_coeff1,
    double acc_coeff2)
{
    if (nb == 0 || nalpha_strs_ == 0 || nbeta_strs_ == 0) return;

    // grid.y can be at most 65535 for ordinary launches; each block covers AY rows.
    // With AY=8, this supports up to 65535*8 = 524280 rows (sufficient for FCI).
    const int grid_x = (nb           + BX - 1) / BX;
    const int grid_y = (int)((nalpha_strs_ + AY - 1) / AY);

    dim3 block(BX, AY);                  // BX * AY threads, e.g. 32*8 = 256
    dim3 grid(grid_x, grid_y);

    inplace_givens_update_beta_only_rowmajor_real<BX, AY><<<grid, block>>>(
        d_Cout,
        sourceb1, targetb1, parityb1, parityb2,
        nb, nalpha_strs_, nbeta_strs_,
        factor, acc_coeff1, acc_coeff2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch inplace_givens_update_beta_only_rowmajor_real<"
                  << BX << "," << AY << "> (" << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_beta_only_rowmajor_real launch failed");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "inplace_givens_update_beta_only_rowmajor_real<"
                  << BX << "," << AY << "> execution failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_beta_only_rowmajor_real execution failed");
    }
}

// Extern "C" wrapper: BX=32, AY=8 (256 threads/block) is the default tuned shape.
// If nb >> 32 and the GPU has enough SMs, BX=64 may give better occupancy.
extern "C" void inplace_givens_update_real_beta_only_rowmajor_wrapper(
    double* d_Cout,
    const int* sourceb1,
    const int* targetb1,
    const double* parityb1,
    const double* parityb2,
    int nb,
    long long nalpha_strs_,
    long long nbeta_strs_,
    double factor,
    double acc_coeff1,
    double acc_coeff2)
{
    if (nb == 0 || nalpha_strs_ == 0 || nbeta_strs_ == 0) return;

    // BX=32 → one full warp per row-tile: optimal coalescing for row-major data.
    // BX=64 can help when nb is large enough that two warps per block are busy,
    // but risks register spill on some architectures.  Default to 32.
    launch_inplace_givens_update_beta_only_rowmajor_real<32, 8>(
        d_Cout,
        sourceb1, targetb1, parityb1, parityb2,
        nb, nalpha_strs_, nbeta_strs_,
        factor, acc_coeff1, acc_coeff2);
}

// ==============================================
// Same Spin kernel implementation (raw COO + sort + reduce)
// ==============================================

__global__ void same_spin_build_raw_keys_vals_kernel(
    unsigned long long* __restrict__ d_keys,   // [nnz_raw]
    cuDoubleComplex* __restrict__ d_vals,      // [nnz_raw]
    const int* __restrict__ d_dexc,            // [states1 * ndexc * 3]
    const cuDoubleComplex* __restrict__ d_h1e, // [norbs2]
    const cuDoubleComplex* __restrict__ d_h2e, // [norbs2 * norbs2]
    int states1,
    int ndexc,
    int norbs)
{
    const int norbs2 = norbs * norbs;

    const long long nnz_per_row = (long long)ndexc * (long long)(ndexc + 1);
    const long long nnz_raw     = (long long)states1 * nnz_per_row;

    // p is the global index into the raw COO arrays
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nnz_raw) return;

    // from p we can get s1 (row index) and position t inside row
    const int s1 = (int)(p / nnz_per_row);
    const int t  = (int)(p - (long long)s1 * nnz_per_row);

    const int base_s1 = 3 * (s1 * ndexc);

    int col = 0;
    cuDoubleComplex val = make_cuDoubleComplex(0.0, 0.0);

    if (t < ndexc) {
        // h1e term: (s1 -> s2)
        const int i = t;

        const int s2      = d_dexc[base_s1 + 3*i + 0];
        const int ijshift = d_dexc[base_s1 + 3*i + 1];
        const int parity1 = d_dexc[base_s1 + 3*i + 2];

        col = s2;

        val = d_h1e[ijshift];
        if (parity1 == -1) { val.x = -val.x; val.y = -val.y; }

    } else {
        // h2e term: (s1 -> target) via s2 row
        const int t2 = t - ndexc;
        const int i  = t2 / ndexc;
        const int j  = t2 - i*ndexc;

        const int s2      = d_dexc[base_s1 + 3*i + 0];
        const int ijshift = d_dexc[base_s1 + 3*i + 1];
        const int parity1 = d_dexc[base_s1 + 3*i + 2];

        const int base_s2 = 3 * (s2 * ndexc);

        const int target  = d_dexc[base_s2 + 3*j + 0];
        const int klshift = d_dexc[base_s2 + 3*j + 1];
        const int parity2 = d_dexc[base_s2 + 3*j + 2];

        col = target;

        val = d_h2e[(long long)ijshift * norbs2 + klshift];
        const int parity = parity1 * parity2;
        if (parity == -1) { val.x = -val.x; val.y = -val.y; }
    }

    // Safety: keep indices in range; if out-of-range, write a 0 entry
    if (col < 0 || col >= states1) {
        col = 0;
        val = make_cuDoubleComplex(0.0, 0.0);
    }

    const unsigned long long key =
        ( (unsigned long long)(unsigned int)s1 << 32 ) |
        ( (unsigned long long)(unsigned int)col );

    d_keys[p] = key;
    d_vals[p] = val;
}

__global__ void same_spin_extract_cols_and_rowcounts_kernel(
    const unsigned long long* __restrict__ d_keys_unique, // [nnz_unique]
    int nnz_unique,
    int* __restrict__ d_col_ind,      // [nnz_unique]
    int* __restrict__ d_row_counts,   // [states1]
    int states1)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz_unique) return;

    unsigned long long key = d_keys_unique[k];
    int row = (int)(key >> 32);
    int col = (int)(key & 0xFFFFFFFFull);

    d_col_ind[k] = col;

    if (row >= 0 && row < states1) {
        atomicAdd(&d_row_counts[row], 1);
    }
}

extern "C" void lm_apply_array12_same_spin_spmm_csr_coalesced_wrapper(
    cuDoubleComplex* d_out,
    const cuDoubleComplex* d_C,
    const int* d_dexc,
    const cuDoubleComplex* d_h1e,
    const cuDoubleComplex* d_h2e,
    int states1,
    int states2,
    int ndexc,
    int norbs,
    int inc1,
    int inc2)
{
    if (states1 <= 0 || states2 <= 0 || ndexc <= 0) return;

    // ------------------------------------------------------------
    // 1) Build RAW COO contributions (duplicates allowed)
    //    raw nnz per row = ndexc + ndexc*ndexc = ndexc*(ndexc+1)
    // ------------------------------------------------------------
    const long long nnz_per_row = (long long)ndexc * (long long)(ndexc + 1);
    const long long nnz_raw_ll  = (long long)states1 * nnz_per_row;

    if (nnz_raw_ll <= 0) return;
    if (nnz_raw_ll > (long long)std::numeric_limits<int>::max()) {
        // You can support larger by switching some ints to int64 in cuSPARSE.
        throw std::runtime_error("nnz_raw too large for this 32-bit implementation");
    }

    const int nnz_raw = (int)nnz_raw_ll;

    thrust::device_vector<unsigned long long> d_keys_raw(nnz_raw);  // (row<<32 | col)
    thrust::device_vector<cuDoubleComplex>    d_vals_raw(nnz_raw);

    {
        int threads = 256;
        int blocks  = (nnz_raw + threads - 1) / threads;

        same_spin_build_raw_keys_vals_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_keys_raw.data()),
            thrust::raw_pointer_cast(d_vals_raw.data()),
            d_dexc, d_h1e, d_h2e,
            states1, ndexc, norbs);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // ------------------------------------------------------------
    // 2) Sort by (row,col) key
    // ------------------------------------------------------------
    thrust::sort_by_key(d_keys_raw.begin(), d_keys_raw.end(), d_vals_raw.begin());

    // ------------------------------------------------------------
    // 3) Reduce duplicates: (row,col) sums its contributions
    // ------------------------------------------------------------
    thrust::device_vector<unsigned long long> d_keys_uni(nnz_raw);
    thrust::device_vector<cuDoubleComplex>    d_vals_uni(nnz_raw);

    auto end_pair = thrust::reduce_by_key(
        d_keys_raw.begin(), d_keys_raw.end(),
        d_vals_raw.begin(),
        d_keys_uni.begin(),
        d_vals_uni.begin(),
        thrust::equal_to<unsigned long long>(),
        cuCadd_op());

    int nnz_unique = (int)(end_pair.first - d_keys_uni.begin());
    if (nnz_unique <= 0) return;

    // Free raw COO data — no longer needed after reduce
    d_keys_raw.clear(); d_keys_raw.shrink_to_fit();
    d_vals_raw.clear(); d_vals_raw.shrink_to_fit();

    d_keys_uni.resize(nnz_unique);
    d_vals_uni.resize(nnz_unique);

    // ------------------------------------------------------------
    // 4) Build CSR row_ptr + col_ind from unique COO
    // ------------------------------------------------------------
    thrust::device_vector<int> d_row_counts(states1);
    thrust::fill(d_row_counts.begin(), d_row_counts.end(), 0);

    thrust::device_vector<int> d_col_ind(nnz_unique);

    {
        int threads = 256;
        int blocks  = (nnz_unique + threads - 1) / threads;

        same_spin_extract_cols_and_rowcounts_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_keys_uni.data()),
            nnz_unique,
            thrust::raw_pointer_cast(d_col_ind.data()),
            thrust::raw_pointer_cast(d_row_counts.data()),
            states1);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Free unique keys — columns already extracted into d_col_ind
    d_keys_uni.clear(); d_keys_uni.shrink_to_fit();

    // CSR row_ptr: exclusive_scan(row_counts) + last element = nnz_unique
    thrust::device_vector<int> d_row_ptr(states1 + 1);
    thrust::exclusive_scan(d_row_counts.begin(), d_row_counts.end(), d_row_ptr.begin());

    CHECK_CUDA(cudaMemcpy(
        thrust::raw_pointer_cast(d_row_ptr.data()) + states1,
        &nnz_unique,
        sizeof(int),
        cudaMemcpyHostToDevice));

    // ------------------------------------------------------------
    // 5) cuSPARSE SpMM: out += A * C
    // A is CSR(states1 x states1), B is dense(states1 x states2), C is dense(states1 x states2)
    // ------------------------------------------------------------
    static cusparseHandle_t handle = nullptr;
    static bool handle_init = false;
    if (!handle_init) {
        CHECK_CUSPARSE(cusparseCreate(&handle));
        handle_init = true;
    }

    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA,
        (int64_t)states1, (int64_t)states1, (int64_t)nnz_unique,
        (void*)thrust::raw_pointer_cast(d_row_ptr.data()),
        (void*)thrust::raw_pointer_cast(d_col_ind.data()),
        (void*)thrust::raw_pointer_cast(d_vals_uni.data()),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_C_64F));

    // Dense matrix descriptors:
    // cuSPARSE only supports ROW or COL order (not arbitrary strides).
    cusparseOrder_t order;
    int ld = 0;

    if (inc2 == 1) {
        // row-major: idx = row*inc1 + col*1, so ld should be #cols == states2
        order = CUSPARSE_ORDER_ROW;
        ld = inc1;
    } else if (inc1 == 1) {
        // col-major: idx = row*1 + col*inc2, so ld should be #rows == states1
        order = CUSPARSE_ORDER_COL;
        ld = inc2;
    } else {
        CHECK_CUSPARSE(cusparseDestroySpMat(matA));
        throw std::runtime_error("Unsupported dense layout (need row-major or col-major contiguous)");
    }

    cusparseDnMatDescr_t matB, matC;
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matB,
        (int64_t)states1, (int64_t)states2, (int64_t)ld,
        (void*)d_C,
        CUDA_C_64F,
        order));

    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matC,
        (int64_t)states1, (int64_t)states2, (int64_t)ld,
        (void*)d_out,
        CUDA_C_64F,
        order));

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta  = make_cuDoubleComplex(1.0, 0.0); // accumulate into existing out

    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA, matB,
        &beta,
        matC,
        CUDA_C_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize));

    void* dBuffer = nullptr;
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    CHECK_CUSPARSE(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA, matB,
        &beta,
        matC,
        CUDA_C_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        dBuffer));

    CHECK_CUDA(cudaFree(dBuffer));

    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ==============================================
// Same Spin kernel implementation (Real version)
// ==============================================

__global__ void same_spin_build_raw_keys_vals_kernel_real(
    unsigned long long* __restrict__ d_keys,   // [nnz_raw]
    double* __restrict__ d_vals,               // [nnz_raw]
    const int* __restrict__ d_dexc,            // [states1 * ndexc * 3]
    const double* __restrict__ d_h1e,          // [norbs2]
    const double* __restrict__ d_h2e,          // [norbs2 * norbs2]
    int states1,
    int ndexc,
    int norbs)
{
    const int norbs2 = norbs * norbs;

    const long long nnz_per_row = (long long)ndexc * (long long)(ndexc + 1);
    const long long nnz_raw     = (long long)states1 * nnz_per_row;

    // p is the global index into the raw COO arrays
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nnz_raw) return;

    // from p we can get s1 (row index) and position t inside row
    const int s1 = (int)(p / nnz_per_row);
    const int t  = (int)(p - (long long)s1 * nnz_per_row);

    const int base_s1 = 3 * (s1 * ndexc);

    int col = 0;
    double val = 0.0;

    if (t < ndexc) {
        // h1e term: (s1 -> s2)
        const int i = t;

        const int s2      = d_dexc[base_s1 + 3*i + 0];
        const int ijshift = d_dexc[base_s1 + 3*i + 1];
        const int parity1 = d_dexc[base_s1 + 3*i + 2];

        col = s2;

        val = d_h1e[ijshift];
        if (parity1 == -1) { val = -val; }

    } else {
        // h2e term: (s1 -> target) via s2 row
        const int t2 = t - ndexc;
        const int i  = t2 / ndexc;
        const int j  = t2 - i*ndexc;

        const int s2      = d_dexc[base_s1 + 3*i + 0];
        const int ijshift = d_dexc[base_s1 + 3*i + 1];
        const int parity1 = d_dexc[base_s1 + 3*i + 2];

        const int base_s2 = 3 * (s2 * ndexc);

        const int target  = d_dexc[base_s2 + 3*j + 0];
        const int klshift = d_dexc[base_s2 + 3*j + 1];
        const int parity2 = d_dexc[base_s2 + 3*j + 2];

        col = target;

        val = d_h2e[(long long)ijshift * norbs2 + klshift];
        const int parity = parity1 * parity2;
        if (parity == -1) { val = -val; }
    }

    // Safety: keep indices in range; if out-of-range, write a 0 entry
    if (col < 0 || col >= states1) {
        col = 0;
        val = 0.0;
    }

    const unsigned long long key =
        ( (unsigned long long)(unsigned int)s1 << 32 ) |
        ( (unsigned long long)(unsigned int)col );

    d_keys[p] = key;
    d_vals[p] = val;
}

extern "C" void lm_apply_array12_same_spin_spmm_csr_coalesced_wrapper_real(
    double* d_out,
    const double* d_C,
    const int* d_dexc,
    const double* d_h1e,
    const double* d_h2e,
    int states1,
    int states2,
    int ndexc,
    int norbs,
    int inc1,
    int inc2)
{
    if (states1 <= 0 || states2 <= 0 || ndexc <= 0) return;

    // ------------------------------------------------------------
    // 1) Build RAW COO contributions (duplicates allowed)
    //    raw nnz per row = ndexc + ndexc*ndexc = ndexc*(ndexc+1)
    // ------------------------------------------------------------
    const long long nnz_per_row = (long long)ndexc * (long long)(ndexc + 1);
    const long long nnz_raw_ll  = (long long)states1 * nnz_per_row;

    if (nnz_raw_ll <= 0) return;
    if (nnz_raw_ll > (long long)std::numeric_limits<int>::max()) {
        throw std::runtime_error("nnz_raw too large for this 32-bit implementation");
    }

    const int nnz_raw = (int)nnz_raw_ll;

    thrust::device_vector<unsigned long long> d_keys_raw(nnz_raw);  // (row<<32 | col)
    thrust::device_vector<double>             d_vals_raw(nnz_raw);

    {
        int threads = 256;
        int blocks  = (nnz_raw + threads - 1) / threads;

        same_spin_build_raw_keys_vals_kernel_real<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_keys_raw.data()),
            thrust::raw_pointer_cast(d_vals_raw.data()),
            d_dexc, d_h1e, d_h2e,
            states1, ndexc, norbs);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // ------------------------------------------------------------
    // 2) Sort by (row,col) key
    // ------------------------------------------------------------
    thrust::sort_by_key(d_keys_raw.begin(), d_keys_raw.end(), d_vals_raw.begin());

    // ------------------------------------------------------------
    // 3) Reduce duplicates: (row,col) sums its contributions
    // ------------------------------------------------------------
    thrust::device_vector<unsigned long long> d_keys_uni(nnz_raw);
    thrust::device_vector<double>             d_vals_uni(nnz_raw);

    auto end_pair = thrust::reduce_by_key(
        d_keys_raw.begin(), d_keys_raw.end(),
        d_vals_raw.begin(),
        d_keys_uni.begin(),
        d_vals_uni.begin(),
        thrust::equal_to<unsigned long long>(),
        thrust::plus<double>());

    int nnz_unique = (int)(end_pair.first - d_keys_uni.begin());
    if (nnz_unique <= 0) return;

    // Free raw COO data — no longer needed after reduce
    d_keys_raw.clear(); d_keys_raw.shrink_to_fit();
    d_vals_raw.clear(); d_vals_raw.shrink_to_fit();

    d_keys_uni.resize(nnz_unique);
    d_vals_uni.resize(nnz_unique);

    // ------------------------------------------------------------
    // 4) Build CSR row_ptr + col_ind from unique COO
    // ------------------------------------------------------------
    thrust::device_vector<int> d_row_counts(states1);
    thrust::fill(d_row_counts.begin(), d_row_counts.end(), 0);

    thrust::device_vector<int> d_col_ind(nnz_unique);

    {
        int threads = 256;
        int blocks  = (nnz_unique + threads - 1) / threads;

        same_spin_extract_cols_and_rowcounts_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_keys_uni.data()),
            nnz_unique,
            thrust::raw_pointer_cast(d_col_ind.data()),
            thrust::raw_pointer_cast(d_row_counts.data()),
            states1);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Free unique keys — columns already extracted into d_col_ind
    d_keys_uni.clear(); d_keys_uni.shrink_to_fit();

    // CSR row_ptr: exclusive_scan(row_counts) + last element = nnz_unique
    thrust::device_vector<int> d_row_ptr(states1 + 1);
    thrust::exclusive_scan(d_row_counts.begin(), d_row_counts.end(), d_row_ptr.begin());

    CHECK_CUDA(cudaMemcpy(
        thrust::raw_pointer_cast(d_row_ptr.data()) + states1,
        &nnz_unique,
        sizeof(int),
        cudaMemcpyHostToDevice));

    // ------------------------------------------------------------
    // 5) cuSPARSE SpMM: out += A * C
    // A is CSR(states1 x states1), B is dense(states1 x states2), C is dense(states1 x states2)
    // ------------------------------------------------------------
    static cusparseHandle_t handle = nullptr;
    static bool handle_init = false;
    if (!handle_init) {
        CHECK_CUSPARSE(cusparseCreate(&handle));
        handle_init = true;
    }

    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA,
        (int64_t)states1, (int64_t)states1, (int64_t)nnz_unique,
        (void*)thrust::raw_pointer_cast(d_row_ptr.data()),
        (void*)thrust::raw_pointer_cast(d_col_ind.data()),
        (void*)thrust::raw_pointer_cast(d_vals_uni.data()),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));

    // Dense matrix descriptors:
    cusparseOrder_t order;
    int ld = 0;

    if (inc2 == 1) {
        // row-major: idx = row*inc1 + col*1, so ld should be #cols == states2
        order = CUSPARSE_ORDER_ROW;
        ld = inc1;
    } else if (inc1 == 1) {
        // col-major: idx = row*1 + col*inc2, so ld should be #rows == states1
        order = CUSPARSE_ORDER_COL;
        ld = inc2;
    } else {
        CHECK_CUSPARSE(cusparseDestroySpMat(matA));
        throw std::runtime_error("Unsupported dense layout (need row-major or col-major contiguous)");
    }

    cusparseDnMatDescr_t matB, matC;
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matB,
        (int64_t)states1, (int64_t)states2, (int64_t)ld,
        (void*)d_C,
        CUDA_R_64F,
        order));

    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matC,
        (int64_t)states1, (int64_t)states2, (int64_t)ld,
        (void*)d_out,
        CUDA_R_64F,
        order));

    double alpha = 1.0;
    double beta  = 1.0; // accumulate into existing out

    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA, matB,
        &beta,
        matC,
        CUDA_R_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize));

    void* dBuffer = nullptr;
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    CHECK_CUSPARSE(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA, matB,
        &beta,
        matC,
        CUDA_R_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        dBuffer));

    CHECK_CUDA(cudaFree(dBuffer));

    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ==============================================
// Same Spin kernel implementation (Mixed: real h, complex state)
// ==============================================

__global__ void same_spin_build_raw_keys_vals_kernel_mixed(
    unsigned long long* __restrict__ d_keys,   // [nnz_raw]
    cuDoubleComplex* __restrict__ d_vals,      // [nnz_raw]
    const int* __restrict__ d_dexc,            // [states1 * ndexc * 3]
    const double* __restrict__ d_h1e,          // [norbs2] - REAL
    const double* __restrict__ d_h2e,          // [norbs2 * norbs2] - REAL
    int states1,
    int ndexc,
    int norbs)
{
    const int norbs2 = norbs * norbs;

    const long long nnz_per_row = (long long)ndexc * (long long)(ndexc + 1);
    const long long nnz_raw     = (long long)states1 * nnz_per_row;

    // p is the global index into the raw COO arrays
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nnz_raw) return;

    // from p we can get s1 (row index) and position t inside row
    const int s1 = (int)(p / nnz_per_row);
    const int t  = (int)(p - (long long)s1 * nnz_per_row);

    const int base_s1 = 3 * (s1 * ndexc);

    int col = 0;
    cuDoubleComplex val = make_cuDoubleComplex(0.0, 0.0);

    if (t < ndexc) {
        // h1e term: (s1 -> s2)
        const int i = t;

        const int s2      = d_dexc[base_s1 + 3*i + 0];
        const int ijshift = d_dexc[base_s1 + 3*i + 1];
        const int parity1 = d_dexc[base_s1 + 3*i + 2];

        col = s2;

        double h_val = d_h1e[ijshift];  // Read real value
        val = make_cuDoubleComplex(h_val, 0.0);  // Convert to complex
        if (parity1 == -1) { val.x = -val.x; val.y = -val.y; }

    } else {
        // h2e term: (s1 -> target) via s2 row
        const int t2 = t - ndexc;
        const int i  = t2 / ndexc;
        const int j  = t2 - i*ndexc;

        const int s2      = d_dexc[base_s1 + 3*i + 0];
        const int ijshift = d_dexc[base_s1 + 3*i + 1];
        const int parity1 = d_dexc[base_s1 + 3*i + 2];

        const int base_s2 = 3 * (s2 * ndexc);

        const int target  = d_dexc[base_s2 + 3*j + 0];
        const int klshift = d_dexc[base_s2 + 3*j + 1];
        const int parity2 = d_dexc[base_s2 + 3*j + 2];

        col = target;

        double h_val = d_h2e[(long long)ijshift * norbs2 + klshift];  // Read real value
        val = make_cuDoubleComplex(h_val, 0.0);  // Convert to complex
        const int parity = parity1 * parity2;
        if (parity == -1) { val.x = -val.x; val.y = -val.y; }
    }

    // Safety: keep indices in range; if out-of-range, write a 0 entry
    if (col < 0 || col >= states1) {
        col = 0;
        val = make_cuDoubleComplex(0.0, 0.0);
    }

    const unsigned long long key =
        ( (unsigned long long)(unsigned int)s1 << 32 ) |
        ( (unsigned long long)(unsigned int)col );

    d_keys[p] = key;
    d_vals[p] = val;
}

extern "C" void lm_apply_array12_same_spin_spmm_csr_coalesced_wrapper_mixed(
    cuDoubleComplex* d_out,
    const cuDoubleComplex* d_C,
    const int* d_dexc,
    const double* d_h1e,
    const double* d_h2e,
    int states1,
    int states2,
    int ndexc,
    int norbs,
    int inc1,
    int inc2)
{
    if (states1 <= 0 || states2 <= 0 || ndexc <= 0) return;

    // ------------------------------------------------------------
    // 1) Build RAW COO contributions (duplicates allowed)
    //    raw nnz per row = ndexc + ndexc*ndexc = ndexc*(ndexc+1)
    // ------------------------------------------------------------
    const long long nnz_per_row = (long long)ndexc * (long long)(ndexc + 1);
    const long long nnz_raw_ll  = (long long)states1 * nnz_per_row;

    if (nnz_raw_ll <= 0) return;
    if (nnz_raw_ll > (long long)std::numeric_limits<int>::max()) {
        throw std::runtime_error("nnz_raw too large for this 32-bit implementation");
    }

    const int nnz_raw = (int)nnz_raw_ll;

    thrust::device_vector<unsigned long long> d_keys_raw(nnz_raw);  // (row<<32 | col)
    thrust::device_vector<cuDoubleComplex>    d_vals_raw(nnz_raw);

    {
        int threads = 256;
        int blocks  = (nnz_raw + threads - 1) / threads;

        same_spin_build_raw_keys_vals_kernel_mixed<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_keys_raw.data()),
            thrust::raw_pointer_cast(d_vals_raw.data()),
            d_dexc, d_h1e, d_h2e,
            states1, ndexc, norbs);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // ------------------------------------------------------------
    // 2) Sort by (row,col) key
    // ------------------------------------------------------------
    thrust::sort_by_key(d_keys_raw.begin(), d_keys_raw.end(), d_vals_raw.begin());

    // ------------------------------------------------------------
    // 3) Reduce duplicates: (row,col) sums its contributions
    // ------------------------------------------------------------
    thrust::device_vector<unsigned long long> d_keys_uni(nnz_raw);
    thrust::device_vector<cuDoubleComplex>    d_vals_uni(nnz_raw);

    auto end_pair = thrust::reduce_by_key(
        d_keys_raw.begin(), d_keys_raw.end(),
        d_vals_raw.begin(),
        d_keys_uni.begin(),
        d_vals_uni.begin(),
        thrust::equal_to<unsigned long long>(),
        cuCadd_op());

    int nnz_unique = (int)(end_pair.first - d_keys_uni.begin());
    if (nnz_unique <= 0) return;

    // Free raw COO data — no longer needed after reduce
    d_keys_raw.clear(); d_keys_raw.shrink_to_fit();
    d_vals_raw.clear(); d_vals_raw.shrink_to_fit();

    d_keys_uni.resize(nnz_unique);
    d_vals_uni.resize(nnz_unique);

    // ------------------------------------------------------------
    // 4) Build CSR row_ptr + col_ind from unique COO
    // ------------------------------------------------------------
    thrust::device_vector<int> d_row_counts(states1);
    thrust::fill(d_row_counts.begin(), d_row_counts.end(), 0);

    thrust::device_vector<int> d_col_ind(nnz_unique);

    {
        int threads = 256;
        int blocks  = (nnz_unique + threads - 1) / threads;

        same_spin_extract_cols_and_rowcounts_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_keys_uni.data()),
            nnz_unique,
            thrust::raw_pointer_cast(d_col_ind.data()),
            thrust::raw_pointer_cast(d_row_counts.data()),
            states1);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Free unique keys — columns already extracted into d_col_ind
    d_keys_uni.clear(); d_keys_uni.shrink_to_fit();

    // CSR row_ptr: exclusive_scan(row_counts) + last element = nnz_unique
    thrust::device_vector<int> d_row_ptr(states1 + 1);
    thrust::exclusive_scan(d_row_counts.begin(), d_row_counts.end(), d_row_ptr.begin());

    CHECK_CUDA(cudaMemcpy(
        thrust::raw_pointer_cast(d_row_ptr.data()) + states1,
        &nnz_unique,
        sizeof(int),
        cudaMemcpyHostToDevice));

    // ------------------------------------------------------------
    // 5) cuSPARSE SpMM: out += A * C
    // A is CSR(states1 x states1), B is dense(states1 x states2), C is dense(states1 x states2)
    // ------------------------------------------------------------
    static cusparseHandle_t handle = nullptr;
    static bool handle_init = false;
    if (!handle_init) {
        CHECK_CUSPARSE(cusparseCreate(&handle));
        handle_init = true;
    }

    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA,
        (int64_t)states1, (int64_t)states1, (int64_t)nnz_unique,
        (void*)thrust::raw_pointer_cast(d_row_ptr.data()),
        (void*)thrust::raw_pointer_cast(d_col_ind.data()),
        (void*)thrust::raw_pointer_cast(d_vals_uni.data()),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_C_64F));

    // Dense matrix descriptors:
    cusparseOrder_t order;
    int ld = 0;

    if (inc2 == 1) {
        order = CUSPARSE_ORDER_ROW;
        ld = inc1;
    } else if (inc1 == 1) {
        order = CUSPARSE_ORDER_COL;
        ld = inc2;
    } else {
        CHECK_CUSPARSE(cusparseDestroySpMat(matA));
        throw std::runtime_error("Unsupported dense layout (need row-major or col-major contiguous)");
    }

    cusparseDnMatDescr_t matB, matC;
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matB,
        (int64_t)states1, (int64_t)states2, (int64_t)ld,
        (void*)d_C,
        CUDA_C_64F,
        order));

    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matC,
        (int64_t)states1, (int64_t)states2, (int64_t)ld,
        (void*)d_out,
        CUDA_C_64F,
        order));

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta  = make_cuDoubleComplex(1.0, 0.0); // accumulate into existing out

    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA, matB,
        &beta,
        matC,
        CUDA_C_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize));

    void* dBuffer = nullptr;
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    CHECK_CUSPARSE(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA, matB,
        &beta,
        matC,
        CUDA_C_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        dBuffer));

    CHECK_CUDA(cudaFree(dBuffer));

    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ==============================================
// Diff Spin kernel implementation
// ==============================================

// ---------------------------------------------------------
// 1) Count alpha excitations per orbid: d_counts[orbid]
//    adexc layout: for each (s1, i) with idx = s1*nadexc + i
//      [ coff, orbij, sign ]
// ---------------------------------------------------------
__global__ void count_alpha_excitations_per_orbid_kernel(
    const int* __restrict__ d_adexc,
    int alpha_states,
    int nadexc,
    int norbs2,
    int* __restrict__ d_counts)
{
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int total = alpha_states * nadexc;
    if (flat >= total) return;

    int base  = 3 * flat;
    int orbij = d_adexc[base + 1];
    if (orbij >= 0 && orbij < norbs2) {
        atomicAdd(&d_counts[orbij], 1);
    }
}

// ---------------------------------------------------------
// 2) Fill CSR arrays for alpha excitations:
//
// d_ad_offsets: size norbs2+1 (CSR offsets)
// d_cursors:    size norbs2, initialized to d_ad_offsets[0..norbs2-1]
// d_ad_coff:    size total_ex (coff)
// d_ad_boff:    size total_ex (boff = alpha state index)
// d_ad_sign:    size total_ex (±1)
// ---------------------------------------------------------
__global__ void fill_alpha_csr_from_adexc_kernel(
    const int* __restrict__ d_adexc,
    int alpha_states,
    int nadexc,
    int norbs2,
    const int* __restrict__ d_ad_offsets,
    int* __restrict__ d_cursors,
    int* __restrict__ d_ad_coff,
    int* __restrict__ d_ad_boff,
    int* __restrict__ d_ad_sign)
{
    int flat  = blockIdx.x * blockDim.x + threadIdx.x;
    int total = alpha_states * nadexc;
    if (flat >= total) return;

    int base   = 3 * flat;
    int coff   = d_adexc[base + 0];
    int orbij  = d_adexc[base + 1];
    int sign   = d_adexc[base + 2];

    if (orbij < 0 || orbij >= norbs2) return;

    int pos = atomicAdd(&d_cursors[orbij], 1);

    d_ad_coff[pos] = coff;
    d_ad_boff[pos] = flat / nadexc; // s1 = row index in adexc
    d_ad_sign[pos] = sign;
}

// ---------------------------------------------------------
// 3) Split beta excitations into SoA:
//    bdexc layout: for each (s2, j) with flat = s2*nbdexc + j
//      [ idx2, orbkl, parity ]
// ---------------------------------------------------------
__global__ void split_bdexc_kernel(
    const int* __restrict__ d_bdexc,
    int beta_states,
    int nbdexc,
    int* __restrict__ d_bd_idx2,
    int* __restrict__ d_bd_orbkl,
    int* __restrict__ d_bd_parity)
{
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int total = beta_states * nbdexc;
    if (flat >= total) return;

    int base = 3 * flat;
    d_bd_idx2[flat]   = d_bdexc[base + 0];
    d_bd_orbkl[flat]  = d_bdexc[base + 1];
    d_bd_parity[flat] = d_bdexc[base + 2];
}

// ---------------------------------------------------------
// Diff spin: cache bdexc row + weights in shared once per block
// ---------------------------------------------------------
__global__ void lm_apply_array12_diff_spin_kernel(
    cuDoubleComplex* __restrict__ d_out,
    const cuDoubleComplex* __restrict__ d_C,
    const int* __restrict__ d_ad_offsets, // [norbs2+1]
    const int* __restrict__ d_ad_coff,    // [total_ex]
    const int* __restrict__ d_ad_boff,    // [total_ex]
    const int* __restrict__ d_ad_sign,    // [total_ex]
    const int* __restrict__ d_bd_idx2,    // [beta_states*nbdexc]
    const int* __restrict__ d_bd_orbkl,   // [beta_states*nbdexc]
    const int* __restrict__ d_bd_parity,  // [beta_states*nbdexc]
    const cuDoubleComplex* __restrict__ d_h2e, // [norbs2*norbs2]
    int alpha_states,
    int beta_states,
    int nbdexc,
    int norbs)
{
    // Warp-centric tile:
    //   x = 32 lanes (one warp) -> alpha excitations within a tile
    //   y = BETA_TILE warps -> multiple s2 handled per block
    constexpr int SIG_TILE  = 32;
    constexpr int BETA_TILE = 8;

    const int norbs2 = norbs * norbs;

    const int orbid = blockIdx.x;
    if (orbid >= norbs2) return;

    const int beta_tile_base = blockIdx.y * BETA_TILE;
    const int local_beta     = threadIdx.y;           // 0..BETA_TILE-1
    const int s2             = beta_tile_base + local_beta;

    // CSR range for this orbid
    const int ad_begin = d_ad_offsets[orbid];
    const int ad_end   = d_ad_offsets[orbid + 1];
    const int nsig     = ad_end - ad_begin;
    if (nsig == 0) return;

    // Pointer to h2e row for this orbid
    const cuDoubleComplex* __restrict__ h2e_block =
        d_h2e + (size_t)orbid * norbs2;

    // ---------------------------
    // Shared memory layout:
    //   int sh_coff[SIG_TILE]
    //   int sh_boff[SIG_TILE]
    //   int sh_sign[SIG_TILE]
    //   int sh_idx2[BETA_TILE * nbdexc]
    //   cuDoubleComplex sh_ttt[BETA_TILE * nbdexc]   (precomputed parity*h2e)
    // ---------------------------
    extern __shared__ unsigned char smem[];

    int* sh_int = reinterpret_cast<int*>(smem);
    int* sh_coff = sh_int;
    int* sh_boff = sh_coff + SIG_TILE;
    int* sh_sign = sh_boff + SIG_TILE;
    int* sh_idx2 = sh_sign + SIG_TILE; // length = BETA_TILE*nbdexc

    // Align to 16 bytes for cuDoubleComplex
    size_t int_bytes = (size_t)(3 * SIG_TILE + BETA_TILE * nbdexc) * sizeof(int);
    size_t int_bytes_aligned = (int_bytes + 15) & ~((size_t)15);

    cuDoubleComplex* sh_ttt =
        reinterpret_cast<cuDoubleComplex*>(smem + int_bytes_aligned);

    // -------------------------------------------------------
    // Stage 0: load bdexc row (idx2) and precompute weights ttt
    // This is done ONCE per block and reused for all sig tiles.
    // Each warp (fixed local_beta) loads its own s2 row.
    // -------------------------------------------------------
    if (s2 < beta_states) {
        for (int jj = threadIdx.x; jj < nbdexc; jj += SIG_TILE) {
            const int flat = s2 * nbdexc + jj;

            const int idx2   = d_bd_idx2[flat];
            const int orbkl  = d_bd_orbkl[flat];
            const int parity = d_bd_parity[flat];

            const int dst = local_beta * nbdexc + jj;
            sh_idx2[dst] = idx2;

            cuDoubleComplex t = h2e_block[orbkl];
            if (parity == -1) { t.x = -t.x; t.y = -t.y; }
            sh_ttt[dst] = t;
        }
    } else {
        // out-of-range s2 lanes: fill dummy
        for (int jj = threadIdx.x; jj < nbdexc; jj += SIG_TILE) {
            const int dst = local_beta * nbdexc + jj;
            sh_idx2[dst] = 0;
            sh_ttt[dst]  = make_cuDoubleComplex(0.0, 0.0);
        }
    }
    __syncthreads();

    // Thread identifiers
    const int local_sig  = threadIdx.x; // 0..31
    const int flat_tid   = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;

    // -------------------------------------------------------
    // Loop over tiles of alpha excitations
    // -------------------------------------------------------
    for (int sig_base = 0; sig_base < nsig; sig_base += SIG_TILE) {

        // Load one alpha tile into shared (coff, boff, sign)
        for (int t = flat_tid; t < SIG_TILE; t += block_size) {
            int g = sig_base + t;
            if (g < nsig) {
                int idx = ad_begin + g;
                sh_coff[t] = d_ad_coff[idx];
                sh_boff[t] = d_ad_boff[idx];
                sh_sign[t] = d_ad_sign[idx];
            }
        }
        __syncthreads();

        const int global_sig = sig_base + local_sig;

        if (s2 < beta_states && global_sig < nsig) {

            const int row_in  = sh_coff[local_sig]; // alpha_to
            const int row_out = sh_boff[local_sig]; // alpha_from
            const int sign    = sh_sign[local_sig]; // ±1

            if ((unsigned)row_in < (unsigned)alpha_states &&
                (unsigned)row_out < (unsigned)alpha_states) {

                const cuDoubleComplex* __restrict__ C_row =
                    d_C + (size_t)row_in * beta_states;

                cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);

                const int base = local_beta * nbdexc;

                // Inner loop: nbdexc dot-product against sparse gather of C_row
                for (int jj = 0; jj < nbdexc; ++jj) {
                    const int idx2 = sh_idx2[base + jj];
                    if ((unsigned)idx2 >= (unsigned)beta_states) continue;

                    const cuDoubleComplex ttt = sh_ttt[base + jj];
                    cuDoubleComplex cval = C_row[idx2];
                    if (sign == -1) { cval.x = -cval.x; cval.y = -cval.y; }

                    // acc += ttt * cval
                    const cuDoubleComplex prod = cuCmul(ttt, cval);
                    acc.x += prod.x;
                    acc.y += prod.y;
                }

                // Scatter-add into output
                const size_t out_idx = (size_t)row_out * beta_states + (size_t)s2;
                atomicAdd(&d_out[out_idx].x, acc.x);
                atomicAdd(&d_out[out_idx].y, acc.y);
            }
        }

        __syncthreads();
    }
}

extern "C" void lm_apply_array12_diff_spin_wrapper(
    cuDoubleComplex* d_out,
    const cuDoubleComplex* d_C,
    const int* d_adexc,
    const int* d_bdexc,
    const cuDoubleComplex* d_h2e,
    int alpha_states,
    int beta_states,
    int nadexc,
    int nbdexc,
    int norbs)
{
    const int norbs2      = norbs * norbs;
    const int nadexc_tot  = alpha_states * nadexc;
    const int betaexc_tot = beta_states * nbdexc;

    // -----------------------------
    // Alpha CSR by orbid (same as v2)
    // -----------------------------
    thrust::device_vector<int> d_counts(norbs2, 0);
    {
        int threads = 256;
        int blocks  = (nadexc_tot + threads - 1) / threads;
        count_alpha_excitations_per_orbid_kernel<<<blocks, threads>>>(
            d_adexc, alpha_states, nadexc, norbs2,
            thrust::raw_pointer_cast(d_counts.data()));
        cudaDeviceSynchronize();
    }

    thrust::device_vector<int> d_ad_offsets(norbs2 + 1);
    thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_ad_offsets.begin());
    int total_ex = thrust::reduce(d_counts.begin(), d_counts.end(), 0, thrust::plus<int>());
    cudaMemcpy(thrust::raw_pointer_cast(d_ad_offsets.data()) + norbs2,
               &total_ex, sizeof(int), cudaMemcpyHostToDevice);

    thrust::device_vector<int> d_ad_coff(total_ex);
    thrust::device_vector<int> d_ad_boff(total_ex);
    thrust::device_vector<int> d_ad_sign(total_ex);

    thrust::device_vector<int> d_cursors(norbs2);
    thrust::copy(d_ad_offsets.begin(), d_ad_offsets.begin() + norbs2, d_cursors.begin());

    {
        int threads = 256;
        int blocks  = (nadexc_tot + threads - 1) / threads;
        fill_alpha_csr_from_adexc_kernel<<<blocks, threads>>>(
            d_adexc, alpha_states, nadexc, norbs2,
            thrust::raw_pointer_cast(d_ad_offsets.data()),
            thrust::raw_pointer_cast(d_cursors.data()),
            thrust::raw_pointer_cast(d_ad_coff.data()),
            thrust::raw_pointer_cast(d_ad_boff.data()),
            thrust::raw_pointer_cast(d_ad_sign.data()));
        cudaDeviceSynchronize();
    }

    // -----------------------------
    // Beta SoA (same as v2)
    // -----------------------------
    thrust::device_vector<int> d_bd_idx2(betaexc_tot);
    thrust::device_vector<int> d_bd_orbkl(betaexc_tot);
    thrust::device_vector<int> d_bd_parity(betaexc_tot);

    {
        int threads = 256;
        int blocks  = (betaexc_tot + threads - 1) / threads;
        split_bdexc_kernel<<<blocks, threads>>>(
            d_bdexc, beta_states, nbdexc,
            thrust::raw_pointer_cast(d_bd_idx2.data()),
            thrust::raw_pointer_cast(d_bd_orbkl.data()),
            thrust::raw_pointer_cast(d_bd_parity.data()));
        cudaDeviceSynchronize();
    }

    // -----------------------------
    // Launch kernel
    // -----------------------------
    constexpr int SIG_TILE  = 32;
    constexpr int BETA_TILE = 8;

    dim3 block(SIG_TILE, BETA_TILE);
    dim3 grid(norbs2, (beta_states + BETA_TILE - 1) / BETA_TILE);

    // shared bytes:
    //   ints: 3*SIG_TILE + BETA_TILE*nbdexc
    //   complex: BETA_TILE*nbdexc
    size_t int_bytes = (size_t)(3 * SIG_TILE + BETA_TILE * nbdexc) * sizeof(int);
    size_t int_bytes_aligned = (int_bytes + 15) & ~((size_t)15);
    size_t complex_bytes = (size_t)(BETA_TILE * nbdexc) * sizeof(cuDoubleComplex);
    size_t shmem_bytes = int_bytes_aligned + complex_bytes;

    lm_apply_array12_diff_spin_kernel<<<grid, block, shmem_bytes>>>(
        d_out,
        d_C,
        thrust::raw_pointer_cast(d_ad_offsets.data()),
        thrust::raw_pointer_cast(d_ad_coff.data()),
        thrust::raw_pointer_cast(d_ad_boff.data()),
        thrust::raw_pointer_cast(d_ad_sign.data()),
        thrust::raw_pointer_cast(d_bd_idx2.data()),
        thrust::raw_pointer_cast(d_bd_orbkl.data()),
        thrust::raw_pointer_cast(d_bd_parity.data()),
        d_h2e,
        alpha_states,
        beta_states,
        nbdexc,
        norbs);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "lm_apply_array12_diff_spin_wrapper failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("lm_apply_array12_diff_spin_wrapper failed");
    }
}

// ==============================================
// Diff Spin kernel implementation (Real version)
// ==============================================

// ---------------------------------------------------------
// Diff spin real version: cache bdexc row + weights in shared once per block
// ---------------------------------------------------------
__global__ void lm_apply_array12_diff_spin_kernel_real(
    double* __restrict__ d_out,
    const double* __restrict__ d_C,
    const int* __restrict__ d_ad_offsets, // [norbs2+1]
    const int* __restrict__ d_ad_coff,    // [total_ex]
    const int* __restrict__ d_ad_boff,    // [total_ex]
    const int* __restrict__ d_ad_sign,    // [total_ex]
    const int* __restrict__ d_bd_idx2,    // [beta_states*nbdexc]
    const int* __restrict__ d_bd_orbkl,   // [beta_states*nbdexc]
    const int* __restrict__ d_bd_parity,  // [beta_states*nbdexc]
    const double* __restrict__ d_h2e,     // [norbs2*norbs2]
    int alpha_states,
    int beta_states,
    int nbdexc,
    int norbs)
{
    // Warp-centric tile:
    //   x = 32 lanes (one warp) -> alpha excitations within a tile
    //   y = BETA_TILE warps -> multiple s2 handled per block
    constexpr int SIG_TILE  = 32;
    constexpr int BETA_TILE = 8;

    const int norbs2 = norbs * norbs;

    const int orbid = blockIdx.x;
    if (orbid >= norbs2) return;

    const int beta_tile_base = blockIdx.y * BETA_TILE;
    const int local_beta     = threadIdx.y;           // 0..BETA_TILE-1
    const int s2             = beta_tile_base + local_beta;

    // CSR range for this orbid
    const int ad_begin = d_ad_offsets[orbid];
    const int ad_end   = d_ad_offsets[orbid + 1];
    const int nsig     = ad_end - ad_begin;
    if (nsig == 0) return;

    // Pointer to h2e row for this orbid
    const double* __restrict__ h2e_block = d_h2e + (size_t)orbid * norbs2;

    // ---------------------------
    // Shared memory layout:
    //   int sh_coff[SIG_TILE]
    //   int sh_boff[SIG_TILE]
    //   int sh_sign[SIG_TILE]
    //   int sh_idx2[BETA_TILE * nbdexc]
    //   double sh_ttt[BETA_TILE * nbdexc]   (precomputed parity*h2e)
    // ---------------------------
    extern __shared__ unsigned char smem[];

    int* sh_int = reinterpret_cast<int*>(smem);
    int* sh_coff = sh_int;
    int* sh_boff = sh_coff + SIG_TILE;
    int* sh_sign = sh_boff + SIG_TILE;
    int* sh_idx2 = sh_sign + SIG_TILE; // length = BETA_TILE*nbdexc

    // Align to 8 bytes for double
    size_t int_bytes = (size_t)(3 * SIG_TILE + BETA_TILE * nbdexc) * sizeof(int);
    size_t int_bytes_aligned = (int_bytes + 7) & ~((size_t)7);

    double* sh_ttt = reinterpret_cast<double*>(smem + int_bytes_aligned);

    // -------------------------------------------------------
    // Stage 0: load bdexc row (idx2) and precompute weights ttt
    // This is done ONCE per block and reused for all sig tiles.
    // Each warp (fixed local_beta) loads its own s2 row.
    // -------------------------------------------------------
    if (s2 < beta_states) {
        for (int jj = threadIdx.x; jj < nbdexc; jj += SIG_TILE) {
            const int flat = s2 * nbdexc + jj;

            const int idx2   = d_bd_idx2[flat];
            const int orbkl  = d_bd_orbkl[flat];
            const int parity = d_bd_parity[flat];

            const int dst = local_beta * nbdexc + jj;
            sh_idx2[dst] = idx2;

            double t = h2e_block[orbkl];
            if (parity == -1) { t = -t; }
            sh_ttt[dst] = t;
        }
    } else {
        // out-of-range s2 lanes: fill dummy
        for (int jj = threadIdx.x; jj < nbdexc; jj += SIG_TILE) {
            const int dst = local_beta * nbdexc + jj;
            sh_idx2[dst] = 0;
            sh_ttt[dst]  = 0.0;
        }
    }
    __syncthreads();

    // Thread identifiers
    const int local_sig  = threadIdx.x; // 0..31
    const int flat_tid   = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;

    // -------------------------------------------------------
    // Loop over tiles of alpha excitations
    // -------------------------------------------------------
    for (int sig_base = 0; sig_base < nsig; sig_base += SIG_TILE) {

        // Load one alpha tile into shared (coff, boff, sign)
        for (int t = flat_tid; t < SIG_TILE; t += block_size) {
            int g = sig_base + t;
            if (g < nsig) {
                int idx = ad_begin + g;
                sh_coff[t] = d_ad_coff[idx];
                sh_boff[t] = d_ad_boff[idx];
                sh_sign[t] = d_ad_sign[idx];
            }
        }
        __syncthreads();

        const int global_sig = sig_base + local_sig;

        if (s2 < beta_states && global_sig < nsig) {

            const int row_in  = sh_coff[local_sig]; // alpha_to
            const int row_out = sh_boff[local_sig]; // alpha_from
            const int sign    = sh_sign[local_sig]; // ±1

            if ((unsigned)row_in < (unsigned)alpha_states &&
                (unsigned)row_out < (unsigned)alpha_states) {

                const double* __restrict__ C_row = d_C + (size_t)row_in * beta_states;

                double acc = 0.0;

                const int base = local_beta * nbdexc;

                // Inner loop: nbdexc dot-product against sparse gather of C_row
                for (int jj = 0; jj < nbdexc; ++jj) {
                    const int idx2 = sh_idx2[base + jj];
                    if ((unsigned)idx2 >= (unsigned)beta_states) continue;

                    const double ttt = sh_ttt[base + jj];
                    double cval = C_row[idx2];
                    if (sign == -1) { cval = -cval; }

                    acc += ttt * cval;
                }

                // Scatter-add into output
                const size_t out_idx = (size_t)row_out * beta_states + (size_t)s2;
                atomicAdd(&d_out[out_idx], acc);
            }
        }

        __syncthreads();
    }
}

extern "C" void lm_apply_array12_diff_spin_wrapper_real(
    double* d_out,
    const double* d_C,
    const int* d_adexc,
    const int* d_bdexc,
    const double* d_h2e,
    int alpha_states,
    int beta_states,
    int nadexc,
    int nbdexc,
    int norbs)
{
    const int norbs2      = norbs * norbs;
    const int nadexc_tot  = alpha_states * nadexc;
    const int betaexc_tot = beta_states * nbdexc;

    // -----------------------------
    // Alpha CSR by orbid
    // -----------------------------
    thrust::device_vector<int> d_counts(norbs2, 0);
    {
        int threads = 256;
        int blocks  = (nadexc_tot + threads - 1) / threads;
        count_alpha_excitations_per_orbid_kernel<<<blocks, threads>>>(
            d_adexc, alpha_states, nadexc, norbs2,
            thrust::raw_pointer_cast(d_counts.data()));
        cudaDeviceSynchronize();
    }

    thrust::device_vector<int> d_ad_offsets(norbs2 + 1);
    thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_ad_offsets.begin());
    int total_ex = thrust::reduce(d_counts.begin(), d_counts.end(), 0, thrust::plus<int>());
    cudaMemcpy(thrust::raw_pointer_cast(d_ad_offsets.data()) + norbs2,
               &total_ex, sizeof(int), cudaMemcpyHostToDevice);

    thrust::device_vector<int> d_ad_coff(total_ex);
    thrust::device_vector<int> d_ad_boff(total_ex);
    thrust::device_vector<int> d_ad_sign(total_ex);

    thrust::device_vector<int> d_cursors(norbs2);
    thrust::copy(d_ad_offsets.begin(), d_ad_offsets.begin() + norbs2, d_cursors.begin());

    {
        int threads = 256;
        int blocks  = (nadexc_tot + threads - 1) / threads;
        fill_alpha_csr_from_adexc_kernel<<<blocks, threads>>>(
            d_adexc, alpha_states, nadexc, norbs2,
            thrust::raw_pointer_cast(d_ad_offsets.data()),
            thrust::raw_pointer_cast(d_cursors.data()),
            thrust::raw_pointer_cast(d_ad_coff.data()),
            thrust::raw_pointer_cast(d_ad_boff.data()),
            thrust::raw_pointer_cast(d_ad_sign.data()));
        cudaDeviceSynchronize();
    }

    // -----------------------------
    // Beta SoA
    // -----------------------------
    thrust::device_vector<int> d_bd_idx2(betaexc_tot);
    thrust::device_vector<int> d_bd_orbkl(betaexc_tot);
    thrust::device_vector<int> d_bd_parity(betaexc_tot);

    {
        int threads = 256;
        int blocks  = (betaexc_tot + threads - 1) / threads;
        split_bdexc_kernel<<<blocks, threads>>>(
            d_bdexc, beta_states, nbdexc,
            thrust::raw_pointer_cast(d_bd_idx2.data()),
            thrust::raw_pointer_cast(d_bd_orbkl.data()),
            thrust::raw_pointer_cast(d_bd_parity.data()));
        cudaDeviceSynchronize();
    }

    // -----------------------------
    // Launch kernel
    // -----------------------------
    constexpr int SIG_TILE  = 32;
    constexpr int BETA_TILE = 8;

    dim3 block(SIG_TILE, BETA_TILE);
    dim3 grid(norbs2, (beta_states + BETA_TILE - 1) / BETA_TILE);

    // shared bytes:
    //   ints: 3*SIG_TILE + BETA_TILE*nbdexc
    //   double: BETA_TILE*nbdexc
    size_t int_bytes = (size_t)(3 * SIG_TILE + BETA_TILE * nbdexc) * sizeof(int);
    size_t int_bytes_aligned = (int_bytes + 7) & ~((size_t)7);
    size_t double_bytes = (size_t)(BETA_TILE * nbdexc) * sizeof(double);
    size_t shmem_bytes = int_bytes_aligned + double_bytes;

    lm_apply_array12_diff_spin_kernel_real<<<grid, block, shmem_bytes>>>(
        d_out,
        d_C,
        thrust::raw_pointer_cast(d_ad_offsets.data()),
        thrust::raw_pointer_cast(d_ad_coff.data()),
        thrust::raw_pointer_cast(d_ad_boff.data()),
        thrust::raw_pointer_cast(d_ad_sign.data()),
        thrust::raw_pointer_cast(d_bd_idx2.data()),
        thrust::raw_pointer_cast(d_bd_orbkl.data()),
        thrust::raw_pointer_cast(d_bd_parity.data()),
        d_h2e,
        alpha_states,
        beta_states,
        nbdexc,
        norbs);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "lm_apply_array12_diff_spin_wrapper_real failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("lm_apply_array12_diff_spin_wrapper_real failed");
    }
}

// ==============================================
// Diff Spin kernel implementation (Mixed: real h, complex state)
// ==============================================

__global__ void lm_apply_array12_diff_spin_kernel_mixed(
    cuDoubleComplex* __restrict__ d_out,
    const cuDoubleComplex* __restrict__ d_C,
    const int* __restrict__ d_ad_offsets, // [norbs2+1]
    const int* __restrict__ d_ad_coff,    // [total_ex]
    const int* __restrict__ d_ad_boff,    // [total_ex]
    const int* __restrict__ d_ad_sign,    // [total_ex]
    const int* __restrict__ d_bd_idx2,    // [beta_states*nbdexc]
    const int* __restrict__ d_bd_orbkl,   // [beta_states*nbdexc]
    const int* __restrict__ d_bd_parity,  // [beta_states*nbdexc]
    const double* __restrict__ d_h2e,     // [norbs2*norbs2] - REAL
    int alpha_states,
    int beta_states,
    int nbdexc,
    int norbs)
{
    constexpr int SIG_TILE  = 32;
    constexpr int BETA_TILE = 8;

    const int norbs2 = norbs * norbs;

    const int orbid = blockIdx.x;
    if (orbid >= norbs2) return;

    const int beta_tile_base = blockIdx.y * BETA_TILE;
    const int local_beta     = threadIdx.y;
    const int s2             = beta_tile_base + local_beta;

    // CSR range for this orbid
    const int ad_begin = d_ad_offsets[orbid];
    const int ad_end   = d_ad_offsets[orbid + 1];
    const int nsig     = ad_end - ad_begin;
    if (nsig == 0) return;

    // Pointer to h2e row for this orbid (REAL)
    const double* __restrict__ h2e_block = d_h2e + (size_t)orbid * norbs2;

    // Shared memory layout:
    //   int sh_coff[SIG_TILE]
    //   int sh_boff[SIG_TILE]
    //   int sh_sign[SIG_TILE]
    //   int sh_idx2[BETA_TILE * nbdexc]
    //   cuDoubleComplex sh_ttt[BETA_TILE * nbdexc]
    extern __shared__ unsigned char smem[];

    int* sh_int = reinterpret_cast<int*>(smem);
    int* sh_coff = sh_int;
    int* sh_boff = sh_coff + SIG_TILE;
    int* sh_sign = sh_boff + SIG_TILE;
    int* sh_idx2 = sh_sign + SIG_TILE;

    size_t int_bytes = (size_t)(3 * SIG_TILE + BETA_TILE * nbdexc) * sizeof(int);
    size_t int_bytes_aligned = (int_bytes + 15) & ~((size_t)15);

    cuDoubleComplex* sh_ttt = reinterpret_cast<cuDoubleComplex*>(smem + int_bytes_aligned);

    // Load bdexc row and precompute weights (convert real h2e to complex)
    if (s2 < beta_states) {
        for (int jj = threadIdx.x; jj < nbdexc; jj += SIG_TILE) {
            const int flat = s2 * nbdexc + jj;

            const int idx2   = d_bd_idx2[flat];
            const int orbkl  = d_bd_orbkl[flat];
            const int parity = d_bd_parity[flat];

            const int dst = local_beta * nbdexc + jj;
            sh_idx2[dst] = idx2;

            double h_val = h2e_block[orbkl];  // Read real value
            cuDoubleComplex t = make_cuDoubleComplex(h_val, 0.0);  // Convert to complex
            if (parity == -1) { t.x = -t.x; t.y = -t.y; }
            sh_ttt[dst] = t;
        }
    } else {
        for (int jj = threadIdx.x; jj < nbdexc; jj += SIG_TILE) {
            const int dst = local_beta * nbdexc + jj;
            sh_idx2[dst] = 0;
            sh_ttt[dst]  = make_cuDoubleComplex(0.0, 0.0);
        }
    }
    __syncthreads();

    const int local_sig  = threadIdx.x;
    const int flat_tid   = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;

    // Loop over tiles of alpha excitations
    for (int sig_base = 0; sig_base < nsig; sig_base += SIG_TILE) {

        // Load one alpha tile into shared
        for (int t = flat_tid; t < SIG_TILE; t += block_size) {
            int g = sig_base + t;
            if (g < nsig) {
                int idx = ad_begin + g;
                sh_coff[t] = d_ad_coff[idx];
                sh_boff[t] = d_ad_boff[idx];
                sh_sign[t] = d_ad_sign[idx];
            }
        }
        __syncthreads();

        const int global_sig = sig_base + local_sig;

        if (s2 < beta_states && global_sig < nsig) {

            const int row_in  = sh_coff[local_sig];
            const int row_out = sh_boff[local_sig];
            const int sign    = sh_sign[local_sig];

            if ((unsigned)row_in < (unsigned)alpha_states &&
                (unsigned)row_out < (unsigned)alpha_states) {

                const cuDoubleComplex* __restrict__ C_row =
                    d_C + (size_t)row_in * beta_states;

                cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);

                const int base = local_beta * nbdexc;

                // Inner loop: nbdexc dot-product
                for (int jj = 0; jj < nbdexc; ++jj) {
                    const int idx2 = sh_idx2[base + jj];
                    if ((unsigned)idx2 >= (unsigned)beta_states) continue;

                    const cuDoubleComplex ttt = sh_ttt[base + jj];
                    cuDoubleComplex cval = C_row[idx2];
                    if (sign == -1) { cval.x = -cval.x; cval.y = -cval.y; }

                    // acc += ttt * cval
                    const cuDoubleComplex prod = cuCmul(ttt, cval);
                    acc.x += prod.x;
                    acc.y += prod.y;
                }

                // Scatter-add into output
                const size_t out_idx = (size_t)row_out * beta_states + (size_t)s2;
                atomicAdd(&d_out[out_idx].x, acc.x);
                atomicAdd(&d_out[out_idx].y, acc.y);
            }
        }

        __syncthreads();
    }
}

extern "C" void lm_apply_array12_diff_spin_wrapper_mixed(
    cuDoubleComplex* d_out,
    const cuDoubleComplex* d_C,
    const int* d_adexc,
    const int* d_bdexc,
    const double* d_h2e,
    int alpha_states,
    int beta_states,
    int nadexc,
    int nbdexc,
    int norbs)
{
    const int norbs2      = norbs * norbs;
    const int nadexc_tot  = alpha_states * nadexc;
    const int betaexc_tot = beta_states * nbdexc;

    // Alpha CSR by orbid
    thrust::device_vector<int> d_counts(norbs2, 0);
    {
        int threads = 256;
        int blocks  = (nadexc_tot + threads - 1) / threads;
        count_alpha_excitations_per_orbid_kernel<<<blocks, threads>>>(
            d_adexc, alpha_states, nadexc, norbs2,
            thrust::raw_pointer_cast(d_counts.data()));
        cudaDeviceSynchronize();
    }

    thrust::device_vector<int> d_ad_offsets(norbs2 + 1);
    thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_ad_offsets.begin());
    int total_ex = thrust::reduce(d_counts.begin(), d_counts.end(), 0, thrust::plus<int>());
    cudaMemcpy(thrust::raw_pointer_cast(d_ad_offsets.data()) + norbs2,
               &total_ex, sizeof(int), cudaMemcpyHostToDevice);

    thrust::device_vector<int> d_ad_coff(total_ex);
    thrust::device_vector<int> d_ad_boff(total_ex);
    thrust::device_vector<int> d_ad_sign(total_ex);

    thrust::device_vector<int> d_cursors(norbs2);
    thrust::copy(d_ad_offsets.begin(), d_ad_offsets.begin() + norbs2, d_cursors.begin());

    {
        int threads = 256;
        int blocks  = (nadexc_tot + threads - 1) / threads;
        fill_alpha_csr_from_adexc_kernel<<<blocks, threads>>>(
            d_adexc, alpha_states, nadexc, norbs2,
            thrust::raw_pointer_cast(d_ad_offsets.data()),
            thrust::raw_pointer_cast(d_cursors.data()),
            thrust::raw_pointer_cast(d_ad_coff.data()),
            thrust::raw_pointer_cast(d_ad_boff.data()),
            thrust::raw_pointer_cast(d_ad_sign.data()));
        cudaDeviceSynchronize();
    }

    // Beta SoA
    thrust::device_vector<int> d_bd_idx2(betaexc_tot);
    thrust::device_vector<int> d_bd_orbkl(betaexc_tot);
    thrust::device_vector<int> d_bd_parity(betaexc_tot);

    {
        int threads = 256;
        int blocks  = (betaexc_tot + threads - 1) / threads;
        split_bdexc_kernel<<<blocks, threads>>>(
            d_bdexc, beta_states, nbdexc,
            thrust::raw_pointer_cast(d_bd_idx2.data()),
            thrust::raw_pointer_cast(d_bd_orbkl.data()),
            thrust::raw_pointer_cast(d_bd_parity.data()));
        cudaDeviceSynchronize();
    }

    // Launch kernel
    constexpr int SIG_TILE  = 32;
    constexpr int BETA_TILE = 8;

    dim3 block(SIG_TILE, BETA_TILE);
    dim3 grid(norbs2, (beta_states + BETA_TILE - 1) / BETA_TILE);

    // shared bytes:
    //   ints: 3*SIG_TILE + BETA_TILE*nbdexc
    //   complex: BETA_TILE*nbdexc
    size_t int_bytes = (size_t)(3 * SIG_TILE + BETA_TILE * nbdexc) * sizeof(int);
    size_t int_bytes_aligned = (int_bytes + 15) & ~((size_t)15);
    size_t complex_bytes = (size_t)(BETA_TILE * nbdexc) * sizeof(cuDoubleComplex);
    size_t shmem_bytes = int_bytes_aligned + complex_bytes;

    lm_apply_array12_diff_spin_kernel_mixed<<<grid, block, shmem_bytes>>>(
        d_out,
        d_C,
        thrust::raw_pointer_cast(d_ad_offsets.data()),
        thrust::raw_pointer_cast(d_ad_coff.data()),
        thrust::raw_pointer_cast(d_ad_boff.data()),
        thrust::raw_pointer_cast(d_ad_sign.data()),
        thrust::raw_pointer_cast(d_bd_idx2.data()),
        thrust::raw_pointer_cast(d_bd_orbkl.data()),
        thrust::raw_pointer_cast(d_bd_parity.data()),
        d_h2e,
        alpha_states,
        beta_states,
        nbdexc,
        norbs);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "lm_apply_array12_diff_spin_wrapper_mixed failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("lm_apply_array12_diff_spin_wrapper_mixed failed");
    }
}

// ==============================================
// Diff Spin v2 tiled implementation
// ==============================================

namespace {

constexpr int DIFF_V2_TILE_B = 16;   // threadIdx.x: beta-output columns
constexpr int DIFF_V2_TILE_A = 8;    // threadIdx.y: alpha-output rows
constexpr int DIFF_V2_CHUNK_A = 8;   // incoming alpha edges staged per row
constexpr int DIFF_V2_CHUNK_B = 8;   // incoming beta edges staged per column

int diff_spin_v2_tiled_grid_dim(long long tiles)
{
    const long long capped = std::min<long long>(tiles, 65535);
    return static_cast<int>(std::max<long long>(capped, 1));
}

size_t diff_spin_v2_tiled_shared_bytes()
{
    const size_t alpha_slots =
        static_cast<size_t>(DIFF_V2_TILE_A) * DIFF_V2_CHUNK_A;
    const size_t beta_slots =
        static_cast<size_t>(DIFF_V2_TILE_B) * DIFF_V2_CHUNK_B;
    return 3 * (alpha_slots + beta_slots) * sizeof(int);
}

} // namespace

__global__ void lm_apply_array12_diff_spin_v2_tiled_kernel_real(
    double* __restrict__ d_out,
    const double* __restrict__ d_C,
    const long long* __restrict__ d_alpha_offsets,
    const int* __restrict__ d_alpha_sources,
    const int* __restrict__ d_alpha_pairs,
    const int* __restrict__ d_alpha_parities,
    const long long* __restrict__ d_beta_offsets,
    const int* __restrict__ d_beta_sources,
    const int* __restrict__ d_beta_pairs,
    const int* __restrict__ d_beta_parities,
    const double* __restrict__ d_h2e,
    long long alpha_states,
    long long beta_states,
    int norbs)
{
    __shared__ long long sh_a_begin[DIFF_V2_TILE_A];
    __shared__ long long sh_a_count[DIFF_V2_TILE_A];
    __shared__ long long sh_b_begin[DIFF_V2_TILE_B];
    __shared__ long long sh_b_count[DIFF_V2_TILE_B];
    __shared__ long long sh_max_a;
    __shared__ long long sh_max_b;

    extern __shared__ int sh_edges[];
    int* sh_a_src = sh_edges;
    int* sh_a_pair = sh_a_src + DIFF_V2_TILE_A * DIFF_V2_CHUNK_A;
    int* sh_a_parity = sh_a_pair + DIFF_V2_TILE_A * DIFF_V2_CHUNK_A;
    int* sh_b_src = sh_a_parity + DIFF_V2_TILE_A * DIFF_V2_CHUNK_A;
    int* sh_b_pair = sh_b_src + DIFF_V2_TILE_B * DIFF_V2_CHUNK_B;
    int* sh_b_parity = sh_b_pair + DIFF_V2_TILE_B * DIFF_V2_CHUNK_B;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;
    const long long norbs2 = static_cast<long long>(norbs) * norbs;

    // One block owns a 2D output tile.  If the CI matrix has more tiles than
    // the grid can expose directly, the block grid-strides over output tiles.
    for (long long a_base = static_cast<long long>(blockIdx.y) * DIFF_V2_TILE_A;
         a_base < alpha_states;
         a_base += static_cast<long long>(gridDim.y) * DIFF_V2_TILE_A) {
        for (long long b_base = static_cast<long long>(blockIdx.x) * DIFF_V2_TILE_B;
             b_base < beta_states;
             b_base += static_cast<long long>(gridDim.x) * DIFF_V2_TILE_B) {

            if (tid < DIFF_V2_TILE_A) {
                const long long a_out = a_base + tid;
                if (a_out < alpha_states) {
                    sh_a_begin[tid] = d_alpha_offsets[a_out];
                    sh_a_count[tid] = d_alpha_offsets[a_out + 1] - sh_a_begin[tid];
                } else {
                    sh_a_begin[tid] = 0;
                    sh_a_count[tid] = 0;
                }
            }
            if (tid < DIFF_V2_TILE_B) {
                const long long b_out = b_base + tid;
                if (b_out < beta_states) {
                    sh_b_begin[tid] = d_beta_offsets[b_out];
                    sh_b_count[tid] = d_beta_offsets[b_out + 1] - sh_b_begin[tid];
                } else {
                    sh_b_begin[tid] = 0;
                    sh_b_count[tid] = 0;
                }
            }
            __syncthreads();

            if (tid == 0) {
                long long max_a = 0;
                long long max_b = 0;
                for (int i = 0; i < DIFF_V2_TILE_A; ++i) {
                    max_a = max_a < sh_a_count[i] ? sh_a_count[i] : max_a;
                }
                for (int i = 0; i < DIFF_V2_TILE_B; ++i) {
                    max_b = max_b < sh_b_count[i] ? sh_b_count[i] : max_b;
                }
                sh_max_a = max_a;
                sh_max_b = max_b;
            }
            __syncthreads();

            const long long a_out = a_base + threadIdx.y;
            const long long b_out = b_base + threadIdx.x;
            const bool active = a_out < alpha_states && b_out < beta_states;
            double acc = 0.0;

            // Irregular incoming-list lengths make the tile ragged.  We stage
            // fixed-size chunks for all rows/columns in the output tile; each
            // thread then consumes only the chunk entries for its owned
            // (a_out, b_out) scalar.
            for (long long a_chunk = 0; a_chunk < sh_max_a; a_chunk += DIFF_V2_CHUNK_A) {
                for (int slot = tid;
                     slot < DIFF_V2_TILE_A * DIFF_V2_CHUNK_A;
                     slot += nthreads) {
                    const int local_a = slot / DIFF_V2_CHUNK_A;
                    const int k = slot - local_a * DIFF_V2_CHUNK_A;
                    const long long edge = sh_a_begin[local_a] + a_chunk + k;
                    if (a_chunk + k < sh_a_count[local_a]) {
                        sh_a_src[slot] = d_alpha_sources[edge];
                        sh_a_pair[slot] = d_alpha_pairs[edge];
                        sh_a_parity[slot] = d_alpha_parities[edge];
                    } else {
                        sh_a_src[slot] = 0;
                        sh_a_pair[slot] = 0;
                        sh_a_parity[slot] = 0;
                    }
                }
                __syncthreads();

                for (long long b_chunk = 0; b_chunk < sh_max_b; b_chunk += DIFF_V2_CHUNK_B) {
                    for (int slot = tid;
                         slot < DIFF_V2_TILE_B * DIFF_V2_CHUNK_B;
                         slot += nthreads) {
                        const int local_b = slot / DIFF_V2_CHUNK_B;
                        const int k = slot - local_b * DIFF_V2_CHUNK_B;
                        const long long edge = sh_b_begin[local_b] + b_chunk + k;
                        if (b_chunk + k < sh_b_count[local_b]) {
                            sh_b_src[slot] = d_beta_sources[edge];
                            sh_b_pair[slot] = d_beta_pairs[edge];
                            sh_b_parity[slot] = d_beta_parities[edge];
                        } else {
                            sh_b_src[slot] = 0;
                            sh_b_pair[slot] = 0;
                            sh_b_parity[slot] = 0;
                        }
                    }
                    __syncthreads();

                    if (active) {
                        const long long a_rem = sh_a_count[threadIdx.y] - a_chunk;
                        const long long b_rem = sh_b_count[threadIdx.x] - b_chunk;
                        const int a_limit = a_rem < DIFF_V2_CHUNK_A
                            ? static_cast<int>(a_rem) : DIFF_V2_CHUNK_A;
                        const int b_limit = b_rem < DIFF_V2_CHUNK_B
                            ? static_cast<int>(b_rem) : DIFF_V2_CHUNK_B;
                        const int a_slot_base = threadIdx.y * DIFF_V2_CHUNK_A;
                        const int b_slot_base = threadIdx.x * DIFF_V2_CHUNK_B;

                        for (int ia = 0; ia < a_limit; ++ia) {
                            const int a_slot = a_slot_base + ia;
                            const int a_src = sh_a_src[a_slot];
                            const int ij = sh_a_pair[a_slot];
                            const int pa = sh_a_parity[a_slot];

                            for (int ib = 0; ib < b_limit; ++ib) {
                                const int b_slot = b_slot_base + ib;
                                const int b_src = sh_b_src[b_slot];
                                const int kl = sh_b_pair[b_slot];
                                const int pb = sh_b_parity[b_slot];
                                const size_t h_idx =
                                    static_cast<size_t>(ij) * static_cast<size_t>(norbs2)
                                    + static_cast<size_t>(kl);
                                const size_t c_idx =
                                    static_cast<size_t>(a_src) * static_cast<size_t>(beta_states)
                                    + static_cast<size_t>(b_src);
                                acc += static_cast<double>(pa * pb) * d_h2e[h_idx] * d_C[c_idx];
                            }
                        }
                    }
                    __syncthreads();
                }
            }

            if (active) {
                const size_t out_idx =
                    static_cast<size_t>(a_out) * static_cast<size_t>(beta_states)
                    + static_cast<size_t>(b_out);
                d_out[out_idx] += acc;
            }
            __syncthreads();
        }
    }
}

__global__ void lm_apply_array12_diff_spin_v2_tiled_kernel_mixed(
    cuDoubleComplex* __restrict__ d_out,
    const cuDoubleComplex* __restrict__ d_C,
    const long long* __restrict__ d_alpha_offsets,
    const int* __restrict__ d_alpha_sources,
    const int* __restrict__ d_alpha_pairs,
    const int* __restrict__ d_alpha_parities,
    const long long* __restrict__ d_beta_offsets,
    const int* __restrict__ d_beta_sources,
    const int* __restrict__ d_beta_pairs,
    const int* __restrict__ d_beta_parities,
    const double* __restrict__ d_h2e,
    long long alpha_states,
    long long beta_states,
    int norbs)
{
    __shared__ long long sh_a_begin[DIFF_V2_TILE_A];
    __shared__ long long sh_a_count[DIFF_V2_TILE_A];
    __shared__ long long sh_b_begin[DIFF_V2_TILE_B];
    __shared__ long long sh_b_count[DIFF_V2_TILE_B];
    __shared__ long long sh_max_a;
    __shared__ long long sh_max_b;

    extern __shared__ int sh_edges[];
    int* sh_a_src = sh_edges;
    int* sh_a_pair = sh_a_src + DIFF_V2_TILE_A * DIFF_V2_CHUNK_A;
    int* sh_a_parity = sh_a_pair + DIFF_V2_TILE_A * DIFF_V2_CHUNK_A;
    int* sh_b_src = sh_a_parity + DIFF_V2_TILE_A * DIFF_V2_CHUNK_A;
    int* sh_b_pair = sh_b_src + DIFF_V2_TILE_B * DIFF_V2_CHUNK_B;
    int* sh_b_parity = sh_b_pair + DIFF_V2_TILE_B * DIFF_V2_CHUNK_B;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;
    const long long norbs2 = static_cast<long long>(norbs) * norbs;

    for (long long a_base = static_cast<long long>(blockIdx.y) * DIFF_V2_TILE_A;
         a_base < alpha_states;
         a_base += static_cast<long long>(gridDim.y) * DIFF_V2_TILE_A) {
        for (long long b_base = static_cast<long long>(blockIdx.x) * DIFF_V2_TILE_B;
             b_base < beta_states;
             b_base += static_cast<long long>(gridDim.x) * DIFF_V2_TILE_B) {

            if (tid < DIFF_V2_TILE_A) {
                const long long a_out = a_base + tid;
                if (a_out < alpha_states) {
                    sh_a_begin[tid] = d_alpha_offsets[a_out];
                    sh_a_count[tid] = d_alpha_offsets[a_out + 1] - sh_a_begin[tid];
                } else {
                    sh_a_begin[tid] = 0;
                    sh_a_count[tid] = 0;
                }
            }
            if (tid < DIFF_V2_TILE_B) {
                const long long b_out = b_base + tid;
                if (b_out < beta_states) {
                    sh_b_begin[tid] = d_beta_offsets[b_out];
                    sh_b_count[tid] = d_beta_offsets[b_out + 1] - sh_b_begin[tid];
                } else {
                    sh_b_begin[tid] = 0;
                    sh_b_count[tid] = 0;
                }
            }
            __syncthreads();

            if (tid == 0) {
                long long max_a = 0;
                long long max_b = 0;
                for (int i = 0; i < DIFF_V2_TILE_A; ++i) {
                    max_a = max_a < sh_a_count[i] ? sh_a_count[i] : max_a;
                }
                for (int i = 0; i < DIFF_V2_TILE_B; ++i) {
                    max_b = max_b < sh_b_count[i] ? sh_b_count[i] : max_b;
                }
                sh_max_a = max_a;
                sh_max_b = max_b;
            }
            __syncthreads();

            const long long a_out = a_base + threadIdx.y;
            const long long b_out = b_base + threadIdx.x;
            const bool active = a_out < alpha_states && b_out < beta_states;
            cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);

            for (long long a_chunk = 0; a_chunk < sh_max_a; a_chunk += DIFF_V2_CHUNK_A) {
                for (int slot = tid;
                     slot < DIFF_V2_TILE_A * DIFF_V2_CHUNK_A;
                     slot += nthreads) {
                    const int local_a = slot / DIFF_V2_CHUNK_A;
                    const int k = slot - local_a * DIFF_V2_CHUNK_A;
                    const long long edge = sh_a_begin[local_a] + a_chunk + k;
                    if (a_chunk + k < sh_a_count[local_a]) {
                        sh_a_src[slot] = d_alpha_sources[edge];
                        sh_a_pair[slot] = d_alpha_pairs[edge];
                        sh_a_parity[slot] = d_alpha_parities[edge];
                    } else {
                        sh_a_src[slot] = 0;
                        sh_a_pair[slot] = 0;
                        sh_a_parity[slot] = 0;
                    }
                }
                __syncthreads();

                for (long long b_chunk = 0; b_chunk < sh_max_b; b_chunk += DIFF_V2_CHUNK_B) {
                    for (int slot = tid;
                         slot < DIFF_V2_TILE_B * DIFF_V2_CHUNK_B;
                         slot += nthreads) {
                        const int local_b = slot / DIFF_V2_CHUNK_B;
                        const int k = slot - local_b * DIFF_V2_CHUNK_B;
                        const long long edge = sh_b_begin[local_b] + b_chunk + k;
                        if (b_chunk + k < sh_b_count[local_b]) {
                            sh_b_src[slot] = d_beta_sources[edge];
                            sh_b_pair[slot] = d_beta_pairs[edge];
                            sh_b_parity[slot] = d_beta_parities[edge];
                        } else {
                            sh_b_src[slot] = 0;
                            sh_b_pair[slot] = 0;
                            sh_b_parity[slot] = 0;
                        }
                    }
                    __syncthreads();

                    if (active) {
                        const long long a_rem = sh_a_count[threadIdx.y] - a_chunk;
                        const long long b_rem = sh_b_count[threadIdx.x] - b_chunk;
                        const int a_limit = a_rem < DIFF_V2_CHUNK_A
                            ? static_cast<int>(a_rem) : DIFF_V2_CHUNK_A;
                        const int b_limit = b_rem < DIFF_V2_CHUNK_B
                            ? static_cast<int>(b_rem) : DIFF_V2_CHUNK_B;
                        const int a_slot_base = threadIdx.y * DIFF_V2_CHUNK_A;
                        const int b_slot_base = threadIdx.x * DIFF_V2_CHUNK_B;

                        for (int ia = 0; ia < a_limit; ++ia) {
                            const int a_slot = a_slot_base + ia;
                            const int a_src = sh_a_src[a_slot];
                            const int ij = sh_a_pair[a_slot];
                            const int pa = sh_a_parity[a_slot];

                            for (int ib = 0; ib < b_limit; ++ib) {
                                const int b_slot = b_slot_base + ib;
                                const int b_src = sh_b_src[b_slot];
                                const int kl = sh_b_pair[b_slot];
                                const int pb = sh_b_parity[b_slot];
                                const size_t h_idx =
                                    static_cast<size_t>(ij) * static_cast<size_t>(norbs2)
                                    + static_cast<size_t>(kl);
                                const size_t c_idx =
                                    static_cast<size_t>(a_src) * static_cast<size_t>(beta_states)
                                    + static_cast<size_t>(b_src);
                                double weight = d_h2e[h_idx];
                                if (pa * pb == -1) {
                                    weight = -weight;
                                }
                                const cuDoubleComplex cval = d_C[c_idx];
                                acc.x += weight * cval.x;
                                acc.y += weight * cval.y;
                            }
                        }
                    }
                    __syncthreads();
                }
            }

            if (active) {
                const size_t out_idx =
                    static_cast<size_t>(a_out) * static_cast<size_t>(beta_states)
                    + static_cast<size_t>(b_out);
                d_out[out_idx].x += acc.x;
                d_out[out_idx].y += acc.y;
            }
            __syncthreads();
        }
    }
}

__global__ void lm_apply_array12_diff_spin_v2_tiled_kernel(
    cuDoubleComplex* __restrict__ d_out,
    const cuDoubleComplex* __restrict__ d_C,
    const long long* __restrict__ d_alpha_offsets,
    const int* __restrict__ d_alpha_sources,
    const int* __restrict__ d_alpha_pairs,
    const int* __restrict__ d_alpha_parities,
    const long long* __restrict__ d_beta_offsets,
    const int* __restrict__ d_beta_sources,
    const int* __restrict__ d_beta_pairs,
    const int* __restrict__ d_beta_parities,
    const cuDoubleComplex* __restrict__ d_h2e,
    long long alpha_states,
    long long beta_states,
    int norbs)
{
    __shared__ long long sh_a_begin[DIFF_V2_TILE_A];
    __shared__ long long sh_a_count[DIFF_V2_TILE_A];
    __shared__ long long sh_b_begin[DIFF_V2_TILE_B];
    __shared__ long long sh_b_count[DIFF_V2_TILE_B];
    __shared__ long long sh_max_a;
    __shared__ long long sh_max_b;

    extern __shared__ int sh_edges[];
    int* sh_a_src = sh_edges;
    int* sh_a_pair = sh_a_src + DIFF_V2_TILE_A * DIFF_V2_CHUNK_A;
    int* sh_a_parity = sh_a_pair + DIFF_V2_TILE_A * DIFF_V2_CHUNK_A;
    int* sh_b_src = sh_a_parity + DIFF_V2_TILE_A * DIFF_V2_CHUNK_A;
    int* sh_b_pair = sh_b_src + DIFF_V2_TILE_B * DIFF_V2_CHUNK_B;
    int* sh_b_parity = sh_b_pair + DIFF_V2_TILE_B * DIFF_V2_CHUNK_B;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;
    const long long norbs2 = static_cast<long long>(norbs) * norbs;

    for (long long a_base = static_cast<long long>(blockIdx.y) * DIFF_V2_TILE_A;
         a_base < alpha_states;
         a_base += static_cast<long long>(gridDim.y) * DIFF_V2_TILE_A) {
        for (long long b_base = static_cast<long long>(blockIdx.x) * DIFF_V2_TILE_B;
             b_base < beta_states;
             b_base += static_cast<long long>(gridDim.x) * DIFF_V2_TILE_B) {

            if (tid < DIFF_V2_TILE_A) {
                const long long a_out = a_base + tid;
                if (a_out < alpha_states) {
                    sh_a_begin[tid] = d_alpha_offsets[a_out];
                    sh_a_count[tid] = d_alpha_offsets[a_out + 1] - sh_a_begin[tid];
                } else {
                    sh_a_begin[tid] = 0;
                    sh_a_count[tid] = 0;
                }
            }
            if (tid < DIFF_V2_TILE_B) {
                const long long b_out = b_base + tid;
                if (b_out < beta_states) {
                    sh_b_begin[tid] = d_beta_offsets[b_out];
                    sh_b_count[tid] = d_beta_offsets[b_out + 1] - sh_b_begin[tid];
                } else {
                    sh_b_begin[tid] = 0;
                    sh_b_count[tid] = 0;
                }
            }
            __syncthreads();

            if (tid == 0) {
                long long max_a = 0;
                long long max_b = 0;
                for (int i = 0; i < DIFF_V2_TILE_A; ++i) {
                    max_a = max_a < sh_a_count[i] ? sh_a_count[i] : max_a;
                }
                for (int i = 0; i < DIFF_V2_TILE_B; ++i) {
                    max_b = max_b < sh_b_count[i] ? sh_b_count[i] : max_b;
                }
                sh_max_a = max_a;
                sh_max_b = max_b;
            }
            __syncthreads();

            const long long a_out = a_base + threadIdx.y;
            const long long b_out = b_base + threadIdx.x;
            const bool active = a_out < alpha_states && b_out < beta_states;
            cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);

            for (long long a_chunk = 0; a_chunk < sh_max_a; a_chunk += DIFF_V2_CHUNK_A) {
                for (int slot = tid;
                     slot < DIFF_V2_TILE_A * DIFF_V2_CHUNK_A;
                     slot += nthreads) {
                    const int local_a = slot / DIFF_V2_CHUNK_A;
                    const int k = slot - local_a * DIFF_V2_CHUNK_A;
                    const long long edge = sh_a_begin[local_a] + a_chunk + k;
                    if (a_chunk + k < sh_a_count[local_a]) {
                        sh_a_src[slot] = d_alpha_sources[edge];
                        sh_a_pair[slot] = d_alpha_pairs[edge];
                        sh_a_parity[slot] = d_alpha_parities[edge];
                    } else {
                        sh_a_src[slot] = 0;
                        sh_a_pair[slot] = 0;
                        sh_a_parity[slot] = 0;
                    }
                }
                __syncthreads();

                for (long long b_chunk = 0; b_chunk < sh_max_b; b_chunk += DIFF_V2_CHUNK_B) {
                    for (int slot = tid;
                         slot < DIFF_V2_TILE_B * DIFF_V2_CHUNK_B;
                         slot += nthreads) {
                        const int local_b = slot / DIFF_V2_CHUNK_B;
                        const int k = slot - local_b * DIFF_V2_CHUNK_B;
                        const long long edge = sh_b_begin[local_b] + b_chunk + k;
                        if (b_chunk + k < sh_b_count[local_b]) {
                            sh_b_src[slot] = d_beta_sources[edge];
                            sh_b_pair[slot] = d_beta_pairs[edge];
                            sh_b_parity[slot] = d_beta_parities[edge];
                        } else {
                            sh_b_src[slot] = 0;
                            sh_b_pair[slot] = 0;
                            sh_b_parity[slot] = 0;
                        }
                    }
                    __syncthreads();

                    if (active) {
                        const long long a_rem = sh_a_count[threadIdx.y] - a_chunk;
                        const long long b_rem = sh_b_count[threadIdx.x] - b_chunk;
                        const int a_limit = a_rem < DIFF_V2_CHUNK_A
                            ? static_cast<int>(a_rem) : DIFF_V2_CHUNK_A;
                        const int b_limit = b_rem < DIFF_V2_CHUNK_B
                            ? static_cast<int>(b_rem) : DIFF_V2_CHUNK_B;
                        const int a_slot_base = threadIdx.y * DIFF_V2_CHUNK_A;
                        const int b_slot_base = threadIdx.x * DIFF_V2_CHUNK_B;

                        for (int ia = 0; ia < a_limit; ++ia) {
                            const int a_slot = a_slot_base + ia;
                            const int a_src = sh_a_src[a_slot];
                            const int ij = sh_a_pair[a_slot];
                            const int pa = sh_a_parity[a_slot];

                            for (int ib = 0; ib < b_limit; ++ib) {
                                const int b_slot = b_slot_base + ib;
                                const int b_src = sh_b_src[b_slot];
                                const int kl = sh_b_pair[b_slot];
                                const int pb = sh_b_parity[b_slot];
                                const size_t h_idx =
                                    static_cast<size_t>(ij) * static_cast<size_t>(norbs2)
                                    + static_cast<size_t>(kl);
                                const size_t c_idx =
                                    static_cast<size_t>(a_src) * static_cast<size_t>(beta_states)
                                    + static_cast<size_t>(b_src);
                                cuDoubleComplex weight = d_h2e[h_idx];
                                if (pa * pb == -1) {
                                    weight.x = -weight.x;
                                    weight.y = -weight.y;
                                }
                                const cuDoubleComplex prod = cuCmul(weight, d_C[c_idx]);
                                acc.x += prod.x;
                                acc.y += prod.y;
                            }
                        }
                    }
                    __syncthreads();
                }
            }

            if (active) {
                const size_t out_idx =
                    static_cast<size_t>(a_out) * static_cast<size_t>(beta_states)
                    + static_cast<size_t>(b_out);
                d_out[out_idx].x += acc.x;
                d_out[out_idx].y += acc.y;
            }
            __syncthreads();
        }
    }
}

extern "C" void lm_apply_array12_diff_spin_v2_tiled_wrapper_real(
    double* d_out,
    const double* d_C,
    const long long* d_alpha_offsets,
    const int* d_alpha_sources,
    const int* d_alpha_pairs,
    const int* d_alpha_parities,
    const long long* d_beta_offsets,
    const int* d_beta_sources,
    const int* d_beta_pairs,
    const int* d_beta_parities,
    const double* d_h2e,
    long long alpha_states,
    long long beta_states,
    int norbs)
{
    const long long beta_tiles =
        (beta_states + DIFF_V2_TILE_B - 1) / DIFF_V2_TILE_B;
    const long long alpha_tiles =
        (alpha_states + DIFF_V2_TILE_A - 1) / DIFF_V2_TILE_A;
    dim3 block(DIFF_V2_TILE_B, DIFF_V2_TILE_A);
    dim3 grid(
        diff_spin_v2_tiled_grid_dim(beta_tiles),
        diff_spin_v2_tiled_grid_dim(alpha_tiles));

    lm_apply_array12_diff_spin_v2_tiled_kernel_real<<<
        grid, block, diff_spin_v2_tiled_shared_bytes()>>>(
            d_out,
            d_C,
            d_alpha_offsets,
            d_alpha_sources,
            d_alpha_pairs,
            d_alpha_parities,
            d_beta_offsets,
            d_beta_sources,
            d_beta_pairs,
            d_beta_parities,
            d_h2e,
            alpha_states,
            beta_states,
            norbs);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "lm_apply_array12_diff_spin_v2_tiled_wrapper_real failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("lm_apply_array12_diff_spin_v2_tiled_wrapper_real failed");
    }
}

extern "C" void lm_apply_array12_diff_spin_v2_tiled_wrapper_mixed(
    cuDoubleComplex* d_out,
    const cuDoubleComplex* d_C,
    const long long* d_alpha_offsets,
    const int* d_alpha_sources,
    const int* d_alpha_pairs,
    const int* d_alpha_parities,
    const long long* d_beta_offsets,
    const int* d_beta_sources,
    const int* d_beta_pairs,
    const int* d_beta_parities,
    const double* d_h2e,
    long long alpha_states,
    long long beta_states,
    int norbs)
{
    const long long beta_tiles =
        (beta_states + DIFF_V2_TILE_B - 1) / DIFF_V2_TILE_B;
    const long long alpha_tiles =
        (alpha_states + DIFF_V2_TILE_A - 1) / DIFF_V2_TILE_A;
    dim3 block(DIFF_V2_TILE_B, DIFF_V2_TILE_A);
    dim3 grid(
        diff_spin_v2_tiled_grid_dim(beta_tiles),
        diff_spin_v2_tiled_grid_dim(alpha_tiles));

    lm_apply_array12_diff_spin_v2_tiled_kernel_mixed<<<
        grid, block, diff_spin_v2_tiled_shared_bytes()>>>(
            d_out,
            d_C,
            d_alpha_offsets,
            d_alpha_sources,
            d_alpha_pairs,
            d_alpha_parities,
            d_beta_offsets,
            d_beta_sources,
            d_beta_pairs,
            d_beta_parities,
            d_h2e,
            alpha_states,
            beta_states,
            norbs);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "lm_apply_array12_diff_spin_v2_tiled_wrapper_mixed failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("lm_apply_array12_diff_spin_v2_tiled_wrapper_mixed failed");
    }
}

extern "C" void lm_apply_array12_diff_spin_v2_tiled_wrapper(
    cuDoubleComplex* d_out,
    const cuDoubleComplex* d_C,
    const long long* d_alpha_offsets,
    const int* d_alpha_sources,
    const int* d_alpha_pairs,
    const int* d_alpha_parities,
    const long long* d_beta_offsets,
    const int* d_beta_sources,
    const int* d_beta_pairs,
    const int* d_beta_parities,
    const cuDoubleComplex* d_h2e,
    long long alpha_states,
    long long beta_states,
    int norbs)
{
    const long long beta_tiles =
        (beta_states + DIFF_V2_TILE_B - 1) / DIFF_V2_TILE_B;
    const long long alpha_tiles =
        (alpha_states + DIFF_V2_TILE_A - 1) / DIFF_V2_TILE_A;
    dim3 block(DIFF_V2_TILE_B, DIFF_V2_TILE_A);
    dim3 grid(
        diff_spin_v2_tiled_grid_dim(beta_tiles),
        diff_spin_v2_tiled_grid_dim(alpha_tiles));

    lm_apply_array12_diff_spin_v2_tiled_kernel<<<
        grid, block, diff_spin_v2_tiled_shared_bytes()>>>(
            d_out,
            d_C,
            d_alpha_offsets,
            d_alpha_sources,
            d_alpha_pairs,
            d_alpha_parities,
            d_beta_offsets,
            d_beta_sources,
            d_beta_pairs,
            d_beta_parities,
            d_h2e,
            alpha_states,
            beta_states,
            norbs);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "lm_apply_array12_diff_spin_v2_tiled_wrapper failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("lm_apply_array12_diff_spin_v2_tiled_wrapper failed");
    }
}
