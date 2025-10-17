// tests/test_scale_kernels.cu
// Self-contained test harness: no header imports beyond standard/CUDA libs.

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>  // shuffle, copy_n
#include <numeric>    // iota


#define CUDA_CHECK(call) do {                                        \
    cudaError_t _e = (call);                                         \
    if (_e != cudaSuccess) {                                         \
        std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__    \
                  << " : " << cudaGetErrorString(_e) << std::endl;   \
        std::exit(1);                                                \
    }                                                                \
} while(0)

// ================================
// Reference kernel (complex path)
// ================================
__global__ void scale_elements_kernel(
    cuDoubleComplex* d_Cout,
    const int* d_first, 
    int first_size,
    const int* d_second, 
    int second_size,
    int nbeta_strs_,
    cuDoubleComplex factor) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < first_size && j < second_size) {
        int idx = d_first[i] * nbeta_strs_ + d_second[j];
        d_Cout[idx] = cuCmul(d_Cout[idx], factor);
    }
}

// Your wrapper (as given)
extern "C" void scale_elements_wrapper_complex(
    cuDoubleComplex* d_Cout,
    const int* d_first, 
    int first_size,
    const int* d_second, 
    int second_size,
    int nbeta_strs_,
    cuDoubleComplex factor) 
{
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

// ================================
// Your split-array kernels (as given)
// ================================
__global__ void scale_elements_kernel_soa_factor_real(
    double* __restrict__ dCr, double* __restrict__ dCi,
    const int* __restrict__ d_first, int first_size,
    const int* __restrict__ d_second, int second_size,
    int nbeta_strs_, double fr)  // factor = fr
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < first_size && j < second_size) {
        int idx = d_first[i] * nbeta_strs_ + d_second[j];
        dCr[idx] *= fr;
        dCi[idx] *= fr;
    }
}

__global__ void scale_elements_kernel_soa_factor_imag(
    double* __restrict__ dCr, double* __restrict__ dCi,
    const int* __restrict__ d_first, int first_size,
    const int* __restrict__ d_second, int second_size,
    int nbeta_strs_, double fi)  // factor = i*fi
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < first_size && j < second_size) {
        int idx = d_first[i] * nbeta_strs_ + d_second[j];
        double xr = dCr[idx], xi = dCi[idx];
        // (xr + i*xi) * (i*fi) = (-fi*xi) + i*(fi*xr)
        dCr[idx] = -fi * xi;
        dCi[idx] =  fi * xr;
    }
}

// ================================
// Test utilities
// ================================
static bool almost_equal(double a, double b, double tol=1e-12) {
    double diff = std::fabs(a - b);
    double scale = 1.0;
    double aa = std::fabs(a), bb = std::fabs(b);
    if (aa > scale) scale = aa;
    if (bb > scale) scale = bb;
    return diff <= tol * scale;
}

struct Sizes {
    int nalpha;       // rows of Cout
    int nbeta;        // columns (nbeta_strs_)
    int first_size;   // |d_first|
    int second_size;  // |d_second|
};

static void fill_unique_indices(int limit, int count, std::vector<int>& out, std::mt19937_64& rng) {
    // count must be <= limit
    out.resize(count);
    std::vector<int> pool(limit);
    std::iota(pool.begin(), pool.end(), 0);
    std::shuffle(pool.begin(), pool.end(), rng);
    std::copy_n(pool.begin(), count, out.begin());
}

static void init_random_unique(Sizes sz,
                               std::vector<int>& h_first,
                               std::vector<int>& h_second,
                               std::vector<cuDoubleComplex>& h_C)
{
    std::mt19937_64 rng(42);
    fill_unique_indices(sz.nalpha, sz.first_size,  h_first,  rng);
    fill_unique_indices(sz.nbeta,  sz.second_size, h_second, rng);

    h_C.resize((size_t)sz.nalpha * sz.nbeta);
    std::uniform_real_distribution<double> dist_v(-1.0, 1.0);
    for (auto& z : h_C) {
        z = make_cuDoubleComplex(dist_v(rng), dist_v(rng));
    }
}


static dim3 grid_for(int nx, int ny, dim3 block=dim3(16,16)) {
    return dim3((nx + block.x - 1)/block.x, (ny + block.y - 1)/block.y);
}

// ================================
// Tests
// ================================
static int test_real_case(const Sizes& sz, double alpha, double tol)
{
    // Host init
    std::vector<int> h_first, h_second;
    std::vector<cuDoubleComplex> h_C;
    init_random_unique(sz, h_first, h_second, h_C);

    size_t N = (size_t)sz.nalpha * sz.nbeta;

    // Device buffers
    cuDoubleComplex* d_Cref = nullptr;
    double *d_Cr = nullptr, *d_Ci = nullptr;
    int *d_first = nullptr, *d_second = nullptr;

    CUDA_CHECK(cudaMalloc(&d_Cref,  N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_Cr,    N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ci,    N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_first,  sz.first_size  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_second, sz.second_size * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_Cref,  h_C.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_first, h_first.data(), sz.first_size  * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_second,h_second.data(),sz.second_size * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize split arrays
    {
        std::vector<double> hr(N), hi(N);
        for (size_t k = 0; k < N; ++k) {
            hr[k] = cuCreal(h_C[k]);
            hi[k] = cuCimag(h_C[k]);
        }
        CUDA_CHECK(cudaMemcpy(d_Cr, hr.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Ci, hi.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Run reference (complex) with real factor
    scale_elements_wrapper_complex(d_Cref, d_first, sz.first_size, d_second, sz.second_size, sz.nbeta,
                                   make_cuDoubleComplex(alpha, 0.0));

    // Run split-SoA kernel (real factor)
    {
        dim3 block(16,16);
        dim3 grid = grid_for(sz.first_size, sz.second_size, block);
        scale_elements_kernel_soa_factor_real<<<grid, block>>>(
            d_Cr, d_Ci, d_first, sz.first_size, d_second, sz.second_size, sz.nbeta, alpha);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Compare
    std::vector<cuDoubleComplex> h_ref(N);
    std::vector<double> hr2(N), hi2(N);
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_Cref, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hr2.data(),   d_Cr,   N * sizeof(double),          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hi2.data(),   d_Ci,   N * sizeof(double),          cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (size_t k = 0; k < N; ++k) {
        double rr = cuCreal(h_ref[k]);
        double ri = cuCimag(h_ref[k]);
        if (!almost_equal(hr2[k], rr, tol) || !almost_equal(hi2[k], ri, tol)) {
            if (++mismatches <= 10) {
                std::cerr << "[REAL] Mismatch at " << k
                          << " got (" << hr2[k] << ", " << hi2[k]
                          << ") vs ref (" << rr << ", " << ri << ")\n";
            }
        }
    }

    cudaFree(d_Cref); cudaFree(d_Cr); cudaFree(d_Ci);
    cudaFree(d_first); cudaFree(d_second);
    return mismatches;
}

static int test_imag_case(const Sizes& sz, double beta, double tol)
{
    // Host init
    std::vector<int> h_first, h_second;
    std::vector<cuDoubleComplex> h_C;
    init_random_unique(sz, h_first, h_second, h_C);

    size_t N = (size_t)sz.nalpha * sz.nbeta;

    // Device buffers
    cuDoubleComplex* d_Cref = nullptr;
    double *d_Cr = nullptr, *d_Ci = nullptr;
    int *d_first = nullptr, *d_second = nullptr;

    CUDA_CHECK(cudaMalloc(&d_Cref,  N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_Cr,    N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ci,    N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_first,  sz.first_size  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_second, sz.second_size * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_Cref,  h_C.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_first, h_first.data(), sz.first_size  * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_second,h_second.data(),sz.second_size * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize split arrays
    {
        std::vector<double> hr(N), hi(N);
        for (size_t k = 0; k < N; ++k) {
            hr[k] = cuCreal(h_C[k]);
            hi[k] = cuCimag(h_C[k]);
        }
        CUDA_CHECK(cudaMemcpy(d_Cr, hr.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Ci, hi.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Run reference (complex) with imaginary factor i*beta
    scale_elements_wrapper_complex(d_Cref, d_first, sz.first_size, d_second, sz.second_size, sz.nbeta,
                                   make_cuDoubleComplex(0.0, beta));

    // Run split-SoA kernel (imag factor)
    {
        dim3 block(16,16);
        dim3 grid = grid_for(sz.first_size, sz.second_size, block);
        scale_elements_kernel_soa_factor_imag<<<grid, block>>>(
            d_Cr, d_Ci, d_first, sz.first_size, d_second, sz.second_size, sz.nbeta, beta);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Compare
    std::vector<cuDoubleComplex> h_ref(N);
    std::vector<double> hr2(N), hi2(N);
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_Cref, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hr2.data(),   d_Cr,   N * sizeof(double),          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hi2.data(),   d_Ci,   N * sizeof(double),          cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (size_t k = 0; k < N; ++k) {
        double rr = cuCreal(h_ref[k]);
        double ri = cuCimag(h_ref[k]);
        if (!almost_equal(hr2[k], rr, tol) || !almost_equal(hi2[k], ri, tol)) {
            if (++mismatches <= 10) {
                std::cerr << "[IMAG] Mismatch at " << k
                          << " got (" << hr2[k] << ", " << hi2[k]
                          << ") vs ref (" << rr << ", " << ri << ")\n";
            }
        }
    }

    cudaFree(d_Cref); cudaFree(d_Cr); cudaFree(d_Ci);
    cudaFree(d_first); cudaFree(d_second);
    return mismatches;
}

int main()
{
    // Dimensions (tweak as you like)
    Sizes sz;
    sz.nalpha = 256;       // rows in Cout
    sz.nbeta  = 384;       // columns (nbeta_strs_)
    sz.first_size  = 173;  // number of row indices touched
    sz.second_size = 289;  // number of col indices touched

    double tol = 1e-12;
    double alpha = 2.25;   // purely real factor
    double beta  = -1.7;   // purely imaginary factor i*beta

    int m1 = test_real_case(sz, alpha, tol);
    int m2 = test_imag_case(sz, beta,  tol);

    if (m1 == 0) std::cout << "REAL factor test: PASS\n";
    else         std::cout << "REAL factor test: FAIL (" << m1 << " mismatches)\n";

    if (m2 == 0) std::cout << "IMAG factor test: PASS\n";
    else         std::cout << "IMAG factor test: FAIL (" << m2 << " mismatches)\n";

    return (m1 == 0 && m2 == 0) ? 0 : 1;
}
