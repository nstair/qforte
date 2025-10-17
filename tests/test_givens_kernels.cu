// tests/test_givens_kernels.cu
// Self-contained test for the tiled Givens-update kernels.
//
// Build:
//   nvcc -O2 -std=c++17 -arch=sm_86 tests/test_givens_kernels.cu -o test_givens_kernels
//
// This constructs inputs that avoid write conflicts by design:
//   - For rows:   sa1 = 0..R-1,  ta1 = R..2R-1 with R = nalpha/2   (disjoint)
//   - For cols:   sb1 = 0..nb-1, tb1 = nb..2*nb-1 (disjoint)
// Requirements we enforce: nalpha is even, nbeta_strs_ >= 2*nb.

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>  // iota/shuffle/copy_n (we'll use iota)
#include <numeric>    // iota

#define CUDA_CHECK(call) do {                                        \
    cudaError_t _e = (call);                                         \
    if (_e != cudaSuccess) {                                         \
        std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__    \
                  << " : " << cudaGetErrorString(_e) << std::endl;   \
        std::exit(1);                                                \
    }                                                                \
} while(0)

static bool almost_equal(double a, double b, double tol=1e-12) {
    double diff = std::fabs(a - b);
    double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    return diff <= tol * scale;
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
    int nbeta_strs_,
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2)
{
    const int ib0 = blockIdx.x * BX;
    if (ib0 >= nb) return;

    const int tx = threadIdx.x;             // [0, BX)
    const int ty = threadIdx.y;             // [0, AY)
    constexpr int AY = 8;
    static_assert(BX % 32 == 0, "Pick BX multiple of warp width for coalescing");

    __shared__ int s_sb1[BX], s_tb1[BX];
    __shared__ cuDoubleComplex s_pb1[BX], s_pb2[BX];

    __shared__ int s_sa1[AY], s_ta1[AY];
    __shared__ cuDoubleComplex s_pa1[AY], s_pa2[AY];

    if (tx + ib0 < nb && ty == 0) {
        const int ib = ib0 + tx;
        s_sb1[tx] = sourceb1[ib];
        s_tb1[tx] = targetb1[ib];
        s_pb1[tx] = parityb1[ib];
        s_pb2[tx] = parityb2[ib];
    }
    __syncthreads();

    for (int ia0 = blockIdx.y * AY; ia0 < nalpha; ia0 += gridDim.y * AY)
    {
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
            const int sa1 = s_sa1[ty];
            const int ta1 = s_ta1[ty];
            const cuDoubleComplex pa1 = s_pa1[ty];
            const cuDoubleComplex pa2 = s_pa2[ty];

            const int sb1  = s_sb1[tx];
            const int tb1  = s_tb1[tx];
            const cuDoubleComplex pb1 = s_pb1[tx];
            const cuDoubleComplex pb2 = s_pb2[tx];

            const int base_u = sa1 * nbeta_strs_;
            const int base_v = ta1 * nbeta_strs_;

            const int idx_u  = base_u + sb1;  // (sa1, sb1)
            const int idx_v  = base_v + tb1;  // (ta1, tb1)

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

template<int BX>
__global__ void givens_update_soa_tiled_factor_real(
    double* __restrict__ dCr, double* __restrict__ dCi,
    const int* __restrict__ sourcea1,
    const int* __restrict__ targeta1,
    const double* __restrict__ pa1r, const double* __restrict__ pa1i,
    const double* __restrict__ pa2r, const double* __restrict__ pa2i,
    const int* __restrict__ sourceb1,
    const int* __restrict__ targetb1,
    const double* __restrict__ pb1r, const double* __restrict__ pb1i,
    const double* __restrict__ pb2r, const double* __restrict__ pb2i,
    int nalpha, int nb, int nbeta_strs_,
    double factor_real,
    double acc1r, double acc1i,
    double acc2r, double acc2i)
{
    constexpr int AY = 8;
    static_assert(BX % 32 == 0, "BX must be a multiple of 32");

    const int ib0 = blockIdx.x * BX;
    if (ib0 >= nb) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ int s_sb1[BX], s_tb1[BX];
    __shared__ double s_pb1r[BX], s_pb1i[BX], s_pb2r[BX], s_pb2i[BX];

    __shared__ int s_sa1[AY], s_ta1[AY];
    __shared__ double s_pa1r[AY], s_pa1i[AY], s_pa2r[AY], s_pa2i[AY];

    if (tx + ib0 < nb && ty == 0) {
        const int ib = ib0 + tx;
        s_sb1[tx] = sourceb1[ib];
        s_tb1[tx] = targetb1[ib];
        s_pb1r[tx] = pb1r[ib];
        s_pb1i[tx] = pb1i[ib];
        s_pb2r[tx] = pb2r[ib];
        s_pb2i[tx] = pb2i[ib];
    }
    __syncthreads();

    for (int ia0 = blockIdx.y * AY; ia0 < nalpha; ia0 += gridDim.y * AY) {
        if (ty < AY && tx == 0) {
            const int ia = ia0 + ty;
            if (ia < nalpha) {
                s_sa1[ty] = sourcea1[ia];
                s_ta1[ty] = targeta1[ia];
                s_pa1r[ty] = pa1r[ia]; s_pa1i[ty] = pa1i[ia];
                s_pa2r[ty] = pa2r[ia]; s_pa2i[ty] = pa2i[ia];
            }
        }
        __syncthreads();

        const int ia = ia0 + ty;
        if (ia < nalpha && tx + ib0 < nb) {
            const int sa1 = s_sa1[ty];
            const int ta1 = s_ta1[ty];
            const int sb1 = s_sb1[tx];
            const int tb1 = s_tb1[tx];

            const int idx_u = sa1 * nbeta_strs_ + sb1;
            const int idx_v = ta1 * nbeta_strs_ + tb1;

            const double ur0 = dCr[idx_u], ui0 = dCi[idx_u];
            const double vr0 = dCr[idx_v], vi0 = dCi[idx_v];

            const double p1r = s_pa1r[ty]*s_pb1r[tx] - s_pa1i[ty]*s_pb1i[tx];
            const double p1i = s_pa1r[ty]*s_pb1i[tx] + s_pa1i[ty]*s_pb1r[tx];

            const double p2r = s_pa2r[ty]*s_pb2r[tx] - s_pa2i[ty]*s_pb2i[tx];
            const double p2i = s_pa2r[ty]*s_pb2i[tx] + s_pa2i[ty]*s_pb2r[tx];

            const double t2r = p2r*vr0 - p2i*vi0;
            const double t2i = p2r*vi0 + p2i*vr0;

            const double a2t2r = acc2r*t2r - acc2i*t2i;
            const double a2t2i = acc2r*t2i + acc2i*t2r;

            const double t1r = p1r*ur0 - p1i*ui0;
            const double t1i = p1r*ui0 + p1i*ur0;

            const double a1t1r = acc1r*t1r - acc1i*t1i;
            const double a1t1i = acc1r*t1i + acc1i*t1r;

            const double fu_r = factor_real * ur0;
            const double fu_i = factor_real * ui0;
            const double fv_r = factor_real * vr0;
            const double fv_i = factor_real * vi0;

            dCr[idx_u] = fu_r + a2t2r;
            dCi[idx_u] = fu_i + a2t2i;
            dCr[idx_v] = fv_r + a1t1r;
            dCi[idx_v] = fv_i + a1t1i;
        }
        __syncthreads();
    }
}

template<int BX>
__global__ void givens_update_soa_tiled_factor_imag(
    double* __restrict__ dCr, double* __restrict__ dCi,
    const int* __restrict__ sourcea1,
    const int* __restrict__ targeta1,
    const double* __restrict__ pa1r, const double* __restrict__ pa1i,
    const double* __restrict__ pa2r, const double* __restrict__ pa2i,
    const int* __restrict__ sourceb1,
    const int* __restrict__ targetb1,
    const double* __restrict__ pb1r, const double* __restrict__ pb1i,
    const double* __restrict__ pb2r, const double* __restrict__ pb2i,
    int nalpha, int nb, int nbeta_strs_,
    double factor_imag,
    double acc1r, double acc1i,
    double acc2r, double acc2i)
{
    constexpr int AY = 8;
    static_assert(BX % 32 == 0, "BX must be a multiple of 32");

    const int ib0 = blockIdx.x * BX;
    if (ib0 >= nb) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ int s_sb1[BX], s_tb1[BX];
    __shared__ double s_pb1r[BX], s_pb1i[BX], s_pb2r[BX], s_pb2i[BX];

    __shared__ int s_sa1[AY], s_ta1[AY];
    __shared__ double s_pa1r[AY], s_pa1i[AY], s_pa2r[AY], s_pa2i[AY];

    if (tx + ib0 < nb && ty == 0) {
        const int ib = ib0 + tx;
        s_sb1[tx] = sourceb1[ib];
        s_tb1[tx] = targetb1[ib];
        s_pb1r[tx] = pb1r[ib];
        s_pb1i[tx] = pb1i[ib];
        s_pb2r[tx] = pb2r[ib];
        s_pb2i[tx] = pb2i[ib];
    }
    __syncthreads();

    for (int ia0 = blockIdx.y * AY; ia0 < nalpha; ia0 += gridDim.y * AY) {
        if (ty < AY && tx == 0) {
            const int ia = ia0 + ty;
            if (ia < nalpha) {
                s_sa1[ty] = sourcea1[ia];
                s_ta1[ty] = targeta1[ia];
                s_pa1r[ty] = pa1r[ia]; s_pa1i[ty] = pa1i[ia];
                s_pa2r[ty] = pa2r[ia]; s_pa2i[ty] = pa2i[ia];
            }
        }
        __syncthreads();

        const int ia = ia0 + ty;
        if (ia < nalpha && tx + ib0 < nb) {
            const int sa1 = s_sa1[ty];
            const int ta1 = s_ta1[ty];
            const int sb1 = s_sb1[tx];
            const int tb1 = s_tb1[tx];

            const int idx_u = sa1 * nbeta_strs_ + sb1;
            const int idx_v = ta1 * nbeta_strs_ + tb1;

            const double ur0 = dCr[idx_u], ui0 = dCi[idx_u];
            const double vr0 = dCr[idx_v], vi0 = dCi[idx_v];

            const double p1r = s_pa1r[ty]*s_pb1r[tx] - s_pa1i[ty]*s_pb1i[tx];
            const double p1i = s_pa1r[ty]*s_pb1i[tx] + s_pa1i[ty]*s_pb1r[tx];

            const double p2r = s_pa2r[ty]*s_pb2r[tx] - s_pa2i[ty]*s_pb2i[tx];
            const double p2i = s_pa2r[ty]*s_pb2i[tx] + s_pa2i[ty]*s_pb2r[tx];

            const double t2r = p2r*vr0 - p2i*vi0;
            const double t2i = p2r*vi0 + p2i*vr0;

            const double a2t2r = acc2r*t2r - acc2i*t2i;
            const double a2t2i = acc2r*t2i + acc2i*t2r;

            const double t1r = p1r*ur0 - p1i*ui0;
            const double t1i = p1r*ui0 + p1i*ur0;

            const double a1t1r = acc1r*t1r - acc1i*t1i;
            const double a1t1i = acc1r*t1i + acc1i*t1r;

            const double fi = factor_imag;
            const double fu_r = -fi * ui0;
            const double fu_i =  fi * ur0;
            const double fv_r = -fi * vi0;
            const double fv_i =  fi * vr0;

            dCr[idx_u] = fu_r + a2t2r;
            dCi[idx_u] = fu_i + a2t2i;
            dCr[idx_v] = fv_r + a1t1r;
            dCi[idx_v] = fv_i + a1t1i;
        }
        __syncthreads();
    }
}

// ------------------------------
// Test driver
// ------------------------------
struct Sizes {
    int nalpha;       // rows of C
    int nb;           // number of column-pairs
    int nbeta;        // nbeta_strs_ (columns)
};

static void make_race_free_maps(const Sizes& s,
                                std::vector<int>& sa1,
                                std::vector<int>& ta1,
                                std::vector<int>& sb1,
                                std::vector<int>& tb1)
{
    /// require nalpha even; let R = nalpha/2
    int R = s.nalpha / 2;
    sa1.resize(s.nalpha);
    ta1.resize(s.nalpha);

    // Unique rows: sa1 is a bijection of ia; ta1 is a disjoint bijection
    for (int ia = 0; ia < s.nalpha; ++ia) {
        sa1[ia] = ia;                 // 0..nalpha-1 (unique)
        ta1[ia] = (ia + R) % s.nalpha; // also unique, and disjoint from sa1 for all ia
    }

    // Column pairs: sb1 = 0..nb-1, tb1 = nb..2*nb-1  (requires nbeta >= 2*nb)
    sb1.resize(s.nb);
    tb1.resize(s.nb);
    for (int ib = 0; ib < s.nb; ++ib) {
        sb1[ib] = ib;
        tb1[ib] = ib + s.nb;
    }
}

static void init_random_state(const Sizes& s,
                              std::vector<cuDoubleComplex>& C,
                              std::vector<cuDoubleComplex>& pa1,
                              std::vector<cuDoubleComplex>& pa2,
                              std::vector<cuDoubleComplex>& pb1,
                              std::vector<cuDoubleComplex>& pb2)
{
    std::mt19937_64 rng(123);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    C.resize((size_t)s.nalpha * s.nbeta);
    for (auto& z : C) z = make_cuDoubleComplex(dist(rng), dist(rng));

    pa1.resize(s.nalpha);
    pa2.resize(s.nalpha);
    for (int i = 0; i < s.nalpha; ++i) {
        pa1[i] = make_cuDoubleComplex(dist(rng), dist(rng));
        pa2[i] = make_cuDoubleComplex(dist(rng), dist(rng));
    }

    pb1.resize(s.nb);
    pb2.resize(s.nb);
    for (int i = 0; i < s.nb; ++i) {
        pb1[i] = make_cuDoubleComplex(dist(rng), dist(rng));
        pb2[i] = make_cuDoubleComplex(dist(rng), dist(rng));
    }
}

template <typename F>
static int compare_with_tol(size_t N,
                            const std::vector<double>& r,
                            const std::vector<double>& i,
                            const std::vector<cuDoubleComplex>& ref,
                            double tol,
                            const char* tag,
                            F printer)
{
    int mism = 0;
    for (size_t k = 0; k < N; ++k) {
        double rr = cuCreal(ref[k]);
        double ri = cuCimag(ref[k]);
        if (!almost_equal(r[k], rr, tol) || !almost_equal(i[k], ri, tol)) {
            if (++mism <= 10) {
                printer(k, r[k], i[k], rr, ri, tag);
            }
        }
    }
    return mism;
}

int main()
{
    // Choose sizes satisfying constraints:
    // nalpha must be even; nbeta >= 2*nb
    const int BX = 32;
    const int AY = 8;

    Sizes s;
    s.nalpha = 128;          // even, and multiple of AY works well
    s.nb     = 64;           // number of column-pairs (multiple of BX is nice)
    s.nbeta  = 256;          // >= 2 * nb (we use sb1 in [0..nb-1], tb1 in [nb..2*nb-1])

    if (s.nalpha % 2 != 0 || s.nbeta < 2*s.nb) {
        std::cerr << "Invalid sizes: require nalpha even and nbeta >= 2*nb.\n";
        return 1;
    }

    // Host data
    std::vector<int> h_sa1, h_ta1, h_sb1, h_tb1;
    make_race_free_maps(s, h_sa1, h_ta1, h_sb1, h_tb1);

    std::vector<cuDoubleComplex> h_C, h_pa1, h_pa2, h_pb1, h_pb2;
    init_random_state(s, h_C, h_pa1, h_pa2, h_pb1, h_pb2);

    // Split parity into SoA
    std::vector<double> h_pa1r(s.nalpha), h_pa1i(s.nalpha),
                        h_pa2r(s.nalpha), h_pa2i(s.nalpha);
    for (int i = 0; i < s.nalpha; ++i) {
        h_pa1r[i] = cuCreal(h_pa1[i]); h_pa1i[i] = cuCimag(h_pa1[i]);
        h_pa2r[i] = cuCreal(h_pa2[i]); h_pa2i[i] = cuCimag(h_pa2[i]);
    }
    std::vector<double> h_pb1r(s.nb), h_pb1i(s.nb),
                        h_pb2r(s.nb), h_pb2i(s.nb);
    for (int i = 0; i < s.nb; ++i) {
        h_pb1r[i] = cuCreal(h_pb1[i]); h_pb1i[i] = cuCimag(h_pb1[i]);
        h_pb2r[i] = cuCreal(h_pb2[i]); h_pb2i[i] = cuCimag(h_pb2[i]);
    }

    // Scalars
    cuDoubleComplex factor_real_c = make_cuDoubleComplex(2.0, 0.0);
    cuDoubleComplex factor_imag_c = make_cuDoubleComplex(0.0, -1.5);
    cuDoubleComplex acc1_c = make_cuDoubleComplex(0.75, -0.25);
    cuDoubleComplex acc2_c = make_cuDoubleComplex(-0.4, 0.9);

    double factor_real = cuCreal(factor_real_c); // 2.0
    double factor_imag = cuCimag(factor_imag_c); // -1.5
    double acc1r = cuCreal(acc1_c), acc1i = cuCimag(acc1_c);
    double acc2r = cuCreal(acc2_c), acc2i = cuCimag(acc2_c);

    // Device allocations
    size_t N  = (size_t)s.nalpha * s.nbeta;
    size_t Cc = N * sizeof(cuDoubleComplex);
    size_t Cd = N * sizeof(double);

    cuDoubleComplex *d_C_ref1 = nullptr, *d_C_ref2 = nullptr;
    double *d_Cr = nullptr, *d_Ci = nullptr;

    int *d_sa1 = nullptr, *d_ta1 = nullptr, *d_sb1 = nullptr, *d_tb1 = nullptr;
    cuDoubleComplex *d_pa1 = nullptr, *d_pa2 = nullptr, *d_pb1 = nullptr, *d_pb2 = nullptr;
    double *d_pa1r = nullptr, *d_pa1i = nullptr, *d_pa2r = nullptr, *d_pa2i = nullptr;
    double *d_pb1r = nullptr, *d_pb1i = nullptr, *d_pb2r = nullptr, *d_pb2i = nullptr;

    CUDA_CHECK(cudaMalloc(&d_C_ref1, Cc));
    CUDA_CHECK(cudaMalloc(&d_C_ref2, Cc));
    CUDA_CHECK(cudaMalloc(&d_Cr,     Cd));
    CUDA_CHECK(cudaMalloc(&d_Ci,     Cd));

    CUDA_CHECK(cudaMalloc(&d_sa1, s.nalpha * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ta1, s.nalpha * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sb1, s.nb     * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tb1, s.nb     * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_pa1, s.nalpha * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_pa2, s.nalpha * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_pb1, s.nb     * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_pb2, s.nb     * sizeof(cuDoubleComplex)));

    CUDA_CHECK(cudaMalloc(&d_pa1r, s.nalpha * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pa1i, s.nalpha * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pa2r, s.nalpha * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pa2i, s.nalpha * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pb1r, s.nb     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pb1i, s.nb     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pb2r, s.nb     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pb2i, s.nb     * sizeof(double)));

    // Upload
    CUDA_CHECK(cudaMemcpy(d_C_ref1, h_C.data(), Cc, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_ref2, h_C.data(), Cc, cudaMemcpyHostToDevice));
    {
        std::vector<double> hr(N), hi(N);
        for (size_t k = 0; k < N; ++k) { hr[k] = cuCreal(h_C[k]); hi[k] = cuCimag(h_C[k]); }
        CUDA_CHECK(cudaMemcpy(d_Cr, hr.data(), Cd, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Ci, hi.data(), Cd, cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMemcpy(d_sa1, h_sa1.data(), s.nalpha * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ta1, h_ta1.data(), s.nalpha * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sb1, h_sb1.data(), s.nb     * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tb1, h_tb1.data(), s.nb     * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_pa1, h_pa1.data(), s.nalpha * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pa2, h_pa2.data(), s.nalpha * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pb1, h_pb1.data(), s.nb     * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pb2, h_pb2.data(), s.nb     * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_pa1r, h_pa1r.data(), s.nalpha * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pa1i, h_pa1i.data(), s.nalpha * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pa2r, h_pa2r.data(), s.nalpha * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pa2i, h_pa2i.data(), s.nalpha * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pb1r, h_pb1r.data(), s.nb     * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pb1i, h_pb1i.data(), s.nb     * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pb2r, h_pb2r.data(), s.nb     * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pb2i, h_pb2i.data(), s.nb     * sizeof(double), cudaMemcpyHostToDevice));

    // Launch configs
    dim3 block(BX, AY);
    dim3 grid((s.nb + BX - 1) / BX, (s.nalpha + AY - 1) / AY);

    // ---- Test 1: factor is real ----
    inplace_givens_update_complex_tiled<32><<<grid, block>>>(
        d_C_ref1,
        d_sa1, d_ta1,
        d_pa1, d_pa2,
        d_sb1, d_tb1,
        d_pb1, d_pb2,
        s.nalpha, s.nb, s.nbeta,
        factor_real_c, acc1_c, acc2_c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    givens_update_soa_tiled_factor_real<32><<<grid, block>>>(
        d_Cr, d_Ci,
        d_sa1, d_ta1,
        d_pa1r, d_pa1i, d_pa2r, d_pa2i,
        d_sb1, d_tb1,
        d_pb1r, d_pb1i, d_pb2r, d_pb2i,
        s.nalpha, s.nb, s.nbeta,
        factor_real, acc1r, acc1i, acc2r, acc2i);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download and compare (real case)
    std::vector<cuDoubleComplex> h_ref1(N);
    std::vector<double> h_Cr1(N), h_Ci1(N);
    CUDA_CHECK(cudaMemcpy(h_ref1.data(), d_C_ref1, Cc, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Cr1.data(),  d_Cr,     Cd, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Ci1.data(),  d_Ci,     Cd, cudaMemcpyDeviceToHost));

    auto printer = [](size_t k, double gr, double gi, double rr, double ri, const char* tag){
        if (tag) std::cerr << "[" << tag << "]";
        std::cerr << " Mismatch @" << k
                  << " got (" << gr << ", " << gi << ")"
                  << " vs ref (" << rr << ", " << ri << ")\n";
    };
    int mism_real = compare_with_tol(N, h_Cr1, h_Ci1, h_ref1, 1e-12, "REAL", printer);

    // Reset SoA (for fair imag test starting from same initial C)
    {
        std::vector<double> hr(N), hi(N);
        for (size_t k = 0; k < N; ++k) { hr[k] = cuCreal(h_C[k]); hi[k] = cuCimag(h_C[k]); }
        CUDA_CHECK(cudaMemcpy(d_Cr, hr.data(), Cd, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Ci, hi.data(), Cd, cudaMemcpyHostToDevice));
    }

    // ---- Test 2: factor is purely imaginary ----
    inplace_givens_update_complex_tiled<32><<<grid, block>>>(
        d_C_ref2,
        d_sa1, d_ta1,
        d_pa1, d_pa2,
        d_sb1, d_tb1,
        d_pb1, d_pb2,
        s.nalpha, s.nb, s.nbeta,
        factor_imag_c, acc1_c, acc2_c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    givens_update_soa_tiled_factor_imag<32><<<grid, block>>>(
        d_Cr, d_Ci,
        d_sa1, d_ta1,
        d_pa1r, d_pa1i, d_pa2r, d_pa2i,
        d_sb1, d_tb1,
        d_pb1r, d_pb1i, d_pb2r, d_pb2i,
        s.nalpha, s.nb, s.nbeta,
        factor_imag, acc1r, acc1i, acc2r, acc2i);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download and compare (imag case)
    std::vector<cuDoubleComplex> h_ref2(N);
    std::vector<double> h_Cr2(N), h_Ci2(N);
    CUDA_CHECK(cudaMemcpy(h_ref2.data(), d_C_ref2, Cc, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Cr2.data(),  d_Cr,     Cd, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Ci2.data(),  d_Ci,     Cd, cudaMemcpyDeviceToHost));

    int mism_imag = compare_with_tol(N, h_Cr2, h_Ci2, h_ref2, 1e-12, "IMAG", printer);

    if (mism_real == 0) std::cout << "GIVENS REAL test: PASS\n";
    else                std::cout << "GIVENS REAL test: FAIL (" << mism_real << " mismatches)\n";

    if (mism_imag == 0) std::cout << "GIVENS IMAG test: PASS\n";
    else                std::cout << "GIVENS IMAG test: FAIL (" << mism_imag << " mismatches)\n";

    // Cleanup
    cudaFree(d_C_ref1); cudaFree(d_C_ref2);
    cudaFree(d_Cr); cudaFree(d_Ci);
    cudaFree(d_sa1); cudaFree(d_ta1); cudaFree(d_sb1); cudaFree(d_tb1);
    cudaFree(d_pa1); cudaFree(d_pa2); cudaFree(d_pb1); cudaFree(d_pb2);
    cudaFree(d_pa1r); cudaFree(d_pa1i); cudaFree(d_pa2r); cudaFree(d_pa2i);
    cudaFree(d_pb1r); cudaFree(d_pb1i); cudaFree(d_pb2r); cudaFree(d_pb2i);

    return (mism_real == 0 && mism_imag == 0) ? 0 : 1;
}
