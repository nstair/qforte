#include "blas_math.h"
#include <complex>

extern "C" {
  // Include the actual BLAS library header
  #include <cblas.h>
}

void math_daxpy(
  const int n, 
  const double alpha, 
  const double* x, 
  const int incx, 
  double* y, 
  const int incy) 
{
  // Call the BLAS DAXPY function
  cblas_daxpy(n, alpha, x, incx, y, incy);
}

void math_zaxpy(
    const int n,
    const std::complex<double> alpha,
    const std::complex<double>* x,
    const int incx,
    std::complex<double>* y,
    const int incy) 
{   
    
    // Call the BLAS ZAXPY function
    cblas_zaxpy(
      n, 
      &alpha, 
      reinterpret_cast<const openblas_complex_double*>(x), 
      incx, 
      reinterpret_cast<openblas_complex_double*>(y), 
      incy);
}

void math_zscale(
    const int n,
    const std::complex<double> alpha,
    std::complex<double>* x,
    const int incx)
{
    // Call the CBLAS ZSCALE function
    cblas_zscal(
      n, 
      &alpha, 
      reinterpret_cast<openblas_complex_double*>(x), 
      incx);
}

void math_zgemm(
    const char transa,
    const char transb,
    const int M,
    const int N,
    const int K,
    const std::complex<double> alpha,
    const std::complex<double>* A,
    const int lda,
    const std::complex<double>* B,
    const int ldb,
    const std::complex<double> beta,
    std::complex<double>* C,
    const int ldc)
{
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;

    if(transa == 'T'){ transA = CblasTrans; }
    if(transa == 'C'){ transA = CblasConjTrans; }

    if(transb == 'T'){ transB = CblasTrans; }
    if(transb == 'C'){ transB = CblasConjTrans; }

    cblas_zgemm(
      CblasColMajor, 
      transA, 
      transB, 
      M, 
      N, 
      K, 
      reinterpret_cast<const openblas_complex_double*>(&alpha), 
      reinterpret_cast<const openblas_complex_double*>(A), 
      lda, 
      reinterpret_cast<const openblas_complex_double*>(B), 
      ldb, 
      reinterpret_cast<const openblas_complex_double*>(&beta), 
      reinterpret_cast<openblas_complex_double*>(C), 
      ldc);
}