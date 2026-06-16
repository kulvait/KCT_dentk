#pragma once

#include <cufft.h>
#include <math_constants.h>
#include <stdio.h>

#ifndef EXECUDA
#define EXECUDA(INF) _assert_CUDA(INF, __FILE__, __LINE__)
#endif
#ifndef EXECUFFT
#define EXECUFFT(INF) _assert_CUFFT(INF, __FILE__, __LINE__)
#endif

//Tested to be good double representation of these values, FOURPISQUARED=4*pi*pi
#define PI 3.141592653589793
#define PISQUARED 9.869604401089358
#define TWOPI 6.283185307179586
#define FOURPISQUARED 39.47841760435743
#define MINUSTWOPISQUARED -19.739208802178716

// -----------------------------------------------------------------------------
// Existing projection-domain and spectral filters
// -----------------------------------------------------------------------------

void CUDARadonFilter(dim3 threads,
                     dim3 blocks,
                     void* x,
                     const int SIZEX,
                     const int SIZEY,
                     const float pixel_size_x,
                     const bool ifftshift = false);

void CUDAParkerFilter(dim3 threads,
                      dim3 blocks,
                      void* x,
                      const int SIZEX,
                      const int SIZEY,
                      const float corpos,
                      const float zslope);

// Multiply Hermitian-packed R2C spectra by a real 1D filter of length SIZEX_FULL/2+1.
// The filter is applied independently to each of SIZEY rows.
// Includes scaling by 1/(SIZEX_FULL * pixel_size_x).
template <typename T, typename W>
void CUDASpectralFilter(dim3 threads,
                        void* GPU_f_in,
                        void* GPU_filter_in,
                        const int SIZEX_FULL,
                        const int SIZEY,
                        const T pixel_size_x);

// -----------------------------------------------------------------------------
// Spatial extension / restriction helpers
// -----------------------------------------------------------------------------


void CUDAconstantMultiplication(dim3 threads,
                                dim3 blocks,
                                void* x,
                                const float factor,
                                const int SIZEX,
                                const int SIZEY,
                                const int TOX,
                                const int TOY);

// -----------------------------------------------------------------------------
// Shift / roll helpers
// -----------------------------------------------------------------------------

void CUDARoll(dim3 threads,
              dim3 blocks,
              void* x_in,
              void* x_out,
              const int shift,
              const int SIZEX,
              const int SIZEY);

void CUDAifftshift(dim3 threads,
                   dim3 blocks,
                   void* x_in,
                   void* x_out,
                   const int SIZEX,
                   const int SIZEY);

void CUDAfftshift(dim3 threads,
                  dim3 blocks,
                  void* x_in,
                  void* x_out,
                  const int SIZEX,
                  const int SIZEY);

void CUDAifftshiftSpectral(dim3 threads,
                           dim3 blocks,
                           void* x,
                           const int SIZEX,
                           const int SIZEY);

// -----------------------------------------------------------------------------
// Ramp / Ram-Lak filter construction
// -----------------------------------------------------------------------------

/**
 * Construct the spatial-domain Ram-Lak convolution kernel directly on the GPU.
 *
 * The output buffer must have SIZE elements:
 *
 *   float  version:  SIZE * sizeof(float)
 *   double version:  SIZE * sizeof(double)
 *
 * This kernel is intended to be transformed by CUFFT R2C/D2Z if the exact
 * frequency response of the discrete Ram-Lak spatial kernel is desired.
 */
template <typename T>
void CUDARamLakKernel1D(void* x, const int SIZE);

/**
 * Construct an ideal ramp response directly in R2C/D2Z half-spectrum Hermitian packed layout layout.
 *
 * SIZE_FULL is the full FFT length, for example NX.
 * SIZE_HERMITIAN is the stored real-to-complex half-spectrum length:
 *
 *   SIZE_HERMITIAN = SIZE_FULL / 2 + 1
 *
 * The output buffer must have SIZE_HERMITIAN elements:
 */
template <typename T>
void CUDAIdealRamp1D(void* OUT_PACKED, const int SIZE_FULL);

// -----------------------------------------------------------------------------
// Ramp window functions
// -----------------------------------------------------------------------------

/**
 * Apply frequency-domain windows to a real-valued half-spectrum filter.
 *
 * The filter buffer must contain SIZE_HERMITIAN real values corresponding to
 * nonnegative R2C/D2Z frequency indices:
 *
 *   k = 0, 1, ..., SIZE_FULL / 2
 *
 * SIZE_FULL is the original FFT length.
 */
void CUDAApplySheppLoganWindow1DFloat(void* filter,
                                      const int SIZE_HERMITIAN,
                                      const int SIZE_FULL);

void CUDAApplySheppLoganWindow1DDouble(void* filter,
                                       const int SIZE_HERMITIAN,
                                       const int SIZE_FULL);

void CUDAApplyCosineWindow1DFloat(void* filter,
                                  const int SIZE_HERMITIAN,
                                  const int SIZE_FULL);

void CUDAApplyCosineWindow1DDouble(void* filter,
                                   const int SIZE_HERMITIAN,
                                   const int SIZE_FULL);

void CUDAApplyHanningWindow1DFloat(void* filter,
                                   const int SIZE_HERMITIAN,
                                   const int SIZE_FULL);

void CUDAApplyHanningWindow1DDouble(void* filter,
                                    const int SIZE_HERMITIAN,
                                    const int SIZE_FULL);

void CUDAApplyHammingWindow1DFloat(void* filter,
                                   const int SIZE_HERMITIAN,
                                   const int SIZE_FULL);

void CUDAApplyHammingWindow1DDouble(void* filter,
                                    const int SIZE_HERMITIAN,
                                    const int SIZE_FULL);

void CUDAApplyKaiserWindow1DFloat(void* filter,
                                  const int SIZE_HERMITIAN,
                                  const int SIZE_FULL,
                                  const float beta);

void CUDAApplyKaiserWindow1DDouble(void* filter,
                                   const int SIZE_HERMITIAN,
                                   const int SIZE_FULL,
                                   const double beta);

// -----------------------------------------------------------------------------
// CUFFT helper kernels
// -----------------------------------------------------------------------------

/**
 * Extract the real part from a CUFFT R2C/D2Z half-spectrum.
 *
 * Input:
 *   float version:  cufftComplex*
 *   double version: cufftDoubleComplex*
 *
 * Output:
 *   float version:  float*
 *   double version: double*
 *
 * SIZE is the number of half-spectrum elements, usually:
 *
 *   SIZE = SIZE_FULL / 2 + 1
 */
void CUDAExtractRealFFTFloat(void* in, void* out, const int SIZE);
void CUDAExtractRealFFTDouble(void* in, void* out, const int SIZE);

// -----------------------------------------------------------------------------
// Spectral Gaussian blur template wrappers
// -----------------------------------------------------------------------------

/**
 * Template argument W is expected to be cufftComplex or cufftDoubleComplex.
 *
 * These functions are implemented and explicitly instantiated in
 * tomographicFiltering.cu for:
 *
 *   CUDASpectralGaussianBlur1D<float,  cufftComplex>
 *   CUDASpectralGaussianBlur1D<double, cufftDoubleComplex>
 *   CUDASpectralGaussianBlur2D<float,  cufftComplex>
 *   CUDASpectralGaussianBlur2D<double, cufftDoubleComplex>
 */
template <typename T, typename W>
void CUDASpectralGaussianBlur1D(dim3 threads,
                                void* GPU_vec,
                                const int SIZEX,
                                const int SIZEY,
                                const T sigma_z);

template <typename T, typename W>
void CUDASpectralGaussianBlur2D(dim3 threads,
                                void* GPU_vec,
                                const int SIZEX,
                                const int SIZEY,
                                const T sigma_x,
                                const T sigma_y);
