#include <cufft.h>
#include <stdio.h>

#define EXECUDA(INF) _assert_CUDA(INF, __FILE__, __LINE__)
#define EXECUFFT(INF) _assert_CUFFT(INF, __FILE__, __LINE__)
#define PI 3.1415926535897931

__global__ void envelopeConstruction(float* __restrict__ GPU_intensity,
                                     float* __restrict__ GPU_phase,
                                     float2* __restrict__ GPU_envelope,
                                     const int dimx,
                                     const int dimy,
                                     const int dimx_padded,
                                     const int dimy_padded,
                                     const bool paddingXSymmetric,
                                     const bool paddingYSymmetric);

void CUDAenvelopeConstruction(dim3 threads,
                              dim3 blocks,
                              void* GPU_intensity,
                              void* GPU_phase,
                              void* GPU_envelope,
                              const int dimx,
                              const int dimy,
                              const int dimx_padded,
                              const int dimy_padded,
                              const bool paddingXSymmetric,
                              const bool paddingYSymmetric);

__global__ void envelopeDecomposition(float* __restrict__ GPU_intensity,
                                      float* __restrict__ GPU_phase,
                                      float2* __restrict__ GPU_envelope,
                                      const int dimx,
                                      const int dimy,
                                      const int dimx_padded,
                                      const int dimy_padded);

void CUDAenvelopeDecomposition(dim3 threads,
                               dim3 blocks,
                               void* GPU_intensity,
                               void* GPU_phase,
                               void* GPU_envelope,
                               const int dimx,
                               const int dimy,
                               const int dimx_padded,
                               const int dimy_padded,
                               bool normalize = true);

__global__ void spectralMultiplicationFresnel(float* __restrict__ GPU_FTenvelope,
                                              const float lambda,
                                              const float propagationDistance,
                                              const int dimx_padded,
                                              const int dimy_padded,
                                              const float pixel_size_x,
                                              const float pixel_size_y);

void CUDAspectralMultiplicationFresnel(dim3 threads,
                                       dim3 blocks,
                                       void* GPU_FTenvelope,
                                       const float lambda,
                                       const float propagationDistance,
                                       const int dimx_padded,
                                       const int dimy_padded,
                                       const float pixel_size_x,
                                       const float pixel_size_y);

__global__ void spectralMultiplicationRayleigh(float* __restrict__ GPU_FTenvelope,
                                               const float lambda,
                                               const float propagationDistance,
                                               const int dimx_padded,
                                               const int dimy_padded,
                                               const float pixel_size_x,
                                               const float pixel_size_y);

void CUDAspectralMultiplicationRayleigh(dim3 threads,
                                        dim3 blocks,
                                        void* GPU_FTenvelope,
                                        const float lambda,
                                        const float propagationDistance,
                                        const int dimx_padded,
                                        const int dimy_padded,
                                        const float pixel_size_x,
                                        const float pixel_size_y);

__global__ void exportKernelFresnel(float* __restrict__ GPU_kernel_re,
                                    float* __restrict__ GPU_kernel_im,
                                    const float lambda,
                                    const float propagationDistance,
                                    const int dimx,
                                    const int dimy,
                                    const float pixel_size_x,
                                    const float pixel_size_y);

void CUDAexportKernelFresnel(dim3 threads,
                             dim3 blocks,
                             void* GPU_kernel_re,
                             void* GPU_kernel_im,
                             const float lambda,
                             const float propagationDistance,
                             const int dimx_padded,
                             const int dimy_padded,
                             const float pixel_size_x,
                             const float pixel_size_y);

__global__ void exportKernelRayleigh(float* __restrict__ GPU_kernel_re,
                                     float* __restrict__ GPU_kernel_im,
                                     const float lambda,
                                     const float propagationDistance,
                                     const int dimx,
                                     const int dimy,
                                     const float pixel_size_x,
                                     const float pixel_size_y);

void CUDAexportKernelRayleigh(dim3 threads,
                              dim3 blocks,
                              void* GPU_kernel_re,
                              void* GPU_kernel_im,
                              const float lambda,
                              const float propagationDistance,
                              const int dimx_padded,
                              const int dimy_padded,
                              const float pixel_size_x,
                              const float pixel_size_y);
