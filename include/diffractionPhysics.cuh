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
                                     const int dimy_padded);

void CUDAenvelopeConstruction(dim3 threads,
                              dim3 blocks,
                              void* GPU_intensity,
                              void* GPU_phase,
                              void* GPU_envelope,
                              const int dimx,
                              const int dimy,
                              const int dimx_padded,
                              const int dimy_padded);

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
                              const int dimy_padded);

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
