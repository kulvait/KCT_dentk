// Logging

#include <cufft.h>
#include <stdio.h>

#define EXECUDA(INF) _assert_CUDA(INF, __FILE__, __LINE__)
#define EXECUFFT(INF) _assert_CUFFT(INF, __FILE__, __LINE__)
#define PI 3.1415926535897931
#define FOURPISQUARED 39.478417604357432

__global__ void spectralDivision(float2* __restrict__ x,
                                 const int SIZEX,
                                 const int SIZEY,
                                 const float pixel_size_x,
                                 const float pixel_size_y);

__global__ void spectralDivisionRegularized(float2* __restrict__ x,
                                            const float epsilon,
                                            const int SIZEX,
                                            const int SIZEY,
                                            const float pixel_size_x,
                                            const float pixel_size_y);

void CUDAspectralDivision(dim3 threads,
                          dim3 blocks,
                          void* x,
                          const int SIZEX,
                          const int SIZEY,
                          const float pixel_size_x,
                          const float pixel_size_y,
                          const float epsilon = 0.0f);

__global__ void spectralMultiplication(float2* __restrict__ x,
                                       const int SIZEX,
                                       const int SIZEY,
                                       const float pixel_size_x,
                                       const float pixel_size_y);

void CUDAspectralMultiplication(dim3 threads,
                                dim3 blocks,
                                void* x,
                                const int SIZEX,
                                const int SIZEY,
                                const float pixel_size_x,
                                const float pixel_size_y);
__global__ void
dirichletExtension(float* __restrict__ f, float* __restrict__ fx, const int SIZEX, const int SIZEY);

void CUDADirichletExtension(
    dim3 threads, dim3 blocks, void* GPU_f, void* GPU_extendedf, const int SIZEX, const int SIZEY);

__global__ void constantMultiplication(float* __restrict__ x,
                                       const float factor,
                                       const int SIZEX,
                                       const int SIZEY,
                                       const int TOX,
                                       const int TOY);

void CUDAconstantMultiplication(dim3 threads,
                                dim3 blocks,
                                void* x,
                                const float factor,
                                const int SIZEX,
                                const int SIZEY,
                                const int TOX,
                                const int TOY);
__global__ void
neumannExtension(float* __restrict__ f, float* __restrict__ fx, const int SIZEX, const int SIZEY);

void CUDANeumannExtension(
    dim3 threads, dim3 blocks, void* GPU_f, void* GPU_extendedf, const int SIZEX, const int SIZEY);

__global__ void functionRestriction(float* __restrict__ fx,
                                    float* __restrict__ f,
                                    const int SIZEX,
                                    const int SIZEY);

void CUDAFunctionRestriction(
    dim3 threads, dim3 blocks, void* GPU_extendedf, void* GPU_f, const int SIZEX, const int SIZEY);

template <typename T>
__global__ void RegularizedSpectralDivisionHermitian(T* __restrict__ xFourier, 
                                                     const T epsilon, 
                                                     const int SIZEX, 
                                                     const int SIZEY, 
                                                     const T pixelSizeX, 
                                                     const T pixelSizeY);

template <typename T>
void CUDAspectralDivisionHermitian(dim3 threads, 
                                   void* GPU_xFourier, 
                                   const int SIZEX, 
                                   const int SIZEY, 
                                   const T pixelSizeX, 
                                   const T pixelSizeY, 
                                   const T epsilon);
