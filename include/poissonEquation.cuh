// Logging

#include <cufft.h>
#include <stdio.h>

#define EXECUDA(INF) _assert_CUDA(INF, __FILE__, __LINE__)
#define EXECUFFT(INF) _assert_CUFFT(INF, __FILE__, __LINE__)
#define PI 3.14159265358979323846

__global__ void spectralDivision(float2* __restrict__ x,
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
                          const float pixel_size_y);
__global__ void
dirichletExtension(float* __restrict__ f, float* __restrict__ fx, const int SIZEX, const int SIZEY);

void CUDADirichletExtension(
    dim3 threads, dim3 blocks, void* GPU_f, void* GPU_extendedf, const int SIZEX, const int SIZEY);

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
