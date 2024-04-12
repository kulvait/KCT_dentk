// Logging

#include <cufft.h>
#include <stdio.h>

#define EXECUDA(INF) _assert_CUDA(INF, __FILE__, __LINE__)
#define EXECUFFT(INF) _assert_CUFFT(INF, __FILE__, __LINE__)
//Tested to be good double representation of these values, FOURPISQUARED=4*pi*pi
#define PI 3.141592653589793
#define TWOPI 6.283185307179586
#define FOURPISQUARED 39.47841760435743

__global__ void RadonFilter(float2* __restrict__ x,
                            const int SIZEX,
                            const int SIZEY,
                            const float pixel_size_x,
                            const bool ifftshift);

void CUDARadonFilter(dim3 threads,
                     dim3 blocks,
                     void* x,
                     const int SIZEX,
                     const int SIZEY,
                     const float pixel_size_x,
                     const bool ifftshift = false);

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

__global__ void roll(float* __restrict__ x_in,
                     float* __restrict__ x_out,
                     const int shift,
                     const int SIZEX,
                     const int SIZEY);

void CUDARoll(dim3 threads,
              dim3 blocks,
              void* x_in,
              void* x_out,
              const int shift,
              const int SIZEX,
              const int SIZEY);

void CUDAifftshift(
    dim3 threads, dim3 blocks, void* x_in, void* x_out, const int SIZEX, const int SIZEY);

void CUDAfftshift(
    dim3 threads, dim3 blocks, void* x_in, void* x_out, const int SIZEX, const int SIZEY);

__global__ void ifftshiftSpectral(float2* __restrict__ x, const int SIZEX, const int SIZEY);

void CUDAifftshiftSpectral(dim3 threads, dim3 blocks, void* x, const int SIZEX, const int SIZEY);

__global__ void ZeroPad(float* __restrict__ IN,
                        float* __restrict__ OUT,
                        const int SIZEX,
                        const int SIZEXPAD,
                        const int SIZEY);

void CUDAZeroPad(dim3 threads,
                 dim3 blocks,
                 void* GPU_in,
                 void* GPU_out,
                 const int SIZEX,
                 const int SIZEXPAD,
                 const int SIZEY);

__global__ void StripPad(float* __restrict__ IN,
                         float* __restrict__ OUT,
                         const int SIZEX,
                         const int SIZEXPAD,
                         const int SIZEY);

void CUDAStripPad(dim3 threads,
                  dim3 blocks,
                  void* GPU_IN,
                  void* GPU_OUT,
                  const int SIZEX,
                  const int SIZEXPAD,
                  const int SIZEY);

__global__ void SymmPad(float* __restrict__ IN,
                        float* __restrict__ OUT,
                        const int SIZEX,
                        const int SIZEXPAD,
                        const int SIZEY);

void CUDASymmPad(dim3 threads,
                 dim3 blocks,
                 void* GPU_in,
                 void* GPU_out,
                 const int SIZEX,
                 const int SIZEXPAD,
                 const int SIZEY);
