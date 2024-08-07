#pragma once

//The padding.cuh header file encapsulates the CUDA kernel functions for
//zero padding, symmetric padding, and antisymmetric padding, along with their host function wrappers.

#include <cuda_runtime.h>
#include <stdio.h>

// Macro for checking CUDA errors following a CUDA launch or API call
#ifndef EXECUDA
#define EXECUDA(INF) _assert_CUDA(INF, __FILE__, __LINE__)
#endif

// Templated CUDA kernel function for zero padding (1D)
template <typename T>
__global__ void ZeroPad(
    T* __restrict__ IN, T* __restrict__ OUT, const int SIZEX, const int SIZEXPAD, const int SIZEY);

// Templated CUDA kernel function for zero padding (2D)
template <typename T>
__global__ void ZeroPad2D(T* __restrict__ IN,
                          T* __restrict__ OUT,
                          const int SIZEX,
                          const int SIZEY,
                          const int SIZEXPAD,
                          const int SIZEYPAD);

// Templated CUDA kernel function for symmetric padding (1D)
template <typename T>
__global__ void SymmPad(
    T* __restrict__ IN, T* __restrict__ OUT, const int SIZEX, const int SIZEXPAD, const int SIZEY);
// Templated CUDA kernel function for symmetric padding (2D)
template <typename T>
__global__ void SymmPad2D(T* __restrict__ IN,
                          T* __restrict__ OUT,
                          const int SIZEX,
                          const int SIZEY,
                          const int SIZEXPAD,
                          const int SIZEYPAD);

// Templated CUDA kernel function for antisymmetric padding (1D)
template <typename T>
__global__ void AsymmPad(
    T* __restrict__ IN, T* __restrict__ OUT, const int SIZEX, const int SIZEXPAD, const int SIZEY);

// Templated CUDA kernel function for antisymmetric padding (2D)
template <typename T>
__global__ void AsymmPad2D(T* __restrict__ IN,
                           T* __restrict__ OUT,
                           const int SIZEX,
                           const int SIZEY,
                           const int SIZEXPAD,
                           const int SIZEYPAD);
// Templated CUDA kernel function for removal of padding
// 2D version is unnecessary as SIZEYPAD exceeds SIZEY
template <typename T>
__global__ void RemovePadding(
    T* __restrict__ IN, T* __restrict__ OUT, const int SIZEX, const int SIZEY, const int SIZEXPAD);
// Host function wrapper for zero padding (1D)
template <typename T>
void CUDAZeroPad(dim3 threads,
                 void* GPU_in,
                 void* GPU_out,
                 const int SIZEX,
                 const int SIZEXPAD,
                 const int SIZEY);

// Host function wrapper for zero padding (2D)
template <typename T>
void CUDAZeroPad2D(dim3 threads,
                   void* GPU_in,
                   void* GPU_out,
                   const int SIZEX,
                   const int SIZEY,
                   const int SIZEXPAD,
                   const int SIZEYPAD);

// Host function wrapper for symmetric padding (1D)
template <typename T>
void CUDASymmPad(dim3 threads,
                 void* GPU_in,
                 void* GPU_out,
                 const int SIZEX,
                 const int SIZEXPAD,
                 const int SIZEY);

// Host function wrapper for symmetric padding (2D)
template <typename T>
void CUDASymmPad2D(dim3 threads,
                   void* GPU_in,
                   void* GPU_out,
                   const int SIZEX,
                   const int SIZEY,
                   const int SIZEXPAD,
                   const int SIZEYPAD);

// Host function wrapper for antisymmetric padding (1D)
template <typename T>
void CUDAAsymmPad(dim3 threads,
                  void* GPU_in,
                  void* GPU_out,
                  const int SIZEX,
                  const int SIZEXPAD,
                  const int SIZEY);

// Host function wrapper for antisymmetric padding (2D)
template <typename T>
void CUDAAsymmPad2D(dim3 threads,
                    void* GPU_in,
                    void* GPU_out,
                    const int SIZEX,
                    const int SIZEY,
                    const int SIZEXPAD,
                    const int SIZEYPAD);

// Host function wrapper for removal of padding
template <typename T>
void CUDARemovePadding(dim3 threads,
                       void* GPU_in,
                       void* GPU_out,
                       const int SIZEX,
                       const int SIZEY,
                       const int SIZEXPAD);

// Kernel to apply antisymmetric padding with Dirichlet boundary condition on both x and y axes.
// This sets the edges at PX == 0 or PY == 0 to 0.
template <typename T>
__global__ void AsymmPadDirichlet2D(T* __restrict__ IN,
                                    T* __restrict__ OUT,
                                    const int SIZEX,
                                    const int SIZEY,
                                    const int SIZEXPAD,
                                    const int SIZEYPAD);

// Wrapper function to launch the AsymmPadDirichlet2D kernel.
// Sets edges at PX == 0 or PY == 0 to 0 to maintain antisymmetry for FFT.
template <typename T>
void CUDAAsymmPadDirichlet2D(dim3 threads,
                             void* GPU_in,
                             void* GPU_out,
                             const int SIZEX,
                             const int SIZEY,
                             const int SIZEXPAD,
                             const int SIZEYPAD);

// Kernel to apply antisymmetric padding with Dirichlet boundary condition on the x-axis only.
// This sets the left edge at PX == 0 to 0.
template <typename T>
__global__ void AsymmPadDirichlet(
    T* __restrict__ IN, T* __restrict__ OUT, const int SIZEX, const int SIZEXPAD, const int SIZEY);

// Wrapper function to launch the AsymmPadDirichlet_x kernel.
// Sets left edge at PX == 0 to 0 to maintain antisymmetry in the x direction.
template <typename T>
void CUDAAsymmPadDirichlet(dim3 threads,
                           void* GPU_in,
                           void* GPU_out,
                           const int SIZEX,
                           const int SIZEXPAD,
                           const int SIZEY);

dim3 getNumBlocks(dim3 threads, int SIZEX, int SIZEY);

