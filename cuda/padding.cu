#include "padding.cuh"

dim3 getNumBlocks(dim3 threads, int SIZEX, int SIZEY)
{
    return dim3((SIZEY + threads.x - 1) / threads.x, (SIZEX + threads.y - 1) / threads.y);
}

template <typename T>
__global__ void ZeroPad2D(T* __restrict__ IN,
                          T* __restrict__ OUT,
                          const int SIZEX,
                          const int SIZEY,
                          const int SIZEXPAD,
                          const int SIZEYPAD)
{
    const int PY = threadIdx.x + blockIdx.x * blockDim.x; // y-dimension
    const int PX = threadIdx.y + blockIdx.y * blockDim.y; // x-dimension

    if(PX >= SIZEXPAD || PY >= SIZEYPAD)
        return;

    const int IDXPAD = SIZEXPAD * PY + PX;

    if(PX >= SIZEX || PY >= SIZEY)
    {
        OUT[IDXPAD] = 0.0;
    } else
    {
        const int IDX = SIZEX * PY + PX;
        OUT[IDXPAD] = IN[IDX];
    }
}

template <typename T>
void CUDAZeroPad2D(dim3 threads,
                   void* GPU_in,
                   void* GPU_out,
                   const int SIZEX,
                   const int SIZEY,
                   const int SIZEXPAD,
                   const int SIZEYPAD)
{
    printf("CUDAZeroPad2D threads=(%d, %d, %d), SIZEX=%d, SIZEY=%d, SIZEXPAD=%d, SIZEYPAD=%d\n",
           threads.x, threads.y, threads.z, SIZEX, SIZEY, SIZEXPAD, SIZEYPAD);

    // Calculate the number of blocks needed
    dim3 numBlocks = getNumBlocks(threads, SIZEXPAD, SIZEYPAD);

    ZeroPad2D<T><<<numBlocks, threads>>>((T*)GPU_in, (T*)GPU_out, SIZEX, SIZEY, SIZEXPAD, SIZEYPAD);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch ZeroPad2D kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template <typename T>
__global__ void ZeroPad(
    T* __restrict__ IN, T* __restrict__ OUT, const int SIZEX, const int SIZEXPAD, const int SIZEY)
{
    const int PY = threadIdx.x + blockIdx.x * blockDim.x; // y-dimension
    const int PX = threadIdx.y + blockIdx.y * blockDim.y; // x-dimension

    if(PX >= SIZEXPAD || PY >= SIZEY)
        return;

    const int IDX = SIZEX * PY + PX;
    const int IDXPAD = SIZEXPAD * PY + PX;

    if(PX >= SIZEX)
    {
        OUT[IDXPAD] = 0.0;
    } else
    {
        OUT[IDXPAD] = IN[IDX];
    }
}

template <typename T>
void CUDAZeroPad(
    dim3 threads, void* GPU_in, void* GPU_out, const int SIZEX, const int SIZEXPAD, const int SIZEY)
{
    printf("CUDAZeroPad threads=(%d, %d, %d), SIZEX=%d, SIZEXPAD=%d, SIZEY=%d\n", threads.x,
           threads.y, threads.z, SIZEX, SIZEXPAD, SIZEY);

    // Calculate the number of blocks needed
    dim3 numBlocks = getNumBlocks(threads, SIZEXPAD, SIZEY);

    ZeroPad<T><<<numBlocks, threads>>>((T*)GPU_in, (T*)GPU_out, SIZEX, SIZEXPAD, SIZEY);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch ZeroPad kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template <typename T>
__global__ void SymmPad(
    T* __restrict__ IN, T* __restrict__ OUT, const int SIZEX, const int SIZEXPAD, const int SIZEY)
{
    const int PY = threadIdx.x + blockIdx.x * blockDim.x; // y-dimension
    const int PX = threadIdx.y + blockIdx.y * blockDim.y; // x-dimension

    if(PX >= SIZEXPAD || PY >= SIZEY)
        return;

    int IDX = SIZEX * PY + PX;
    int IDXPAD = SIZEXPAD * PY + PX;

    // Reflect the index PX to handle symmetric padding
    int PX_reflected = PX % (2 * SIZEX - 2);
    if(PX_reflected >= SIZEX)
    {
        PX_reflected = 2 * SIZEX - 2 - PX_reflected;
    }

    IDX = SIZEX * PY + PX_reflected;
    OUT[IDXPAD] = IN[IDX];
}

template <typename T>
void CUDASymmPad(
    dim3 threads, void* GPU_in, void* GPU_out, const int SIZEX, const int SIZEXPAD, const int SIZEY)
{
    printf("CUDASymmPad threads=(%d, %d, %d), SIZEX=%d, SIZEXPAD=%d, SIZEY=%d\n", threads.x,
           threads.y, threads.z, SIZEX, SIZEXPAD, SIZEY);

    // Calculate the number of blocks needed
    dim3 numBlocks = getNumBlocks(threads, SIZEXPAD, SIZEY);

    SymmPad<T><<<numBlocks, threads>>>((T*)GPU_in, (T*)GPU_out, SIZEX, SIZEXPAD, SIZEY);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch SymmPad kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template <typename T>
__global__ void SymmPad2D(T* __restrict__ IN,
                          T* __restrict__ OUT,
                          const int SIZEX,
                          const int SIZEY,
                          const int SIZEXPAD,
                          const int SIZEYPAD)
{
    const int PY = threadIdx.x + blockIdx.x * blockDim.x; // y-dimension
    const int PX = threadIdx.y + blockIdx.y * blockDim.y; // x-dimension

    if(PX >= SIZEXPAD || PY >= SIZEYPAD)
        return;

    int IDXPAD = SIZEXPAD * PY + PX;

    // Reflect the index PX and PY to handle symmetric padding
    int PX_reflected = PX % (2 * SIZEX - 2);
    if(PX_reflected >= SIZEX)
    {
        PX_reflected = 2 * SIZEX - 2 - PX_reflected;
    }

    int PY_reflected = PY % (2 * SIZEY - 2);
    if(PY_reflected >= SIZEY)
    {
        PY_reflected = 2 * SIZEY - 2 - PY_reflected;
    }

    int IDX = SIZEX * PY_reflected + PX_reflected;
    OUT[IDXPAD] = IN[IDX];
}

template <typename T>
void CUDASymmPad2D(dim3 threads,
                   void* GPU_in,
                   void* GPU_out,
                   const int SIZEX,
                   const int SIZEY,
                   const int SIZEXPAD,
                   const int SIZEYPAD)
{
    printf("CUDASymmPad2D threads=(%d, %d, %d), SIZEX=%d, SIZEY=%d, SIZEXPAD=%d, SIZEYPAD=%d\n",
           threads.x, threads.y, threads.z, SIZEX, SIZEY, SIZEXPAD, SIZEYPAD);

    // Calculate the number of blocks needed
    dim3 numBlocks = getNumBlocks(threads, SIZEXPAD, SIZEYPAD);

    SymmPad2D<T><<<numBlocks, threads>>>((T*)GPU_in, (T*)GPU_out, SIZEX, SIZEY, SIZEXPAD, SIZEYPAD);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch SymmPad2D kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template <typename T>
__global__ void AsymmPad(
    T* __restrict__ IN, T* __restrict__ OUT, const int SIZEX, const int SIZEXPAD, const int SIZEY)
{
    const int PY = threadIdx.x + blockIdx.x * blockDim.x; // y-dimension
    const int PX = threadIdx.y + blockIdx.y * blockDim.y; // x-dimension

    if(PX >= SIZEXPAD || PY >= SIZEY)
        return;

    int IDX = SIZEX * PY + PX;
    int IDXPAD = SIZEXPAD * PY + PX;

    // Reflect the index PX to handle antisymmetric padding
    int PX_reflected = PX % (2 * SIZEX - 2);
    bool invert_sign = false;
    if(PX_reflected >= SIZEX)
    {
        PX_reflected = 2 * SIZEX - 2 - PX_reflected;
        invert_sign = true;
    }

    IDX = SIZEX * PY + PX_reflected;
    OUT[IDXPAD] = invert_sign ? -IN[IDX] : IN[IDX];
}

template <typename T>
void CUDAAsymmPad(
    dim3 threads, void* GPU_in, void* GPU_out, const int SIZEX, const int SIZEXPAD, const int SIZEY)
{
    printf("CUDAAsymmPad threads=(%d, %d, %d), SIZEX=%d, SIZEXPAD=%d, SIZEY=%d\n", threads.x,
           threads.y, threads.z, SIZEX, SIZEXPAD, SIZEY);

    // Calculate the number of blocks needed
    dim3 numBlocks = getNumBlocks(threads, SIZEXPAD, SIZEY);

    AsymmPad<T><<<numBlocks, threads>>>((T*)GPU_in, (T*)GPU_out, SIZEX, SIZEXPAD, SIZEY);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch AsymmPad kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template <typename T>
__global__ void AsymmPad2D(T* __restrict__ IN,
                           T* __restrict__ OUT,
                           const int SIZEX,
                           const int SIZEY,
                           const int SIZEXPAD,
                           const int SIZEYPAD)
{
    const int PY = threadIdx.x + blockIdx.x * blockDim.x; // y-dimension
    const int PX = threadIdx.y + blockIdx.y * blockDim.y; // x-dimension

    if(PX >= SIZEXPAD || PY >= SIZEYPAD)
        return;

    int IDXPAD = SIZEXPAD * PY + PX;

    // Reflect the index PX and PY to handle antisymmetric padding
    int PX_reflected = PX % (2 * SIZEX - 2);
    bool invert_sign_x = false;
    if(PX_reflected >= SIZEX)
    {
        PX_reflected = 2 * SIZEX - 2 - PX_reflected;
        invert_sign_x = true;
    }

    int PY_reflected = PY % (2 * SIZEY - 2);
    bool invert_sign_y = false;
    if(PY_reflected >= SIZEY)
    {
        PY_reflected = 2 * SIZEY - 2 - PY_reflected;
        invert_sign_y = true;
    }

    int IDX = SIZEX * PY_reflected + PX_reflected;
    OUT[IDXPAD] = (invert_sign_x ^ invert_sign_y) ? -IN[IDX] : IN[IDX];
}

template <typename T>
void CUDAAsymmPad2D(dim3 threads,
                    void* GPU_in,
                    void* GPU_out,
                    const int SIZEX,
                    const int SIZEY,
                    const int SIZEXPAD,
                    const int SIZEYPAD)
{
    printf("CUDAAsymmPad2D threads=(%d, %d, %d), SIZEX=%d, SIZEY=%d, SIZEXPAD=%d, SIZEYPAD=%d\n",
           threads.x, threads.y, threads.z, SIZEX, SIZEY, SIZEXPAD, SIZEYPAD);

    // Calculate the number of blocks needed
    dim3 numBlocks = getNumBlocks(threads, SIZEXPAD, SIZEYPAD);

    AsymmPad2D<T>
        <<<numBlocks, threads>>>((T*)GPU_in, (T*)GPU_out, SIZEX, SIZEY, SIZEXPAD, SIZEYPAD);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch AsymmPad2D kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template <typename T>
__global__ void RemovePadding(
    T* __restrict__ IN, T* __restrict__ OUT, const int SIZEX, const int SIZEY, const int SIZEXPAD)
{
    const int PY = threadIdx.x + blockIdx.x * blockDim.x; // y-dimension
    const int PX = threadIdx.y + blockIdx.y * blockDim.y; // x-dimension

    if(PX >= SIZEX || PY >= SIZEY)
        return;

    int IDX = SIZEX * PY + PX;
    int IDXPAD = SIZEXPAD * PY + PX;

    OUT[IDX] = IN[IDXPAD];
}

template <typename T>
void CUDARemovePadding(
    dim3 threads, void* GPU_in, void* GPU_out, const int SIZEX, const int SIZEY, const int SIZEXPAD)
{
    printf("CUDARemovePadding threads=(%d, %d, %d), SIZEX=%d, SIZEY=%d, SIZEXPAD=%d\n", threads.x,
           threads.y, threads.z, SIZEX, SIZEY, SIZEXPAD);

    // Calculate the number of blocks needed
    dim3 numBlocks = getNumBlocks(threads, SIZEX, SIZEY);

    RemovePadding<T><<<numBlocks, threads>>>((T*)GPU_in, (T*)GPU_out, SIZEX, SIZEY, SIZEXPAD);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch RemovePadding kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// To maintain antisymmetry and ensure the zeroth element (both in PX and PY) is set to zero, while preserving antisymmetry around this element in a circular sense, you should set only the PX == 0 and PY == 0 conditions to zero.
template <typename T>
__global__ void AsymmPadDirichlet2D(T* __restrict__ IN,
                                    T* __restrict__ OUT,
                                    const int SIZEX,
                                    const int SIZEY,
                                    const int SIZEXPAD,
                                    const int SIZEYPAD)
{
    const int PY = threadIdx.x + blockIdx.x * blockDim.x; // y-dimension
    const int PX = threadIdx.y + blockIdx.y * blockDim.y; // x-dimension

    if(PX >= SIZEXPAD || PY >= SIZEYPAD)
        return;

    int IDXPAD = SIZEXPAD * PY + PX;

    // Reflect the index PX and PY to handle antisymmetric padding
    int PX_reflected = PX % (2 * SIZEX - 2);
    bool invert_sign_x = false;
    if(PX_reflected >= SIZEX)
    {
        PX_reflected = 2 * SIZEX - 2 - PX_reflected;
        invert_sign_x = true;
    }

    int PY_reflected = PY % (2 * SIZEY - 2);
    bool invert_sign_y = false;
    if(PY_reflected >= SIZEY)
    {
        PY_reflected = 2 * SIZEY - 2 - PY_reflected;
        invert_sign_y = true;
    }

    // Apply Dirichlet boundary condition: set the top-left edges to 0
    if(PX == 0 || PY == 0)
    {
        OUT[IDXPAD] = 0;
    } else
    {
        int IDX = SIZEX * PY_reflected + PX_reflected;
        OUT[IDXPAD] = (invert_sign_x ^ invert_sign_y) ? -IN[IDX] : IN[IDX];
    }
}

template <typename T>
void CUDAAsymmPadDirichlet2D(dim3 threads,
                             void* GPU_in,
                             void* GPU_out,
                             const int SIZEX,
                             const int SIZEY,
                             const int SIZEXPAD,
                             const int SIZEYPAD)
{
    printf("CUDAAsymmPadDirichlet2D threads=(%d, %d, %d), SIZEX=%d, SIZEY=%d, SIZEXPAD=%d, "
           "SIZEYPAD=%d\n",
           threads.x, threads.y, threads.z, SIZEX, SIZEY, SIZEXPAD, SIZEYPAD);

    // Calculate the number of blocks needed
    dim3 numBlocks = getNumBlocks(threads, SIZEXPAD, SIZEYPAD);

    AsymmPadDirichlet2D<T>
        <<<numBlocks, threads>>>((T*)GPU_in, (T*)GPU_out, SIZEX, SIZEY, SIZEXPAD, SIZEYPAD);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch AsymmPadDirichlet2D kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template <typename T>
__global__ void AsymmPadDirichlet(
    T* __restrict__ IN, T* __restrict__ OUT, const int SIZEX, const int SIZEXPAD, const int SIZEY)
{
    const int PY = threadIdx.x + blockIdx.x * blockDim.x; // y-dimension
    const int PX = threadIdx.y + blockIdx.y * blockDim.y; // x-dimension

    if(PX >= SIZEXPAD || PY >= SIZEY)
        return;

    int IDX = SIZEX * PY + PX;
    int IDXPAD = SIZEXPAD * PY + PX;

    // Reflect the index PX to handle antisymmetric padding
    int PX_reflected = PX % (2 * SIZEX - 2);
    bool invert_sign = false;
    if(PX_reflected >= SIZEX)
    {
        PX_reflected = 2 * SIZEX - 2 - PX_reflected;
        invert_sign = true;
    }

    // Apply Dirichlet boundary condition: set the left edges to 0
    if(PX == 0)
    {
        OUT[IDXPAD] = 0;
    } else
    {
        IDX = SIZEX * PY + PX_reflected;
        OUT[IDXPAD] = invert_sign ? -IN[IDX] : IN[IDX];
    }
}

template <typename T>
void CUDAAsymmPadDirichlet(
    dim3 threads, void* GPU_in, void* GPU_out, const int SIZEX, const int SIZEXPAD, const int SIZEY)
{
    printf("CUDAAsymmPadDirichlet threads=(%d, %d, %d), SIZEX=%d, SIZEXPAD=%d, SIZEY=%d\n",
           threads.x, threads.y, threads.z, SIZEX, SIZEXPAD, SIZEY);

    // Calculate the number of blocks needed
    dim3 numBlocks = getNumBlocks(threads, SIZEXPAD, SIZEY);

    AsymmPadDirichlet<T><<<numBlocks, threads>>>((T*)GPU_in, (T*)GPU_out, SIZEX, SIZEXPAD, SIZEY);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch AsymmPadDirichlet kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

//Explicit instantinations of CUDA functions for float and double
template void CUDAZeroPad<float>(dim3 threads,
                                 void* GPU_in,
                                 void* GPU_out,
                                 const int SIZEX,
                                 const int SIZEY,
                                 const int SIZEXPAD);

template void CUDAZeroPad<double>(dim3 threads,
                                  void* GPU_in,
                                  void* GPU_out,
                                  const int SIZEX,
                                  const int SIZEY,
                                  const int SIZEXPAD);

template void CUDAZeroPad2D<float>(dim3 threads,
                                   void* GPU_in,
                                   void* GPU_out,
                                   const int SIZEX,
                                   const int SIZEY,
                                   const int SIZEXPAD,
                                   const int SIZEYPAD);

template void CUDAZeroPad2D<double>(dim3 threads,
                                    void* GPU_in,
                                    void* GPU_out,
                                    const int SIZEX,
                                    const int SIZEY,
                                    const int SIZEXPAD,
                                    const int SIZEYPAD);

template void CUDASymmPad<float>(dim3 threads,
                                 void* GPU_in,
                                 void* GPU_out,
                                 const int SIZEX,
                                 const int SIZEXPAD,
                                 const int SIZEY);

template void CUDASymmPad<double>(dim3 threads,
                                  void* GPU_in,
                                  void* GPU_out,
                                  const int SIZEX,
                                  const int SIZEXPAD,
                                  const int SIZEY);

template void CUDASymmPad2D<float>(dim3 threads,
                                   void* GPU_in,
                                   void* GPU_out,
                                   const int SIZEX,
                                   const int SIZEY,
                                   const int SIZEXPAD,
                                   const int SIZEYPAD);

template void CUDASymmPad2D<double>(dim3 threads,
                                    void* GPU_in,
                                    void* GPU_out,
                                    const int SIZEX,
                                    const int SIZEY,
                                    const int SIZEXPAD,
                                    const int SIZEYPAD);

template void CUDAAsymmPad<float>(dim3 threads,
                                  void* GPU_in,
                                  void* GPU_out,
                                  const int SIZEX,
                                  const int SIZEXPAD,
                                  const int SIZEY);

template void CUDAAsymmPad<double>(dim3 threads,
                                   void* GPU_in,
                                   void* GPU_out,
                                   const int SIZEX,
                                   const int SIZEXPAD,
                                   const int SIZEY);

template void CUDAAsymmPad2D<float>(dim3 threads,
                                    void* GPU_in,
                                    void* GPU_out,
                                    const int SIZEX,
                                    const int SIZEY,
                                    const int SIZEXPAD,
                                    const int SIZEYPAD);

template void CUDAAsymmPad2D<double>(dim3 threads,
                                     void* GPU_in,
                                     void* GPU_out,
                                     const int SIZEX,
                                     const int SIZEY,
                                     const int SIZEXPAD,
                                     const int SIZEYPAD);

template void CUDARemovePadding<float>(dim3 threads,
                                       void* GPU_in,
                                       void* GPU_out,
                                       const int SIZEX,
                                       const int SIZEY,
                                       const int SIZEXPAD);

template void CUDARemovePadding<double>(dim3 threads,
                                        void* GPU_in,
                                        void* GPU_out,
                                        const int SIZEX,
                                        const int SIZEY,
                                        const int SIZEXPAD);

// Explicit instantiation for AsymmPadDirichlet2D
template void CUDAAsymmPadDirichlet2D<float>(dim3 threads,
                                             void* GPU_in,
                                             void* GPU_out,
                                             const int SIZEX,
                                             const int SIZEY,
                                             const int SIZEXPAD,
                                             const int SIZEYPAD);
template void CUDAAsymmPadDirichlet2D<double>(dim3 threads,
                                              void* GPU_in,
                                              void* GPU_out,
                                              const int SIZEX,
                                              const int SIZEY,
                                              const int SIZEXPAD,
                                              const int SIZEYPAD);
// Explicit instantiation for CUDAAsymmPadDirichlet
template void CUDAAsymmPadDirichlet<float>(dim3 threads,
                                           void* GPU_in,
                                           void* GPU_out,
                                           const int SIZEX,
                                           const int SIZEXPAD,
                                           const int SIZEY);

template void CUDAAsymmPadDirichlet<double>(dim3 threads,
                                            void* GPU_in,
                                            void* GPU_out,
                                            const int SIZEX,
                                            const int SIZEXPAD,
                                            const int SIZEY);
