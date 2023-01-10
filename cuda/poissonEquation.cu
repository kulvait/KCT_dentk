// Logging

#include "poissonEquation.cuh"

__global__ void spectralDivision(float2* __restrict__ x,
                                 const int SIZEX,
                                 const int SIZEY,
                                 const float pixel_size_x,
                                 const float pixel_size_y)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    const int xSizeHermitan = SIZEX / 2 + 1;
    const int IDX = xSizeHermitan * PY + PX;
    const float sizeDivide = (float)SIZEX * SIZEY;
    if((PX >= xSizeHermitan) || (PY >= SIZEY))
        return;
    x[IDX].x /= sizeDivide;
    x[IDX].y /= sizeDivide;
    if(PX == 0 && PY == 0)
    {
        x[0].x = 0;
        x[0].y = 0;
    } else
    {
        // See https://atmos.washington.edu/~breth/classes/AM585/lect/DFT_FS_585_notes.pdf
        float kx, ky;
        if(PX < SIZEX / 2)
        {
            kx = 2.0f * PI * PX / (SIZEX * pixel_size_x);
        } else
        {
            kx = 2.0f * PI * (PX - SIZEX) / (SIZEX * pixel_size_x);
        }
        if(PY < SIZEY / 2)
        {
            ky = 2.0f * PI * PY / (SIZEY * pixel_size_y);
        } else
        {
            ky = 2.0f * PI * (PY - SIZEY) / (SIZEY * pixel_size_y);
        }
        float K = -(kx * kx + ky * ky);
        x[IDX].x /= K;
        x[IDX].y /= K;
    }
}

void CUDAspectralDivision(dim3 threads,
                          dim3 blocks,
                          void* x,
                          const int SIZEX,
                          const int SIZEY,
                          const float pixel_size_x,
                          const float pixel_size_y)
{
    printf("threads=(%d,%d,%d), blocks(%d, %d, %d) SIZEX=%d, SIZEY=%d\n", threads.x, threads.y,
           threads.z, blocks.x, blocks.y, blocks.z, SIZEX, SIZEY);
    spectralDivision<<<blocks, threads>>>((float2*)x, SIZEX, SIZEY, pixel_size_x, pixel_size_y);
}

__global__ void
dirichletExtension(float* __restrict__ f, float* __restrict__ fx, const int SIZEX, const int SIZEY)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    if((PX >= SIZEX) || (PY >= SIZEY))
        return;
    const int SIZEXEXT = 2 * SIZEX;
    const int SIZEYEXT = 2 * SIZEY;
    const int IDX = SIZEX * PY + PX;
    float val = f[IDX];
    fx[PY * SIZEXEXT + PX] = val;
    fx[PY * SIZEXEXT + SIZEXEXT - PX - 1] = -val;
    fx[(SIZEYEXT - PY - 1) * SIZEXEXT + PX] = -val;
    fx[(SIZEYEXT - PY - 1) * SIZEXEXT + SIZEXEXT - PX - 1] = val;
}

void CUDADirichletExtension(
    dim3 threads, dim3 blocks, void* GPU_f, void* GPU_extendedf, const int SIZEX, const int SIZEY)
{
    printf("CUDADirichletExtension threads=(%d,%d,%d), blocks(%d, %d, %d) SIZEX=%d, SIZEY=%d\n",
           threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z, SIZEX, SIZEY);
    dirichletExtension<<<blocks, threads>>>((float*)GPU_f, (float*)GPU_extendedf, SIZEX, SIZEY);
}

__global__ void
neumannExtension(float* __restrict__ f, float* __restrict__ fx, const int SIZEX, const int SIZEY)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    if((PX >= SIZEX) || (PY >= SIZEY))
        return;
    const int SIZEXEXT = 2 * SIZEX;
    const int SIZEYEXT = 2 * SIZEY;
    const int IDX = SIZEX * PY + PX;
    float val = f[IDX];
    fx[PY * SIZEXEXT + SIZEXEXT - PX - 1] = val;
    fx[(SIZEYEXT - PY - 1) * SIZEXEXT + PX] = val;
    fx[(SIZEYEXT - PY - 1) * SIZEXEXT + SIZEXEXT - PX - 1] = val;
    fx[PY * SIZEXEXT + PX] = val;
}

void CUDANeumannExtension(
    dim3 threads, dim3 blocks, void* GPU_f, void* GPU_extendedf, const int SIZEX, const int SIZEY)
{
    printf("CUDANeumannExtension threads=(%d,%d,%d), blocks(%d, %d, %d) SIZEX=%d, SIZEY=%d\n",
           threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z, SIZEX, SIZEY);
    neumannExtension<<<blocks, threads>>>((float*)GPU_f, (float*)GPU_extendedf, SIZEX, SIZEY);
}

__global__ void
functionRestriction(float* __restrict__ fx, float* __restrict__ f, const int SIZEX, const int SIZEY)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    if((PX >= SIZEX) || (PY >= SIZEY))
        return;
    const int SIZEXEXT = 2 * SIZEX;
    f[SIZEX * PY + PX] = fx[SIZEXEXT * PY + PX];
}

void CUDAFunctionRestriction(
    dim3 threads, dim3 blocks, void* GPU_extendedf, void* GPU_f, const int SIZEX, const int SIZEY)
{
    printf("CUDAFunctionRestriction threads=(%d,%d,%d), blocks(%d, %d, %d) SIZEX=%d, SIZEY=%d\n",
           threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z, SIZEX, SIZEY);
    functionRestriction<<<blocks, threads>>>((float*)GPU_extendedf, (float*)GPU_f, SIZEX, SIZEY);
}
