// Logging

#include "spectralMethod.cuh"

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
    if((PX >= xSizeHermitan) || (PY >= SIZEY))
        return;
    if(PX == 0 && PY == 0)
    {
        x[0].x = 0.0f;
        x[0].y = 0.0f;
    } else
    {
        //        double K = (double)SIZEX * (double)SIZEY;
        double K = FOURPISQUARED;
        /*
            const float sizeDivide = (float)SIZEX * SIZEY;
            x[IDX].x /= sizeDivide;
            x[IDX].y /= sizeDivide;*/
        // See https://atmos.washington.edu/~breth/classes/AM585/lect/DFT_FS_585_notes.pdf
        double kx2scale, ky2scale, kx, ky;
        kx2scale = ((double)SIZEX) / (((double)SIZEY) * pixel_size_x * pixel_size_x);
        ky2scale = ((double)SIZEY) / (((double)SIZEX) * pixel_size_y * pixel_size_y);
        //        if(PX <= SIZEX / 2) ... always satisfied by hermitan symmetry choice
        //        {
        kx = PX;
        /*
                } else
                {
                    kx = 2.0 * PI * (PX - SIZEX) / (SIZEX * pixel_size_x);
                }*/
        if(PY <= SIZEY / 2)
        {
            ky = PY;
        } else
        {
            ky = PY - SIZEY;
        }
        K *= -((kx2scale * kx * kx + ky2scale * ky * ky));
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

__global__ void spectralMultiplication(float2* __restrict__ x,
                                       const int SIZEX,
                                       const int SIZEY,
                                       const float pixel_size_x,
                                       const float pixel_size_y)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    const int xSizeHermitan = SIZEX / 2 + 1;
    const int IDX = xSizeHermitan * PY + PX;
    if((PX >= xSizeHermitan) || (PY >= SIZEY))
        return;
    if(PX == 0 && PY == 0)
    {
        x[0].x = 0.0f;
        x[0].y = 0.0f;
    } else
    {
        //        double K = 1.0/( (double)SIZEX * (double)SIZEY);
        double K = FOURPISQUARED / ((double)SIZEX * (double)SIZEY);
        /*
            const float sizeDivide = (float)SIZEX * SIZEY;
            x[IDX].x /= sizeDivide;
            x[IDX].y /= sizeDivide;*/
        // See https://atmos.washington.edu/~breth/classes/AM585/lect/DFT_FS_585_notes.pdf
        double kx, ky, kx2scale, ky2scale;
        kx2scale = ((double)SIZEX * (double)SIZEX * pixel_size_x * pixel_size_x);
        ky2scale = ((double)SIZEY * (double)SIZEY * pixel_size_y * pixel_size_y);
        //        if(PX <= SIZEX / 2) ... always satisfied by hermitan symmetry choice
        //        {
        kx = PX;
        /*
                } else
                {
                    kx = 2.0 * PI * (PX - SIZEX) / (SIZEX * pixel_size_x);
                }*/
        if(PY <= SIZEY / 2)
        {
            ky = PY;
        } else
        {
            ky = (PY - SIZEY);
        }
        K *= -(kx * kx / kx2scale + ky * ky / ky2scale);
        x[IDX].x *= K;
        x[IDX].y *= K;
    }
}

void CUDAspectralMultiplication(dim3 threads,
                                dim3 blocks,
                                void* x,
                                const int SIZEX,
                                const int SIZEY,
                                const float pixel_size_x,
                                const float pixel_size_y)
{
    printf("threads=(%d,%d,%d), blocks(%d, %d, %d) SIZEX=%d, SIZEY=%d\n", threads.x, threads.y,
           threads.z, blocks.x, blocks.y, blocks.z, SIZEX, SIZEY);
    spectralMultiplication<<<blocks, threads>>>((float2*)x, SIZEX, SIZEY, pixel_size_x,
                                                pixel_size_y);
}

__global__ void constantMultiplication(float* __restrict__ x,
                                       const float factor,
                                       const int SIZEX,
                                       const int SIZEY,
                                       const int TOX,
                                       const int TOY)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    if((PX >= TOX) || (PY >= TOY))
        return;
    const int IDX = SIZEX * PY + PX;
    x[IDX] *= factor;
}

void CUDAconstantMultiplication(dim3 threads,
                                dim3 blocks,
                                void* x,
                                const float factor,
                                const int SIZEX,
                                const int SIZEY,
                                const int TOX,
                                const int TOY)
{
    printf("CUDAconstantMultiplication threads=(%d,%d,%d), blocks(%d, %d, %d) SIZEX=%d, SIZEY=%d\n",
           threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z, SIZEX, SIZEY);
    constantMultiplication<<<blocks, threads>>>((float*)x, factor, SIZEX, SIZEY, TOX, TOY);
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
    /* //STANDARD ANTISYMMETRIC EXTENSION
    fx[PY * SIZEXEXT + PX] = val;
    fx[PY * SIZEXEXT + SIZEXEXT - PX - 1] = -val;
    fx[(SIZEYEXT - PY - 1) * SIZEXEXT + PX] = -val;
    fx[(SIZEYEXT - PY - 1) * SIZEXEXT + SIZEXEXT - PX - 1] = val;
    */
    // DFT SYMMETRIC EXTENSION
    if(PX != 0 && PY != 0)
    {
        fx[PY * SIZEXEXT + PX] = val;
        fx[PY * SIZEXEXT + SIZEXEXT - PX] = -val;
        fx[(SIZEYEXT - PY) * SIZEXEXT + PX] = -val;
        fx[(SIZEYEXT - PY) * SIZEXEXT + SIZEXEXT - PX] = val;
    } else if(PX == 0 && PY != 0)
    {
        fx[PY * SIZEXEXT + PX] = 0.0f;
        fx[(SIZEYEXT - PY) * SIZEXEXT + PX] = 0.0f;
    } else if(PX != 0 && PY == 0)
    {
        fx[PY * SIZEXEXT + PX] = 0.0f;
        fx[PY * SIZEXEXT + SIZEXEXT - PX] = 0.0f;
    }
    if(PX == SIZEX - 1 && PY == SIZEY - 1)
    {
        fx[SIZEY * SIZEXEXT + SIZEX] = 0.0f;
    }
    if(PX == SIZEX - 1)
    {
        fx[PY * SIZEXEXT + SIZEX] = 0.0f;
        if(PY != 0)
        {
            fx[(SIZEYEXT - PY) * SIZEXEXT + SIZEX] = 0.0f;
        }
    }
    if(PY == SIZEY - 1)
    {
        fx[SIZEY * SIZEXEXT + PX] = 0.0f;
        if(PX != 0)
        {
            fx[SIZEY * SIZEXEXT + SIZEXEXT - PX] = 0.0f;
        }
    }
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
    fx[PY * SIZEXEXT + PX] = val;
    /* //STANDARD SYMMETRIC EXTENSION
        fx[PY * SIZEXEXT + SIZEXEXT - PX - 1] = val;
        fx[(SIZEYEXT - PY - 1) * SIZEXEXT + PX] = val;
        fx[(SIZEYEXT - PY - 1) * SIZEXEXT + SIZEXEXT - PX - 1] = val;
    */
    // DFT SYMMETRIC EXTENSION
    if(PX != 0 && PY != 0)
    {
        fx[PY * SIZEXEXT + SIZEXEXT - PX] = val;
        fx[(SIZEYEXT - PY) * SIZEXEXT + PX] = val;
        fx[(SIZEYEXT - PY) * SIZEXEXT + SIZEXEXT - PX] = val;
    } else if(PX == 0 && PY != 0)
    {

        fx[(SIZEYEXT - PY) * SIZEXEXT + PX] = val;
    } else if(PX != 0 && PY == 0)
    {
        fx[PY * SIZEXEXT + SIZEXEXT - PX] = val;
    }
    if(PX == SIZEX - 1 && PY == SIZEY - 1)
    {
        fx[SIZEY * SIZEXEXT + SIZEX] = val;
    }
    if(PX == SIZEX - 1)
    {
        fx[PY * SIZEXEXT + SIZEX] = val;
        if(PY != 0)
        {
            fx[(SIZEYEXT - PY) * SIZEXEXT + SIZEX] = val;
        }
    }
    if(PY == SIZEY - 1)
    {
        fx[SIZEY * SIZEXEXT + PX] = val;
        if(PX != 0)
        {
            fx[SIZEY * SIZEXEXT + SIZEXEXT - PX] = val;
        }
    }
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
