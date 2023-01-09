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
