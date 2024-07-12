// Logging

#include "tomographicFiltering.cuh"

__global__ void ifftshiftSpectral(float2* __restrict__ x, const int SIZEX, const int SIZEY)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    const int xSizeHermitan = SIZEX / 2 + 1;
    const int IDX = xSizeHermitan * PY + PX;
    if((PX >= xSizeHermitan) || (PY >= SIZEY))
        return;
    int centerShift = (SIZEX + 1) / 2;
    int freq = PX;
    double shiftingAngle = -TWOPI * freq * centerShift / SIZEX;
    double sinVal, cosVal;
    sincos(shiftingAngle, &sinVal, &cosVal);
    float2 xval = x[IDX];
    x[IDX].x = xval.x * cosVal - xval.y * sinVal;
    x[IDX].y = xval.y * cosVal + xval.x * sinVal;
}

void CUDAifftshiftSpectral(dim3 threads, dim3 blocks, void* x, const int SIZEX, const int SIZEY)
{
    ifftshiftSpectral<<<blocks, threads>>>((float2*)x, SIZEX, SIZEY);
}

__global__ void RadonFilter(float2* __restrict__ x,
                            const int SIZEX,
                            const int SIZEY,
                            const float pixel_size_x,
                            const bool ifftshift)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    const int xSizeHermitan = SIZEX / 2 + 1;
    const int IDX = xSizeHermitan * PY + PX;
    if((PX >= xSizeHermitan) || (PY >= SIZEY))
        return;
    double L = ((double)SIZEX) * pixel_size_x;
    //Note that K shall be PX/L but 1/L is a global scaling factor, which is additionally used here
    //See Kak_Slaney, Ch 3, p. 66, (41)
    double K = PX / (L * L);
    //ifftshiftSpectral
    //Note normally we would call fftshift(fft(ifftshift(image))
    //Now just transfrom results as if we input sequence f[ifftshift(x)]
    if(ifftshift)
    {

        int centerShift = (SIZEX + 1) / 2;
        int freq = PX;
        double shiftingAngle = -TWOPI * freq * centerShift / SIZEX;
        double sinVal, cosVal;
        sincos(shiftingAngle, &sinVal, &cosVal);
        float2 xval = x[IDX];
        xval.x *= K;
        xval.y *= K;
        x[IDX].x = xval.x * cosVal - xval.y * sinVal;
        x[IDX].y = xval.y * cosVal + xval.x * sinVal;
    } else
    {
        x[IDX].x *= K;
        x[IDX].y *= K;
    }
}

void CUDARadonFilter(dim3 threads,
                     dim3 blocks,
                     void* x,
                     const int SIZEX,
                     const int SIZEY,
                     const float pixel_size_x,
                     bool ifftshift)
{
    /*
    printf("threads=(%d,%d,%d), blocks(%d, %d, %d) SIZEX=%d, SIZEY=%d L=%f 1/L=%f pixel_size_x=%f "
           "ifftshift=%s\n",
           threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z, SIZEX, SIZEY,
           ((double)SIZEX) * pixel_size_x, 1 / (((double)SIZEX) * pixel_size_x), pixel_size_x,
           ifftshift ? "true" : "false");
*/
    RadonFilter<<<blocks, threads>>>((float2*)x, SIZEX, SIZEY, pixel_size_x, ifftshift);
}

inline __device__ float scaledSigmoid(float x, float scale)
{
    float minval = 1.0f / (1.0f + expf(scale));
    float sigmoid = 1.0f / (1.0f + expf(-scale * x));
    return (sigmoid - minval) / (1.0f - 2.0f * minval);
}

__global__ void ParkerFilter(
    float* __restrict__ x, const int SIZEX, const int SIZEY, const float corpos, const float zslope)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    const int IDX = SIZEX * PY + PX;
    float cor = corpos + PY * zslope;
    float dist;
    float radius;
    if(cor > 0.5f * (SIZEX - 1))
    {
        radius = (SIZEX - 1) - cor + 0.5;
        dist = cor - PX;
    } else
    {
        radius = cor + 0.5;
        dist = PX - cor;
    }
    if(abs(dist) <= radius)
    {
        float distFromCor = dist / radius; //[-1,1]
        float factor;
        //factor = 0.5f + 0.5f * distFromCor;//linear factor [0,1]
        factor = scaledSigmoid(distFromCor, 5.0f); //sigmoid factor [0,1]
        //	factor = 0.0f;
        x[IDX] *= factor;
    }
    /*
	else
	{
		//do nothing
		//x[IND] = x[IND];
	}
*/
}

void CUDAParkerFilter(dim3 threads,
                      dim3 blocks,
                      void* x,
                      const int SIZEX,
                      const int SIZEY,
                      const float corpos,
                      const float zslope)
{
    ParkerFilter<<<blocks, threads>>>((float*)x, SIZEX, SIZEY, corpos, zslope);
}

//circular shift
//shift .. The number of places by which elements are shifted to the right.
__global__ void roll(float* __restrict__ x_in,
                     float* __restrict__ x_out,
                     const int shift,
                     const int SIZEX,
                     const int SIZEY)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    if((PX >= SIZEX) || (PY >= SIZEY))
        return;
    int PX_shift;
    if(shift > PX)
    {
        PX_shift = PX + SIZEX - shift;
    } else
    {
        PX_shift = PX - shift;
    }
    int IDX_IN = SIZEX * PY + PX_shift;
    int IDX_OUT = SIZEX * PY + PX;
    x_out[IDX_OUT] = x_in[IDX_IN];
}

void CUDARoll(dim3 threads,
              dim3 blocks,
              void* x_in,
              void* x_out,
              const int shift,
              const int SIZEX,
              const int SIZEY)
{
    int modulusShift = shift;
    while(modulusShift < 0)
    {
        modulusShift += SIZEX;
    }
    while(modulusShift >= SIZEX)
    {
        modulusShift -= SIZEX;
    }
    roll<<<blocks, threads>>>((float*)x_in, (float*)x_out, modulusShift, SIZEX, SIZEY);
}

void CUDAifftshift(
    dim3 threads, dim3 blocks, void* x_in, void* x_out, const int SIZEX, const int SIZEY)
{
    int shift = (SIZEX + 1) / 2;
    roll<<<blocks, threads>>>((float*)x_in, (float*)x_out, shift, SIZEX, SIZEY);
}

void CUDAfftshift(
    dim3 threads, dim3 blocks, void* x_in, void* x_out, const int SIZEX, const int SIZEY)
{
    int shift = SIZEX / 2;
    roll<<<blocks, threads>>>((float*)x_in, (float*)x_out, shift, SIZEX, SIZEY);
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

__global__ void ZeroPad(float* __restrict__ IN,
                        float* __restrict__ OUT,
                        const int SIZEX,
                        const int SIZEXPAD,
                        const int SIZEY)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    if((PX >= SIZEXPAD) || (PY >= SIZEY))
        return;
    const int IDX = SIZEX * PY + PX;
    const int IDXPAD = SIZEXPAD * PY + PX;
    if(PX >= SIZEX)
    {
        OUT[IDXPAD] = 0.0f;
    } else
    {
        float val = IN[IDX];
        OUT[IDXPAD] = val;
    }
}

void CUDAZeroPad(dim3 threads,
                 dim3 blocks,
                 void* GPU_in,
                 void* GPU_out,
                 const int SIZEX,
                 const int SIZEXPAD,
                 const int SIZEY)
{
    /*
    printf("CUDAZeroPad threads=(%d,%d,%d), blocks(%d, %d, %d) SIZEX=%d, SIZEY=%d\n",
           threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z, SIZEX, SIZEY);
*/
    ZeroPad<<<blocks, threads>>>((float*)GPU_in, (float*)GPU_out, SIZEX, SIZEXPAD, SIZEY);
}

__global__ void SymmPad(float* __restrict__ IN,
                        float* __restrict__ OUT,
                        const int SIZEX,
                        const int SIZEXPAD,
                        const int SIZEY)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    if((PX >= SIZEXPAD) || (PY >= SIZEY))
        return;
    const int SIZEXPERIOD = 2 * SIZEX - 2;
    const int IDXPAD = SIZEXPAD * PY + PX;
    int PX_ORIGIN = PX;
    while(PX_ORIGIN > SIZEXPERIOD)
    {
        PX_ORIGIN -= SIZEXPERIOD;
    }
    if(PX_ORIGIN >= SIZEX)
    {
        PX_ORIGIN = SIZEX - 2 - (PX_ORIGIN - SIZEX);
    }
    int IDX = SIZEX * PY + PX_ORIGIN;
    OUT[IDXPAD] = IN[IDX];
}

void CUDASymmPad(dim3 threads,
                 dim3 blocks,
                 void* GPU_in,
                 void* GPU_out,
                 const int SIZEX,
                 const int SIZEXPAD,
                 const int SIZEY)
{
    /*
    printf("CUDAZeroPad threads=(%d,%d,%d), blocks(%d, %d, %d) SIZEX=%d, SIZEY=%d\n",
           threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z, SIZEX, SIZEY);
*/
    if(SIZEX < 2)
    {
        printf("This method is invalid for SIZEX<2 but SIZEX=%d", SIZEX);
        return;
    }
    if(SIZEXPAD != 2 * SIZEX - 2)
    {
        printf("SIZEXPAD=%d is not 2*%d-2 = %d", SIZEXPAD, SIZEX, 2 * SIZEX - 2);
    }
    SymmPad<<<blocks, threads>>>((float*)GPU_in, (float*)GPU_out, SIZEX, SIZEXPAD, SIZEY);
}

__global__ void StripPad(float* __restrict__ IN,
                         float* __restrict__ OUT,
                         const int SIZEX,
                         const int SIZEXPAD,
                         const int SIZEY)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    if((PX >= SIZEX) || (PY >= SIZEY))
        return;
    OUT[SIZEX * PY + PX] = IN[SIZEXPAD * PY + PX];
}

void CUDAStripPad(dim3 threads,
                  dim3 blocks,
                  void* GPU_IN,
                  void* GPU_OUT,
                  const int SIZEX,
                  const int SIZEXPAD,
                  const int SIZEY)
{
    StripPad<<<blocks, threads>>>((float*)GPU_IN, (float*)GPU_OUT, SIZEX, SIZEXPAD, SIZEY);
}

template <typename T>
__device__ T computeGaussianKernel(int u, int v, int width, int height, T sigma_x, T sigma_y)
{
    T sigma_x2 = sigma_x * sigma_x;
    T sigma_y2 = sigma_y * sigma_y;
    T val = exp(-2 * M_PI * M_PI * (sigma_x2 * u * u / width + sigma_y2 * v * v / height));
    return val;
}

//Template argument W is of the type cufftComplex or cufftDoubleComplex
//Assuming array is of the size SIZEX_HERMITIAN * SIZEY
template <typename T, typename W>
__global__ void
SpectralGaussianBlur2D(W* __restrict__ VEC, const int SIZEX, const int SIZEY, T sigma_x, T sigma_y)
{
    const int SIZEX_HERMITIAN = SIZEX / 2 + 1;

    const int PY = threadIdx.x + blockIdx.x * blockDim.x; // y-dimension
    const int PX = threadIdx.y + blockIdx.y * blockDim.y; // x-dimension

    if(PX >= SIZEX_HERMITIAN || PY >= SIZEY)
    {
        return;
    }

    const int IDX = SIZEX_HERMITIAN * PY + PX;

    T real_in = VEC[IDX].x;
    T imag_in = VEC[IDX].y;
    T real_ker = computeGaussianKernel<T>(PX <= SIZEX / 2 ? PX : PX - SIZEX,
                                          PY <= SIZEY / 2 ? PY : PY - SIZEY, SIZEX, SIZEY, sigma_x,
                                          sigma_y);
    VEC[IDX].x = real_in * real_ker;
    VEC[IDX].y = imag_in * real_ker;

    /*
        float imag_ker = 0.0f;  // Imaginary part of Gaussian kernel is zero
        OUT[IDX].x = real_in * real_ker - imag_in * imag_ker;
        OUT[IDX].y = real_in * imag_ker + imag_in * real_ker;
*/
}

template <typename T, typename W>
void CUDASpectralGaussianBlur2D(
    dim3 threads, void* GPU_vec, const int SIZEX, const int SIZEY, const T sigma_x, const T sigma_y)
{
    printf("CUDASpectralGaussianBlur threads=(%d, %d, %d), SIZEX=%d, SIZEY=%d, sigma_x=%f, "
           "sigma_y=%f\n",
           threads.x, threads.y, threads.z, SIZEX, SIZEY, sigma_x, sigma_y);

    // Calculate the number of blocks needed
    const int SIZEX_HERMITIAN = SIZEX / 2 + 1;
    dim3 numBlocks((SIZEY + threads.x - 1) / threads.x,
                   (SIZEX_HERMITIAN + threads.y - 1) / threads.y);

    SpectralGaussianBlur2D<T, W>
        <<<numBlocks, threads>>>((T*)GPU_vec, SIZEX, SIZEY, sigma_x, sigma_y);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch SpectralGaussianBlur kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

