#include "cudaUtil.cuh"
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
                            const int SIZEX_FULL,
                            const int SIZEY,
                            const float pixel_size_x,
                            const bool ifftshift)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    const int SIZEX_FULL_HERMITIAN = SIZEX_FULL / 2 + 1;
    const int IDX = SIZEX_FULL_HERMITIAN * PY + PX;
    if((PX >= SIZEX_FULL_HERMITIAN) || (PY >= SIZEY))
        return;
    double L = ((double)SIZEX_FULL) * pixel_size_x;
    //Note that K shall be PX/L but 1/L is a global scaling factor, see Kak_Slaney, Ch 3, p. 66, (41)
    //double K = PX / (L * L);
    //Note that K shall be PX/L but 1/SIZEX_FULL is a global scaling factor for unscaled IFFT
    double K = PX / (L * L);
    //ifftshiftSpectral
    //Note normally we would call fftshift(fft(ifftshift(image))
    //Now just transfrom results as if we input sequence f[ifftshift(x)]
    if(ifftshift)
    {

        int centerShift = (SIZEX_FULL + 1) / 2;
        int freq = PX;
        double shiftingAngle = -TWOPI * freq * centerShift / SIZEX_FULL;
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

template <typename T, typename W>
__global__ void SpectralFilter(W* __restrict__ X,
                               const T* __restrict__ FILTER,
                               const int SIZEX,
                               const int SIZEY,
                               const T pixel_size_x)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    const int SIZEX_HERMITIAN = SIZEX / 2 + 1;

    if(PX >= SIZEX_HERMITIAN || PY >= SIZEY)
        return;

    //Pixel compensation for projection/bacprojection pair
    const int IDX = SIZEX_HERMITIAN * PY + PX;
    T factor = T(1) / (static_cast<T>(SIZEX) * pixel_size_x * pixel_size_x);
    T F = FILTER[PX] * factor;
    X[IDX].x *= F;
    X[IDX].y *= F;
}

template <typename T, typename W>
void CUDASpectralFilter(dim3 threads,
                        void* GPU_f_in,
                        void* GPU_filter_in,
                        const int SIZEX_FULL,
                        const int SIZEY,
                        const T pixel_size_x)
{
    const int SIZEX_HERMITIAN = SIZEX_FULL / 2 + 1;
    const dim3 numBlocks = getNumBlocks(threads, SIZEX_HERMITIAN, SIZEY);

    SpectralFilter<T, W>
        <<<numBlocks, threads>>>((W*)GPU_f_in, (T*)GPU_filter_in, SIZEX_FULL, SIZEY, pixel_size_x);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch SpectralFilter kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * Construct a circularly even discrete spatial-domain Ram-Lak convolution kernel.
 * see https://doi.org/10.1073/pnas.68.9.2236
 *
 * The generated sequence satisfies
 *
 *   h[n] = h[(N - n) mod N],
 *
 * so its DFT is real-valued in exact arithmetic. CUFFT R2C/D2Z still returns a
 * complex half-spectrum, but the imaginary components should be zero up to
 * floating-point roundoff. Therefore, after transforming this kernel, it is
 * sufficient to keep only the real part of the frequency response and multiply
 * projection spectra by a real scalar.
 *
 * The circular FFT layout is:
 *
 *   h[0] = 1/4
 *   h[ i] = -1 / (pi^2 i^2),  for positive odd i
 *   h[-i] = -1 / (pi^2 i^2),  for positive odd i
 *   h[ i] = 0,                otherwise
 *
 * In array indexing, negative spatial index -i is stored as h[N - i].
 *
 * This layout is intended to be transformed directly by CUFFT R2C/D2Z.
 * Do not fftshift this kernel before the FFT.
 */
template <typename T>
__global__ void RamLakKernel1D(T* __restrict__ OUT, const int SIZE)
{
    const int PX = threadIdx.x + blockIdx.x * blockDim.x;

    if(PX >= SIZE)
    {
        return;
    }

    if(PX == 0)
    {
        OUT[PX] = static_cast<T>(0.25);
        return;
    }

    // Circular distance from zero:
    //
    //   PX = 1          -> i = 1
    //   PX = 2          -> i = 2
    //   ...
    //   PX = SIZE - 1   -> i = 1  equivalent to h[-1]
    //   PX = SIZE - 2   -> i = 2  equivalent to h[-2]
    //
    const int i = min(PX, SIZE - PX);

    // Match Python:
    //
    //   for i in range(SIZE // 2):
    //       ...
    //
    // Therefore i == SIZE / 2 is excluded.
    if((i & 1) == 0)
    {
        OUT[PX] = static_cast<T>(0);
        return;
    }

    const T denom = static_cast<T>(PISQUARED) * static_cast<T>(i) * static_cast<T>(i);

    OUT[PX] = -static_cast<T>(1) / denom;
}

template <typename T>
void CUDARamLakKernel1D(void* x, const int SIZE)
{
    const int THREADS = 256;
    const int BLOCKS = (SIZE + THREADS - 1) / THREADS;

    RamLakKernel1D<T><<<BLOCKS, THREADS>>>(static_cast<T*>(x), SIZE);
}

template <typename T>
__global__ void
IdealRamp1D(T* __restrict__ OUT_PACKED, const int SIZE_HERMITIAN, const int SIZE_FULL)
{
    const int IND = threadIdx.x + blockIdx.x * blockDim.x;
    if(IND >= SIZE_HERMITIAN)
    {
        return;
    }
    // R2C half-spectrum stores frequencies:
    //   IND = 0, 1, 2, ..., SIZE_FULL / 2
    // so the normalized nonnegative frequency is IND / SIZE_FULL.
    OUT_PACKED[IND] = static_cast<T>(IND) / static_cast<T>(SIZE_FULL);
}

// Vector X_PACKED is of the size SIZE_HERMITIAN.
template <typename T>
void CUDAIdealRamp1D(void* OUT_PACKED, const int SIZE_FULL)
{
    const int SIZE_HERMITIAN = SIZE_FULL / 2 + 1;
    const int THREADS = 256;
    const int BLOCKS = (SIZE_HERMITIAN + THREADS - 1) / THREADS;

    IdealRamp1D<T><<<BLOCKS, THREADS>>>(static_cast<T*>(OUT_PACKED), SIZE_HERMITIAN, SIZE_FULL);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch IdealRamp1D kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template <typename T>
__global__ void
ApplySheppLoganWindow1D(T* __restrict__ FILTER, const int SIZE_HERMITIAN, const int SIZE_FULL)
{
    const int IND = threadIdx.x + blockIdx.x * blockDim.x;

    if(IND >= SIZE_HERMITIAN)
    {
        return;
    }

    if(IND == 0)
    {
        return;
    }

    const T pi = static_cast<T>(3.141592653589793238462643383279502884);
    const T omega = pi * static_cast<T>(IND) / static_cast<T>(SIZE_FULL);

    FILTER[IND] *= sin(omega) / omega;
}

template <typename T>
__global__ void
ApplyCosineWindow1D(T* __restrict__ FILTER, const int SIZE_HERMITIAN, const int SIZE_FULL)
{
    const int IND = threadIdx.x + blockIdx.x * blockDim.x;

    if(IND >= SIZE_HERMITIAN)
    {
        return;
    }

    const T pi = static_cast<T>(3.141592653589793238462643383279502884);
    const T omega = pi * static_cast<T>(IND) / static_cast<T>(SIZE_FULL);

    FILTER[IND] *= cos(omega);
}

template <typename T>
__global__ void
ApplyHanningWindow1D(T* __restrict__ FILTER, const int SIZE_HERMITIAN, const int SIZE_FULL)
{
    const int IND = threadIdx.x + blockIdx.x * blockDim.x;

    if(IND >= SIZE_HERMITIAN)
    {
        return;
    }

    const T pi = static_cast<T>(3.141592653589793238462643383279502884);
    const T theta = static_cast<T>(2) * pi * static_cast<T>(IND) / static_cast<T>(SIZE_FULL);
    const T window = static_cast<T>(0.5) + static_cast<T>(0.5) * cos(theta);

    FILTER[IND] *= window;
}

template <typename T>
__global__ void
ApplyHammingWindow1D(T* __restrict__ FILTER, const int SIZE_HERMITIAN, const int SIZE_FULL)
{
    const int IND = threadIdx.x + blockIdx.x * blockDim.x;

    if(IND >= SIZE_HERMITIAN)
    {
        return;
    }

    const T pi = static_cast<T>(3.141592653589793238462643383279502884);
    const T theta = static_cast<T>(2) * pi * static_cast<T>(IND) / static_cast<T>(SIZE_FULL);
    const T window = static_cast<T>(0.54) + static_cast<T>(0.46) * cos(theta);

    FILTER[IND] *= window;
}

template <typename T>
__device__ T BesselI0Approx(T x)
{
    const T ax = fabs(x);

    if(ax < static_cast<T>(3.75))
    {
        const T y = (x / static_cast<T>(3.75)) * (x / static_cast<T>(3.75));

        return static_cast<T>(1.0)
            + y
            * (static_cast<T>(3.5156229)
               + y
                   * (static_cast<T>(3.0899424)
                      + y
                          * (static_cast<T>(1.2067492)
                             + y
                                 * (static_cast<T>(0.2659732)
                                    + y
                                        * (static_cast<T>(0.0360768)
                                           + y * static_cast<T>(0.0045813))))));
    }

    const T y = static_cast<T>(3.75) / ax;

    return (exp(ax) / sqrt(ax))
        * (static_cast<T>(0.39894228)
           + y
               * (static_cast<T>(0.01328592)
                  + y
                      * (static_cast<T>(0.00225319)
                         + y
                             * (-static_cast<T>(0.00157565)
                                + y
                                    * (static_cast<T>(0.00916281)
                                       + y
                                           * (-static_cast<T>(0.02057706)
                                              + y
                                                  * (static_cast<T>(0.02635537)
                                                     + y
                                                         * (-static_cast<T>(0.01647633)
                                                            + y
                                                                * static_cast<T>(
                                                                    0.00392377)))))))));
}

template <typename T>
__global__ void ApplyKaiserWindow1D(T* __restrict__ FILTER,
                                    const int SIZE_HERMITIAN,
                                    const int SIZE_FULL,
                                    const T beta)
{
    const int IND = threadIdx.x + blockIdx.x * blockDim.x;

    if(IND >= SIZE_HERMITIAN)
    {
        return;
    }

    const T alpha = static_cast<T>(SIZE_FULL - 1) / static_cast<T>(2);

    if(alpha <= static_cast<T>(0))
    {
        return;
    }

    const T r = static_cast<T>(IND) / alpha;
    const T oneMinusRSquared = max(static_cast<T>(0), static_cast<T>(1) - r * r);

    const T numerator = BesselI0Approx(beta * sqrt(oneMinusRSquared));
    const T denominator = BesselI0Approx(beta);

    FILTER[IND] *= numerator / denominator;
}

void CUDAApplySheppLoganWindow1DFloat(void* filter, const int SIZE_HERMITIAN, const int SIZE_FULL)
{
    const int THREADS = 256;
    const int BLOCKS = (SIZE_HERMITIAN + THREADS - 1) / THREADS;

    ApplySheppLoganWindow1D<float>
        <<<BLOCKS, THREADS>>>(static_cast<float*>(filter), SIZE_HERMITIAN, SIZE_FULL);
}

void CUDAApplySheppLoganWindow1DDouble(void* filter, const int SIZE_HERMITIAN, const int SIZE_FULL)
{
    const int THREADS = 256;
    const int BLOCKS = (SIZE_HERMITIAN + THREADS - 1) / THREADS;

    ApplySheppLoganWindow1D<double>
        <<<BLOCKS, THREADS>>>(static_cast<double*>(filter), SIZE_HERMITIAN, SIZE_FULL);
}

void CUDAApplyCosineWindow1DFloat(void* filter, const int SIZE_HERMITIAN, const int SIZE_FULL)
{
    const int THREADS = 256;
    const int BLOCKS = (SIZE_HERMITIAN + THREADS - 1) / THREADS;

    ApplyCosineWindow1D<float>
        <<<BLOCKS, THREADS>>>(static_cast<float*>(filter), SIZE_HERMITIAN, SIZE_FULL);
}

void CUDAApplyCosineWindow1DDouble(void* filter, const int SIZE_HERMITIAN, const int SIZE_FULL)
{
    const int THREADS = 256;
    const int BLOCKS = (SIZE_HERMITIAN + THREADS - 1) / THREADS;

    ApplyCosineWindow1D<double>
        <<<BLOCKS, THREADS>>>(static_cast<double*>(filter), SIZE_HERMITIAN, SIZE_FULL);
}

void CUDAApplyHanningWindow1DFloat(void* filter, const int SIZE_HERMITIAN, const int SIZE_FULL)
{
    const int THREADS = 256;
    const int BLOCKS = (SIZE_HERMITIAN + THREADS - 1) / THREADS;

    ApplyHanningWindow1D<float>
        <<<BLOCKS, THREADS>>>(static_cast<float*>(filter), SIZE_HERMITIAN, SIZE_FULL);
}

void CUDAApplyHanningWindow1DDouble(void* filter, const int SIZE_HERMITIAN, const int SIZE_FULL)
{
    const int THREADS = 256;
    const int BLOCKS = (SIZE_HERMITIAN + THREADS - 1) / THREADS;

    ApplyHanningWindow1D<double>
        <<<BLOCKS, THREADS>>>(static_cast<double*>(filter), SIZE_HERMITIAN, SIZE_FULL);
}

void CUDAApplyHammingWindow1DFloat(void* filter, const int SIZE_HERMITIAN, const int SIZE_FULL)
{
    const int THREADS = 256;
    const int BLOCKS = (SIZE_HERMITIAN + THREADS - 1) / THREADS;

    ApplyHammingWindow1D<float>
        <<<BLOCKS, THREADS>>>(static_cast<float*>(filter), SIZE_HERMITIAN, SIZE_FULL);
}

void CUDAApplyHammingWindow1DDouble(void* filter, const int SIZE_HERMITIAN, const int SIZE_FULL)
{
    const int THREADS = 256;
    const int BLOCKS = (SIZE_HERMITIAN + THREADS - 1) / THREADS;

    ApplyHammingWindow1D<double>
        <<<BLOCKS, THREADS>>>(static_cast<double*>(filter), SIZE_HERMITIAN, SIZE_FULL);
}

void CUDAApplyKaiserWindow1DFloat(void* filter,
                                  const int SIZE_HERMITIAN,
                                  const int SIZE_FULL,
                                  const float beta)
{
    const int THREADS = 256;
    const int BLOCKS = (SIZE_HERMITIAN + THREADS - 1) / THREADS;

    ApplyKaiserWindow1D<float>
        <<<BLOCKS, THREADS>>>(static_cast<float*>(filter), SIZE_HERMITIAN, SIZE_FULL, beta);
}

void CUDAApplyKaiserWindow1DDouble(void* filter,
                                   const int SIZE_HERMITIAN,
                                   const int SIZE_FULL,
                                   const double beta)
{
    const int THREADS = 256;
    const int BLOCKS = (SIZE_HERMITIAN + THREADS - 1) / THREADS;

    ApplyKaiserWindow1D<double>
        <<<BLOCKS, THREADS>>>(static_cast<double*>(filter), SIZE_HERMITIAN, SIZE_FULL, beta);
}

__global__ void
ExtractRealFFTFloat(const cufftComplex* __restrict__ IN, float* __restrict__ OUT, const int SIZE)
{
    const int PX = threadIdx.x + blockIdx.x * blockDim.x;

    if(PX >= SIZE)
    {
        return;
    }

    OUT[PX] = IN[PX].x;
}

__global__ void ExtractRealFFTDouble(const cufftDoubleComplex* __restrict__ IN,
                                     double* __restrict__ OUT,
                                     const int SIZE)
{
    const int PX = threadIdx.x + blockIdx.x * blockDim.x;

    if(PX >= SIZE)
    {
        return;
    }

    OUT[PX] = IN[PX].x;
}

void CUDAExtractRealFFTFloat(void* in, void* out, const int SIZE)
{
    const int THREADS = 256;
    const int BLOCKS = (SIZE + THREADS - 1) / THREADS;

    ExtractRealFFTFloat<<<BLOCKS, THREADS>>>(static_cast<const cufftComplex*>(in),
                                             static_cast<float*>(out), SIZE);
}

void CUDAExtractRealFFTDouble(void* in, void* out, const int SIZE)
{
    const int THREADS = 256;
    const int BLOCKS = (SIZE + THREADS - 1) / THREADS;

    ExtractRealFFTDouble<<<BLOCKS, THREADS>>>(static_cast<const cufftDoubleComplex*>(in),
                                              static_cast<double*>(out), SIZE);
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

//Template argument W is of the type cufftComplex or cufftDoubleComplex
//Assuming array is of the size SIZEX_HERMITIAN * chunkSize
template <typename T, typename W>
__global__ void
SpectralGaussianBlur1D(W* __restrict__ VEC, const int SIZEX, const int SIZEY, const T sigma_z)
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
    // Inline Gaussian kernel computation
    int u = PX <= SIZEX / 2 ? PX : PX - SIZEX;
    T sigma_z_u_over_width = sigma_z * u / SIZEX;
    T real_ker = exp(MINUSTWOPISQUARED * (sigma_z_u_over_width * sigma_z_u_over_width));
    //CUDA IFFT normalization
    real_ker /= SIZEX;

    VEC[IDX].x = real_in * real_ker;
    VEC[IDX].y = imag_in * real_ker;
}

template <typename T, typename W>
void CUDASpectralGaussianBlur1D(
    dim3 threads, void* GPU_vec, const int SIZEX, const int SIZEY, const T sigma_z)
{
    // Calculate the number of blocks needed
    const int SIZEX_HERMITIAN = SIZEX / 2 + 1;
    dim3 numBlocks((SIZEY + threads.x - 1) / threads.x,
                   (SIZEX_HERMITIAN + threads.y - 1) / threads.y);

    SpectralGaussianBlur1D<T, W><<<numBlocks, threads>>>((W*)GPU_vec, SIZEX, SIZEY, sigma_z);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch SpectralGaussianBlur kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

//Template argument W is of the type cufftComplex or cufftDoubleComplex
//Assuming array is of the size SIZEX_HERMITIAN * SIZEY
template <typename T, typename W>
__global__ void SpectralGaussianBlur2D(
    W* __restrict__ VEC, const int SIZEX, const int SIZEY, const T sigma_x, const T sigma_y)
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
    // Inline Gaussian kernel computation
    int u = PX <= SIZEX / 2 ? PX : PX - SIZEX;
    int v = PY <= SIZEY / 2 ? PY : PY - SIZEY;
    T sigma_x_u_over_width = sigma_x * u / SIZEX;
    T sigma_y_v_over_height = sigma_y * v / SIZEY;
    T real_ker = exp(MINUSTWOPISQUARED
                     * (sigma_x_u_over_width * sigma_x_u_over_width
                        + sigma_y_v_over_height * sigma_y_v_over_height));
    //CUDA IFFT normalization
    real_ker /= (SIZEX * SIZEY);
    VEC[IDX].x = real_in * real_ker;
    VEC[IDX].y = imag_in * real_ker;

    /*
        float imag_ker = 0.0f;  // Imaginary part of Gaussian kernel is zero
        OUT[IDX].x = real_in * real_ker - imag_in * imag_ker;
        OUT[IDX].y = real_in * imag_ker + imag_in * real_ker;
*/
}
/*
template <typename T>
T ComputeNormalizationFactor(const int SIZEX, const int SIZEY, const T sigma_x, const T sigma_y)
{
    T normalization_factor = 0.0;
    const int SIZEX_HERMITIAN = SIZEX / 2 + 1;

    for(int PY = 0; PY < SIZEY; ++PY)
    {
        for(int PX = 0; PX < SIZEX_HERMITIAN; ++PX)
        {
            int u = PX <= SIZEX / 2 ? PX : PX - SIZEX;
            int v = PY <= SIZEY / 2 ? PY : PY - SIZEY;
            T sigma_x_u_over_width = sigma_x * u / SIZEX;
            T sigma_y_v_over_height = sigma_y * v / SIZEY;
            T real_ker = exp(MINUSTWOPISQUARED
                             * (sigma_x_u_over_width * sigma_x_u_over_width
                                + sigma_y_v_over_height * sigma_y_v_over_height));
            normalization_factor += real_ker;
        }
    }

    return normalization_factor;
}
*/
template <typename T, typename W>
void CUDASpectralGaussianBlur2D(
    dim3 threads, void* GPU_vec, const int SIZEX, const int SIZEY, const T sigma_x, const T sigma_y)
{
    //  T normalization_factor = ComputeNormalizationFactor(SIZEX, SIZEY, sigma_x, sigma_y);
    //  printf("Normalization factor: %f\n", normalization_factor);
    /*
    printf("CUDASpectralGaussianBlur threads=(%d, %d, %d), SIZEX=%d, SIZEY=%d, sigma_x=%f, "
           "sigma_y=%f\n",
           threads.x, threads.y, threads.z, SIZEX, SIZEY, sigma_x, sigma_y);
*/
    // Calculate the number of blocks needed
    const int SIZEX_HERMITIAN = SIZEX / 2 + 1;
    dim3 numBlocks((SIZEY + threads.x - 1) / threads.x,
                   (SIZEX_HERMITIAN + threads.y - 1) / threads.y);

    SpectralGaussianBlur2D<T, W>
        <<<numBlocks, threads>>>((W*)GPU_vec, SIZEX, SIZEY, sigma_x, sigma_y);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch SpectralGaussianBlur kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

//Explicit instance of CUDASpectralGaussianBlur2D for float and double
template void CUDASpectralGaussianBlur2D<float, cufftComplex>(dim3 threads,
                                                              void* GPU_vec,
                                                              const int SIZEX,
                                                              const int SIZEY,
                                                              const float sigma_x,
                                                              const float sigma_y);

template void CUDASpectralGaussianBlur2D<double, cufftDoubleComplex>(dim3 threads,
                                                                     void* GPU_vec,
                                                                     const int SIZEX,
                                                                     const int SIZEY,
                                                                     const double sigma_x,
                                                                     const double sigma_y);

//Explicit instance of CUDASpectralGaussianBlur1D for float and double
template void CUDASpectralGaussianBlur1D<float, cufftComplex>(
    dim3 threads, void* GPU_vec, const int SIZEX, const int SIZEY, const float sigma_z);

template void CUDASpectralGaussianBlur1D<double, cufftDoubleComplex>(
    dim3 threads, void* GPU_vec, const int SIZEX, const int SIZEY, const double sigma_z);

//Explicit instances of SpectralGaussianBlur2D for float and double
template __global__ void SpectralGaussianBlur2D<float, cufftComplex>(cufftComplex* __restrict__ VEC,
                                                                     const int SIZEX,
                                                                     const int SIZEY,
                                                                     const float sigma_x,
                                                                     const float sigma_y);

template __global__ void
SpectralGaussianBlur2D<double, cufftDoubleComplex>(cufftDoubleComplex* __restrict__ VEC,
                                                   const int SIZEX,
                                                   const int SIZEY,
                                                   const double sigma_x,
                                                   const double sigma_y);

//Explicit instances of SpectralGaussianBlur1D for float and double
template __global__ void SpectralGaussianBlur1D<float, cufftComplex>(cufftComplex* __restrict__ VEC,
                                                                     const int SIZEX,
                                                                     const int SIZEY,
                                                                     const float sigma_z);

template __global__ void SpectralGaussianBlur1D<double, cufftDoubleComplex>(
    cufftDoubleComplex* __restrict__ VEC, const int SIZEX, const int SIZEY, const double sigma_z);

template __global__ void SpectralFilter<float, cufftComplex>(cufftComplex* __restrict__ X,
                                                             const float* __restrict__ FILTER,
                                                             const int SIZEX,
                                                             const int SIZEY,
                                                             const float pixel_size_x);

template __global__ void
SpectralFilter<double, cufftDoubleComplex>(cufftDoubleComplex* __restrict__ X,
                                           const double* __restrict__ FILTER,
                                           const int SIZEX,
                                           const int SIZEY,
                                           const double pixel_size_x);

template void CUDASpectralFilter<float, cufftComplex>(dim3 threads,
                                                      void* GPU_f_in,
                                                      void* GPU_filter_in,
                                                      const int SIZEX_FULL,
                                                      const int SIZEY,
                                                      const float pixel_size_x);

template void CUDASpectralFilter<double, cufftDoubleComplex>(dim3 threads,
                                                             void* GPU_f_in,
                                                             void* GPU_filter_in,
                                                             const int SIZEX_FULL,
                                                             const int SIZEY,
                                                             const double pixel_size_x);
//IdealRamp1D
template void CUDAIdealRamp1D<float>(void* OUT_PACKED, const int SIZE_FULL);
template void CUDAIdealRamp1D<double>(void* OUT_PACKED, const int SIZE_FULL);
//RamLakKernel1D
template void CUDARamLakKernel1D<float>(void* x, const int SIZE);
template void CUDARamLakKernel1D<double>(void* x, const int SIZE);
//Windowing functions
