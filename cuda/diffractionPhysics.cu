#include "diffractionPhysics.cuh"

__global__ void envelopeConstruction(float* __restrict__ GPU_intensity,
                                     float* __restrict__ GPU_phase,
                                     float2* __restrict__ GPU_envelope,
                                     const int dimx,
                                     const int dimy,
                                     const int dimx_padded,
                                     const int dimy_padded)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    int IDX, IDX_padded;
    float2 v;
    if(PX >= dimx_padded || PY >= dimy_padded)
    {
        return;
    } else if(PX >= dimx || PY >= dimy)
    {
        IDX_padded = dimx_padded * PY + PX;
        v.x = 0.0f;
        v.y = 0.0f;
        GPU_envelope[IDX_padded] = v;
    } else
    {
        float intensity, phase, amplitude;
        IDX = dimx * PY + PX;
        IDX_padded = dimx_padded * PY + PX;
        intensity = GPU_intensity[IDX];
        phase = GPU_phase[IDX];
        amplitude = sqrt(intensity);
        v.x = amplitude * cosf(phase);
        v.y = amplitude * sinf(phase);
        GPU_envelope[IDX_padded] = v;
    }
}

void CUDAenvelopeConstruction(dim3 threads,
                              dim3 blocks,
                              void* GPU_intensity,
                              void* GPU_phase,
                              void* GPU_envelope,
                              const int dimx,
                              const int dimy,
                              const int dimx_padded,
                              const int dimy_padded)
{
    printf("CUDAenvelopeConstruction dimx=%d dimx_padded=%d dimy=%d dimy_padded=%d "
           "threads=(%d,%d,%d), blocks(%d, %d, %d) dimx=%d, dimy=%d\n",
           dimx, dimx_padded, dimy, dimy_padded, threads.x, threads.y, threads.z, blocks.x,
           blocks.y, blocks.z, dimx, dimy);
    envelopeConstruction<<<blocks, threads>>>((float*)GPU_intensity, (float*)GPU_phase,
                                              (float2*)GPU_envelope, dimx, dimy, dimx_padded,
                                              dimy_padded);
}

__global__ void envelopeDecomposition(float* __restrict__ GPU_intensity,
                                      float* __restrict__ GPU_phase,
                                      float2* __restrict__ GPU_envelope,
                                      const int dimx,
                                      const int dimy,
                                      const int dimx_padded,
                                      const int dimy_padded,
                                      const float normalizationFactor)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    if(PX < dimx && PY < dimy)
    {
        int IDX, IDX_padded;
        float2 v;
        float intensity, phase;
        IDX = dimx * PY + PX;
        IDX_padded = dimx_padded * PY + PX;
        v = GPU_envelope[IDX_padded];
        v.x *= normalizationFactor;
        v.y *= normalizationFactor;
        intensity = v.x * v.x + v.y * v.y;
        phase = atan2(v.y, v.x);
        GPU_intensity[IDX] = intensity;
        GPU_phase[IDX] = phase;
    }
}

void CUDAenvelopeDecomposition(dim3 threads,
                               dim3 blocks,
                               void* GPU_intensity,
                               void* GPU_phase,
                               void* GPU_envelope,
                               const int dimx,
                               const int dimy,
                               const int dimx_padded,
                               const int dimy_padded)
{
    printf("CUDAenvelopeDecomposition threads=(%d,%d,%d), blocks(%d, %d, %d) dimx=%d, dimy=%d\n",
           threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z, dimx, dimy);
    float normalizationFactor = 1.0f / ((float)dimx_padded * (float)dimy_padded);
    envelopeDecomposition<<<blocks, threads>>>((float*)GPU_intensity, (float*)GPU_phase,
                                               (float2*)GPU_envelope, dimx, dimy, dimx_padded,
                                               dimy_padded, normalizationFactor);
}

__global__ void spectralMultiplicationFresnel(float2* __restrict__ GPU_FTenvelope,
                                              const float lambda,
                                              const float propagationDistance,
                                              const int dimx,
                                              const int dimy,
                                              const float pixel_size_x,
                                              const float pixel_size_y)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    if(PX < dimx && PY < dimy)
    {
        int IDX = dimx * PY + PX;
        float2 v_in, v_out;
        float kx, ky;
        float detectorSizeX, detectorSizeY;
        float exponentPrefactor, exponentPhase, exponent;
        float2 kernelMultiplier;
        v_in = GPU_FTenvelope[IDX];
        detectorSizeX = dimx * pixel_size_x;
        detectorSizeY = dimy * pixel_size_y;
        if(PX <= dimx / 2)
        {
            kx = PX;
        } else
        {
            kx = PX - dimx;
        }
        if(PY <= dimy / 2)
        {
            ky = PY;
        } else
        {
            ky = PY - dimy;
        }
        kx /= detectorSizeX;
        ky /= detectorSizeY;
        exponentPhase = kx * kx + ky * ky;
        exponentPrefactor = -lambda * PI * propagationDistance;
        exponent = exponentPrefactor * exponentPhase;
        kernelMultiplier.x = cosf(exponent);
        kernelMultiplier.y = sinf(exponent);
        v_out.x = (v_in.x * kernelMultiplier.x - v_in.y * kernelMultiplier.y);
        v_out.y = (v_in.x * kernelMultiplier.y + v_in.y * kernelMultiplier.x);
        GPU_FTenvelope[IDX] = v_out;
    }
}

void CUDAspectralMultiplicationFresnel(dim3 threads,
                                       dim3 blocks,
                                       void* GPU_FTenvelope,
                                       const float lambda,
                                       const float propagationDistance,
                                       const int dimx_padded,
                                       const int dimy_padded,
                                       const float pixel_size_x,
                                       const float pixel_size_y)
{
    printf("dimx_padded=%d dimy_padded=%d pixel_size_x=%f 10^-6m pixel_size_y=%f 10^-6m\n",
           dimx_padded, dimy_padded, pixel_size_x * 1e6, pixel_size_y * 1e6);
    spectralMultiplicationFresnel<<<blocks, threads>>>((float2*)GPU_FTenvelope, lambda,
                                                       propagationDistance, dimx_padded,
                                                       dimy_padded, pixel_size_x, pixel_size_y);
}

// Assumption of evancescent waves see also
// https://mathoverflow.net/questions/442396/is-this-formula-for-2d-fourier-integral-of-diffraction-kernel-correct
__global__ void spectralMultiplicationRayleigh(float2* __restrict__ GPU_FTenvelope,
                                               const float lambda,
                                               const float propagationDistance,
                                               const int dimx,
                                               const int dimy,
                                               const float pixel_size_x,
                                               const float pixel_size_y)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    if(PX < dimx && PY < dimy)
    {
        int IDX = dimx * PY + PX;
        double kx, ky;
        float2 v_in, v_out;
        double detectorSizeX, detectorSizeY;
        double phaseLambdaBall;
        double exponentPrefactor, exponentPartialSum, exponent;
        float2 kernelMultiplier;
        detectorSizeX = dimx * pixel_size_x;
        detectorSizeY = dimy * pixel_size_y;
        if(PX <= dimx / 2)
        {
            kx = PX;
        } else
        {
            kx = PX - dimx;
        }
        if(PY <= dimy / 2)
        {
            ky = PY;
        } else
        {
            ky = PY - dimy;
        }
        kx /= detectorSizeX;
        ky /= detectorSizeY;
        phaseLambdaBall = kx * kx + ky * ky;
        phaseLambdaBall = phaseLambdaBall * lambda * lambda;
        if(phaseLambdaBall >= 1.0f)
        {
            v_out.x = 0.0f;
            v_out.y = 0.0f;
            GPU_FTenvelope[IDX] = v_out;
        } else
        {
            v_in = GPU_FTenvelope[IDX];
            exponentPrefactor = 2 * PI * propagationDistance / lambda;
            double x_partial = phaseLambdaBall *0.5;
            double x_cur = x_partial;
            exponentPartialSum = -x_partial;
            x_cur = x_cur * x_partial * 0.5;
            exponentPartialSum -= x_cur; // x^2/8
            x_cur = x_cur * x_partial;
            exponentPartialSum -= x_cur; // x^3/16
            x_cur = x_cur * x_partial * 1.25;
            exponentPartialSum -= x_cur; // 5x^4/128
            x_cur = x_cur * x_partial * 1.4;
            exponentPartialSum -= x_cur; // 7x^5/256
            exponent = exponentPrefactor * exponentPartialSum;
            kernelMultiplier.x = cos(exponent);
            kernelMultiplier.y = sin(exponent);
            v_out.x = (v_in.x * kernelMultiplier.x - v_in.y * kernelMultiplier.y);
            v_out.y = (v_in.x * kernelMultiplier.y + v_in.y * kernelMultiplier.x);
            GPU_FTenvelope[IDX] = v_out;
        }
    }
}

void CUDAspectralMultiplicationRayleigh(dim3 threads,
                                        dim3 blocks,
                                        void* GPU_FTenvelope,
                                        const float lambda,
                                        const float propagationDistance,
                                        const int dimx_padded,
                                        const int dimy_padded,
                                        const float pixel_size_x,
                                        const float pixel_size_y)
{
    spectralMultiplicationRayleigh<<<blocks, threads>>>((float2*)GPU_FTenvelope, lambda,
                                                        propagationDistance, dimx_padded,
                                                        dimy_padded, pixel_size_x, pixel_size_y);
}

// Debugging routines to export kernels

__global__ void exportKernelFresnel(float* __restrict__ GPU_kernel_re,
                                    float* __restrict__ GPU_kernel_im,
                                    const float lambda,
                                    const float propagationDistance,
                                    const int dimx,
                                    const int dimy,
                                    const float pixel_size_x,
                                    const float pixel_size_y)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    if(PX < dimx && PY < dimy)
    {
        int IDX = dimx * PY + PX;
        float kx, ky;
        float detectorSizeX, detectorSizeY;
        float exponentPrefactor, exponentPhase, exponent;
        float2 kernelMultiplier;
        detectorSizeX = dimx * pixel_size_x;
        detectorSizeY = dimy * pixel_size_y;
        if(PX <= dimx / 2)
        {
            kx = PX;
        } else
        {
            kx = PX - dimx;
        }
        if(PY <= dimy / 2)
        {
            ky = PY;
        } else
        {
            ky = PY - dimy;
        }
        kx /= detectorSizeX;
        ky /= detectorSizeY;
        exponentPhase = kx * kx + ky * ky;
        exponentPrefactor = -lambda * PI * propagationDistance;
        exponent = exponentPrefactor * exponentPhase;
        kernelMultiplier.x = cosf(exponent);
        kernelMultiplier.y = sinf(exponent);
        GPU_kernel_re[IDX] = kernelMultiplier.x;
        GPU_kernel_im[IDX] = kernelMultiplier.y;
    }
}

void CUDAexportKernelFresnel(dim3 threads,
                             dim3 blocks,
                             void* GPU_kernel_re,
                             void* GPU_kernel_im,
                             const float lambda,
                             const float propagationDistance,
                             const int dimx_padded,
                             const int dimy_padded,
                             const float pixel_size_x,
                             const float pixel_size_y)
{
    printf("CUDAexportKernelFresnel dimx_padded=%d dimy_padded=%d pixel_size_x=%f 10^-6m "
           "pixel_size_y=%f 10^-6m\n",
           dimx_padded, dimy_padded, pixel_size_x * 1e6, pixel_size_y * 1e6);
    exportKernelFresnel<<<blocks, threads>>>((float*)GPU_kernel_re, (float*)GPU_kernel_im, lambda,
                                             propagationDistance, dimx_padded, dimy_padded,
                                             pixel_size_x, pixel_size_y);
}

__global__ void exportKernelRayleigh(float* __restrict__ GPU_kernel_re,
                                     float* __restrict__ GPU_kernel_im,
                                     const float lambda,
                                     const float propagationDistance,
                                     const int dimx,
                                     const int dimy,
                                     const float pixel_size_x,
                                     const float pixel_size_y)
{
    const int PX = threadIdx.y + blockIdx.y * blockDim.y;
    const int PY = threadIdx.x + blockIdx.x * blockDim.x;
    if(PX < dimx && PY < dimy)
    {
        int IDX = dimx * PY + PX;
        double kx, ky;
        double detectorSizeX, detectorSizeY;
        double phaseLambdaBall;
        double exponentPrefactor, exponent;
        float2 kernelMultiplier;
        detectorSizeX = dimx * pixel_size_x;
        detectorSizeY = dimy * pixel_size_y;
        if(PX <= dimx / 2)
        {
            kx = PX;
        } else
        {
            kx = PX - dimx;
        }
        if(PY <= dimy / 2)
        {
            ky = PY;
        } else
        {
            ky = PY - dimy;
        }
        kx /= detectorSizeX;
        ky /= detectorSizeY;
        phaseLambdaBall = kx * kx + ky * ky;
        phaseLambdaBall = phaseLambdaBall * lambda * lambda;
        if(phaseLambdaBall >= 1.0f)
        {
            GPU_kernel_re[IDX] = 0.0f;
            GPU_kernel_im[IDX] = 0.0f;
            printf("Filterring evanescent wave for PX=%d PY=%d", PX, PY);
        } else
        {
            exponentPrefactor = 2 * PI * propagationDistance / lambda;
            /*
                exponentPostfactor = sqrt(1.0 - phaseLambdaBall);
                exponent = exponentPrefactor * (exponentPostfactor - 1); 
                //Very bad implementation since exponentPostfactor \ sim 1
            */

            // Implement sqrt(1-x) - 1 = -x/2 - x^2/8 - x^3/16-5x^4/128-7x^5/256
            double exponentPartialSum;
            double x_partial = phaseLambdaBall *0.5;
            double x_cur = x_partial;
            exponentPartialSum = -x_partial;
            x_cur = x_cur * x_partial * 0.5;
            exponentPartialSum -= x_cur; // x^2/8
            x_cur = x_cur * x_partial;
            exponentPartialSum -= x_cur; // x^3/16
            x_cur = x_cur * x_partial * 1.25;
            exponentPartialSum -= x_cur; // 5x^4/128
            x_cur = x_cur * x_partial * 1.4;
            exponentPartialSum -= x_cur; // 7x^5/256
            exponent = exponentPrefactor * exponentPartialSum;

            // Fresnel like approximation
            // exponentPrefactor = -PI * lambda * propagationDistance;
            // exponent = exponentPrefactor * (kx * kx + ky * ky);

            kernelMultiplier.x = cos(exponent);
            kernelMultiplier.y = sin(exponent);
            GPU_kernel_re[IDX] = kernelMultiplier.x;
            GPU_kernel_im[IDX] = kernelMultiplier.y;
        }
    }
}

void CUDAexportKernelRayleigh(dim3 threads,
                              dim3 blocks,
                              void* GPU_kernel_re,
                              void* GPU_kernel_im,
                              const float lambda,
                              const float propagationDistance,
                              const int dimx_padded,
                              const int dimy_padded,
                              const float pixel_size_x,
                              const float pixel_size_y)
{
    printf("CUDAexportKernelRayleigh threads=[%d %d %d] blocks=[%d %d %d], DIMX=%d DIMY=%d "
           "dimx_padded=%d "
           "dimy_padded=%d pixel_size_x=%f 10^-6m pixel_size_y=%f 10^-6m\n",
           threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z, threads.x * blocks.x,
           threads.y * blocks.y, dimx_padded, dimy_padded, pixel_size_x * 1e6, pixel_size_y * 1e6);
    exportKernelRayleigh<<<blocks, threads>>>((float*)GPU_kernel_re, (float*)GPU_kernel_im, lambda,
                                              propagationDistance, dimx_padded, dimy_padded,
                                              pixel_size_x, pixel_size_y);
}
