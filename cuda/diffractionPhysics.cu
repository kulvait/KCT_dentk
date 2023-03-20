// Logging

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
    printf("CUDAenvelopeConstruction threads=(%d,%d,%d), blocks(%d, %d, %d) dimx=%d, dimy=%d\n",
           threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z, dimx, dimy);
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
                                      const int dimy_padded)
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
    envelopeDecomposition<<<blocks, threads>>>((float*)GPU_intensity, (float*)GPU_phase,
                                               (float2*)GPU_envelope, dimx, dimy, dimx_padded,
                                               dimy_padded);
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
        v_in = GPU_FTenvelope[IDX];
        detectorSizeX = dimx * pixel_size_x;
        detectorSizeY = dimy * pixel_size_y;
        if(PX <= dimy / 2)
        {
            kx = PX;
        } else
        {
            kx = PX - dimx;
        }
        if(PY <= dimx / 2)
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
        v_out.x = v_in.x * cosf(exponent);
        v_out.y = v_in.y * sinf(exponent);
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
        float kx, ky;
        float2 v_in, v_out;
        float detectorSizeX, detectorSizeY;
        float phaseLambdaBall;
        float exponentPrefactor, exponentPostfactor, exponent;
        detectorSizeX = dimx * pixel_size_x;
        detectorSizeY = dimy * pixel_size_y;
        if(PX <= dimy / 2)
        {
            kx = PX;
        } else
        {
            kx = PX - dimx;
        }
        if(PY <= dimx / 2)
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
            exponentPostfactor = sqrt(1.0 - phaseLambdaBall);
            exponentPrefactor = 2 * PI * propagationDistance / lambda;
            exponent = exponentPrefactor
                * (exponentPostfactor - 1); // To get a wave envelope without e^ikz
            v_out.x = v_in.x * cosf(exponent);
            v_out.y = v_in.y * sinf(exponent);
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
