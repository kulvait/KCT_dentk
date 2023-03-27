// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctype.h>
#include <cuda.h> //CUDA
#include <cuda_runtime.h> //CUDA
#include <cuda_runtime_api.h>
#include <cufft.h> //Nvidia CUDA FFT
//#include <cufftw.h> //Plan dependent on particular data, not convenient when I would like it to be
// performed on multiple frames
#include "diffractionPhysics.cuh"
#include <iostream>
#include <regex>
#include <string>

// External libraries
#include "CLI/CLI.hpp" //Command line parser

// Internal libraries
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DBufferedWritter.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/ArgumentsCTDetector.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/ArgumentsVerbose.hpp"
#include "PROG/Program.hpp"
#include "PROG/parseArgs.h"
#include "ftpl.h"

using namespace KCT;
using namespace KCT::util;

// class declarations
class Args : public ArgumentsForce,
             public ArgumentsVerbose,
             public ArgumentsFramespec,
             public ArgumentsThreading,
             public ArgumentsCTDetector
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName)
        , ArgumentsVerbose(argc, argv, prgName)
        , ArgumentsFramespec(argc, argv, prgName)
        , ArgumentsThreading(argc, argv, prgName)
        , ArgumentsCTDetector(argc, argv, prgName){};
    std::string input_intensity = "";
    std::string input_phase = "";
    std::string output_intensity = "";
    std::string output_phase = "";
    uint32_t dimx, dimy, dimz;
    uint64_t frameSize;
    io::DenSupportedType dataType;

    double propagationDistance = 1;
    double waveEnergyKeV = 20;
    double lambda;
    bool outputFilesExist;

    bool paddingNone = false;
    bool paddingZero = false;
    bool paddingSymmetric = false;
    bool paddingSymmetricxZeroy = false;
    bool paddingSymmetricyZerox = false;

    bool propagatorFresnel = false;
    bool propagatorRayleigh = false;
    bool exportKernel = false;
};

void Args::defineArguments()
{
    cliApp
        ->add_option("input_intensity", input_intensity,
                     "Componenent I(x,y,0) in E = I e^{i \\Phi}.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("input_phase", input_phase, "Component \\Phi(x,y,0) in E = I e^{i \\Phi}.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp
        ->add_option("output_intensity", output_intensity,
                     "Componenent I(x,y,propagation_distance) in E = I e^{i \\Phi}.")
        ->required();
    cliApp
        ->add_option("output_phase", output_phase,
                     "Component \\Phi(x,y,propagation_distance) in E = I e^{i \\Phi}.")
        ->required();
    cliApp->add_option(
        "--propagation-distance", propagationDistance,
        io::xprintf("Propagation distance in meters, [defaults to %f].", propagationDistance));
    cliApp->add_option("--wave-energy", waveEnergyKeV,
                       io::xprintf("Energy of the wave in keV, [defaults to %f].", waveEnergyKeV));
    // Adding radio group see https://github.com/CLIUtils/CLI11/pull/234
    CLI::Option_group* op_clg
        = cliApp->add_option_group("Padding", "Padding to use for convolution.");
    op_clg->add_flag("--padding-none", paddingNone,
                     "Use no padding and perform cyclic convolution.");
    op_clg->add_flag("--padding-zero", paddingZero,
                     "Use zero padding to perform linear convolution.");
    op_clg->add_flag("--padding-symmetric", paddingSymmetric,
                     "Use symmetric padding to mitigate cross detector edge effects.");
    op_clg->add_flag("--padding-symmetricx-zeroy", paddingSymmetricxZeroy,
                     "Use symmetric padding at X dimension and zero at Y dimension.");
    op_clg->add_flag("--padding-symmetricy-zerox", paddingSymmetricyZerox,
                     "Use symmetric padding at Y dimension and zero at X dimension.");
    op_clg->require_option(1);
    op_clg = cliApp->add_option_group("Propagator operator", "Padding to use for convolution.");
    op_clg->add_flag("--fresnel", propagatorFresnel, "Use Fresnel propagator.");
    op_clg->add_flag("--rayleigh", propagatorRayleigh, "Use Rayleigh-Sommerfeld propagator.");
    op_clg->require_option(1);
    cliApp->add_flag(
        "--export-kernel", exportKernel,
        "Just export kernel, object with which multiplication in Fourier space is performed, to "
        "the output files instead of computing propagation, output_intensity "
        "will contain real and output_phase imaginary part of the kernel.");

    addForceArgs();
    addPixelSizeArgs(1.0, 1.0);
    addFramespecArgs();
    addThreadingArgs();
}

int Args::postParse()
{
    std::string ERR;
    // Test if minuend and subtraend are of the same type and dimensions
    io::DenFileInfo input_intensity_inf(input_intensity);
    io::DenFileInfo input_phase_inf(input_phase);

    dimx = input_intensity_inf.dimx();
    dimy = input_intensity_inf.dimy();
    dimz = input_intensity_inf.dimz();
    frameSize = (uint64_t)dimx * (uint64_t)dimy;
    dataType = input_intensity_inf.getElementType();
    if(input_phase_inf.dimx() != dimx || input_phase_inf.dimy() != dimy
       || input_phase_inf.dimz() != dimz || input_phase_inf.getElementType() != dataType)
    {
        ERR = io::xprintf("Dimensions of intensity %s and phase %s are incompatible!",
                          input_intensity.c_str(), input_phase.c_str());
    }
    // I will simply remove files if they exist
    int flagPhase, flagIntensity;
    flagIntensity = handleFileExistence(output_intensity, force, input_intensity_inf);
    flagPhase = handleFileExistence(output_phase, force, input_phase_inf);
    if(flagPhase == -1 && flagIntensity == -1 && !exportKernel)
    {
        outputFilesExist = true;
    } else if(flagPhase == 0 && flagIntensity == 0)
    {
        outputFilesExist = false;
    } else
    { // Remove existing file
        int flag = handleFileExistence(output_intensity, force, force);
        if(flag == 1)
        {
            return 1;
        }
        flag = handleFileExistence(output_phase, force, force);
        if(flag == 1)
        {
            return 1;
        }
        outputFilesExist = false;
    }
    // Compute lambda in meters
    double planckConstantEvOverHz = 4.135667696e-15;
    double c = 299792458.0;
    double hckev = planckConstantEvOverHz * c / 1000.0;
    lambda = hckev / waveEnergyKeV;
    fillFramesVector(dimz);
    return 0;
}

// See
// https://stackoverflow.com/a/48025679
// http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
void _assert_CUDA(cudaError_t code, const char* file, int line, bool abort = true)
{
    std::string ERR;
    if(code != cudaSuccess)
    {
        ERR = io::xprintf("CUDA Error %d: %s %s %d", code, cudaGetErrorString(code), file, line);
        if(abort)
        {
            KCTERR(ERR);
        } else
        {
            LOGE << ERR;
        }
    }
}

#define EXECUDA(INF) _assert_CUDA(INF, __FILE__, __LINE__)

static const std::string cufftResultString(cufftResult inf)
{
    switch(inf)
    {
    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
        return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_INVALID_DEVICE:
        return "CUFFT_INVALID_DEVICE";
    case CUFFT_PARSE_ERROR:
        return "CUFFT_PARSE_ERROR";
    case CUFFT_NO_WORKSPACE:
        return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:
        return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_LICENSE_ERROR:
        return "CUFFT_LICENSE_ERROR";
    case CUFFT_NOT_SUPPORTED:
        return "CUFFT_NOT_SUPPORTED";
    default:
        return io::xprintf("UNKNOWNcufftResult(%d)", inf);
    }
}

void _assert_CUFFT(cufftResult inf, const char* file, int line, bool abort = true)
{
    std::string ERR;
    if(inf != CUFFT_SUCCESS)
    {
        std::string err = cufftResultString(inf);
        ERR = io::xprintf("CUFFT Error %d: %s %s %d", inf, err.c_str(), file, line);
        if(abort)
        {
            KCTERR(ERR);
        } else
        {
            LOGE << ERR;
        }
    }
}

#define EXECUFFT(INF) _assert_CUFFT(INF, __FILE__, __LINE__)

template <typename T>
class GPUThreadPool
{
public:
    GPUThreadPool(Args ARG)
        : numWorkers(ARG.threads)
        , killWorkers(false)
        , ARG(ARG)

    {
        dataType = ARG.dataType;
        dimx = ARG.dimx;
        dimy = ARG.dimy;
        if(ARG.paddingNone)
        {
            dimx_padded = ARG.dimx;
            dimy_padded = ARG.dimy;
        } else
        {
            dimx_padded = 2 * ARG.dimx;
            dimy_padded = 2 * ARG.dimy;
        }
        frameSizePadded = (uint64_t)dimx_padded * (uint64_t)dimy_padded;
        paddingXSymmetric = ARG.paddingSymmetric || ARG.paddingSymmetricxZeroy;
        paddingYSymmetric = ARG.paddingSymmetric || ARG.paddingSymmetricyZerox;
        // GPU init
        THREADSIZE1 = 32;
        THREADSIZE2 = 32;
        threads = dim3(THREADSIZE1, THREADSIZE2);
        // Run propagator
        io::DenFileInfo di(ARG.input_intensity);
        intensityReader
            = std::make_shared<io::DenFrame2DReader<T>>(ARG.input_intensity, ARG.threads);
        phaseReader = std::make_shared<io::DenFrame2DReader<T>>(ARG.input_phase, ARG.threads);
        if(ARG.outputFilesExist)
        {
            intensityWritter = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(
                ARG.output_intensity, ARG.dimx, ARG.dimy, ARG.dimz);
            phaseWritter = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(
                ARG.output_phase, ARG.dimx, ARG.dimy, ARG.dimz);
        } else
        {
            intensityWritter = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(
                ARG.output_intensity, ARG.dimx, ARG.dimy, ARG.frames.size());
            phaseWritter = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(
                ARG.output_phase, ARG.dimx, ARG.dimy, ARG.frames.size());
        }

        for(uint32_t workerID = 0; workerID < numWorkers; workerID++)
        {
            std::thread worker([this, workerID]() {
                while(true)
                {
                    std::function<void(uint32_t)> task;
                    {
                        std::unique_lock lock(lck);
                        cv.wait(lock,
                                [this]() { return !frameProcessTasks.empty() || killWorkers; });
                        if(frameProcessTasks.empty() && killWorkers)
                            return;
                        task = std::move(frameProcessTasks.front());
                        frameProcessTasks.pop();
                    }
                    task(workerID);
                }
            });
        }
        if(numWorkers > 0)
        {
            GPUResourcesCount = numWorkers;
        } else
        {
            GPUResourcesCount = 1; // No threading
        }
        for(uint32_t workerID = 0; workerID < GPUResourcesCount; workerID++)
        {
            cufftHandle FFT;
            void* GPU_intensity;
            void* GPU_phase;
            void* GPU_envelope;
            void* GPU_FTenvelope;
            EXECUFFT(cufftPlan2d(&FFT, dimy_padded, dimx_padded, CUFFT_C2C));
            EXECUDA(cudaMalloc((void**)&GPU_intensity, ARG.frameSize * sizeof(T)));
            EXECUDA(cudaMalloc((void**)&GPU_phase, ARG.frameSize * sizeof(T)));
            EXECUDA(cudaMalloc((void**)&GPU_envelope, frameSizePadded * 2 * sizeof(T)));
            EXECUDA(cudaMalloc((void**)&GPU_FTenvelope, frameSizePadded * 2 * sizeof(T)));
            FFT_handles.emplace_back(std::move(FFT));
            _GPU_intensity.emplace_back(GPU_intensity);
            _GPU_phase.emplace_back(GPU_phase);
            _GPU_envelope.emplace_back(GPU_envelope);
            _GPU_FTenvelope.emplace_back(GPU_FTenvelope);
        }
    }

    void process()
    {
        if(GPUResourcesCount == 0)
        {
            std::string ERR = io::xprintf("Out of GPU!");
            KCTERR(ERR);
        }

        uint32_t k_in, k_out;
        for(uint32_t IND = 0; IND != ARG.frames.size(); IND++)
        {
            k_in = ARG.frames[IND];
            if(ARG.outputFilesExist)
            {
                k_out = k_in; // To be able to do dentk-calc --force --multiply -f 0,end zero.den
                              // BETA.den BETA.den
            } else
            {
                k_out = IND;
            }
            LOGI << io::xprintf("k_in=%d k_out=%d", k_in, k_out);
            std::function<void(uint32_t)> task;
            if(dataType == io::DenSupportedType::FLOAT32)
            {
                task = std::bind(&GPUThreadPool::processFrameFloat, this, std::placeholders::_1,
                                 k_in, k_out);
            } else if(dataType == io::DenSupportedType::FLOAT64)
            {
                task = std::bind(&GPUThreadPool::processFrameDouble, this, std::placeholders::_1,
                                 k_in, k_out);
            } else
            {
                std::string ERR = io::xprintf("Unknown dataType=%s",
                                              DenSupportedTypeToString(dataType).c_str());
                KCTERR(ERR);
            }

            if(numWorkers > 0)
            {
                std::lock_guard<std::mutex> lock(lck);
                frameProcessTasks.push(task);
            } else
            {
                uint32_t workerID = 0;
                task(workerID);
            }
        }
        finishProcessing();
    }

    virtual ~GPUThreadPool() { finishProcessing(); }

private:
    GPUThreadPool(const GPUThreadPool&) = delete;
    GPUThreadPool(GPUThreadPool&&) = delete;
    GPUThreadPool& operator=(const GPUThreadPool&) = delete;
    GPUThreadPool& operator=(GPUThreadPool&&) = delete;

    void processFrameFloat(uint32_t workerID, uint32_t k_in, uint32_t k_out)
    {
        cufftHandle FFT = FFT_handles[workerID];
        void* GPU_intensity = _GPU_intensity[workerID];
        void* GPU_phase = _GPU_phase[workerID];
        void* GPU_envelope = _GPU_envelope[workerID];
        void* GPU_FTenvelope = _GPU_FTenvelope[workerID];

        std::shared_ptr<io::BufferedFrame2D<T>> I_f = intensityReader->readBufferedFrame(k_in);
        std::shared_ptr<io::BufferedFrame2D<T>> P_f = phaseReader->readBufferedFrame(k_in);
        // We transform it to E_0 and decompose back to I and P in GPU memmory
        T* I_array = I_f->getDataPointer();
        T* P_array = P_f->getDataPointer();

        // Do something here
        // Try without distinguishing types
        EXECUDA(cudaMemcpy((void*)GPU_intensity, (void*)I_array, ARG.frameSize * sizeof(T),
                           cudaMemcpyHostToDevice));
        EXECUDA(cudaMemcpy((void*)GPU_phase, (void*)P_array, ARG.frameSize * sizeof(T),
                           cudaMemcpyHostToDevice));

        // Different to the padding code
        dim3 unpaddedBlocks((dimy + THREADSIZE1 - 1) / THREADSIZE1,
                            (dimx + THREADSIZE2 - 1) / THREADSIZE2);
        dim3 paddedBlocks((dimy_padded + THREADSIZE1 - 1) / THREADSIZE1,
                          (dimx_padded + THREADSIZE2 - 1) / THREADSIZE2);
        CUDAenvelopeConstruction(threads, unpaddedBlocks, GPU_intensity, GPU_phase, GPU_envelope,
                                 dimx, dimy, dimx_padded, dimy_padded, paddingXSymmetric,
                                 paddingYSymmetric);
        EXECUDA(cudaPeekAtLastError());
        EXECUDA(cudaDeviceSynchronize());
        EXECUFFT(cufftExecC2C(FFT, (cufftComplex*)GPU_envelope, (cufftComplex*)GPU_FTenvelope,
                              CUFFT_FORWARD));
        if(ARG.propagatorFresnel)
        {
            CUDAspectralMultiplicationFresnel(threads, paddedBlocks, GPU_FTenvelope, ARG.lambda,
                                              ARG.propagationDistance, dimx_padded, dimy_padded,
                                              ARG.pixelSizeX * 1e-3, ARG.pixelSizeY * 1e-3);

        } else if(ARG.propagatorRayleigh)
        {
            CUDAspectralMultiplicationRayleigh(threads, paddedBlocks, GPU_FTenvelope, ARG.lambda,
                                               ARG.propagationDistance, dimx_padded, dimy_padded,
                                               ARG.pixelSizeX * 1e-3, ARG.pixelSizeY * 1e-3);
        } else
        {
            KCTERR("No propagator specified");
        }
        EXECUDA(cudaPeekAtLastError());
        EXECUDA(cudaDeviceSynchronize());
        EXECUFFT(cufftExecC2C(FFT, (cufftComplex*)GPU_FTenvelope, (cufftComplex*)GPU_envelope,
                              CUFFT_INVERSE));

        CUDAenvelopeDecomposition(threads, unpaddedBlocks, GPU_intensity, GPU_phase, GPU_envelope,
                                  dimx, dimy, dimx_padded, dimy_padded);
        EXECUDA(cudaMemcpy((void*)I_array, (void*)GPU_intensity, ARG.frameSize * sizeof(T),
                           cudaMemcpyDeviceToHost));
        EXECUDA(cudaMemcpy((void*)P_array, (void*)GPU_phase, ARG.frameSize * sizeof(T),
                           cudaMemcpyDeviceToHost));
        intensityWritter->writeBufferedFrame(*I_f, k_out);
        phaseWritter->writeBufferedFrame(*P_f, k_out);
        if(ARG.verbose)
        {
            uint32_t dimz = phaseWritter->dimz();
            if(k_in == k_out)
            {
                LOGD << io::xprintf("Processed frame %d/%d.", k_in, dimz);
            } else
            {
                LOGD << io::xprintf("Processed frame %d->%d/%d.", k_in, k_out, dimz);
            }
        }
    }

    void processFrameDouble(uint32_t workerID, uint32_t k_in, uint32_t k_out)
    {
        KCTERR("Not yet implemented, need to implement double CUDA functions.");
    }

    void finishProcessing()
    {
        if(numWorkers > 0)
        {
            {
                std::lock_guard lock(lck);
                killWorkers = true;
            }
            cv.notify_all();
            for(auto& worker : workers)
            {
                worker.join();
            }
        }
        workers.clear();
        numWorkers = 0;
        for(uint32_t i = 0; i != GPUResourcesCount; i++)
        {
            void* GPU_intensity = _GPU_intensity[i];
            void* GPU_phase = _GPU_phase[i];
            void* GPU_envelope = _GPU_envelope[i];
            void* GPU_FTenvelope = _GPU_FTenvelope[i];
            EXECUDA(cudaFree(GPU_intensity));
            EXECUDA(cudaFree(GPU_phase));
            EXECUDA(cudaFree(GPU_envelope));
            EXECUDA(cudaFree(GPU_FTenvelope));
        }
        _GPU_intensity.clear();
        _GPU_phase.clear();
        _GPU_envelope.clear();
        _GPU_FTenvelope.clear();
        _GPU_FTenvelope.clear();
        GPUResourcesCount = 0;
    }

    uint32_t numWorkers;
    std::vector<std::thread> workers;
    std::queue<std::function<void(uint32_t)>> frameProcessTasks;

    std::mutex lck;
    std::condition_variable cv;
    bool killWorkers = false;

    // GPU items
    uint32_t THREADSIZE1 = 32;
    uint32_t THREADSIZE2 = 32;
    dim3 threads;
    std::vector<cufftHandle> FFT_handles;
    std::vector<void*> _GPU_intensity;
    std::vector<void*> _GPU_phase;
    std::vector<void*> _GPU_envelope;
    std::vector<void*> _GPU_FTenvelope;
    uint32_t GPUResourcesCount;
    // Program params
    Args ARG;
    io::DenSupportedType dataType;
    uint32_t dimx, dimy;
    uint32_t dimx_padded, dimy_padded, frameSizePadded;
    bool paddingXSymmetric;
    bool paddingYSymmetric;
    std::shared_ptr<io::DenFrame2DReader<T>> intensityReader, phaseReader;
    std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>> intensityWritter, phaseWritter;
};

template <typename T>
void exportKernelFloat(Args ARG,
                       io::DenSupportedType dataType,
                       std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>>& kernelReWritter,
                       std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>>& kernelImWritter)
{
    uint32_t dimx, dimy, dimx_padded, dimy_padded;
    dimx = ARG.dimx;
    dimy = ARG.dimy;
    if(ARG.paddingNone)
    {
        dimx_padded = ARG.dimx;
        dimy_padded = ARG.dimy;
    } else
    {
        dimx_padded = 2 * ARG.dimx;
        dimy_padded = 2 * ARG.dimy;
    }
    uint64_t frameSizePadded = (uint64_t)dimx_padded * (uint64_t)dimy_padded;
    uint32_t THREADSIZE1 = 32;
    uint32_t THREADSIZE2 = 32;
    dim3 threads(THREADSIZE1, THREADSIZE2);
    // Do something here
    // Try without distinguishing types
    io::BufferedFrame2D<T> kernel_re_f(T(0), dimx_padded, dimy_padded);
    io::BufferedFrame2D<T> kernel_im_f(T(0), dimx_padded, dimy_padded);
    T* K_re_array = kernel_re_f.getDataPointer();
    T* K_im_array = kernel_im_f.getDataPointer();
    void* GPU_kernel_re;
    void* GPU_kernel_im;
    EXECUDA(cudaMalloc((void**)&GPU_kernel_re, frameSizePadded * sizeof(T)));
    EXECUDA(cudaMalloc((void**)&GPU_kernel_im, frameSizePadded * sizeof(T)));

    // Different to the padding code
    dim3 blocks((dimy_padded + THREADSIZE1 - 1) / THREADSIZE1,
                (dimx_padded + THREADSIZE2 - 1) / THREADSIZE2);
    if(ARG.propagatorFresnel)
    {
        CUDAexportKernelFresnel(threads, blocks, GPU_kernel_re, GPU_kernel_im, ARG.lambda,
                                ARG.propagationDistance, dimx_padded, dimy_padded,
                                ARG.pixelSizeX * 1e-3, ARG.pixelSizeY * 1e-3);

    } else if(ARG.propagatorRayleigh)
    {
        CUDAexportKernelRayleigh(threads, blocks, GPU_kernel_re, GPU_kernel_im, ARG.lambda,
                                 ARG.propagationDistance, dimx_padded, dimy_padded,
                                 ARG.pixelSizeX * 1e-3, ARG.pixelSizeY * 1e-3);
    } else
    {
        KCTERR("No propagator specified");
    }
    EXECUDA(cudaPeekAtLastError());
    EXECUDA(cudaDeviceSynchronize());
    EXECUDA(cudaMemcpy((void*)K_re_array, (void*)GPU_kernel_re, frameSizePadded * sizeof(T),
                       cudaMemcpyDeviceToHost));
    EXECUDA(cudaMemcpy((void*)K_im_array, (void*)GPU_kernel_im, frameSizePadded * sizeof(T),
                       cudaMemcpyDeviceToHost));
    EXECUDA(cudaFree(GPU_kernel_re));
    EXECUDA(cudaFree(GPU_kernel_im));
    kernelReWritter->writeBufferedFrame(kernel_re_f, 0);
    kernelImWritter->writeBufferedFrame(kernel_im_f, 0);
    if(ARG.verbose)
    {
        LOGD << io::xprintf("Kernel was exported.");
    }
}

template <typename T>
void exportKernelDouble(Args ARG,
                        io::DenSupportedType dataType,
                        std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>>& kernelReWritter,
                        std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>>& kernelImWritter)
{
    KCTERR("Not yet implemented, need to implement double CUDA functions.");
}

template <typename T>
void exportKernel(Args ARG,
                  io::DenSupportedType dataType,
                  std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>>& kernelReWritter,
                  std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>>& kernelImWritter)
{
    if(dataType == io::DenSupportedType::FLOAT32)
    {
        exportKernelFloat<T>(ARG, dataType, kernelReWritter, kernelImWritter);
    } else if(dataType == io::DenSupportedType::FLOAT64)
    {
        exportKernelDouble<T>(ARG, dataType, kernelReWritter, kernelImWritter);
    }
}

template <typename T>
void processFiles(Args ARG, io::DenSupportedType dataType)
{
    if(ARG.exportKernel)
    {
        uint32_t dimx_padded, dimy_padded;
        if(ARG.paddingNone)
        {
            dimx_padded = ARG.dimx;
            dimy_padded = ARG.dimy;
        } else
        {
            dimx_padded = 2 * ARG.dimx;
            dimy_padded = 2 * ARG.dimy;
        }
        std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>> outputIntensityWritter,
            kernelImWritter;
        outputIntensityWritter = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(
            ARG.output_intensity, dimx_padded, dimy_padded, 1);
        kernelImWritter = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(
            ARG.output_phase, dimx_padded, dimy_padded, 1);
        exportKernel<T>(ARG, dataType, outputIntensityWritter, kernelImWritter);
    } else
    {
        GPUThreadPool<T> GTP(ARG);
        GTP.process();
    }
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    const std::string prgInfo = "Implements Fresnel and Rayleigh-Sommerfeld propagators of the "
                                "discrete plane wave  with CUDA.";
    Args ARG(argc, argv, prgInfo);
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    // After init parsing arguments
    io::DenFileInfo di(ARG.input_intensity);
    io::DenSupportedType dataType = di.getElementType();
    switch(dataType)
    {
    case io::DenSupportedType::FLOAT32: {
        processFiles<float>(ARG, dataType);
        break;
    }
    case io::DenSupportedType::FLOAT64: {
        processFiles<double>(ARG, dataType);
        break;
    }
    default: {
        std::string errMsg = io::xprintf("Unsupported data type %s.",
                                         io::DenSupportedTypeToString(dataType).c_str());
        KCTERR(errMsg);
    }
    }
    PRG.endLog();
}

