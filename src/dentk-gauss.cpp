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
#include "padding.cuh"
#include "tomographicFiltering.cuh"
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
#include "PROG/ThreadPool.hpp"
#include "PROG/parseArgs.h"

using namespace KCT;
using namespace KCT::util;

template <typename T>
using READER = io::DenFrame2DReader<T>;

template <typename T>
using READERPTR = std::shared_ptr<READER<T>>;

template <typename T>
using WRITER = io::DenAsyncFrame2DBufferedWritter<T>;

template <typename T>
using WRITERPTR = std::shared_ptr<WRITER<T>>;

template <typename T>
using FRAME = io::BufferedFrame2D<T>;

template <typename T>
using FRAMEPTR = std::shared_ptr<FRAME<T>>;

//Enum used for computation of the running average
enum PaddingMode { NOPAD, ZEROPAD, SYMPAD };

// class declarations
class Args : public ArgumentsForce,
             public ArgumentsVerbose,
             public ArgumentsFramespec,
             public ArgumentsThreading
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
        , ArgumentsThreading(argc, argv, prgName){};
    std::string input_den = "";
    std::string output_den = "";
    double sigma_x = 0.0;
    double sigma_y = 0.0;
    double sigma_z = 0.0;
    PaddingMode zpadding = NOPAD;
    PaddingMode xypadding = NOPAD;
    uint32_t dimx, dimy, dimz;
    uint64_t frameSize;
    uint64_t totalSize;
    //To remove
    double pixelSizeX = 1.0;
    double pixelSizeY = 1.0;
};

void Args::defineArguments()
{
    cliApp->add_option("input_den", input_den, "Input file to add Gaussian blur.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("output_den", output_den, "Output file with Gaussian blur.")->required();
    cliApp
        ->add_option("--sigma-x", sigma_x, "Sigma in x dimension for Gaussian blur [default=0.0].")
        ->check(CLI::NonNegativeNumber);
    cliApp
        ->add_option("--sigma-y", sigma_y, "Sigma in y dimension for Gaussian blur [default=0.0].")
        ->check(CLI::NonNegativeNumber);
    cliApp
        ->add_option("--sigma-z", sigma_z, "Sigma in z dimension for Gaussian blur [default=0.0].")
        ->check(CLI::NonNegativeNumber);
    CLI::Option_group* op_xpd = cliApp->add_option_group(
        "Padding in xy dimension", "Padding in xy dimension, defaults to PaddingMode::NOPAD.");
    op_xpd->add_flag_function(
        "--padxy-wrap", [this](std::int64_t count) { xypadding = PaddingMode::NOPAD; },
        "Wrap data circularly in xy dimension causing wrap in Fourier domain convolution, "
        "PaddingMode::NOPAD.");
    op_xpd->add_flag_function(
        "--padxy-zero", [this](std::int64_t count) { xypadding = PaddingMode::ZEROPAD; },
        "Zero pad data in xy dimension in xy dimension extending Fourier domain, "
        "PaddingMode::ZEROPAD.");
    op_xpd->add_flag_function(
        "--padxy-sym", [this](std::int64_t count) { xypadding = PaddingMode::SYMPAD; },
        "Symmetrically pad data in xy dimension extending Fourier domain, PaddingMode::SYMPAD.");
    op_xpd->require_option(0, 1);

    CLI::Option_group* op_zpd
        = cliApp->add_option_group("Padding in z dimension", "Padding in z dimension.");
    op_zpd->add_flag_function(
        "--padz-wrap", [this](std::int64_t count) { zpadding = PaddingMode::NOPAD; },
        "Wrap data circularly in z dimension causing wrap in Fourier domain convolution, "
        "PaddingMode::NOPAD.");
    op_zpd->add_flag_function(
        "--padz-zero", [this](std::int64_t count) { zpadding = PaddingMode::ZEROPAD; },
        "Zero pad data in z dimension in z dimension extending Fourier domain, "
        "PaddingMode::ZEROPAD.");
    op_zpd->add_flag_function(
        "--padz-sym", [this](std::int64_t count) { zpadding = PaddingMode::SYMPAD; },
        "Symmetrically pad data in z dimension extending Fourier domain, PaddingMode::SYMPAD.");
    op_zpd->require_option(0, 1);
    addForceArgs();
    addFramespecArgs();
    addThreadingArgs();
}

int Args::postParse()
{
    // Test if minuend and subtraend are of the same type and dimensions
    io::DenFileInfo input_den_inf(input_den);
    bool removeIfExists = force;
    int existFlag = handleFileExistence(output_den, force, removeIfExists);
    if(existFlag == 1)
    {
        return 1;
    }
    dimx = input_den_inf.dimx();
    dimy = input_den_inf.dimy();
    dimz = input_den_inf.dimz();
    frameSize = static_cast<uint64_t>(dimx) * static_cast<uint64_t>(dimy);
    LOGI << io::xprintf("Filling frames vector with dimz=%d.", input_den_inf.getFrameCount());
    fillFramesVector(input_den_inf.getFrameCount());
    LOGD << io::xprintf("Total number of frames: %d", frames.size());
    totalSize = frameSize * static_cast<uint64_t>(frames.size());
    return 0;
}

// See

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

#ifndef EXECUDA
#define EXECUDA(INF) _assert_CUDA(INF, __FILE__, __LINE__)
#endif

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
struct WORKER
{
    WORKER(Args& ARG)
        : ARG(ARG)
        , dataType(io::DenSupportedType::FLOAT32)
        , CUDADeviceID(0)
        , isGPUInitialized(false)
        , GPU_f(nullptr)
        , GPU_extendedf(nullptr)
        , GPU_FTf(nullptr)
        , padding(false)
        , dimx(0)
        , dimy(0)
        , dimx_padded(0)
        , dimy_padded(0)
        , dimx_padded_hermitian(0)
        , frameSize_padded(0)
        , frameSize_padded_complex(0){};
    Args ARG;
    io::DenSupportedType dataType;
    cufftHandle FFT;
    cufftHandle IFT;
    uint32_t CUDADeviceID;
    bool isGPUInitialized;
    void* GPU_f;
    void* GPU_extendedf;
    void* GPU_FTf;
    bool padding;
    uint64_t dimx;
    uint64_t dimy;
    uint64_t dimx_padded;
    uint64_t dimy_padded;
    uint64_t dimx_padded_hermitian;
    uint64_t frameSize_padded;
    uint64_t frameSize_padded_complex;
};

template <typename T>
struct WORKER1D
{
    WORKER1D(Args& ARG)
        : ARG(ARG)
        , dataType(io::DenSupportedType::FLOAT32)
        , CUDADeviceID(0)
        , isGPUInitialized(false)
        , GPU_f(nullptr)
        , GPU_extendedf(nullptr)
        , GPU_FTf(nullptr)
        , padding(false)
        , dimx(0)
        , dimx_padded(0)
        , dimx_padded_hermitian(0)
        , chunkSize(0)
        , totalArrayCount(0){};
    Args ARG;
    io::DenSupportedType dataType;
    cufftHandle FFT;
    cufftHandle IFT;
    uint32_t CUDADeviceID;
    bool isGPUInitialized;
    void* GPU_f;
    void* GPU_extendedf;
    void* GPU_FTf;
    bool padding;
    uint64_t dimx;
    uint64_t dimx_padded;
    uint64_t dimx_padded_hermitian;
    uint64_t chunkSize;
    uint64_t chunkSize_times_dimx;
    uint64_t chunkSize_times_dimx_padded;
    uint64_t chunkSize_times_dimx_padded_hermitian;
    uint64_t totalArrayCount;
};

void checkCudaError(cudaError_t result, const char* func)
{
    if(result != cudaSuccess)
    {
        std::cerr << "CUDA error in " << func << ": " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void queryGPUs(std::vector<size_t>& gpuMemSizes)
{
    int deviceCount;
    checkCudaError(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");

    for(int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp prop;
        checkCudaError(cudaGetDeviceProperties(&prop, i), "cudaGetDeviceProperties");
        gpuMemSizes.push_back(prop.totalGlobalMem);
        std::cout << "GPU " << i << ": " << prop.name
                  << ", Total Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB"
                  << std::endl;
    }
}

size_t estimateChunkSize(const std::vector<size_t>& gpuMemSizes, size_t blurrArrayLength)
{
    size_t maxChunkSize = 0;
    size_t requiredMemoryPerChunk = 5 * blurrArrayLength * sizeof(double);
    size_t arrayLengthPadded = 2 * blurrArrayLength - 2;
    size_t requiredMemoryFFT
        = (arrayLengthPadded + 2 * (arrayLengthPadded / 2 + 1)) * sizeof(double);
    size_t requiredMemoryBuffers = requiredMemoryFFT + blurrArrayLength * sizeof(double);
    size_t requiredMemoryTotal = requiredMemoryPerChunk + requiredMemoryBuffers;

    for(size_t memSize : gpuMemSizes)
    {
        size_t availableMemory = static_cast<size_t>(0.8 * memSize); // Use 80% of GPU memory
        size_t chunkSize = availableMemory / requiredMemoryTotal;
        if(chunkSize > maxChunkSize)
        {
            maxChunkSize = chunkSize;
        }
    }
    return maxChunkSize;
}

template <typename T>
using WORKER1DPTR = std::shared_ptr<WORKER1D<T>>;

template <typename T>
using TP1D = io::ThreadPool<WORKER1D<T>>;

template <typename T>
using TP1DPTR = std::shared_ptr<TP1D<T>>;

template <typename T>
using TP1DINFO = typename TP1D<T>::ThreadInfo;

template <typename T>
using TP1DINFOPTR = std::shared_ptr<TP1DINFO<T>>;

template <typename T>
void processZ(std::shared_ptr<typename TP1D<T>::ThreadInfo> threadInfo, uint32_t k_in, T* data)
{
    std::shared_ptr<WORKER1D<T>> worker = threadInfo->worker;
    if(!worker->isGPUInitialized)
    {
        EXECUDA(cudaSetDevice(worker->CUDADeviceID));
        if(worker->ARG.zpadding == NOPAD)
        {
            //Initialize worker
            worker->padding = false;
            worker->dimx_padded = worker->dimx;
            worker->dimx_padded_hermitian = worker->dimx_padded / 2 + 1;
            worker->chunkSize_times_dimx = worker->chunkSize * worker->dimx;
            worker->chunkSize_times_dimx_padded = worker->chunkSize * worker->dimx_padded;
            worker->chunkSize_times_dimx_padded_hermitian
                = worker->chunkSize * worker->dimx_padded_hermitian;
            //Device memory allocation for GPU_f and GPU_FTf, GPU_extendedf is not needed to be allocated
            EXECUDA(cudaMalloc((void**)&worker->GPU_f, worker->chunkSize_times_dimx * sizeof(T)));
            EXECUDA(cudaMalloc((void**)&worker->GPU_FTf,
                               worker->chunkSize_times_dimx_padded_hermitian * 2 * sizeof(T)));
        } else
        {
            //Initialize worker
            worker->padding = true;
            worker->dimx_padded = 2 * worker->dimx - 2; //Proper padding for DFT symmetry
            worker->dimx_padded_hermitian = worker->dimx_padded / 2 + 1;
            worker->chunkSize_times_dimx = worker->chunkSize * worker->dimx;
            worker->chunkSize_times_dimx_padded = worker->chunkSize * worker->dimx_padded;
            worker->chunkSize_times_dimx_padded_hermitian
                = worker->chunkSize * worker->dimx_padded_hermitian;
            //Device memory allocation
            EXECUDA(cudaMalloc((void**)&worker->GPU_f, worker->chunkSize_times_dimx * sizeof(T)));
            EXECUDA(cudaMalloc((void**)&worker->GPU_extendedf,
                               worker->chunkSize_times_dimx_padded * sizeof(T)));
            EXECUDA(cudaMalloc((void**)&worker->GPU_FTf,
                               worker->chunkSize_times_dimx_padded_hermitian * 2 * sizeof(T)));
        }
        if(worker->dataType == io::DenSupportedType::FLOAT32)
        {
            EXECUFFT(cufftPlan1d(&worker->FFT, worker->dimx_padded, CUFFT_R2C, worker->chunkSize));
            EXECUFFT(cufftPlan1d(&worker->IFT, worker->dimx_padded, CUFFT_C2R, worker->chunkSize));
        } else if(worker->dataType == io::DenSupportedType::FLOAT64)
        {
            EXECUFFT(cufftPlan1d(&worker->FFT, worker->dimx_padded, CUFFT_D2Z, worker->chunkSize));
            EXECUFFT(cufftPlan1d(&worker->IFT, worker->dimx_padded, CUFFT_Z2D, worker->chunkSize));
        }
        worker->isGPUInitialized = true;
    } else
    {
        //This is needed to be done in case the worker is already initialized and the device ID is different
        EXECUDA(cudaSetDevice(worker->CUDADeviceID));
    }
    PaddingMode zpadding = worker->ARG.zpadding;
    T sigma_z = worker->ARG.sigma_z;
    //cufftHandle FFT = worker->FFT;
    //We read proper memory block into GPU memory
    T* framePointer = data + k_in * worker->chunkSize_times_dimx;
    //Copy data to host memory
    size_t currentChunkSize = worker->chunkSize;
    size_t currentChunkSize_times_dimx;
    if(k_in * worker->chunkSize + worker->chunkSize > worker->totalArrayCount)
    {
        currentChunkSize = worker->totalArrayCount - k_in * worker->chunkSize;
        currentChunkSize_times_dimx = currentChunkSize * worker->dimx;
    } else
    {
        currentChunkSize = worker->chunkSize;
        currentChunkSize_times_dimx = currentChunkSize * worker->dimx;
    }
    EXECUDA(cudaMemcpy((void*)worker->GPU_f, (void*)framePointer,
                       currentChunkSize_times_dimx * sizeof(T), cudaMemcpyHostToDevice));
    dim3 threads(16, 16);
    //Do padding if needed or at least do pointer copy so that the GPU data are pointed by worker->GPU_extendedf and their size is worker->dimx_padded*worker->dimy_padded
    if(zpadding == NOPAD)
    {

        worker->GPU_extendedf = worker->GPU_f;
    } else
    {
        if(zpadding == SYMPAD)
        { //Neumann extension provides exactly symmetric padding
            CUDASymmPad<T>(threads, worker->GPU_f, worker->GPU_extendedf, worker->dimx,
                           currentChunkSize, worker->dimx_padded);
        } else if(zpadding == ZEROPAD)
        {
            CUDAZeroPad<T>(threads, worker->GPU_f, worker->GPU_extendedf, worker->dimx,
                           currentChunkSize, worker->dimx_padded);
        }
        EXECUDA(cudaPeekAtLastError());
        EXECUDA(cudaDeviceSynchronize());
    }
    //FFT of GPU_extendedf
    if(worker->dataType == io::DenSupportedType::FLOAT32)
    {
        EXECUFFT(cufftExecR2C(worker->FFT, (cufftReal*)worker->GPU_extendedf,
                              (cufftComplex*)worker->GPU_FTf));
    } else if(worker->dataType == io::DenSupportedType::FLOAT64)
    {
        EXECUFFT(cufftExecD2Z(worker->FFT, (cufftDoubleReal*)worker->GPU_extendedf,
                              (cufftDoubleComplex*)worker->GPU_FTf));
    }
    //Multiplication with the Gaussian kernel
    if(worker->dataType == io::DenSupportedType::FLOAT32)
    {
        CUDASpectralGaussianBlur1D<float, cufftComplex>(
            threads, worker->GPU_FTf, worker->dimx_padded, currentChunkSize, sigma_z);
    } else if(worker->dataType == io::DenSupportedType::FLOAT64)
    {
        CUDASpectralGaussianBlur1D<double, cufftDoubleComplex>(
            threads, worker->GPU_FTf, worker->dimx_padded, currentChunkSize, sigma_z);
    }
    //Perform IFFT
    if(worker->dataType == io::DenSupportedType::FLOAT32)
    {
        EXECUFFT(cufftExecC2R(worker->IFT, (cufftComplex*)worker->GPU_FTf,
                              (cufftReal*)worker->GPU_extendedf));
    } else if(worker->dataType == io::DenSupportedType::FLOAT64)
    {
        EXECUFFT(cufftExecZ2D(worker->IFT, (cufftDoubleComplex*)worker->GPU_FTf,
                              (cufftDoubleReal*)worker->GPU_extendedf));
    }
    //Remove padding if needed
    if(zpadding == NOPAD)
    {
        worker->GPU_f = worker->GPU_extendedf;
    } else
    {
        CUDARemovePadding<T>(threads, worker->GPU_extendedf, worker->GPU_f, worker->dimx,
                             currentChunkSize, worker->dimx_padded);
    }
    //Copy data back to host memory
    EXECUDA(cudaMemcpy((void*)framePointer, (void*)worker->GPU_f,
                       currentChunkSize_times_dimx * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void processTransformed1DData(Args& ARG,
                              io::DenSupportedType dataType,
                              T* dataBuffer,
                              uint64_t blurrArrayLength,
                              uint64_t blurrArrayCount)

{
    std::vector<size_t> gpuMemSizes;
    queryGPUs(gpuMemSizes);
    uint64_t deviceCount = gpuMemSizes.size();
    size_t heuristicChunkSize = estimateChunkSize(gpuMemSizes, blurrArrayLength);
    size_t heuristicChunkCount = (blurrArrayCount + heuristicChunkSize - 1) / heuristicChunkSize;
    LOGI << io::xprintf("Estimated heuristicChunkSize=%lu with given blurrArrayLength=%lu means "
                        "that blurArrayCount=%lu fits %d times",
                        heuristicChunkSize, blurrArrayLength, blurrArrayCount, heuristicChunkCount);
    size_t chunkSize = std::min(heuristicChunkSize, blurrArrayCount);
    size_t chunkCount = (blurrArrayCount + chunkSize - 1) / chunkSize;
    if(chunkCount < ARG.dimx && chunkCount < ARG.dimy)
    {
        if(heuristicChunkCount < gpuMemSizes.size())
        {
            chunkCount = gpuMemSizes.size();
            chunkSize = (blurrArrayCount + chunkCount - 1) / chunkCount;
        }
    }
    LOGI << io::xprintf("Processing %d chunks of size %d", chunkCount, chunkSize);
    uint64_t threadCount = std::max(1lu, std::min(static_cast<uint64_t>(ARG.threads), chunkCount));
    threadCount = std::min(threadCount, deviceCount);
    std::vector<WORKER1DPTR<T>> workers;
    std::shared_ptr<WORKER1D<T>> worker;
    for(uint64_t t = 0; t < threadCount; t++)
    {
        worker = std::make_shared<WORKER1D<T>>(ARG);
        worker->ARG = ARG;
        worker->dataType = dataType;
        worker->CUDADeviceID = t % deviceCount;
        worker->isGPUInitialized = false;
        worker->chunkSize = chunkSize;
        worker->dimx = blurrArrayLength;
        worker->totalArrayCount = blurrArrayCount;
        workers.push_back(worker);
    }
    TP1DPTR<T> threadpool = std::make_shared<TP1D<T>>(threadCount, workers);
    /*
    worker = workers[0];
    TP1DINFOPTR<T> threadInfo
        = std::make_shared<typename TP1D<T>::ThreadInfo>(TP1DINFO<T>{ 0, 0, workers[0] });
    for(uint64_t k = 0; k < chunkCount; k++)
    {
        LOGI << io::xprintf("Processing chunk %d/%d", k, chunkCount);
        LOGI << io::xprintf("worker->CUDADeviceID=%d, worker->dataType=%d, worker->chunkSize=%d, "
                            "worker->dimx=%d, worker->totalArrayCount=%d",
                            worker->CUDADeviceID, worker->dataType, worker->chunkSize, worker->dimx,
                            worker->totalArrayCount);
        processZ<T>(threadInfo, k, dataBuffer);
    }

    */
    for(uint64_t k = 0; k < chunkCount; k++)
    {
        LOGI << io::xprintf("Processing chunk %d/%d", k, chunkCount);
        threadpool->submit(processZ<T>, k, dataBuffer);
    }
    threadpool->waitAll();
}

template <typename T>
using WORKERPTR = std::shared_ptr<WORKER<T>>;

template <typename T>
using TP = io::ThreadPool<WORKER<T>>;

template <typename T>
using TPPTR = std::shared_ptr<TP<T>>;

template <typename T>
using TPINFO = typename TP<T>::ThreadInfo;

template <typename T>
using TPINFOPTR = std::shared_ptr<TPINFO<T>>;

template <typename T>
void processXYFrame(std::shared_ptr<typename TP<T>::ThreadInfo> threadInfo, uint32_t k_in, T* data)
{
    std::shared_ptr<WORKER<T>> worker = threadInfo->worker;
    if(!worker->isGPUInitialized)
    {
        EXECUDA(cudaSetDevice(worker->CUDADeviceID));
        worker->dimx = worker->ARG.dimx;
        worker->dimy = worker->ARG.dimy;
        if(worker->ARG.xypadding == NOPAD)
        {
            //Initialize worker
            worker->padding = false;
            worker->dimx_padded = worker->dimx;
            worker->dimy_padded = worker->dimy;
            worker->dimx_padded_hermitian = worker->dimx_padded / 2 + 1;
            worker->frameSize_padded = worker->dimx_padded * worker->dimy_padded;
            worker->frameSize_padded_complex = worker->dimy_padded * worker->dimx_padded_hermitian;
            //Device memory allocation for GPU_f and GPU_FTf, GPU_extendedf is not needed to be allocated
            EXECUDA(cudaMalloc((void**)&worker->GPU_f, worker->ARG.frameSize * sizeof(T)));
            EXECUDA(cudaMalloc((void**)&worker->GPU_FTf,
                               worker->frameSize_padded_complex * 2 * sizeof(T)));
        } else
        {
            //Initialize worker
            worker->padding = true;
            worker->dimx_padded = 2 * worker->dimx - 2; //Proper padding for DFT symmetry
            worker->dimy_padded = 2 * worker->dimy - 2; //Proper padding for DFT symmetry
            worker->dimx_padded_hermitian = worker->dimx_padded / 2 + 1;
            worker->frameSize_padded = worker->dimx_padded * worker->dimy_padded;
            worker->frameSize_padded_complex = worker->dimy_padded * worker->dimx_padded_hermitian;
            //Device memory allocation
            EXECUDA(cudaMalloc((void**)&worker->GPU_f, worker->ARG.frameSize * sizeof(T)));
            EXECUDA(
                cudaMalloc((void**)&worker->GPU_extendedf, worker->frameSize_padded * sizeof(T)));
            EXECUDA(cudaMalloc((void**)&worker->GPU_FTf,
                               worker->frameSize_padded_complex * 2 * sizeof(T)));
        }
        if(worker->dataType == io::DenSupportedType::FLOAT32)
        {
            //In cufftPlan2d first argument is slowest changing dimension then fastest changing dimension
            EXECUFFT(
                cufftPlan2d(&worker->FFT, worker->dimy_padded, worker->dimx_padded, CUFFT_R2C));
            EXECUFFT(
                cufftPlan2d(&worker->IFT, worker->dimy_padded, worker->dimx_padded, CUFFT_C2R));
        } else if(worker->dataType == io::DenSupportedType::FLOAT64)
        {
            //In cufftPlan2d first argument is slowest changing dimension then fastest changing dimension
            EXECUFFT(
                cufftPlan2d(&worker->FFT, worker->dimy_padded, worker->dimx_padded, CUFFT_D2Z));
            EXECUFFT(
                cufftPlan2d(&worker->IFT, worker->dimy_padded, worker->dimx_padded, CUFFT_Z2D));
        }
        worker->isGPUInitialized = true;
    } else
    {
        //This is needed to be done in case the worker is already initialized and the device ID is different
        EXECUDA(cudaSetDevice(worker->CUDADeviceID));
    }
    Args ARG = worker->ARG;
    //cufftHandle FFT = worker->FFT;
    //cufftHandle IFT = worker->IFT;
    //We read proper memory block into GPU memory
    T* framePointer = data + k_in * ARG.frameSize;
    //Copy data to host memory
    EXECUDA(cudaMemcpy((void*)worker->GPU_f, (void*)framePointer, ARG.frameSize * sizeof(T),
                       cudaMemcpyHostToDevice));
    dim3 threads(16, 16);
    //Do padding if needed or at least do pointer copy so that the GPU data are pointed by worker->GPU_extendedf and their size is worker->dimx_padded*worker->dimy_padded
    if(ARG.xypadding == NOPAD)
    {
        worker->GPU_extendedf = worker->GPU_f;
    } else
    {
        if(ARG.xypadding == SYMPAD)
        { //Neumann extension provides exactly symmetric padding
            CUDASymmPad2D<T>(threads, worker->GPU_f, worker->GPU_extendedf, worker->dimx,
                             worker->dimy, worker->dimx_padded, worker->dimy_padded);
        } else if(ARG.xypadding == ZEROPAD)
        {
            CUDAZeroPad2D<T>(threads, worker->GPU_f, worker->GPU_extendedf, worker->dimx,
                             worker->dimy, worker->dimx_padded, worker->dimy_padded);
        }
        EXECUDA(cudaPeekAtLastError());
        EXECUDA(cudaDeviceSynchronize());
    }
    //FFT of GPU_extendedf
    if(worker->dataType == io::DenSupportedType::FLOAT32)
    {
        EXECUFFT(cufftExecR2C(worker->FFT, (cufftReal*)worker->GPU_extendedf,
                              (cufftComplex*)worker->GPU_FTf));
    } else if(worker->dataType == io::DenSupportedType::FLOAT64)
    {
        EXECUFFT(cufftExecD2Z(worker->FFT, (cufftDoubleReal*)worker->GPU_extendedf,
                              (cufftDoubleComplex*)worker->GPU_FTf));
    }
    //Multiplication with the Gaussian kernel
    if(worker->dataType == io::DenSupportedType::FLOAT32)
    {
        CUDASpectralGaussianBlur2D<float, cufftComplex>(threads, worker->GPU_FTf,
                                                        worker->dimx_padded, worker->dimy_padded,
                                                        ARG.sigma_x, ARG.sigma_y);
    } else if(worker->dataType == io::DenSupportedType::FLOAT64)
    {
        CUDASpectralGaussianBlur2D<double, cufftDoubleComplex>(
            threads, worker->GPU_FTf, worker->dimx_padded, worker->dimy_padded, ARG.sigma_x,
            ARG.sigma_y);
    }
    //Perform IFFT
    if(worker->dataType == io::DenSupportedType::FLOAT32)
    {
        EXECUFFT(cufftExecC2R(worker->IFT, (cufftComplex*)worker->GPU_FTf,
                              (cufftReal*)worker->GPU_extendedf));
    } else if(worker->dataType == io::DenSupportedType::FLOAT64)
    {
        EXECUFFT(cufftExecZ2D(worker->IFT, (cufftDoubleComplex*)worker->GPU_FTf,
                              (cufftDoubleReal*)worker->GPU_extendedf));
    }
    //Remove padding if needed
    if(ARG.xypadding == NOPAD)
    {
        worker->GPU_f = worker->GPU_extendedf;
    } else
    {
        CUDARemovePadding<T>(threads, worker->GPU_extendedf, worker->GPU_f, worker->dimx,
                             worker->dimy, worker->dimx_padded);
    }
    //Copy data back to host memory
    EXECUDA(cudaMemcpy((void*)framePointer, (void*)worker->GPU_f, ARG.frameSize * sizeof(T),
                       cudaMemcpyDeviceToHost));
}

template <typename T>
void readFileToMemory(Args ARG, std::shared_ptr<io::DenFrame2DReader<T>>& inputReader, T* inputData)
{
    uint64_t frameCount = ARG.frames.size();
    uint64_t threadCount = std::max(1lu, std::min(static_cast<uint64_t>(ARG.threads), frameCount));
    uint64_t frames_per_thread = (frameCount + threadCount - 1) / threadCount;
    std::vector<std::thread> threads;
    for(uint32_t t = 0; t < threadCount; t++)
    {
        uint64_t start_frame = t * frames_per_thread;
        uint64_t end_frame
            = std::min(start_frame + frames_per_thread, static_cast<uint64_t>(frameCount));
        threads.push_back(std::thread([&ARG, &inputReader, &inputData, start_frame, end_frame]() {
            uint64_t k_in;
            std::shared_ptr<io::BufferedFrame2D<T>> f;
            T* f_array;
            for(uint64_t IND = start_frame; IND < end_frame; IND++)
            {
                k_in = ARG.frames[IND];
                f = inputReader->readBufferedFrame(k_in);
                f_array = f->getDataPointer();
                std::copy(f_array, f_array + ARG.frameSize, inputData + IND * ARG.frameSize);
            }
        }));
    }
    for(auto& t : threads)
    {
        t.join();
    }
}

// Function to write frames in parallel
template <typename T>
void writeFramesParallel(Args& ARG, io::DenSupportedType dataType, T* inputBuffer)
{
    uint64_t frameCount = ARG.frames.size();
    uint64_t frameSize = ARG.frameSize;
    uint64_t frameByteSize = frameSize * sizeof(T);
    uint64_t threadCount = std::max(1lu, std::min(static_cast<uint64_t>(ARG.threads), frameCount));
    uint64_t frames_per_thread = (frameCount + threadCount - 1) / threadCount;
    std::string outputFile = ARG.output_den;
    io::DenFileInfo::createEmpty3DDenFile(outputFile, dataType, ARG.dimx, ARG.dimy, frameCount);

    std::vector<std::thread> threads;
    for(uint64_t t = 0; t < threadCount; t++)
    {
        uint64_t start_frame = t * frames_per_thread;
        uint64_t end_frame
            = std::min(start_frame + frames_per_thread, static_cast<uint64_t>(frameCount));
        uint64_t bufferSize
            = std::min(static_cast<uint64_t>(10u), end_frame - start_frame) * frameByteSize;
        threads.emplace_back([&outputFile, &inputBuffer, frameSize, start_frame, end_frame,
                              bufferSize]() {
            std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>> outputWriter
                = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(outputFile, bufferSize);
            T* f_array;
            for(uint32_t k = start_frame; k < end_frame; k++)
            {
                f_array = inputBuffer + k * frameSize;
                outputWriter->writeBuffer(f_array, k);
            }
        });
    }
    for(auto& t : threads)
    {
        t.join();
    }
}

template <typename T>
/**
* @brief Partial transform of (x, y, z) array to (x, z, y) array
*
* @param array_in
* @param array_out
* @param dimx
* @param dimy
* @param dimz
* @param dimy_from
* @param dimy_to
*/
void swapArrayPartial(T* array_in,
                      T* array_out,
                      uint64_t dimx,
                      uint64_t dimy,
                      uint64_t dimz,
                      uint64_t dimy_from,
                      uint64_t dimy_to)
{
    if(dimy_from >= dimy_to)
    {
        return;
    }

    uint64_t frameSizeIn = dimy * dimx;
    uint64_t frameSizeOut = dimz * dimx;
    for(uint64_t k = 0; k < dimz; ++k)
    {
        T* startKIn = array_in + k * frameSizeIn;
        for(uint64_t j = dimy_from; j < dimy_to; ++j)
        {
            T* startIndexIn = startKIn + j * dimx;
            T* startIndexOut = array_out + j * frameSizeOut + k * dimx;
            std::copy(startIndexIn, startIndexIn + dimx, startIndexOut);
        }
    }
}

template <typename T>
void transposeArrayPartial(T* array_in,
                           T* array_out,
                           uint64_t dimx,
                           uint64_t dimy,
                           uint64_t dimz,
                           uint64_t dimz_from,
                           uint64_t dimz_to)
{
    if(dimz_from >= dimz_to)
    {
        return;
    }
    uint64_t frameSize = dimy * dimx;
    for(uint64_t k = dimz_from; k < dimz_to; ++k)
    {
        T* startKIn = array_in + k * frameSize;
        T* startKOut = array_out + k * frameSize;
        for(uint64_t j = 0; j < dimy; ++j)
        {
            for(uint64_t i = 0; i < dimx; ++i)
            {
                startKOut[i * dimy + j] = startKIn[j * dimx + i];
            }
        }
    }
}

template <typename T>
void processFiles(Args ARG, io::DenSupportedType dataType)
{
    //First I determine number of CUDA capable devices
    //I also test whether CUDA is available and if not I will raise exception
    int deviceCount;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    if(cudaStatus != cudaSuccess)
    {
        std::string errMsg = io::xprintf("CUDA error: %s", cudaGetErrorString(cudaStatus));
        KCTERR(errMsg);
    }
    if(deviceCount == 0)
    {
        std::string errMsg = "No CUDA capable devices found!";
        LOGI << errMsg;
        KCTERR(errMsg);
    }
    LOGI << io::xprintf("Number of CUDA capable devices: %d executing on %d threads", deviceCount,
                        ARG.threads);

    //I will use so called Gaussian Filter Separability property and peform first 2D convolution in x and y followed by 1D convolution in z
    //For doing so I will allocate block of memory for the whole stack of frames
    //So first read the whole stack into the memory
    uint64_t frameCount = ARG.frames.size();
    LOGI << io::xprintf("Allocating memory for %d frames.", frameCount);
    T* dataBuffer = new T[ARG.totalSize];
    std::shared_ptr<io::DenFrame2DReader<T>> inputReader
        = std::make_shared<io::DenFrame2DReader<T>>(ARG.input_den, ARG.threads);
    LOGI << io::xprintf("Reading file %s into memory.", ARG.input_den.c_str());
    readFileToMemory<T>(ARG, inputReader, dataBuffer);
    if(ARG.sigma_x > 0.0 || ARG.sigma_y > 0.0)
    {
        LOGI << io::xprintf("Adding gauss blur in xy frames with sigma_x=%f, sigma_y=%f.",
                            ARG.sigma_x, ARG.sigma_y);
        uint64_t threadCount
            = std::max(1lu, std::min(static_cast<uint64_t>(ARG.threads), frameCount));
        std::vector<WORKERPTR<T>> workers;
        std::shared_ptr<WORKER<T>> worker;
        for(uint64_t t = 0; t < threadCount; t++)
        {
            worker = std::make_shared<WORKER<T>>(ARG);
            worker->ARG = ARG;
            worker->dataType = dataType;
            worker->CUDADeviceID = t % deviceCount;
            worker->isGPUInitialized = false;
            workers.push_back(worker);
        }
        TPPTR<T> threadpool = std::make_shared<TP<T>>(threadCount, workers);

        for(uint64_t k = 0; k < frameCount; k++)
        {
            //            LOGI << io::xprintf("Processing %d/%d frame.", k, frameCount);
            threadpool->submit(processXYFrame<T>, k, dataBuffer);
        }
        threadpool->waitAll();
    }
    if(ARG.sigma_z > 0.0)
    {
        LOGI << io::xprintf("Adding gauss blur in z direction with sigma_z=%f.", ARG.sigma_z);
        LOGI << io::xprintf("Allocating temporary memory");
        T* swpBuffer = new T[ARG.totalSize];
        uint64_t dimy = ARG.dimy;
        uint64_t threadCount = std::max(1lu, std::min(static_cast<uint64_t>(ARG.threads), dimy));
        uint32_t rowsPerThread = (dimy + threadCount - 1) / threadCount;
        std::vector<std::future<void>> futures_swap;
        LOGI << io::xprintf("Swapping data with %d threads and %d rows per thread", threadCount,
                            rowsPerThread);
        uint64_t startRow = 0, endRow = 0;
        while(startRow < dimy)
        {
            endRow = std::min(startRow + rowsPerThread, static_cast<uint64_t>(dimy));
            futures_swap.emplace_back(std::async(std::launch::async, swapArrayPartial<T>,
                                                 dataBuffer, swpBuffer, ARG.dimx, ARG.dimy,
                                                 frameCount, startRow, endRow));
            startRow = endRow;
        }
        for(auto& f : futures_swap)
        {
            f.get();
        }
        LOGI << io::xprintf("Transposing data with %d threads", threadCount);
        std::vector<std::future<void>> futures_transpose;
        uint64_t startK = 0, endK = 0;
        while(startK < dimy)
        {
            endK = std::min(startK + rowsPerThread, static_cast<uint64_t>(dimy));
            futures_transpose.emplace_back(std::async(std::launch::async, transposeArrayPartial<T>,
                                                      swpBuffer, dataBuffer, ARG.dimx, frameCount,
                                                      ARG.dimy, startK, endK));
            startK = endK;
        }
        for(auto& f : futures_transpose)
        {
            f.get();
        }
        LOGI << io::xprintf("Adding Gauss blurr in z direction with sigma_z=%f.", ARG.sigma_z);
        //Now I treat dataBuffer as a ARG.dimx * ARG.dimy individual frameCount arrays to process
        uint64_t arrayCount = static_cast<uint64_t>(ARG.dimx) * static_cast<uint64_t>(ARG.dimy);
        processTransformed1DData<T>(ARG, dataType, dataBuffer, frameCount, arrayCount);
        LOGI << io::xprintf("Transposing data back with %d threads", threadCount);
        std::vector<std::future<void>> futures_transpose_back;
        startK = 0;
        endK = 0;
        while(startK < dimy)
        {
            endK = std::min(startK + rowsPerThread, static_cast<uint64_t>(dimy));
            futures_transpose_back.emplace_back(
                std::async(std::launch::async, transposeArrayPartial<T>, dataBuffer, swpBuffer,
                           frameCount, ARG.dimx, ARG.dimy, startK, endK));
            startK = endK;
        }
        for(auto& f : futures_transpose_back)
        {
            f.get();
        }
        LOGI << io::xprintf("Swapping data back with %d threads", threadCount);
        std::vector<std::future<void>> futures_swap_back;
        threadCount = std::max(1lu, std::min(static_cast<uint64_t>(ARG.threads), frameCount));
        uint32_t framesPerThread = (frameCount + threadCount - 1) / threadCount;
        uint64_t startFrame = 0, endFrame = 0;
        startFrame = 0;
        endFrame = 0;
        while(startFrame < frameCount)
        {
            endFrame = std::min(startFrame + framesPerThread, static_cast<uint64_t>(frameCount));
            futures_swap_back.emplace_back(std::async(std::launch::async, swapArrayPartial<T>,
                                                      swpBuffer, dataBuffer, ARG.dimx, frameCount,
                                                      ARG.dimy, startFrame, endFrame));
            startFrame = endFrame;
        }
        for(auto& f : futures_swap_back)
        {
            f.get();
        }
        LOGI << io::xprintf("Swap and transpose done, deleting temporary buffer.");
        delete[] swpBuffer;
    }
    LOGI << "Writing frames...";
    writeFramesParallel<T>(ARG, dataType, dataBuffer);
    LOGI << "Deleting buffers...";
    delete[] dataBuffer;
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    const std::string prgInfo = "Applies Gaussian blur to a 3D DEN file. The convolution is "
                                "performed in Fourier domain.";
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
    if(ARG.zpadding == NOPAD)
    {
        LOGI << "Padding in z dimension: NOPAD";
    } else if(ARG.zpadding == ZEROPAD)
    {
        LOGI << "Padding in z dimension: ZEROPAD";
    } else if(ARG.zpadding == SYMPAD)
    {
        LOGI << "Padding in z dimension: SYMPAD";
    }
    if(ARG.xypadding == NOPAD)
    {
        LOGI << "Padding in xy dimension: NOPAD";
    } else if(ARG.xypadding == ZEROPAD)
    {
        LOGI << "Padding in xy dimension: ZEROPAD";
    } else if(ARG.xypadding == SYMPAD)
    {
        LOGI << "Padding in xy dimension: SYMPAD";
    }
    LOGI << io::xprintf(
        "ARG.input_den=%s, ARG.output_den=%s, ARG.sigma_x=%f, ARG.sigma_y=%f, "
        "ARG.sigma_z=%f, "
        "ARG.zpadding=%d, ARG.xypadding=%d, ARG.dimx=%d, ARG.dimy=%d, ARG.dimz=%d, "
        "ARG.frameSize=%d, ARG.frames.size()=%d, ARG.pixelSizeX=%f, ARG.pixelSizeY=%f",
        ARG.input_den.c_str(), ARG.output_den.c_str(), ARG.sigma_x, ARG.sigma_y, ARG.sigma_z,
        ARG.zpadding, ARG.xypadding, ARG.dimx, ARG.dimy, ARG.dimz, ARG.frameSize, ARG.frames.size(),
        ARG.pixelSizeX, ARG.pixelSizeY);
    // After init parsing arguments
    io::DenFileInfo di(ARG.input_den);
    io::DenSupportedType dataType = di.getElementType();
    switch(dataType)
    {
    case io::DenSupportedType::FLOAT32: {
        processFiles<float>(ARG, dataType);
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
