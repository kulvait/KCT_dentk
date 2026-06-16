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
#include <iostream>
#include <mutex>
#include <regex>
#include <string>

//KCT specific
#include "BufferedFrame2D.hpp"
#include "BufferedFrame2DI.hpp"
#include "DEN/DenAsyncFrame2DBufferedWritter.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "PROG/ArgumentsCTDetector.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/ArgumentsThreadingCUDA.hpp"
#include "PROG/ArgumentsVerbose.hpp"
#include "PROG/Program.hpp"
#include "PROG/ThreadPool.hpp"
#include "PROG/parseArgs.h"
#include "padding.cuh"
#include "tomographicFiltering.cuh"

using namespace KCT;
using namespace KCT::util;

enum class BaseFilter { IdealFrequencyRamp, RamLakDiscrete };
enum class RampWindow { None, SheppLogan, Cosine, Hanning, Hamming, Kaiser };
// class declarations
class Args : public ArgumentsForce,
             public ArgumentsVerbose,
             public ArgumentsFramespec,
             public ArgumentsThreadingCUDA,
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
        , ArgumentsThreadingCUDA(argc, argv, prgName)
        , ArgumentsCTDetector(argc, argv, prgName){};
    std::string input_den = "";
    std::string output_den = "";
    std::string frameSpecs = "";
    uint32_t dimx, dimy;
    uint32_t frameCount;
    uint64_t frameSize;
    bool outputFileExists = false;
    bool pad_none = false, pad_symm = false, pad_zero = false;

    inline std::string to_string(BaseFilter f)
    {
        switch(f)
        {
        case BaseFilter::IdealFrequencyRamp:
            return "IdealFrequencyRamp";
        case BaseFilter::RamLakDiscrete:
            return "RamLakDiscrete";
        }
        return "Unknown";
    }

    inline std::string to_string(RampWindow w)
    {
        switch(w)
        {
        case RampWindow::None:
            return "None";
        case RampWindow::SheppLogan:
            return "SheppLogan";
        case RampWindow::Cosine:
            return "Cosine";
        case RampWindow::Hanning:
            return "Hanning";
        case RampWindow::Hamming:
            return "Hamming";
        case RampWindow::Kaiser:
            return "Kaiser";
        }
        return "Unknown";
    }

    BaseFilter baseFilter = BaseFilter::RamLakDiscrete;
    RampWindow rampWindow = RampWindow::None;
    double kaiserBeta = 0.0; // Only used if rampWindow == RampWindow::Kaiser
};

void Args::defineArguments()
{
    cliApp->add_option("input_den", input_den, "Input projection data.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("output_den", output_den, "Output filtered data.")->required();
    addForceArgs();
    //Ramp filter selection
    CLI::Option_group* og_base_filter = cliApp->add_option_group(
        "Base filter",
        io::xprintf("Ramp construction method, defaults to %s.", to_string(baseFilter).c_str()));
    og_base_filter->add_flag_callback(
        "--filter-ideal-ramp", [this]() { baseFilter = BaseFilter::IdealFrequencyRamp; },
        "Ideal frequency ramp |w|.");
    og_base_filter->add_flag_callback(
        "--filter-ram-lak", [this]() { baseFilter = BaseFilter::RamLakDiscrete; },
        "Ram-Lak discrete ramp.");
    og_base_filter->require_option(0, 1);
    //Ramp window selection
    CLI::Option_group* og_window = cliApp->add_option_group(
        "Ramp window",
        io::xprintf("Window to apply on the ramp filter, defaults to %s.",
                    to_string(rampWindow).c_str()));
    og_window->add_flag_callback(
        "--window-ramp", [this]() { rampWindow = RampWindow::None; }, "No window, pure ramp.");
    og_window->add_flag_callback(
        "--window-shepp-logan", [this]() { rampWindow = RampWindow::SheppLogan; },
        "Shepp-Logan window.");
    og_window->add_flag_callback(
        "--window-cosine", [this]() { rampWindow = RampWindow::Cosine; }, "Cosine window.");
    og_window->add_flag_callback(
        "--window-hanning", [this]() { rampWindow = RampWindow::Hanning; }, "Hanning window.");
    og_window->add_flag_callback(
        "--window-hamming", [this]() { rampWindow = RampWindow::Hamming; }, "Hamming window.");
    CLI::Option* opt_kaiser = og_window->add_flag_callback(
        "--window-kaiser", [this]() { rampWindow = RampWindow::Kaiser; }, "Kaiser window.");
    og_window->require_option(0, 1);
    cliApp
        ->add_option("--window-kaiser-beta", kaiserBeta,
                     "Beta parameter for Kaiser window, only used if --window-kaiser is selected.")
        ->check(CLI::PositiveNumber)
        ->needs(opt_kaiser);
    //Padding
    CLI::Option_group* op_clg = cliApp->add_option_group("Padding strategy", "Padding to use.");
    op_clg->add_flag("--pad-none", pad_none, "No padding.");
    op_clg->add_flag("--pad-symm", pad_symm, "Symmetric or reflection padding.");
    op_clg->add_flag("--pad-zero", pad_zero, "Padding with zeros.");
    op_clg->require_option(1);

    // Natural derivatives
    addPixelSizeArgs(1.0, 1.0);
    addFramespecArgs();
    addThreadingArgs();
}

int Args::postParse()
{
    // Test if minuend and subtraend are of the same type and dimensions
    io::DenFileInfo input_inf(input_den);
    int existFlag = handleFileExistence(output_den, force, input_den);
    if(existFlag == 1)
    {
        return 1;
    } else if(existFlag == -1)
    {
        outputFileExists = true;
    }
    dimx = input_inf.dimx();
    dimy = input_inf.dimy();
    frameCount = input_inf.getFrameCount();
    frameSize = static_cast<uint64_t>(dimx) * static_cast<uint64_t>(dimy);
    fillFramesVector(frameCount);
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
using READER = io::DenFrame2DReader<T>;

template <typename T>
using READERPTR = std::shared_ptr<READER<T>>;

template <typename T>
using WRITER = io::DenAsyncFrame2DBufferedWritter<T>;

template <typename T>
using WRITERPTR = std::shared_ptr<WRITER<T>>;

template <typename T>
class GPUWorker
{
public:
    int gpuID;
    Args ARG;
    io::DenSupportedType dataType;
    READERPTR<T> reader;
    WRITERPTR<T> writer;

    cufftHandle FFT = 0;
    cufftHandle IFT = 0;
    cufftHandle RamLakFFT = 0;
    uint32_t NX;
    uint32_t xSizeHermitian;
    uint32_t frameByteSize;
    uint32_t paddedFrameByteSize;
    uint32_t complexBufferByteSize;

    std::mutex gpuMutex;

    GPUWorker(int gpuID_,
              Args ARG_,
              io::DenSupportedType dataType_,
              READERPTR<T> reader_,
              WRITERPTR<T> writer_)
        : gpuID(gpuID_)
        , ARG(ARG_)
        , dataType(dataType_)
        , reader(reader_)
        , writer(writer_)
    {
        EXECUDA(cudaSetDevice(gpuID));
        if(ARG.pad_none)
        {
            NX = ARG.dimx;
        } else if(ARG.pad_zero)
        {
            uint32_t padPower
                = static_cast<uint32_t>(std::ceil(std::log2(static_cast<float>(ARG.dimx)))) + 1;
            NX = 1 << padPower; // see https://stackoverflow.com/a/30357743
            LOGI << io::xprintf("ARG.dimx=%d padded NX=%d", ARG.dimx, NX);
        } else //Symmetric padding is a special case
        {
            if(ARG.dimx < 2)
            {
                std::string err
                    = io::xprintf("For symmetric padding ARG.dimx > 1 but ARG.dimx=%d", ARG.dimx);
                KCTERR(err);
            }
            NX = 2 * ARG.dimx - 2;
        }
        if(dataType == io::DenSupportedType::FLOAT32)
        {
            EXECUFFT(cufftPlan1d(&FFT, NX, CUFFT_R2C, ARG.dimy));
            EXECUFFT(cufftPlan1d(&IFT, NX, CUFFT_C2R, ARG.dimy));
        } else if(dataType == io::DenSupportedType::FLOAT64)
        {
            EXECUFFT(cufftPlan1d(&FFT, NX, CUFFT_D2Z, ARG.dimy));
            EXECUFFT(cufftPlan1d(&IFT, NX, CUFFT_Z2D, ARG.dimy));
        } else
        {
            KCTERR(io::xprintf("Unsupported DenSupportedType %s for FFT plan creation.",
                               io::DenSupportedTypeToString(dataType).c_str()));
        }

        xSizeHermitian = NX / 2 + 1;

        frameByteSize = ARG.frameSize * sizeof(T);
        paddedFrameByteSize = static_cast<uint64_t>(NX) * ARG.dimy * sizeof(T);
        complexBufferByteSize = static_cast<uint64_t>(ARG.dimy) * xSizeHermitian * 2 * sizeof(T);

        EXECUDA(cudaMalloc(&GPU_f, frameByteSize));
        EXECUDA(cudaMalloc(&GPU_FTf, complexBufferByteSize));

        if(!ARG.pad_none)
        {
            EXECUDA(cudaMalloc(&GPU_f_padded, paddedFrameByteSize));
        }

        EXECUDA(cudaMalloc(&GPU_f_shifted, frameByteSize));
        EXECUDA(cudaMalloc(&GPU_filter, xSizeHermitian * sizeof(T)));
        if(ARG.baseFilter == BaseFilter::RamLakDiscrete)
        {
            EXECUDA(cudaMalloc(&GPU_RamLak, NX * sizeof(T)));
            EXECUDA(cudaMalloc(&GPU_RamLak_FFT, xSizeHermitian * 2 * sizeof(T)));
            if(dataType == io::DenSupportedType::FLOAT32)
            {
                CUDARamLakKernel1DFloat(GPU_RamLak, NX);
                EXECUFFT(cufftPlan1d(&RamLakFFT, NX, CUFFT_R2C, 1));
                EXECUFFT(
                    cufftExecR2C(RamLakFFT, (cufftReal*)GPU_RamLak, (cufftComplex*)GPU_RamLak_FFT));
                CUDAExtractRealFFTFloat(GPU_RamLak_FFT, GPU_filter, xSizeHermitian);
                EXECUDA(cudaPeekAtLastError());
            } else if(dataType == io::DenSupportedType::FLOAT64)
            {
                CUDARamLakKernel1DDouble(GPU_RamLak, NX);
                EXECUFFT(cufftPlan1d(&RamLakFFT, NX, CUFFT_D2Z, 1));
                EXECUFFT(cufftExecD2Z(RamLakFFT, (cufftDoubleReal*)GPU_RamLak,
                                      (cufftDoubleComplex*)GPU_RamLak_FFT));
                CUDAExtractRealFFTDouble(GPU_RamLak_FFT, GPU_filter, xSizeHermitian);
                EXECUDA(cudaPeekAtLastError());
            } else
            {
                KCTERR("Unsupported data type for filter creation.");
            }
        } else if(ARG.baseFilter == BaseFilter::IdealFrequencyRamp)
        {
            CUDAIdealRamp1D<T>(GPU_filter, NX);
        }

        switch(ARG.rampWindow)
        {
        case RampWindow::None:
            break;
        case RampWindow::SheppLogan:
            if(dataType == io::DenSupportedType::FLOAT32)
            {
                CUDAApplySheppLoganWindow1DFloat(GPU_filter, xSizeHermitian, NX);
            } else if(dataType == io::DenSupportedType::FLOAT64)
            {
                CUDAApplySheppLoganWindow1DDouble(GPU_filter, xSizeHermitian, NX);
            }
            break;
        case RampWindow::Cosine:
            if(dataType == io::DenSupportedType::FLOAT32)
            {
                CUDAApplyCosineWindow1DFloat(GPU_filter, xSizeHermitian, NX);
            } else if(dataType == io::DenSupportedType::FLOAT64)
            {
                CUDAApplyCosineWindow1DDouble(GPU_filter, xSizeHermitian, NX);
            }
            break;
        case RampWindow::Hanning:
            if(dataType == io::DenSupportedType::FLOAT32)
            {
                CUDAApplyHanningWindow1DFloat(GPU_filter, xSizeHermitian, NX);
            } else if(dataType == io::DenSupportedType::FLOAT64)
            {
                CUDAApplyHanningWindow1DDouble(GPU_filter, xSizeHermitian, NX);
            }
            break;
        case RampWindow::Hamming:
            if(dataType == io::DenSupportedType::FLOAT32)
            {
                CUDAApplyHammingWindow1DFloat(GPU_filter, xSizeHermitian, NX);
            } else if(dataType == io::DenSupportedType::FLOAT64)
            {
                CUDAApplyHammingWindow1DDouble(GPU_filter, xSizeHermitian, NX);
            }
            break;
        case RampWindow::Kaiser:
            if(dataType == io::DenSupportedType::FLOAT32)
            {
                CUDAApplyKaiserWindow1DFloat(GPU_filter, xSizeHermitian, NX, ARG.kaiserBeta);
            } else if(dataType == io::DenSupportedType::FLOAT64)
            {
                CUDAApplyKaiserWindow1DDouble(GPU_filter, xSizeHermitian, NX, ARG.kaiserBeta);
            }
            break;
        default:
            KCTERR("Unknown ramp window selected.");
        }
    }

    ~GPUWorker()
    {
        cudaSetDevice(gpuID);

        if(GPU_f != nullptr)
        {
            cudaFree(GPU_f);
        }

        if(GPU_f_padded != nullptr)
        {
            cudaFree(GPU_f_padded);
        }

        if(GPU_f_shifted != nullptr)
        {
            cudaFree(GPU_f_shifted);
        }

        if(GPU_FTf != nullptr)
        {
            cudaFree(GPU_FTf);
        }

        if(FFT != 0)
        {
            cufftDestroy(FFT);
        }

        if(IFT != 0)
        {
            cufftDestroy(IFT);
        }

        if(FFT != 0)
        {
            cufftDestroy(FFT);
        }

        if(IFT != 0)
        {
            cufftDestroy(IFT);
        }
    }

    void setDevice() { EXECUDA(cudaSetDevice(gpuID)); }

    cufftHandle fft() { return FFT; }

    cufftHandle ift() { return IFT; }

    WRITERPTR<T> getWriter() { return writer; }

    uint32_t getNX() const { return NX; }

    void processFrameNopad(const io::BufferedFrame2D<T>& frame_in,
                           io::BufferedFrame2D<T>& frame_out)
    {
        uint32_t THREADSIZE1 = 32;
        uint32_t THREADSIZE2 = 32;
        dim3 threads(THREADSIZE1, THREADSIZE2);
        {
            std::lock_guard<std::mutex> lock(gpuMutex);
            setDevice();

            EXECUDA(cudaMemcpy((void*)GPU_f, (void*)frame_in.data(), ARG.frameSize * sizeof(T),
                               cudaMemcpyHostToDevice));
            if(dataType == io::DenSupportedType::FLOAT32)
            {
                EXECUFFT(cufftExecR2C(FFT, (cufftReal*)GPU_f, (cufftComplex*)GPU_FTf));
                CUDASpectralFilter<float, cufftComplex>(threads, GPU_FTf, GPU_filter, NX, ARG.dimy,
                                                        ARG.pixelSizeX);
                EXECUFFT(cufftExecC2R(IFT, (cufftComplex*)GPU_FTf, (cufftReal*)GPU_f));
            } else if(dataType == io::DenSupportedType::FLOAT64)
            {
                EXECUFFT(cufftExecD2Z(FFT, (cufftDoubleReal*)GPU_f, (cufftDoubleComplex*)GPU_FTf));
                CUDASpectralFilter<double, cufftDoubleComplex>(threads, GPU_FTf, GPU_filter, NX,
                                                               ARG.dimy, ARG.pixelSizeX);
                EXECUFFT(cufftExecZ2D(IFT, (cufftDoubleComplex*)GPU_FTf, (cufftDoubleReal*)GPU_f));
            } else
            {
                KCTERR("Unsupported data type for FFT.");
            }
            EXECUDA(cudaPeekAtLastError());
            EXECUDA(cudaDeviceSynchronize());
            EXECUDA(cudaMemcpy((void*)frame_out.data(), (void*)GPU_f, frameByteSize,
                               cudaMemcpyDeviceToHost));
        }
    }

    void processFramePad(const io::BufferedFrame2D<T>& frame_in, io::BufferedFrame2D<T>& frame_out)
    {
        uint32_t THREADSIZE1 = 32;
        uint32_t THREADSIZE2 = 32;
        dim3 threads(THREADSIZE1, THREADSIZE2);
        //int xSizeHermitan = NX / 2 + 1;
        {
            std::lock_guard<std::mutex> lock(gpuMutex);
            setDevice();
            EXECUDA(cudaMemcpy((void*)GPU_f, (void*)frame_in.data(), ARG.frameSize * sizeof(T),
                               cudaMemcpyHostToDevice));
            if(ARG.pad_zero)
            {
                CUDAZeroPad<T>(threads, GPU_f, GPU_f_padded, ARG.dimx, NX, ARG.dimy);
            } else
            {
                CUDASymmPadZero<T>(threads, GPU_f, GPU_f_padded, ARG.dimx, NX, ARG.dimy);
            }
            if(dataType == io::DenSupportedType::FLOAT32)
            {
                EXECUFFT(cufftExecR2C(FFT, (cufftReal*)GPU_f_padded, (cufftComplex*)GPU_FTf));
                CUDASpectralFilter<float, cufftComplex>(threads, GPU_FTf, GPU_filter, NX, ARG.dimy,
                                                        ARG.pixelSizeX);
                EXECUFFT(cufftExecC2R(IFT, (cufftComplex*)GPU_FTf, (cufftReal*)GPU_f_padded));
            } else if(dataType == io::DenSupportedType::FLOAT64)
            {
                EXECUFFT(cufftExecD2Z(FFT, (cufftDoubleReal*)GPU_f_padded,
                                      (cufftDoubleComplex*)GPU_FTf));
                CUDASpectralFilter<double, cufftDoubleComplex>(threads, GPU_FTf, GPU_filter, NX,
                                                               ARG.dimy, ARG.pixelSizeX);
                EXECUFFT(cufftExecZ2D(IFT, (cufftDoubleComplex*)GPU_FTf,
                                      (cufftDoubleReal*)GPU_f_padded));
            } else
            {
                KCTERR("Unsupported data type for FFT.");
            }
            CUDARemovePadding<T>(threads, GPU_f_padded, GPU_f, ARG.dimx, ARG.dimy, NX);
            EXECUDA(cudaMemcpy((void*)frame_out.data(), (void*)GPU_f, frameByteSize,
                               cudaMemcpyDeviceToHost));
        }
    }

    void processFrame(uint32_t k_in, uint32_t k_out)
    {
        io::BufferedFrame2D<T> frame_in(ARG.dimx, ARG.dimy);
        reader->readFrameIntoBuffer(k_in, frame_in.data());
        io::BufferedFrame2D<T> frame_out(T(0), ARG.dimx, ARG.dimy);

        if(ARG.pad_none)
        {
            processFrameNopad(frame_in, frame_out);
        } else
        {
            processFramePad(frame_in, frame_out);
        }
        writer->writeBufferedFrame(frame_out, k_out);
    }

private:
    //GPU buffers
    void* GPU_RamLak = nullptr;
    void* GPU_RamLak_FFT = nullptr;
    void* GPU_filter = nullptr;
    void* GPU_f = nullptr;
    void* GPU_FTf = nullptr;
    //void* GPU_filter = nullptr;
    void* GPU_f_shifted = nullptr;
    void* GPU_f_padded = nullptr;
};

template <typename T>
using GPUWORKERPTR = std::shared_ptr<GPUWorker<T>>;

template <typename T>
using TP = io::ThreadPool<GPUWorker<T>>;

template <typename T>
using TPPTR = std::shared_ptr<TP<T>>;

template <typename T>
using TPINFO = typename TP<T>::ThreadInfo;

template <typename T>
using TPINFOPTR = std::shared_ptr<TPINFO<T>>;

//Legacy implementation to be removed
template <typename T>
void processFrameNopad(TPINFOPTR<T> threadInfo, uint32_t k_in, uint32_t k_out)
{
    GPUWORKERPTR<T> worker = threadInfo->worker;
    Args ARG = worker->ARG;
    READERPTR<T> fReader = worker->reader;
    WRITERPTR<T> outputWritter = worker->writer;
    io::DenSupportedType dataType = worker->dataType;
    cufftHandle FFT = worker->fft();
    cufftHandle IFT = worker->ift();

    std::shared_ptr<io::BufferedFrame2DI<T>> F = fReader->readBufferedFrame(k_in);
    io::BufferedFrame2D<T> x(T(0), ARG.dimx, ARG.dimy);
    T* F_array = F->data();
    T* x_array = x.data();
    //std::copy(F_array, F_array + ARG.frameSize, x_array);
    //outputWritter->writeBufferedFrame(x, k_out);
    //return;
    uint32_t THREADSIZE1 = 32;
    uint32_t THREADSIZE2 = 32;
    dim3 threads(THREADSIZE1, THREADSIZE2);
    int xSizeHermitan = ARG.dimx / 2 + 1;
    dim3 blocksHermitan((ARG.dimy + THREADSIZE1 - 1) / THREADSIZE1,
                        (xSizeHermitan + THREADSIZE2 - 1) / THREADSIZE2);
    dim3 blocks((ARG.dimy + THREADSIZE1 - 1) / THREADSIZE1,
                (ARG.dimx + THREADSIZE2 - 1) / THREADSIZE2);
    // Mutex protected GPU access for avoiding issues with multiple threads using the same GPU
    {
        std::lock_guard<std::mutex> lock(worker->gpuMutex);
        worker->setDevice();
        void* GPU_f;
        void* GPU_FTf;

        EXECUDA(cudaMalloc((void**)&GPU_f, ARG.frameSize * sizeof(T)));
        EXECUDA(cudaMemcpy((void*)GPU_f, (void*)F_array, ARG.frameSize * sizeof(T),
                           cudaMemcpyHostToDevice));
        uint64_t complexBufferSize = ARG.dimy * xSizeHermitan;
        EXECUDA(cudaMalloc((void**)&GPU_FTf, complexBufferSize * 2 * sizeof(T)));
        if(dataType == io::DenSupportedType::FLOAT32)
        {
            bool withoutFftShift = false;
            if(withoutFftShift)
            {
                EXECUFFT(cufftExecR2C(FFT, (cufftReal*)GPU_f, (cufftComplex*)GPU_FTf));
                CUDARadonFilter(threads, blocksHermitan, GPU_FTf, ARG.dimx, ARG.dimy,
                                ARG.pixelSizeX, false);
                EXECUFFT(cufftExecC2R(IFT, (cufftComplex*)GPU_FTf, (cufftReal*)GPU_f));
            } else
            {
                void* GPU_f_shifted;
                EXECUDA(cudaMalloc((void**)&GPU_f_shifted, ARG.frameSize * sizeof(T)));
                bool doSpectralIfftshift = false;
                if(doSpectralIfftshift)
                {
                    EXECUFFT(cufftExecR2C(FFT, (cufftReal*)GPU_f, (cufftComplex*)GPU_FTf));
                    CUDARadonFilter(threads, blocksHermitan, GPU_FTf, ARG.dimx, ARG.dimy,
                                    ARG.pixelSizeX, true);
                    EXECUDA(cudaPeekAtLastError());
                    EXECUDA(cudaDeviceSynchronize());

                    EXECUFFT(cufftExecC2R(IFT, (cufftComplex*)GPU_FTf, (cufftReal*)GPU_f_shifted));
                    CUDAfftshift(threads, blocks, (cufftReal*)GPU_f_shifted, (cufftReal*)GPU_f,
                                 ARG.dimx, ARG.dimy);
                } else
                {
                    CUDAifftshift(threads, blocks, (cufftReal*)GPU_f, (cufftReal*)GPU_f_shifted,
                                  ARG.dimx, ARG.dimy);
                    EXECUFFT(cufftExecR2C(FFT, (cufftReal*)GPU_f_shifted, (cufftComplex*)GPU_FTf));
                    CUDARadonFilter(threads, blocksHermitan, GPU_FTf, ARG.dimx, ARG.dimy,
                                    ARG.pixelSizeX, false);
                    EXECUFFT(cufftExecC2R(IFT, (cufftComplex*)GPU_FTf, (cufftReal*)GPU_f_shifted));
                    CUDAfftshift(threads, blocks, (cufftReal*)GPU_f_shifted, (cufftReal*)GPU_f,
                                 ARG.dimx, ARG.dimy);
                }
                EXECUDA(cudaFree(GPU_f_shifted));
            }
            //Divide by number of projections for avoiding backprojection aditivity
            /* Not necessary as --backprojector-natural-scaling was introduced in kct-pb2d-backprojector 5b5bc55
        float frameCount = static_cast<float>(ARG.frameCount);
        float factor = PI / (frameCount * ARG.dimx);
        CUDAconstantMultiplication(threads, blocks, (cufftReal*)GPU_f, factor, ARG.dimx, ARG.dimy,
                                   ARG.dimx, ARG.dimy);
		*/

        } else
        {
            KCTERR("Implemented just for FLOAT32!");
        }

        EXECUDA(cudaMemcpy((void*)x_array, (void*)GPU_f, ARG.frameSize * sizeof(T),
                           cudaMemcpyDeviceToHost));
        EXECUDA(cudaFree(GPU_f));
        EXECUDA(cudaFree(GPU_FTf));
    }
    outputWritter->writeBufferedFrame(x, k_out);
    if(ARG.verbose)
    {
        if(k_in == k_out)
        {
            LOGD << io::xprintf("Processed frame %d/%d.", k_in, outputWritter->getFrameCount());
        } else
        {
            LOGD << io::xprintf("Processed frame %d->%d/%d.", k_in, k_out,
                                outputWritter->getFrameCount());
        }
    }
}

//Legacy implementation to be removed
template <typename T>
void processFramePad(TPINFOPTR<T> threadInfo, uint32_t k_in, uint32_t k_out)
{
    GPUWORKERPTR<T> worker = threadInfo->worker;
    Args ARG = worker->ARG;
    READERPTR<T> fReader = worker->reader;
    WRITERPTR<T> outputWritter = worker->writer;
    io::DenSupportedType dataType = worker->dataType;
    cufftHandle FFT = worker->fft();
    cufftHandle IFT = worker->ift();
    uint32_t NX = worker->getNX();

    std::shared_ptr<io::BufferedFrame2DI<T>> F = fReader->readBufferedFrame(k_in);
    io::BufferedFrame2D<T> x(T(0), ARG.dimx, ARG.dimy);
    T* F_array = F->data();
    T* x_array = x.data();
    uint32_t frameSizePad = NX * ARG.dimy;
    uint32_t THREADSIZE1 = 32;
    uint32_t THREADSIZE2 = 32;
    dim3 threads(THREADSIZE1, THREADSIZE2);
    int xSizeHermitan = NX / 2 + 1;
    dim3 blocksHermitan((ARG.dimy + THREADSIZE1 - 1) / THREADSIZE1,
                        (NX + THREADSIZE2 - 1) / THREADSIZE2);
    dim3 blocks((ARG.dimy + THREADSIZE1 - 1) / THREADSIZE1, (NX + THREADSIZE2 - 1) / THREADSIZE2);
    dim3 blocksNopad((ARG.dimy + THREADSIZE1 - 1) / THREADSIZE1,
                     (ARG.dimx + THREADSIZE2 - 1) / THREADSIZE2);
    // Mutex protected GPU access for avoiding issues with multiple threads using the same GPU
    {
        std::lock_guard<std::mutex> lock(worker->gpuMutex);
        worker->setDevice();
        void* GPU_f;
        void* GPU_f_padded;
        void* GPU_FTf;
        EXECUDA(cudaMalloc((void**)&GPU_f, ARG.frameSize * sizeof(T)));
        EXECUDA(cudaMalloc((void**)&GPU_f_padded, frameSizePad * sizeof(T)));
        EXECUDA(cudaMemcpy((void*)GPU_f, (void*)F_array, ARG.frameSize * sizeof(T),
                           cudaMemcpyHostToDevice));
        uint64_t complexBufferSize = ARG.dimy * xSizeHermitan;
        EXECUDA(cudaMalloc((void**)&GPU_FTf, complexBufferSize * 2 * sizeof(T)));

        if(ARG.pad_zero)
        {
            CUDAZeroPad<T>(threads, GPU_f, GPU_f_padded, ARG.dimx, NX, ARG.dimy);
        } else
        {
            CUDASymmPadZero<T>(threads, GPU_f, GPU_f_padded, ARG.dimx, NX, ARG.dimy);
        }
        if(dataType == io::DenSupportedType::FLOAT32)
        {
            bool withoutFftShift = false;
            if(withoutFftShift)
            {
                EXECUFFT(cufftExecR2C(FFT, (cufftReal*)GPU_f_padded, (cufftComplex*)GPU_FTf));
                CUDARadonFilter(threads, blocksHermitan, GPU_FTf, NX, ARG.dimy, ARG.pixelSizeX,
                                false);
                EXECUFFT(cufftExecC2R(IFT, (cufftComplex*)GPU_FTf, (cufftReal*)GPU_f_padded));

            } else
            {
                void* GPU_f_shifted;
                EXECUDA(cudaMalloc((void**)&GPU_f_shifted, frameSizePad * sizeof(T)));
                bool doSpectralIfftshift = false;
                if(doSpectralIfftshift)
                {
                    EXECUFFT(cufftExecR2C(FFT, (cufftReal*)GPU_f_padded, (cufftComplex*)GPU_FTf));
                    CUDARadonFilter(threads, blocksHermitan, GPU_FTf, NX, ARG.dimy, ARG.pixelSizeX,
                                    true);
                    EXECUDA(cudaPeekAtLastError());
                    EXECUDA(cudaDeviceSynchronize());

                    EXECUFFT(cufftExecC2R(IFT, (cufftComplex*)GPU_FTf, (cufftReal*)GPU_f_shifted));
                    CUDAfftshift(threads, blocks, (cufftReal*)GPU_f_shifted,
                                 (cufftReal*)GPU_f_padded, NX, ARG.dimy);
                } else
                {
                    CUDAifftshift(threads, blocks, (cufftReal*)GPU_f_padded,
                                  (cufftReal*)GPU_f_shifted, NX, ARG.dimy);
                    //EXECUDA(cudaMemcpy((void*)GPU_f_shifted, (void*)GPU_f_padded,
                    //                   NX * ARG.dimy * sizeof(T), cudaMemcpyDeviceToDevice));
                    EXECUFFT(cufftExecR2C(FFT, (cufftReal*)GPU_f_shifted, (cufftComplex*)GPU_FTf));
                    CUDARadonFilter(threads, blocksHermitan, GPU_FTf, NX, ARG.dimy, ARG.pixelSizeX,
                                    false);
                    EXECUFFT(cufftExecC2R(IFT, (cufftComplex*)GPU_FTf, (cufftReal*)GPU_f_shifted));
                    CUDAfftshift(threads, blocks, (cufftReal*)GPU_f_shifted,
                                 (cufftReal*)GPU_f_padded, NX, ARG.dimy);
                    //EXECUDA(cudaMemcpy((void*)GPU_f_padded, (void*)GPU_f_shifted,
                    //                   NX * ARG.dimy * sizeof(T), cudaMemcpyDeviceToDevice));
                }
                EXECUDA(cudaFree(GPU_f_shifted));
            }

        } else if(dataType == io::DenSupportedType::FLOAT64)
        {
            KCTERR("Implemented just for FLOAT32!");
        }
        CUDARemovePadding<T>(threads, GPU_f_padded, GPU_f, ARG.dimx, ARG.dimy, NX);
        //Divide by number of projections for avoiding backprojection aditivity
        /* Not necessary as --backprojector-natural-scaling was introduced in kct-pb2d-backprojector 5b5bc55
        float frameCount = static_cast<float>(ARG.frameCount);
        float factor = PI / (frameCount * ARG.dimx);
        CUDAconstantMultiplication(threads, blocks, (cufftReal*)GPU_f, factor, ARG.dimx, ARG.dimy,
                                   ARG.dimx, ARG.dimy);
		*/
        //Test if there are some errors, wrong GPU ...
        EXECUDA(cudaPeekAtLastError());
        EXECUDA(cudaDeviceSynchronize());

        EXECUDA(cudaMemcpy((void*)x_array, (void*)GPU_f, ARG.frameSize * sizeof(T),
                           cudaMemcpyDeviceToHost));
        EXECUDA(cudaFree(GPU_f));
        EXECUDA(cudaFree(GPU_FTf));
        EXECUDA(cudaFree(GPU_f_padded));
    }
    outputWritter->writeBufferedFrame(x, k_out);
    if(ARG.verbose)
    {
        if(k_in == k_out)
        {
            LOGD << io::xprintf("Processed frame %d/%d.", k_in, outputWritter->getFrameCount());
        } else
        {
            LOGD << io::xprintf("Processed frame %d->%d/%d.", k_in, k_out,
                                outputWritter->getFrameCount());
        }
    }
}

template <typename T>
void processFrame(TPINFOPTR<T> threadInfo, uint32_t k_in, uint32_t k_out)
{
    GPUWORKERPTR<T> worker = threadInfo->worker;
    worker->processFrame(k_in, k_out);
    /*
    Args ARG = worker->ARG;
    if(ARG.pad_none)
    {
        processFrameNopad<T>(threadInfo, k_in, k_out);
    } else
    {
        processFramePad<T>(threadInfo, k_in, k_out);
    }*/
}

template <typename T>
void processFiles(Args ARG, io::DenSupportedType dataType)
{
    std::shared_ptr<io::DenFrame2DReader<T>> fReader
        = std::make_shared<io::DenFrame2DReader<T>>(ARG.input_den, ARG.threads);
    std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>> outputWritter;
    if(ARG.outputFileExists)
    {
        outputWritter = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(
            ARG.output_den, ARG.dimx, ARG.dimy, ARG.frameCount);
    } else
    {
        outputWritter = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(
            ARG.output_den, ARG.dimx, ARG.dimy, ARG.frames.size());
    }
    //Figure out how many GPUs are available
    int gpuCount = 0;
    EXECUDA(cudaGetDeviceCount(&gpuCount));
    if(gpuCount <= 0)
    {
        KCTERR("No CUDA device available.");
    }
    TPPTR<T> threadpool = nullptr;
    std::shared_ptr<typename TP<T>::ThreadInfo> thread_info = nullptr;
    //Regardelss if we use threadpool or not, we create workers vector so that without threading we can use workers[0] for processing frames.
    //This is done for better code structure and to avoid ifs in the processing function.
    std::vector<GPUWORKERPTR<T>> workers;
    uint32_t threadCount = ARG.threads > 0 ? ARG.threads : 1;
    uint32_t workerCount = std::min(threadCount, static_cast<uint32_t>(gpuCount));
    uint32_t divisionBoundaries = (threadCount + workerCount - 1) / workerCount;
    GPUWORKERPTR<T> wp = nullptr;
    for(uint32_t i = 0; i < threadCount; i++)
    {
        if(i % divisionBoundaries == 0 || wp == nullptr)
        {
            wp = std::make_shared<GPUWorker<T>>(i % gpuCount, ARG, dataType, fReader,
                                                outputWritter);
        }
        workers.push_back(wp);
    }

    if(ARG.threads > 0)
    {
        threadpool = std::make_shared<TP<T>>(ARG.threads, workers);
    } else
    {
        wp = workers[0];
        thread_info = std::make_shared<typename TP<T>::ThreadInfo>(TPINFO<T>{ 0, 0, wp });
    }

    uint32_t k_in, k_out;
    LOGI << io::xprintf("Processing %d frames using %d thread(s) and %d GPU(s).", ARG.frames.size(),
                        ARG.threads, gpuCount);
    for(uint32_t IND = 0; IND != ARG.frames.size(); IND++)
    {
        k_in = ARG.frames[IND];
        if(ARG.outputFileExists)
        {
            k_out = k_in; // To be able to do dentk-calc --force --multiply -f 0,end zero.den
            // BETA.den BETA.den
        } else
        {
            k_out = IND;
        }
        if(threadpool)
        {
            threadpool->submit(processFrame<T>, k_in, k_out);
        } else
        {
            processFrame<T>(thread_info, k_in, k_out);
        }
    }
    if(threadpool != nullptr)
    {
        threadpool->waitAll();
        threadpool = nullptr;
    }
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    const std::string prgInfo = "Filter projection data to be backprojected using FBP.";
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
    io::DenFileInfo di(ARG.input_den);
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
    PRG.endLog(true);
}
