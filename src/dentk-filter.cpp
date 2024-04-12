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
#include <regex>
#include <string>

//KCT specific
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DBufferedWritter.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "PROG/ArgumentsCTDetector.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/ArgumentsVerbose.hpp"
#include "PROG/Program.hpp"
#include "PROG/parseArgs.h"
#include "ftpl.h"
#include "tomographicFiltering.cuh"

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
    std::string input_den = "";
    std::string output_den = "";
    std::string frameSpecs = "";
    uint32_t dimx, dimy;
    uint32_t frameCount;
    uint64_t frameSize;
    bool outputFileExists = false;
    bool pad_none = false, pad_symm = false, pad_zero = false;
};

void Args::defineArguments()
{
    cliApp->add_option("input_den", input_den, "Input projection data.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("output_den", output_den, "Output filtered data.")->required();
    addForceArgs();
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
void processFrameNopad(int _FTPLID,
                       Args ARG,
                       io::DenSupportedType dataType,
                       cufftHandle FFT,
                       cufftHandle IFT,
                       uint32_t k_in,
                       uint32_t k_out,
                       std::shared_ptr<io::DenFrame2DReader<T>>& fReader,
                       std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>>& outputWritter)
{
    std::shared_ptr<io::BufferedFrame2D<T>> F = fReader->readBufferedFrame(k_in);
    io::BufferedFrame2D<T> x(T(0), ARG.dimx, ARG.dimy);
    T* F_array = F->getDataPointer();
    T* x_array = x.getDataPointer();
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
    // Do something here
    // Try without distinguishing types
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
            CUDARadonFilter(threads, blocksHermitan, GPU_FTf, ARG.dimx, ARG.dimy, ARG.pixelSizeX,
                            false);
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
void processFramePad(int _FTPLID,
                     Args ARG,
                     io::DenSupportedType dataType,
                     cufftHandle FFT,
                     cufftHandle IFT,
                     uint32_t NX,
                     uint32_t k_in,
                     uint32_t k_out,
                     std::shared_ptr<io::DenFrame2DReader<T>>& fReader,
                     std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>>& outputWritter)
{
    std::shared_ptr<io::BufferedFrame2D<T>> F = fReader->readBufferedFrame(k_in);
    io::BufferedFrame2D<T> x(T(0), ARG.dimx, ARG.dimy);
    T* F_array = F->getDataPointer();
    T* x_array = x.getDataPointer();
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
    void* GPU_f;
    void* GPU_f_padded;
    void* GPU_FTf;
    EXECUDA(cudaMalloc((void**)&GPU_f, ARG.frameSize * sizeof(T)));
    EXECUDA(cudaMalloc((void**)&GPU_f_padded, frameSizePad * sizeof(T)));
    EXECUDA(cudaMemcpy((void*)GPU_f, (void*)F_array, ARG.frameSize * sizeof(T),
                       cudaMemcpyHostToDevice));
    uint64_t complexBufferSize = ARG.dimy * xSizeHermitan;
    EXECUDA(cudaMalloc((void**)&GPU_FTf, complexBufferSize * 2 * sizeof(T)));

    if(dataType == io::DenSupportedType::FLOAT32)
    {
        if(ARG.pad_zero)
        {
            CUDAZeroPad(threads, blocks, GPU_f, GPU_f_padded, ARG.dimx, NX, ARG.dimy);
        } else
        {
            CUDASymmPad(threads, blocksNopad, GPU_f, GPU_f_padded, ARG.dimx, NX, ARG.dimy);
        }
        bool withoutFftShift = false;
        if(withoutFftShift)
        {
            EXECUFFT(cufftExecR2C(FFT, (cufftReal*)GPU_f_padded, (cufftComplex*)GPU_FTf));
            CUDARadonFilter(threads, blocksHermitan, GPU_FTf, NX, ARG.dimy, ARG.pixelSizeX, false);
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
                CUDAfftshift(threads, blocks, (cufftReal*)GPU_f_shifted, (cufftReal*)GPU_f_padded,
                             NX, ARG.dimy);
            } else
            {
                CUDAifftshift(threads, blocks, (cufftReal*)GPU_f_padded, (cufftReal*)GPU_f_shifted,
                              NX, ARG.dimy);
                //EXECUDA(cudaMemcpy((void*)GPU_f_shifted, (void*)GPU_f_padded,
                //                   NX * ARG.dimy * sizeof(T), cudaMemcpyDeviceToDevice));
                EXECUFFT(cufftExecR2C(FFT, (cufftReal*)GPU_f_shifted, (cufftComplex*)GPU_FTf));
                CUDARadonFilter(threads, blocksHermitan, GPU_FTf, NX, ARG.dimy, ARG.pixelSizeX,
                                false);
                EXECUFFT(cufftExecC2R(IFT, (cufftComplex*)GPU_FTf, (cufftReal*)GPU_f_shifted));
                CUDAfftshift(threads, blocks, (cufftReal*)GPU_f_shifted, (cufftReal*)GPU_f_padded,
                             NX, ARG.dimy);
                //EXECUDA(cudaMemcpy((void*)GPU_f_padded, (void*)GPU_f_shifted,
                //                   NX * ARG.dimy * sizeof(T), cudaMemcpyDeviceToDevice));
            }
        }
        CUDAStripPad(threads, blocksNopad, GPU_f_padded, GPU_f, ARG.dimx, NX, ARG.dimy);
        //Divide by number of projections for avoiding backprojection aditivity
        /* Not necessary as --backprojector-natural-scaling was introduced in kct-pb2d-backprojector 5b5bc55
        float frameCount = static_cast<float>(ARG.frameCount);
        float factor = PI / (frameCount * ARG.dimx);
        CUDAconstantMultiplication(threads, blocks, (cufftReal*)GPU_f, factor, ARG.dimx, ARG.dimy,
                                   ARG.dimx, ARG.dimy);
		*/

    } else if(dataType == io::DenSupportedType::FLOAT64)
    {
        KCTERR("Implemented just for FLOAT32!");
    }

    EXECUDA(cudaMemcpy((void*)x_array, (void*)GPU_f, ARG.frameSize * sizeof(T),
                       cudaMemcpyDeviceToHost));
    EXECUDA(cudaFree(GPU_f));
    EXECUDA(cudaFree(GPU_FTf));
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
void processFrame(int _FTPLID,
                  Args ARG,
                  io::DenSupportedType dataType,
                  cufftHandle FFT,
                  cufftHandle IFT,
                  uint32_t NX,
                  uint32_t k_in,
                  uint32_t k_out,
                  std::shared_ptr<io::DenFrame2DReader<T>>& fReader,
                  std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>>& outputWritter)
{
    if(ARG.pad_none)
    {
        processFrameNopad<T>(_FTPLID, ARG, dataType, FFT, IFT, k_in, k_out, fReader, outputWritter);
    } else
    {
        processFramePad<T>(_FTPLID, ARG, dataType, FFT, IFT, NX, k_in, k_out, fReader,
                           outputWritter);
    }
}

template <typename T>
void processFiles(Args ARG, io::DenSupportedType dataType)
{
    ftpl::thread_pool* threadpool = nullptr;
    if(ARG.threads > 0)
    {
        threadpool = new ftpl::thread_pool(ARG.threads);
    }
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
    const int dummy_FTPLID = 0;
    uint32_t k_in, k_out;
    cufftHandle FFT, IFT;
    // First is  slowest changing dimension, last is  slowest changing dimension
    uint32_t NX; //Padding size
    if(ARG.pad_none)
    {
        NX = ARG.dimx;
        EXECUFFT(cufftPlan1d(&FFT, ARG.dimx, CUFFT_R2C, ARG.dimy));
        EXECUFFT(cufftPlan1d(&IFT, ARG.dimx, CUFFT_C2R, ARG.dimy));
    } else if(ARG.pad_zero)
    {
        uint32_t padPower
            = static_cast<uint32_t>(std::ceil(std::log2(static_cast<float>(ARG.dimx)))) + 1;
        NX = 1 << padPower; // see https://stackoverflow.com/a/30357743
        LOGI << io::xprintf("ARG.dimx=%d padded NX=%d", ARG.dimx, NX);
        EXECUFFT(cufftPlan1d(&FFT, NX, CUFFT_R2C, ARG.dimy));
        EXECUFFT(cufftPlan1d(&IFT, NX, CUFFT_C2R, ARG.dimy));
    } else //Symmetric padding is a special case
    {
        if(ARG.dimx < 2)
        {
            std::string err
                = io::xprintf("For symmetric padding ARG.dimx > 1 but ARG.dimx=%d", ARG.dimx);
            KCTERR(err);
        }
        NX = 2 * ARG.dimx - 2;
        EXECUFFT(cufftPlan1d(&FFT, NX, CUFFT_R2C, ARG.dimy));
        EXECUFFT(cufftPlan1d(&IFT, NX, CUFFT_C2R, ARG.dimy));
    }
    LOGI << io::xprintf("Processing %d frames.", ARG.frames.size());
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
            threadpool->push(processFrame<T>, ARG, dataType, FFT, IFT, NX, k_in, k_out, fReader,
                             outputWritter);
        } else
        {
            processFrame<T>(dummy_FTPLID, ARG, dataType, FFT, IFT, NX, k_in, k_out, fReader,
                            outputWritter);
        }
    }
    if(threadpool != nullptr)
    {
        threadpool->stop(true);
        delete threadpool;
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
    PRG.endLog();
}
