// Logging
#include "PLOG/PlogSetup.h"

//#define PI 3.14159265358979323846
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
#include "spectralMethod.cuh"
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
#include "PROG/ArgumentsThreadingCUDA.hpp"
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
    std::string input_f = "";
    std::string output_x = "";
    std::string frameSpecs = "";
    uint32_t dimx, dimy, dimz;
    uint64_t frameSize;
    bool periodicBCs = false;
    bool neumannBCs = false;
    bool dirichletBCs = false;
    bool outputFileExists = false;
};

void Args::defineArguments()
{
    cliApp->add_option("input_f", input_f, "Component f in the equation \\Delta x = f.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("output_x", output_x, "Component x in the equation \\Delta x = f.")
        ->required();
    // Adding radio group see https://github.com/CLIUtils/CLI11/pull/234
    CLI::Option_group* op_clg
        = cliApp->add_option_group("Boundary conditions", "Boundary conditions to use.");
    op_clg->add_flag("--bc-neumann", neumannBCs, "Neumann boundary conditions.");
    op_clg->add_flag("--bc-dirichlet", dirichletBCs, "Dirichlet boundary conditions.");
    op_clg->add_flag("--bc-periodic", periodicBCs, "Periodic boundary conditions.");
    op_clg->require_option(1);
    addForceArgs();
    // Natural derivatives
    addPixelSizeArgs(1.0, 1.0);
    addFramespecArgs();
    addThreadingArgs();
}

int Args::postParse()
{
    // Test if minuend and subtraend are of the same type and dimensions
    io::DenFileInfo input_f_inf(input_f);
    int existFlag = handleFileExistence(output_x, force, input_f_inf);
    if(existFlag == 1)
    {
        return 1;
    } else if(existFlag == -1)
    {
        outputFileExists = true;
    }
    dimx = input_f_inf.dimx();
    dimy = input_f_inf.dimy();
    dimz = input_f_inf.dimz();
    frameSize = (uint64_t)dimx * (uint64_t)dimy;
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
void processFramePeriodic(int _FTPLID,
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
    uint32_t THREADSIZE1 = 32;
    uint32_t THREADSIZE2 = 32;
    dim3 threads(THREADSIZE1, THREADSIZE2);
    int xSizeHermitan = ARG.dimx / 2 + 1;
    dim3 blocks((ARG.dimy + THREADSIZE1 - 1) / THREADSIZE1,
                (xSizeHermitan + THREADSIZE2 - 1) / THREADSIZE2);
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
        /*
                            cufftReal *GPU_f;
                    EXECUDA(cudaMalloc((void **) &GPU_f, ARG.frameSize * sizeof(T)));
                            cufftComplex *GPU_FTf;*/
        EXECUFFT(cufftExecR2C(FFT, (cufftReal*)GPU_f, (cufftComplex*)GPU_FTf));
        // Now divide by (k_x^2+k_y^2)
        CUDAspectralMultiplication(threads, blocks, GPU_FTf, ARG.dimx, ARG.dimy, ARG.pixelSizeX,
                             ARG.pixelSizeY);
        EXECUDA(cudaPeekAtLastError());
        EXECUDA(cudaDeviceSynchronize());

        EXECUFFT(cufftExecC2R(IFT, (cufftComplex*)GPU_FTf, (cufftReal*)GPU_f));

    } else if(dataType == io::DenSupportedType::FLOAT64)
    {
        /*
                            cufftDoubleReal *GPU_f;
                            cufftDoubleComplex *GPU_FTf;*/
        EXECUFFT(cufftExecD2Z(FFT, (cufftDoubleReal*)GPU_f, (cufftDoubleComplex*)GPU_FTf));
        // Now divide by (k_x^2+k_y^2)
        EXECUFFT(cufftExecZ2D(IFT, (cufftDoubleComplex*)GPU_FTf, (cufftDoubleReal*)GPU_f));
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
            LOGD << io::xprintf("Processed frame %d->%d/%d.", k_in, k_out, outputWritter->getFrameCount());
        }
    }
}

template <typename T>
void processFrameNonperiodic(int _FTPLID,
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
    uint32_t THREADSIZE1 = 32;
    uint32_t THREADSIZE2 = 32;
    dim3 threads(THREADSIZE1, THREADSIZE2);
    // Do something here
    // Try without distinguishing types
    void* GPU_f;
    void* GPU_extendedf;
    void* GPU_FTf;
    EXECUDA(cudaMalloc((void**)&GPU_f, ARG.frameSize * sizeof(T)));
    EXECUDA(cudaMemcpy((void*)GPU_f, (void*)F_array, ARG.frameSize * sizeof(T),
                       cudaMemcpyHostToDevice));

    EXECUDA(cudaMalloc((void**)&GPU_extendedf, ARG.frameSize * 4 * sizeof(T)));
    if(ARG.dirichletBCs)
    {
        dim3 blocks((ARG.dimy + THREADSIZE1 - 1) / THREADSIZE1,
                    (ARG.dimx + THREADSIZE2 - 1) / THREADSIZE2);
        CUDADirichletExtension(threads, blocks, GPU_f, GPU_extendedf, ARG.dimx, ARG.dimy);
        EXECUDA(cudaPeekAtLastError());
        EXECUDA(cudaDeviceSynchronize());
    } else if(ARG.neumannBCs)
    {
        dim3 blocks((ARG.dimy + THREADSIZE1 - 1) / THREADSIZE1,
                    (ARG.dimx + THREADSIZE2 - 1) / THREADSIZE2);
        CUDANeumannExtension(threads, blocks, GPU_f, GPU_extendedf, ARG.dimx, ARG.dimy);
        EXECUDA(cudaPeekAtLastError());
        EXECUDA(cudaDeviceSynchronize());
    }
    int xSizeHermitan = 2 * ARG.dimx / 2 + 1;
    uint64_t complexBufferSize = 2 * ARG.dimy * xSizeHermitan;
    EXECUDA(cudaMalloc((void**)&GPU_FTf, complexBufferSize * 2 * sizeof(T)));

    if(dataType == io::DenSupportedType::FLOAT32)
    {
        EXECUFFT(cufftExecR2C(FFT, (cufftReal*)GPU_extendedf, (cufftComplex*)GPU_FTf));
        // Now divide by (k_x^2+k_y^2)
        int xSizeHermitan = 2 * ARG.dimx / 2 + 1;
        dim3 blocks((2 * ARG.dimy + THREADSIZE1 - 1) / THREADSIZE1,
                    (xSizeHermitan + THREADSIZE2 - 1) / THREADSIZE2);
        CUDAspectralMultiplication(threads, blocks, GPU_FTf, 2 * ARG.dimx, 2 * ARG.dimy, ARG.pixelSizeX,
                             ARG.pixelSizeY);
        EXECUDA(cudaPeekAtLastError());
        EXECUDA(cudaDeviceSynchronize());

        EXECUFFT(cufftExecC2R(IFT, (cufftComplex*)GPU_FTf, (cufftReal*)GPU_extendedf));

    } else if(dataType == io::DenSupportedType::FLOAT64)
    {
        EXECUFFT(cufftExecD2Z(FFT, (cufftDoubleReal*)GPU_f, (cufftDoubleComplex*)GPU_FTf));
        // Now divide by (k_x^2+k_y^2)
        EXECUFFT(cufftExecZ2D(IFT, (cufftDoubleComplex*)GPU_FTf, (cufftDoubleReal*)GPU_f));
    }
    dim3 blocks((ARG.dimy + THREADSIZE1 - 1) / THREADSIZE1,
                (ARG.dimx + THREADSIZE2 - 1) / THREADSIZE2);
    CUDAFunctionRestriction(threads, blocks, GPU_extendedf, GPU_f, ARG.dimx, ARG.dimy);
    EXECUDA(cudaPeekAtLastError());
    EXECUDA(cudaDeviceSynchronize());
    EXECUDA(cudaMemcpy((void*)x_array, (void*)(GPU_f), ARG.frameSize * sizeof(T),
                       cudaMemcpyDeviceToHost));
    EXECUDA(cudaFree(GPU_f));
    EXECUDA(cudaFree(GPU_extendedf));
    EXECUDA(cudaFree(GPU_FTf));
    outputWritter->writeBufferedFrame(x, k_out);
    if(ARG.verbose)
    {
        if(k_in == k_out)
        {
            LOGD << io::xprintf("Processed frame %d/%d.", k_in, outputWritter->getFrameCount());
        } else
        {
            LOGD << io::xprintf("Processed frame %d->%d/%d.", k_in, k_out, outputWritter->getFrameCount());
        }
    }
}

template <typename T>
void processFrame(int _FTPLID,
                  Args ARG,
                  io::DenSupportedType dataType,
                  cufftHandle FFT,
                  cufftHandle IFT,
                  uint32_t k_in,
                  uint32_t k_out,
                  std::shared_ptr<io::DenFrame2DReader<T>>& fReader,
                  std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>>& outputWritter)
{
    if(ARG.periodicBCs)
    {
        processFramePeriodic<T>(_FTPLID, ARG, dataType, FFT, IFT, k_in, k_out, fReader,
                                outputWritter);
    } else
    {
        processFrameNonperiodic<T>(_FTPLID, ARG, dataType, FFT, IFT, k_in, k_out, fReader,
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
        = std::make_shared<io::DenFrame2DReader<T>>(ARG.input_f, ARG.threads);
    std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>> outputWritter;
    if(ARG.outputFileExists)
    {
        outputWritter = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(
            ARG.output_x, ARG.dimx, ARG.dimy, ARG.dimz);
    } else
    {
        outputWritter = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(
            ARG.output_x, ARG.dimx, ARG.dimy, ARG.frames.size());
    }
    const int dummy_FTPLID = 0;
    uint32_t k_in, k_out;
    cufftHandle FFT, IFT;
    // First is  slowest changing dimension, last is  slowest changing dimension
    if(ARG.periodicBCs)
    {
        EXECUFFT(cufftPlan2d(&FFT, ARG.dimy, ARG.dimx, CUFFT_R2C));
        EXECUFFT(cufftPlan2d(&IFT, ARG.dimy, ARG.dimx, CUFFT_C2R));
    } else
    {
        EXECUFFT(cufftPlan2d(&FFT, 2 * ARG.dimy, 2 * ARG.dimx, CUFFT_R2C));
        EXECUFFT(cufftPlan2d(&IFT, 2 * ARG.dimy, 2 * ARG.dimx, CUFFT_C2R));
    }
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
        LOGI << io::xprintf("k_in=%d k_out=%d", k_in, k_out);
        if(threadpool)
        {
            threadpool->push(processFrame<T>, ARG, dataType, FFT, IFT, k_in, k_out, fReader,
                             outputWritter);
        } else
        {
            processFrame<T>(dummy_FTPLID, ARG, dataType, FFT, IFT, k_in, k_out, fReader,
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
    const std::string prgInfo
        = "Compute 2D Laplace's operator \\Delta f using spectral method frame-wise with CUDA.";
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
    io::DenFileInfo di(ARG.input_f);
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
