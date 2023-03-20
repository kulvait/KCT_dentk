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

    double propagationDistance = 1;
    double waveEnergyKeV = 20;
    double lambda;
    bool outputFilesExist;

    bool paddingNone = false;
    bool paddingZero = false;
    bool propagatorFresnel = false;
    bool propagatorRayleigh = false;
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
    op_clg->require_option(1);
    op_clg = cliApp->add_option_group("Propagator operator", "Padding to use for convolution.");
    op_clg->add_flag("--fresnel", propagatorFresnel, "Use Fresnel propagator.");
    op_clg->add_flag("--rayleigh", propagatorRayleigh, "Use Rayleigh-Sommerfeld propagator.");
    op_clg->require_option(1);

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
    if(input_phase_inf.dimx() != dimx || input_phase_inf.dimy() != dimy
       || input_phase_inf.dimz() != dimz
       || input_phase_inf.getElementType() != input_intensity_inf.getElementType())
    {
        ERR = io::xprintf("Dimensions of intensity %s and phase %s are incompatible!",
                          input_intensity.c_str(), input_phase.c_str());
    }
    // I will simply remove files if they exist
    int flagPhase, flagIntensity;
    flagIntensity = handleFileExistence(output_intensity, force, input_intensity_inf);
    flagPhase = handleFileExistence(output_phase, force, input_phase_inf);
    if(flagPhase == -1 && flagIntensity == -1)
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
void processFrameFloat(int _FTPLID,
                       Args ARG,
                       io::DenSupportedType dataType,
                       cufftHandle FFT,
                       cufftHandle IFT,
                       uint32_t k_in,
                       uint32_t k_out,
                       std::shared_ptr<io::DenFrame2DReader<T>>& intensityReader,
                       std::shared_ptr<io::DenFrame2DReader<T>>& phaseReader,
                       std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>>& intensityWritter,
                       std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>>& phaseWritter)
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

    std::shared_ptr<io::BufferedFrame2D<T>> I_f = intensityReader->readBufferedFrame(k_in);
    std::shared_ptr<io::BufferedFrame2D<T>> P_f = phaseReader->readBufferedFrame(k_in);
    // We transform it to E_0 and decompose back to I and P in GPU memmory

    io::BufferedFrame2D<T> x(T(0), ARG.dimx, ARG.dimy);
    T* I_array = I_f->getDataPointer();
    T* P_array = P_f->getDataPointer();

    uint32_t THREADSIZE1 = 32;
    uint32_t THREADSIZE2 = 32;
    dim3 threads(THREADSIZE1, THREADSIZE2);
    // Do something here
    // Try without distinguishing types
    void* GPU_intensity;
    void* GPU_phase;
    void* GPU_envelope;
    void* GPU_FTenvelope;
    EXECUDA(cudaMalloc((void**)&GPU_intensity, ARG.frameSize * sizeof(T)));
    EXECUDA(cudaMemcpy((void*)GPU_intensity, (void*)I_array, ARG.frameSize * sizeof(T),
                       cudaMemcpyHostToDevice));
    EXECUDA(cudaMalloc((void**)&GPU_phase, ARG.frameSize * sizeof(T)));
    EXECUDA(cudaMemcpy((void*)GPU_phase, (void*)P_array, ARG.frameSize * sizeof(T),
                       cudaMemcpyHostToDevice));

    // Different to the padding code
    EXECUDA(cudaMalloc((void**)&GPU_envelope, frameSizePadded * 2 * sizeof(T)));
    EXECUDA(cudaMalloc((void**)&GPU_FTenvelope, frameSizePadded * 2 * sizeof(T)));
    dim3 blocks((dimy_padded + THREADSIZE1 - 1) / THREADSIZE1,
                (dimx_padded + THREADSIZE2 - 1) / THREADSIZE2);
    CUDAenvelopeConstruction(threads, blocks, GPU_intensity, GPU_phase, GPU_envelope, dimx, dimy,
                             dimx_padded, dimy_padded);
    EXECUDA(cudaPeekAtLastError());
    EXECUDA(cudaDeviceSynchronize());
    EXECUFFT(cufftExecC2C(FFT, (cufftComplex*)GPU_envelope, (cufftComplex*)GPU_FTenvelope,
                          CUFFT_FORWARD));
    if(ARG.propagatorFresnel)
    {
        CUDAspectralMultiplicationFresnel(threads, blocks, GPU_FTenvelope, ARG.lambda,
                                          ARG.propagationDistance, dimx_padded, dimy_padded,
                                          ARG.pixelSizeX, ARG.pixelSizeY);
    } else if(ARG.propagatorRayleigh)
    {
        CUDAspectralMultiplicationRayleigh(threads, blocks, GPU_FTenvelope, ARG.lambda,
                                           ARG.propagationDistance, dimx_padded, dimy_padded,
                                           ARG.pixelSizeX, ARG.pixelSizeY);
    } else
    {
        KCTERR("No propagator specified");
    }
    EXECUDA(cudaPeekAtLastError());
    EXECUDA(cudaDeviceSynchronize());
    EXECUFFT(cufftExecC2C(IFT, (cufftComplex*)GPU_FTenvelope, (cufftComplex*)GPU_envelope,
                          CUFFT_INVERSE));

    dim3 blocksOrig((dimy + THREADSIZE1 - 1) / THREADSIZE1, (dimx + THREADSIZE2 - 1) / THREADSIZE2);

    CUDAenvelopeDecomposition(threads, blocksOrig, GPU_intensity, GPU_phase, GPU_envelope, dimx,
                              dimy, dimx_padded, dimy_padded);
    EXECUDA(cudaMemcpy((void*)I_array, (void*)GPU_intensity, ARG.frameSize * sizeof(T),
                       cudaMemcpyDeviceToHost));
    EXECUDA(cudaMemcpy((void*)P_array, (void*)GPU_phase, ARG.frameSize * sizeof(T),
                       cudaMemcpyDeviceToHost));
    EXECUDA(cudaFree(GPU_intensity));
    EXECUDA(cudaFree(GPU_phase));
    EXECUDA(cudaFree(GPU_envelope));
    EXECUDA(cudaFree(GPU_FTenvelope));
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

template <typename T>
void processFrameDouble(int _FTPLID,
                        Args ARG,
                        io::DenSupportedType dataType,
                        cufftHandle FFT,
                        cufftHandle IFT,
                        uint32_t k_in,
                        uint32_t k_out,
                        std::shared_ptr<io::DenFrame2DReader<T>>& intensityReader,
                        std::shared_ptr<io::DenFrame2DReader<T>>& phaseReader,
                        std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>>& intensityWritter,
                        std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>>& phaseWritter)
{
    KCTERR("Not yet implemented, need to implement double CUDA functions.");
}

template <typename T>
void processFrame(int _FTPLID,
                  Args ARG,
                  io::DenSupportedType dataType,
                  cufftHandle FFT,
                  cufftHandle IFT,
                  uint32_t k_in,
                  uint32_t k_out,
                  std::shared_ptr<io::DenFrame2DReader<T>>& intensityReader,
                  std::shared_ptr<io::DenFrame2DReader<T>>& phaseReader,
                  std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>>& intensityWritter,
                  std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>>& phaseWritter)
{
    if(dataType == io::DenSupportedType::FLOAT32)
    {
        processFrameFloat<T>(_FTPLID, ARG, dataType, FFT, IFT, k_in, k_out, intensityReader,
                             phaseReader, intensityWritter, phaseWritter);
    } else if(dataType == io::DenSupportedType::FLOAT64)
    {
        processFrameDouble<T>(_FTPLID, ARG, dataType, FFT, IFT, k_in, k_out, intensityReader,
                              phaseReader, intensityWritter, phaseWritter);
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
    std::shared_ptr<io::DenFrame2DReader<T>> intensityReader
        = std::make_shared<io::DenFrame2DReader<T>>(ARG.input_intensity, ARG.threads);
    std::shared_ptr<io::DenFrame2DReader<T>> phaseReader
        = std::make_shared<io::DenFrame2DReader<T>>(ARG.input_phase, ARG.threads);
    std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>> outputIntensityWritter,
        outputPhaseWritter;
    if(ARG.outputFilesExist)
    {
        outputIntensityWritter = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(
            ARG.output_intensity, ARG.dimx, ARG.dimy, ARG.dimz);
        outputPhaseWritter = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(
            ARG.output_phase, ARG.dimx, ARG.dimy, ARG.dimz);
    } else
    {
        outputIntensityWritter = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(
            ARG.output_intensity, ARG.dimx, ARG.dimy, ARG.frames.size());

        outputPhaseWritter = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(
            ARG.output_phase, ARG.dimx, ARG.dimy, ARG.frames.size());
    }
    const int dummy_FTPLID = 0;
    uint32_t k_in, k_out;
    cufftHandle FFT, IFT;
    // First is  slowest changing dimension, last is  slowest changing dimension
    if(ARG.paddingNone)
    {
        EXECUFFT(cufftPlan2d(&FFT, ARG.dimy, ARG.dimx, CUFFT_C2C));
        EXECUFFT(cufftPlan2d(&IFT, ARG.dimy, ARG.dimx, CUFFT_C2C));
    } else
    {
        EXECUFFT(cufftPlan2d(&FFT, 2 * ARG.dimy, 2 * ARG.dimx, CUFFT_C2C));
        EXECUFFT(cufftPlan2d(&IFT, 2 * ARG.dimy, 2 * ARG.dimx, CUFFT_C2C));
    }
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
        if(threadpool)
        {
            threadpool->push(processFrame<T>, ARG, dataType, FFT, IFT, k_in, k_out, intensityReader,
                             phaseReader, outputIntensityWritter, outputPhaseWritter);
        } else
        {
            processFrame<T>(dummy_FTPLID, ARG, dataType, FFT, IFT, k_in, k_out, intensityReader,
                            phaseReader, outputIntensityWritter, outputPhaseWritter);
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
