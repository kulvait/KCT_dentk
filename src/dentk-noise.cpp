// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctype.h>
#include <iostream>
#include <random>
#include <regex>
#include <string>

// External libraries
#include "CLI/CLI.hpp" //Command line parser
#include "ctpl_stl.h" //Threadpool
#include "mkl.h"

// Internal libraries
#include "AsyncFrame2DWritterI.hpp"
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DI.hpp"
#include "Frame2DReaderI.hpp"
#include "FrameMemoryViewer2D.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/Program.hpp"
#include "PROG/parseArgs.h"

using namespace CTL;
using namespace CTL::util;

class Args : public ArgumentsFramespec, public ArgumentsThreading
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsFramespec(argc, argv, prgName)
        , ArgumentsThreading(argc, argv, prgName){};
    std::string inputFile;
    std::string outputFile;
    bool gauss = false;
    double standardDeviation = 1.0;
    double mean = 0.0;
    bool poisson = false;
    double K0 = 7500.0;
    uint32_t dimx;
    uint32_t dimy;
    uint32_t dimz;
    bool force = false;
};

void Args::defineArguments()
{
    cliApp->add_option("input_den_file", inputFile, "File in a DEN format to process add noise to.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("output_den_file", outputFile, "File in a DEN format to output.")
        ->required();
    cliApp->add_flag("--force", force, "Overwrite outputFile if it exists.");
    CLI::Option_group* op_clg
        = cliApp->add_option_group("Noise type", "Choose the type of the noise.");
    CLI::Option* gaussDistribution
        = op_clg->add_flag("--gauss", gauss, "Gauss distribution is added to the data");
    CLI::Option* poissonDistribution
        = op_clg->add_flag("--poisson", poisson,
                           "Poisson noise with lambda=K. Data are transformed to "
                           "compute K=K0exp(-x), poisson value L is generated, "
                           "then x is substituted by ln(K0/L).");
    op_clg->require_option(1);
    op_clg = cliApp->add_option_group("Gauss distribution parameters",
                                      "Parameters of the standard distribution.");
    std::string optSpec;
    optSpec = io::xprintf("Standard distribution mean, [defaults to %f].", mean);
    CLI::Option* gaussMean = op_clg->add_option("--mean", mean, optSpec);
    optSpec = io::xprintf("Standard distribution sigma, [defaults to %f].", standardDeviation);
    CLI::Option* gaussSigma
        = op_clg->add_option("--standard-deviation", standardDeviation, optSpec);
    op_clg = cliApp->add_option_group("Poisson distribution parameters",
                                      "Parameters of the poisson distribution.");
    optSpec = io::xprintf("Poisson lambda that equals K0, number of photons per blank scan "
                          "detector [defaults to %f].",
                          K0);
    CLI::Option* poissonLambda = op_clg->add_option("--k0", K0, optSpec);
    poissonLambda->needs(poissonDistribution);
    gaussMean->needs(gaussDistribution);
    gaussSigma->needs(gaussDistribution);

    addFramespecArgs();
    addThreadingArgs();
}

int Args::postParse()
{
    // If force is not set, then check if output file does not exist
    std::string err;
    if(!force)
    {
        if(io::pathExists(outputFile))
        {
            err = "Error: output file already exists, use --force to force overwrite.";
            LOGE << err;
            return 1;
        }
    }
    // How many projection matrices is there in total
    io::DenFileInfo di(inputFile);
    dimx = di.dimx();
    dimy = di.dimy();
    dimz = di.dimz();
    fillFramesVector(dimz);
    return 0;
}

template <typename T>
void addGaussFrame(int id,
                   int fromId,
                   std::shared_ptr<io::Frame2DReaderI<T>> inputReader,
                   int toId,
                   std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter,
                   double mean,
                   double sd)
{
    uint32_t dimx = imagesWritter->dimx();
    uint32_t dimy = imagesWritter->dimy();
    uint32_t frameSize = dimx * dimy;
    T* noise = new T[frameSize];
    std::random_device randomInts;
    int seed = fromId;
    std::mt19937 engine(seed);
    std::normal_distribution<T> dis(mean, sd);
    auto gen = [&dis, &engine]() { return dis(engine); };
    std::generate(noise, noise + frameSize, gen);
    std::shared_ptr<io::Frame2DI<T>> curframe = inputReader->readFrame(fromId);
    for(uint32_t y = 0; y != dimy; y++)
    {
        for(uint32_t x = 0; x != dimx; x++)
        {
            noise[y * dimx + x] += curframe->get(x, y);
            if(noise[y * dimx + x] < 0.0)
            {
                noise[y * dimx + x] = 0.0;
            }
        }
    }
    std::unique_ptr<io::Frame2DI<T>> f
        = std::make_unique<io::FrameMemoryViewer2D<T>>(noise, dimx, dimy);
    imagesWritter->writeFrame(*f, toId);
    delete[] noise;
}

template <typename T>
void addPoissonFrame(int id,
                     int fromId,
                     std::shared_ptr<io::Frame2DReaderI<T>> inputReader,
                     int toId,
                     std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter,
                     double k0lambda)
{
    uint32_t dimx = imagesWritter->dimx();
    uint32_t dimy = imagesWritter->dimy();
    uint32_t frameSize = dimx * dimy;
    std::srand(std::time(nullptr)); // use current time as seed for random generator
    // MKL_UINT seed = fromId;
    MKL_UINT seed = std::rand();
    MKL_INT status;
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_WH, seed);
    T* frame = new T[frameSize];
    double K;
    int L;
    std::shared_ptr<io::Frame2DI<T>> curframe = inputReader->readFrame(fromId);
    for(uint32_t y = 0; y != dimy; y++)
    {
        for(uint32_t x = 0; x != dimx; x++)
        {
            K = k0lambda * exp(-curframe->get(x, y));
            status = viRngPoisson(VSL_RNG_METHOD_POISSON_PTPE, stream, 1, &L, K);
            if(status == VSL_STATUS_OK)
            {
                if(L > k0lambda)
                {
                    frame[y * dimx + x] = 0;
                } else if(L == 0)
                {
                    frame[y * dimx + x] = log(k0lambda / 0.5);
                } else
                {
                    frame[y * dimx + x] = log(k0lambda / L);
                }
            } else
            {
                std::string ERR = io::xprintf(
                    "The status for frame fromID %d toID %d x=%d y=%d is not OK and is %d", fromId,
                    toId, x, y, status);
                LOGE << ERR;
                throw std::runtime_error(ERR);
            }
        }
    }
    vslDeleteStream(&stream);
    std::unique_ptr<io::Frame2DI<T>> f
        = std::make_unique<io::FrameMemoryViewer2D<T>>(frame, dimx, dimy);
    imagesWritter->writeFrame(*f, toId);
    delete[] frame;
}

template <typename T>
void elementWiseNoise(Args a)
{
    std::shared_ptr<io::Frame2DReaderI<T>> inputReader;
    inputReader = std::make_shared<io::DenFrame2DReader<T>>(a.inputFile);
    uint32_t dimx = inputReader->dimx();
    uint32_t dimy = inputReader->dimy();
    ctpl::thread_pool* threadpool = nullptr;
    if(a.threads != 0)
    {
        threadpool = new ctpl::thread_pool(a.threads);
    }
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter
        = std::make_shared<io::DenAsyncFrame2DWritter<T>>(a.outputFile, dimx, dimy,
                                                          a.frames.size());
    for(uint32_t i = 0; i != a.frames.size(); i++)
    {
        if(threadpool != nullptr)
        {
            if(a.gauss)
            {
                threadpool->push(addGaussFrame<T>, a.frames[i], inputReader, i, imagesWritter,
                                 a.mean, a.standardDeviation);
            }
            if(a.poisson)
            {
                threadpool->push(addPoissonFrame<T>, a.frames[i], inputReader, i, imagesWritter,
                                 a.K0);
            }
        } else
        {
            if(a.gauss)
            {
                addGaussFrame<T>(0, a.frames[i], inputReader, i, imagesWritter, a.mean,
                                 a.standardDeviation);
            }
            if(a.poisson)
            {
                addPoissonFrame<T>(0, a.frames[i], inputReader, i, imagesWritter, a.K0);
            }
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
    Args ARG(argc, argv,
             "Add a noise to a DEN file, normal or Poisson distributed on transformed data.");
    // Argument parsing
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    // Frames to process
    io::DenFileInfo inf(ARG.inputFile);
    io::DenSupportedType dataType = inf.getDataType();
    switch(dataType)
    {
    case io::DenSupportedType::float_:
    {
        elementWiseNoise<float>(ARG);
        break;
    }
    case io::DenSupportedType::double_:
    {
        elementWiseNoise<double>(ARG);
        break;
    }
    case io::DenSupportedType::uint16_t_:
    {
        // elementWiseNoise<uint16_t>(ARG);
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(dataType));
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
    }
    default:
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(dataType));
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
    }
    LOGI << io::xprintf("END %s", argv[0]);
}
