#include "PLOG/PlogSetup.h"

// External libraries
#include "CLI/CLI.hpp" //Command line parser
#include "ctpl_stl.h" //Threadpool
#include <cmath>

// Internal libraries
#include "AsyncFrame2DWritterI.hpp"
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "DEN/DenSupportedType.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/Program.hpp"
#include "littleEndianAlignment.h"
#include "rawop.h"

using namespace KCT;
using namespace KCT::util;

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
    float geq = -std::numeric_limits<float>::infinity();
    float leq = std::numeric_limits<float>::infinity();
    float gt = -std::numeric_limits<float>::infinity();
    float lt = std::numeric_limits<float>::infinity();
    float upq = 1.0;
    float loq = 1.0;
    bool nan = false;
    bool inf = false;
};

/**Argument parsing
 *
 */
void Args::defineArguments()
{
    cliApp->add_option("input_file", inputFile, "Input DEN file.")
        ->required()
        ->check(CLI::ExistingFile);
    addFramespecArgs();
    addThreadingArgs();
    cliApp->add_option("--leq", leq, "Find values less or equal than x.");
    cliApp->add_option("--geq", geq, "Find values greater or equal than x.");
    cliApp->add_option("--lt", lt, "Find values less than x.");
    cliApp->add_option("--gt", gt, "Find values greater than x.");
    cliApp->add_option("--upper-quantile", upq, "Find the upq highest values.")
        ->check(CLI::Range(0.0, 1.0));
    cliApp->add_option("--lower-quantile", loq, "Find the loq smalest values.")
        ->check(CLI::Range(0.0, 1.0));
    cliApp->add_flag("--nan", nan, "Find NAN values.");
    cliApp->add_flag("--inf", inf, "Find INF values.");
}

int Args::postParse()
{
    io::DenFileInfo inf(inputFile);
    fillFramesVector(inf.dimz());
    return 0;
}

/**
 * @brief Results are 0 or 1 based on the
 *
 * @param f
 */
template <typename T>
void findValues(int id,
                int fromId,
                std::shared_ptr<io::Frame2DReaderI<T>> denSliceReader,
                T leq,
                T geq,
                T lt,
                T gt,
                bool nan,
                bool inf)
{
    std::shared_ptr<io::Frame2DI<T>> f = denSliceReader->readFrame(fromId);
    for(std::size_t i = 0; i != f->dimx(); i++)
    {
        for(std::size_t j = 0; j != f->dimy(); j++)
        {
            T elm = f->get(i, j);
            if(nan)
            {
                if(std::isnan(double(elm)))
                {
                    std::cout << io::xprintf("val(%d, %d, %d) = NAN\n", i, j, fromId);
                }
            } else if(inf)
            {
                if(std::isinf(double(elm)))
                {
                    if(double(elm) > 0)
                    {
                        std::cout << io::xprintf("val(%d, %d, %d) = +INF\n", i, j, fromId);
                    } else
                    {
                        std::cout << io::xprintf("val(%d, %d, %d) = -INF\n", i, j, fromId);
                    }
                }
            } else if(elm >= geq && elm <= leq && elm > gt && elm < lt)
            {
                std::cout << io::xprintf("val(%d, %d, %d) = %f\n", i, j, fromId, double(elm));
            }
        }
    }
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    Args ARG(argc, argv, "Find values with their positions in DEN files.");
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    io::DenFileInfo inputFileInfo(ARG.inputFile);
    io::DenSupportedType dataType = inputFileInfo.getDataType();
    int dimx = inputFileInfo.dimx();
    int dimy = inputFileInfo.dimy();
    uint64_t totalSize = dimx * dimy * ARG.frames.size();
    if(totalSize > std::numeric_limits<uint32_t>::max())
    {
        LOGI << io::xprintf("The size of file %s is %d that is bigger than MAX_UINT32!",
                            ARG.inputFile.c_str(), totalSize);
    }
    ctpl::thread_pool* threadpool = nullptr;
    if(ARG.threads > 0)
    {
        threadpool = new ctpl::thread_pool(ARG.threads);
    }
    switch(dataType)
    {
    case io::DenSupportedType::UINT16: {
        std::shared_ptr<io::Frame2DReaderI<uint16_t>> denSliceReader
            = std::make_shared<io::DenFrame2DReader<uint16_t>>(ARG.inputFile);
        uint16_t geq, leq, gt, lt;
        geq = (uint16_t)ARG.geq;
        leq = (uint16_t)ARG.leq;
        gt = (uint16_t)ARG.gt;
        lt = (uint16_t)ARG.lt;
        if(ARG.geq == -std::numeric_limits<float>::infinity())
        {
            geq = 0;
        }
        if(ARG.leq == std::numeric_limits<float>::infinity())
        {
            leq = 65535;
        }
        if(ARG.loq != 1.0 || ARG.upq != 1.0)
        {
            uint16_t* x = new uint16_t[totalSize];
            uint32_t frameSize = dimx * dimy;
            for(uint32_t i = 0; i != ARG.frames.size(); i++)
            {
                io::readBytesFrom(ARG.inputFile,
                                  uint64_t(ARG.frames[i]) * frameSize * sizeof(uint16_t)
                                      + inputFileInfo.getOffset(),
                                  (uint8_t*)&x[i * frameSize], frameSize * sizeof(uint16_t));
            }
            std::sort(x, x + totalSize, std::less<uint32_t>());
            uint32_t loqElements, upqElements;
            loqElements = (uint32_t)(double(ARG.loq) * (totalSize - 1));
            upqElements = (uint32_t)(double(ARG.upq) * (totalSize - 1));
            leq = std::min(leq, x[loqElements]);
            geq = std::max(geq, x[totalSize - 1 - upqElements]);
            delete[] x;
        }
        for(uint32_t i = 0; i != ARG.frames.size(); i++)
        {
            if(threadpool != nullptr)
            {
                threadpool->push(findValues<uint16_t>, ARG.frames[i], denSliceReader, leq, geq, lt,
                                 gt, ARG.nan, ARG.inf);
            } else
            {
                findValues<uint16_t>(0, ARG.frames[i], denSliceReader, leq, geq, lt, gt, ARG.nan,
                                     ARG.inf);
            }
        }
        break;
    }
    case io::DenSupportedType::FLOAT32: {
        std::shared_ptr<io::Frame2DReaderI<float>> denSliceReader
            = std::make_shared<io::DenFrame2DReader<float>>(ARG.inputFile);
        float leq = ARG.leq;
        float geq = ARG.geq;
        float lt = ARG.lt;
        float gt = ARG.gt;
        if(ARG.loq != 1.0 || ARG.upq != 1.0)
        {
            float* x = new float[totalSize];
            uint32_t frameSize = dimx * dimy;
            for(uint32_t i = 0; i != ARG.frames.size(); i++)
            {
                io::readBytesFrom(ARG.inputFile,
                                  uint64_t(ARG.frames[i]) * frameSize * sizeof(float)
                                      + inputFileInfo.getOffset(),
                                  (uint8_t*)&x[i * frameSize], frameSize * sizeof(float));
            }
            std::sort(x, x + totalSize, std::less<float>());
            uint32_t loqElements, upqElements;
            loqElements = (uint32_t)(double(ARG.loq) * double(totalSize - 1));
            upqElements = (uint32_t)(double(ARG.upq) * double(totalSize - 1));
            leq = std::min(leq, x[loqElements]);
            geq = std::max(geq, x[totalSize - 1 - upqElements]);
            delete[] x;
        }
        for(uint32_t i = 0; i != ARG.frames.size(); i++)
        {
            if(threadpool != nullptr)
            {
                threadpool->push(findValues<float>, ARG.frames[i], denSliceReader, leq, geq, lt, gt,
                                 ARG.nan, ARG.inf);
            } else
            {

                findValues<float>(0, ARG.frames[i], denSliceReader, leq, geq, lt, gt, ARG.nan,
                                  ARG.inf);
            }
        }
        break;
    }
    case io::DenSupportedType::FLOAT64: {
        std::shared_ptr<io::Frame2DReaderI<double>> denSliceReader
            = std::make_shared<io::DenFrame2DReader<double>>(ARG.inputFile);
        double leq = ARG.leq;
        double geq = ARG.geq;
        double lt = ARG.lt;
        double gt = ARG.gt;
        if(ARG.loq != 1.0 || ARG.upq != 1.0)
        {
            double* x = new double[totalSize];
            uint32_t frameSize = dimx * dimy;
            for(uint32_t i = 0; i != ARG.frames.size(); i++)
            {
                io::readBytesFrom(ARG.inputFile,
                                  uint64_t(ARG.frames[i]) * frameSize * sizeof(double)
                                      + inputFileInfo.getOffset(),
                                  (uint8_t*)&x[i * frameSize], frameSize * sizeof(double));
            }
            std::sort(x, x + totalSize, std::less<double>());
            uint32_t loqElements, upqElements;
            loqElements = (uint32_t)(double(ARG.loq) * (totalSize - 1));
            upqElements = (uint32_t)(double(ARG.upq) * (totalSize - 1));
            leq = std::min(leq, x[loqElements]);
            geq = std::max(geq, x[totalSize - 1 - upqElements]);
            delete[] x;
        }
        for(uint32_t i = 0; i != ARG.frames.size(); i++)
        {
            if(threadpool != nullptr)
            {
                threadpool->push(findValues<double>, ARG.frames[i], denSliceReader, leq, geq, lt,
                                 gt, ARG.nan, ARG.inf);
            } else
            {

                findValues<double>(0, ARG.frames[i], denSliceReader, leq, geq, lt, gt, ARG.nan,
                                   ARG.inf);
            }
        }
        break;
    }
    default:
        std::string errMsg = io::xprintf("Unsupported data type %s.",
                                         io::DenSupportedTypeToString(dataType).c_str());
        KCTERR(errMsg);
    }
    if(threadpool != nullptr)
    {
        threadpool->stop(true);
        delete threadpool;
    }
    PRG.endLog();
}
