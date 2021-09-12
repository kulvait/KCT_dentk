#include "PLOG/PlogSetup.h"

// External libraries
#include "CLI/CLI.hpp" //Command line parser
#include "ctpl_stl.h" //Threadpool

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

class Args : public ArgumentsFramespec, public ArgumentsForce, public ArgumentsThreading
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsFramespec(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName)
        , ArgumentsThreading(argc, argv, prgName){};

    std::string inputFile;
    std::string outputFile;
    float geq = -std::numeric_limits<float>::infinity();
    float leq = std::numeric_limits<float>::infinity();
    float gt = -std::numeric_limits<float>::infinity();
    float lt = std::numeric_limits<float>::infinity();
    float upq = 1.0;
    float loq = 1.0;
};

/**Argument parsing
 *
 */
void Args::defineArguments()
{
    cliApp->add_option("input_file", inputFile, "Input DEN file.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("output_file", outputFile, "Output DEN file.")->required();
    addFramespecArgs();
    addThreadingArgs();
    addForceArgs();
    cliApp->add_option("--leq", leq,
                       "In alpha channel will be just the items less or equal than x.");
    cliApp->add_option("--geq", geq,
                       "In alpha channel will be just the items greater or equal than x.");
    cliApp->add_option("--lt", lt, "In alpha channel will be just the items less than x.");
    cliApp->add_option("--gt", gt, "In alpha channel will be just the items greater than x.");
    cliApp
        ->add_option("--upper-quantile", upq,
                     "In alpha channel will be just the upq highest values.")
        ->check(CLI::Range(0.0, 1.0));
    cliApp
        ->add_option("--lower-quantile", loq,
                     "In alpha channel will be just the loq smalest values.")
        ->check(CLI::Range(0.0, 1.0));
}

int Args::postParse()
{
    if(!force)
    {
        if(io::pathExists(outputFile))
        {
            std::string msg = "Error: output file already exists, use --force to force overwrite.";
            LOGE << msg;
            return 1;
        }
    }
    if(inputFile == outputFile)
    {
        LOGE << "Error: input and output files must differ!";
        return 1;
    }
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
void writeAlphaChannel(int id,
                       int fromId,
                       std::shared_ptr<io::Frame2DReaderI<T>> denSliceReader,
                       int toId,
                       std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter,
                       T leq,
                       T geq,
                       T lt,
                       T gt)
{
    std::shared_ptr<io::Frame2DI<T>> f = denSliceReader->readFrame(fromId);
    io::BufferedFrame2D<T> alpha(T(0), f->dimx(), f->dimy());
    for(std::size_t i = 0; i != f->dimx(); i++)
    {
        for(std::size_t j = 0; j != f->dimy(); j++)
        {
            T elm = f->get(i, j);
            if(elm >= geq && elm <= leq && elm > gt && elm < lt)
            {
                alpha.set(T(1), i, j);
            }
        }
    }
    imagesWritter->writeFrame(alpha, toId);
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    Args ARG(argc, argv, "Create alpha channel or mask based on the DEN file.");
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
    case io::DenSupportedType::uint16_t_:
    {
        std::shared_ptr<io::Frame2DReaderI<uint16_t>> denSliceReader
            = std::make_shared<io::DenFrame2DReader<uint16_t>>(ARG.inputFile);
        std::shared_ptr<io::AsyncFrame2DWritterI<uint16_t>> imagesWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<uint16_t>>(ARG.outputFile, dimx, dimy,
                                                                     ARG.frames.size());

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
                threadpool->push(writeAlphaChannel<uint16_t>, ARG.frames[i], denSliceReader, i,
                                 imagesWritter, leq, geq, lt, gt);
            } else
            {

                writeAlphaChannel<uint16_t>(0, ARG.frames[i], denSliceReader, i, imagesWritter, leq,
                                            geq, lt, gt);
            }

            // Try asynchronous calls
            // threadpool->push(writeFrameUint16, ARG.frames[i], denSliceReader, i, imagesWritter);
        }
        break;
    }
    case io::DenSupportedType::float_:
    {
        std::shared_ptr<io::Frame2DReaderI<float>> denSliceReader
            = std::make_shared<io::DenFrame2DReader<float>>(ARG.inputFile);
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> imagesWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(ARG.outputFile, dimx, dimy,
                                                                  ARG.frames.size());
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
                threadpool->push(writeAlphaChannel<float>, ARG.frames[i], denSliceReader, i,
                                 imagesWritter, leq, geq, lt, gt);
            } else
            {

                writeAlphaChannel<float>(0, ARG.frames[i], denSliceReader, i, imagesWritter, leq,
                                         geq, lt, gt);
            }
        }
        break;
    }
    case io::DenSupportedType::double_:
    {
        std::shared_ptr<io::Frame2DReaderI<double>> denSliceReader
            = std::make_shared<io::DenFrame2DReader<double>>(ARG.inputFile);
        std::shared_ptr<io::AsyncFrame2DWritterI<double>> imagesWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<double>>(ARG.outputFile, dimx, dimy,
                                                                   ARG.frames.size());
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
                threadpool->push(writeAlphaChannel<double>, ARG.frames[i], denSliceReader, i,
                                 imagesWritter, leq, geq, lt, gt);
            } else
            {

                writeAlphaChannel<double>(0, ARG.frames[i], denSliceReader, i, imagesWritter, leq,
                                          geq, lt, gt);
            }
        }
        break;
    }
    default:
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(dataType));
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
    }
    if(threadpool != nullptr)
    {
        threadpool->stop(true);
        delete threadpool;
    }
    PRG.endLog();
}
