#include "PLOG/PlogSetup.h"

// External libraries
#include "CLI/CLI.hpp" //Command line parser
#include "ctpl_stl.h" //Threadpool

// Internal libraries
#include "ARGPARSE/parseArgs.h"
#include "AsyncFrame2DWritterI.hpp"
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "DEN/DenSupportedType.hpp"
#include "Frame2DReaderI.hpp"
#include "littleEndianAlignment.h"
#include "rawop.h"

using namespace CTL;

struct Args
{
    std::string inputFile;
    std::string outputFile;
    std::string frameSpecs = "";
    std::vector<int> frames;
    int threads = 0;
    bool force = false;
    float geq = -std::numeric_limits<float>::infinity();
    float leq = std::numeric_limits<float>::infinity();
    float upq = 1.0;
    float loq = 1.0;
    int parseArguments(int argc, char* argv[]);
};

/**Argument parsing
 *
 */
int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Create alpha channel or mask of a DEN file." };
    app.add_flag("-f,--force", force, "Overwrite outputDenFile if it exists.");
    app.add_option("input_file", inputFile, "Input DEN file.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("output_file", outputFile, "Output DEN file.")->required();
    app.add_option("--leq", leq, "In alpha channel will be just the items less or equal than x.");
    app.add_option("--geq", geq,
                   "In alpha channel will be just the items greater or equal than x.");
    app.add_option("--upper-quantile", upq, "In alpha channel will be just the upq highest values.")
        ->check(CLI::Range(0.0, 1.0));
    app.add_option("--lower-quantile", loq, "In alpha channel will be just the loq smalest values.")
        ->check(CLI::Range(0.0, 1.0));
    app.add_option("-j,--threads", threads, "Number of extra threads that application can use.")
        ->check(CLI::Range(0, 65535));
    try
    {
        app.parse(argc, argv);
        if(!force)
        {
            if(io::fileExists(outputFile))
            {
                std::string msg
                    = "Error: output file already exists, use --force to force overwrite.";
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
        frames = util::processFramesSpecification(frameSpecs, inf.dimz());
    } catch(const CLI::ParseError& e)
    {
        int exitcode = app.exit(e);
        if(exitcode == 0) // Help message was printed
        {
            return 1;
        } else
        {
            LOGE << io::xprintf("There was perse error catched.\n %s", app.help().c_str());
            return -1;
        }
    }
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
                       T geq)
{
    std::shared_ptr<io::Frame2DI<T>> f = denSliceReader->readFrame(fromId);
    io::BufferedFrame2D<T> alpha(T(0), f->dimx(), f->dimy());
    for(std::size_t i = 0; i != f->dimx(); i++)
    {
        for(std::size_t j = 0; j != f->dimy(); j++)
        {
            T elm = f->get(i, j);
            if(elm >= geq && elm <= leq)
            {
                alpha.set(T(1), i, j);
            }
        }
    }
    imagesWritter->writeFrame(alpha, toId);
}

int main(int argc, char* argv[])
{
    plog::Severity verbosityLevel = plog::debug; // debug, info, ...
    std::string csvLogFile = io::xprintf(
        "/tmp/%s.csv", io::getBasename(std::string(argv[0])).c_str()); // Set NULL to disable
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    // Argument parsing
    Args a;
    int parseResult = a.parseArguments(argc, argv);
    if(parseResult != 0)
    {
        if(parseResult > 0)
        {
            return 0; // Exited sucesfully, help message printed
        } else
        {
            return -1; // Exited somehow wrong
        }
    }
    LOGI << io::xprintf("START %s", argv[0]);
    io::DenFileInfo di(a.inputFile);
    io::DenSupportedType dataType = di.getDataType();
    int dimx = di.dimx();
    int dimy = di.dimy();
    uint32_t totalSize = dimx * dimy * a.frames.size();
    ctpl::thread_pool* threadpool = nullptr;
    if(a.threads > 0)
    {
        threadpool = new ctpl::thread_pool(a.threads);
    }
    switch(dataType)
    {
    case io::DenSupportedType::uint16_t_:
    {
        std::shared_ptr<io::Frame2DReaderI<uint16_t>> denSliceReader
            = std::make_shared<io::DenFrame2DReader<uint16_t>>(a.inputFile);
        std::shared_ptr<io::AsyncFrame2DWritterI<uint16_t>> imagesWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<uint16_t>>(a.outputFile, dimx, dimy,
                                                                     a.frames.size());

        uint16_t geq, leq;
        geq = (uint16_t)a.geq;
        leq = (uint16_t)a.leq;
        if(a.geq == -std::numeric_limits<float>::infinity())
        {
            geq = 0;
        }

        if(a.leq == std::numeric_limits<float>::infinity())
        {
            leq = 65535;
        }
        if(a.loq != 1.0 || a.upq != 1.0)
        {
            uint16_t* x = new uint16_t[totalSize];
            uint32_t frameSize = dimx * dimy;
            for(uint32_t i = 0; i != a.frames.size(); i++)
            {
                io::readBytesFrom(a.inputFile, a.frames[i] * frameSize * sizeof(uint16_t) + 6,
                                  (uint8_t*)&x[i * frameSize], frameSize * sizeof(uint16_t));
            }
            std::sort(x, x + totalSize, std::less<uint32_t>());
            uint32_t loqElements, upqElements;
            loqElements = (uint32_t)(a.loq * (totalSize - 1));
            upqElements = (uint32_t)(a.upq * (totalSize - 1));
            leq = std::min(leq, x[loqElements]);
            geq = std::max(geq, x[totalSize - 1 - upqElements]);
            delete[] x;
        }
        for(uint32_t i = 0; i != a.frames.size(); i++)
        {

            if(threadpool != nullptr)
            {
                threadpool->push(writeAlphaChannel<uint16_t>, a.frames[i], denSliceReader, i,
                                 imagesWritter, leq, geq);
            } else
            {

                writeAlphaChannel<uint16_t>(0, a.frames[i], denSliceReader, i, imagesWritter, leq,
                                            geq);
            }

            // Try asynchronous calls
            // threadpool->push(writeFrameUint16, a.frames[i], denSliceReader, i, imagesWritter);
        }
        break;
    }
    case io::DenSupportedType::float_:
    {
        std::shared_ptr<io::Frame2DReaderI<float>> denSliceReader
            = std::make_shared<io::DenFrame2DReader<float>>(a.inputFile);
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> imagesWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(a.outputFile, dimx, dimy,
                                                                  a.frames.size());
        float leq = a.leq;
        float geq = a.geq;
        if(a.loq != 1.0 || a.upq != 1.0)
        {
            float* x = new float[totalSize];
            uint32_t frameSize = dimx * dimy;
            for(uint32_t i = 0; i != a.frames.size(); i++)
            {
                io::readBytesFrom(a.inputFile, a.frames[i] * frameSize * sizeof(float) + 6,
                                  (uint8_t*)&x[i * frameSize], frameSize * sizeof(float));
            }
            std::sort(x, x + totalSize, std::less<float>());
            uint32_t loqElements, upqElements;
            loqElements = (uint32_t)(a.loq * (totalSize - 1));
            upqElements = (uint32_t)(a.upq * (totalSize - 1));
            leq = std::min(leq, x[loqElements]);
            geq = std::max(geq, x[totalSize - 1 - upqElements]);
            delete[] x;
        }
        for(uint32_t i = 0; i != a.frames.size(); i++)
        {
            if(threadpool != nullptr)
            {
                threadpool->push(writeAlphaChannel<float>, a.frames[i], denSliceReader, i,
                                 imagesWritter, leq, geq);
            } else
            {

                writeAlphaChannel<float>(0, a.frames[i], denSliceReader, i, imagesWritter, leq,
                                         geq);
            }
        }
        break;
    }
    case io::DenSupportedType::double_:
    {
        std::shared_ptr<io::Frame2DReaderI<double>> denSliceReader
            = std::make_shared<io::DenFrame2DReader<double>>(a.inputFile);
        std::shared_ptr<io::AsyncFrame2DWritterI<double>> imagesWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<double>>(a.outputFile, dimx, dimy,
                                                                   a.frames.size());
        double leq = a.leq;
        double geq = a.geq;
        if(a.loq != 1.0 || a.upq != 1.0)
        {
            double* x = new double[totalSize];
            uint32_t frameSize = dimx * dimy;
            for(uint32_t i = 0; i != a.frames.size(); i++)
            {
                io::readBytesFrom(a.inputFile, a.frames[i] * frameSize * sizeof(double) + 6,
                                  (uint8_t*)&x[i * frameSize], frameSize * sizeof(double));
            }
            std::sort(x, x + totalSize, std::less<double>());
            uint32_t loqElements, upqElements;
            loqElements = (uint32_t)(a.loq * (totalSize - 1));
            upqElements = (uint32_t)(a.upq * (totalSize - 1));
            leq = std::min(leq, x[loqElements]);
            geq = std::max(geq, x[totalSize - 1 - upqElements]);
            delete[] x;
        }
        for(uint32_t i = 0; i != a.frames.size(); i++)
        {
            if(threadpool != nullptr)
            {
                threadpool->push(writeAlphaChannel<double>, a.frames[i], denSliceReader, i,
                                 imagesWritter, leq, geq);
            } else
            {

                writeAlphaChannel<double>(0, a.frames[i], denSliceReader, i, imagesWritter, leq,
                                          geq);
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
    LOGI << io::xprintf("END %s", argv[0]);
}
