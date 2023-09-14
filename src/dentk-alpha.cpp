#include "PLOG/PlogSetup.h"

// External libraries
#include "CLI/CLI.hpp" //Command line parser
#include "ftpl.h" //Threadpool

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
    float leq = std::numeric_limits<float>::infinity();
    float geq = -std::numeric_limits<float>::infinity();
    float lt = std::numeric_limits<float>::infinity();
    float gt = -std::numeric_limits<float>::infinity();
    float loq = 1.0;
    float upq = 1.0;
    float loq_frame = 1.0;
    float upq_frame = 1.0;
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
        ->add_option("--lower-quantile", loq,
                     "In alpha channel will be just the loq smalest values computed over all "
                     "admissible frames.")
        ->check(CLI::Range(0.0, 1.0));
    cliApp
        ->add_option("--upper-quantile", upq,
                     "In alpha channel will be just the upq highest values computed over all "
                     "admissible frames.")
        ->check(CLI::Range(0.0, 1.0));
    cliApp
        ->add_option("--frame-lower-quantile", loq_frame,
                     "In alpha channel will be just the loq smalest values computed frame wise.")
        ->check(CLI::Range(0.0, 1.0));
    cliApp
        ->add_option("--frame-upper-quantile", upq_frame,
                     "In alpha channel will be just the upq highest values computed frame wise.")
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
void writeAlphaChannel(int ftpl_id,
                       int fromId,
                       std::shared_ptr<io::DenFrame2DReader<T>> denFrameReader,
                       int toId,
                       std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter,
                       T leq,
                       T geq,
                       T lt,
                       T gt,
                       float loq_frame,
                       float upq_frame)
{
    std::shared_ptr<io::BufferedFrame2D<T>> f = denFrameReader->readBufferedFrame(fromId);
    uint64_t frameSize = (uint64_t)f->dimx() * (uint64_t)f->dimy();
    io::BufferedFrame2D<T> alpha(T(0), f->dimx(), f->dimy());
    T* f_array = f->getDataPointer();
    T* a_array = alpha.getDataPointer();
    if(loq_frame != 1.0f || upq_frame != 1.0f)
    {
        T* x = new T[frameSize];
        std::copy(f_array, f_array + frameSize, x);
        std::sort(x, x + frameSize, std::less<T>());
        uint32_t loqElements, upqElements;
        loqElements = (uint32_t)(loq_frame * (frameSize - 1));
        upqElements = (uint32_t)(upq_frame * (frameSize - 1));
        leq = std::min(leq, x[loqElements]);
        geq = std::max(geq, x[frameSize - 1 - upqElements]);
        delete[] x;
    }
    std::transform(f_array, f_array + frameSize, a_array, [geq, leq, gt, lt](const T& elm) {
        if(elm >= geq && elm <= leq && elm > gt && elm < lt)
        {
            return T(1);
        } else
        {
            return T(0);
        }
    });
    imagesWritter->writeFrame(alpha, toId);
}

template <typename T>
void processAlpha(Args ARG)
{
    ftpl::thread_pool* threadpool = nullptr;
    if(ARG.threads > 0)
    {
        threadpool = new ftpl::thread_pool(ARG.threads);
    }
    io::DenFileInfo inputFileInfo(ARG.inputFile);
    io::DenSupportedType dataType = inputFileInfo.getElementType();
    uint32_t dimx = inputFileInfo.dimx();
    uint32_t dimy = inputFileInfo.dimy();
    uint64_t totalSize = (uint64_t)dimx * (uint64_t)dimy * (uint64_t)ARG.frames.size();
    std::shared_ptr<io::DenFrame2DReader<T>> denSliceReader
        = std::make_shared<io::DenFrame2DReader<T>>(ARG.inputFile);
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter
        = std::make_shared<io::DenAsyncFrame2DWritter<T>>(ARG.outputFile, dimx, dimy,
                                                          ARG.frames.size());
    T geq, leq, gt, lt;
    geq = (T)ARG.geq;
    leq = (T)ARG.leq;
    gt = (T)ARG.gt;
    lt = (T)ARG.lt;
    if(dataType == io::DenSupportedType::UINT16)
    {
        if(ARG.geq <= 0)
        {
            geq = 0;
        }

        if(ARG.leq >= 65535.0)
        {
            leq = 65535;
        }
    }
    if(ARG.loq != 1.0 || ARG.upq != 1.0)
    {
        T* x = new T[totalSize];
        uint64_t frameSize = (uint64_t)dimx * (uint64_t)dimy;
        for(uint32_t i = 0; i != ARG.frames.size(); i++)
        {
            io::readBytesFrom(ARG.inputFile,
                              uint64_t(ARG.frames[i]) * frameSize * sizeof(T)
                                  + inputFileInfo.getOffset(),
                              (uint8_t*)&x[i * frameSize], frameSize * sizeof(T));
        }
        std::sort(x, x + totalSize, std::less<T>());
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
            threadpool->push(writeAlphaChannel<T>, ARG.frames[i], denSliceReader, i, imagesWritter,
                             leq, geq, lt, gt, ARG.loq_frame, ARG.upq_frame);
        } else
        {

            writeAlphaChannel<T>(0, ARG.frames[i], denSliceReader, i, imagesWritter, leq, geq, lt,
                                 gt, ARG.loq_frame, ARG.upq_frame);
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
    Args ARG(argc, argv, "Create alpha channel or mask based on the DEN file.");
    int parseResult = ARG.parse(false);
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    io::DenFileInfo inputFileInfo(ARG.inputFile);
    io::DenSupportedType dataType = inputFileInfo.getElementType();
    uint32_t dimx = inputFileInfo.dimx();
    uint32_t dimy = inputFileInfo.dimy();
    uint64_t totalSize = dimx * dimy * ARG.frames.size();
    if(totalSize > std::numeric_limits<uint32_t>::max())
    {
        LOGI << io::xprintf("The size of file %s is %lu that is bigger than MAX_UINT32!",
                            ARG.inputFile.c_str(), totalSize);
    }
    switch(dataType)
    {

    case io::DenSupportedType::UINT16:
        processAlpha<uint16_t>(ARG);
        break;
    case io::DenSupportedType::FLOAT32:
        processAlpha<float>(ARG);
        break;
    case io::DenSupportedType::FLOAT64:
        processAlpha<double>(ARG);
        break;
    default:
        std::string errMsg = io::xprintf("Unsupported data type %s.",
                                         io::DenSupportedTypeToString(dataType).c_str());
        KCTERR(errMsg);
        break;
    }
    PRG.endLog(true);
}
