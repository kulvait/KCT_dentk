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
    float loq_line = 1.0;
    float upq_line = 1.0;
    bool output_float32 = false, output_float64 = false, output_uint16 = false;
    io::DenSupportedType inputDataType;
    io::DenSupportedType outputDataType;
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
                       "In alpha channel will be just the items less or equal than x, quantile "
                       "operations might decrease size of alpha channel.");
    cliApp->add_option("--geq", geq,
                       "In alpha channel will be just the items greater or equal than x, quantile "
                       "operations might decrease size of alpha channel.");
    cliApp->add_option("--lt", lt,
                       "In alpha channel will be just the items less than x, quantile operations "
                       "might decrease size of alpha channel.");
    cliApp->add_option("--gt", gt,
                       "In alpha channel will be just the items greater than x, quantile "
                       "operations might decrease size of alpha channel.");
    CLI::Option* opt_loq
        = cliApp
              ->add_option("--lower-quantile", loq,
                           "In alpha channel will be just the loq smalest values computed over all "
                           "admissible frames.")
              ->check(CLI::Range(0.0, 1.0));
    CLI::Option* opt_upq
        = cliApp
              ->add_option("--upper-quantile", upq,
                           "In alpha channel will be just the upq highest values computed over all "
                           "admissible frames.")
              ->check(CLI::Range(0.0, 1.0));
    CLI::Option* opt_loq_frame
        = cliApp
              ->add_option(
                  "--frame-lower-quantile", loq_frame,
                  "In alpha channel will be just the loq smalest values computed frame wise.")
              ->check(CLI::Range(0.0, 1.0));
    CLI::Option* opt_upq_frame
        = cliApp
              ->add_option(
                  "--frame-upper-quantile", upq_frame,
                  "In alpha channel will be just the upq highest values computed frame wise.")
              ->check(CLI::Range(0.0, 1.0));
    CLI::Option* opt_loq_line
        = cliApp
              ->add_option(
                  "--line-lower-quantile", loq_line,
                  "In alpha channel will be just the loq smalest values computed line wise.")
              ->check(CLI::Range(0.0, 1.0));
    CLI::Option* opt_upq_line
        = cliApp
              ->add_option(
                  "--line-upper-quantile", upq_line,
                  "In alpha channel will be just the upq highest values computed line wise.")
              ->check(CLI::Range(0.0, 1.0));
    opt_upq_line->excludes(opt_upq_frame)
        ->excludes(opt_loq_frame)
        ->excludes(opt_upq)
        ->excludes(opt_loq);
    opt_loq_line->excludes(opt_upq_frame)
        ->excludes(opt_loq_frame)
        ->excludes(opt_upq)
        ->excludes(opt_loq);
    opt_upq_frame->excludes(opt_upq_line)
        ->excludes(opt_loq_line)
        ->excludes(opt_upq)
        ->excludes(opt_loq);
    opt_loq_frame->excludes(opt_upq_line)
        ->excludes(opt_loq_line)
        ->excludes(opt_upq)
        ->excludes(opt_loq);
    opt_upq->excludes(opt_upq_frame)
        ->excludes(opt_loq_frame)
        ->excludes(opt_upq_line)
        ->excludes(opt_loq_line);
    opt_loq->excludes(opt_upq_frame)
        ->excludes(opt_loq_frame)
        ->excludes(opt_upq_line)
        ->excludes(opt_loq_line);
    CLI::Option_group* op_clg = cliApp->add_option_group(
        "Output type", "Select type of output DEN file, by default FLOAT32.");
    registerOptionGroup("operation", op_clg);
    registerOption("output-float32",
                   op_clg->add_flag("--outupt-float32", output_float32, "FLOAT32 output."));
    registerOption("output-float64",
                   op_clg->add_flag("--output-float64", output_float64, "FLOAT64 output."));
    registerOption("output-uint16",
                   op_clg->add_flag("--output-uint16", output_uint16, "UINT16 output."));
    op_clg->require_option(0, 1);
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
    io::DenFileInfo inputFileInfo(inputFile);
    inputDataType = inputFileInfo.getElementType();
    if(output_float64)
    {
        outputDataType = io::DenSupportedType::FLOAT64;
    } else if(output_uint16)
    {
        outputDataType = io::DenSupportedType::UINT16;
    } else
    {
        outputDataType = io::DenSupportedType::FLOAT32;
    }
    return 0;
}

/**
 * @brief Results are 0 or 1 based on the
 *
 * @param f
 */
template <typename T, typename W>
void writeAlphaChannel(int ftpl_id,
                       int fromId,
                       std::shared_ptr<io::DenFrame2DReader<T>> denFrameReader,
                       int toId,
                       std::shared_ptr<io::AsyncFrame2DWritterI<W>> imagesWritter,
                       T leq,
                       T geq,
                       T lt,
                       T gt,
                       float loq_frame,
                       float upq_frame,
                       float loq_line,
                       float upq_line)
{
    std::shared_ptr<io::BufferedFrame2D<T>> f = denFrameReader->readBufferedFrame(fromId);
    uint64_t dimx, dimy;
    dimx = f->dimx();
    dimy = f->dimy();
    uint64_t frameSize = dimx * dimy;
    io::BufferedFrame2D<W> alpha(W(0), dimx, dimy);
    T* f_array = f->getDataPointer();
    W* a_array = alpha.getDataPointer();
    if(loq_line != 1.0f || upq_line != 1.0f)
    {
        T* x = new T[dimx];
        uint32_t loqElements, upqElements;
        T l, h;
        for(uint64_t j = 0; j != dimy; j++)
        {
            T* STARTIND = f_array + j * dimx;
            T* ENDIND = f_array + (j + 1) * dimx;
            std::copy(STARTIND, ENDIND, x);
            std::sort(x, x + dimx, std::less<T>());
            loqElements = (uint32_t)(loq_line * (dimx - 1));
            upqElements = (uint32_t)(upq_line * (dimx - 1));
            l = std::min(leq, x[loqElements]);
            h = std::max(geq, x[dimx - 1 - upqElements]);
            std::transform(STARTIND, ENDIND, a_array + j * dimx, [h, l](const W& elm) {
                if(elm >= h && elm <= l)
                {
                    return W(1);
                } else
                {
                    return W(0);
                }
            });
        }
        delete[] x;
    } else
    {
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
        std::transform(f_array, f_array + frameSize, a_array, [geq, leq, gt, lt](const W& elm) {
            if(elm >= geq && elm <= leq && elm > gt && elm < lt)
            {
                return W(1);
            } else
            {
                return W(0);
            }
        });
    }
    imagesWritter->writeFrame(alpha, toId);
}

template <typename T, typename W>
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
    std::shared_ptr<io::DenFrame2DReader<T>> denFrameReader
        = std::make_shared<io::DenFrame2DReader<T>>(ARG.inputFile);
    std::shared_ptr<io::AsyncFrame2DWritterI<W>> imagesWritter
        = std::make_shared<io::DenAsyncFrame2DWritter<W>>(ARG.outputFile, dimx, dimy,
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
            threadpool->push(writeAlphaChannel<T, W>, ARG.frames[i], denFrameReader, i, imagesWritter,
                             leq, geq, lt, gt, ARG.loq_frame, ARG.upq_frame, ARG.loq_line,
                             ARG.upq_line);
        } else
        {

            writeAlphaChannel<T, W>(0, ARG.frames[i], denFrameReader, i, imagesWritter, leq, geq, lt,
                                 gt, ARG.loq_frame, ARG.upq_frame, ARG.loq_line, ARG.upq_line);
        }
    }
    if(threadpool != nullptr)
    {
        threadpool->stop(true);
        delete threadpool;
    }
}
template <typename T>
void preprocessAlpha(Args ARG)
{
    switch(ARG.outputDataType)
    {

    case io::DenSupportedType::UINT16:
        processAlpha<T, uint16_t>(ARG);
        break;
    case io::DenSupportedType::FLOAT32:
        processAlpha<T, float>(ARG);
        break;
    case io::DenSupportedType::FLOAT64:
        processAlpha<T, double>(ARG);
        break;
    default:
        std::string errMsg = io::xprintf("Unsupported output data type %s.",
                                         io::DenSupportedTypeToString(ARG.outputDataType).c_str());
        KCTERR(errMsg);
        break;
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
    uint64_t dimx = inputFileInfo.dimx();
    uint64_t dimy = inputFileInfo.dimy();
    uint64_t totalSize = dimx * dimy * ARG.frames.size();
    if(totalSize > std::numeric_limits<uint32_t>::max())
    {
        LOGI << io::xprintf("The size of file %s to process is %lu that is bigger than MAX_UINT32!",
                            ARG.inputFile.c_str(), totalSize);
    }
    switch(ARG.inputDataType)
    {

    case io::DenSupportedType::UINT16:
        preprocessAlpha<uint16_t>(ARG);
        break;
    case io::DenSupportedType::FLOAT32:
        preprocessAlpha<float>(ARG);
        break;
    case io::DenSupportedType::FLOAT64:
        preprocessAlpha<double>(ARG);
        break;
    default:
        std::string errMsg = io::xprintf("Unsupported data type %s.",
                                         io::DenSupportedTypeToString(ARG.inputDataType).c_str());
        KCTERR(errMsg);
        break;
    }
    PRG.endLog(true);
}
