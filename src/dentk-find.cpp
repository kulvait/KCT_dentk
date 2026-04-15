#include "PLOG/PlogSetup.h"

// External libraries
#include "CLI/CLI.hpp" //Command line parser
#include "ftpl.h" //Threadpool
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
    uint64_t dimx, dimy;
    uint64_t totalSize;
    io::DenSupportedType dataType;
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
    io::DenFileInfo inputFileInfo(inputFile);
    dataType = inputFileInfo.getElementType();
    dimx = inputFileInfo.dimx();
    dimy = inputFileInfo.dimy();
    totalSize = dimx * dimy * frames.size();
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

template <typename T>
void preprocessFloatType(Args& ARG, io::DenSupportedType dataType)
{
    std::shared_ptr<io::Frame2DReaderI<T>> frameReader
        = std::make_shared<io::DenFrame2DReader<T>>(ARG.inputFile);
    T leq = ARG.leq;
    T geq = ARG.geq;
    T lt = ARG.lt;
    T gt = ARG.gt;
    if(ARG.loq != 1.0 || ARG.upq != 1.0)
    {
        T* filebuffer = new T[ARG.totalSize];
        uint64_t frameSize = frameReader->getFrameSize();
        for(uint32_t i = 0; i != ARG.frames.size(); i++)
        {
            frameReader->readFrameIntoBuffer(ARG.frames[i], filebuffer + i * frameSize);
        }
        std::sort(filebuffer, filebuffer + ARG.totalSize, std::less<T>());
        uint32_t loqElements, upqElements;
        loqElements = (uint32_t)(double(ARG.loq) * (ARG.totalSize - 1));
        upqElements = (uint32_t)(double(ARG.upq) * (ARG.totalSize - 1));
        leq = std::min(leq, filebuffer[loqElements]);
        geq = std::max(geq, filebuffer[ARG.totalSize - 1 - upqElements]);
        delete[] filebuffer;
    }
    ftpl::thread_pool* threadpool = nullptr;
    if(ARG.threads > 0)
    {
        threadpool = new ftpl::thread_pool(ARG.threads);
    }
    for(uint32_t i = 0; i != ARG.frames.size(); i++)
    {
        if(threadpool != nullptr)
        {
            threadpool->push(findValues<T>, ARG.frames[i], frameReader, leq, geq, lt, gt,
                             ARG.nan, ARG.inf);
        } else
        {

            findValues<T>(0, ARG.frames[i], frameReader, leq, geq, lt, gt, ARG.nan, ARG.inf);
        }
    }
    if(threadpool != nullptr)
    {
        threadpool->stop(true);
        delete threadpool;
    }
    return;
}

//Integer types does not have concept of NAN and INF, so we ignore those flags and only check the range.
//lt and gt are ignored for integer types, because they do not make much sense. If the user wants to use them, we convert to leq and geq, so we do not need to check them here.
template <typename T>
void findIntegerValues(
    int id, int fromId, std::shared_ptr<io::Frame2DReaderI<T>> denSliceReader, T leq, T geq)
{
    std::shared_ptr<io::Frame2DI<T>> f = denSliceReader->readFrame(fromId);
    for(std::size_t i = 0; i != f->dimx(); i++)
    {
        for(std::size_t j = 0; j != f->dimy(); j++)
        {
            T elm = f->get(i, j);
            if(elm >= geq && elm <= leq)
            {
                std::cout << io::xprintf("val(%d, %d, %d) = %d\n", i, j, fromId, int64_t(elm));
            }
        }
    }
}

template <typename T>
void preprocessIntegerType(Args& ARG, io::DenSupportedType dataType)
{
    std::shared_ptr<io::Frame2DReaderI<T>> frameReader
        = std::make_shared<io::DenFrame2DReader<T>>(ARG.inputFile);
    auto clamp_to_T = [](float v) -> T {
        if(v <= static_cast<float>(std::numeric_limits<T>::lowest()))
            return std::numeric_limits<T>::lowest();
        if(v >= static_cast<float>(std::numeric_limits<T>::max()))
            return std::numeric_limits<T>::max();
        return static_cast<T>(v);
    };

    T leq = std::numeric_limits<T>::max();
    T geq = std::numeric_limits<T>::lowest();
    if(ARG.geq != -std::numeric_limits<float>::infinity())
    {
        geq = clamp_to_T(ARG.geq);
    }
    if(ARG.leq != std::numeric_limits<float>::infinity())
    {
        leq = clamp_to_T(ARG.leq);
    }
    if(ARG.lt != std::numeric_limits<float>::infinity())
    {
        T lt = clamp_to_T(ARG.lt);
        if(lt == std::numeric_limits<T>::lowest())
        {
            //No such value can exist in the data type, so we can just ignore this condition, but we warn the user about it.
            LOGW << io::xprintf("The value of lt is too small for the data type %s.",
                                io::DenSupportedTypeToString(dataType).c_str());
            return;
        }
        T lt_leq = lt - 1;
        leq = std::min(leq, lt_leq);
    }
    if(ARG.gt != -std::numeric_limits<float>::infinity())
    {
        T gt = clamp_to_T(ARG.gt);
        if(gt == std::numeric_limits<T>::max())
        {
            //No such value can exist in the data type, so we can just ignore this condition, but we warn the user about it.
            LOGW << io::xprintf("The value of gt is too big for the data type %s.",
                                io::DenSupportedTypeToString(dataType).c_str());
            return;
        }
        T gt_geq = gt + 1;
        geq = std::max(geq, gt_geq);
    }
    if(ARG.loq != 1.0 || ARG.upq != 1.0)
    {
        T* filebuffer = new T[ARG.totalSize];
        uint64_t frameSize = frameReader->getFrameSize();
        for(uint32_t i = 0; i != ARG.frames.size(); i++)
        {
            frameReader->readFrameIntoBuffer(ARG.frames[i], filebuffer + i * frameSize);
        }
        std::sort(filebuffer, filebuffer + ARG.totalSize, std::less<T>());
        uint32_t loqElements, upqElements;
        loqElements = (uint32_t)(double(ARG.loq) * (ARG.totalSize - 1));
        upqElements = (uint32_t)(double(ARG.upq) * (ARG.totalSize - 1));
        leq = std::min(leq, filebuffer[loqElements]);
        geq = std::max(geq, filebuffer[ARG.totalSize - 1 - upqElements]);
        delete[] filebuffer;
    }
    LOGI << io::xprintf("Finding values in file %s with geq=%d, leq=%d",
                        io::DenSupportedTypeToString(dataType).c_str(), int64_t(geq), int64_t(leq));
    ftpl::thread_pool* threadpool = nullptr;
    if(ARG.threads > 0)
    {
        threadpool = new ftpl::thread_pool(ARG.threads);
    }
    for(uint32_t i = 0; i != ARG.frames.size(); i++)
    {
        if(threadpool != nullptr)
        {
            threadpool->push(findIntegerValues<T>, ARG.frames[i], frameReader, leq, geq);
        } else
        {
            findIntegerValues<T>(0, ARG.frames[i], frameReader, leq, geq);
        }
    }
    if(threadpool != nullptr)
    {
        threadpool->stop(true);
        delete threadpool;
    }
    return;
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
    io::DenSupportedType dataType = inputFileInfo.getElementType();
    uint64_t dimx = inputFileInfo.dimx();
    uint64_t dimy = inputFileInfo.dimy();
    uint64_t totalSize = dimx * dimy * ARG.frames.size();
    if(totalSize > std::numeric_limits<uint32_t>::max())
    {
        LOGI << io::xprintf("The size of file %s is %lu that is bigger than MAX_UINT32!",
                            ARG.inputFile.c_str(), totalSize);
    }
    switch(dataType)
    {
    case io::DenSupportedType::UINT8: {
        preprocessIntegerType<uint8_t>(ARG, dataType);
        break;
    }

    case io::DenSupportedType::UINT16: {
        preprocessIntegerType<uint16_t>(ARG, dataType);
        break;
    }
    case io::DenSupportedType::UINT32: {
        preprocessIntegerType<uint32_t>(ARG, dataType);
        break;
    }
    case io::DenSupportedType::UINT64: {
        preprocessIntegerType<uint64_t>(ARG, dataType);
        break;
    }
    case io::DenSupportedType::INT16: {
        preprocessIntegerType<int16_t>(ARG, dataType);
        break;
    }
    case io::DenSupportedType::INT32: {
        preprocessIntegerType<int32_t>(ARG, dataType);
        break;
    }
    case io::DenSupportedType::INT64: {
        preprocessIntegerType<int64_t>(ARG, dataType);
        break;
    }
    case io::DenSupportedType::FLOAT32: {
        preprocessFloatType<float>(ARG, dataType);
        break;
    }
    case io::DenSupportedType::FLOAT64: {
        preprocessFloatType<double>(ARG, dataType);
        break;
    }
    default:
        std::string errMsg = io::xprintf("Unsupported data type %s.",
                                         io::DenSupportedTypeToString(dataType).c_str());
        KCTERR(errMsg);
    }
    PRG.endLog();
}
