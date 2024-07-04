// The purpose of this tool is to filter out outer bone structures.
// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctype.h>
#include <future>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

// External libraries
#include "CLI/CLI.hpp" //Command line parser

// Internal libraries
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DBufferedWritter.hpp"
#include "DEN/DenFile.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "FrameMemoryViewer2D.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/Program.hpp"

using namespace KCT;
using namespace KCT::util;

template <typename T>
using READER = io::DenFrame2DReader<T>;

template <typename T>
using READERPTR = std::shared_ptr<READER<T>>;

template <typename T>
using WRITER = io::DenAsyncFrame2DBufferedWritter<T>;

template <typename T>
using WRITERPTR = std::shared_ptr<WRITER<T>>;

template <typename T>
using FRAME = io::BufferedFrame2D<T>;

template <typename T>
using FRAMEPTR = std::shared_ptr<FRAME<T>>;

using namespace KCT::util;

// class declarations
// class declarations
class Args : public ArgumentsForce, public ArgumentsFramespec, public ArgumentsThreading
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName)
        , ArgumentsFramespec(argc, argv, prgName)
        , ArgumentsThreading(argc, argv, prgName){};
    std::string input_den = "";
    std::string output_den = "";
    bool sum = false;
    bool avg = false;
    bool variance = false;
    bool sampleVariance = false;
    bool standardDeviation = false;
    bool sampleStandardDeviation = false;
    bool mad = false;
    bool min = false;
    bool max = false;
    bool median = false;
};

// Function to apply a given operation on a subset of frames
template <typename T, typename Op>
FRAMEPTR<T> aggregateFramesPartial(Args ARG, uint64_t start, uint64_t end, Op operation)
{
    io::DenFileInfo di(ARG.input_den);
    uint64_t frameSize = di.getFrameSize();
    READERPTR<T> denReader = std::make_shared<READER<T>>(ARG.input_den);

    // Read the first frame
    FRAMEPTR<T> F = denReader->readBufferedFrame(ARG.frames[start]);
    T* F_array = F->getDataPointer();

    // Apply the operation for each subsequent frame in the range
    for(uint64_t k = start + 1; k < end; ++k)
    {
        FRAMEPTR<T> A = denReader->readBufferedFrame(ARG.frames[k]);
        T* A_array = A->getDataPointer();
        std::transform(F_array, F_array + frameSize, A_array, F_array, operation);
    }

    return F;
}

// General function to aggregate frames using the specified operation and combination operation
template <typename T, typename Op, typename AggOp>
FRAMEPTR<T> aggregateFrames(Args ARG, Op operation, AggOp combineOp)
{
    io::DenFileInfo di(ARG.input_den);
    uint64_t frameSize = di.getFrameSize();
    uint64_t frameCount = ARG.frames.size();
    uint32_t threadCount = ARG.threads;

    if(threadCount == 0)
    {
        // Single-threaded processing
        return aggregateFramesPartial<T>(ARG, 0, frameCount, operation);
    } else
    {
        // Multi-threaded processing
        uint64_t framesPerThread = frameCount / threadCount;
        std::vector<std::future<FRAMEPTR<T>>> futures;

        // Launch threads to process subsets of frames
        for(uint32_t i = 0; i < threadCount; ++i)
        {
            uint64_t startFrame = i * framesPerThread;
            uint64_t endFrame = std::min(startFrame + framesPerThread, frameCount);
            futures.emplace_back(std::async(std::launch::async, aggregateFramesPartial<T, Op>, ARG,
                                            startFrame, endFrame, operation));
        }

        // Aggregate results from each thread
        FRAMEPTR<T> F = futures[0].get();
        T* result = F->getDataPointer();
        for(uint32_t i = 1; i < threadCount; ++i)
        {
            FRAMEPTR<T> A = futures[i].get();
            T* A_array = A->getDataPointer();
            std::transform(result, result + frameSize, A_array, result, combineOp);
        }

        return F;
    }
}

template <typename T>
FRAMEPTR<T> sumFrames(Args ARG)
{
    return aggregateFrames<T>(ARG, std::plus<T>(), std::plus<T>());
}

template <typename T>
FRAMEPTR<T> minFrames(Args ARG)
{
    return aggregateFrames<T>(
        ARG, [](T a, T b) { return std::min(a, b); }, [](T a, T b) { return std::min(a, b); });
}

template <typename T>
FRAMEPTR<T> maxFrames(Args ARG)
{
    return aggregateFrames<T>(
        ARG, [](T a, T b) { return std::max(a, b); }, [](T a, T b) { return std::max(a, b); });
}

template <typename T>
FRAMEPTR<T> averageFrames(Args ARG)
{
    io::DenFileInfo di(ARG.input_den);
    uint64_t frameSize = di.getFrameSize();
    uint64_t frameCount = ARG.frames.size();
    FRAMEPTR<T> F = sumFrames<T>(ARG);
    T* sum = F->getDataPointer();
    std::transform(sum, sum + frameSize, sum, [frameCount](T x) { return x / frameCount; });
    return F;
}

template <typename T>
FRAMEPTR<T>
sumOfSquaredDifferecesOfFramesPartial(Args ARG, FRAMEPTR<T> avg, uint64_t start, uint64_t end)
{
    io::DenFileInfo di(ARG.input_den);
    uint64_t frameSize = di.getFrameSize();
    READERPTR<T> denReader = std::make_shared<READER<T>>(ARG.input_den);
    T* avg_array = avg->getDataPointer();

    // Read the first frame
    FRAMEPTR<T> F = denReader->readBufferedFrame(ARG.frames[start]);
    T* F_array = F->getDataPointer();
    std::transform(F_array, F_array + frameSize, avg_array, F_array, [](T el, T avg) {
        T vel = el - avg;
        return vel * vel;
    });
    // Apply the operation for each subsequent frame in the range
    for(uint64_t k = start + 1; k < end; ++k)
    {
        FRAMEPTR<T> A = denReader->readBufferedFrame(ARG.frames[k]);
        T* A_array = A->getDataPointer();
        std::transform(A_array, A_array + frameSize, avg_array, A_array, [](T el, T avg) {
            T vel = el - avg;
            return vel * vel;
        });
        std::transform(A_array, A_array + frameSize, F_array, F_array,
                       [](T x, T v) { return x + v; });
    }
    return F;
}

template <typename T>
FRAMEPTR<T> sumOfSquaredDifferecesOfFrames(Args ARG, FRAMEPTR<T> avg)
{
    io::DenFileInfo di(ARG.input_den);
    uint64_t frameSize = di.getFrameSize();
    uint64_t frameCount = ARG.frames.size();
    uint32_t threadCount = ARG.threads;

    if(threadCount == 0)
    {
        // Single-threaded processing
        return sumOfSquaredDifferecesOfFramesPartial<T>(ARG, avg, 0, frameCount);
    } else
    {
        // Multi-threaded processing
        uint64_t framesPerThread = frameCount / threadCount;
        std::vector<std::future<FRAMEPTR<T>>> futures;

        // Launch threads to process subsets of frames
        for(uint32_t i = 0; i < threadCount; ++i)
        {
            uint64_t startFrame = i * framesPerThread;
            uint64_t endFrame = std::min(startFrame + framesPerThread, frameCount);
            futures.emplace_back(std::async(std::launch::async,
                                            sumOfSquaredDifferecesOfFramesPartial<T>, ARG, avg,
                                            startFrame, endFrame));
        }

        // Aggregate results from each thread
        FRAMEPTR<T> F = futures[0].get();
        T* result = F->getDataPointer();
        for(uint32_t i = 1; i < threadCount; ++i)
        {
            FRAMEPTR<T> A = futures[i].get();
            T* A_array = A->getDataPointer();
            std::transform(result, result + frameSize, A_array, result,
                           [](T a, T b) { return a + b; });
        }
        return F;
    }
}

// Utilizing two pass algorithm as discribed
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
template <typename T>
FRAMEPTR<T> varianceOfFrames(Args ARG, bool sampleVariance = false)
{
    io::DenFileInfo di(ARG.input_den);
    uint64_t frameCount = ARG.frames.size();
    uint64_t frameSize = di.getFrameSize();
    FRAMEPTR<T> AVG = averageFrames<T>(ARG);
    FRAMEPTR<T> F = sumOfSquaredDifferecesOfFrames<T>(ARG, AVG);
    T* sumSquaredDifferences = F->getDataPointer();
    uint32_t divideFactor = frameCount;
    if(sampleVariance)
    {
        divideFactor = frameCount - 1;
    }
    std::transform(sumSquaredDifferences, sumSquaredDifferences + frameSize, sumSquaredDifferences,
                   [divideFactor](T x) { return x / divideFactor; });
    return F;
}

// Utilizing two pass algorithm as discribed
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
template <typename T>
FRAMEPTR<T> framesStandardDeviation(Args ARG, bool sampleStandardDeviation = false)
{
    io::DenFileInfo di(ARG.input_den);
    uint64_t frameCount = ARG.frames.size();
    uint64_t frameSize = di.getFrameSize();
    FRAMEPTR<T> AVG = averageFrames<T>(ARG);
    FRAMEPTR<T> F = sumOfSquaredDifferecesOfFrames<T>(ARG, AVG);
    T* sumSquaredDifferences = F->getDataPointer();
    uint32_t divideFactor = frameCount;
    if(sampleStandardDeviation)
    {
        divideFactor = frameCount - 1;
    }
    std::transform(sumSquaredDifferences, sumSquaredDifferences + frameSize, sumSquaredDifferences,
                   [divideFactor](T x) { return std::sqrt(x / divideFactor); });
    return F;
}

template <typename T>
T getMedian(T* array, uint32_t arrayLen)
{
    uint32_t n = arrayLen / 2;
    std::nth_element(array, array + n, array + arrayLen);
    T v = array[n];
    if(arrayLen % 2 == 1)
    {
        return v;
    } else
    {
        T* max_it = std::max_element(array, array + n);
        T v_second = *max_it;
        return (v + v_second) / 2;
    }
}

template <typename T>
void medianFramesPartial(Args ARG,
                         std::shared_ptr<io::DenFile<T>> denFile,
                         T* medianArray,
                         uint64_t startFramePos,
                         uint64_t endFramePos)
{
    uint64_t frameCount = ARG.frames.size();
    uint32_t IND;
    std::vector<T> elements;
    elements.reserve(frameCount);
    for(uint64_t i = startFramePos; i < endFramePos; ++i)
    {
        for(uint64_t k = 0; k < frameCount; ++k)
        {
            IND = ARG.frames[k];
            T* framePtr = denFile->getFramePointer(IND);
            elements.push_back(*(framePtr + i));
        }
        medianArray[i] = getMedian(elements.data(), elements.size());
    }
}

template <typename T>
void madFramesPartial(Args ARG,
                      std::shared_ptr<io::DenFile<T>> denFile,
                      T* avgArray,
                      T* madArray,
                      uint64_t startFramePos,
                      uint64_t endFramePos)
{
    uint64_t frameCount = ARG.frames.size();
    uint32_t IND;
    std::vector<T> elements;
    elements.reserve(frameCount);
    for(uint64_t i = startFramePos; i < endFramePos; ++i)
    {
        for(uint64_t k = 0; k < frameCount; ++k)
        {
            IND = ARG.frames[k];
            T* framePtr = denFile->getFramePointer(IND);
            T elm = *(framePtr + i);
            T val = std::abs(elm - avgArray[i]);
            elements.push_back(val);
        }
        madArray[i] = getMedian(elements.data(), elements.size());
    }
}

template <typename T>
/**
 * @brief Here for computation of medians is needed to fill an array with all the values, sort it
 * and obtain its median. That is memory intensive for the whole dataset. But it would be I/O
 * intensive for every frame element. So we decided to do I/O row wise.
 *
 * @param a
 *
 * @return
 */
FRAMEPTR<T> medianFrames(Args ARG)
{
    io::DenFileInfo di(ARG.input_den);
    uint32_t dimx = di.dimx();
    uint32_t dimy = di.dimy();
    uint64_t frameSize = di.getFrameSize();
    FRAMEPTR<T> F = std::make_shared<FRAME<T>>(T(0), dimx, dimy);
    T* medianArray = F->getDataPointer();
    std::shared_ptr<io::DenFile<T>> denFile
        = std::make_shared<io::DenFile<T>>(ARG.input_den, ARG.threads);
    if(ARG.threads <= 1)
    {
        medianFramesPartial(ARG, denFile, medianArray, 0, frameSize);
    } else
    {
        uint32_t elementsPerThread = frameSize / ARG.threads;
        std::vector<std::future<void>> futures;
        for(uint32_t i = 0; i < ARG.threads; ++i)
        {
            uint64_t startFramePos = i * elementsPerThread;
            uint64_t endFramePos = std::min(startFramePos + elementsPerThread, frameSize);
            futures.emplace_back(std::async(std::launch::async, medianFramesPartial<T>, ARG,
                                            denFile, medianArray, startFramePos, endFramePos));
        }
    }
    return F;
}

template <typename T>
/**
 * @brief Here for computation of medians is needed to fill an array with all the values, sort it
 * and obtain its median. That is memory intensive for the whole dataset. But it would be I/O
 * intensive for every frame element. So we decided to do I/O row wise.
 *
 * @param a
 *
 * @return
 */
FRAMEPTR<T> madFrames(Args ARG)
{
    io::DenFileInfo di(ARG.input_den);
    uint32_t dimx = di.dimx();
    uint32_t dimy = di.dimy();
    uint64_t frameSize = di.getFrameSize();
    FRAMEPTR<T> AVG = averageFrames<T>(ARG);
    FRAMEPTR<T> F = std::make_shared<FRAME<T>>(T(0), dimx, dimy);
    T* madArray = F->getDataPointer();
    T* avgArray = AVG->getDataPointer();
    std::shared_ptr<io::DenFile<T>> denFile
        = std::make_shared<io::DenFile<T>>(ARG.input_den, ARG.threads);
    if(ARG.threads <= 1)
    {
        madFramesPartial(ARG, denFile, avgArray, madArray, 0, frameSize);
    } else
    {
        uint32_t elementsPerThread = frameSize / ARG.threads;
        std::vector<std::future<void>> futures;
        for(uint32_t i = 0; i < ARG.threads; ++i)
        {
            uint64_t startFramePos = i * elementsPerThread;
            uint64_t endFramePos = std::min(startFramePos + elementsPerThread, frameSize);
            futures.emplace_back(std::async(std::launch::async, madFramesPartial<T>, ARG, denFile,
                                            avgArray, madArray, startFramePos, endFramePos));
        }
    }
    return F;
}

template <typename T>
void processFiles(Args ARG)
{
    io::DenFileInfo di(ARG.input_den);
    uint32_t dimx = di.dimx();
    uint32_t dimy = di.dimy();
    WRITERPTR<T> outputWritter = std::make_shared<WRITER<T>>(ARG.output_den, dimx, dimy, 1);
    FRAMEPTR<T> f = nullptr;
    if(ARG.sum)
    {
        f = sumFrames<T>(ARG);
    } else if(ARG.avg)
    {
        f = averageFrames<T>(ARG);
    } else if(ARG.min)
    {
        f = minFrames<T>(ARG);
    } else if(ARG.max)
    {
        f = maxFrames<T>(ARG);
    } else if(ARG.variance)
    {
        f = varianceOfFrames<T>(ARG, false);
    } else if(ARG.sampleVariance)
    {
        f = varianceOfFrames<T>(ARG, true);
    } else if(ARG.standardDeviation)
    {
        f = framesStandardDeviation<T>(ARG, false);
    } else if(ARG.sampleStandardDeviation)
    {
        f = framesStandardDeviation<T>(ARG, true);
    } else if(ARG.median)
    {
        f = medianFrames<T>(ARG);
    } else if(ARG.mad)
    {
        f = madFrames<T>(ARG);
    } else
    {
        KCTERR("No operation selected.");
    }
    if(f != nullptr)
    {
        outputWritter->writeFrame(*f, 0);
    }
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    Args ARG(argc, argv, "Aggregate data along the XY frame.");
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    io::DenFileInfo di(ARG.input_den);
    io::DenSupportedType dataType = di.getElementType();
    switch(dataType)
    {
    case io::DenSupportedType::UINT16: {
        processFiles<uint16_t>(ARG);
        break;
    }
    case io::DenSupportedType::FLOAT32: {
        processFiles<float>(ARG);
        break;
    }
    case io::DenSupportedType::FLOAT64: {
        processFiles<double>(ARG);
        break;
    }
    default: {
        std::string errMsg = io::xprintf("Unsupported data type %s.",
                                         io::DenSupportedTypeToString(dataType).c_str());
        KCTERR(errMsg);
    }
    }
    PRG.endLog(true);
}

void Args::defineArguments()
{
    cliApp->add_option("input_den", input_den, "Input file.")->check(CLI::ExistingFile)->required();
    cliApp->add_option("output_den", output_den, "Output file.")->required();
    addForceArgs();
    addFramespecArgs();
    addThreadingArgs();
    CLI::Option_group* op_clg = cliApp->add_option_group(
        "Operation", "Mathematical operation f to perform element wise to get OUTPUT=f(INPUT).");
    registerOptionGroup("operation", op_clg);
    registerOption("sum", op_clg->add_flag("--sum", sum, "Sum of the data aggregated by frame."));
    registerOption("avg",
                   op_clg->add_flag("--avg", avg, "Average of the data aggregated by frame."));
    registerOption("variance",
                   op_clg->add_flag("--variance", variance,
                                    "Variance of the data over frames, this might be biassed if "
                                    "the frames does not represent whole sample."));
    registerOption(
        "samplevariance",
        op_clg->add_flag("--sample-variance", sampleVariance,
                         "Sample variance estimator, which use (n-1) sample frames to divide."));
    registerOption(
        "stdev",
        op_clg->add_flag("--standard-deviation", standardDeviation,
                         "Standard deviation of the data over frames, this might be biassed if "
                         "the frames does not represent whole sample."));
    registerOption(
        "samplestdev",
        op_clg->add_flag("--sample-standard-deviation", sampleStandardDeviation,
                         "Sample standard deviation estimator of the data over frames, mihgt be "
                         "slightly biased, see https://en.wikipedia.org/wiki/Standard_deviation."));
    registerOption("max", op_clg->add_flag("--max", max, "Max of the data aggregated by frame."));
    registerOption("min", op_clg->add_flag("--min", min, "Min of the data aggregated by frame."));
    registerOption("median",
                   op_clg->add_flag("--median", median, "Median of the data aggregated by frame."));
    registerOption("mad",
                   op_clg->add_flag("--mad", mad,
                                    "Median absolute deviation of the data aggregated by frame."));
    op_clg->require_option(1);
}

int Args::postParse()
{
    if(!force)
    {
        if(io::pathExists(output_den))
        {
            LOGE << "Error: output file already exists, use --force to force overwrite.";
            return 1;
        }
    }
    if(getRegisteredOption("avg")->count() > 0)
    {
        avg = true;
    }
    if(getRegisteredOption("sum")->count() > 0)
    {
        sum = true;
    }
    io::DenFileInfo di(input_den);
    fillFramesVector(di.dimz());
    return 0;
}
