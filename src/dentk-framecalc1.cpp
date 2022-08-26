// The purpose of this tool is to filter out outer bone structures.
// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctype.h>
#include <iostream>
#include <regex>
#include <string>

// External libraries
#include "CLI/CLI.hpp" //Command line parser

// Internal libraries
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "FrameMemoryViewer2D.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/Program.hpp"

using namespace KCT;
using namespace KCT::util;

// class declarations
// class declarations
class Args : public ArgumentsFramespec, public ArgumentsForce
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsFramespec(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName){};
    std::string input_den = "";
    std::string output_den = "";
    bool sum = false;
    bool avg = false;
    bool variance = false;
    bool sampleVariance = false;
    bool standardDeviation = false;
    bool sampleStandardDeviation = false;
    bool min = false;
    bool max = false;
    bool median = false;
};

template <typename T>
void sumFrames(Args a, T* sum)
{
    io::DenFileInfo di(a.input_den);
    uint32_t dimx = di.dimx();
    uint32_t dimy = di.dimy();
    uint32_t frameSize = dimx * dimy;
    std::fill(sum, sum + frameSize, T(0));
    std::shared_ptr<io::Frame2DReaderI<T>> denReader
        = std::make_shared<io::DenFrame2DReader<T>>(a.input_den);
    for(const int& k : a.frames)
    {
        std::shared_ptr<io::Frame2DI<T>> A = denReader->readFrame(k);
        for(int j = 0; j != dimy; j++)
        {
            for(int i = 0; i != dimx; i++)
            {
                sum[i + j * dimx] += A->get(i, j);
            }
        }
    }
}

template <typename T>
void averageFrames(Args a, T* avg)
{
    io::DenFileInfo di(a.input_den);
    uint32_t dimx = di.dimx();
    uint32_t dimy = di.dimy();
    uint32_t frameSize = dimx * dimy;
    uint32_t frameCount = a.frames.size();
    std::fill(avg, avg + frameSize, T(0));
    std::shared_ptr<io::Frame2DReaderI<T>> denReader
        = std::make_shared<io::DenFrame2DReader<T>>(a.input_den);
    for(const int& k : a.frames)
    {
        std::shared_ptr<io::Frame2DI<T>> A = denReader->readFrame(k);
        for(int j = 0; j != dimy; j++)
        {
            for(int i = 0; i != dimx; i++)
            {
                avg[i + j * dimx] += A->get(i, j);
            }
        }
    }
    for(int i = 0; i != frameSize; i++)
    {
        avg[i] = avg[i] / frameCount;
    }
}

// Utilizing two pass algorithm as discribed
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
template <typename T>
void framesVariance(Args a, T* var, bool sampleVariance = false)
{
    io::DenFileInfo di(a.input_den);
    uint32_t dimx = di.dimx();
    uint32_t dimy = di.dimy();
    uint32_t frameSize = dimx * dimy;
    uint32_t frameCount = a.frames.size();
    std::fill(var, var + frameSize, T(0));
    T* avg = new T[frameSize];
    averageFrames(a, avg);
    std::shared_ptr<io::Frame2DReaderI<T>> denReader
        = std::make_shared<io::DenFrame2DReader<T>>(a.input_den);
    uint32_t divideFactor = frameCount;
    if(sampleVariance)
    {
        divideFactor = frameCount - 1;
    }
    for(const int& k : a.frames)
    {
        std::shared_ptr<io::Frame2DI<T>> A = denReader->readFrame(k);
        for(int j = 0; j != dimy; j++)
        {
            for(int i = 0; i != dimx; i++)
            {
                T a = A->get(i, j) - avg[i + j * dimx];
                a = a * a;
                var[i + j * dimx] += a;
            }
        }
    }
    delete[] avg;
    for(int i = 0; i != frameSize; i++)
    {
        var[i] = var[i] / divideFactor;
    }
}

// Utilizing two pass algorithm as discribed
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
template <typename T>
void framesStandardDeviation(Args a, T* stdev, bool sampleStandardDeviation = false)
{
    io::DenFileInfo di(a.input_den);
    uint32_t dimx = di.dimx();
    uint32_t dimy = di.dimy();
    uint32_t frameSize = dimx * dimy;
    uint32_t frameCount = a.frames.size();
    std::fill(stdev, stdev + frameSize, T(0));
    T* avg = new T[frameSize];
    averageFrames(a, avg);
    std::shared_ptr<io::Frame2DReaderI<T>> denReader
        = std::make_shared<io::DenFrame2DReader<T>>(a.input_den);
    uint32_t divideFactor = frameCount;
    if(sampleStandardDeviation)
    {
        divideFactor = frameCount - 1;
    }
    for(const int& k : a.frames)
    {
        std::shared_ptr<io::Frame2DI<T>> A = denReader->readFrame(k);
        for(int j = 0; j != dimy; j++)
        {
            for(int i = 0; i != dimx; i++)
            {
                T a = A->get(i, j) - avg[i + j * dimx];
                a = a * a;
                stdev[i + j * dimx] += a;
            }
        }
    }
    delete[] avg;
    for(int i = 0; i != frameSize; i++)
    {
        stdev[i] = std::sqrt(stdev[i] / divideFactor);
    }
}

template <typename T>
std::shared_ptr<io::Frame2DI<T>> minFrames(Args a)
{
    io::DenFileInfo di(a.input_den);
    uint32_t dimx = di.dimx();
    uint32_t dimy = di.dimy();
    uint32_t frameCount = a.frames.size();
    std::shared_ptr<io::Frame2DReaderI<T>> denReader
        = std::make_shared<io::DenFrame2DReader<T>>(a.input_den);
    std::shared_ptr<io::Frame2DI<T>> F = denReader->readFrame(a.frames[0]);
    for(int k = 1; k < frameCount; k++)
    {
        std::shared_ptr<io::Frame2DI<T>> A = denReader->readFrame(a.frames[k]);
        for(int j = 0; j != dimy; j++)
        {
            for(int i = 0; i != dimx; i++)
            {
                F->set(std::min(A->get(i, j), F->get(i, j)), i, j);
            }
        }
    }
    return F;
}

template <typename T>
std::shared_ptr<io::Frame2DI<T>> maxFrames(Args a)
{
    io::DenFileInfo di(a.input_den);
    uint32_t dimx = di.dimx();
    uint32_t dimy = di.dimy();
    uint32_t frameCount = a.frames.size();
    std::shared_ptr<io::Frame2DReaderI<T>> denReader
        = std::make_shared<io::DenFrame2DReader<T>>(a.input_den);
    std::shared_ptr<io::Frame2DI<T>> F = denReader->readFrame(a.frames[0]);
    for(int k = 1; k < frameCount; k++)
    {
        std::shared_ptr<io::Frame2DI<T>> A = denReader->readFrame(a.frames[k]);
        for(int j = 0; j != dimy; j++)
        {
            for(int i = 0; i != dimx; i++)
            {
                F->set(std::max(A->get(i, j), F->get(i, j)), i, j);
            }
        }
    }
    return F;
}

template <typename T>
T getMedian(T* array, uint32_t count)
{
    std::sort(array, array + count);
    if(count % 2 == 0)
    {
        return (array[count / 2 - 1] + array[count / 2]) / 2;
    } else
    {
        return array[count / 2];
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
std::shared_ptr<io::Frame2DI<T>> medianFrames(Args a)
{
    io::DenFileInfo di(a.input_den);
    uint32_t dimx = di.dimx();
    uint32_t dimy = di.dimy();
    uint32_t frameCount = a.frames.size();
    std::shared_ptr<io::Frame2DReaderI<T>> denReader
        = std::make_shared<io::DenFrame2DReader<T>>(a.input_den);
    std::shared_ptr<io::Frame2DI<T>> F = std::make_shared<io::BufferedFrame2D<T>>(T(0), dimx, dimy);
    uint32_t frameSize = dimx * dimy;
    uint32_t maxArrayNum = 2147483647 / frameCount;
    uint32_t arraysCount = std::min(frameSize, maxArrayNum);
    T** rowArrays = new T*[arraysCount];
    for(int i = 0; i != arraysCount; i++)
    {
        rowArrays[i] = new T[frameCount];
    }
    for(int ind = 0; ind != 1 + ((frameSize-1) / arraysCount); ind++)
    {
        int maxjnd = std::min(arraysCount, frameSize - ind * arraysCount);
        for(int k = 0; k < frameCount; k++)
        {
            std::shared_ptr<io::Frame2DI<T>> A = denReader->readFrame(a.frames[k]);
            for(int jnd = 0; jnd < maxjnd; jnd++)
            {
                int flatindex = ind * arraysCount + jnd;
                int xindex = flatindex % dimx;
                int yindex = flatindex / dimx;
                rowArrays[jnd][k] = A->get(xindex, yindex);
            }
        }
        for(int jnd = 0; jnd < maxjnd; jnd++)
        {
            int flatindex = ind * arraysCount + jnd;
            int xindex = flatindex % dimx;
            int yindex = flatindex / dimx;
            T median = getMedian(rowArrays[jnd], frameCount);
            F->set(median, xindex, yindex);
        }
    }
    for(int i = 0; i != arraysCount; i++)
    {
        delete[] rowArrays[i];
    }
    delete[] rowArrays;
    return F;
}

template <typename T>
void processFiles(Args a)
{
    io::DenFileInfo di(a.input_den);
    uint32_t dimx = di.dimx();
    uint32_t dimy = di.dimy();
    uint32_t frameSize = dimx * dimy;
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> outputWritter
        = std::make_shared<io::DenAsyncFrame2DWritter<T>>(a.output_den, dimx, dimy, 1);
    if(a.sum)
    {
        T* sum = new T[frameSize];
        sumFrames(a, sum);
        std::unique_ptr<io::Frame2DI<T>> f
            = std::make_unique<io::FrameMemoryViewer2D<T>>(sum, dimx, dimy);
        outputWritter->writeFrame(*f, 0);
        delete[] sum;
    }

    if(a.avg)
    {
        T* avg = new T[frameSize];
        averageFrames(a, avg);
        std::unique_ptr<io::Frame2DI<T>> f
            = std::make_unique<io::FrameMemoryViewer2D<T>>(avg, dimx, dimy);
        outputWritter->writeFrame(*f, 0);
        delete[] avg;
    }

    if(a.variance)
    {
        T* var = new T[frameSize];
        framesVariance(a, var, false);
        std::unique_ptr<io::Frame2DI<T>> f
            = std::make_unique<io::FrameMemoryViewer2D<T>>(var, dimx, dimy);
        outputWritter->writeFrame(*f, 0);
        delete[] var;
    }

    if(a.sampleVariance)
    {
        T* var = new T[frameSize];
        framesVariance(a, var, true);
        std::unique_ptr<io::Frame2DI<T>> f
            = std::make_unique<io::FrameMemoryViewer2D<T>>(var, dimx, dimy);
        outputWritter->writeFrame(*f, 0);
        delete[] var;
    }

    if(a.standardDeviation)
    {
        T* stdev = new T[frameSize];
        framesStandardDeviation(a, stdev, false);
        std::unique_ptr<io::Frame2DI<T>> f
            = std::make_unique<io::FrameMemoryViewer2D<T>>(stdev, dimx, dimy);
        outputWritter->writeFrame(*f, 0);
        delete[] stdev;
    }

    if(a.sampleStandardDeviation)
    {
        T* stdev = new T[frameSize];
        framesStandardDeviation(a, stdev, true);
        std::unique_ptr<io::Frame2DI<T>> f
            = std::make_unique<io::FrameMemoryViewer2D<T>>(stdev, dimx, dimy);
        outputWritter->writeFrame(*f, 0);
        delete[] stdev;
    }

    if(a.max)
    {
        std::shared_ptr<io::Frame2DI<T>> f = maxFrames<T>(a);
        outputWritter->writeFrame(*f, 0);
    }

    if(a.min)
    {
        std::shared_ptr<io::Frame2DI<T>> f = minFrames<T>(a);
        outputWritter->writeFrame(*f, 0);
    }

    if(a.median)
    {
        std::shared_ptr<io::Frame2DI<T>> f = medianFrames<T>(a);
        outputWritter->writeFrame(*f, 0);
    }

    // given angle attenuation is maximal
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
    io::DenSupportedType dataType = di.getDataType();
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
    PRG.endLog();
}

void Args::defineArguments()
{
    cliApp->add_option("input_den", input_den, "Input file.")->check(CLI::ExistingFile)->required();
    cliApp->add_option("output_den", output_den, "Output file.")->required();
    addForceArgs();
    addFramespecArgs();
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
