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
#include "ftpl.h"

// Internal libraries
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DBufferedWritter.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DCachedReader.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/ArgumentsVerbose.hpp"
#include "PROG/Program.hpp"
#include "PROG/parseArgs.h"
#include "frameop.h"

using namespace KCT;
using namespace KCT::util;

// class declarations
class Args : public ArgumentsForce,
             public ArgumentsVerbose,
             public ArgumentsFramespec,
             public ArgumentsThreading
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
        , ArgumentsThreading(argc, argv, prgName){};
    std::string input = "";
    std::string output = "";
    uint32_t dimx, dimy, dimz;
    uint64_t frameSize;
    bool zeroAtQuantile = false;
    double zeroQuantile = std::numeric_limits<double>::quiet_NaN(); // From 0.0 to 1.0
    bool meanScaleFrame = false;
    bool meanShiftFrame = false;
    double meanAfterNormalization = std::numeric_limits<double>::quiet_NaN();
};

void Args::defineArguments()
{
    cliApp->add_option("input_op1", input, "Input DEN file.")->required()->check(CLI::ExistingFile);
    cliApp->add_option("output", output, "Output DEN file.")->required();
    // Adding radio group see https://github.com/CLIUtils/CLI11/pull/234
    CLI::Option* optZeroQuantile
        = cliApp->add_option("--zero-quantile", zeroQuantile,
                             "Specify the quantile at which will be 0 in each "
                             "frame, must be in the range [0,1].");
    CLI::Option* optMeanAfter
        = cliApp
              ->add_option(
                  "--mean-after-normalization", meanAfterNormalization,
                  "Specify the mean of each frame after normalization. When the mean of the "
                  "frame before the normalization is zero, then --mean-frame-scala can not be "
                  "performed and the frame mean will stay the same.")
              ->check(CLI::Range(0.0, 1.0));
    CLI::Option_group* op_clg = cliApp->add_option_group("Operation", "Normalization to perform.");
    CLI::Option* flagZeroQuantile
        = op_clg->add_flag("--zero-at-quantile", zeroAtQuantile,
                           "Shift the data so that the zero is at certain quantile");
    op_clg->add_flag("--mean-frame-shift", meanShiftFrame,
                     "Adjust the frame mean to a particular value by scaling all the frames, by "
                     "default the mean will be adjusted to the mean of means of frames, if "
                     "specified --mean-after-normalization will be used.");
    op_clg->add_flag("--mean-frame-scale", meanScaleFrame,
                     "Adjust the frame mean to a particular value by shifting values in all the "
                     "frames, by default the mean will be adjusted to the mean of means of frames, "
                     "if specified --mean-after-normalization will be used.");
    op_clg->require_option(1);
    flagZeroQuantile->needs(optZeroQuantile);
    optZeroQuantile->needs(flagZeroQuantile);
    flagZeroQuantile->excludes(optMeanAfter);
    addForceArgs();
    addVerboseArgs();
    addFramespecArgs();
    addThreadingArgs();
}

int Args::postParse()
{
    int e = handleFileExistence(output, force, force);
    if(e != 0)
    {
        return e;
    }
    io::DenFileInfo inf(input);
    dimx = inf.dimx();
    dimy = inf.dimy();
    dimz = inf.dimz();
    frameSize = (uint64_t)dimx * (uint64_t)dimy;
    fillFramesVector(dimz);
    return 0;
}

template <typename T>
std::shared_ptr<io::BufferedFrame2D<T>> quantileShiftedBuffframe(
    std::shared_ptr<io::BufferedFrame2D<T>> f_in, uint32_t k, double zeroAtQuantile)
{
    if(std::isnan(zeroAtQuantile))
    {
        KCTERR("Pos is NAN");
    } else if(!std::isfinite(zeroAtQuantile))
    {
        KCTERR("Pos is INF");
    } else if(zeroAtQuantile < 0.0 || zeroAtQuantile > 1.0)
    {
        std::string ERR = io::xprintf("Pos=%f is not in the range[0,1].", zeroAtQuantile);
        KCTERR(ERR);
    }
    uint64_t frameSize = f_in->getFrameSize();
    if(frameSize == 0)
    {
        KCTERR("Can not compute quantile on empty frame!");
    }
    std::shared_ptr<io::BufferedFrame2D<T>> f_out = std::make_shared<io::BufferedFrame2D<T>>(
        *f_in); // Construct copy not to destroy original array
    T* in_array = f_in->getDataPointer();
    T* out_array = f_out->getDataPointer();
    std::sort(out_array, out_array + frameSize);
    uint64_t quantileIndex = (uint64_t)(zeroAtQuantile * frameSize - 1);
    T quantile = out_array[quantileIndex];
    LOGD << io::xprintf("k=%d %f quantile is %f", k, zeroAtQuantile, quantile);
    std::transform(in_array, in_array + frameSize, out_array,
                   [quantile](const T& x) { return x - quantile; });
    return f_out;
}

template <typename T>
std::shared_ptr<io::BufferedFrame2D<T>>
shiftedBuffframe(std::shared_ptr<io::BufferedFrame2D<T>> f_in, double shift)
{
    if(std::isnan(shift))
    {
        KCTERR("Pos is NAN");
    } else if(!std::isfinite(shift))
    {
        KCTERR("Pos is INF");
    }
    uint64_t frameSize = f_in->getFrameSize();
    if(frameSize == 0)
    {
        KCTERR("Can not compute quantile on empty frame!");
    }
    std::shared_ptr<io::BufferedFrame2D<T>> f_out = std::make_shared<io::BufferedFrame2D<T>>(
        T(0), f_in->dimx(), f_in->dimy()); // Construct copy not to destroy original array
    T* in_array = f_in->getDataPointer();
    T* out_array = f_out->getDataPointer();
    std::transform(in_array, in_array + frameSize, out_array,
                   [shift](const T& x) { return T(x + shift); });
    return f_out;
}

template <typename T>
std::shared_ptr<io::BufferedFrame2D<T>>
scaledBuffframe(std::shared_ptr<io::BufferedFrame2D<T>> f_in, double factor)
{
    if(std::isnan(factor))
    {
        KCTERR("Pos is NAN");
    } else if(!std::isfinite(factor))
    {
        KCTERR("Pos is INF");
    }
    uint64_t frameSize = f_in->getFrameSize();
    if(frameSize == 0)
    {
        KCTERR("Can not compute quantile on empty frame!");
    }
    std::shared_ptr<io::BufferedFrame2D<T>> f_out = std::make_shared<io::BufferedFrame2D<T>>(
        T(0), f_in->dimx(), f_in->dimy()); // Construct copy not to destroy original array
    T* in_array = f_in->getDataPointer();
    T* out_array = f_out->getDataPointer();
    std::transform(in_array, in_array + frameSize, out_array,
                   [factor](const T& x) { return T(x * factor); });
    return f_out;
}

template <typename T>
void quantileShiftFrame(int _FTPLID,
                        double zeroQuantile,
                        uint32_t k_in,
                        uint32_t k_out,
                        std::shared_ptr<io::DenFrame2DCachedReader<T>>& inReader,
                        std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>> outputWritter)
{
    std::shared_ptr<io::BufferedFrame2D<T>> f_in = inReader->readBufferedFrame(k_in);
    std::shared_ptr<io::BufferedFrame2D<T>> f_out
        = quantileShiftedBuffframe(f_in, k_in, zeroQuantile);
    outputWritter->writeBufferedFrame(*f_out, k_out);
}
template <typename T>
void shiftFrame(int _FTPLID,
                double targetMean,
                double currentMean,
                uint32_t k_in,
                uint32_t k_out,
                std::shared_ptr<io::DenFrame2DCachedReader<T>>& inReader,
                std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>> outputWritter)
{
    std::shared_ptr<io::BufferedFrame2D<T>> f_in = inReader->readBufferedFrame(k_in);
    std::shared_ptr<io::BufferedFrame2D<T>> f_out
        = shiftedBuffframe(f_in, targetMean - currentMean);
    outputWritter->writeBufferedFrame(*f_out, k_out);
}

template <typename T>
void scaleFrame(int _FTPLID,
                double targetMean,
                double currentMean,
                uint32_t k_in,
                uint32_t k_out,
                std::shared_ptr<io::DenFrame2DCachedReader<T>>& inReader,
                std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>> outputWritter)
{
    std::shared_ptr<io::BufferedFrame2D<T>> f_in = inReader->readBufferedFrame(k_in);
    std::shared_ptr<io::BufferedFrame2D<T>> f_out = scaledBuffframe(f_in, targetMean / currentMean);
    outputWritter->writeBufferedFrame(*f_out, k_out);
}

template <typename T>
void sumFrame(int _FTPLID,
              uint32_t k,
              std::shared_ptr<io::DenFrame2DCachedReader<T>>& inReader,
              std::vector<double>* sumVector)
{
    (*sumVector)[k] = bufferedFrameSum(inReader->readBufferedFrame(k));
}

template <typename T>
void processFiles(Args ARG)
{
    ftpl::thread_pool* threadpool = nullptr;
    if(ARG.threads > 0)
    {
        threadpool = new ftpl::thread_pool(ARG.threads);
    }
    std::shared_ptr<io::DenFrame2DCachedReader<T>> fReader
        = std::make_shared<io::DenFrame2DCachedReader<T>>(ARG.input, ARG.threads);
    std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>> outputWritter
        = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(ARG.output, ARG.dimx, ARG.dimy,
                                                                  ARG.frames.size());

    const int dummy_FTPLID = 0;
    uint32_t k_in, k_out;
    if(ARG.zeroAtQuantile)
    {
        for(uint32_t IND = 0; IND != ARG.frames.size(); IND++)
        {
            k_in = ARG.frames[IND];
            k_out = IND;
            if(threadpool)
            {
                threadpool->push(quantileShiftFrame<T>, ARG.zeroQuantile, k_in, k_out, fReader,
                                 outputWritter);
            } else
            {
                quantileShiftFrame<T>(dummy_FTPLID, ARG.zeroQuantile, k_in, k_out, fReader,
                                      outputWritter);
            }
        }
    } else
    {
        std::vector<double> sumVector;
        sumVector.resize(ARG.dimz);
        for(uint32_t IND = 0; IND != ARG.frames.size(); IND++)
        {
            k_in = ARG.frames[IND];
            k_out = IND;
            if(threadpool)
            {
                threadpool->push(sumFrame<T>, k_in, fReader, &sumVector);
            } else
            {
                sumFrame<T>(dummy_FTPLID, k_in, fReader, &sumVector);
            }
        }
        if(threadpool != nullptr)
        {
            threadpool->stop(true);
            threadpool->init();
            threadpool->resize(ARG.threads);
        }
        if(std::isnan(ARG.meanAfterNormalization))
        {
            // Vector of means that are in the frames vector to easily compute its mean
            std::vector<double> validMeans;
            uint32_t k;
            for(uint32_t IND = 0; IND != ARG.frames.size(); IND++)
            {
                k = ARG.frames[IND];
                validMeans.emplace_back(sumVector[k] / ARG.frameSize);
            }
            ARG.meanAfterNormalization
                = std::reduce(validMeans.begin(), validMeans.end()) / validMeans.size();
        }
        if(ARG.meanShiftFrame)
        {
            for(uint32_t IND = 0; IND != ARG.frames.size(); IND++)
            {
                k_in = ARG.frames[IND];
                k_out = IND;
                if(threadpool)
                {
                    // threadpool->push(shiftFrame<T>, ARG.meanAfterNormalization,
                    //               sumVector[k_in] / ARG.frameSize, k_in, k_out, fReader,
                    //             outputWritter);
                } else
                {
                    shiftFrame<T>(dummy_FTPLID, ARG.meanAfterNormalization,
                                  sumVector[k_in] / ARG.frameSize, k_in, k_out, fReader,
                                  outputWritter);
                }
            }
        }
        if(ARG.meanScaleFrame)
        {
            for(uint32_t IND = 0; IND != ARG.frames.size(); IND++)
            {
                k_in = ARG.frames[IND];
                k_out = IND;
                if(threadpool)
                {
                    threadpool->push(scaleFrame<T>, ARG.meanAfterNormalization,
                                     sumVector[k_in] / ARG.frameSize, k_in, k_out, fReader,
                                     outputWritter);
                } else
                {
                    scaleFrame<T>(dummy_FTPLID, ARG.meanAfterNormalization,
                                  sumVector[k_in] / ARG.frameSize, k_in, k_out, fReader,
                                  outputWritter);
                }
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
    const std::string prgInfo = "Frame normalization.";
    Args ARG(argc, argv, prgInfo);
    int parseResult = ARG.parse(false);
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    // After init parsing arguments
    io::DenFileInfo di(ARG.input);
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

