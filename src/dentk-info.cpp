// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctype.h>
#include <iostream>
#include <numeric>
#include <regex>
#include <string>

// External libraries
#include "CLI/CLI.hpp"

// Internal libraries
#include "AsyncFrame2DWritterI.hpp"
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFrame2DCachedReader.hpp"
#include "DEN/DenSupportedType.hpp"
#include "Frame2DI.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/Arguments.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/Program.hpp"
#include "PROG/parseArgs.h"
#include "frameop.h"
#include "ftpl.h"

using namespace KCT;
using namespace KCT::util;

// class declarations
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
    std::string input_file;
    uint32_t dimx, dimy, dimz;
    uint64_t frameSize;
    bool returnDimensions = false;
    bool l2norm = false;
};

void Args::defineArguments()
{
    cliApp->add_option("input_den_file", input_file, "File in a DEN format to process.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_flag("--l2norm", l2norm, "Print l2 norm of the frame specs.");
    cliApp->add_flag("--dim", returnDimensions,
                     "Return only the dimensions in a format x\\ty\\tz\\n and quit.");
    addFramespecArgs();
    threads = 32;
    addThreadingArgs();
}

int Args::postParse()
{
    cliApp->parse(argc, argv);
    if(returnDimensions)
    {
        return 0; // Do not process frames and print a log message.
    }
    if(cliApp->count("--frames") > 0)
    {
        framesSpecified = true;
    }
    io::DenFileInfo inf(input_file);
    dimx = inf.dimx();
    dimy = inf.dimy();
    dimz = inf.dimz();
    frameSize = inf.getFrameSize();
    fillFramesVector(inf.dimz());
    return 0;
}

template <typename T>
void processFrame2(int _FTPLID,
                   uint32_t k,
                   std::shared_ptr<io::DenFrame2DCachedReader<T>> inputReader,
                   double shift,
                   std::vector<io::onepassData<T>>* frameDataVector)
{
    std::shared_ptr<io::BufferedFrame2D<T>> f = inputReader->readBufferedFrame(k);
    (*frameDataVector)[k] = onepassBuffframeInfo(f, shift);
}

template <typename T>
void processFrame(int _FTPLID,
                  uint32_t k,
                  std::shared_ptr<io::DenFrame2DCachedReader<T>> inputReader,
                  double shift,
                  T* min,
                  T* max,
                  double* sum,
                  double* sumSquares,
                  double* shiftedSum,
                  double* shiftedSumSquares,
                  uint64_t* NANcount,
                  uint64_t* INFcount)
{
    LOGD << io::xprintf("Processing frame %d", k);
    std::shared_ptr<io::BufferedFrame2D<T>> f = inputReader->readBufferedFrame(k);
    LOGD << io::xprintf("Read frame %d", k);
    onepassBuffframeInfo(f, shift, min, max, sum, sumSquares, shiftedSum, shiftedSumSquares,
                         NANcount, INFcount);
    LOGD << io::xprintf("Processed frame %d", k);
}

template <typename T>
void processFile(Args ARG, io::DenFileInfo di)
{
    io::DenSupportedType denDataType = di.getElementType();
    ftpl::thread_pool* threadpool = nullptr;
    if(ARG.threads > 0)
    {
        threadpool = new ftpl::thread_pool(ARG.threads);
    }
    std::shared_ptr<io::DenFrame2DCachedReader<T>> fReader
        = std::make_shared<io::DenFrame2DCachedReader<T>>(ARG.input_file, ARG.threads);
    const int dummy_FTPLID = 0;
    // As a heuristic we compute shift as a mean of the first frame
    std::shared_ptr<io::BufferedFrame2D<T>> f0 = fReader->readBufferedFrame(0);
    double shift = meanFrameValue(*f0);
    std::vector<io::onepassData<T>> frameDataVector;
    frameDataVector.resize(ARG.dimz);

    std::vector<T> minVector;
    std::vector<T> maxVector;
    std::vector<double> sumVector;
    std::vector<double> sumSquaresVector;
    std::vector<double> shiftedSumVector;
    std::vector<double> shiftedSumSquaresVector;
    std::vector<uint64_t> NANcountVector;
    std::vector<uint64_t> INFcountVector;

    for(uint32_t k = 0; k != ARG.dimz; k++)
    {
        if(threadpool)
        {
            threadpool->push(processFrame2<T>, k, fReader, shift, &frameDataVector);
        } else
        {
            processFrame2<T>(dummy_FTPLID, k, fReader, shift, &frameDataVector);
        }
    }
    if(threadpool != nullptr)
    {
        threadpool->stop(true);
        delete threadpool;
    }
    for(uint32_t k = 0; k != ARG.dimz; k++)
    {
        minVector.emplace_back(frameDataVector[k].min);
        maxVector.emplace_back(frameDataVector[k].max);
        sumVector.emplace_back(frameDataVector[k].sum);
        sumSquaresVector.emplace_back(frameDataVector[k].sumSquares);
        shiftedSumVector.emplace_back(frameDataVector[k].shiftedSum);
        shiftedSumSquaresVector.emplace_back(frameDataVector[k].shiftedSumSquares);
        NANcountVector.emplace_back(frameDataVector[k].NANcount);
        INFcountVector.emplace_back(frameDataVector[k].INFcount);
    }

    T min = *std::min_element(minVector.begin(), minVector.end());
    T max = *std::max_element(maxVector.begin(), maxVector.end());
    uint64_t NANcount = std::accumulate(NANcountVector.begin(), NANcountVector.end(), T(0));
    uint64_t INFcount = std::accumulate(INFcountVector.begin(), INFcountVector.end(), T(0));
    uint64_t N = ARG.frameSize * ARG.dimz - NANcount - INFcount;
    double sum = std::accumulate(sumVector.begin(), sumVector.end(), 0.0);
    double mean = sum / N;
    double sumSquares = std::accumulate(sumSquaresVector.begin(), sumSquaresVector.end(), 0.0);
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    double shiftedSum = std::accumulate(shiftedSumVector.begin(), shiftedSumVector.end(), 0.0);
    double shiftedSumSquares
        = std::accumulate(shiftedSumSquaresVector.begin(), shiftedSumSquaresVector.end(), 0.0);
    double variance = (shiftedSumSquares - (shiftedSum * shiftedSum) / N) / N;

    if(denDataType == io::DenSupportedType::UINT16)
    {
        std::cout << io::xprintf("Global [min, max] = [%d, %d], mean=%f, stdev=%f.\n", (int)min,
                                 (int)max, mean, std::pow(variance, 0.5));
    } else
    {
        std::cout << io::xprintf("Global [min, max] = [%0.3f, %0.3f], mean=%f, stdev=%f, ", min,
                                 max, mean, std::pow(variance, 0.5));
        if(NANcount != 0 || INFcount != 0)
        {

            std::cout << io::xprintf_red("%d NAN and %d INF values.\n", NANcount, INFcount);
        } else
        {
            std::cout << io::xprintf("no NAN or INF values.\n");
        }
    }
    if(ARG.l2norm)
    {
        double RMSEDenominator = std::sqrt(double(N));
        double l2 = sumSquares;
        std::cout << io::xprintf(
            "Global l2 norm is %0.1f and its square is %0.1f. That imply L2=%f, RMSE=%E\n",
            std::pow(l2, 0.5), l2, std::pow(l2, 0.5), std::pow(l2, 0.5) / RMSEDenominator);
    }

    if(ARG.framesSpecified)
    {
        double l2 = 0.0;
        for(uint32_t IND = 0; IND != ARG.frames.size(); IND++)
        {
            uint32_t k = ARG.frames[IND];
            T min = minVector[k];
            T max = maxVector[k];
            uint64_t NANcount = NANcountVector[k];
            uint64_t INFcount = INFcountVector[k];
            uint64_t N = ARG.frameSize - NANcount - INFcount;
            double sum = sumVector[k];
            double mean = sum / N;
            double sumSquares = sumSquaresVector[k];
            // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            double shiftedSum = shiftedSumVector[k];
            double shiftedSumSquares = shiftedSumSquaresVector[k];
            double variance = (shiftedSumSquares - (shiftedSum * shiftedSum) / N) / N;

            if(denDataType == io::DenSupportedType::UINT16)
            {
                std::cout << io::xprintf(
                    "||frame %d||_2=%E RMSE=%E, [min, max] = [%d, %d], mean=%f, stdev=%f.\n", k,
                    std::pow(sumSquares, 0.5), std::pow(sumSquares / N, 0.5), (int)min, (int)max,
                    mean, std::pow(variance, 0.5));
            } else
            {
                std::cout << io::xprintf(
                    "||frame %d||_2=%E RMSE=%E, [min, max] = [%0.3f, %0.3f], mean=%f, stdev=%f, ", k,
                    std::pow(sumSquares, 0.5), std::pow(sumSquares / N, 0.5), min, max, mean,
                    std::pow(variance, 0.5));
                if(NANcount != 0 || INFcount != 0)
                {

                    std::cout << io::xprintf_red("%d NAN and %d INF values.\n", NANcount, INFcount);
                } else
                {
                    std::cout << io::xprintf("no NAN or INF values.\n");
                }
            }
            l2 += sumSquaresVector[k];
        }
        if(ARG.l2norm)
        {
            std::cout << io::xprintf("Square of l2 norm of frames is %0.3f and l2 norm is %0.3f.\n",
                                     l2, std::pow(l2, 0.5));
        }
    }
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    const std::string prgInfo = "Information about DEN file and its individual frames.";
    Args ARG(argc, argv, prgInfo);
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    io::DenFileInfo di(ARG.input_file);
    uint32_t dimx = di.dimx();
    uint32_t dimy = di.dimy();
    uint32_t dimz = di.dimz();
    uint32_t dimCount = di.getDimCount();
    bool isExtended = di.isExtended();
    bool isXMajor = di.hasXMajorAlignment();
    if(ARG.returnDimensions)
    {
        std::cout << io::xprintf("%d\t%d\t%d\n", dimx, dimy, dimz);
        return 0;
    }
    // int elementSize = di.elementByteSize();
    io::DenSupportedType t = di.getElementType();
    std::string elm = io::DenSupportedTypeToString(t);
    std::cout << io::xprintf("The %s is %dD %s DEN %s file of dimensions (x,y,z)=(%d, "
                             "%d, %d). Frames of x*y=%d elements are %s.\n",
                             ARG.input_file.c_str(), dimCount, (isExtended ? "extended" : "legacy"),
                             elm.c_str(), dimx, dimy, dimz, dimx * dimy,
                             (isXMajor ? "xmajor" : "ymajor"));
    switch(t)
    {
    case io::DenSupportedType::UINT16: {
        processFile<uint16_t>(ARG, di);
        break;
    }
    case io::DenSupportedType::FLOAT32: {
        processFile<float>(ARG, di);
        break;
    }
    case io::DenSupportedType::FLOAT64: {
        processFile<double>(ARG, di);
        break;
    }
    default:
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(t).c_str());
        KCTERR(errMsg);
    }
}

