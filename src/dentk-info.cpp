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
#include "CLI/CLI.hpp"

// Internal libraries
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "DEN/DenSupportedType.hpp"
#include "Frame2DI.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/Arguments.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/Program.hpp"
#include "PROG/parseArgs.h"
#include "frameop.h"

using namespace KCT;
using namespace KCT::util;

// class declarations
struct Args : public ArgumentsFramespec
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsFramespec(argc, argv, prgName){};
    std::string input_file;
    bool returnDimensions = false;
    bool l2norm = false;
};

template <typename T>
void printFrameStatistics(const io::Frame2DI<T>& f, uint32_t index)
{
    double min = (double)io::minFrameValue<T>(f);
    double max = (double)io::maxFrameValue<T>(f);
    double avg = io::meanFrameValue<T>(f);
    double l2norm = io::normFrame<T>(f, 2);
    std::cout << io::xprintf("\tMinimum, maximum, average values: %.3f, %0.3f, %0.3f.\n", min, max,
                             avg);
    double RMSEDenominator = std::sqrt(f.dimx() * f.dimy());
    std::cout << io::xprintf("\t||frame %d||_2, RMSE: %E, %E\n", index, l2norm,
                             l2norm / RMSEDenominator);
    int nonFiniteCount = io::sumNonfiniteValues<T>(f);
    if(nonFiniteCount == 0)
    {
        std::cout << io::xprintf("\tNo NAN or not finite number.\n\n");
    } else
    {
        std::cout << io::xprintf("\tThere is %d non finite numbers. \tFrom that %d NAN.\n\n",
                                 io::sumNonfiniteValues<T>(f), io::sumNanValues<T>(f));
    }
}

template <typename T>
void printBasicStatistics(const io::DenFileInfo& di, const Args& ARG)
{

    double min = di.getMinVal<T>();
    double max = di.getMaxVal<T>();
    double mean = di.getMean<T>();
    double variance = di.getVariance<T>();
    if(di.getDataType() == io::DenSupportedType::UINT16)
    {
        std::cout << io::xprintf(
            "Global minimum and maximum values are (%d, %d), mean=%f, stdev=%f.\n", (int)min,
            (int)max, mean, std::pow(variance, 0.5));
    } else
    {
        std::cout << io::xprintf(
            "Global minimum and maximum values are (%0.3f, %0.3f), mean=%f, stdev=%f.\n", min, max,
            mean, std::pow(variance, 0.5));
    }
    if(ARG.l2norm)
    {
        io::DenFileInfo di(ARG.input_file);
        uint64_t dimx = di.dimx();
        uint64_t dimy = di.dimy();
        uint64_t dimz = di.dimz();
        double RMSEDenominator = std::sqrt(double(dimx * dimy * dimz));
        double l2 = di.getl2Square<T>();
        std::cout << io::xprintf(
            "Global l2 norm is %0.1f and its square is %0.1f. That imply L2=%f, RMSE=%E\n",
            std::pow(l2, 0.5), l2, std::pow(l2, 0.5), std::pow(l2, 0.5) / RMSEDenominator);
    }
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    const std::string prgInfo = "Information about DEN file and its individual slices.";
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
    int dimx = di.getNumCols();
    int dimy = di.getNumRows();
    int dimz = di.getNumSlices();
    if(ARG.returnDimensions)
    {
        std::cout << io::xprintf("%d\t%d\t%d\n", dimx, dimy, dimz);
        return 0;
    }
    // int elementSize = di.elementByteSize();
    io::DenSupportedType t = di.getDataType();
    std::string elm = io::DenSupportedTypeToString(t);
    std::cout << io::xprintf(
        "The file %s of type %s has dimensions (x,y,z)=(cols,rows,slices)=(%d, "
        "%d, %d), each cell has x*y=%d pixels.\n",
        ARG.input_file.c_str(), elm.c_str(), dimx, dimy, dimz, dimx * dimy);
    switch(t)
    {
    case io::DenSupportedType::UINT16: {
        printBasicStatistics<uint16_t>(di, ARG);
        break;
    }
    case io::DenSupportedType::FLOAT32: {
        printBasicStatistics<float>(di, ARG);
        break;
    }
    case io::DenSupportedType::FLOAT64: {
        printBasicStatistics<double>(di, ARG);
        break;
    }
    default:
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(t).c_str());
        KCTERR(errMsg);
    }
    if(ARG.framesSpecified)
    {
        double l2 = 0.0;
        double val;
        switch(t)
        {
        case io::DenSupportedType::UINT16: {
            std::shared_ptr<io::Frame2DReaderI<uint16_t>> denSliceReader
                = std::make_shared<io::DenFrame2DReader<uint16_t>>(ARG.input_file);
            for(const int f : ARG.frames)
            {
                std::cout << io::xprintf("Statistic of %d-th frame:\n", f);
                std::shared_ptr<io::Frame2DI<uint16_t>> framePtr = denSliceReader->readFrame(f);
                printFrameStatistics<uint16_t>(*framePtr, f);
                if(ARG.l2norm)
                {
                    val = io::l2square<uint16_t>(*framePtr);
                    l2 += val;
                }
            }
            if(ARG.l2norm)
            {
                std::cout << io::xprintf(
                    "Square of l2 norm of frames is %0.3f and l2 norm is %0.3f.\n", l2,
                    std::pow(l2, 0.5));
            }
            break;
        }
        case io::DenSupportedType::FLOAT32: {

            std::shared_ptr<io::Frame2DReaderI<float>> denSliceReader
                = std::make_shared<io::DenFrame2DReader<float>>(ARG.input_file);
            for(const int f : ARG.frames)
            {
                std::cout << io::xprintf("Statistic of %d-th frame:\n", f);
                std::shared_ptr<io::Frame2DI<float>> framePtr = denSliceReader->readFrame(f);
                printFrameStatistics<float>(*framePtr, f);
                if(ARG.l2norm)
                {
                    val = io::l2square<float>(*framePtr);
                    l2 += val;
                }
            }
            if(ARG.l2norm)
            {
                std::cout << io::xprintf(
                    "Square of l2 norm of frames is %0.3f and l2 norm is %0.3f.\n", l2,
                    std::pow(l2, 0.5));
            }
            break;
        }
        case io::DenSupportedType::FLOAT64: {

            std::shared_ptr<io::Frame2DReaderI<double>> denSliceReader
                = std::make_shared<io::DenFrame2DReader<double>>(ARG.input_file);
            for(const int f : ARG.frames)
            {
                std::cout << io::xprintf("Statistic of %d-th frame:\n", f);
                std::shared_ptr<io::Frame2DI<double>> framePtr = denSliceReader->readFrame(f);
                printFrameStatistics<double>(*framePtr, f);
                if(ARG.l2norm)
                {
                    val = io::l2square<double>(*framePtr);
                    l2 += val;
                }
            }
            if(ARG.l2norm)
            {
                std::cout << io::xprintf(
                    "Square of l2 norm of frames is %0.3f and l2 norm is %0.3f.\n", l2,
                    std::pow(l2, 0.5));
            }
            break;
        }
        default:
            std::string errMsg = io::xprintf("Frame statistic for %s is unsupported.",
                                             io::DenSupportedTypeToString(t).c_str());
            KCTERR(errMsg);
        }
    }
}

void Args::defineArguments()
{
    cliApp->add_option("input_den_file", input_file, "File in a DEN format to process.")
        ->required()
        ->check(CLI::ExistingFile);
    addFramespecArgs();
    cliApp->add_flag("--l2norm", l2norm, "Print l2 norm of the frame specs.");
    cliApp->add_flag("--dim", returnDimensions,
                     "Return only the dimensions in a format x\\ty\\tz\\n and quit.");
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
    fillFramesVector(inf.dimz());
    return 0;
}
