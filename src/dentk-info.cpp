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
#include "ARGPARSE/parseArgs.h"
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "DEN/DenSupportedType.hpp"
#include "Frame2DI.hpp"
#include "Frame2DReaderI.hpp"
#include "frameop.h"

using namespace CTL;

// class declarations
struct Args
{
    int parseArguments(int argc, char* argv[]);
    std::string input_file;
    std::string frameSpecs = "";
    std::vector<int> frames;
    bool framesSpecified = false;
    bool returnDimensions = false;
    bool l2norm = false;
};

template <typename T>
void printFrameStatistics(const io::Frame2DI<T>& f)
{
    double min = (double)io::minFrameValue<T>(f);
    double max = (double)io::maxFrameValue<T>(f);
    double avg = io::meanFrameValue<T>(f);
    double l2norm = io::normFrame<T>(f, 2);
    std::cout << io::xprintf("\tMinimum, maximum, average values: %.3f, %0.3f, %0.3f.\n", min, max,
                             avg);
    std::cout << io::xprintf("\tEuclidean 2-norm of the frame: %E.\n", l2norm);
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
void printBasicStatistics(const io::DenFileInfo& di, const Args& a)
{

    double min = di.getMinVal<T>();
    double max = di.getMaxVal<T>();
    if(di.getDataType() == io::DenSupportedType::uint16_t_)
    {
        std::cout << io::xprintf("Global minimum and maximum values are (%d, %d).\n", (int)min,
                                 (int)max);
    } else
    {
        std::cout << io::xprintf("Global minimum and maximum values are (%0.3f, %0.3f).\n", min,
                                 max);
    }
    if(a.l2norm)
    {
        double l2 = di.getl2Square<T>();
        std::cout << io::xprintf("Global l2 norm is %0.1f and its square is %0.1f.\n",
                                 std::pow(l2, 0.5), l2);
    }
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
    io::DenFileInfo di(a.input_file);
    int dimx = di.getNumCols();
    int dimy = di.getNumRows();
    int dimz = di.getNumSlices();
    if(a.returnDimensions)
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
        a.input_file.c_str(), elm.c_str(), dimx, dimy, dimz, dimx * dimy);
    switch(t)
    {
    case io::DenSupportedType::uint16_t_:
    {
        printBasicStatistics<uint16_t>(di, a);
        break;
    }
    case io::DenSupportedType::float_:
    {
        printBasicStatistics<float>(di, a);
        break;
    }
    case io::DenSupportedType::double_:
    {
        printBasicStatistics<double>(di, a);
        break;
    }
    default:
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(t));
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
    }
    if(a.framesSpecified)
    {
        double l2 = 0.0;
        double val;
        switch(t)
        {
        case io::DenSupportedType::uint16_t_:
        {
            std::shared_ptr<io::Frame2DReaderI<uint16_t>> denSliceReader
                = std::make_shared<io::DenFrame2DReader<uint16_t>>(a.input_file);
            for(const int f : a.frames)
            {
                std::cout << io::xprintf("Statistic of %d-th frame:\n", f);
                std::shared_ptr<io::Frame2DI<uint16_t>> framePtr = denSliceReader->readFrame(f);
                printFrameStatistics<uint16_t>(*framePtr);
                if(a.l2norm)
                {
                    val = io::l2square<uint16_t>(*framePtr);
                    l2 += val;
                }
            }
            if(a.l2norm)
            {
                std::cout << io::xprintf(
                    "Square of l2 norm of frames is %0.3f and l2 norm is %0.3f.\n", l2,
                    std::pow(l2, 0.5));
            }
            break;
        }
        case io::DenSupportedType::float_:
        {

            std::shared_ptr<io::Frame2DReaderI<float>> denSliceReader
                = std::make_shared<io::DenFrame2DReader<float>>(a.input_file);
            for(const int f : a.frames)
            {
                std::cout << io::xprintf("Statistic of %d-th frame:\n", f);
                std::shared_ptr<io::Frame2DI<float>> framePtr = denSliceReader->readFrame(f);
                printFrameStatistics<float>(*framePtr);
                if(a.l2norm)
                {
                    val = io::l2square<float>(*framePtr);
                    l2 += val;
                }
            }
            if(a.l2norm)
            {
                std::cout << io::xprintf(
                    "Square of l2 norm of frames is %0.3f and l2 norm is %0.3f.\n", l2,
                    std::pow(l2, 0.5));
            }
            break;
        }
        case io::DenSupportedType::double_:
        {

            std::shared_ptr<io::Frame2DReaderI<double>> denSliceReader
                = std::make_shared<io::DenFrame2DReader<double>>(a.input_file);
            for(const int f : a.frames)
            {
                std::cout << io::xprintf("Statistic of %d-th frame:\n", f);
                std::shared_ptr<io::Frame2DI<double>> framePtr = denSliceReader->readFrame(f);
                printFrameStatistics<double>(*framePtr);
                if(a.l2norm)
                {
                    val = io::l2square<double>(*framePtr);
                    l2 += val;
                }
            }
            if(a.l2norm)
            {
                std::cout << io::xprintf(
                    "Square of l2 norm of frames is %0.3f and l2 norm is %0.3f.\n", l2,
                    std::pow(l2, 0.5));
            }
            break;
        }
        default:
            std::string errMsg = io::xprintf("Frame statistic for %s is unsupported.",
                                             io::DenSupportedTypeToString(t));
            LOGE << errMsg;
            throw std::runtime_error(errMsg);
        }
    }
}

int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Information about DEN file and its individual slices." };
    app.add_option("input_den_file", input_file, "File in a DEN format to process.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("-f,--frames", frameSpecs,
                   "Specify only particular frames to process. You can input range i.e. 0-20 or "
                   "also individual comma separated frames i.e. 1,8,9. Order does matter. Accepts "
                   "end literal that means total number of slices of the input.");
    app.add_flag("--l2norm", l2norm, "Print l2 norm of the frame specs.");
    app.add_flag("--dim", returnDimensions,
                 "Return only the dimensions in a format x\\ty\\tz\\n and quit.");

    try
    {
        app.parse(argc, argv);
        if(returnDimensions)
        {
            return 0; // Do not process frames and print a log message.
        }
        if(app.count("--frames") > 0)
        {
            framesSpecified = true;
        }
        io::DenFileInfo inf(input_file);
        frames = util::processFramesSpecification(frameSpecs, inf.dimz());
    } catch(const CLI::ParseError& e)
    {
        int exitcode = app.exit(e);
        if(exitcode == 0) // Help message was printed
        {
            return 1;
        } else
        {
            LOGE << "Parse error catched";
            // Negative value should be returned
            return -1;
        }
    }

    return 0;
}
