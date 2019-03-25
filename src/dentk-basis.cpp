#define DEBUG
// Logging
#include "PLOG/PlogSetup.h"

// External libraries

// External libraries
#include "CLI/CLI.hpp"
#include "matplotlibcpp.h"
#include "mkl.h"

// Internal libraries
#include "AsyncFrame2DWritterI.hpp"
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "FUN/StepFunction.hpp"
#include "Frame2DI.hpp"
#include "frameop.h"
#include "rawop.h"

namespace plt = matplotlibcpp;

using namespace CTL;

// class declarations
struct Args
{
    int parseArguments(int argc, char* argv[]);
    std::string input_file;
    bool vizualize = false, info = false;
    std::string output_file = "";
    bool force;
    uint16_t resampledGranularity = 0;
    bool orthogonalize = false;
    int shiftBasis = 0;
    uint16_t granularity;
    uint16_t baseSize;
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
    // int elementSize = di.elementByteSize();
    io::DenSupportedType t = di.getDataType();
    std::string elm = io::DenSupportedTypeToString(t);
    if(a.info)
    {
        std::cout << io::xprintf(
            "The file %s of type %s contains %d basis functions with the discretization %d.\n",
            a.input_file.c_str(), elm.c_str(), a.baseSize, a.granularity);
    }
    if(a.vizualize)
    {
	LOGI << io::xprintf("Granularity of %s is %d", a.input_file.c_str(), a.granularity);
        util::StepFunction b(a.baseSize, a.input_file, 0.0, double(a.granularity - 1));
        b.plotFunctions(a.granularity);
    }
    switch(t)
    {
    case io::DenSupportedType::uint16_t_:
    {
        std::cout << io::xprintf("Not implemented yet.\n");
        break;
    }
    case io::DenSupportedType::float_:
    {
        if(a.shiftBasis != 0)
        {
            std::shared_ptr<io::Frame2DReaderI<float>> denSliceReader
                = std::make_shared<io::DenFrame2DReader<float>>(a.input_file);
            std::shared_ptr<io::AsyncFrame2DWritterI<float>> imagesWritter
                = std::make_shared<io::DenAsyncFrame2DWritter<float>>(a.output_file, a.granularity, 1,
                                                                      a.baseSize);

            std::shared_ptr<io::Frame2DI<float>> f;
            io::BufferedFrame2D<float> bf(0.0, a.granularity, 1);
            for(uint32_t i = 0; i != a.baseSize; i++)
            {
                f = denSliceReader->readFrame(i);
                for(int j = 0; j != a.granularity; j++)
                {
                    int index = j + a.shiftBasis;
                    if(index < 0)
                    {
                        index = 0;
                    } else if(index >= a.granularity)
                    {
                        index = a.granularity - 1;
                    }
                    bf.set(f->get(index, 0), j, 0);
                }
                imagesWritter->writeFrame(bf, i);
            }
        }
        break;
    }
    case io::DenSupportedType::double_:
    {
        std::cout << io::xprintf("Not implemented yet.\n");
        break;
    }
    default:
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(t));
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
    }
}

int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Engineered bases processing. These bases are stored in a DEN file, where x is "
                  "granularity y is 1 and z is number of basis functions." };
    app.add_option("input_den_file", input_file, "File in a DEN format to process.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_flag("-v,--vizualize", vizualize, "Vizualize the basis.");
    app.add_flag("-i,--info", info, "Print information about the basis.");
    CLI::Option* _output = app.add_option("-o", output_file, "File to output.");
    CLI::Option* _force = app.add_flag("-f,--force", force, "Force overwrite output file.");
    _force->needs(_output);
    CLI::Option* _resample
        = app.add_option("--resample", resampledGranularity,
                         "Resample discretization to respect new granularity size.");
    _resample->needs(_output);
    CLI::Option* _shift
        = app.add_option("--shift", shiftBasis, "Shift basis and make it continuous at boundaries. Specify coordinate of 0 in old basis coordinates.");
    _shift->needs(_output);
    _shift->excludes(_resample);
    _resample->excludes(_shift);
    try
    {
        app.parse(argc, argv);
        if(!force)
        {
            if(app.count("-o") > 0 && io::fileExists(output_file))
            {
                std::string msg = io::xprintf(
                    "Error: output file %s already exists, use -f to force overwrite.",
                    output_file.c_str());
                LOGE << msg;
                return 1;
            }
        }
        io::DenFileInfo di(input_file);
        granularity = di.dimx();
        baseSize = di.dimz();
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
