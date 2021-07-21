// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <algorithm>
#include <cstdlib>
#include <ctype.h>
#include <iostream>
#include <regex>
#include <string>

// External libraries
#include "CLI/CLI.hpp" //Command line parser

// Internal libraries
#include "rawop.h"
#include "stringFormatter.h"
#include <BufferedFrame2D.hpp>
#include <DEN/DenAsyncFrame2DWritter.hpp>

using namespace CTL;

struct Args
{
    int parseArguments(int argc, char* argv[]);
    uint16_t dimx, dimy, dimz;
    uint64_t elementByteSize;
    std::string type = "float";
    double value = 0.0;
    bool force;
    std::string outputFile;
};

int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Create empty den file." };
    app.add_option("-t,--type", type,
                   "Type of the base data unit in the DEN file, might be float, double or "
                   "uint16_t, default is float.");
    app.add_option("--value", value, io::xprintf("Default value, defaults to %f", value));
    app.add_option("dimx", dimx, "X dimension.")->required()->check(CLI::Range(0, 65535));
    app.add_option("dimy", dimy, "Y dimension.")->required()->check(CLI::Range(0, 65535));
    app.add_option("dimz", dimz, "Z dimension.")->required()->check(CLI::Range(0, 65535));
    app.add_option("output_den_file", outputFile, "File in a DEN format to output.")->required();
    app.add_flag("-f,--force", force, "Overwrite outputFile if it exists.");

    try
    {
        app.parse(argc, argv);
        // If force is not set, then check if output file does not exist
        if(!force)
        {
            if(io::pathExists(outputFile))
            {
                std::string msg
                    = "Error: output file already exists, use --force to force overwrite.";
                LOGE << msg;
                return -1;
            }
        }
        if(type == "float")
        {
            elementByteSize = 4;
            LOGD << io::xprintf(
                "Creating file %s with data type float and dimensions (x,y,z) = (%d, %d, %d).",
                outputFile.c_str(), dimx, dimy, dimz);
        } else if(type == "double")
        {
            elementByteSize = 8;
            LOGD << io::xprintf(
                "Creating file %s with data type double and dimensions (x,y,z) = (%d, %d, %d).",
                outputFile.c_str(), dimx, dimy, dimz);
        } else if(type == "uint16_t")
        {
            elementByteSize = 2;
            LOGD << io::xprintf(
                "Creating file %s with data type uint16_t and dimensions (x,y,z) = (%d, %d, %d).",
                outputFile.c_str(), dimx, dimy, dimz);
        } else
        {
            std::string err = io::xprintf("Unrecognized data type %s, for help run dentk-empty -h.",
                                          type.c_str());
            LOGE << err;
            return -1;
        }
    } catch(const CLI::CallForHelp e)
    {
        app.exit(e); // Prints help message
        return 1;
    } catch(const CLI::ParseError& e)
    {
        int exitcode = app.exit(e);
        LOGE << io::xprintf("There was perse error with exit code %d catched.\n %s", exitcode,
                            app.help().c_str());
        return -1;
    } catch(...)
    {
        LOGE << "Unknown exception catched";
    }
    return 0;
}

template <typename T>
void createConstantDEN(
    std::string fileName, uint32_t sizex, uint32_t sizey, uint32_t sizez, T value)
{
    using namespace CTL;
    io::DenAsyncFrame2DWritter<T> dw(fileName, sizex, sizey, sizez);
    io::BufferedFrame2D<T> f(value, sizex, sizey);
    for(uint32_t k = 0; k != sizez; k++)
    {
        dw.writeFrame(f, k);
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
    LOGI << io::xprintf("START %s", argv[0]);
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
    LOGI << io::xprintf("START %s", argv[0]);
    if(a.elementByteSize == 2)
    {
        createConstantDEN<uint16_t>(a.outputFile, a.dimx, a.dimy, a.dimz, (uint16_t)a.value);
    }
    if(a.elementByteSize == 4)
    {
        createConstantDEN<float>(a.outputFile, a.dimx, a.dimy, a.dimz, (float)a.value);
    }
    if(a.elementByteSize == 8)
    {
        createConstantDEN<double>(a.outputFile, a.dimx, a.dimy, a.dimz, a.value);
    }
    LOGI << io::xprintf("END %s", argv[0]);
}
