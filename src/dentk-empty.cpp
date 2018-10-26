// Logging
#include <utils/PlogSetup.h>

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

using namespace CTL;

int main(int argc, char* argv[])
{
    plog::Severity verbosityLevel
        = plog::debug; // Set to debug to see the debug messages, info messages
    std::string csvLogFile = "/tmp/imageRegistrationLog.csv"; // Set NULL to disable
    bool logToConsole = true;
    util::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    // Argument parsing
    int a_dimx, a_dimy, a_dimz;
    std::string a_outputFile;
    std::string a_type = "float";
    CLI::App app{ "Create empty den file." };
    app.add_option("-t,--type", a_type,
                   "Type of the base data unit in the DEN file, might be float, double or "
                   "uint16_t, default is float.");
    app.add_option("dimx", a_dimx, "X dimension.")->required()->check(CLI::Range(0, 65535));
    app.add_option("dimy", a_dimy, "Y dimension.")->required()->check(CLI::Range(0, 65535));
    app.add_option("dimz", a_dimz, "Z dimension.")->required()->check(CLI::Range(0, 65535));
    app.add_option("output_den_file1", a_outputFile, "File in a DEN format to output.")->required();
    CLI11_PARSE(app, argc, argv);
    long elementByteSize;
    if(a_type == "float")
    {
        elementByteSize = 4;
        LOGD << io::xprintf(
            "Creating file %s with data type float and dimensions (x,y,z) = (%d, %d, %d).",
            a_outputFile.c_str(), a_dimx, a_dimy, a_dimz);
    } else if(a_type == "double")
    {
        elementByteSize = 8;
        LOGD << io::xprintf(
            "Creating file %s with data type double and dimensions (x,y,z) = (%d, %d, %d).",
            a_outputFile.c_str(), a_dimx, a_dimy, a_dimz);
    } else if(a_type == "uint16_t")
    {
        elementByteSize = 2;
        LOGD << io::xprintf(
            "Creating file %s with data type uint16_t and dimensions (x,y,z) = (%d, %d, %d).",
            a_outputFile.c_str(), a_dimx, a_dimy, a_dimz);
    } else
    {
        std::string err = io::xprintf("Unrecognized data type %s, for help run dentk-empty -h.",
                                      a_type.c_str());
        LOGE << err;
        throw std::runtime_error(err);
    }
    long totalFileSize = 6 + elementByteSize * a_dimx * a_dimy * a_dimz;
    io::createEmptyFile(a_outputFile, totalFileSize, true);
    uint8_t buffer[6];
    util::putUint16((uint16_t)a_dimy, &buffer[0]);
    util::putUint16((uint16_t)a_dimx, &buffer[2]);
    util::putUint16((uint16_t)a_dimz, &buffer[4]);
    io::writeFirstBytes(a_outputFile, buffer, 6);
}
