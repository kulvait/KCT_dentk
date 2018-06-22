//Logging
#include <utils/PlogSetup.h>

//External libraries
#include <algorithm>
#include <cstdlib>
#include <ctype.h>
#include <iostream>
#include <regex>
#include <string>

//External libraries
#include "CLI/CLI.hpp" //Command line parser

//Internal libraries
#include "io/AsyncImageWritterI.hpp"
#include "io/DenAsyncWritter.hpp"

using namespace CTL;

int main(int argc, char* argv[])
{
    plog::Severity verbosityLevel = plog::debug; //Set to debug to see the debug messages, info messages
    std::string csvLogFile = "/tmp/imageRegistrationLog.csv"; //Set NULL to disable
    bool logToConsole = true;
    util::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    //Argument parsing
    int a_dimx, a_dimy, a_dimz;
    std::string a_outputFile;
    CLI::App app{ "Create empty den file." };
    app.add_option("dimx", a_dimx, "X dimension.")->required()->check(CLI::Range(0, 65535));
    app.add_option("dimx", a_dimy, "Y dimension.")->required()->check(CLI::Range(0, 65535));
    app.add_option("dimx", a_dimz, "Z dimension.")->required()->check(CLI::Range(0, 65535));
    app.add_option("output_den_file1", a_outputFile, "File in a DEN format to output.")->required();
    CLI11_PARSE(app, argc, argv);
    LOGD << io::xprintf("Creating file %s with dimensions (x,y,z) = (%d, %d, %d).", a_outputFile.c_str(), a_dimx, a_dimy, a_dimz);
    std::shared_ptr<io::AsyncImageWritterI<float>> imagesWritter = std::make_shared<io::DenAsyncWritter<float>>(a_outputFile, a_dimx, a_dimy, a_dimz);
    //    imagesWritter->writeSlice(*(denSliceReader->readSlice(fromId)), toId);
}
