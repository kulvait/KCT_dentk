// Logging
#include <utils/PlogSetup.h>

// External libraries
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <ctype.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <string>

// External libraries
#include "CLI/CLI.hpp"

// Internal libraries
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "littleEndianAlignment.h"
#include "rawop.h"

using namespace CTL;

// class declarations
struct Args
{
    int parseArguments(int argc, char* argv[]);
    std::string input_file;
    std::string output_file;
    bool forceOverwrite;
};

int main(int argc, char* argv[])
{
    plog::Severity verbosityLevel
        = plog::debug; // Set to debug to see the debug messages, info messages
    std::string csvLogFile = "/tmp/imageRegistrationLog.csv"; // Set NULL to disable
    bool logToConsole = true;
    util::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    LOGI << "dentk-fen2den";
    Args a;
    int parseResult = a.parseArguments(argc, argv);
    if(parseResult != 0)
    {
        if(parseResult > 0)
        {
            return 0; // Exited successfully, help message printed
        } else
        {
            return -1; // Exited somehow wrong
        }
    }
    // First perform byte copy of input to output
    if(a.input_file != a.output_file)
    {
        std::ifstream source(a.input_file, std::ios::binary);
        std::ofstream dest(a.output_file, std::ios::binary);

        std::istreambuf_iterator<char> begin_source(source);
        std::istreambuf_iterator<char> end_source;
        std::ostreambuf_iterator<char> begin_dest(dest);
        std::copy(begin_source, end_source, begin_dest);

        source.close();
        dest.close();
    }
    uint8_t buffer[1024];
    io::readFirstBytes(a.input_file, buffer, 6);

    int i1 = util::nextUint16(&buffer[0]);
    int i2 = util::nextUint16(&buffer[2]);
    int i3 = util::nextUint16(&buffer[4]);
    // Now flip first two shorts
    util::putUint16(i1, &buffer[2]);
    util::putUint16(i2, &buffer[0]);
    io::writeFirstBytes(a.output_file, buffer, 6);
    return 0;
}

int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Information about DEN file and its individual slices." };
    app.add_option("input_fen_file", input_file,
                   "File in a fake den format that should be corrected, first and second header "
                   "items will be flipped.")
        ->check(CLI::ExistingFile)
        ->required();
    app.add_option("output_den_file", output_file, "Corrected file in a DEN format to output.")
        ->required();
    app.add_flag("-f,--force", forceOverwrite, "Force overwriting output file if it exists.");
    try
    {
        app.parse(argc, argv);
        if(!forceOverwrite)
        {
            if(io::fileExists(output_file))
            {
                std::string msg = "Error: output file already exists, use -f to force overwrite.";
                LOGE << msg;
                return 1;
            }
        }
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
