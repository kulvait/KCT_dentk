// Logging
#include "PLOG/PlogSetup.h"

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
    std::string csvLogFile = "/tmp/dentk-transpose.csv"; // Set NULL to disable
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    LOGI << "dentk-merge";
    // Command line parsing
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
    io::DenSupportedType dataType = di.getDataType();
    int i3 = di.getNumSlices();
    switch(dataType)
    {
    case io::DenSupportedType::uint16_t_:
    {
        std::shared_ptr<io::Frame2DReaderI<uint16_t>> sliceReader
            = std::make_shared<io::DenFrame2DReader<uint16_t>>(a.input_file);
        std::shared_ptr<io::AsyncFrame2DWritterI<uint16_t>> imagesWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<uint16_t>>(
                a.output_file, sliceReader->dimy(), sliceReader->dimx(), i3);
        std::shared_ptr<io::Frame2DI<uint16_t>> chunk, transposed;
        for(int i = 0; i != i3; i++)
        {
            chunk = sliceReader->readFrame(i);
            std::shared_ptr<io::BufferedFrame2D<uint16_t>> retyped;
            retyped
                = std::dynamic_pointer_cast<io::BufferedFrame2D<uint16_t>, io::Frame2DI<uint16_t>>(
                    chunk);
            transposed = retyped->transposed();
            imagesWritter->writeFrame(*transposed, i);
        }
        break;
    }
    case io::DenSupportedType::float_:
    {
        std::shared_ptr<io::Frame2DReaderI<float>> sliceReader
            = std::make_shared<io::DenFrame2DReader<float>>(a.input_file);
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> imagesWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
                a.output_file, sliceReader->dimy(), sliceReader->dimx(), i3);
        std::shared_ptr<io::Frame2DI<float>> chunk, transposed;

        for(int i = 0; i != i3; i++)
        {
            chunk = sliceReader->readFrame(i);
            std::shared_ptr<io::BufferedFrame2D<float>> retyped;
            retyped
                = std::dynamic_pointer_cast<io::BufferedFrame2D<float>, io::Frame2DI<float>>(chunk);
            transposed = retyped->transposed();
            imagesWritter->writeFrame(*transposed, i);
        }
        break;
    }
    case io::DenSupportedType::double_:
    {
        std::shared_ptr<io::Frame2DReaderI<double>> sliceReader
            = std::make_shared<io::DenFrame2DReader<double>>(a.input_file);
        std::shared_ptr<io::AsyncFrame2DWritterI<double>> imagesWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<double>>(
                a.output_file, sliceReader->dimy(), sliceReader->dimx(), i3);
        std::shared_ptr<io::Frame2DI<double>> chunk, transposed;
        for(int i = 0; i != i3; i++)
        {
            chunk = sliceReader->readFrame(i);
            std::shared_ptr<io::BufferedFrame2D<double>> retyped;
            retyped = std::dynamic_pointer_cast<io::BufferedFrame2D<double>, io::Frame2DI<double>>(
                chunk);
            transposed = retyped->transposed();
            imagesWritter->writeFrame(*transposed, i);
        }
        break;
    }
    default:
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(dataType));
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
    }

    return 0;
}

int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Transpose all frames in a DEN file." };
    app.add_option("input_den_file", input_file, "File that will be transposed.")
        ->check(CLI::ExistingFile)
        ->required();
    app.add_option("output_den_file", output_file, "Transposed file in a DEN format to output.")
        ->required();
    app.add_flag("-f,--force", forceOverwrite, "Force overwriting output file if it exists.");
    try
    {
        app.parse(argc, argv);
        if(!forceOverwrite)
        {
            if(io::pathExists(output_file))
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
