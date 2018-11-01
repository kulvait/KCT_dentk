// Logging
#include "PLOG/PlogSetup.h"
// External libraries
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctype.h>
#include <experimental/filesystem>
#include <iostream>
#include <regex>
#include <string>
// External libraries
#include "CLI/CLI.hpp" //Command line parser
// Internal libraries
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
using namespace CTL;
namespace fs = std::experimental::filesystem;
struct Args
{
    int parseArguments(int argc, char* argv[]);
    std::string input_file = "";
    std::string output_file = "";
    float water_value = 0.027;
};

int main(int argc, char* argv[])
{
    plog::Severity verbosityLevel
        = plog::debug; // Set to debug to see the debug messages, info messages
    std::string csvLogFile = "/tmp/dentk-fromhu.csv"; // Set NULL to disable
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    LOGI << "dentk-tohu";
    // Process arguments
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
    io::DenSupportedType dataType = di.getDataType();
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> outputWritter
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(a.output_file, dimx, dimy, dimz);
    float* buffer = new float[dimx * dimy];
    std::shared_ptr<io::Frame2DI<float>> x
        = std::make_shared<io::BufferedFrame2D<float>>(buffer, dimx, dimy);
    delete buffer;
    switch(dataType)
    {
    case io::DenSupportedType::uint16_t_:
    {
        std::shared_ptr<io::Frame2DReaderI<uint16_t>> sliceReader
            = std::make_shared<io::DenFrame2DReader<uint16_t>>(a.input_file);
        for(int k = 0; k != dimz; k++)
        {
            std::shared_ptr<io::Frame2DI<uint16_t>> f = sliceReader->readFrame(k);
            for(int i = 0; i != dimx; i++)
                for(int j = 0; j != dimy; j++)
                {
                    float v = (float)f->get(i, j); // From uint16_t
                    x->set(v * float(1000.0 / a.water_value), i, j);
                }
            outputWritter->writeFrame(*x, k);
        }
        break;
    }
    case io::DenSupportedType::float_:
    {
        std::shared_ptr<io::Frame2DReaderI<float>> sliceReader
            = std::make_shared<io::DenFrame2DReader<float>>(a.input_file);
        for(int k = 0; k != dimz; k++)
        {
            std::shared_ptr<io::Frame2DI<float>> f = sliceReader->readFrame(k);
            for(int i = 0; i != dimx; i++)
                for(int j = 0; j != dimy; j++)
                {
                    float v = f->get(i, j); // From uint16_t
                    x->set(v * float(1000 / a.water_value) - float(1000.0), i, j);
                }
            outputWritter->writeFrame(*x, k);
        }
        break;
    }
    case io::DenSupportedType::double_:
    {
        std::shared_ptr<io::Frame2DReaderI<float>> sliceReader
            = std::make_shared<io::DenFrame2DReader<float>>(a.input_file);
        for(int k = 0; k != dimz; k++)
        {
            std::shared_ptr<io::Frame2DI<float>> f = sliceReader->readFrame(k);
            for(int i = 0; i != dimx; i++)
                for(int j = 0; j != dimy; j++)
                {
                    float v = (float)f->get(i, j); // From double
                    x->set(v * float(1000.0 / a.water_value) - float(1000.0), i, j);
                }
            outputWritter->writeFrame(*x, k);
        }
        break;
    }
    default:
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(dataType));
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
    }
}

int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{
        "Convert DEN file with data in nonscaled to the Hounsfield units containing floats."
        "For input_file containing floats or doubles coefs = coefs*(1000/w)-1000."
        "For uint16 coefs = coefs*(1000/w). Default w=0.027. Ouput den file contains floats."
    };
    app.add_option("den_file", input_file, "File in a DEN format in HU to process.")
        ->check(CLI::ExistingFile);
    app.add_option("output_DICOM_dir", output_file, "File in a float DEN format after conversion.")
        ->required();
    app.add_option("-w,--water-value", water_value, "Water value to use, defaults to 0.027")
        ->check(CLI::Range(0.0, 1.0));
    try
    {
        app.parse(argc, argv);
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
