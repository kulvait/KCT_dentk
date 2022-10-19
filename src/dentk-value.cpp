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
#include "ftpl.h" //Threadpool

// Internal libraries
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/Arguments.hpp"
#include "PROG/Program.hpp"

using namespace KCT;
using namespace KCT::util;

// Function declarations (definition at the end of the file)

// class declarations
struct Args : public Arguments
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName){};
    std::string input_file;
    uint32_t x, y, z;
    double value = std::numeric_limits<double>::quiet_NaN();
};

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    Args ARG(argc, argv, "Extract single value from the file.");
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    io::DenFileInfo di(ARG.input_file);
    io::DenSupportedType dataType = di.getElementType();
    switch(dataType)
    {
    case io::DenSupportedType::UINT16: {
        std::shared_ptr<io::Frame2DReaderI<uint16_t>> denFrameReader
            = std::make_shared<io::DenFrame2DReader<uint16_t>>(ARG.input_file);
        std::shared_ptr<io::Frame2DI<uint16_t>> f = denFrameReader->readFrame(ARG.z);
        uint16_t v = f->get(ARG.x, ARG.y);
        std::cout << v;
        if(!std::isnan(ARG.value))
        {
            f->set(ARG.value, ARG.x, ARG.y);
            std::shared_ptr<io::AsyncFrame2DWritterI<uint16_t>> denFrameWritter
                = std::make_shared<io::DenAsyncFrame2DWritter<uint16_t>>(ARG.input_file);
            denFrameWritter->writeFrame(*f, ARG.z);
        }
        break;
    }
    case io::DenSupportedType::FLOAT32: {
        std::shared_ptr<io::Frame2DReaderI<float>> denFrameReader
            = std::make_shared<io::DenFrame2DReader<float>>(ARG.input_file);
        std::shared_ptr<io::Frame2DI<float>> f = denFrameReader->readFrame(ARG.z);
        float v = f->get(ARG.x, ARG.y);
        std::cout << v;
        if(!std::isnan(ARG.value))
        {
            f->set(ARG.value, ARG.x, ARG.y);
            std::shared_ptr<io::AsyncFrame2DWritterI<float>> denFrameWritter
                = std::make_shared<io::DenAsyncFrame2DWritter<float>>(ARG.input_file);
            denFrameWritter->writeFrame(*f, ARG.z);
        }
        break;
    }
    case io::DenSupportedType::FLOAT64: {
        std::shared_ptr<io::Frame2DReaderI<double>> denFrameReader
            = std::make_shared<io::DenFrame2DReader<double>>(ARG.input_file);
        std::shared_ptr<io::Frame2DI<double>> f = denFrameReader->readFrame(ARG.z);
        double v = f->get(ARG.x, ARG.y);
        std::cout << v;
        if(!std::isnan(ARG.value))
        {
            f->set(ARG.value, ARG.x, ARG.y);
            std::shared_ptr<io::AsyncFrame2DWritterI<double>> denFrameWritter
                = std::make_shared<io::DenAsyncFrame2DWritter<double>>(ARG.input_file);
            denFrameWritter->writeFrame(*f, ARG.z);
        }
        break;
    }
    default:
        std::string errMsg = io::xprintf("Unsupported data type %s.",
                                         io::DenSupportedTypeToString(dataType).c_str());
        KCTERR(errMsg);
    }
}

void Args::defineArguments()
{
    cliApp->add_option("x", x, "X coordinate.")->required();
    cliApp->add_option("y", y, "Y coordinate.")->required();
    cliApp->add_option("z", z, "Z coordinate.")->required();
    cliApp->add_option("input_den_file", input_file, "File in a DEN format to process.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("--set-value", value,
                       "If the parameter is specified, the value of is set for given coordinates.");
}

int Args::postParse()
{
    std::string err;
    io::DenFileInfo di(input_file);
    uint32_t dimx = di.dimx();
    uint32_t dimy = di.dimy();
    uint32_t dimz = di.dimz();
    if(!(x < dimx && y < dimy && z < dimz))
    {
        err = io::xprintf("Specified coordinates (x,y,z) are out of the range of the DEN file!");
        LOGE << err;
        return 1;
    }
    return 0;
}
