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
#include "ctpl_stl.h" //Threadpool

// Internal libraries
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/Program.hpp"
#include "PROG/Arguments.hpp"

using namespace CTL;
using namespace CTL::util;

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
    io::DenSupportedType dataType = di.getDataType();
    switch(dataType)
    {
    case io::DenSupportedType::uint16_t_:
    {
        std::shared_ptr<io::Frame2DReaderI<uint16_t>> denFrameReader
            = std::make_shared<io::DenFrame2DReader<uint16_t>>(ARG.input_file);
        uint16_t v = denFrameReader->readFrame(ARG.z)->get(ARG.x, ARG.y);
        std::cout << v;
        break;
    }
    case io::DenSupportedType::float_:
    {
        std::shared_ptr<io::Frame2DReaderI<float>> denFrameReader
            = std::make_shared<io::DenFrame2DReader<float>>(ARG.input_file);
        float v = denFrameReader->readFrame(ARG.z)->get(ARG.x, ARG.y);
        std::cout << v;
        break;
    }
    case io::DenSupportedType::double_:
    {
        std::shared_ptr<io::Frame2DReaderI<double>> denFrameReader
            = std::make_shared<io::DenFrame2DReader<double>>(ARG.input_file);
        double v = denFrameReader->readFrame(ARG.z)->get(ARG.x, ARG.y);
        std::cout << v;
        break;
    }
    default:
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(dataType));
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
    }
}

void Args::defineArguments()
{
    cliApp->add_option("x", x, "X coordinate.")->required();
    cliApp->add_option("x", y, "X coordinate.")->required();
    cliApp->add_option("x", z, "X coordinate.")->required();
    cliApp->add_option("input_den_file", input_file, "File in a DEN format to process.")
        ->required()
        ->check(CLI::ExistingFile);
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
