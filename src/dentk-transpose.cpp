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
#include "DEN/DenAsyncFrame2DBufferedWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/Program.hpp"
#include "littleEndianAlignment.h"
#include "rawop.h"

using namespace KCT;
using namespace KCT::util;

// class declarations
class Args : public ArgumentsForce
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName){};
    int parseArguments(int argc, char* argv[]);
    std::string input_file;
    std::string output_file;
    bool outputFileExists = false;
};

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    const std::string prgInfo = "Frame-wise transpose of DEN file.";
    Args ARG(argc, argv, prgInfo);
    int parseResult = ARG.parse(false);
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    // After init parsing arguments
    io::DenFileInfo di(ARG.input_file);
    io::DenSupportedType dataType = di.getElementType();
    uint16_t dimCount = di.getDimCount();
    uint32_t K = di.getFrameCount();
    std::vector<uint32_t> dim;
    //Transposed specification
    dim.push_back(di.dim(1));
    dim.push_back(di.dim(0));
    for(uint16_t i = 2; i < dimCount; i++)
    {
        dim.push_back(di.dim(i));
    }
    switch(dataType)
    {
    case io::DenSupportedType::UINT16: {
        std::shared_ptr<io::Frame2DReaderI<uint16_t>> sliceReader
            = std::make_shared<io::DenFrame2DReader<uint16_t>>(ARG.input_file);
        std::shared_ptr<io::AsyncFrame2DWritterI<uint16_t>> imagesWritter
            = std::make_shared<io::DenAsyncFrame2DBufferedWritter<uint16_t>>(
                ARG.output_file, dimCount, &dim.front());
        std::shared_ptr<io::Frame2DI<uint16_t>> chunk, transposed;
        for(int i = 0; i != K; i++)
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
    case io::DenSupportedType::FLOAT32: {
        std::shared_ptr<io::Frame2DReaderI<float>> sliceReader
            = std::make_shared<io::DenFrame2DReader<float>>(ARG.input_file);
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> imagesWritter
            = std::make_shared<io::DenAsyncFrame2DBufferedWritter<float>>(ARG.output_file, dimCount,
                                                                          &dim.front());
        std::shared_ptr<io::Frame2DI<float>> chunk, transposed;

        for(int i = 0; i != K; i++)
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
    case io::DenSupportedType::FLOAT64: {
        std::shared_ptr<io::Frame2DReaderI<double>> sliceReader
            = std::make_shared<io::DenFrame2DReader<double>>(ARG.input_file);
        std::shared_ptr<io::AsyncFrame2DWritterI<double>> imagesWritter
            = std::make_shared<io::DenAsyncFrame2DBufferedWritter<double>>(ARG.output_file,
                                                                           dimCount, &dim.front());
        std::shared_ptr<io::Frame2DI<double>> chunk, transposed;
        for(int i = 0; i != K; i++)
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
        std::string errMsg = io::xprintf("Unsupported data type %s.",
                                         io::DenSupportedTypeToString(dataType).c_str());
        KCTERR(errMsg);
    }
    PRG.endLog(true);
    return 0;
}

void Args::defineArguments()
{
    cliApp->add_option("input_den_file", input_file, "File that will be transposed.")
        ->check(CLI::ExistingFile)
        ->required();
    cliApp->add_option("output_den_file", output_file, "Transposed file in a DEN format to output.")
        ->required();
    addForceArgs();
}

int Args::postParse()
{
    int existFlag = handleFileExistence(output_file, force, input_file);
    if(existFlag == 1)
    {
        return 1;
    } else if(existFlag == -1)
    {
        outputFileExists = true;
    }
    io::DenFileInfo di(input_file);
    if(di.getDimCount() < 2)
    {
        std::string ERR = io::xprintf("The file %s has just %d<2 dimensions!", input_file.c_str(),
                                      di.getDimCount());
        KCTERR(ERR);
    }
    return 0;
}
