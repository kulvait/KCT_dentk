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
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/Program.hpp"

using namespace KCT;
using namespace KCT::util;

// Function declarations (definition at the end of the file)

// class declarations
struct Args : public ArgumentsFramespec, public ArgumentsThreading
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsFramespec(argc, argv, prgName)
        , ArgumentsThreading(argc, argv, prgName){};
    std::string input_file;
    std::string output_file;
};

template <class T>
void writeFrame(int id,
                uint32_t fromId,
                std::shared_ptr<io::Frame2DReaderI<T>> denFrameReader,
                uint32_t toId,
                std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter)
{
    imagesWritter->writeFrame(*(denFrameReader->readFrame(fromId)), toId);
    //    LOGD << io::xprintf("Writting %d th slice from %d th image.", toId, fromId);
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    const std::string prgInfo = "Extract and reorder particular frames from DEN file.";
    Args ARG(argc, argv, prgInfo);
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    io::DenFileInfo di(ARG.input_file);
    io::DenSupportedType dataType = di.getDataType();
    uint32_t dimx = di.dimx();
    uint32_t dimy = di.dimy();
    ctpl::thread_pool* threadpool = nullptr;
    if(ARG.threads != 0)
    {
        threadpool = new ctpl::thread_pool(ARG.threads);
    }
    switch(dataType)
    {
    case io::DenSupportedType::UINT16: {
        std::shared_ptr<io::Frame2DReaderI<uint16_t>> denFrameReader
            = std::make_shared<io::DenFrame2DReader<uint16_t>>(ARG.input_file);
        std::shared_ptr<io::AsyncFrame2DWritterI<uint16_t>> imagesWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<uint16_t>>(ARG.output_file, dimx, dimy,
                                                                     ARG.frames.size());
        for(uint32_t i = 0; i != ARG.frames.size(); i++)
        {
            // Try asynchronous calls
            if(threadpool != nullptr)
            {
                threadpool->push(writeFrame<uint16_t>, ARG.frames[i], denFrameReader, i,
                                 imagesWritter);
            } else
            {
                writeFrame<uint16_t>(0, ARG.frames[i], denFrameReader, i, imagesWritter);
            }
        }
        break;
    }
    case io::DenSupportedType::FLOAT32: {
        std::shared_ptr<io::Frame2DReaderI<float>> denFrameReader
            = std::make_shared<io::DenFrame2DReader<float>>(ARG.input_file);
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> imagesWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(ARG.output_file, dimx, dimy,
                                                                  ARG.frames.size());
        for(uint32_t i = 0; i != ARG.frames.size(); i++)
        {
            // Try asynchronous calls
            if(threadpool != nullptr)
            {
                threadpool->push(writeFrame<float>, ARG.frames[i], denFrameReader, i,
                                 imagesWritter);
            } else
            {
                writeFrame<float>(0, ARG.frames[i], denFrameReader, i, imagesWritter);
            }
        }
        break;
    }
    case io::DenSupportedType::FLOAT64: {
        std::shared_ptr<io::Frame2DReaderI<double>> denFrameReader
            = std::make_shared<io::DenFrame2DReader<double>>(ARG.input_file);
        std::shared_ptr<io::AsyncFrame2DWritterI<double>> imagesWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<double>>(ARG.output_file, dimx, dimy,
                                                                   ARG.frames.size());
        for(uint32_t i = 0; i != ARG.frames.size(); i++)
        {
            // Try asynchronous calls
            if(threadpool != nullptr)
            {
                threadpool->push(writeFrame<double>, ARG.frames[i], denFrameReader, i,
                                 imagesWritter);
            } else
            {
                writeFrame<double>(0, ARG.frames[i], denFrameReader, i, imagesWritter);
            }
        }
        break;
    }
    default:
        std::string errMsg = io::xprintf("Unsupported data type %s.",
                                         io::DenSupportedTypeToString(dataType).c_str());
        KCTERR(errMsg);
    }
    if(threadpool != nullptr)
    {
        threadpool->stop(true);
        delete threadpool;
    }
    PRG.endLog();
}

void Args::defineArguments()
{
    cliApp->add_option("input_den_file", input_file, "File in a DEN format to process.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("output_den_file", output_file, "File in a DEN format to output.")
        ->required();
    addFramespecArgs();
    addThreadingArgs();
}

int Args::postParse()
{
    std::string err;
    if(input_file.compare(output_file) == 0)
    {
        err = io::xprintf("Input and output files should be different!");
        LOGE << err;
        return -1;
    }
    io::DenFileInfo inf(input_file);
    fillFramesVector(inf.dimz());
    return 0;
}
