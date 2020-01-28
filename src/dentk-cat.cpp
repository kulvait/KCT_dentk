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
#include "PROG/Arguments.hpp"
#include "PROG/Program.hpp"
#include "PROG/parseArgs.h"

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
    std::string output_file;
    std::string frameSpecs = "";
    std::vector<int> frames;
    uint32_t threads = 0;
    int k = 1;
    bool reverse_order = false;
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
    Args ARG(argc, argv, "Extract and reorder particular frames from DEN file.");
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog();
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
    case io::DenSupportedType::uint16_t_:
    {
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
    case io::DenSupportedType::float_:
    {
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
    case io::DenSupportedType::double_:
    {
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
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(dataType));
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
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
    cliApp->add_flag("-r,--reverse_order", reverse_order,
                     "Output in the reverse order of input or reverse specified frames.");
    cliApp->add_option(
        "-f,--frames", frameSpecs,
        "Specify only particular frames to process. You can input range i.e. 0-20 or "
        "also individual coma separated frames i.e. 1,8,9. Order does matter. Accepts "
        "end literal that means total number of slices of the input.");
    cliApp
        ->add_option(
            "-k,--each-kth", k,
            "Process only each k-th frame specified by k to output. The frames to output "
            "are then 1st specified, 1+kN, N=1...\\infty if such frame exists. Parametter k "
            "must be positive integer.")
        ->check(CLI::Range(1, 65535));
    cliApp
        ->add_option("-j,--threads", threads,
                     "Number of extra threads that cliApplication can use. Defaults to 0, disabled "
                     "threading.")
        ->check(CLI::Range(0, 65535));
    cliApp->add_option("input_den_file", input_file, "File in a DEN format to process.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("output_den_file", output_file, "File in a DEN format to output.")
        ->required();
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
    std::vector<int> f = util::processFramesSpecification(frameSpecs, inf.dimz());
    if(reverse_order)
    {
        std::reverse(f.begin(), f.end()); // It really does!
    }
    for(std::size_t i = 0; i != f.size(); i++)
    {
        if(i % k == 0)
        {
            frames.push_back(f[i]);
        }
    }
    return 0;
}
