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
#include "ARGPARSE/parseArgs.h"
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"

using namespace CTL;

// Function declarations (definition at the end of the file)

// class declarations
struct Args
{
    int parseArguments(int argc, char* argv[]);
    std::string input_file;
    std::string output_file;
    std::string frames = "";
    int threads = 1;
    int k = 1;
    bool reverse_order = false;
};

void writeFrameFloat(int id,
                     int fromId,
                     std::shared_ptr<io::Frame2DReaderI<float>> denSliceReader,
                     int toId,
                     std::shared_ptr<io::AsyncFrame2DWritterI<float>> imagesWritter)
{
    imagesWritter->writeFrame(*(denSliceReader->readFrame(fromId)), toId);
    //    LOGD << io::xprintf("Writting %d th slice from %d th image.", toId, fromId);
}

void writeFrameDouble(int id,
                      int fromId,
                      std::shared_ptr<io::Frame2DReaderI<double>> denSliceReader,
                      int toId,
                      std::shared_ptr<io::AsyncFrame2DWritterI<double>> imagesWritter)
{
    imagesWritter->writeFrame(*(denSliceReader->readFrame(fromId)), toId);
    //    LOGD << io::xprintf("Writting %d th slice from %d th image.", toId, fromId);
}

void writeFrameUint16(int id,
                      int fromId,
                      std::shared_ptr<io::Frame2DReaderI<uint16_t>> denSliceReader,
                      int toId,
                      std::shared_ptr<io::AsyncFrame2DWritterI<uint16_t>> imagesWritter)
{
    imagesWritter->writeFrame(*(denSliceReader->readFrame(fromId)), toId);
    //    LOGD << io::xprintf("Writting %d th slice from %d th image.", toId, fromId);
}

int main(int argc, char* argv[])
{
    plog::Severity verbosityLevel
        = plog::debug; // Set to debug to see the debug messages, info messages
    std::string csvLogFile = "/tmp/imageRegistrationLog.csv"; // Set NULL to disable
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    LOGI << "dentk-cat";
    // Argument parsing
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
    int dimx = di.getNumCols();
    int dimy = di.getNumRows();
    std::vector<int> framesToProcess
        = util::processFramesSpecification(a.frames, di.getNumSlices());
    if(a.reverse_order)
    {
        std::reverse(framesToProcess.begin(), framesToProcess.end()); // It really does!
    }
    std::vector<int> framesToOutput;
    for(int i = 0; i != framesToProcess.size(); i++)
    {
        if(i % a.k == 0)
        {
            framesToOutput.push_back(framesToProcess[i]);
        }
    }
    ctpl::thread_pool* threadpool = new ctpl::thread_pool(a.threads);
    switch(dataType)
    {
    case io::DenSupportedType::uint16_t_:
    {
        std::shared_ptr<io::Frame2DReaderI<uint16_t>> denSliceReader
            = std::make_shared<io::DenFrame2DReader<uint16_t>>(a.input_file);
        std::shared_ptr<io::AsyncFrame2DWritterI<uint16_t>> imagesWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<uint16_t>>(a.output_file, dimx, dimy,
                                                                     framesToOutput.size());
        for(int i = 0; i != framesToOutput.size(); i++)
        {
            // Try asynchronous calls
            threadpool->push(writeFrameUint16, framesToOutput[i], denSliceReader, i, imagesWritter);
        }
        break;
    }
    case io::DenSupportedType::float_:
    {
        std::shared_ptr<io::Frame2DReaderI<float>> denSliceReader
            = std::make_shared<io::DenFrame2DReader<float>>(a.input_file);
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> imagesWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(a.output_file, dimx, dimy,
                                                                  framesToOutput.size());
        for(int i = 0; i != framesToOutput.size(); i++)
        {
            // Try asynchronous calls
            threadpool->push(writeFrameFloat, framesToOutput[i], denSliceReader, i, imagesWritter);
        }
        break;
    }
    case io::DenSupportedType::double_:
    {
        std::shared_ptr<io::Frame2DReaderI<double>> denSliceReader
            = std::make_shared<io::DenFrame2DReader<double>>(a.input_file);
        std::shared_ptr<io::AsyncFrame2DWritterI<double>> imagesWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<double>>(a.output_file, dimx, dimy,
                                                                   framesToOutput.size());
        for(int i = 0; i != framesToOutput.size(); i++)
        {
            // Try asynchronous calls
            threadpool->push(writeFrameDouble, framesToOutput[i], denSliceReader, i, imagesWritter);
        }
        break;
    }
    default:
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(dataType));
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
    }

            threadpool->stop(true);
    delete threadpool;
}

int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Extract particular frames from DEN file." };
    app.add_flag("-r,--reverse_order", reverse_order,
                 "Output in the reverse order of input or reverse specified frames.");
    app.add_option("-f,--frames", frames,
                   "Specify only particular frames to process. You can input range i.e. 0-20 or "
                   "also individual coma separated frames i.e. 1,8,9. Order does matter. Accepts "
                   "end literal that means total number of slices of the input.");
    app.add_option("-k,--each-kth", k,
                   "Process only each k-th frame specified by k to output. The frames to output "
                   "are then 1st specified, 1+kN, N=1...\\infty if such frame exists. Parametter k "
                   "must be positive integer.")
        ->check(CLI::Range(1, 65535));
    app.add_option("-j,--threads", threads, "Number of extra threads that application can use.")
        ->check(CLI::Range(1, 65535));
    app.add_option("input_den_file", input_file, "File in a DEN format to process.")->required();
    app.add_option("output_den_file", output_file, "File in a DEN format to output.")->required();

    try
    {
        app.parse(argc, argv);
        LOGD << io::xprintf("Input file is %s and output file is %s.", input_file.c_str(),
                            output_file.c_str());
    } catch(const CLI::ParseError& e)
    {
        int exitcode = app.exit(e);
        if(exitcode == 0) // Help message was printed
        {
            return 1;
        } else
        {
            LOGE << "Parse error catched";
            return -1;
        }
    }
    return 0;
}
