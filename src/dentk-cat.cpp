// Logging
#include <utils/PlogSetup.h>

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
#include "strtk.hpp"

// Internal libraries
#include "io/AsyncImageWritterI.hpp"
#include "io/Chunk2DReaderI.hpp"
#include "io/DenAsyncWritter.hpp"
#include "io/DenChunk2DReader.hpp"

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

std::vector<int> processResultingFrames(std::string frameSpecification, int dimz)
{
    // Remove spaces
    for(int i = 0; i < frameSpecification.length(); i++)
        if(frameSpecification[i] == ' ')
            frameSpecification.erase(i, 1);
    frameSpecification = std::regex_replace(frameSpecification, std::regex("end"),
                                            io::xprintf("%d", dimz - 1).c_str());
    LOGD << io::xprintf("Processing frame specification '%s'.", frameSpecification.c_str());
    std::vector<int> frames;
    if(frameSpecification.empty())
    {
        LOGD << "Frame specification is empty, putting all frames.";
        for(int i = 0; i != dimz; i++)
            frames.push_back(i);
    } else
    {
        std::list<std::string> string_list;
        LOGW << "Before parse";
        strtk::parse(frameSpecification, ",", string_list);
        LOGW << "After parse";
        auto it = string_list.begin();
        while(it != string_list.end())
        {
            LOGD << io::xprintf("Iteration where the iterator value is '%s'.", (*it).c_str());
            size_t numRangeSigns = std::count(it->begin(), it->end(), '-');
            if(numRangeSigns > 1)
            {
                std::string msg = io::xprintf("Wrong number of range specifiers in the string %s.",
                                              (*it).c_str());
                LOGE << msg;
                throw std::runtime_error(msg);
            } else if(numRangeSigns == 1)
            {
                std::vector<int> int_vector;
                strtk::parse((*it), "-", int_vector);
                if(0 <= int_vector[0] && int_vector[0] <= int_vector[1] && int_vector[1] < dimz)
                {
                    for(int k = int_vector[0]; k != int_vector[1] + 1; k++)
                    {
                        frames.push_back(k);
                    }
                } else
                {
                    std::string msg
                        = io::xprintf("String %s is invalid range specifier.", (*it).c_str());
                    LOGE << msg;
                    throw std::runtime_error(msg);
                }
            } else
            {
                int index = atoi(it->c_str());
                if(0 <= index && index < dimz)
                {
                    frames.push_back(index);
                } else
                {
                    std::string msg = io::xprintf(
                        "String %s is invalid specifier for the value in the range [0,%d).",
                        (*it).c_str(), dimz);
                    LOGE << msg;
                    throw std::runtime_error(msg);
                }
            }
            it++;
        }
    }
    return frames;
}

void writeFrame(int id,
                int fromId,
                std::shared_ptr<io::Chunk2DReaderI<float>> denSliceReader,
                int toId,
                std::shared_ptr<io::AsyncImageWritterI<float>> imagesWritter)
{
    imagesWritter->writeSlice(*(denSliceReader->readSlice(fromId)), toId);
    LOGD << io::xprintf("Writting %d th slice from %d th image.", toId, fromId);
}

int main(int argc, char* argv[])
{
    plog::Severity verbosityLevel
        = plog::debug; // Set to debug to see the debug messages, info messages
    std::string csvLogFile = "/tmp/imageRegistrationLog.csv"; // Set NULL to disable
    bool logToConsole = true;
    util::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
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
    LOGD << "Parsed arguments, entering main function.";
    std::shared_ptr<io::Chunk2DReaderI<float>> denSliceReader
        = std::make_shared<io::DenChunk2DReader<float>>(a.input_file);
    int dimx = denSliceReader->dimx();
    int dimy = denSliceReader->dimy();
    int dimz = denSliceReader->count();
    LOGD << io::xprintf("The file has dimensions (x,y,z)=(%d, %d, %d)", dimx, dimy, dimz);
    std::vector<int> framesToProcess = processResultingFrames(a.frames, dimz);
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
    std::shared_ptr<io::AsyncImageWritterI<float>> imagesWritter
        = std::make_shared<io::DenAsyncWritter<float>>(a.output_file, dimx, dimy,
                                                       framesToOutput.size());
    for(int i = 0; i != framesToOutput.size(); i++)
    {
        // Try asynchronous calls
        threadpool->push(writeFrame, framesToOutput[i], denSliceReader, i, imagesWritter);
    }
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

    LOGD << io::xprintf("Input file is %s and output file is %s.", input_file.c_str(),
                        output_file.c_str());
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
            return -1;
        }
    }
    return 0;
}
