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
    LOGD << io::xprintf("Writting %d th slice of file %s to %d th slice of file %s.", fromId,
                        (std::dynamic_pointer_cast<io::DenChunk2DReader<float>>(denSliceReader))
                            ->getFileName()
                            .c_str(),
                        toId,
                        (std::dynamic_pointer_cast<io::DenAsyncWritter<float>>(imagesWritter))
                            ->getFileName()
                            .c_str());
    imagesWritter->writeSlice(*(denSliceReader->readSlice(fromId)), toId);
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
    bool a_interlacing = false;
    std::string a_frameSpecs = "";
    int a_eachkth = 1;
    int a_threads = 1;
    std::vector<std::string> a_inputDenFiles;
    std::string a_outputDen;
    CLI::App app{ "Merge multiple DEN files together." };
    app.add_flag("-i,--interlacing", a_interlacing,
                 "First n frames in the output will be from the first n DEN files.");
    app.add_option("-f,--frames", a_frameSpecs,
                   "Specify only particular frames to process. You can input range i.e. 0-20 or "
                   "also individual coma separated frames i.e. 1,8,9. Order does matter. Accepts "
                   "end literal that means total number of slices of the input.");
    app.add_option("-k,--each-kth", a_eachkth,
                   "Process only each k-th frame specified by k to output. The frames to output "
                   "are then 1st specified, 1+kN, N=1...\\infty if such frame exists. Parametter k "
                   "must be positive integer.")
        ->check(CLI::Range(1, 65535));
    app.add_option("-j,--threads", a_threads, "Number of extra threads that application can use.")
        ->check(CLI::Range(1, 65535));
    app.add_option("output_den_file", a_outputDen, "File in a DEN format to output.")
        ->required()
        ->check(CLI::NonexistentPath);
    app.add_option("input_den_file1 ... input_den_filen output_den_file", a_inputDenFiles,
                   "Files in a DEN format to process. These files should have the same x,y and z "
                   "dimension as the first file of input.")
        ->required()
        ->check(CLI::ExistingFile);
    CLI11_PARSE(app, argc, argv);
    LOGD << io::xprintf("Optional parameters: interlacing=%d, frames=%s, eachkth=%d, threads=%d "
                        "and %d input files.",
                        a_interlacing, a_frameSpecs.c_str(), a_eachkth, a_threads,
                        a_inputDenFiles.size());
    // Frames to process
    std::vector<std::shared_ptr<io::Chunk2DReaderI<float>>> denSliceReaders;
    for(int i = 0; i != a_inputDenFiles.size(); i++)
    {
        denSliceReaders.push_back(
            std::make_shared<io::DenChunk2DReader<float>>(a_inputDenFiles[i]));
    }
    int dimx = denSliceReaders[0]->dimx();
    int dimy = denSliceReaders[0]->dimy();
    int dimz = denSliceReaders[0]->count();
    LOGD << io::xprintf("The file %s has dimensions (x,y,z)=(%d, %d, %d)",
                        a_inputDenFiles[0].c_str(), dimx, dimy, dimz);
    std::vector<int> framesToProcess = processResultingFrames(a_frameSpecs, dimz);
    std::vector<int> framesToOutput;
    for(int i = 0; i != framesToProcess.size(); i++)
    {
        if(i % a_eachkth == 0)
        {
            framesToOutput.push_back(framesToProcess[i]);
        }
    }
    ctpl::thread_pool* threadpool = new ctpl::thread_pool(a_threads);
    std::shared_ptr<io::AsyncImageWritterI<float>> imagesWritter
        = std::make_shared<io::DenAsyncWritter<float>>(
            a_outputDen, dimx, dimy, a_inputDenFiles.size() * framesToOutput.size());
    int outputFiles = a_inputDenFiles.size();
    for(int i = 0; i != framesToOutput.size(); i++)
    {
        for(int j = 0; j != a_inputDenFiles.size(); j++)
        {
            if(a_interlacing)
                threadpool->push(writeFrame, framesToOutput[i], denSliceReaders[j],
                                 i * outputFiles + j, imagesWritter);
            else
                threadpool->push(writeFrame, framesToOutput[i], denSliceReaders[j], j * dimz + i,
                                 imagesWritter);
        }
    }
    delete threadpool;
}
