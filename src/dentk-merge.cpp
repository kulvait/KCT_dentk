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

void writeFrame(int id,
                int fromId,
                std::shared_ptr<io::Frame2DReaderI<float>> denSliceReader,
                int toId,
                std::shared_ptr<io::AsyncFrame2DWritterI<float>> imagesWritter)
{
    // LOGD << io::xprintf(
    //    "Writing %d th slice of file %s to %d th slice of file %s.", fromId,
    //    (std::dynamic_pointer_cast<io::DenFrame2DReader<float>>(denSliceReader))
    //        ->getFileName()
    //        .c_str(),
    //    toId,
    //    (std::dynamic_pointer_cast<io::DenAsyncFrame2DWritter<float>>(imagesWritter))
    //        ->getFileName()
    //        .c_str());
    imagesWritter->writeFrame(*(denSliceReader->readFrame(fromId)), toId);
}

int main(int argc, char* argv[])
{
    plog::Severity verbosityLevel
        = plog::debug; // Set to debug to see the debug messages, info messages
    std::string csvLogFile = "/tmp/dentk-merge.csv"; // Set NULL to disable
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    LOGI << "dentk-merge";
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
                   "also individual comma separated frames i.e. 1,8,9. Order does matter. Accepts "
                   "end literal that means total number of slices of the input.");
    app.add_option("-k,--each-kth", a_eachkth,
                   "Process only each k-th frame specified by k to output. The frames to output "
                   "are then 1st specified, 1+kN, N=1...\\infty if such frame exists. Parameter k "
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
    std::vector<std::shared_ptr<io::Frame2DReaderI<float>>> denSliceReaders;
    LOGD << io::xprintf("Will merge file %s from specified files.", a_outputDen.c_str());
    for(int i = 0; i != a_inputDenFiles.size(); i++)
    {
        denSliceReaders.push_back(
            std::make_shared<io::DenFrame2DReader<float>>(a_inputDenFiles[i]));
    }
    int dimx = denSliceReaders[0]->dimx();
    int dimy = denSliceReaders[0]->dimy();
    int dimz = denSliceReaders[0]->dimz();
    // LOGD << io::xprintf("The file %s has dimensions (x,y,z)=(%d, %d, %d)",
    //                    a_inputDenFiles[0].c_str(), dimx, dimy, dimz);
    std::vector<int> framesToProcess = util::processFramesSpecification(a_frameSpecs, dimz);
    std::vector<int> framesToOutput;
    for(int i = 0; i != framesToProcess.size(); i++)
    {
        if(i % a_eachkth == 0)
        {
            framesToOutput.push_back(framesToProcess[i]);
        }
    }
    ctpl::thread_pool* threadpool = new ctpl::thread_pool(a_threads);
    LOGD << io::xprintf("Size of framesToOutput is %d.", framesToOutput.size());
    std::shared_ptr<io::AsyncFrame2DWritterI<float>> imagesWritter
        = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
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
