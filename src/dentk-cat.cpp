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

// Internal libraries
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DBufferedWritter.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/Program.hpp"

using namespace KCT;
using namespace KCT::util;

// Function declarations (definition at the end of the file)

// class declarations
struct Args : public ArgumentsFramespec, public ArgumentsThreading, public ArgumentsForce
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsFramespec(argc, argv, prgName)
        , ArgumentsThreading(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName){};
    std::string input_file;
    std::string output_file;
};

template <typename T>
void process(Args& ARG, io::DenFileInfo& di)
{
    std::string outputFile = ARG.output_file;
    uint64_t frameSize = di.getFrameSize();
    uint64_t frameByteSize = frameSize * sizeof(T);
    uint64_t frameCount = ARG.frames.size();
    uint64_t num_threads
        = std::min(static_cast<uint64_t>(ARG.threads), static_cast<uint64_t>(ARG.frames.size()));
    num_threads = std::max(num_threads, 1lu);
    std::shared_ptr<io::DenFrame2DReader<T>> inputReader
        = std::make_shared<io::DenFrame2DReader<T>>(ARG.input_file, num_threads);
    uint64_t frames_per_thread = (frameCount + num_threads - 1) / num_threads; // ceil
    std::vector<std::thread> threads;
    for(uint64_t t = 0; t < num_threads; t++)
    {
        uint64_t start_frame = t * frames_per_thread;
        uint64_t end_frame
            = std::min((t + 1) * frames_per_thread, static_cast<uint64_t>(frameCount));
        if(start_frame >= end_frame)
        {
            break;
        }
        uint64_t bufferSize
            = std::min(static_cast<uint64_t>(10u), end_frame - start_frame) * frameByteSize;
        threads.emplace_back([&inputReader, &ARG, bufferSize, outputFile, start_frame, end_frame]() {
            std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>> outputWriter
                = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(outputFile, bufferSize);
            std::shared_ptr<io::BufferedFrame2D<T>> f;
            for(uint32_t k = start_frame; k < end_frame; k++)
            {
                uint32_t IND = ARG.frames[k];
                f = inputReader->readBufferedFrame(IND);
                outputWriter->writeBufferedFrame(*f, k);
            }
        });
    }
    for(auto& t : threads)
    {
        t.join();
    }
}

int main(int argc, char** argv)
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
    io::DenSupportedType dataType = di.getElementType();
    uint32_t dimx = di.dimx();
    uint32_t dimy = di.dimy();
    uint32_t dimz = ARG.frames.size();

    io::DenFileInfo::createEmpty3DDenFile(ARG.output_file, dataType, dimx, dimy, dimz);
    LOGI << io::xprintf("Output file %s created to store swap result.", ARG.output_file.c_str());
    switch(dataType)
    {
    case io::DenSupportedType::UINT16:
        process<uint16_t>(ARG, di);
        break;
    case io::DenSupportedType::FLOAT32:
        process<float>(ARG, di);
        break;
    case io::DenSupportedType::FLOAT64:
        process<double>(ARG, di);
        break;
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
    cliApp->add_option("input_den_file", input_file, "File in a DEN format to process.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("output_den_file", output_file, "File in a DEN format to output.")
        ->required();
    addFramespecArgs();
    addForceArgs();
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
    bool removeIfExists = true;
    int existFlag = handleFileExistence(output_file, force, removeIfExists);
    if(existFlag != 0)
    {
        return 1;
    }
    io::DenFileInfo inf(input_file);
    if(inf.getDimCount() != 3)
    {
        std::string ERR
            = io::xprintf("The file %s has %d dimensions!", input_file.c_str(), inf.getDimCount());
        LOGE << ERR;
        return 1;
    }
    fillFramesVector(inf.dimz());
    return 0;
}
