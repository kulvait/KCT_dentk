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
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/Program.hpp"
#include "PROG/parseArgs.h"

using namespace CTL;
using namespace CTL::util;

struct Args : public ArgumentsFramespec, public ArgumentsForce, public ArgumentsThreading
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsFramespec(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName)
        , ArgumentsThreading(argc, argv, prgName){};
    bool interlacing = false;
    std::vector<std::string> inputFiles;
    std::string outputFile;
};

/**Argument parsing
 *
 */
void Args::defineArguments()
{
    cliApp->add_option("output_den_file", outputFile, "File in a DEN format to output.")
        ->required();
    cliApp
        ->add_option("input_den_file1 ... input_den_filen output_den_file", inputFiles,
                     "Files in a DEN format to process. These files should have the same x,y and z "
                     "dimension as the first file of input.")
        ->required()
        ->check(CLI::ExistingFile);
    addForceArgs();
    addFramespecArgs();
    addThreadingArgs();
    cliApp->add_flag("-i,--interlacing", interlacing,
                     "First n frames in the output will be from the first n DEN files.");
}

int Args::postParse()
{
    // LOGD << io::xprintf("Optional parameters: interlacing=%d, frames=%s, eachkth=%d, threads=%d "
    //                    "and %d input files.",
    //                    interlacing, frameSpecs.c_str(), eachkth, threads, inputFiles.size());
    // If force is not set, then check if output file does not exist
    if(!force)
    {
        if(io::pathExists(outputFile))
        {
            std::string msg = "Error: output file already exists, use --force to force overwrite.";
            LOGE << msg;
            return 1;
        }
    }
    // How many projection matrices is there in total
    io::DenFileInfo di(inputFiles[0]);
    io::DenSupportedType dataType = di.getDataType();
    uint32_t dimx = di.dimx();
    uint32_t dimy = di.dimy();
    uint32_t dimz = di.dimz();
    std::string err;
    for(std::string const& f : inputFiles)
    {
        io::DenFileInfo df(f);
        if(df.getDataType() != dataType)
        {
            err = io::xprintf("File %s and %s are of different element types.",
                              inputFiles[0].c_str(), f.c_str());
            LOGE << err;
            return -1;
        }
        if(df.dimx() != dimx || df.dimy() != dimy)
        {
            err = io::xprintf("Files %s and %s do not have the same x and y dimensions.",
                              inputFiles[0].c_str(), f.c_str());
            LOGE << err;
            return -1;
        }
        if(interlacing && df.dimz() != dimz)
        {
            err = io::xprintf("Files %s and %s do not have the same z dimensions. Since "
                              "interlacing, eachkth or frame specification was given, this is "
                              "important.",
                              inputFiles[0].c_str(), f.c_str());
            LOGE << err;
            return -1;
        }
    }
    return 0;
}

template <typename T>
void writeFrame(int id,
                uint64_t fromId,
                std::shared_ptr<io::Frame2DReaderI<T>> denSliceReader,
                uint64_t toId,
                std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter)
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

template <typename T>
void mergeFiles(Args a)
{
    std::vector<std::shared_ptr<io::Frame2DReaderI<T>>> denSliceReaders;
    ctpl::thread_pool* threadpool = nullptr;
    if(a.threads > 0)
    {
        threadpool = new ctpl::thread_pool(a.threads);
    }
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter;
    LOGD << io::xprintf("Will merge file %s from specified files.", a.outputFile.c_str());
    if(a.interlacing)
    {
        // Each file must have the same dimensions
        io::DenFileInfo INF(a.inputFiles[0]);
        uint64_t dimx = INF.dimx();
        uint64_t dimy = INF.dimy();
        uint64_t dimz = INF.dimz();
        a.fillFramesVector(dimz);
        for(const std::string& f : a.inputFiles)
        {
            denSliceReaders.push_back(std::make_shared<io::DenFrame2DReader<T>>(f));
        }
        uint64_t i;
        imagesWritter = std::make_shared<io::DenAsyncFrame2DWritter<T>>(
            a.outputFile, dimx, dimy, a.frames.size() * a.inputFiles.size());
        for(std::size_t ind = 0; ind != a.frames.size(); ind++)
        {
            for(std::size_t j = 0; j != a.inputFiles.size(); j++)
            {
                i = a.frames[ind];
                if(threadpool != nullptr)
                {
                    threadpool->push(writeFrame<T>, i, denSliceReaders[j],
                                     ind * a.inputFiles.size() + j, imagesWritter);
                } else
                {
                    writeFrame<T>(0, i, denSliceReaders[j], ind * a.inputFiles.size() + j,
                                  imagesWritter);
                }
            }
        }
    } else
    {
        uint64_t dimx, dimy, dimz = 0;
        uint32_t local_dimz;
        std::vector<std::vector<int>> frameSpecifications;
        for(const std::string& f : a.inputFiles)
        {
            io::DenFileInfo INF(f);
            local_dimz = INF.dimz();
            denSliceReaders.push_back(std::make_shared<io::DenFrame2DReader<T>>(f));
            a.fillFramesVector(local_dimz);
            frameSpecifications.push_back(a.frames);
            dimz += a.frames.size();
        }
        dimx = denSliceReaders[0]->dimx();
        dimy = denSliceReaders[0]->dimy();
        imagesWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<T>>(a.outputFile, dimx, dimy, dimz);
        uint64_t i;
        uint64_t writeOffset = 0;
        for(std::size_t j = 0; j != a.inputFiles.size(); j++)
        {
            std::vector<int> frameSpecification = frameSpecifications[j];
            for(std::size_t ind = 0; ind != frameSpecification.size(); ind++)
            {
                i = frameSpecification[ind];
                if(threadpool != nullptr)
                {
                    threadpool->push(writeFrame<T>, i, denSliceReaders[j], writeOffset + ind,
                                     imagesWritter);
                } else
                {
                    writeFrame<T>(0, i, denSliceReaders[j], writeOffset + ind, imagesWritter);
                }
            }
            writeOffset += frameSpecification.size();
        }
    }
    if(threadpool != nullptr)
    {
        threadpool->stop(true);
        delete threadpool;
    }
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    const std::string prgInfo = "Merge multiple DEN files together.";
    Args ARG(argc, argv, prgInfo);
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    // Frames to process
    io::DenFileInfo inf(ARG.inputFiles[0]);
    io::DenSupportedType dataType = inf.getDataType();
    switch(dataType)
    {
    case io::DenSupportedType::uint16_t_:
    {
        mergeFiles<uint16_t>(ARG);
        break;
    }
    case io::DenSupportedType::float_:
    {
        mergeFiles<float>(ARG);
        break;
    }
    case io::DenSupportedType::double_:
    {
        mergeFiles<double>(ARG);
        break;
    }
    default:
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(dataType));
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
    }
    PRG.endLog();
}
