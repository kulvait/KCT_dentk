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
#include "ftpl.h" //Threadpool

// Internal libraries
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DBufferedWritter.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/Program.hpp"
#include "PROG/ThreadPool.hpp"
#include "littleEndianAlignment.h"
#include "rawop.h"

using namespace KCT;
using namespace KCT::util;

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

template <typename T>
using READER = io::DenFrame2DReader<T>;

template <typename T>
using READERPTR = std::shared_ptr<READER<T>>;

template <typename T>
using WRITER = io::DenAsyncFrame2DBufferedWritter<T>;

template <typename T>
using WRITERPTR = std::shared_ptr<WRITER<T>>;

template <typename T>
using TP = io::ThreadPool<WRITER<T>>;

template <typename T>
using TPPTR = std::shared_ptr<TP<T>>;

template <typename T>
using TPINFO = typename TP<T>::ThreadInfo;

template <typename T>
using TPINFOPTR = std::shared_ptr<TPINFO<T>>;

// Function that will be submitted to the thread pool
template <typename T>
void writeFrameTask(std::shared_ptr<typename TP<T>::ThreadInfo> dummyInfo,
                    READERPTR<T> inputReader,
                    uint64_t readerIndex,
                    uint64_t writerIndex)
{
    WRITERPTR<T> writer = dummyInfo->worker;
    std::shared_ptr<io::BufferedFrame2D<T>> f = inputReader->readBufferedFrame(readerIndex);
    writer->writeBufferedFrame(*f, writerIndex);
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
void mergeFiles(Args ARG)
{
    // Fill frameReaders and get actual dimensions of the output file
    io::DenFileInfo INF(ARG.inputFiles[0]);
    io::DenSupportedType dataType = INF.getElementType();
    uint64_t frameByteSize = INF.getFrameByteSize();
    uint64_t dimx = INF.dimx();
    uint64_t dimy = INF.dimy();
    uint64_t dimz = 0;
    uint64_t inputFileCount = ARG.inputFiles.size();
    //Only for else
    std::vector<READERPTR<T>> frameReaders;
    std::vector<std::vector<uint64_t>> frameSpecifications;
    if(ARG.interlacing)
    {
        // Each file must have the same dimensions
        dimz = INF.dimz();
        ARG.fillFramesVector(dimz);
        for(const std::string& f : ARG.inputFiles)
        {
            frameReaders.push_back(std::make_shared<READER<T>>(f, ARG.threads));
        }
        dimz = ARG.frames.size() * ARG.inputFiles.size();
    } else
    {
        uint64_t local_dimz;
        for(const std::string& f : ARG.inputFiles)
        {
            READERPTR<T> a = std::make_shared<READER<T>>(f, ARG.threads);
            local_dimz = a->getFrameCount();
            frameReaders.push_back(a);
            ARG.fillFramesVector(local_dimz);
            frameSpecifications.push_back(ARG.frames);
            dimz += ARG.frames.size();
        }
    }
    io::DenFileInfo::createEmpty3DDenFile(ARG.outputFile, dataType, dimx, dimy, dimz);

    TPPTR<T> threadpool = nullptr;
    std::shared_ptr<typename TP<T>::ThreadInfo> dummyInfo = nullptr;
    if(ARG.threads > 0)
    {
        uint32_t workerCount
            = std::min(ARG.threads, 10u); //Maximum 10 workers that will be shared between threads
        uint32_t divisionBoundaries = (ARG.threads + workerCount - 1) / workerCount;
        WRITERPTR<T> wp = nullptr;
        std::vector<WRITERPTR<T>> workers;
        for(uint32_t i = 0; i < ARG.threads; i++)
        {
            if(i % divisionBoundaries == 0 || wp == nullptr)
            {
                //wp = std::make_shared<WRITER<T>>(ARG.output_file);
                //Optimalized buffer size
                wp = std::make_shared<WRITER<T>>(ARG.outputFile, 5 * frameByteSize);
            }
            workers.push_back(wp);
        }
        threadpool = std::make_shared<TP<T>>(ARG.threads, workers);
    } else
    {
        WRITERPTR<T> singleThreadWritter = std::make_shared<WRITER<T>>(ARG.outputFile);
        dummyInfo
            = std::make_shared<typename TP<T>::ThreadInfo>(TPINFO<T>{ 0, 0, singleThreadWritter });
    }

    LOGD << io::xprintf("Will merge file %s from specified files.", ARG.outputFile.c_str());
    if(ARG.interlacing)
    {
        uint64_t i;
        for(std::size_t ind = 0; ind != ARG.frames.size(); ind++)
        {
            for(std::size_t j = 0; j != inputFileCount; j++)
            {
                i = ARG.frames[ind];
                if(threadpool != nullptr)
                {
                    threadpool->submit(writeFrameTask<T>, frameReaders[j], i,
                                       ind * inputFileCount + j);
                } else
                {
                    writeFrameTask<T>(dummyInfo, frameReaders[j], i, ind * inputFileCount + j);
                }
            }
        }
    } else
    {
        uint64_t i;
        uint64_t writeOffset = 0;
        for(std::size_t j = 0; j != inputFileCount; j++)
        {
            std::vector<uint64_t> frameSpecification = frameSpecifications[j];
            for(std::size_t ind = 0; ind != frameSpecification.size(); ind++)
            {
                i = frameSpecification[ind];
                if(threadpool != nullptr)
                {
                    threadpool->submit(writeFrameTask<T>, frameReaders[j], i, writeOffset + ind);
                } else
                {
                    writeFrameTask<T>(dummyInfo, frameReaders[j], i, writeOffset + ind);
                }
            }
            writeOffset += frameSpecification.size();
        }
    }
    if(threadpool != nullptr)
    {
        threadpool->waitAll();
        threadpool = nullptr;
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
    PRG.startLog(true);
    // Frames to process
    io::DenFileInfo inf(ARG.inputFiles[0]);
    io::DenSupportedType dataType = inf.getElementType();
    switch(dataType)
    {
    case io::DenSupportedType::UINT16: {
        mergeFiles<uint16_t>(ARG);
        break;
    }
    case io::DenSupportedType::FLOAT32: {
        mergeFiles<float>(ARG);
        break;
    }
    case io::DenSupportedType::FLOAT64: {
        mergeFiles<double>(ARG);
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
    // If force is not set, then check if output file does not exist
    bool removeIfExists = true;
    int existFlag = handleFileExistence(outputFile, force, removeIfExists);
    if(existFlag != 0)
    {
        std::string msg
            = io::xprintf("Error: output file %s already exists, use --force to force overwrite.",
                          outputFile.c_str());
        LOGE << msg;
        return 1;
    }
    // How many projection matrices is there in total
    io::DenFileInfo di(inputFiles[0]);
    io::DenSupportedType dataType = di.getElementType();
    uint32_t dimx = di.dimx();
    uint32_t dimy = di.dimy();
    uint32_t dimz = di.dimz();
    std::string err;
    for(std::string const& f : inputFiles)
    {
        io::DenFileInfo df(f);
        if(df.getElementType() != dataType)
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
    // LOGD << io::xprintf("Optional parameters: interlacing=%d, frames=%s, eachkth=%d, threads=%d "
    //                    "and %d input files.",
    //                    interlacing, frameSpecs.c_str(), eachkth, threads, inputFiles.size());
    return 0;
}
