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
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/Program.hpp"
#include "PROG/ThreadPool.hpp"
#include "littleEndianAlignment.h"
#include "rawop.h"

using namespace KCT;
using namespace KCT::util;

// class declarations
class Args : public ArgumentsForce, public ArgumentsThreading
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName)
        , ArgumentsThreading(argc, argv, prgName){};
    int parseArguments(int argc, char* argv[]);
    std::string input_file;
    std::string output_file;
    bool outputFileExists = false;
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

template <typename T>
void transposeArray(const T* input, T* transposed, uint32_t dimx, uint32_t dimy)
{
    for(uint64_t j = 0; j < dimy; j++)
    {
        for(uint64_t i = 0; i < dimx; i++)
        {
            transposed[i * dimy + j] = input[j * dimx + i];
        }
    }
}

// Function that will be submitted to the thread pool
template <typename T>
void transposeFrameTask(std::shared_ptr<typename TP<T>::ThreadInfo> thread_info,
                        READERPTR<T> inputReader,
                        uint64_t frameIndex)
{
    WRITERPTR<T> writer = thread_info->worker;
    uint64_t frameSize = inputReader->getFrameSize();
    uint32_t dimx = inputReader->dimx();
    uint32_t dimy = inputReader->dimy();
    std::shared_ptr<io::BufferedFrame2D<T>> f = inputReader->readBufferedFrame(frameIndex);
    T* input = f->getDataPointer();
    T* trasposed = new T[frameSize];
    transposeArray(input, trasposed, dimx, dimy);
    writer->writeBuffer(trasposed, frameIndex);
    delete[] trasposed;
}

// Main processing function
template <typename T>
int process(Args& ARG, io::DenFileInfo& input_inf)
{
    io::DenSupportedType dataType = input_inf.getElementType();
    uint16_t dimCount = input_inf.getDimCount();
    uint64_t K = input_inf.getFrameCount();
    std::vector<uint32_t> dim;
    // Transposed specification
    uint32_t dimx = input_inf.dim(0);
    uint32_t dimy = input_inf.dim(1);
    uint64_t frameByteSize = input_inf.getFrameByteSize();
    dim.push_back(dimy);
    dim.push_back(dimx);
    for(uint16_t i = 2; i < dimCount; i++)
    {
        dim.push_back(input_inf.dim(i));
    }
    io::DenFileInfo::createEmptyDenFile(ARG.output_file, dataType, dimCount, dim.data(),
                                        input_inf.hasXMajorAlignment());
    LOGI << io::xprintf("Output file %s created to store transposed result.",
                        ARG.output_file.c_str());

    READERPTR<T> inputReader = std::make_shared<READER<T>>(ARG.input_file, ARG.threads);
    TPPTR<T> threadpool = nullptr;
    std::shared_ptr<typename TP<T>::ThreadInfo> thread_info = nullptr;
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
                wp = std::make_shared<WRITER<T>>(ARG.output_file, 5 * frameByteSize);
            }
            workers.push_back(wp);
        }
        threadpool = std::make_shared<TP<T>>(ARG.threads, workers);
    } else
    {
        WRITERPTR<T> singleThreadWritter
            = std::make_shared<WRITER<T>>(ARG.output_file, 5 * frameByteSize);
        thread_info
            = std::make_shared<typename TP<T>::ThreadInfo>(TPINFO<T>{ 0, 0, singleThreadWritter });
    }

    for(uint64_t k = 0; k < K; k++)
    {
        if(threadpool != nullptr)
        {
            threadpool->submit(transposeFrameTask<T>, inputReader, k);
        } else
        {
            transposeFrameTask<T>(thread_info, inputReader, k);
        }
    }

    if(threadpool != nullptr)
    {
        threadpool->waitAll();
        threadpool = nullptr;
    }
    return 0;
}

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
}

void Args::defineArguments()
{
    cliApp->add_option("input_den_file", input_file, "File that will be transposed.")
        ->check(CLI::ExistingFile)
        ->required();
    cliApp->add_option("output_den_file", output_file, "Transposed file in a DEN format to output.")
        ->required();
    addForceArgs();
    addThreadingArgs();
}

int Args::postParse()
{
    bool removeIfExists = true;
    int existFlag = handleFileExistence(output_file, force, removeIfExists);
    if(existFlag != 0)
    {
        std::string msg
            = io::xprintf("Error: output file %s already exists, use --force to force overwrite.",
                          output_file.c_str());
        LOGE << msg;
        return 1;
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
