// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <algorithm>
#include <cstdlib>
#include <ctype.h>
#include <iostream>
#include <random>
#include <regex>
#include <string>

// External libraries
#include "CLI/CLI.hpp" //Command line parser

// Internal libraries
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DBufferedWritter.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/KCTException.hpp"
#include "PROG/Program.hpp"
#include "PROG/ThreadPool.hpp"
#include "rawop.h"
#include "stringFormatter.h"

using namespace KCT;
using namespace KCT::util;

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

class Args : public ArgumentsThreading, public ArgumentsForce
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsThreading(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName){};

    uint32_t dimx, dimy, dimz;
    uint64_t elementByteSize;
    std::string type = "FLOAT32";
    io::DenSupportedType dataType = io::DenSupportedType::FLOAT32;
    double value = 0.0;
    bool noise = false;
    std::string outputFile;
};

void Args::defineArguments()
{
    cliApp->add_option("-t,--type", type,
                       "Type of the base data unit in the DEN file, might be float, double or "
                       "uint16_t, default is float.");
    cliApp->add_option("--value", value, io::xprintf("Default value, defaults to %f", value));
    cliApp->add_flag("--noise", noise, io::xprintf("Pseudorandom noise from [0,1)"));
    cliApp->add_option("dimx", dimx, "X dimension.")->required();
    cliApp->add_option("dimy", dimy, "Y dimension.")->required();
    cliApp->add_option("dimz", dimz, "Z dimension.")->required();
    cliApp->add_option("output_den_file", outputFile, "File in a DEN format to output.")
        ->required();
    addThreadingArgs();
    addForceArgs();
}

int Args::postParse()
{
    // If force is not set, then check if output file does not exist
    if(!force)
    {
        if(io::pathExists(outputFile))
        {
            std::string msg = "Error: output file already exists, use --force to force overwrite.";
            LOGE << msg;
            return -1;
        }
    }
    if(type == "FLOAT32")
    {
        dataType = io::DenSupportedType::FLOAT32;
        LOGD << io::xprintf(
            "Creating file %s with data type float and dimensions (x,y,z) = (%d, %d, %d).",
            outputFile.c_str(), dimx, dimy, dimz);
    } else if(type == "FLOAT64")
    {
        dataType = io::DenSupportedType::FLOAT64;
        LOGD << io::xprintf(
            "Creating file %s with data type double and dimensions (x,y,z) = (%d, %d, %d).",
            outputFile.c_str(), dimx, dimy, dimz);
    } else if(type == "UINT16")
    {
        dataType = io::DenSupportedType::UINT16;
        LOGD << io::xprintf(
            "Creating file %s with data type uint16_t and dimensions (x,y,z) = (%d, %d, %d).",
            outputFile.c_str(), dimx, dimy, dimz);
    } else
    {
        std::string err
            = io::xprintf("Unrecognized data type %s, for help run dentk-empty -h.", type.c_str());
        LOGE << err;
        return -1;
    }
    return 0;
}

template <typename T>
void writeFrame(std::shared_ptr<typename TP<T>::ThreadInfo> threadInfo,
                std::shared_ptr<io::BufferedFrame2D<T>> f,
                uint32_t k)
{
    WRITERPTR<T> writer = threadInfo->worker;
    writer->writeBufferedFrame(*f, k);
}

template <typename T>
void writeValue(std::shared_ptr<typename TP<T>::ThreadInfo> threadInfo,
                T value,
                uint32_t dimx,
                uint32_t dimy,
                uint32_t k)
{
    WRITERPTR<T> writer = threadInfo->worker;
    io::BufferedFrame2D<T> f(value, dimx, dimy);
    writer->writeBufferedFrame(f, k);
}

template <typename T>
void createConstantDEN(
    std::string fileName, uint32_t sizex, uint32_t sizey, uint32_t K, T value, uint32_t threads = 0)
{
    if(threads > K)
    {
        threads = K;
    }
    if(K == 0)
    {
        return;
    }
    uint64_t frameByteSize = static_cast<uint64_t>(sizex) * sizey * sizeof(T);
    uint64_t writerBufferSize = std::min(K, 5u) * frameByteSize;
    std::shared_ptr<io::BufferedFrame2D<T>> f
        = std::make_shared<io::BufferedFrame2D<T>>(value, sizex, sizey);
    TPPTR<T> threadpool = nullptr;
    std::shared_ptr<typename TP<T>::ThreadInfo> thread_info = nullptr;
    if(threads > 0)
    {
        WRITERPTR<T> wp = nullptr;
        std::vector<WRITERPTR<T>> workers;
        uint32_t workerCount
            = std::min(threads, 10u); //Maximum 10 workers that will be shared between threads
        uint32_t divisionBoundaries = (threads + workerCount - 1) / workerCount;
        for(uint32_t i = 0; i < threads; i++)
        {
            if(i % divisionBoundaries == 0 || wp == nullptr)
            {
                wp = std::make_shared<WRITER<T>>(fileName, writerBufferSize);
            }
            workers.push_back(wp);
        }
        threadpool = std::make_shared<TP<T>>(threads, workers);
    } else
    {
        //Optimalized buffer size
        WRITERPTR<T> singleThreadWritter
            = std::make_shared<WRITER<T>>(fileName, std::min(K, 10u) * frameByteSize);
        thread_info
            = std::make_shared<typename TP<T>::ThreadInfo>(TPINFO<T>{ 0, 0, singleThreadWritter });
    }
    for(uint64_t k = 0; k < K; k++)
    {
        if(threadpool != nullptr)
        {
            /*threadpool->submit(writeFrame<T>, f, k);*/
            threadpool->submit(writeValue<T>, value, sizex, sizey, k);
        } else
        {
            writeValue<T>(thread_info, value, sizex, sizey, k);
        }
    }
    if(threadpool != nullptr)
    {
        threadpool->waitAll();
        threadpool = nullptr;
    }
}

template <typename T>
void createNoisyDEN(std::string fileName,
                    uint32_t sizex,
                    uint32_t sizey,
                    uint32_t K,
                    std::mt19937 gen,
                    T from = 0.0,
                    T to = 1.0,
                    uint32_t threads = 0)
{
    using namespace KCT;
    std::uniform_real_distribution<T> dis(from, to);
    uint64_t frameSize = static_cast<uint64_t>(sizex) * static_cast<uint64_t>(sizey);
    uint64_t frameByteSize = frameSize * sizeof(T);
    std::shared_ptr<io::BufferedFrame2D<T>> f;
    TPPTR<T> threadpool = nullptr;
    std::shared_ptr<typename TP<T>::ThreadInfo> thread_info = nullptr;
    if(threads > 0)
    {
        uint32_t workerCount
            = std::min(threads, 10u); //Maximum 10 workers that will be shared between threads
        uint32_t divisionBoundaries = (threads + workerCount - 1) / workerCount;
        WRITERPTR<T> wp = nullptr;
        std::vector<WRITERPTR<T>> workers;
        for(uint32_t i = 0; i < threads; i++)
        {
            if(i % divisionBoundaries == 0 || wp == nullptr)
            {
                //wp = std::make_shared<WRITER<T>>(ARG.output_file);
                //Optimalized buffer size
                wp = std::make_shared<WRITER<T>>(fileName, 5 * frameByteSize);
            }
            workers.push_back(wp);
        }
        threadpool = std::make_shared<TP<T>>(threads, workers);
    } else
    {
        WRITERPTR<T> singleThreadWritter
            = std::make_shared<WRITER<T>>(fileName, 10 * frameByteSize);
        thread_info
            = std::make_shared<typename TP<T>::ThreadInfo>(TPINFO<T>{ 0, 0, singleThreadWritter });
    }
    for(uint64_t k = 0; k < K; k++)
    {
        //I can not reuse the same frame, because the noise shall be different for each frame
        f = std::make_shared<io::BufferedFrame2D<T>>(T(0), sizex, sizey);
        T* f_array = f->getDataPointer();
        for(uint64_t i = 0; i < frameSize; i++)
        {
            f_array[i] = dis(gen);
        }
        if(threadpool != nullptr)
        {
            threadpool->submit(writeFrame<T>, f, k);
        } else
        {
            writeFrame<T>(thread_info, f, k);
        }
    }
    if(threadpool != nullptr)
    {
        threadpool->waitAll();
        threadpool = nullptr;
    }
}

template <typename T>
void process(Args& ARG)
{
    if(ARG.noise && ARG.dataType == io::DenSupportedType::UINT16)
    {
        KCTERR("Noise for uint16_t not implemented");
    }
    io::DenFileInfo::createEmpty3DDenFile(ARG.outputFile, ARG.dataType, ARG.dimx, ARG.dimy,
                                          ARG.dimz);
    if(ARG.noise)
    {
        std::mt19937 gen(0);
        createNoisyDEN<T>(ARG.outputFile, ARG.dimx, ARG.dimy, ARG.dimz, gen, ARG.threads);
    } else
    {
        createConstantDEN<T>(ARG.outputFile, ARG.dimx, ARG.dimy, ARG.dimz,
                             static_cast<T>(ARG.value), ARG.threads);
    }
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    const std::string prgInfo = "Create a DEN file with constant or noisy data.";
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
    // After init parsing arguments

    switch(ARG.dataType)
    {
    case io::DenSupportedType::UINT16:
        if(ARG.noise)
        {
            //createNoisyDEN<uint16_t> triggers a compile error due to std::uniform_real_distribution<uint16_t>
            KCTERR("Noise for uint16_t not implemented");
        } else
        {
            io::DenFileInfo::createEmpty3DDenFile(ARG.outputFile, ARG.dataType, ARG.dimx, ARG.dimy,
                                                  ARG.dimz);
            createConstantDEN<uint16_t>(ARG.outputFile, ARG.dimx, ARG.dimy, ARG.dimz,
                                        static_cast<uint16_t>(ARG.value), ARG.threads);
        }
        break;
    case io::DenSupportedType::FLOAT32:
        process<float>(ARG);
        break;
    case io::DenSupportedType::FLOAT64:
        process<double>(ARG);
        break;
    default:
        std::string errMsg = io::xprintf("Unsupported data type %s.",
                                         io::DenSupportedTypeToString(ARG.dataType).c_str());
        KCTERR(errMsg);
    }
    PRG.endLog(true);
    return 0;
}
