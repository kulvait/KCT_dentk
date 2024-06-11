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
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenAsyncFrame2DBufferedWritter.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/KCTException.hpp"
#include "PROG/Program.hpp"
#include "ftpl.h"
#include "rawop.h"
#include "stringFormatter.h"

using namespace KCT;
using namespace KCT::util;

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
    std::string type = "float";
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
    if(type == "float")
    {
        elementByteSize = 4;
        LOGD << io::xprintf(
            "Creating file %s with data type float and dimensions (x,y,z) = (%d, %d, %d).",
            outputFile.c_str(), dimx, dimy, dimz);
    } else if(type == "double")
    {
        elementByteSize = 8;
        LOGD << io::xprintf(
            "Creating file %s with data type double and dimensions (x,y,z) = (%d, %d, %d).",
            outputFile.c_str(), dimx, dimy, dimz);
    } else if(type == "uint16_t")
    {
        elementByteSize = 2;
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
void writeFrame(int _FTPLID,
                uint32_t k,
                std::shared_ptr<io::BufferedFrame2D<T>> f,
                std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>> dw)
{
    dw->writeFrame(*f, k);
}

template <typename T>
void createConstantDEN(std::string fileName,
                       uint32_t sizex,
                       uint32_t sizey,
                       uint32_t sizez,
                       T value,
                       uint32_t threads = 0)
{
    ftpl::thread_pool* threadpool = nullptr;
    if(threads > 0)
    {
        threadpool = new ftpl::thread_pool(threads);
    }
    std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>> dw
        = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(fileName, sizex, sizey, sizez);
    std::shared_ptr<io::BufferedFrame2D<T>> f
        = std::make_shared<io::BufferedFrame2D<T>>(value, sizex, sizey);
    const int dummy_FTPLID = 0;
    for(uint32_t k = 0; k != sizez; k++)
    {
        if(threadpool != nullptr)
        {
            threadpool->push(writeFrame<T>, k, f, dw);
        } else
        {
            writeFrame<T>(dummy_FTPLID, k, f, dw);
        }
    }

    if(threadpool != nullptr)
    {
        threadpool->stop(true);
        delete threadpool;
    }
}

template <typename T>
void createNoisyDEN(std::string fileName,
                    uint32_t sizex,
                    uint32_t sizey,
                    uint32_t sizez,
                    std::mt19937 gen,
                    T from = 0.0,
                    T to = 1.0)
{
    using namespace KCT;
    std::uniform_real_distribution<T> dis(from, to);
    io::DenAsyncFrame2DBufferedWritter<T> dw(fileName, sizex, sizey, sizez);
    io::BufferedFrame2D<T> f(0.0f, sizex, sizey);
    for(uint32_t k = 0; k != sizez; k++)
    {
        for(uint32_t i = 0; i != sizex; i++)
        {
            for(uint32_t j = 0; j != sizey; j++)
            {
                f.set(dis(gen), i, j);
            }
        }
        dw.writeFrame(f, k);
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
    if(ARG.elementByteSize == 2)
    {
        if(ARG.noise)
        {
            KCTERR("Noise for uint16_t not implemented");
        } else
        {
            createConstantDEN<uint16_t>(ARG.outputFile, ARG.dimx, ARG.dimy, ARG.dimz,
                                        (uint16_t)ARG.value, ARG.threads);
        }
    }
    if(ARG.elementByteSize == 4)
    {
        if(ARG.noise)
        {
            std::mt19937 gen(0);
            createNoisyDEN<float>(ARG.outputFile, ARG.dimx, ARG.dimy, ARG.dimz, gen);
        } else
        {
            createConstantDEN<float>(ARG.outputFile, ARG.dimx, ARG.dimy, ARG.dimz, (float)ARG.value,
                                     ARG.threads);
        }
    }
    if(ARG.elementByteSize == 8)
    {
        if(ARG.noise)
        {
            std::mt19937 gen(0);
            createNoisyDEN<double>(ARG.outputFile, ARG.dimx, ARG.dimy, ARG.dimz, gen);
        } else
        {
            createConstantDEN<double>(ARG.outputFile, ARG.dimx, ARG.dimy, ARG.dimz, ARG.value,
                                      ARG.threads);
        }
    }
    PRG.endLog(true);
    return 0;
}
