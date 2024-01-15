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
#include "CLI/CLI.hpp" //Command line parser
#include "ftpl.h" //Threadpool

// Internal libraries
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DBufferedWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/ArgumentsVerbose.hpp"
#include "PROG/Program.hpp"
#include "littleEndianAlignment.h"
#include "rawop.h"

using namespace KCT;
using namespace KCT::util;

// class declarations
class Args : public ArgumentsForce,
             public ArgumentsFramespec,
             public ArgumentsThreading,
             public ArgumentsVerbose
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName)
        , ArgumentsFramespec(argc, argv, prgName)
        , ArgumentsThreading(argc, argv, prgName)
        , ArgumentsVerbose(argc, argv, prgName){};
    int parseArguments(int argc, char* argv[]);
    std::string input_file;
    std::string output_file;
    uint32_t dimx, dimy;
    bool outputFileExists = false;
    bool flipRows = false;
    bool flipColumns = false;
};

void Args::defineArguments()
{
    cliApp->add_option("input_den_file", input_file, "File that will be transposed.")
        ->check(CLI::ExistingFile)
        ->required();
    cliApp->add_option("output_den_file", output_file, "Flipped file in a DEN format to output.")
        ->required();
    CLI::Option_group* fl_clg = cliApp->add_option_group("Operation", "Flip to perform.");
    fl_clg->add_flag("--flip-rows", flipRows, "Flip rows, that are sequences where y=const");
    fl_clg->add_flag("--flip-columns", flipColumns,
                     "Flip columns, that are sequences where x=const");
    fl_clg->require_option(1, 2);
    addForceArgs();
    addFramespecArgs();
    addThreadingArgs();
    addVerboseArgs();
}

int Args::postParse()
{
    int existFlag = handleFileExistence(output_file, force, input_file);
    if(existFlag == 1)
    {
        return 1;
    } else if(existFlag == -1)
    {
        outputFileExists = true;
    }
    io::DenFileInfo di(input_file);
    if(di.getDimCount() < 2)
    {
        std::string ERR = io::xprintf("The file %s has just %d<2 dimensions!", input_file.c_str(),
                                      di.getDimCount());
        LOGE << ERR;
        return 1;
    }
    dimx = di.dim(0);
    dimy = di.dim(1);
    fillFramesVector(di.getFrameCount());
    return 0;
}

template <typename T>
void processFrame(int _FTPLID,
                  Args ARG,
                  uint64_t k_in,
                  uint64_t k_out,
                  std::shared_ptr<io::DenFrame2DReader<T>>& fr,
                  std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>>& fw)
{
    std::shared_ptr<io::BufferedFrame2D<T>> A = fr->readBufferedFrame(k_in);
    io::BufferedFrame2D<T> X(T(0), ARG.dimx, ARG.dimy);
    T val;
    uint32_t i_out, j_out;
    for(uint32_t j = 0; j != ARG.dimy; j++)
    {
        if(ARG.flipColumns)
        {
            j_out = ARG.dimy - j - 1;
        } else
        {
            j_out = j;
        }
        for(uint32_t i = 0; i != ARG.dimx; i++)
        {
            if(ARG.flipRows)
            {
                i_out = ARG.dimx - i - 1;
            } else
            {
                i_out = i;
            }
            val = A->get(i, j);
            X.set(val, i_out, j_out);
        }
    }
    fw->writeFrame(X, k_out);
}

template <typename T>
void run(Args ARG)
{
    io::DenFileInfo di(ARG.input_file);
    uint16_t dimCount = di.getDimCount();
    std::vector<uint32_t> dim;
    for(uint16_t i = 0; i < dimCount; i++)
    {
        dim.push_back(di.dim(i));
    }
    std::shared_ptr<io::DenFrame2DReader<T>> fr
        = std::make_shared<io::DenFrame2DReader<T>>(ARG.input_file);
    std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>> fw;
    bool consistentIndexing;
    if(ARG.outputFileExists || dimCount != 3)
    {
        fw = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(ARG.output_file, dimCount,
                                                                     &dim.front());
        consistentIndexing = true;
    } else
    {
        fw = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(ARG.output_file, dim[0],
                                                                     dim[1], ARG.frames.size());
        consistentIndexing = false;
    }

    ftpl::thread_pool* threadpool = nullptr;
    if(ARG.threads != 0)
    {
        threadpool = new ftpl::thread_pool(ARG.threads);
    }
    uint64_t k_out = -1;
    for(uint64_t k_in : ARG.frames)
    {
        if(consistentIndexing)
        {
            k_out = k_in;
        } else
        {
            k_out++;
        }
        if(threadpool != nullptr)
        {
            threadpool->push(processFrame<T>, ARG, k_in, k_out, fr, fw);
        } else
        {
            processFrame<T>(0, ARG, k_in, k_out, fr, fw);
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
    const std::string prgInfo = "Frame-wise flip of DEN file.";
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
    io::DenFileInfo di(ARG.input_file);
    io::DenSupportedType dataType = di.getElementType();
    switch(dataType)
    {
    case io::DenSupportedType::UINT16: {
        run<uint16_t>(ARG);
        break;
    }
    case io::DenSupportedType::FLOAT32: {
        run<float>(ARG);
        break;
    }
    case io::DenSupportedType::FLOAT64: {
        run<double>(ARG);
        break;
    }
    default: {
        std::string errMsg = io::xprintf("Unsupported data type %s.",
                                         io::DenSupportedTypeToString(dataType).c_str());
        KCTERR(errMsg);
    }
    }
    PRG.endLog(true);
    return 0;
}

