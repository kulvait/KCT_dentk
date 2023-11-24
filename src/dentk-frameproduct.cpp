// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctype.h>
#include <iostream>
#include <regex>
#include <string>

// External libraries
#include "CLI/CLI.hpp" //Command line parser

// Internal libraries
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DBufferedWritter.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DCachedReader.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/ArgumentsVerbose.hpp"
#include "PROG/Program.hpp"
#include "PROG/parseArgs.h"
#include "ftpl.h"

using namespace KCT;
using namespace KCT::util;

// class declarations
class Args : public ArgumentsForce,
             public ArgumentsVerbose,
             public ArgumentsFramespec,
             public ArgumentsThreading
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };
    CLI::Option *a_count_opt, *b_count_opt;

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName)
        , ArgumentsVerbose(argc, argv, prgName)
        , ArgumentsFramespec(argc, argv, prgName)
        , ArgumentsThreading(argc, argv, prgName){};
    std::string input_op1 = "";
    std::string input_op2 = "";
    std::string output = "";
    uint32_t dimx_a, dimy_a, dimz_a;
    uint32_t dimx_b, dimy_b, dimz_b;
    uint64_t frameSize;
    uint32_t dimz_a_count = 0;
    uint32_t dimz_b_count = 0;
    io::DenSupportedType dataType;
};

void Args::defineArguments()
{
    cliApp->add_option("input_op1", input_op1, "Component A in the equation C=frame_product(A, B).")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp
        ->add_option("input_op2", input_op2, "Component B in the equation C=frame_product(A,  B).")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp
        ->add_option("output", output,
                     "Component C in the equation C=frame_product(A, B), with dimensions "
                     "dimz_a_count, dimz_b_count.")
        ->required();
    a_count_opt = cliApp->add_option("--a-count", dimz_a_count,
                                     "Count of frames to use in A, defaults to dimz_a.");
    b_count_opt = cliApp->add_option("--b-count", dimz_b_count,
                                     "Count of frames to use in B, defaults to dimz_b.");
    addForceArgs();
    addFramespecArgs();
    addThreadingArgs();
}

int Args::postParse()
{
    if(!force)
    {
        if(io::pathExists(output))
        {
            LOGE << "Error: output file already exists, use --force to force overwrite.";
            return 1;
        }
    }
    // Test if minuend and subtraend are of the same type and dimensions
    io::DenFileInfo input_op1_inf(input_op1);
    io::DenFileInfo input_op2_inf(input_op2);
    dimx_a = input_op1_inf.dimx();
    dimy_a = input_op1_inf.dimy();
    dimz_a = input_op1_inf.dimz();
    dimx_b = input_op2_inf.dimx();
    dimy_b = input_op2_inf.dimy();
    dimz_b = input_op2_inf.dimz();
    if(input_op1_inf.getElementType() != input_op2_inf.getElementType())
    {
        LOGE << io::xprintf("Type incompatibility while the file %s is of type %s and file %s has "
                            "type %s.",
                            input_op1.c_str(),
                            io::DenSupportedTypeToString(input_op1_inf.getElementType()).c_str(),
                            input_op2.c_str(),
                            io::DenSupportedTypeToString(input_op2_inf.getElementType()).c_str());
        return 1;
    }
    dataType = input_op1_inf.getElementType();
    if(dimx_a != dimx_b || dimy_a != dimy_b)
    {
        LOGE << io::xprintf(
            "Files %s and %s have incompatible dimensions.\nFile %s of the type %s has "
            "dimensions (x, y, z) = (%d, %d, %d).\nFile %s of the type %s has "
            "dimensions (x, y, z) = (%d, %d, %d).",
            input_op1.c_str(), input_op2.c_str(), input_op1.c_str(),
            io::DenSupportedTypeToString(input_op1_inf.getElementType()).c_str(),
            input_op1_inf.getNumCols(), input_op1_inf.getNumRows(), input_op1_inf.getNumSlices(),
            input_op2.c_str(), io::DenSupportedTypeToString(input_op2_inf.getElementType()).c_str(),
            input_op2_inf.getNumCols(), input_op2_inf.getNumRows(), input_op2_inf.getNumSlices());
        return 1;
    }
    if(a_count_opt->count() == 0)
    {
        dimz_a_count = dimz_a;
    }
    if(b_count_opt->count() == 0)
    {
        dimz_b_count = dimz_b;
    }
    if(dimz_a_count > dimz_a)
    {
        LOGE << io::xprintf("File %s has %d frames but dimz_a_count = %d exceeds them.",
                            input_op1.c_str(), dimz_a, dimz_a_count);
    }
    if(dimz_b_count > dimz_b)
    {
        LOGE << io::xprintf("File %s has %d frames but dimz_a_count = %d exceeds them.",
                            input_op2.c_str(), dimz_b, dimz_b_count);
    }
    frameSize = (uint64_t)dimx_a * (uint64_t)dimy_a;
    return 0;
}

template <typename T>
void processFrame(int _FTPLID,
                  const Args& ARG,
                  const std::shared_ptr<io::DenFrame2DCachedReader<T>>& aReader,
                  const std::shared_ptr<io::DenFrame2DCachedReader<T>>& bReader,
                  const uint32_t& i,
                  const uint32_t& j,
                  T* elm)
{
    std::shared_ptr<io::BufferedFrame2D<T>> A = aReader->readBufferedFrame(i);
    std::shared_ptr<io::BufferedFrame2D<T>> B = bReader->readBufferedFrame(j);
    T* A_array = A->getDataPointer();
    T* B_array = B->getDataPointer();
    *elm = std::inner_product(A_array, A_array + ARG.frameSize, B_array, 0.0);
}

template <typename T>
void processFiles(Args ARG)
{
    ftpl::thread_pool* threadpool = nullptr;
    if(ARG.threads > 0)
    {
        threadpool = new ftpl::thread_pool(ARG.threads);
    }
    io::BufferedFrame2D<T> X(T(0), ARG.dimz_a_count, ARG.dimz_b_count);
    T* X_array = X.getDataPointer();
    std::shared_ptr<io::DenFrame2DCachedReader<T>> aReader
        = std::make_shared<io::DenFrame2DCachedReader<T>>(ARG.input_op1, ARG.threads, ARG.threads);
    std::shared_ptr<io::DenFrame2DCachedReader<T>> bReader
        = std::make_shared<io::DenFrame2DCachedReader<T>>(ARG.input_op2, ARG.threads, ARG.threads);
    const int dummy_FTPLID = 0;
    for(uint32_t j = 0; j != ARG.dimz_b_count; j++)
    {
        for(uint32_t i = 0; i != ARG.dimz_a_count; i++)
        {
            T* elm = X_array + i + j * ARG.dimz_a_count;
            //	T elm;
            if(threadpool)
            {
                threadpool->push(processFrame<T>, ARG, aReader, bReader, i, j, elm);
            } else
            {
                processFrame<T>(dummy_FTPLID, ARG, aReader, bReader, i, j, elm);
            }
        }
    }
    if(threadpool != nullptr)
    {
        threadpool->stop(true);
        delete threadpool;
    }
    std::array<uint32_t, 2> dim;
    dim[0] = ARG.dimz_a_count;
    dim[1] = ARG.dimz_b_count;
    io::DenFileInfo::createDenFileFromArray(X_array, true, ARG.output, ARG.dataType, 2, std::begin(dim), true);
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    const std::string prgInfo
        = "Frame-wise operation C_i = A_i op B  where the same operation is performed with "
          "every frame in A.";
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
    io::DenFileInfo di(ARG.input_op1);
    io::DenSupportedType dataType = di.getElementType();
    switch(dataType)
    {
    case io::DenSupportedType::UINT16: {
        processFiles<uint16_t>(ARG);
        break;
    }
    case io::DenSupportedType::FLOAT32: {
        processFiles<float>(ARG);
        break;
    }
    case io::DenSupportedType::FLOAT64: {
        processFiles<double>(ARG);
        break;
    }
    default: {
        std::string errMsg = io::xprintf("Unsupported data type %s.",
                                         io::DenSupportedTypeToString(dataType).c_str());
        KCTERR(errMsg);
    }
    }
    PRG.endLog();
}
