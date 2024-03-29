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
#include "DEN/DenFrame2DReader.hpp"
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
    std::string frameSpecs = "";
    uint32_t dimx, dimy, dimz;
    uint64_t frameSize;
    bool add = false;
    bool subtract = false;
    bool flippedSubtract = false;
    bool divide = false;
    bool flippedDivide = false;
    bool multiply = false;
    bool max = false;
    bool min = false;
};

void Args::defineArguments()
{
    cliApp->add_option("input_op1", input_op1, "Component A in the equation C=A_i op B.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp
        ->add_option("input_op2", input_op2,
                     "Component B in the equation C=A_i op B, only frame with the index 0 is used.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("output", output, "Component C in the equation C=A op B.")->required();
    // Adding radio group see https://github.com/CLIUtils/CLI11/pull/234
    CLI::Option_group* op_clg
        = cliApp->add_option_group("Operation", "Mathematical operation to perform.");
    op_clg->add_flag("--add", add, "op1 + op2");
    op_clg->add_flag("--subtract", subtract, "op1 - op2");
    op_clg->add_flag("--flipped-subtract", flippedSubtract, "op2 - op1");
    op_clg->add_flag("--multiply", multiply, "op1 * op2");
    op_clg->add_flag("--divide", divide, "op1 / op2");
    op_clg->add_flag("--flipped-divide", flippedDivide, "op2 / op1");
    op_clg->add_flag("--max", max, "max(op1, op2)");
    op_clg->add_flag("--min", min, "min(op1, op2)");
    op_clg->require_option(1);
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
    if(input_op1_inf.dimx() != input_op2_inf.dimx() || input_op1_inf.dimy() != input_op2_inf.dimy())
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
    if(input_op2_inf.dimz() != 1)
    {
        LOGW << io::xprintf("Second operand %s has %d frames but only the first will be used.",
                            input_op2.c_str(), input_op2_inf.dimz());
    }
    if(!add && !subtract && !divide && !multiply && !max && !min && !flippedDivide && !flippedSubtract)
    {
        LOGE << "You must provide one of supported operations (add, subtract, divide, multiply, "
                "flipped-divide, flipped-subtract)";
        return 1;
    }
    dimx = input_op1_inf.dimx();
    dimy = input_op1_inf.dimy();
    dimz = input_op1_inf.dimz();
    frameSize = (uint64_t)dimx * (uint64_t)dimy;
    fillFramesVector(dimz);
    return 0;
}

template <typename T>
void processFrame(int _FTPLID,
                  Args ARG,
                  uint32_t k_in,
                  uint32_t k_out,
                  std::shared_ptr<io::DenFrame2DReader<T>>& aReader,
                  std::shared_ptr<io::BufferedFrame2D<T>>& B,
                  std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>>& outputWritter)
{
    std::shared_ptr<io::BufferedFrame2D<T>> A = aReader->readBufferedFrame(k_in);
    io::BufferedFrame2D<T> x(T(0), ARG.dimx, ARG.dimy);
    T* A_array = A->getDataPointer();
    T* B_array = B->getDataPointer();
    T* x_array = x.getDataPointer();
    if(ARG.multiply)
    {
        std::transform(A_array, A_array + ARG.frameSize, B_array, x_array, std::multiplies());
    } else if(ARG.divide)
    {
        std::transform(A_array, A_array + ARG.frameSize, B_array, x_array, std::divides());
    } else if(ARG.flippedDivide)
    {
        std::transform(A_array, A_array + ARG.frameSize, B_array, x_array,
                       [](T i, T j) { return j / i; });
    } else if(ARG.add)
    {
        std::transform(A_array, A_array + ARG.frameSize, B_array, x_array, std::plus());
    } else if(ARG.subtract)
    {
        std::transform(A_array, A_array + ARG.frameSize, B_array, x_array, std::minus());
    } else if(ARG.flippedSubtract)
    {
        std::transform(A_array, A_array + ARG.frameSize, B_array, x_array,
                       [](T i, T j) { return j - i; });
    } else if(ARG.max)
    {
        std::transform(A_array, A_array + ARG.frameSize, B_array, x_array,
                       [](T i, T j) { return std::max(i, j); });
    } else if(ARG.min)
    {
        std::transform(A_array, A_array + ARG.frameSize, B_array, x_array,
                       [](T i, T j) { return std::min(i, j); });
    }
    outputWritter->writeBufferedFrame(x, k_out);
    if(ARG.verbose)
    {
        if(k_in == k_out)
        {
            LOGD << io::xprintf("Processed frame %d/%d.", k_in, outputWritter->getFrameCount());
        } else
        {
            LOGD << io::xprintf("Processed frame %d->%d/%d.", k_in, k_out, outputWritter->getFrameCount());
        }
    }
}

template <typename T>
void processFiles(Args ARG)
{
    ftpl::thread_pool* threadpool = nullptr;
    if(ARG.threads > 0)
    {
        threadpool = new ftpl::thread_pool(ARG.threads);
    }
    std::shared_ptr<io::DenFrame2DReader<T>> aReader
        = std::make_shared<io::DenFrame2DReader<T>>(ARG.input_op1, ARG.threads);
    io::DenFrame2DReader<T> bReader(ARG.input_op2);
    std::shared_ptr<io::BufferedFrame2D<T>> B = bReader.readBufferedFrame(0);
    std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>> outputWritter
        = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(ARG.output, ARG.dimx, ARG.dimy,
                                                                  ARG.frames.size());
    const int dummy_FTPLID = 0;
    uint32_t k_in, k_out;
    for(uint32_t IND = 0; IND != ARG.frames.size(); IND++)
    {
        k_in = ARG.frames[IND];
        k_out = IND;
        if(threadpool)
        {
            threadpool->push(processFrame<T>, ARG, k_in, k_out, aReader, B, outputWritter);
        } else
        {
            processFrame<T>(dummy_FTPLID, ARG, k_in, k_out, aReader, B, outputWritter);
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
