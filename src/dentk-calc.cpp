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
    uint32_t dimx, dimy, dimz;
    uint64_t frameSize;
    bool outputFileExists = false;
    bool add = false;
    bool subtract = false;
    bool divide = false;
    bool inverseDivide = false;
    bool multiply = false;
    bool max = false;
    bool min = false;
};

void Args::defineArguments()
{
    cliApp->add_option("input_op1", input_op1, "Component A in the equation C=A op B.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("input_op2", input_op2, "Component B in the equation C=A op B.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("output", output, "Component C in the equation C=A op B.")->required();
    // Adding radio group see https://github.com/CLIUtils/CLI11/pull/234
    CLI::Option_group* op_clg
        = cliApp->add_option_group("Operation", "Mathematical operation to perform.");
    op_clg->add_flag("--add", add, "op1 + op2");
    op_clg->add_flag("--subtract", subtract, "op1 - op2");
    op_clg->add_flag("--multiply", multiply, "op1 * op2");
    op_clg->add_flag("--divide", divide, "op1 / op2");
    op_clg->add_flag("--inverse-divide", inverseDivide, "op2 / op1");
    op_clg->add_flag("--max", max, "max(op1, op2)");
    op_clg->add_flag("--min", min, "min(op1, op2)");
    op_clg->require_option(1);
    addForceArgs();
    addVerboseArgs();
    addFramespecArgs();
    addThreadingArgs();
}

int Args::postParse()
{
    io::DenFileInfo input_op1_inf(input_op1);
    io::DenFileInfo input_op2_inf(input_op2);
    // If output exists, force is true and is compatible, leave it
    int existFlag = handleFileExistence(output, force, input_op1);
    if(existFlag == 1)
    {
        return 1;
    } else if(existFlag == -1)
    {
        outputFileExists = true;
    }
    // Test if minuend and subtraend are of the same type and dimensions
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
    if(input_op1_inf.dimx() != input_op2_inf.dimx() || input_op1_inf.dimy() != input_op2_inf.dimy()
       || input_op1_inf.dimz() != input_op2_inf.dimz())
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
    if(!add && !subtract && !divide && !multiply && !max && !min && !inverseDivide)
    {
        LOGE << "You must provide one of supported operations (add, subtract, divide, multiply, "
                "inverse-divide)";
        return 1;
    }
    dimx = input_op1_inf.dimx();
    dimy = input_op1_inf.dimy();
    dimz = input_op1_inf.dimz();
    if(io::pathExists(output))
    {
        io::DenFileInfo output_inf(output);
        if(output_inf.dimx() != dimx || output_inf.dimy() != dimy || output_inf.dimz() != dimz)
        {
            LOGI << io::xprintf("Existing output file %s has incompatible dimensions and will be "
                                "romoved before dentk-calc calclulation.",
                                output.c_str());
            std::remove(output.c_str());
        }
    }
    frameSize = (uint64_t)dimx * (uint64_t)dimy;
    fillFramesVector(dimz);
    return 0;
}

// See
// https://stackoverflow.com/questions/29265451/how-to-typedef-a-function-pointer-with-template-arguments
template <typename T>
using Operator = T (*)(const T&, const T&);

template <typename T>
void processFrame(int _FTPLID,
                  Args ARG,
                  uint32_t k_in,
                  uint32_t k_out,
                  std::shared_ptr<io::DenFrame2DReader<T>>& aReader,
                  std::shared_ptr<io::DenFrame2DReader<T>>& bReader,
                  std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>>& outputWritter)
{
    std::shared_ptr<io::BufferedFrame2D<T>> A = aReader->readBufferedFrame(k_in);
    std::shared_ptr<io::BufferedFrame2D<T>> B = bReader->readBufferedFrame(k_in);
    io::BufferedFrame2D<T> x(T(0), ARG.dimx, ARG.dimy);
    T* A_array = A->getDataPointer();
    T* B_array = B->getDataPointer();
    T* x_array = x.getDataPointer();
    Operator<T> op = nullptr;

    if(ARG.multiply)
    {
        op = [](const T& i, const T& j) { return T(i * j); };
    } else if(ARG.divide)
    {
        op = [](const T& i, const T& j) { return T(i / j); };
    } else if(ARG.inverseDivide)
    {
        op = [](const T& i, const T& j) { return T(j / i); };
    } else if(ARG.add)
    {
        op = [](const T& i, const T& j) { return T(i + j); };
    } else if(ARG.subtract)
    {
        op = [](const T& i, const T& j) { return T(i - j); };
    } else if(ARG.max)
    {
        op = [](const T& i, const T& j) { return std::max(i, j); };
    } else if(ARG.min)
    {
        op = [](const T& i, const T& j) { return std::min(i, j); };
    }
    std::transform(A_array, A_array + ARG.frameSize, B_array, x_array, op);
    outputWritter->writeBufferedFrame(x, k_out);
    if(ARG.verbose)
    {
        if(k_in == k_out)
        {
            LOGD << io::xprintf("Processed frame %d/%d.", k_in, outputWritter->dimz());
        } else
        {
            LOGD << io::xprintf("Processed frame %d->%d/%d.", k_in, k_out, outputWritter->dimz());
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
    std::shared_ptr<io::DenFrame2DReader<T>> bReader
        = std::make_shared<io::DenFrame2DReader<T>>(ARG.input_op2, ARG.threads);
    std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>> outputWritter;
    if(ARG.outputFileExists)
    {
        outputWritter = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(
            ARG.output, ARG.dimx, ARG.dimy, ARG.dimz);
    } else
    {
        outputWritter = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(
            ARG.output, ARG.dimx, ARG.dimy, ARG.frames.size());
    }
    const int dummy_FTPLID = 0;
    uint32_t k_in, k_out;
    for(uint32_t IND = 0; IND != ARG.frames.size(); IND++)
    {
        k_in = ARG.frames[IND];
        if(ARG.outputFileExists)
        {
            k_out = k_in; // To be able to do dentk-calc --force --multiply -f 0,end zero.den
                          // BETA.den BETA.den
        } else
        {
            k_out = IND;
        }
        if(threadpool)
        {
            threadpool->push(processFrame<T>, ARG, k_in, k_out, aReader, bReader, outputWritter);
        } else
        {
            processFrame<T>(dummy_FTPLID, ARG, k_in, k_out, aReader, bReader, outputWritter);
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
    const std::string prgInfo = "Element-wise operation on two DEN files with the same dimensions.";
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
    PRG.endLog(true);
}

