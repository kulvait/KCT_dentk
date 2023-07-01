// The purpose of this tool is to filter out outer bone structures.
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
#include "ftpl.h"

using namespace KCT;
using namespace KCT::util;

// class declarations
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
    std::string input_den = "";
    std::string output_den = "";
    uint32_t dimx, dimy, dimz;
    uint64_t frameSize;
    bool nanandinftozero = false;
    bool logarithm = false;
    bool exponentiation = false;
    bool squareroot = false;
    bool square = false;
    bool absoluteValue = false;
    bool multiplyByConstant = false;
    double constantToMultiply = 0.0;
    bool addConstant = false;
    double constantToAdd = 0.0;
    bool invert = false;
    bool max = false;
    bool min = false;
    double maxArgument = 0.0;
    double minArgument = 0.0;
};

void Args::defineArguments()
{
    cliApp->add_option("input_den", input_den, "Input file.")->check(CLI::ExistingFile)->required();
    cliApp->add_option("output_den", output_den, "Output file.")->required();
    addForceArgs();
    addVerboseArgs();
    addFramespecArgs();
    addThreadingArgs();
    CLI::Option_group* op_clg = cliApp->add_option_group(
        "Operation", "Mathematical operation f to perform element wise to get OUTPUT=f(INPUT).");
    registerOptionGroup("operation", op_clg);
    registerOption("log", op_clg->add_flag("--log", logarithm, "Natural logarithm."));
    registerOption("exp", op_clg->add_flag("--exp", exponentiation, "Exponentiation."));
    registerOption("sqrt", op_clg->add_flag("--sqrt", squareroot, "Square root."));
    registerOption("square", op_clg->add_flag("--square", square, "Square."));
    registerOption("abs", op_clg->add_flag("--abs", absoluteValue, "Absolute value."));
    registerOption("inv", op_clg->add_flag("--inv", invert, "Invert value."));
    registerOption("nan-and-inf-to-zero",
                   op_clg->add_flag("--nan-and-inf-to-zero", nanandinftozero,
                                    "Convert NaN and Inf values to zero."));
    registerOption(
        "multiply",
        op_clg->add_option("--multiply", constantToMultiply, "Multiplication with a constant."));
    registerOption("add", op_clg->add_option("--add", constantToAdd, "Add a constant."));
    registerOption("min", op_clg->add_option("--min", minArgument, "Perform min(x, minArgument)."));
    registerOption("max", op_clg->add_option("--max", maxArgument, "Perform max(x, maxArgument)."));
    op_clg->require_option(1);
}

int Args::postParse()
{
    int e = handleFileExistence(output_den, force, force);
    if(e != 0)
    {
        return e;
    }
    if(getRegisteredOption("multiply")->count() > 0)
    {
        multiplyByConstant = true;
    }
    if(getRegisteredOption("add")->count() > 0)
    {
        addConstant = true;
    }
    if(getRegisteredOption("min")->count() > 0)
    {
        min = true;
    }
    if(getRegisteredOption("max")->count() > 0)
    {
        max = true;
    }
    io::DenFileInfo inf(input_den);
    dimx = inf.dimx();
    dimy = inf.dimy();
    dimz = inf.dimz();
    frameSize = (uint64_t)dimx * (uint64_t)dimy;
    fillFramesVector(dimz);
    return 0;
}

template <typename T>
using Transform = T (*)(const T&);

template <typename T>
void processFrame(int _FTPLID,
                  Args ARG,
                  uint32_t k_in,
                  uint32_t k_out,
                  std::shared_ptr<io::DenFrame2DReader<T>>& denReader,
                  std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>>& outputWritter)
{
    io::BufferedFrame2D<T> f(T(0), ARG.dimx, ARG.dimy);
    std::shared_ptr<io::BufferedFrame2D<T>> A = denReader->readBufferedFrame(k_in);
    T* A_array = A->getDataPointer();
    T* x_array = f.getDataPointer();
    Transform<T> o = nullptr;
    T constantToMultiply;
    T constantToAdd;
    T maxArgument;
    T minArgument;
    if(ARG.logarithm)
    {
        o = [](const T& x) { return T(std::log(x)); };
    }
    if(ARG.exponentiation)
    {
        o = [](const T& x) { return T(std::exp(x)); };
    }
    if(ARG.squareroot)
    {
        o = [](const T& x) { return T(std::sqrt(x)); };
    }
    if(ARG.square)
    {
        o = [](const T& x) { return T(x * x); };
    }
    if(ARG.absoluteValue)
    {
        //#pragma GCC diagnostic push
        //#pragma GCC diagnostic ignored "-Wabsolute-value"
        o = [](const T& x) { return T(std::abs(x)); };
        //#pragma GCC diagnostic pop
    }
    if(ARG.invert)
    {
        o = [](const T& x) { return T(T(1) / x); };
    }
    if(ARG.nanandinftozero)
    {
        o = [](const T& x) {
            double val_double = (double)x;
            if(std::isnan(val_double))
            {
                return T(0);
            } else if(std::isinf(val_double))
            {
                return T(0);
            } else
            {
                return x;
            }
        };
    }
    if(o != nullptr)
    {
        std::transform(A_array, A_array + ARG.frameSize, x_array, o);
    }
    if(ARG.multiplyByConstant)
    {
        constantToMultiply = ARG.constantToMultiply;
        std::transform(A_array, A_array + ARG.frameSize, x_array,
                       [constantToMultiply](const T& x) { return T(constantToMultiply * x); });
    }
    if(ARG.addConstant)
    {
        constantToAdd = ARG.constantToAdd;
        std::transform(A_array, A_array + ARG.frameSize, x_array,
                       [constantToAdd](const T& x) { return T(x + constantToAdd); });
    }
    if(ARG.min)
    {
        minArgument = ARG.minArgument;
        std::transform(A_array, A_array + ARG.frameSize, x_array,
                       [minArgument](const T& x) { return x < minArgument ? x : minArgument; });
    }
    if(ARG.max)
    {
        maxArgument = ARG.maxArgument;
        std::transform(A_array, A_array + ARG.frameSize, x_array,
                       [maxArgument](const T& x) { return x > maxArgument ? x : maxArgument; });
    }
    outputWritter->writeBufferedFrame(f, k_out);
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
    std::shared_ptr<io::DenFrame2DReader<T>> denReader
        = std::make_shared<io::DenFrame2DReader<T>>(ARG.input_den, ARG.threads);
    std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>> outputWritter
        = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(
            ARG.output_den, ARG.dimx, ARG.dimy,
            ARG.frames.size()); // I write regardless to frame specification to original position
    const int dummy_FTPLID = 0;
    uint32_t k_in, k_out;
    for(uint32_t IND = 0; IND != ARG.frames.size(); IND++)
    {
        k_in = ARG.frames[IND];
        k_out = IND;
        if(threadpool)
        {
            threadpool->push(processFrame<T>, ARG, k_in, k_out, denReader, outputWritter);
        } else
        {
            processFrame<T>(dummy_FTPLID, ARG, k_in, k_out, denReader, outputWritter);
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
    Args ARG(argc, argv,
             "Calculate an unary operation or transformation pointwise on the input file to "
             "produce output file.");
    int parseResult = ARG.parse(false);
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    io::DenFileInfo di(ARG.input_den);
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

