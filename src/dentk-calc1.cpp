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
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/Program.hpp"

using namespace KCT;
using namespace KCT::util;

// class declarations
// class declarations
class Args : public ArgumentsFramespec, public ArgumentsForce
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsFramespec(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName){};
    std::string input_den = "";
    std::string output_den = "";
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
};

template <typename T>
void processFiles(Args a)
{
    io::DenFileInfo di(a.input_den);
    int dimx = di.dimx();
    int dimy = di.dimy();
    int dimz = di.dimz();
    std::shared_ptr<io::Frame2DReaderI<T>> denReader
        = std::make_shared<io::DenFrame2DReader<T>>(a.input_den);
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> outputWritter
        = std::make_shared<io::DenAsyncFrame2DWritter<T>>(
            a.output_den, dimx, dimy,
            dimz); // I write regardless to frame specification to original position
    for(const int& k : a.frames)
    {
        io::BufferedFrame2D<T> f(T(0), dimx, dimy);
        std::shared_ptr<io::Frame2DI<T>> A = denReader->readFrame(k);
        for(int i = 0; i != dimx; i++)
        {
            for(int j = 0; j != dimy; j++)
            {
                if(a.logarithm)
                {
                    f.set(std::log(A->get(i, j)), i, j);
                }
                if(a.exponentiation)
                {
                    f.set(std::exp(A->get(i, j)), i, j);
                }
                if(a.squareroot)
                {
                    f.set(std::sqrt(A->get(i, j)), i, j);
                }
                if(a.square)
                {
                    T val = A->get(i, j);
                    val = val * val;
                    f.set(val, i, j);
                }
                if(a.multiplyByConstant)
                {
                    f.set(T(A->get(i, j) * a.constantToMultiply), i, j);
                }
                if(a.addConstant)
                {
                    f.set(T(A->get(i, j) + a.constantToAdd), i, j);
                }
                if(a.absoluteValue)
                {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wabsolute-value"
                    f.set(std::abs(A->get(i, j)), i, j);
#pragma GCC diagnostic pop
                }
                if(a.invert)
                {
                    f.set(1.0 / A->get(i, j), i, j);
                }
                if(a.nanandinftozero)
                {
                    T val = A->get(i, j);
                    double val_double = (double)val;
                    if(std::isnan(val_double))
                    {
                        f.set(T(0), i, j);
                    } else if(std::isinf(val_double))
                    {
                        f.set(T(0), i, j);
                    } else
                    {
                        f.set(val, i, j);
                    }
                }
            }
        }
        outputWritter->writeFrame(f, k);
    }
    // given angle attenuation is maximal
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    Args ARG(argc, argv,
             "Calculate an unary operation or transformation pointwise on the input file to "
             "produce output file.");
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    io::DenFileInfo di(ARG.input_den);
    io::DenSupportedType dataType = di.getDataType();
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

void Args::defineArguments()
{
    cliApp->add_option("input_den", input_den, "Input file.")->check(CLI::ExistingFile)->required();
    cliApp->add_option("output_den", output_den, "Output file.")->required();
    addForceArgs();
    addFramespecArgs();
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
    op_clg->require_option(1);
}

int Args::postParse()
{
    if(!force)
    {
        if(io::pathExists(output_den))
        {
            LOGE << "Error: output file already exists, use --force to force overwrite.";
            return 1;
        }
    }
    if(getRegisteredOption("multiply")->count() > 0)
    {
        multiplyByConstant = true;
    }
    if(getRegisteredOption("add")->count() > 0)
    {
        addConstant = true;
    }
    io::DenFileInfo di(input_den);
    fillFramesVector(di.dimz());
    return 0;
}
