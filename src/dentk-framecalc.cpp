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
#include "PROG/Program.hpp"
#include "PROG/parseArgs.h"

using namespace KCT;

// class declarations
struct Args
{
    int parseArguments(int argc, char* argv[]);
    std::string input_op1 = "";
    std::string input_op2 = "";
    std::string output = "";
    std::string frameSpecs = "";
    std::vector<int> frames;
    uint32_t dimx, dimy, dimz;
    bool force = false;
    bool add = false;
    bool subtract = false;
    bool divide = false;
    bool inverseDivide = false;
    bool multiply = false;
    bool max = false;
    bool min = false;
};

template <typename T>
void processFiles(Args a)
{
    io::DenFileInfo di(a.input_op1);
    uint32_t dimx = di.dimx();
    uint32_t dimy = di.dimy();
    uint32_t frameSize = dimx * dimy;
    std::shared_ptr<io::Frame2DReaderI<T>> aReader
        = std::make_shared<io::DenFrame2DReader<T>>(a.input_op1);
    std::shared_ptr<io::Frame2DReaderI<T>> bReader
        = std::make_shared<io::DenFrame2DReader<T>>(a.input_op2);
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> outputWritter
        = std::make_shared<io::DenAsyncFrame2DWritter<T>>(
            a.output, dimx, dimy,
            a.frames.size()); // I write regardless to frame specification to original position
    std::shared_ptr<io::BufferedFrame2D<T>> B
        = std::dynamic_pointer_cast<io::BufferedFrame2D<T>>(bReader->readFrame(0));
    T* B_array = B->getDataPointer();
    T* A_array;
    io::BufferedFrame2D<T> x(T(0), dimx, dimy);
    T* x_array = x.getDataPointer();
    for(int k = 0; k != a.frames.size(); k++)
    {
        /*
        std::shared_ptr<io::Frame2DI<T>> A = aReader->readFrame(a.frames[k]);
        double val;
                for(int i = 0; i != dimx; i++)
                {
                    for(int j = 0; j != dimy; j++)
                    {
                        if(a.multiply)
                        {
                            val = A->get(i, j) * B->get(i, j);
                        } else if(a.divide)
                        {
                            val = A->get(i, j) / B->get(i, j);
                        } else if(a.inverseDivide)
                        {
                            val = B->get(i, j) / A->get(i, j);
                        } else if(a.add)
                        {
                            val = A->get(i, j) + B->get(i, j);
                        } else if(a.subtract)
                        {
                            val = A->get(i, j) - B->get(i, j);
                        } else if(a.max)
                        {
                            val = std::max(A->get(i, j), B->get(i, j));
                        } else if(a.min)
                        {
                            val = std::min(A->get(i, j), B->get(i, j));
                        }
                        x.set(T(val), i, j);
                    }
                }*/
        std::shared_ptr<io::BufferedFrame2D<T>> A
            = std::dynamic_pointer_cast<io::BufferedFrame2D<T>>(aReader->readFrame(a.frames[k]));
        A_array = A->getDataPointer();
        for(uint32_t IND = 0; IND != frameSize; IND++)
        {
            if(a.multiply)
            {
                x_array[IND] = A_array[IND] * B_array[IND];
            } else if(a.divide)
            {
                x_array[IND] = A_array[IND] / B_array[IND];
            } else if(a.inverseDivide)
            {
                x_array[IND] = B_array[IND] / A_array[IND];
            } else if(a.add)
            {
                x_array[IND] = A_array[IND] + B_array[IND];
            } else if(a.subtract)
            {
                x_array[IND] = A_array[IND] - B_array[IND];
            } else if(a.max)
            {
                x_array[IND] = std::max(A_array[IND], B_array[IND]);
            } else if(a.min)
            {
                x_array[IND] = std::min(A_array[IND], B_array[IND]);
            }
        }
        outputWritter->writeFrame(x, k);
    }
}

int main(int argc, char* argv[])
{
    using namespace KCT::util;
    Program PRG(argc, argv);
    // After init parsing arguments
    Args ARG;
    int parseResult = ARG.parseArguments(argc, argv);
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog();
    io::DenFileInfo di(ARG.input_op1);
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

int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Frame-wise operation C_i = A_i op B  where the same operation is performed with "
                  "every frame in A." };
    app.add_option("input_op1", input_op1, "Component A in the equation C=A_i op B.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("input_op2", input_op2, "Component B in the equation C=A_i op B.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("output", output, "Component C in the equation C=A op B.")->required();
    // Adding radio group see https://github.com/CLIUtils/CLI11/pull/234
    CLI::Option_group* op_clg
        = app.add_option_group("Operation", "Mathematical operation to perform.");
    op_clg->add_flag("--add", add, "op1 + op2");
    op_clg->add_flag("--subtract", subtract, "op1 - op2");
    op_clg->add_flag("--multiply", multiply, "op1 * op2");
    op_clg->add_flag("--divide", divide, "op1 / op2");
    op_clg->add_flag("--inverse-divide", inverseDivide, "op2 / op1");
    op_clg->add_flag("--max", max, "max(op1, op2)");
    op_clg->add_flag("--min", min, "min(op1, op2)");
    op_clg->require_option(1);
    app.add_flag("--force", force, "Overwrite output file if it exists.");
    app.add_option("-f,--frames", frameSpecs,
                   "Specify only particular frames to process. You can input range i.e. 0-20 or "
                   "also individual coma separated frames i.e. 1,8,9. Order does matter. Accepts "
                   "end literal that means total number of slices of the input.");
    try
    {
        app.parse(argc, argv);
        io::DenFileInfo inf(input_op1);
        frames = util::processFramesSpecification(frameSpecs, inf.dimz());
    } catch(const CLI::ParseError& e)
    {
        int exitcode = app.exit(e);
        if(exitcode == 0) // Help message was printed
        {
            return 1;
        } else
        {
            LOGE << "Parse error catched";
            // Negative value should be returned
            return -1;
        }
    }
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
    if(input_op1_inf.getDataType() != input_op2_inf.getDataType())
    {
        LOGE << io::xprintf(
            "Type incompatibility while the file %s is of type %s and file %s has "
            "type %s.",
            input_op1.c_str(), io::DenSupportedTypeToString(input_op1_inf.getDataType()).c_str(),
            input_op2.c_str(), io::DenSupportedTypeToString(input_op2_inf.getDataType()).c_str());
        return 1;
    }
    if(input_op1_inf.dimx() != input_op2_inf.dimx() || input_op1_inf.dimy() != input_op2_inf.dimy())
    {
        LOGE << io::xprintf(
            "Files %s and %s have incompatible dimensions.\nFile %s of the type %s has "
            "dimensions (x, y, z) = (%d, %d, %d).\nFile %s of the type %s has "
            "dimensions (x, y, z) = (%d, %d, %d).",
            input_op1.c_str(), input_op2.c_str(), input_op1.c_str(),
            io::DenSupportedTypeToString(input_op1_inf.getDataType()).c_str(),
            input_op1_inf.getNumCols(), input_op1_inf.getNumRows(), input_op1_inf.getNumSlices(),
            input_op2.c_str(), io::DenSupportedTypeToString(input_op2_inf.getDataType()).c_str(),
            input_op2_inf.getNumCols(), input_op2_inf.getNumRows(), input_op2_inf.getNumSlices());
        return 1;
    }
    if(input_op2_inf.dimz() != 1)
    {
        LOGE << io::xprintf("Second operand %s shall have only one frame but has %d.",
                            input_op2_inf.dimz());
        return 1;
    }
    if(!add && !subtract && !divide && !multiply && !max && !min)
    {
        LOGE << "You must provide one of supported operations (add, subtract, divide, multiply)";
    }
    return 0;
}
