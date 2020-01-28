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

using namespace CTL;

// class declarations
struct Args
{
    int parseArguments(int argc, char* argv[]);
    std::string input_op1 = "";
    std::string input_op2 = "";
    std::string output = "";
    std::string frameSpecs = "";
    std::vector<int> frames;
    bool force = false;
    bool add = false;
    bool subtract = false;
    bool divide = false;
    bool multiply = false;
};

int main(int argc, char* argv[])
{
    using namespace CTL::util;
    Program PRG(argc, argv);
    // After init parsing arguments
    Args a;
    int parseResult = a.parseArguments(argc, argv);
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog();
    io::DenFileInfo di(a.input_op1);
    int dimx = di.getNumCols();
    int dimy = di.getNumRows();
    int dimz = di.getNumSlices();
    io::DenSupportedType dataType = di.getDataType();
    switch(dataType)
    {
    case io::DenSupportedType::uint16_t_:
    {
        std::shared_ptr<io::Frame2DReaderI<uint16_t>> aReader
            = std::make_shared<io::DenFrame2DReader<uint16_t>>(a.input_op1);
        std::shared_ptr<io::Frame2DReaderI<uint16_t>> bReader
            = std::make_shared<io::DenFrame2DReader<uint16_t>>(a.input_op2);
        std::shared_ptr<io::AsyncFrame2DWritterI<uint16_t>> outputWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<uint16_t>>(
                a.output, dimx, dimy,
                dimz); // I write regardless to frame specification to original position
        io::BufferedFrame2D<uint16_t> x(uint16_t(0), dimx, dimy);
        for(const int& k : a.frames)
        {
            uint16_t val;
            std::shared_ptr<io::Frame2DI<uint16_t>> A = aReader->readFrame(k);
            std::shared_ptr<io::Frame2DI<uint16_t>> B = bReader->readFrame(k);
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
                    } else if(a.add)
                    {
                        val = A->get(i, j) + B->get(i, j);
                    } else // subtract
                    {
                        val = A->get(i, j) - B->get(i, j);
                    }
                    x.set(val, i, j);
                }
            }
            outputWritter->writeFrame(x, k);
        }
        break;
    }
    case io::DenSupportedType::float_:
    {
        std::shared_ptr<io::Frame2DReaderI<float>> aReader
            = std::make_shared<io::DenFrame2DReader<float>>(a.input_op1);
        std::shared_ptr<io::Frame2DReaderI<float>> bReader
            = std::make_shared<io::DenFrame2DReader<float>>(a.input_op2);
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> outputWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(
                a.output, dimx, dimy,
                dimz); // I write regardless to frame specification to original position
        io::BufferedFrame2D<float> x(float(0), dimx, dimy);
        for(const int& k : a.frames)
        {
            float val;
            std::shared_ptr<io::Frame2DI<float>> A = aReader->readFrame(k);
            std::shared_ptr<io::Frame2DI<float>> B = bReader->readFrame(k);
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
                    } else if(a.add)
                    {
                        val = A->get(i, j) + B->get(i, j);
                    } else // subtract
                    {
                        val = A->get(i, j) - B->get(i, j);
                    }
                    x.set(val, i, j);
                }
            }
            outputWritter->writeFrame(x, k);
        }
        break;
    }
    case io::DenSupportedType::double_:
    {
        std::shared_ptr<io::Frame2DReaderI<double>> aReader
            = std::make_shared<io::DenFrame2DReader<double>>(a.input_op1);
        std::shared_ptr<io::Frame2DReaderI<double>> bReader
            = std::make_shared<io::DenFrame2DReader<double>>(a.input_op2);
        std::shared_ptr<io::AsyncFrame2DWritterI<double>> outputWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<double>>(
                a.output, dimx, dimy,
                dimz); // I write regardless to frame specification to original position
        io::BufferedFrame2D<double> x(double(0), dimx, dimy);
        for(const int& k : a.frames)
        {
            double val;
            std::shared_ptr<io::Frame2DI<double>> A = aReader->readFrame(k);
            std::shared_ptr<io::Frame2DI<double>> B = bReader->readFrame(k);
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
                    } else if(a.add)
                    {
                        val = A->get(i, j) + B->get(i, j);
                    } else // subtract
                    {
                        val = A->get(i, j) - B->get(i, j);
                    }
                    x.set(val, i, j);
                }
            }
            outputWritter->writeFrame(x, k);
        }
        break;
    }
    default:
    {
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(dataType));
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
    }
    }
    PRG.endLog();
}

int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Point-wise operation on two DEN files with the same dimensions." };
    app.add_option("input_op1", input_op1, "Component A in the equation C=A op B.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("input_op2", input_op2, "Component B in the equation C=A op B.")
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
    op_clg->require_option(1);
    app.add_flag("--force", force, "Overwrite output file if it exists.");
    app.add_option("-f,--frames", frameSpecs,
                   "Specify only particular frames to process. You can input range i.e. 0-20 or "
                   "also individual coma separated frames i.e. 1,8,9. Order does matter. Accepts "
                   "end literal that means total number of slices of the input.");
    try
    {
        app.parse(argc, argv);
        io::DenFileInfo inf(input_op2);
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
    io::DenFileInfo disub(input_op1);
    io::DenFileInfo dimin(input_op2);
    if(disub.getDataType() != dimin.getDataType())
    {
        LOGE << io::xprintf(
            "Type incompatibility while the file %s is of type %s and file %s has "
            "type %s.",
            input_op1.c_str(), io::DenSupportedTypeToString(disub.getDataType()).c_str(),
            input_op2.c_str(), io::DenSupportedTypeToString(dimin.getDataType()).c_str());
        return 1;
    }
    if(disub.getNumCols() != dimin.getNumCols() || disub.getNumRows() != dimin.getNumRows()
       || disub.getNumSlices() != dimin.getNumSlices())
    {
        LOGE << io::xprintf(
            "Files %s and %s have incompatible dimensions.\nFile %s of the type %s has "
            "dimensions (x, y, z) = (%d, %d, %d).\nFile %s of the type %s has "
            "dimensions (x, y, z) = (%d, %d, %d).",
            input_op1.c_str(), input_op2.c_str(), input_op1.c_str(),
            io::DenSupportedTypeToString(disub.getDataType()).c_str(), disub.getNumCols(),
            disub.getNumRows(), disub.getNumSlices(), input_op2.c_str(),
            io::DenSupportedTypeToString(dimin.getDataType()).c_str(), dimin.getNumCols(),
            dimin.getNumRows(), dimin.getNumSlices());
        return 1;
    }
    if(!add && !subtract && !divide && !multiply)
    {
        LOGE << "You must provide one of supported operations (add, subtract, divide, multiply)";
    }
    return 0;
}
