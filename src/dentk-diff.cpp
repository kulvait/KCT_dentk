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
#include "ARGPARSE/parseArgs.h"
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"

using namespace CTL;

// class declarations
struct Args
{
    int parseArguments(int argc, char* argv[]);
    std::string input_subtrahend = "";
    std::string input_minuend = "";
    std::string output_difference = "";
    std::string frameSpecs = "";
    std::vector<int> frames;
    bool force = false;
};

int main(int argc, char* argv[])
{
    plog::Severity verbosityLevel
        = plog::debug; // Set to debug to see the debug messages, info messages
    std::string csvLogFile = "/tmp/dentk-diff.csv"; // Set NULL to disable
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    LOGI << "dentk-diff";
    // Process arguments
    Args a;
    int parseResult = a.parseArguments(argc, argv);
    if(parseResult != 0)
    {
        if(parseResult > 0)
        {
            return 0; // Exited sucesfully, help message printed
        } else
        {
            return -1; // Exited somehow wrong
        }
    }
    io::DenFileInfo di(a.input_subtrahend);
    int dimx = di.getNumCols();
    int dimy = di.getNumRows();
    int dimz = di.getNumSlices();
    io::DenSupportedType dataType = di.getDataType();
    switch(dataType)
    {
    case io::DenSupportedType::uint16_t_:
    {
        std::shared_ptr<io::Frame2DReaderI<uint16_t>> aReader
            = std::make_shared<io::DenFrame2DReader<uint16_t>>(a.input_subtrahend);
        std::shared_ptr<io::Frame2DReaderI<uint16_t>> bReader
            = std::make_shared<io::DenFrame2DReader<uint16_t>>(a.input_minuend);
        std::shared_ptr<io::AsyncFrame2DWritterI<uint16_t>> outputWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<uint16_t>>(a.output_difference, dimx,
                                                                     dimy, dimz);//I write regardless to frame specification to original position
        uint16_t* buffer = new uint16_t[dimx * dimy];
        io::BufferedFrame2D<uint16_t> x(buffer, dimx, dimy);
        delete[] buffer;
        for(const int &k : a.frames)
        {
            std::shared_ptr<io::Frame2DI<uint16_t>> A = aReader->readFrame(k);
            std::shared_ptr<io::Frame2DI<uint16_t>> B = bReader->readFrame(k);
            for(int i = 0; i != dimx; i++)
                for(int j = 0; j != dimy; j++)
                {
                    uint16_t difference = A->get(i, j) - B->get(i, j); // From uint16_t
                    x.set(difference, i, j);
                }
            outputWritter->writeFrame(x, k);
        }
        break;
    }
    case io::DenSupportedType::float_:
    {
        std::shared_ptr<io::Frame2DReaderI<float>> aReader
            = std::make_shared<io::DenFrame2DReader<float>>(a.input_subtrahend);
        std::shared_ptr<io::Frame2DReaderI<float>> bReader
            = std::make_shared<io::DenFrame2DReader<float>>(a.input_minuend);
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> outputWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(a.output_difference, dimx, dimy,
                                                                  dimz);//I write regardless to frame specification to original position
        float* buffer = new float[dimx * dimy];
        io::BufferedFrame2D<float> x(buffer, dimx, dimy);
        delete[] buffer;
        for(const int &k : a.frames)
        {
            std::shared_ptr<io::Frame2DI<float>> A = aReader->readFrame(k);
            std::shared_ptr<io::Frame2DI<float>> B = bReader->readFrame(k);
            for(int i = 0; i != dimx; i++)
                for(int j = 0; j != dimy; j++)
                {
                    float difference = A->get(i, j) - B->get(i, j); // From uint16_t
                    x.set(difference, i, j);
                }
            outputWritter->writeFrame(x, k);
        }
        break;
    }
    case io::DenSupportedType::double_:
    {
        std::shared_ptr<io::Frame2DReaderI<double>> aReader
            = std::make_shared<io::DenFrame2DReader<double>>(a.input_subtrahend);
        std::shared_ptr<io::Frame2DReaderI<double>> bReader
            = std::make_shared<io::DenFrame2DReader<double>>(a.input_minuend);
        std::shared_ptr<io::AsyncFrame2DWritterI<double>> outputWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<double>>(a.output_difference, dimx, dimy,
                                                                   dimz);//I write regardless to frame specification to original position
        double* buffer = new double[dimx * dimy];
        io::BufferedFrame2D<double> x(buffer, dimx, dimy);
        delete[] buffer;
        for(const int &k : a.frames)
        {
            std::shared_ptr<io::Frame2DI<double>> A = aReader->readFrame(k);
            std::shared_ptr<io::Frame2DI<double>> B = bReader->readFrame(k);
            for(int i = 0; i != dimx; i++)
                for(int j = 0; j != dimy; j++)
                {
                    double difference = A->get(i, j) - B->get(i, j); // From uint16_t
                    x.set(difference, i, j);
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
}

int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Subtract two DEN files with the same dimensions from each other." };
    app.add_flag("--force", force, "Force rewrite output file if it exists.");
    app.add_option("-f,--frames", frameSpecs,
                   "Specify only particular frames to process. You can input range i.e. 0-20 or "
                   "also individual coma separated frames i.e. 1,8,9. Order does matter. Accepts "
                   "end literal that means total number of slices of the input.");
    app.add_option("input_subtrahend", input_subtrahend, "Component A in the equation C=A-B.")
        ->check(CLI::ExistingFile);
    app.add_option("input_minuend", input_minuend, "Component B in the equation C=A-B.")
        ->check(CLI::ExistingFile);
    app.add_option("output_difference", output_difference, "Component C in the equation C=A-B.");
    try
    {
        app.parse(argc, argv);
	io::DenFileInfo inf(input_minuend);
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
        if(io::fileExists(output_difference))
        {
            LOGE << "Error: output file already exists, use -f to force overwrite.";
            return 1;
        }
    }
    // Test if minuend and subtraend are of the same type and dimensions
    io::DenFileInfo disub(input_subtrahend);
    io::DenFileInfo dimin(input_minuend);
    if(disub.getNumCols() != dimin.getNumCols() || disub.getNumRows() != dimin.getNumRows()
       || disub.getNumSlices() != dimin.getNumSlices()
       || disub.getDataType() != dimin.getDataType())
    {
        LOGE << io::xprintf("The files %s and %s are uncompatible.\nFile %s of the type %s has "
                            "dimensions (x, y, z) = (%d, %d, %d).\nFile %s of the type %s has "
                            "dimensions (x, y, z) = (%d, %d, %d).",
                            input_subtrahend, input_minuend, input_subtrahend,
                            io::DenSupportedTypeToString(disub.getDataType()), disub.getNumCols(),
                            disub.getNumRows(), disub.getNumSlices(), input_minuend,
                            io::DenSupportedTypeToString(dimin.getDataType()), dimin.getNumCols(),
                            dimin.getNumRows(), dimin.getNumSlices());
        return 1;
    }
    return 0;
}
