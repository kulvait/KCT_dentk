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
#include "CLI/CLI.hpp"

// Internal libraries
#include "ARGPARSE/parseArgs.h"
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "DEN/DenSupportedType.hpp"
#include "Frame2DI.hpp"
#include "Frame2DReaderI.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "frameop.h"

using namespace CTL;

// class declarations
struct Args
{
    int parseArguments(int argc, char* argv[]);
    std::string input_file;
    std::string frameSpecs = "";
    std::vector<int> frames;
    bool framesSpecified = false;
};

template <typename T>
void printFrameStatistics(const io::Frame2DI<T>& f)
{
    double min = (double)io::minFrameValue<T>(f);
    double max = (double)io::maxFrameValue<T>(f);
    double avg = io::meanFrameValue<T>(f);
    double l2norm = io::normFrame<T>(f, 2);
    std::cout << io::xprintf("\tMinimum, maximum, average values: %.3f, %0.3f, %0.3f.\n", min, max,
                             avg);
    std::cout << io::xprintf("\tEuclidean 2-norm of the frame: %E.\n", l2norm);
    int nonFiniteCount = io::sumNonfiniteValues<T>(f);
    if(nonFiniteCount == 0)
    {
        std::cout << io::xprintf("\tNo NAN or not finite number.\n\n");
    } else
    {
        std::cout << io::xprintf("\tThere is %d non finite numbers. \tFrom that %d NAN.\n\n",
                                 io::sumNonfiniteValues<T>(f), io::sumNanValues<T>(f));
    }
}

int main(int argc, char* argv[])
{
    plog::Severity verbosityLevel = plog::debug; // debug, info, ...
    std::string csvLogFile = io::xprintf(
        "/tmp/%s.csv", io::getBasename(std::string(argv[0])).c_str()); // Set NULL to disable
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    // Argument parsing
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
    LOGI << io::xprintf("START %s", argv[0]);
    if(a.framesSpecified)
    {
        std::shared_ptr<io::DenProjectionMatrixReader> dcr
            = std::make_shared<io::DenProjectionMatrixReader>(a.input_file);
        for(const int f : a.frames)
        {
            std::cout << io::xprintf("Camera matrix from %d-th frame:\n", f);
            matrix::ProjectionMatrix pm = dcr->readMatrix(f);
            std::cout << pm.toString();
        }
    }
}

int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Information about projection matrices in a DEN file." };
    app.add_option("-f,--frames", frameSpecs,
                   "Specify only particular frames to process. You can input range i.e. 0-20 or "
                   "also individual comma separated frames i.e. 1,8,9. Order does matter. Accepts "
                   "end literal that means total number of slices of the input.");
    app.add_option("input_camera_matrix", input_file, "File in a DEN format to process.")
        ->required()
        ->check(CLI::ExistingFile);
    try
    {
        app.parse(argc, argv);
        if(app.count("--frames") > 0)
        {
            framesSpecified = true;
        }
        io::DenFileInfo inf(
            input_file); // Print description of the file before potential throwing error
        int dimx = inf.dimx();
        int dimy = inf.dimy();
        int dimz = inf.dimz();
        // int elementSize = di.elementByteSize();
        io::DenSupportedType t = inf.getDataType();
        std::string elm = io::DenSupportedTypeToString(t);
        std::cout << io::xprintf(
            "The file %s of type %s has dimensions (x,y,z)=(cols,rows,slices)=(%d, "
            "%d, %d), each cell has x*y=%d pixels.\n",
            input_file.c_str(), elm.c_str(), dimx, dimy, dimz, dimx * dimy);
        if(inf.dimx() != 4 || inf.dimy() != 3)
        {
            io::throwerr("Provided file do not have correct dimensions 3x4 but %dx%d.", inf.dimx(),
                         inf.dimy());
        }
        if(t != io::DenSupportedType::double_)
        {
            io::throwerr("Camera matrix must be of the type double but it is of the type %s.",
                         DenSupportedTypeToString(t));
        }
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

    return 0;
}
