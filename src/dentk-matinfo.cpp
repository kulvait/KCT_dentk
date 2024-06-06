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
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "DEN/DenGeometry3DParallelReader.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "DEN/DenSupportedType.hpp"
#include "Frame2DI.hpp"
#include "Frame2DReaderI.hpp"
#include "GEOMETRY/Geometry3DParallel.hpp"
#include "GEOMETRY/Geometry3DParallelI.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "PROG/KCTException.hpp"
#include "PROG/parseArgs.h"
#include "frameop.h"

using namespace KCT;

// class declarations
struct Args
{
    int parseArguments(int argc, char* argv[]);
    std::string input_file;
    bool CBMAT = false;
    bool PBMAT = false;
    bool estimateReconstructionSize = false;
    uint32_t detectorSize = 512;
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
    if(a.estimateReconstructionSize)
    {
        if(!a.PBMAT)
        {
            KCTERR("Estimation of the reconstruction size is only implemented for PBMAT.");
        }
        std::shared_ptr<io::DenGeometry3DParallelReader> geometryReader
            = std::make_shared<io::DenGeometry3DParallelReader>(a.input_file);
        std::shared_ptr<geometry::Geometry3DParallel> geometry;
        uint32_t matrixCount = geometryReader->count();
        double sum = 0.0;
        for(std::uint32_t i = 0; i < matrixCount; i++)
        {
            geometry
                = std::make_shared<geometry::Geometry3DParallel>(geometryReader->readGeometry(i));
            std::array<double, 8> pm = geometry->projectionMatrixAsVector8();
            sum += pm[3];
        }
        double COR = sum / matrixCount;
        double distToCor = std::max(a.detectorSize - COR - 0.5, COR + 0.5);
        double optimalDetectorSize = 2 * distToCor;
        int optimelDetectorSizeRounded = static_cast<int>(optimalDetectorSize + 0.5);
        //Let's make it multiple of 256
        int roundedDetectorSize = (optimelDetectorSizeRounded / 256 + 1) * 256;
        std::cout << io::xprintf(
            "%d pixels is the optimal detector size for COR=%f and detectorSize=%d.\n",
            roundedDetectorSize, COR, a.detectorSize);
        return 0;
    }
    LOGI << io::xprintf("START %s", argv[0]);
    if(a.framesSpecified)
    {
        if(a.CBMAT)
        {
            std::shared_ptr<io::DenProjectionMatrixReader> dcr
                = std::make_shared<io::DenProjectionMatrixReader>(a.input_file);
            for(const int f : a.frames)
            {
                std::cout << io::xprintf("CBCT matrix from %d-th frame:\n", f);
                matrix::ProjectionMatrix pm = dcr->readMatrix(f);
                std::cout << pm.toString();
            }
        } else if(a.PBMAT)
        {
            std::shared_ptr<io::DenGeometry3DParallelReader> geometryReader
                = std::make_shared<io::DenGeometry3DParallelReader>(a.input_file);
            std::shared_ptr<geometry::Geometry3DParallel> geometry;
            for(const int f : a.frames)
            {
                std::cout << io::xprintf("PBCT matrix from %d-th frame:\n", f);
                geometry = std::make_shared<geometry::Geometry3DParallel>(
                    geometryReader->readGeometry(f));
                std::array<double, 8> pm = geometry->projectionMatrixAsVector8();
                std::cout << io::xprintf("[[%f %f %f %f],[%f %f %f %f]]\n", pm[0], pm[1], pm[2],
                                         pm[3], pm[4], pm[5], pm[6], pm[7]);
            }
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
    app.add_flag("--estimate-reconstruction-size", estimateReconstructionSize,
                 "Estimate the size of the reconstruction from the projection matrices.");
    std::string optstring
        = io::xprintf("Size of the detector in pixels, defaults to %d.", detectorSize);
    app.add_option("--detector-size", detectorSize, optstring);
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
        io::DenSupportedType t = inf.getElementType();
        std::string elm = io::DenSupportedTypeToString(t);
        std::cout << io::xprintf(
            "The file %s of type %s has dimensions (x,y,z)=(cols,rows,slices)=(%d, "
            "%d, %d), each cell has x*y=%d pixels.\n",
            input_file.c_str(), elm.c_str(), dimx, dimy, dimz, dimx * dimy);
        std::string ERR;
        if(inf.dimx() == 4 && inf.dimy() == 3)
        {
            CBMAT = true;
        } else if(inf.dimx() == 4 && inf.dimy() == 2)
        {
            PBMAT = true;
        } else
        {
            ERR = io::xprintf("Provided file do not have correct dimensions to be CBMAT (3x4) or "
                              "PBMAT 2x4 but %dx%d.",
                              inf.dimy(), inf.dimx());
            KCTERR(ERR);
        }
        if(t != io::DenSupportedType::FLOAT64)
        {
            ERR = io::xprintf("Camera matrix must be of the type double but it is of the type %s.",
                              DenSupportedTypeToString(t).c_str());
            KCTERR(ERR);
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
