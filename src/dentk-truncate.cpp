// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <algorithm>
#include <cstdlib>
#include <ctype.h>
#include <iostream>
#include <regex>
#include <string>

// External libraries
#include "CLI/CLI.hpp" //Command line parser
#include "ctpl_stl.h" //Threadpool

// Internal libraries
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "Frame2DI.hpp"
#include "Frame2DReaderI.hpp"
#include "FrameMemoryViewer2D.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "PROG/parseArgs.h"

using namespace CTL;

// Function declarations (definition at the end of the file)

// class declarations
struct Args
{
    int parseArguments(int argc, char* argv[]);
    std::string input_projections;
    std::string input_matrices;
    std::string output_projections;
    std::string output_matrices;
    uint16_t threads = 0;
    uint32_t left_cut = 0, bottom_cut = 0, right_cut = 0, top_cut = 0;
    bool force;
};

template <typename T>
void writeReducedFrame(int id,
                       uint16_t k,
                       std::shared_ptr<io::Frame2DReaderI<T>> denFrameReader,
                       std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter,
                       Args a)
{
    uint16_t dimx_new = denFrameReader->dimx() - a.left_cut - a.right_cut;
    uint16_t dimy_new = denFrameReader->dimy() - a.bottom_cut - a.top_cut;
    std::shared_ptr<io::Frame2DI<T>> origf = denFrameReader->readFrame(k);
    io::BufferedFrame2D<T> f(T(0), dimx_new, dimy_new);
    for(uint16_t i = a.left_cut; i != denFrameReader->dimx() - a.right_cut; i++)
    {
        for(uint16_t j = a.top_cut; j != denFrameReader->dimy() - a.bottom_cut; j++)
        {
            f.set(origf->get(i, j), i - a.left_cut, j - a.top_cut);
        }
    }
    imagesWritter->writeFrame(f, k);
}

void writeShiftedCammat(int id,
                        uint16_t k,
                        std::shared_ptr<io::DenProjectionMatrixReader> dcr,
                        std::shared_ptr<io::AsyncFrame2DWritterI<double>> cmw,
                        Args a)
{
    matrix::ProjectionMatrix pm = dcr->readMatrix(k);
    matrix::ProjectionMatrix newpm = pm.shiftDetectorOrigin(double(a.left_cut), double(a.top_cut));
    io::FrameMemoryViewer2D<double> fmw(newpm.getPtr(), 4, 3);
    cmw->writeFrame(fmw, k);
}

template <typename T>
void dentkTruncate(Args a)
{
    io::DenFileInfo di(a.input_projections);
    uint16_t dimx = di.dimx();
    uint16_t dimy = di.dimy();
    uint16_t dimz = di.dimz();
    uint16_t dimx_new = dimx - a.left_cut - a.right_cut;
    uint16_t dimy_new = dimy - a.bottom_cut - a.top_cut;
    ctpl::thread_pool* threadpool = nullptr;
    if(a.threads != 0)
    {
        threadpool = new ctpl::thread_pool(a.threads);
    }
    std::shared_ptr<io::Frame2DReaderI<T>> denFrameReader
        = std::make_shared<io::DenFrame2DReader<T>>(a.input_projections);
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter
        = std::make_shared<io::DenAsyncFrame2DWritter<T>>(a.output_projections, dimx_new, dimy_new,
                                                          dimz);
    std::shared_ptr<io::DenProjectionMatrixReader> dcr = nullptr;
    std::shared_ptr<io::AsyncFrame2DWritterI<double>> dcw = nullptr;
    if(!a.input_matrices.empty())
    {
        dcr = std::make_shared<io::DenProjectionMatrixReader>(a.input_matrices);
        dcw = std::make_shared<io::DenAsyncFrame2DWritter<double>>(a.output_matrices, 4, 3, dimz);
    }
    for(uint32_t k = 0; k != dimz; k++)
    {
        if(threadpool != nullptr)
        {
            threadpool->push(writeReducedFrame<T>, k, denFrameReader, imagesWritter, a);
            if(dcw != nullptr)
            {
                threadpool->push(writeShiftedCammat, k, dcr, dcw, a);
            }

        } else
        {
            writeReducedFrame<T>(0, k, denFrameReader, imagesWritter, a);
            if(dcw != nullptr)
            {
                writeShiftedCammat(0, k, dcr, dcw, a);
            }
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
    io::DenFileInfo di(a.input_projections);
    io::DenSupportedType dataType = di.getDataType();
    switch(dataType)
    {
    case io::DenSupportedType::uint16_t_:
    {
        dentkTruncate<uint16_t>(a);
        break;
    }
    case io::DenSupportedType::float_:
    {
        dentkTruncate<float>(a);
        break;
    }
    case io::DenSupportedType::double_:
    {
        dentkTruncate<double>(a);
        break;
    }
    default:
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(dataType));
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
    }

    LOGI << io::xprintf("END %s", argv[0]);
}

int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Truncate projections and adjust projection matrices." };
    app.add_option("-j,--threads", threads,
                   "Number of extra threads that application can use 0 for turn off threading.")
        ->check(CLI::Range(0, 65535));
    app.add_option("--left_cut", left_cut, "Crop the image from the left by this number of pixels.")
        ->check(CLI::Range(0, 65535));
    app.add_option("--bottom_cut", bottom_cut,
                   "Crop the image from the bottom by this number of pixels.")
        ->check(CLI::Range(0, 65535));
    app.add_option("--right_cut", right_cut,
                   "Crop the image from the right by this number of pixels.")
        ->check(CLI::Range(0, 65535));
    app.add_option("--top_cut", top_cut, "Crop the image from the top by this number of pixels.")
        ->check(CLI::Range(0, 65535));
    app.add_option("input_projections", input_projections, "Projections in a DEN format.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("output_projections", output_projections,
                   "Projections in a DEN format to output.")
        ->required();
    CLI::Option* inpCM
        = app.add_option("--input_matrices", input_matrices, "Projection matrices in a DEN format.")
              ->check(CLI::ExistingFile);
    CLI::Option* outCM = app.add_option("--output_matrices", output_matrices,
                                        "Projection matrices in a DEN format to output.");
    inpCM->needs(outCM);
    outCM->needs(inpCM);
    app.add_flag("-f,--force", force, "Overwrite outputFile if it exists.");

    try
    {
        app.parse(argc, argv);
        io::DenFileInfo inf(input_projections);
        uint16_t dimx = inf.dimx();
        uint16_t dimy = inf.dimy();
        uint16_t dimz = inf.dimz();
        if(!input_matrices.empty())
        {
            io::DenFileInfo ima(input_matrices);
            if(dimz != ima.dimz())
            {
                LOGE << io::xprintf("Projections and matrices do not have the same z dimension.");
                return -1;
            }
            if(ima.dimx() != 4 || ima.dimy() != 3)
            {
                LOGE << io::xprintf(
                    "Matrices in file %s do not have proper dimension 3x4 but %dx%d.",
                    input_matrices.c_str(), ima.dimy(), ima.dimx());
                return -1;
            }
        }
        if(!force)
        {
            if(io::pathExists(output_projections))
            {
                std::string msg = io::xprintf("Error: output projections file %s already exists, "
                                              "use --force to force overwrite.",
                                              input_projections.c_str());
                LOGE << msg;
                return -1;
            }
            if(!input_matrices.empty() && io::pathExists(output_matrices))
            {
                std::string msg = io::xprintf("Error: output matrices file %s already exists, use "
                                              "--force to force overwrite.",
                                              input_matrices.c_str());
                LOGE << msg;
                return -1;
            }
        } else
        {
            if(input_projections == output_projections)
            {
                std::string msg = io::xprintf("Error: output projections file %s could not be same "
                                              "as input projections file.",
                                              input_projections.c_str());
                LOGE << msg;
                return -1;
            }
            if(!input_matrices.empty() && input_matrices == output_matrices)
            {
                std::string msg = io::xprintf(
                    "Error: output matrices file %s could not be same as input projections file.",
                    input_matrices.c_str());
                LOGE << msg;
                return -1;
            }
        }
        if(left_cut + right_cut >= dimx)
        {
            LOGE << io::xprintf(
                "Size of left_cut %d and right_cut %d could not sum equal or more than dimx %d.",
                left_cut, right_cut, dimx);
            return -1;
        }
        if(top_cut + bottom_cut >= dimy)
        {
            LOGE << io::xprintf(
                "Size of top_cut %d and bottom_cut %d cout not sum equal or more than dimy %d.",
                top_cut, bottom_cut, dimy);
            return -1;
        }
    } catch(const CLI::ParseError& e)
    {
        int exitcode = app.exit(e);
        if(exitcode == 0) // Help message was printed
        {
            return 1;
        } else
        {
            LOGE << "Parse error catched";
            return -1;
        }
    }
    return 0;
}
