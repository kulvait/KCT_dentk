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
#include "ftpl.h" //Threadpool

// Internal libraries
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DBufferedWritter.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFrame2DCachedReader.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "DEN/DenProjectionMatrixReader.hpp"
#include "Frame2DI.hpp"
#include "Frame2DReaderI.hpp"
#include "FrameMemoryViewer2D.hpp"
#include "MATRIX/ProjectionMatrix.hpp"
#include "PROG/Arguments.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/Program.hpp"
#include "PROG/parseArgs.h"

using namespace KCT;
using namespace KCT::util;
using namespace KCT::io;

// Function declarations (definition at the end of the file)

// class declarations
class Args : public ArgumentsForce, public ArgumentsThreading
{
public:
    Args(int argc, char** argv, std::string programName)
        : Arguments(argc, argv, programName)
        , ArgumentsForce(argc, argv, programName)
        , ArgumentsThreading(argc, argv, programName){};
    int preParse() { return 0; };
    int postParse();
    void defineArguments();
    std::string input_projections;
    std::string input_matrices;
    std::string output_projections;
    std::string output_matrices;
    uint32_t left_cut = 0, bottom_cut = 0, right_cut = 0, top_cut = 0;
    uint32_t x_to, y_to;
    uint32_t dimx, dimy, dimz;
    uint32_t dimx_new, dimy_new;
    bool XMajorAlignment;
};

void Args::defineArguments()
{
    cliApp->add_option("input_projections", input_projections, "Projections in a DEN format.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp
        ->add_option("output_projections", output_projections,
                     "Projections in a DEN format to output.")
        ->required();
    CLI::Option* inpCM = cliApp
                             ->add_option("--input-matrices", input_matrices,
                                          "Projection matrices in a DEN format.")
                             ->check(CLI::ExistingFile);
    CLI::Option* outCM = cliApp->add_option("--output-matrices", output_matrices,
                                            "Projection matrices in a DEN format to output.");
    inpCM->needs(outCM);
    outCM->needs(inpCM);
    cliApp->add_option("--x-from,--left-cut", left_cut, "Crop x range [x_from, x_to).")
        ->check(CLI::NonNegativeNumber);
    CLI::Option* o_xt = cliApp->add_option("--x-to", x_to, "Crop x range [x_from, x_to).")
                            ->check(CLI::NonNegativeNumber);
    CLI::Option* o_rc = cliApp
                            ->add_option("--right-cut", right_cut,
                                         "Crop the image from the right by this number of pixels.")
                            ->check(CLI::NonNegativeNumber);
    cliApp->add_option("--y-from,--top-cut", top_cut, "Crop y range [y_from, y_to).")
        ->check(CLI::NonNegativeNumber);
    CLI::Option* o_yt = cliApp->add_option("--y-to", y_to, "Crop y range [y_from, y_to).")
                            ->check(CLI::NonNegativeNumber);
    CLI::Option* o_bc = cliApp
                            ->add_option("--bottom-cut", bottom_cut,
                                         "Crop the image from the bottom by this number of pixels.")
                            ->check(CLI::NonNegativeNumber);
    o_rc->excludes(o_xt);
    o_xt->excludes(o_rc);
    o_bc->excludes(o_yt);
    o_yt->excludes(o_bc);
    addThreadingArgs();
    addForceArgs();
}

int Args::postParse()
{
    io::DenFileInfo inf(input_projections);
    dimx = inf.dimx();
    dimy = inf.dimy();
    dimz = inf.dimz();
    XMajorAlignment = inf.hasXMajorAlignment();
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
            LOGE << io::xprintf("Matrices in file %s do not have proper dimension 3x4 but %dx%d.",
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
    if(cliApp->count("--x-to") > 0)
    {
        if(x_to > dimx)
        {
            LOGE << io::xprintf("Size of x_to %d can not exceed dimx %d.", x_to, dimx);
            return -1;
        }
        right_cut = dimx - x_to;
    }
    if(cliApp->count("--y-to") > 0)
    {
        if(y_to > dimy)
        {
            LOGE << io::xprintf("Size of y_to %d can not exceed dimy %d.", y_to, dimy);
            return -1;
        }
        bottom_cut = dimy - y_to;
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
    dimx_new = dimx - right_cut - left_cut;
    dimy_new = dimy - top_cut - bottom_cut;
    x_to = dimx - right_cut;
    y_to = dimy - bottom_cut;
    return 0;
}

template <typename T>
void writeReducedFrame(int id,
                       uint32_t k,
                       std::shared_ptr<io::DenFrame2DCachedReader<T>> denFrameReader,
                       std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>> imagesWritter,
                       std::shared_ptr<Args> a)
{
    //Frame is Xmajor structure by definition
    std::shared_ptr<io::BufferedFrame2D<T>> origf = denFrameReader->readBufferedFrame(k);
    io::BufferedFrame2D<T> f(T(0), a->dimx_new, a->dimy_new);
    T* inBuffer = origf->getDataPointer();
    T* outBuffer = f.getDataPointer();
    //Now I make strided copy
    T* inStridedStart;
    T* ouStridedStart;
    for(uint64_t y_new = 0; y_new != a->dimy_new; y_new++)
    {
        inStridedStart = inBuffer + (a->top_cut + y_new) * a->dimx + a->left_cut;
        ouStridedStart = outBuffer + y_new * a->dimx_new;
        std::copy(inStridedStart, inStridedStart + a->dimx_new, ouStridedStart);
    }
    imagesWritter->writeFrame(f, k);
}

void writeShiftedCammat(int id,
                        uint32_t k,
                        std::shared_ptr<io::DenProjectionMatrixReader> dcr,
                        std::shared_ptr<io::AsyncFrame2DWritterI<double>> cmw,
                        std::shared_ptr<Args> a)
{
    matrix::ProjectionMatrix pm = dcr->readMatrix(k);
    matrix::ProjectionMatrix newpm
        = pm.shiftDetectorOrigin(double(a->left_cut), double(a->top_cut));
    io::FrameMemoryViewer2D<double> fmw(newpm.getPtr(), 4, 3);
    cmw->writeFrame(fmw, k);
}

template <typename T>
void dentkTruncate(std::shared_ptr<Args> a)
{
    ftpl::thread_pool* threadpool = nullptr;
    if(a->threads != 0)
    {
        threadpool = new ftpl::thread_pool(a->threads);
    }
    std::shared_ptr<io::DenFrame2DCachedReader<T>> denFrameReader
        = std::make_shared<io::DenFrame2DCachedReader<T>>(a->input_projections, a->threads);
    std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>> imagesWritter
        = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(
            a->output_projections, a->dimx_new, a->dimy_new, a->dimz, a->XMajorAlignment);
    std::shared_ptr<io::DenProjectionMatrixReader> dcr = nullptr;
    std::shared_ptr<io::AsyncFrame2DWritterI<double>> dcw = nullptr;
    if(!a->input_matrices.empty())
    {
        dcr = std::make_shared<io::DenProjectionMatrixReader>(a->input_matrices);
        dcw = std::make_shared<io::DenAsyncFrame2DWritter<double>>(a->output_matrices, 4, 3,
                                                                   a->dimz);
    }
    for(uint32_t k = 0; k != a->dimz; k++)
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
    Program PRG(argc, argv);
    std::string prgInfo = "Truncate projections and if specified adjust projection matrices for CBCT.";
    std::shared_ptr<Args> ARG = std::make_shared<Args>(argc, argv, prgInfo);

    // Argument parsing
    int parseResult = ARG->parse(true);
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    io::DenFileInfo di(ARG->input_projections);
    io::DenSupportedType dataType = di.getElementType();
    switch(dataType)
    {
    case io::DenSupportedType::UINT16: {
        dentkTruncate<uint16_t>(ARG);
        break;
    }
    case io::DenSupportedType::FLOAT32: {
        dentkTruncate<float>(ARG);
        break;
    }
    case io::DenSupportedType::FLOAT64: {
        dentkTruncate<double>(ARG);
        break;
    }
    default:
        std::string errMsg = io::xprintf("Unsupported data type %s.",
                                         io::DenSupportedTypeToString(dataType).c_str());
        KCTERR(errMsg);
    }
    PRG.endLog(true);
}

