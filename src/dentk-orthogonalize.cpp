// Logging
#include "PLOG/PlogSetup.h"

// External libraries

// External libraries
#include "CLI/CLI.hpp"
#include "matplotlibcpp.h"
#include "mkl.h"

// Internal libraries
#include "AsyncFrame2DWritterI.hpp"
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "FUN/StepFunction.hpp"
#include "Frame2DI.hpp"
#include "MATRIX/Matrix.hpp"
#include "MATRIX/RQFactorization.hpp"
#include "frameop.h"
#include "rawop.h"

namespace plt = matplotlibcpp;

using namespace KCT;

// class declarations
struct Args
{
    int parseArguments(int argc, char* argv[]);
    std::string input_file;
    std::string output_file = "";
    bool force;
    uint16_t granularity;
    uint16_t baseSize;
};

template <typename T>
/**
 * Orthogonalization by RQ decomposition from the last vector. So we reorder.
 *
 * @param inputFile
 * @param outputFile
 */
void orthonormalize(std::string inputFile, std::string outputFile)
{
    std::shared_ptr<io::Frame2DReaderI<T>> denSliceReader
        = std::make_shared<io::DenFrame2DReader<T>>(inputFile);
    uint32_t granularity = denSliceReader->dimx();
    uint32_t baseSize = denSliceReader->dimz();
    std::shared_ptr<io::Frame2DI<T>> f;
    io::BufferedFrame2D<T> bf(T(0.0), granularity, 1);
    double* values = new double[baseSize * granularity];
    for(uint32_t i = 0; i != baseSize; i++)
    {
        f = denSliceReader->readFrame(baseSize - 1 - i);
        for(uint32_t j = 0; j != granularity; j++)
        {
            values[i * granularity + j] = double(f->get(j, 0));
        }
    }
    matrix::RQFactorization rq;
    std::shared_ptr<matrix::Matrix> B
        = std::make_shared<matrix::Matrix>(baseSize, granularity, values);
    rq.factorize(B);
    auto C = rq.getRMatrix(); // baseSize*baseSize
    auto Q = rq.getQMatrix(); // baseSize*granularity
    double a = 0.0;
    int writeIndex = 0;
    for(uint32_t i = 0; i != baseSize; i++)
    {
        a = 0.0;
        for(uint32_t j = 0; j != baseSize; j++)
        {
            a += C->get(j, i) * C->get(j, i);
        }
        if(std::sqrt(a) > 1e-10)
        {
            writeIndex++;
        }
    }
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter
        = std::make_shared<io::DenAsyncFrame2DWritter<T>>(outputFile, granularity, 1, writeIndex);
    writeIndex--;
    for(uint32_t i = 0; i != baseSize; i++)
    {
        a = 0.0;
        for(uint32_t j = 0; j != baseSize; j++)
        {
            a += C->get(j, i) * C->get(j, i);
        }
        if(std::sqrt(a) > 1e-10)
        {
            for(uint32_t j = 0; j != granularity; j++)
            {
                bf.set(Q->get(i, j), j, 0);
            }
            imagesWritter->writeFrame(bf, writeIndex--);
        }
    }
    delete[] values;
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
    io::DenFileInfo di(a.input_file);
    // int elementSize = di.elementByteSize();
    io::DenSupportedType t = di.getElementType();
    std::string elm = io::DenSupportedTypeToString(t);
    switch(t)
    {
    case io::DenSupportedType::UINT16: {
        orthonormalize<uint16_t>(a.input_file, a.output_file);
        break;
    }
    case io::DenSupportedType::FLOAT32: {
        orthonormalize<float>(a.input_file, a.output_file);
        break;
    }
    case io::DenSupportedType::FLOAT64: {
        orthonormalize<double>(a.input_file, a.output_file);
        break;
    }
    default:
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(t).c_str());
        KCTERR(errMsg);
    }
}

int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Engineered bases orthogonalization by QR algorithm, cut colinear vectors." };
    app.add_option("input_den_file", input_file, "File in a DEN format to process.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("output_den_file", output_file, "File to output.")->required();
    app.add_flag("-f,--force", force, "Force overwrite output file.");
    try
    {
        app.parse(argc, argv);
        if(!force)
        {
            if(io::pathExists(output_file))
            {
                std::string msg = io::xprintf(
                    "Error: output file %s already exists, use -f to force overwrite.",
                    output_file.c_str());
                LOGE << msg;
                return 1;
            }
        }
        io::DenFileInfo di(input_file);
        granularity = di.dimx();
        baseSize = di.dimz();
        if(di.dimy() != 1)
        {
            std::string msg = io::xprintf(
                "Error: input file %s has invalid y dimension that must be 1 and is %d!",
                input_file.c_str(), di.dimy());
            LOGE << msg;
            return 1;
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
            // Negative value should be returned
            return -1;
        }
    }

    return 0;
}
