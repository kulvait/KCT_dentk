#define DEBUG
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
#include "FUN/LegendrePolynomial.h"
#include "FUN/StepFunction.hpp"
#include "Frame2DI.hpp"
#include "SPLINE/SplineFitter.hpp"
#include "frameop.h"
#include "rawop.h"

namespace plt = matplotlibcpp;

using namespace KCT;

// class declarations
struct Args
{
    int parseArguments(int argc, char* argv[]);
    std::string input_file;
    bool vizualize = false, info = false;
    std::string output_file = "";
    bool force;
    uint16_t resampledGranularity = 0;
    bool orthogonalize = false;
    int shiftBasis = 0;
    uint16_t granularity;
    uint16_t baseSize;
    uint16_t legendrePolynomialsAdded = 0;
};

template <typename T>
void shiftBasis(std::string inputFile, std::string outputFile, int shift)
{
    std::shared_ptr<io::Frame2DReaderI<T>> denSliceReader
        = std::make_shared<io::DenFrame2DReader<T>>(inputFile);
    uint16_t granularity = denSliceReader->dimx();
    uint16_t baseSize = denSliceReader->dimz();
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter
        = std::make_shared<io::DenAsyncFrame2DWritter<T>>(outputFile, granularity, 1, baseSize);
    std::shared_ptr<io::Frame2DI<T>> f;
    io::BufferedFrame2D<T> bf(0.0, granularity, 1);
    for(uint32_t i = 0; i != baseSize; i++)
    {
        f = denSliceReader->readFrame(i);
        for(int j = 0; j != granularity; j++)
        {
            int index = j + shift;
            if(index < 0)
            {
                index = 0;
            } else if(index >= granularity)
            {
                index = granularity - 1;
            }
            bf.set(f->get(index, 0), j, 0);
        }
        imagesWritter->writeFrame(bf, i);
    }
}

template <typename T>
void resampleFunctions(std::string inputFile, std::string outputFile, uint16_t resampledGranularity)
{
    std::shared_ptr<io::Frame2DReaderI<T>> denSliceReader
        = std::make_shared<io::DenFrame2DReader<T>>(inputFile);
    int granularity = denSliceReader->dimx();
    uint16_t baseSize = denSliceReader->dimz();
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter
        = std::make_shared<io::DenAsyncFrame2DWritter<T>>(outputFile, resampledGranularity, 1,
                                                          baseSize);
    std::shared_ptr<io::Frame2DI<T>> f;
    io::BufferedFrame2D<T> bf(T(0.0), resampledGranularity, 1);
    math::SplineFitter* sf = new math::SplineFitter(granularity, DF_PP_CUBIC, DF_PP_AKIMA);
    double* values = new double[baseSize * granularity];
    double* times = new double[granularity];
    double* newTimes = new double[resampledGranularity];
    double* newValues = new double[resampledGranularity];
    MKL_INT bc_type = DF_BC_1ST_LEFT_DER | DF_BC_1ST_RIGHT_DER;
    double bc[2] = { 0.0, 0.0 };
    for(int i = 0; i != baseSize; i++)
    {
        f = denSliceReader->readFrame(i);
        for(int j = 0; j != granularity; j++)
        {
            values[i * granularity + j] = double(f->get(j, 0));
        }
    }
    for(int j = 0; j != granularity; j++)
    {
        times[j] = double(j);
    }
    for(int j = 0; j != resampledGranularity; j++)
    {
        newTimes[j] = double(j) * double(granularity - 1) / double(resampledGranularity - 1);
    }

    for(uint16_t i = 0; i != baseSize; i++)
    {
        sf->buildSpline(times, &values[i * granularity], bc_type, bc);
        sf->interpolateAt(resampledGranularity, newTimes, newValues);
        for(uint32_t j = 0; j != resampledGranularity; j++)
        {
            bf.set(T(newValues[j]), j, 0);
        }
        imagesWritter->writeFrame(bf, i);
    }
    delete sf;
    delete[] newValues;
    delete[] values;
    delete[] times;
    delete[] newTimes;
}

template <typename T>
void addLegendrePolynomials(std::string inputFile, std::string outputFile, uint16_t n)
{
    std::shared_ptr<io::Frame2DReaderI<T>> denSliceReader
        = std::make_shared<io::DenFrame2DReader<T>>(inputFile);
    uint16_t granularity = denSliceReader->dimx();
    uint16_t baseSize = denSliceReader->dimz();
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter
        = std::make_shared<io::DenAsyncFrame2DWritter<T>>(outputFile, granularity, 1, n + baseSize);
    std::shared_ptr<io::Frame2DI<T>> f;
    io::BufferedFrame2D<T> bf(0.0, granularity, 1);
    util::LegendrePolynomial l(n - 1, 0.0, double(granularity - 1));
    double* values = new double[n * granularity];
    for(int j = 0; j != granularity; j++)
    {
        l.valuesAt(double(j), &values[j * n]);
    }
    for(uint32_t i = 0; i != n; i++)
    {
        for(int j = 0; j != granularity; j++)
        {
            bf.set(T(values[j * n + i]), j, 0);
        }
        imagesWritter->writeFrame(bf, i);
    }
    for(uint32_t i = 0; i != baseSize; i++)
    {
        f = denSliceReader->readFrame(i);
        imagesWritter->writeFrame(*f, n + i);
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
    if(a.info)
    {
        std::cout << io::xprintf(
            "The file %s of type %s contains %d basis functions with the discretization %d.\n",
            a.input_file.c_str(), elm.c_str(), a.baseSize, a.granularity);
    }
    if(a.vizualize)
    {
        //      LOGI << io::xprintf("Granularity of %s is %d", a.input_file.c_str(), a.granularity);
        util::StepFunction b(a.input_file, a.baseSize, 0.0, double(a.granularity - 1));
        b.plotFunctions(a.granularity);
    }
    switch(t)
    {
    case io::DenSupportedType::UINT16: {
        if(a.shiftBasis != 0)
        {
            shiftBasis<uint16_t>(a.input_file, a.output_file, a.shiftBasis);
        }
        if(a.resampledGranularity != 0)
        {
            resampleFunctions<uint16_t>(a.input_file, a.output_file, a.resampledGranularity);
        }
        if(a.legendrePolynomialsAdded != 0)
        {
            addLegendrePolynomials<uint16_t>(a.input_file, a.output_file,
                                             a.legendrePolynomialsAdded);
        }
        break;
    }
    case io::DenSupportedType::FLOAT32: {
        if(a.shiftBasis != 0)
        {
            shiftBasis<float>(a.input_file, a.output_file, a.shiftBasis);
        }
        if(a.resampledGranularity != 0)
        {
            resampleFunctions<float>(a.input_file, a.output_file, a.resampledGranularity);
        }
        if(a.legendrePolynomialsAdded != 0)
        {
            addLegendrePolynomials<float>(a.input_file, a.output_file, a.legendrePolynomialsAdded);
        }
        break;
    }
    case io::DenSupportedType::FLOAT64: {
        if(a.shiftBasis != 0)
        {
            shiftBasis<double>(a.input_file, a.output_file, a.shiftBasis);
        }
        if(a.resampledGranularity != 0)
        {
            resampleFunctions<double>(a.input_file, a.output_file, a.resampledGranularity);
        }
        if(a.legendrePolynomialsAdded != 0)
        {
            addLegendrePolynomials<double>(a.input_file, a.output_file, a.legendrePolynomialsAdded);
        }
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
    CLI::App app{ "Engineered bases processing. These bases are stored in a DEN file, where x is "
                  "granularity y is 1 and z is number of basis functions." };
    app.add_option("input_den_file", input_file, "File in a DEN format to process.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_flag("-v,--vizualize", vizualize, "Vizualize the basis.");
    app.add_flag("-i,--info", info, "Print information about the basis.");
    CLI::Option* _output = app.add_option("-o", output_file, "File to output.");
    CLI::Option* _force = app.add_flag("-f,--force", force, "Force overwrite output file.");
    _force->needs(_output);
    CLI::Option* _resample
        = app.add_option("--resample", resampledGranularity,
                         "Resample discretization to respect new granularity size.")
              ->check(CLI::Range(1, 65535));
    _resample->needs(_output);
    CLI::Option* _shift = app.add_option("--shift", shiftBasis,
                                         "Shift basis and make it continuous at boundaries. "
                                         "Specify coordinate of 0 in old basis coordinates.")
                              ->check(CLI::Range(-65535, 65535));
    _shift->needs(_output);
    CLI::Option* _addLegendre
        = app.add_option("--addLegendreBasis", legendrePolynomialsAdded,
                         "Add given number of members of Legendre polynomial basis. 1 means to add "
                         "constant, 2 means to add constant and linear function")
              ->check(CLI::Range(1, 65535));
    _addLegendre->needs(_output);
    _shift->excludes(_resample);
    _shift->excludes(_addLegendre);
    _resample->excludes(_shift);
    _resample->excludes(_addLegendre);
    _addLegendre->excludes(_resample);
    _addLegendre->excludes(_shift);
    try
    {
        app.parse(argc, argv);
        if(!force)
        {
            if(app.count("-o") > 0 && io::pathExists(output_file))
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
