// Logging
#include "PLOG/PlogSetup.h"
// External libraries
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctype.h>
#include <experimental/filesystem>
#include <iostream>
#include <regex>
#include <string>
// External libraries
#include "CLI/CLI.hpp" //Command line parser
// Internal libraries
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
using namespace KCT;
namespace fs = std::experimental::filesystem;
struct Args
{
    int parseArguments(int argc, char* argv[]);
    std::string input_file = "";
    std::string output_file = "";
    float water_value = 0.027;
    /// If the minimum value in hounsfield scale is 1024, then base2=true
    bool base2 = false;
};

int main(int argc, char* argv[])
{
    plog::Severity verbosityLevel = plog::debug; // debug, info, ...
    std::string csvLogFile = io::xprintf(
        "/tmp/%s.csv", io::getBasename(std::string(argv[0])).c_str()); // Set NULL to disable
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
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

    io::DenFileInfo di(a.input_file);
    int dimx = di.getNumCols();
    int dimy = di.getNumRows();
    int dimz = di.getNumSlices();
    io::DenSupportedType dataType = di.getDataType();
    switch(dataType)
    {
    case io::DenSupportedType::UINT16: {
        std::shared_ptr<io::AsyncFrame2DWritterI<uint16_t>> outputWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<uint16_t>>(a.output_file, dimx, dimy,
                                                                     dimz);
        std::shared_ptr<io::Frame2DI<uint16_t>> x
            = std::make_shared<io::BufferedFrame2D<uint16_t>>(0, dimx, dimy);
        double N;
        if(a.base2)
        {
            N = 1024.0;
        } else
        {
            N = 1000.0;
        }
        // N / W
        double NW = N / a.water_value;
        std::shared_ptr<io::Frame2DReaderI<uint16_t>> sliceReader
            = std::make_shared<io::DenFrame2DReader<uint16_t>>(a.input_file);
        for(int k = 0; k != dimz; k++)
        {
            std::shared_ptr<io::Frame2DI<uint16_t>> f = sliceReader->readFrame(k);
            for(int i = 0; i != dimx; i++)
                for(int j = 0; j != dimy; j++)
                {
                    double UINT16MAX = 65535;
                    double v = (double)f->get(i, j); // From uint16_t
                    double hu = v * NW + 0.5; // Do not offset
                    if(hu < 0.0)
                    {
                        x->set(0, i, j); // Do not offset
                    } else if(hu > UINT16MAX)
                    {
                        x->set(65535, i, j);
                    } else
                    {
                        x->set((uint16_t)hu, i, j);
                    }
                }
            outputWritter->writeFrame(*x, k);
        }
        break;
    }
    case io::DenSupportedType::FLOAT32: {
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> outputWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(a.output_file, dimx, dimy, dimz);
        std::shared_ptr<io::Frame2DI<float>> x
            = std::make_shared<io::BufferedFrame2D<float>>(0.0, dimx, dimy);
        float N;
        if(a.base2)
        {
            N = 1024.0;
        } else
        {
            N = 1000.0;
        }
        // N / W
        float NW = N / a.water_value;
        std::shared_ptr<io::Frame2DReaderI<float>> sliceReader
            = std::make_shared<io::DenFrame2DReader<float>>(a.input_file);
        for(int k = 0; k != dimz; k++)
        {
            std::shared_ptr<io::Frame2DI<float>> f = sliceReader->readFrame(k);
            for(int i = 0; i != dimx; i++)
                for(int j = 0; j != dimy; j++)
                {
                    float v = f->get(i, j); // From uint16_t
                    x->set(v * NW - N, i, j);
                }
            outputWritter->writeFrame(*x, k);
        }
        break;
    }
    case io::DenSupportedType::FLOAT64: {
        std::shared_ptr<io::AsyncFrame2DWritterI<double>> outputWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<double>>(a.output_file, dimx, dimy, dimz);
        std::shared_ptr<io::Frame2DI<double>> x
            = std::make_shared<io::BufferedFrame2D<double>>(0.0, dimx, dimy);
        double N;
        if(a.base2)
        {
            N = 1024.0;
        } else
        {
            N = 1000.0;
        }
        // N / W
        double NW = N / a.water_value;
        std::shared_ptr<io::Frame2DReaderI<double>> sliceReader
            = std::make_shared<io::DenFrame2DReader<double>>(a.input_file);
        for(int k = 0; k != dimz; k++)
        {
            std::shared_ptr<io::Frame2DI<double>> f = sliceReader->readFrame(k);
            for(int i = 0; i != dimx; i++)
                for(int j = 0; j != dimy; j++)
                {
                    double v = f->get(i, j); // From double
                    x->set(v * NW - N, i, j);
                }
            outputWritter->writeFrame(*x, k);
        }
        break;
    }
    default:
        std::string errMsg = io::xprintf("Unsupported data type %s.",
                                         io::DenSupportedTypeToString(dataType).c_str());
        KCTERR(errMsg);
    }
}

int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{
        "Convert DEN file with data in nonscaled to the Hounsfield units."
        "For input_file containing floats or doubles coefs = coefs*(1000/w)-1000."
        "For uint16 coefs = coefs*(1000/w). Default w=0.027. Ouput den file contains floats."
    };
    app.add_option("den_file", input_file, "File in a DEN format in HU to process.")
        ->check(CLI::ExistingFile);
    app.add_option("output_DICOM_dir", output_file, "File in a float DEN format after conversion.")
        ->required();
    app.add_option("-w,--water-value", water_value, "Water value to use, defaults to 0.027")
        ->check(CLI::Range(0.0, 1.0));
    app.add_flag("--base2", base2,
                 "If this flag is specified than the minimum value in HU is -1024.");
    try
    {
        app.parse(argc, argv);
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
