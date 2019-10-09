// The purpose of this tool is to filter out outer bone structures.
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

#define PI 3.14159265

using namespace CTL;

// class declarations
struct Args
{
    int parseArguments(int argc, char* argv[]);
    std::string input_den = "";
    std::string output_den = "";
    std::string frameSpecs = "";
    std::vector<int> frames;
    float scale = 1.0;
    bool force = false;
};

template <typename T>
int getMaxAttenuationDistance(double alpha, int dimx, int dimy, std::shared_ptr<io::Frame2DI<T>> A)
{
    int max_x = dimx / 2;
    int max_y = dimy / 2;
    int max_r = 0;
    int cur_r = 0;
    double maximum = A->get(max_x, max_y);
    double x = double(max_x);
    double y = double(max_y);
    double x_inc = std::cos(double(alpha) * PI / 180);
    double y_inc = std::sin(double(alpha) * PI / 180);
    while(true)
    {
        cur_r++;
        x += x_inc;
        y += y_inc;
        if((int)x >= 0 && (int)x < dimx && (int)y >= 0 && (int)y < dimy)
        {
            double v = A->get(x, y);
            if(v > maximum)
            {
                max_r = cur_r;
                maximum = v;
            }
        } else
        {
            break;
        }
    }
    return max_r;
}

template <typename T>
void processFiles(Args a)
{
    io::DenFileInfo di(a.input_den);
    int dimx = di.dimx();
    int dimy = di.dimy();
    int dimz = di.dimz();
    std::shared_ptr<io::Frame2DReaderI<T>> denReader
        = std::make_shared<io::DenFrame2DReader<T>>(a.input_den);
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> outputWritter
        = std::make_shared<io::DenAsyncFrame2DWritter<T>>(
            a.output_den, dimx, dimy,
            dimz); // I write regardless to frame specification to original position
    int* distancesFromCenter = new int[360]; // How far from center is the point in which the
                                             // attenuation is maximal for given angle
    int center_x = dimx / 2;
    int center_y = dimy / 2;
    for(const int& k : a.frames)
    {
        io::BufferedFrame2D<T> f(T(0), dimx, dimy);
        std::shared_ptr<io::Frame2DI<T>> A = denReader->readFrame(k);
        for(int alpha = 0; alpha != 360; alpha++)
        {
            distancesFromCenter[alpha] = getMaxAttenuationDistance(alpha, dimx, dimy, A);
        }
        for(int i = 0; i != dimx; i++)
        {
            for(int j = 0; j != dimy; j++)
            {
                double x = i - center_x;
                double y = j - center_y;
                int alpha = (int)(std::atan2(y, x) * 180.0 / PI);
                if(alpha < 0)
                {
                    alpha = alpha + 360;
                }
                int r = (int)std::sqrt(x * x + y * y);
                if(r < distancesFromCenter[alpha] * a.scale)
                {
                    f.set(T(1), i, j);
                }
            }
        }
        outputWritter->writeFrame(f, k);
    }
    // given angle attenuation is maximal
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
    io::DenFileInfo di(a.input_den);
    io::DenSupportedType dataType = di.getDataType();
    switch(dataType)
    {
    case io::DenSupportedType::uint16_t_:
    {
        processFiles<uint16_t>(a);
        break;
    }
    case io::DenSupportedType::float_:
    {
        processFiles<float>(a);
        break;
    }
    case io::DenSupportedType::double_:
    {
        processFiles<double>(a);
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
    LOGI << io::xprintf("END %s", argv[0]);
}

int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Create file with ones inside the area from the center to the highest "
                  "attenuation in given direction." };
    app.add_option("input_den", input_den, "Input file.")->check(CLI::ExistingFile)->required();
    app.add_option("output_den", output_den, "Output file.")->required();
    app.add_flag("--force", force, "Owerwrite output file if it exists.");
    app.add_option("-f,--frames", frameSpecs,
                   "Specify only particular frames to process. You can input range i.e. 0-20 or "
                   "also individual coma separated frames i.e. 1,8,9. Order does matter. Accepts "
                   "end literal that means total number of slices of the input.");
    app.add_option("--scale", scale, "Scale size of the detected area.");
    try
    {
        app.parse(argc, argv);
        LOGD << io::xprintf("Input file is %s and output file is %s.", input_den.c_str(),
                            output_den.c_str());
        io::DenFileInfo inf(input_den);
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
        if(io::fileExists(output_den))
        {
            LOGE << "Error: output file already exists, use -f to force overwrite.";
            return 1;
        }
    }
    return 0;
}
