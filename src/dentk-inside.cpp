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
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/Program.hpp"

#define PI 3.14159265

using namespace CTL;
using namespace CTL::util;

// class declarations
// class declarations
class Args : public ArgumentsFramespec, public ArgumentsForce
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsFramespec(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName){};
    std::string input_den = "";
    std::string output_den = "";
    float stopMax = std::numeric_limits<float>::infinity();
    float stopMin = -std::numeric_limits<float>::infinity();
    float scale = 1.0;
};

template <typename T>
/**
* Returns the distance from the center in a voxel cut in which there is either maximum or the value lt stopMin or gt stopMax.
*
* @param alpha
* @param stopMin
* @param stopMax
* @param A
*
* @return 
*/
int getMaxAttenuationDistance(double alpha,
                              float stopMin,
                              float stopMax,
                              std::shared_ptr<io::Frame2DI<T>> A)
{
    uint32_t dimx = A->dimx();
    uint32_t dimy = A->dimy();
    uint32_t max_x = dimx / 2;
    uint32_t max_y = dimy / 2;
    uint32_t max_r = 0;
    uint32_t cur_r = 0;
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
        if((int)x >= 0 && (int)x < (int)dimx && (int)y >= 0 && (int)y < (int)dimy)
        {
            double v = A->get(x, y);
            if(v > stopMax || v < stopMin)
            {
                break;
            }
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
            distancesFromCenter[alpha] = getMaxAttenuationDistance(alpha, a.stopMin, a.stopMax, A);
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
    Program PRG(argc, argv);
    // Argument parsing
    Args ARG(argc, argv,
             "Create file with ones inside the area from the center to the highest attenuation in "
             "given direction.");
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    io::DenFileInfo di(ARG.input_den);
    io::DenSupportedType dataType = di.getDataType();
    switch(dataType)
    {
    case io::DenSupportedType::uint16_t_:
    {
        processFiles<uint16_t>(ARG);
        break;
    }
    case io::DenSupportedType::float_:
    {
        processFiles<float>(ARG);
        break;
    }
    case io::DenSupportedType::double_:
    {
        processFiles<double>(ARG);
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

void Args::defineArguments()
{
    cliApp->add_option("input_den", input_den, "Input file.")->check(CLI::ExistingFile)->required();
    cliApp->add_option("output_den", output_den, "Output file.")->required();
    addForceArgs();
    cliApp->add_option("--scale", scale, "Scale size of the detected area.");
    cliApp->add_option(
        "--stop-max", stopMax,
        "Stop the search for the maximum when cliApp->oaching value greater than stop_max.");
    cliApp->add_option(
        "--stop-min", stopMin,
        "Stop the search for the maximum when cliApp->oaching value less than stop_min.");
}

int Args::postParse()
{
    if(!force)
    {
        if(io::pathExists(output_den))
        {
            LOGE << "Error: output file already exists, use -f to force overwrite.";
            return 1;
        }
    }
    return 0;
}
