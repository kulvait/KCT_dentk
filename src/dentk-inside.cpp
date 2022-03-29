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
#include <tuple>

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

using namespace KCT;
using namespace KCT::util;

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
    float bfDiameter = 1.0;
    bool boundaryFill = false;
};

uint32_t level = 0;

template <typename T>
bool boundaryFlip(const Args& a,
                  uint32_t dimx,
                  uint32_t dimy,
                  uint32_t x,
                  uint32_t y,
                  const std::shared_ptr<io::Frame2DI<T>>& F,
                  const std::shared_ptr<io::Frame2DI<T>>& ALPHA)
{
    double v = F->get(x, y);
    double va = ALPHA->get(x, y);
    if(va != 1 && (!(v > a.stopMax || v < a.stopMin)))
    {
        ALPHA->set(T(1), x, y);
        return true;
    }
    return false;
}

using P = std::tuple<uint32_t, uint32_t>;
std::deque<P> processingQueue;
void enquePoint(io::BufferedFrame2D<bool>& visited, uint32_t x, uint32_t y)
{
    if(!visited.get(x, y))
    {
        P p({ x, y });
        processingQueue.emplace_back(p);
        visited.set(true, x, y);
    }
}

void enquePoint(uint32_t x, uint32_t y)
{
    P p({ x, y });
    processingQueue.emplace_back(p);
}

template <typename T>
void boundaryFill(const Args& a,
                  uint32_t dimx,
                  uint32_t dimy,
                  uint32_t x,
                  uint32_t y,
                  const std::shared_ptr<io::Frame2DI<T>>& F,
                  const std::shared_ptr<io::Frame2DI<T>>& ALPHA)
{
    io::BufferedFrame2D<bool> visited(false, dimx, dimy);
    enquePoint(visited, x, y);
    while(!processingQueue.empty())
    {
        uint32_t px, py;
        std::tie(px, py) = processingQueue[0];
        /*
                for(auto iter = processingQueue.begin(); iter != processingQueue.end(); ++iter)
                {
                    uint32_t a, b;
                    std::tie(a, b) = *iter;
                    std::cout << io::xprintf(",[%d, %d]", a, b);
                }
                std::cout << std::endl;
                LOGI << io::xprintf("Processing x=%d, y=%d, length=%d", px, py,
           processingQueue.size());
        */
        processingQueue.pop_front();
        double v = F->get(px, py);
        double va = ALPHA->get(px, py);
        if(va != 1 && (!(v > a.stopMax || v < a.stopMin)))
        {
            ALPHA->set(T(1), px, py);
            if(px != 0)
            {
                enquePoint(visited, px - 1, py);
            }
            if(px + 1 != dimx)
            {
                enquePoint(visited, px + 1, py);
            }
            if(py != 0)
            {
                enquePoint(visited, px, py - 1);
            }
            if(py + 1 != dimy)
            {
                enquePoint(visited, px, py + 1);
            }
        }
    }
}

template <typename T>
void processBoundaryFill(Args a)
{
    io::DenFileInfo di(a.input_den);
    uint32_t dimx = di.dimx();
    uint32_t dimy = di.dimy();
    uint32_t dimz = di.dimz();
    uint32_t center_x = dimx / 2;
    uint32_t center_y = dimy / 2;
    std::shared_ptr<io::Frame2DReaderI<T>> denReader
        = std::make_shared<io::DenFrame2DReader<T>>(a.input_den);
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> outputWritter
        = std::make_shared<io::DenAsyncFrame2DWritter<T>>(
            a.output_den, dimx, dimy,
            dimz); // I write regardless to frame specification to original position
    uint32_t centerFrameID = a.frames.size() / 2;
    uint32_t k = a.frames[centerFrameID];
    std::shared_ptr<io::Frame2DI<T>> A = denReader->readFrame(k);
    std::shared_ptr<io::BufferedFrame2D<T>> centerF
        = std::make_shared<io::BufferedFrame2D<T>>(T(0), dimx, dimy);
    LOGI << io::xprintf("Seed point [x,y,z] = [%d, %d, %d] with value %f", center_x, center_y, k,
                        A->get(center_x, center_y));
    LOGI << io::xprintf("Seed point [x,y,z] = [%d, %d, %d] with value %f", center_x + 10,
                        center_y + 10, k, A->get(center_x + 10, center_y + 10));
    LOGI << io::xprintf("Seed point [x,y,z] = [%d, %d, %d] with value %f", center_x - 10,
                        center_y - 10, k, A->get(center_x - 10, center_y - 10));
    LOGI << io::xprintf("Seed point [x,y,z] = [%d, %d, %d] with value %f", center_x + 10,
                        center_y - 10, k, A->get(center_x + 10, center_y - 10));
    LOGI << io::xprintf("Seed point [x,y,z] = [%d, %d, %d] with value %f", center_x - 10,
                        center_y + 10, k, A->get(center_x - 10, center_y + 10));
    boundaryFill<T>(a, dimx, dimy, center_x, center_y, A, centerF);
    boundaryFill<T>(a, dimx, dimy, center_x + 10, center_y + 10, A, centerF);
    boundaryFill<T>(a, dimx, dimy, center_x - 10, center_y - 10, A, centerF);
    boundaryFill<T>(a, dimx, dimy, center_x + 10, center_y - 10, A, centerF);
    boundaryFill<T>(a, dimx, dimy, center_x - 10, center_y + 10, A, centerF);
    outputWritter->writeFrame(*centerF, k);
    std::shared_ptr<io::Frame2DI<T>> lastF = centerF;
    float radiusSquare = 0.25 * (dimx * dimx + dimy * dimy);
    radiusSquare = radiusSquare * a.bfDiameter * a.bfDiameter;
    for(uint32_t ind = centerFrameID + 1; ind < a.frames.size(); ind++)
    {
        k = a.frames[ind];
        int lasti = -1;
        int lastj = -1;
        A = denReader->readFrame(k);
        for(uint32_t i = 0; i != dimx; i++)
        {
            for(uint32_t j = 0; j != dimy; j++)
            {
                float x0 = float(i) - float(center_x);
                float y0 = float(j) - float(center_y);
                if(x0 * x0 + y0 * y0 < radiusSquare)
                {
                    double v = A->get(i, j);
                    double va = lastF->get(i, j);
                    if(!(v > a.stopMax || v < a.stopMin) && va == 1.0)
                    {
                        enquePoint(i, j);
                        lasti = i;
                        lastj = j;
                    }
                }
            }
        }
        lastF = std::make_shared<io::BufferedFrame2D<T>>(T(0), dimx, dimy);
        if(lasti != -1)
        {
            boundaryFill<T>(a, dimx, dimy, lasti, lastj, A, lastF);
        }
        outputWritter->writeFrame(*lastF, k);
    }
    lastF = centerF;
    for(int ind = centerFrameID - 1; ind > -1; ind--)
    {
        k = a.frames[ind];
        int lasti = -1;
        int lastj = -1;
        A = denReader->readFrame(k);
        for(uint32_t i = 0; i != dimx; i++)
        {
            for(uint32_t j = 0; j != dimy; j++)
            {
                float x0 = float(i) - float(center_x);
                float y0 = float(j) - float(center_y);
                if(x0 * x0 + y0 * y0 < radiusSquare)
                {
                    double v = A->get(i, j);
                    double va = lastF->get(i, j);
                    if(!(v > a.stopMax || v < a.stopMin) && va == 1.0)
                    {
                        enquePoint(i, j);
                        lasti = i;
                        lastj = j;
                    }
                }
            }
        }
        lastF = std::make_shared<io::BufferedFrame2D<T>>(T(0), dimx, dimy);
        if(lasti != -1)
        {
            boundaryFill<T>(a, dimx, dimy, lasti, lastj, A, lastF);
        }
        outputWritter->writeFrame(*lastF, k);
    }
    /*
for(const int& k : a.frames)
{
    LOGI << io::xprintf("Creating frame %d", k);
    std::shared_ptr<io::Frame2DI<T>> f
        = std::make_shared<io::BufferedFrame2D<T>>(T(0), dimx, dimy);
    std::shared_ptr<io::Frame2DI<T>> A = denReader->readFrame(k);
    boundaryFill<T>(a, dimx, dimy, center_x, center_y, A, f);
    io::xprintf("Writing output");
}*/
}

template <typename T>
/**
 * Returns the distance from the center in a voxel cut in which there is either maximum or the
 * value lt stopMin or gt stopMax.
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
    if(a.boundaryFill)
    {
        processBoundaryFill<T>(a);
        return;
    }
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
    case io::DenSupportedType::UINT16: {
        processFiles<uint16_t>(ARG);
        break;
    }
    case io::DenSupportedType::FLOAT32: {
        processFiles<float>(ARG);
        break;
    }
    case io::DenSupportedType::FLOAT64: {
        processFiles<double>(ARG);
        break;
    }
    default: {
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(dataType).c_str());
        KCTERR(errMsg);
    }
    }
    PRG.endLog();
}

void Args::defineArguments()
{
    cliApp->add_option("input_den", input_den, "Input file.")->check(CLI::ExistingFile)->required();
    cliApp->add_option("output_den", output_den, "Output file.")->required();
    addForceArgs();
    addFramespecArgs();
    cliApp->add_option("--scale", scale, "Scale size of the detected area.")
        ->check(CLI::Range(0.0, 1.0));
    cliApp->add_option(
        "--stop-max", stopMax,
        "Stop the search for the maximum when approaching value greater than stop_max.");
    cliApp->add_option(
        "--stop-min", stopMin,
        "Stop the search for the maximum when approaching value less than stop_min.");
    cliApp->add_flag("--boundary-fill", boundaryFill, "Use boundary fill.");
    cliApp
        ->add_option("--boundary-fill-seed", bfDiameter,
                     "Seed of the next layer only inside given circle.")
        ->check(CLI::Range(0.0, 1.0));
}

int Args::postParse()
{
    if(!force)
    {
        if(io::pathExists(output_den))
        {
            LOGE << "Error: output file already exists, use --force to force overwrite.";
            return 1;
        }
    }
    io::DenFileInfo di(input_den);
    fillFramesVector(di.dimz());
    return 0;
}
