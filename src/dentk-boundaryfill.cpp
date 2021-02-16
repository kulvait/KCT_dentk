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

using namespace CTL;
using namespace CTL::util;

// class declarations
// class declarations
class Args : public ArgumentsForce
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName){};
    std::string input_den = "";
    std::string output_den = "";
    float stopMax = std::numeric_limits<float>::infinity();
    float stopMin = -std::numeric_limits<float>::infinity();
    float scale = 1.0;
    float seedSearchConstraint = 1.0;
    uint32_t dimX, dimY, dimZ;
    uint32_t seedX;
    uint32_t seedY;
    uint32_t seedZ;
    bool additionalSeeds = false;
    int additionalSeedsOffset = 10;
};

void Args::defineArguments()
{
    cliApp->add_option("input_den", input_den, "Input file.")->check(CLI::ExistingFile)->required();
    cliApp->add_option("output_den", output_den, "Output file.")->required();
    addForceArgs();
    cliApp->add_option("--scale", scale, "Scale size of the detected area.")
        ->check(CLI::Range(0.0, 1.0));
    cliApp->add_option(
        "--stop-max", stopMax,
        "Stop the search for the maximum when approaching value greater than stop_max.");
    cliApp->add_option(
        "--stop-min", stopMin,
        "Stop the search for the maximum when approaching value less than stop_min.");
    cliApp
        ->add_option("--seed-constraint", seedSearchConstraint,
                     "Seed of the next layer only inside given circle, relative to the half of the "
                     "diagonal radius.")
        ->check(CLI::Range(0.0, 1.0));
    cliApp->add_option("--seed-x", seedX, "Coordinate of the seed x.");
    cliApp->add_option("--seed-y", seedY, "Coordinate of the seed y.");
    cliApp->add_option("--seed-z", seedZ, "Coordinate of the seed z.");
    CLI::Option* as_opt
        = cliApp->add_flag("--additional-seeds", additionalSeeds,
                           "Use additional seeds in the xy direction to protect for outliers.");
    std::string optstring
        = io::xprintf("Offset of additional seeds, defaults to %d.", additionalSeedsOffset);
    cliApp->add_option("--additional-seeds-offset", additionalSeedsOffset, optstring)
        ->needs(as_opt);
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
    dimX = di.dimx();
    dimY = di.dimy();
    dimZ = di.dimz();
    if(cliApp->count("--seed-x") == 0)
    {
        seedX = dimX / 2;
    }
    if(cliApp->count("--seed-y") == 0)
    {
        seedY = dimY / 2;
    }
    if(cliApp->count("--seed-z") == 0)
    {
        seedZ = dimZ / 2;
    }
    std::string err;
    if(seedX >= dimX)
    {
        err = io::xprintf("SeedX must %d be less than dimX %d!", seedX, dimX);
        LOGE << err;
        return -1;
    }
    if(seedY >= dimY)
    {
        err = io::xprintf("SeedX must %d be less than dimX %d!", seedX, dimX);
        LOGE << err;
        return -1;
    }
    if(seedZ >= dimZ)
    {
        err = io::xprintf("SeedX must %d be less than dimX %d!", seedX, dimX);
        LOGE << err;
        return -1;
    }
    return 0;
}

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
void seedBoundaryFill(const Args& a,
                      uint32_t x,
                      uint32_t y,
                      const std::shared_ptr<io::Frame2DI<T>>& seedF,
                      const std::shared_ptr<io::Frame2DI<T>>& alphaF)
{
    if(x < a.dimX && y < a.dimY)
    {
        LOGI << io::xprintf("Seed point [x,y,z] = [%d, %d, %d] with value %f", x, y, a.seedZ,
                            seedF->get(x, y));
        boundaryFill<T>(a, a.dimX, a.dimY, x, y, seedF, alphaF);
    }
}

template <typename T>
std::shared_ptr<io::Frame2DI<T>>
fillNewFrame(const Args& a,
             uint32_t k,
             std::shared_ptr<io::Frame2DReaderI<T>> denReader,
             std::shared_ptr<io::AsyncFrame2DWritterI<T>> outputWritter,
             std::shared_ptr<io::Frame2DI<T>> alphaFrame)
{
    float radiusSquare = 0.25 * (a.dimX * a.dimX + a.dimY * a.dimY);
    radiusSquare = radiusSquare * a.seedSearchConstraint * a.seedSearchConstraint;
    int lasti = -1;
    int lastj = -1;
    std::shared_ptr<io::Frame2DI<T>> currentFrame = denReader->readFrame(k);
    std::shared_ptr<io::Frame2DI<T>> newAlphaFrame
        = std::make_shared<io::BufferedFrame2D<T>>(T(0), a.dimX, a.dimY);
    for(uint32_t i = 0; i != a.dimX; i++)
    {
        for(uint32_t j = 0; j != a.dimX; j++)
        {
            float x0 = float(i) - float(a.seedX);
            float y0 = float(j) - float(a.seedY);
            if(x0 * x0 + y0 * y0 < radiusSquare)
            {
                double v = currentFrame->get(i, j);
                double va = alphaFrame->get(i, j);
                if(!(v > a.stopMax || v < a.stopMin) && va == 1.0)
                {
                    enquePoint(i, j);
                    lasti = i;
                    lastj = j;
                }
            }
        }
    }
    if(lasti != -1)
    {
        boundaryFill<T>(a, a.dimX, a.dimY, lasti, lastj, currentFrame, newAlphaFrame);
    }
    outputWritter->writeFrame(*newAlphaFrame, k);
    return newAlphaFrame;
}

template <typename T>
void processBoundaryFill(Args a)
{
    std::shared_ptr<io::Frame2DReaderI<T>> denReader
        = std::make_shared<io::DenFrame2DReader<T>>(a.input_den);
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> outputWritter
        = std::make_shared<io::DenAsyncFrame2DWritter<T>>(
            a.output_den, a.dimX, a.dimY,
            a.dimZ); // I write regardless to frame specification to original position
    std::shared_ptr<io::Frame2DI<T>> currentFrame = denReader->readFrame(a.seedZ);
    std::shared_ptr<io::BufferedFrame2D<T>> centerAlphaFrame
        = std::make_shared<io::BufferedFrame2D<T>>(T(0), a.dimX, a.dimY);
    seedBoundaryFill<T>(a, a.seedX, a.seedY, currentFrame, centerAlphaFrame);
    if(a.additionalSeeds)
    {
        seedBoundaryFill<T>(a, a.seedX + a.additionalSeedsOffset, a.seedY, currentFrame,
                            centerAlphaFrame);
        seedBoundaryFill<T>(a, a.seedX + a.additionalSeedsOffset, a.seedY + a.additionalSeedsOffset,
                            currentFrame, centerAlphaFrame);
        seedBoundaryFill<T>(a, a.seedX, a.seedY + a.additionalSeedsOffset, currentFrame,
                            centerAlphaFrame);
        seedBoundaryFill<T>(a, a.seedX - a.additionalSeedsOffset, a.seedY + a.additionalSeedsOffset,
                            currentFrame, centerAlphaFrame);
        seedBoundaryFill<T>(a, a.seedX - a.additionalSeedsOffset, a.seedY, currentFrame,
                            centerAlphaFrame);
        seedBoundaryFill<T>(a, a.seedX - a.additionalSeedsOffset, a.seedY - a.additionalSeedsOffset,
                            currentFrame, centerAlphaFrame);
        seedBoundaryFill<T>(a, a.seedX, a.seedY - a.additionalSeedsOffset, currentFrame,
                            centerAlphaFrame);
        seedBoundaryFill<T>(a, a.seedX - a.additionalSeedsOffset, a.seedY - a.additionalSeedsOffset,
                            currentFrame, centerAlphaFrame);
        seedBoundaryFill<T>(a, a.seedX, a.seedY - a.additionalSeedsOffset, currentFrame,
                            centerAlphaFrame);
    }
    outputWritter->writeFrame(*centerAlphaFrame, a.seedZ);

    std::shared_ptr<io::Frame2DI<T>> lastAlphaFrame = centerAlphaFrame;
    for(uint32_t ind = a.seedZ + 1; ind < a.dimZ; ind++)
    {
        lastAlphaFrame = fillNewFrame<T>(a, ind, denReader, outputWritter, lastAlphaFrame);
    }
    lastAlphaFrame = centerAlphaFrame;
    for(int ind = int(a.seedZ) - 1; ind > -1; ind--)
    {
        lastAlphaFrame = fillNewFrame<T>(a, ind, denReader, outputWritter, lastAlphaFrame);
    }
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    Args ARG(argc, argv, "Boundary fill algorithm to mask (using 0 and 1) particular region.");
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
        processBoundaryFill<uint16_t>(ARG);
        break;
    }
    case io::DenSupportedType::float_:
    {
        processBoundaryFill<float>(ARG);
        break;
    }
    case io::DenSupportedType::double_:
    {
        processBoundaryFill<double>(ARG);
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
