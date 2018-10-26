// Logging
#include <utils/PlogSetup.h>

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
#include "strtk.hpp"

// Internal libraries
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"

using namespace CTL;

// class declarations
struct Args
{
    int parseArguments(int argc, char* argv[]);
    float h = 1.0;
    std::string input_file = "";
    std::string output_x = "", output_y = "", output_z = "";
    std::string frames = "";
    bool force = false;
};

std::string program_name = "";

std::vector<int> processResultingFrames(std::string frameSpecification, int dimz)
{
    // Remove spaces
    for(int i = 0; i < frameSpecification.length(); i++)
        if(frameSpecification[i] == ' ')
            frameSpecification.erase(i, 1);
    frameSpecification = std::regex_replace(frameSpecification, std::regex("end"),
                                            io::xprintf("%d", dimz - 1).c_str());
    std::vector<int> frames;
    if(frameSpecification.empty())
    {
        for(int i = 0; i != dimz; i++)
            frames.push_back(i);
    } else
    {
        std::list<std::string> string_list;
        strtk::parse(frameSpecification, ",", string_list);
        auto it = string_list.begin();
        while(it != string_list.end())
        {
            size_t numRangeSigns = std::count(it->begin(), it->end(), '-');
            if(numRangeSigns > 1)
            {
                std::string msg = io::xprintf("Wrong number of range specifiers in the string %s.",
                                              (*it).c_str());
                LOGE << msg;
                throw std::runtime_error(msg);
            } else if(numRangeSigns == 1)
            {
                std::vector<int> int_vector;
                strtk::parse((*it), "-", int_vector);
                if(0 <= int_vector[0] && int_vector[0] <= int_vector[1] && int_vector[1] < dimz)
                {
                    for(int k = int_vector[0]; k != int_vector[1] + 1; k++)
                    {
                        frames.push_back(k);
                    }
                } else
                {
                    std::string msg
                        = io::xprintf("String %s is invalid range specifier.", (*it).c_str());
                    LOGE << msg;
                    throw std::runtime_error(msg);
                }
            } else
            {
                int index = atoi(it->c_str());
                if(0 <= index && index < dimz)
                {
                    frames.push_back(index);
                } else
                {
                    std::string msg = io::xprintf(
                        "String %s is invalid specifier for the value in the range [0,%d).",
                        (*it).c_str(), dimz);
                    LOGE << msg;
                    throw std::runtime_error(msg);
                }
            }
            it++;
        }
    }
    return frames;
}

int main(int argc, char* argv[])
{
    plog::Severity verbosityLevel
        = plog::debug; // Set to debug to see the debug messages, info messages
    std::string csvLogFile = "/tmp/dentk-grad.csv"; // Set NULL to disable
    bool logToConsole = true;
    util::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    // Process arguments
    program_name = argv[0];
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
    std::vector<int> framesToOutput;
    framesToOutput = processResultingFrames(a.frames, dimz);
    switch(dataType)
    {
    case io::DenSupportedType::float_:
    {
        std::shared_ptr<io::Frame2DReaderI<float>> reader
            = std::make_shared<io::DenFrame2DReader<float>>(a.input_file);
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> ox
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(a.output_x, dimx, dimy,
                                                                  framesToOutput.size());
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> oy
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(a.output_y, dimx, dimy,
                                                                  framesToOutput.size());
        std::shared_ptr<io::AsyncFrame2DWritterI<float>> oz
            = std::make_shared<io::DenAsyncFrame2DWritter<float>>(a.output_z, dimx, dimy,
                                                                  framesToOutput.size());
        io::BufferedFrame2D<float> x(nullptr, dimx, dimy);
        io::BufferedFrame2D<float> y(nullptr, dimx, dimy);
        io::BufferedFrame2D<float> z(nullptr, dimx, dimy);
        int k;
        for(int ind = 0; ind != framesToOutput.size(); ind++)
        {
            k = framesToOutput[ind];
            std::shared_ptr<io::Frame2DI<float>> f = reader->readFrame(k);
            std::shared_ptr<io::Frame2DI<float>> fprev, fnext;
            if(k > 0)
            {
                fprev = reader->readFrame(k - 1);
            } else
            {
                fprev = std::make_shared<io::BufferedFrame2D<float>>(0.0, dimx, dimy);
            }
            if(k < dimz - 1)
            {
                fnext = reader->readFrame(k + 1);
            } else
            {
                fnext = std::make_shared<io::BufferedFrame2D<float>>(0.0, dimx, dimy);
            }

            for(int i = 0; i != dimx; i++)
                for(int j = 0; j != dimy; j++)
                {

                    float centralDifferenceX = (i + 1 == dimx ? 0.0 : f->get(i + 1, j))
                        - (i - 1 == -1 ? 0.0 : f->get(i - 1, j));
                    float centralDifferenceY = (j + 1 == dimy ? 0.0 : f->get(i, j + 1))
                        - (j - 1 == -1 ? 0.0 : f->get(i, j - 1));
                    float centralDifferenceZ = fnext->get(i, j) - fprev->get(i, j);
                    x.set(centralDifferenceX / (2 * a.h), i, j);
                    y.set(centralDifferenceY / (2 * a.h), i, j);
                    z.set(centralDifferenceZ / (2 * a.h), i, j);
                }
            ox->writeFrame(x, k);
            oy->writeFrame(y, k);
            oz->writeFrame(z, k);
        }
    }
    break;
    default:
    {
        std::string errMsg
            = io::xprintf("Unsupported data type %s, currently only float data type is supported.",
                          io::DenSupportedTypeToString(dataType));
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
    }
    }
}

int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Subtract two DEN files with the same dimensions from each other." };
    app.add_flag("-f,--force", force, "Force rewrite output file if it exists.");
    app.add_option("-k,--frames", frames,
                   "Specify only particular frames to process. You can input range i.e. 0-20 or "
                   "also individual coma separated frames i.e. 1,8,9. Order does matter. Accepts "
                   "end literal that means total number of slices of the input.");

    CLI::Option* ox = app.add_option("--output-x", output_x, "Derivative x direction.");
    CLI::Option* oy = app.add_option("--output-y", output_y, "Derivative y direction.");
    CLI::Option* oz = app.add_option("--output-z", output_z, "Derivative z direction.");
    ox->needs(oy)->needs(oz);
    oy->needs(ox)->needs(oz);
    oz->needs(ox)->needs(oy);
    app.add_option("input_file", input_file, "File to compute gradient from.")
        ->check(CLI::ExistingFile);
    app.add_option("--pixel-spacing", h, "Spacing of pixels, defaults to 1.0.");
    try
    {
        app.parse(argc, argv);
    if(output_x.empty())
    {
        std::string prefix = input_file.substr(0, input_file.find(".", 0));
        output_x = io::xprintf("%s_x.den", prefix.c_str());
        output_y = io::xprintf("%s_y.den", prefix.c_str());
        output_z = io::xprintf("%s_z.den", prefix.c_str());
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
    if(!force)
    {
        if(io::fileExists(output_x) || io::fileExists(output_y) || io::fileExists(output_z))
        {
            LOGE << "Error: output file already exists, use -f to force overwrite.";
            return 1;
        }
    }
    return 0;
}
