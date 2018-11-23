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
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "DENITK/DenFrame2DReaderItk.hpp"
#include "DENITK/itkop.h"
#include "Frame2DReaderI.hpp"

using namespace CTL;

// class declarations
struct Args
{
    int parseArguments(int argc, char* argv[]);
    std::string input_file = "";
    std::string input_file_red = "";
    std::string input_file_green = "";
    std::string input_file_blue = "";
    std::string frameSpecs = "";
    std::vector<int> frames;
    std::string output_directory = "/tmp";
    std::string file_prefix = "slice";
    float min;
    float max;
    bool intervalSpecified = false;
    bool framesSpecified = false;
    bool colorsSpecified = false;
    bool grayscaleSpecified = false;
};

std::string program_name = "";

int main(int argc, char* argv[])
{
    plog::Severity verbosityLevel
        = plog::debug; // Set to debug to see the debug messages, info messages
    std::string csvLogFile = "/tmp/dentk-jpg.csv"; // Set NULL to disable
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    LOGI << "dentk-jpg";
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
    if(a.grayscaleSpecified)
    {
        std::shared_ptr<io::Frame2DReaderItkI<float>> sliceReader
            = std::make_shared<io::DenFrame2DReaderItk<float>>(a.input_file);
        for(const int &k : a.frames)
        {
            std::string tmpImg = io::xprintf("%s/%s%03d.bmp", a.output_directory.c_str(),
                                             a.file_prefix.c_str(), k);
            if(a.intervalSpecified)
            {
                io::writeCastedImage<float>(sliceReader->readChunk2DAsItkImage(k),
                                            tmpImg, a.min, a.max);
            } else
            {
                io::writeCastedImage<float>(sliceReader->readChunk2DAsItkImage(k),
                                            tmpImg);
            }
        }

    } else
    {
        if(a.input_file_red.empty() && a.input_file_green.empty() && a.input_file_blue.empty())
        {
            std::string msg = io::xprintf("There was no input files specified.");
            LOGE << msg;
            throw std::runtime_error(msg);
        }
        std::shared_ptr<io::Frame2DReaderI<float>> redSliceReader, greenSliceReader,
            blueSliceReader;
        redSliceReader = NULL;
        greenSliceReader = NULL;
        blueSliceReader = NULL;
        if(!a.input_file_red.empty())
        {
            redSliceReader = std::make_shared<io::DenFrame2DReader<float>>(a.input_file_red);
        }
        if(!a.input_file_green.empty())
        {
            greenSliceReader = std::make_shared<io::DenFrame2DReader<float>>(a.input_file_green);
        }
        if(!a.input_file_blue.empty())
        {
            blueSliceReader = std::make_shared<io::DenFrame2DReader<float>>(a.input_file_blue);
        }
        for(const int &k : a.frames)
        {
            std::string tmpImg = io::xprintf("%s/%s%03d.bmp", a.output_directory.c_str(),
                                             a.file_prefix.c_str(), k);
            std::shared_ptr<io::Frame2DI<float>> imgR, imgG, imgB;
            if(redSliceReader)
            {
                imgR = redSliceReader->readFrame(k);
            } else
            {
                imgR = NULL;
            }
            if(greenSliceReader)
            {
                imgG = greenSliceReader->readFrame(k);
            } else
            {
                imgG = NULL;
            }
            if(blueSliceReader)
            {
                imgB = blueSliceReader->readFrame(k);
            } else
            {
                imgB = NULL;
            }
            if(a.intervalSpecified)
            {
                io::writeChannelsRGB<float>(imgR, imgG, imgB, tmpImg, a.min, a.max);
            } else
            {
                io::writeChannelsRGB<float>(imgR, imgG, imgB, tmpImg);
            }
        }
    }
}

int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Process different frames as color channels." };
    app.add_option("-R,--red-component", input_file_red,
                   "File in a DEN format to be interpretted as a red component.")
        ->check(CLI::ExistingFile);
    app.add_option("-B,--blue-component", input_file_blue,
                   "File in a DEN format to be interpretted as a blue component.")
        ->check(CLI::ExistingFile);
    app.add_option("-G,--green-component", input_file_green,
                   "File in a DEN format to be interpretted as a green component.")
        ->check(CLI::ExistingFile);
    app.add_option("-f,--frames", frames,
                   "Specify only particular frames to process. You can input range i.e. 0-20 or "
                   "also individual coma separated frames i.e. 1,8,9. Order does matter. Accepts "
                   "end literal that means total number of slices of the input.");
    app.add_option("-d,--output-dir-name", output_directory,
                   "Output directory. Defaults to '/tmp'.")
        ->check(CLI::ExistingDirectory);
    app.add_option("-p,--file-prefix", file_prefix, "Prefix of files to write, defaults to slice.");
    CLI::Option* minopt = app.add_option("--min", min,
                                         "Float value that will be interpreted as a minimum for "
                                         "interpolation [min,max] -> [0,255].")
                              ->check(CLI::Range(0.0, 1000.0));
    CLI::Option* maxopt = app.add_option("--max", max,
                                         "Float value that will be interpreted as a maximum for "
                                         "interpolation [min,max] -> [0,255].")
                              ->needs(minopt)
                              ->check(CLI::Range(0.0, 1000.0));
    minopt->needs(maxopt);
    app.add_option("den_file", input_file,
                   "File in a DEN format to produce a grayscale image. If specified, no other (R, "
                   "G, B) components are processed.")
        ->check(CLI::ExistingFile);
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
    if(app.count("--frames") > 0)
        framesSpecified = true;
    if(app.count("--red-component") > 0 || app.count("--blue-component") > 0
       || app.count("--green-component") > 0)
    {
        colorsSpecified = true;
    }
    if(app.count("--max") > 0)
        intervalSpecified = true;
    if(!input_file.empty())
    {
        grayscaleSpecified = true;
    }
    if(!grayscaleSpecified && !colorsSpecified)
    {
        std::string msg = io::xprintf("There was no input files specified.");
        LOGE << msg;
        std::cout << app.help();
        return -1;
    } else
    {
        std::string f = "";
        if(!input_file.empty())
        {
            f = input_file;
        } else if(!input_file_green.empty())
        {
            f = input_file_green;
        } else if(!input_file_blue.empty())
        {
            f = input_file_blue;
        } else if(!input_file_red.empty())
        {
            f = input_file_red;
        } else
        {//If all the strings are empty
            std::string msg = io::xprintf("There was no input files specified.");
            LOGE << msg;
            std::cout << app.help();
        }
        io::DenFileInfo di(f);
        frames = util::processFramesSpecification(frameSpecs, di.getNumSlices());
    }
    return 0;
}
