// Logging
#include <utils/PlogSetup.h>

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
#include "strtk.hpp"

// Internal libraries
#include "io/AsyncImageWritterItkI.hpp"
#include "io/Chunk2DReaderI.hpp"
#include "io/DICOMAsyncWritterItk.hpp"
#include "io/DenAsyncWritter.hpp"
#include "io/DenChunk2DReader.hpp"
#include "io/DenChunk2DReaderItk.hpp"
#include "io/DenFileInfo.hpp"
#include "io/itkop.h"

using namespace CTL;
namespace fs = std::experimental::filesystem;

// class declarations
struct Args
{
    int parseArguments(int argc, char* argv[]);
    std::string input_file = "";
    std::string output_dir = "";
    std::string frames = "";
    std::string file_prefix = "";
    bool framesSpecified = false;
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
                int index = std::stoi(*it);
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
    std::string csvLogFile = "/tmp/imageRegistrationLog.csv"; // Set NULL to disable
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
    float globalMinValue, globalMaxValue;
    io::DenFileInfo di(a.input_file);
    globalMinValue = di.getMinVal<float>();
    globalMaxValue = di.getMaxVal<float>();
    io::DenSupportedType dataType = di.getDataType();
    //    float min, max;
    //    min = std::numeric_limits<float>::infinity();
    //    max = -std::numeric_limits<float>::infinity();
    //    float tmpmin, tmpmax;
    //    for(int i = 0; i != framesToOutput.size(); i++)
    //    {
    //        tmpmin = (double)sliceReader->readSlice(framesToOutput[i])->minValue();
    //        tmpmax = (double)sliceReader->readSlice(framesToOutput[i])->maxValue();
    //        min = (min < tmpmin ? min : tmpmin);
    //        max = (max > tmpmax ? max : tmpmax);
    //    }
    //    LOGD << io::xprintf("Global min=%f, min=%f, global max=%f, max=%f", globalMinValue, min,
    //    globalMaxValue, max);
    std::vector<int> framesToOutput;
    framesToOutput = processResultingFrames(a.frames, di.getNumSlices());
    switch(dataType)
    {
    case io::DenSupportedType::uint16_t_:
    {
        std::shared_ptr<io::Chunk2DReaderItkI<uint16_t>> sliceReader
            = std::make_shared<io::DenChunk2DReaderItk<uint16_t>>(a.input_file);
        std::shared_ptr<io::AsyncImageWritterItkI<uint16_t>> dicomWritter
            = std::make_shared<io::DICOMAsyncWritterItk<uint16_t>>(
                a.output_dir, a.file_prefix, sliceReader->dimx(), sliceReader->dimy(),
                framesToOutput.size(), globalMinValue, globalMaxValue);
        for(int i = 0; i != framesToOutput.size(); i++)
        {
            // For each frame I write one slice into the output directory.
            LOGD << io::xprintf("Processing frame %d.", framesToOutput[i]);
            dicomWritter->writeSlice(sliceReader->readChunk2DAsItkImage(framesToOutput[i]), i);
        }
        break;
    }
    case io::DenSupportedType::float_:
    {
        std::shared_ptr<io::Chunk2DReaderItkI<float>> sliceReader
            = std::make_shared<io::DenChunk2DReaderItk<float>>(a.input_file);
        std::shared_ptr<io::AsyncImageWritterItkI<float>> dicomWritter
            = std::make_shared<io::DICOMAsyncWritterItk<float>>(
                a.output_dir, a.file_prefix, sliceReader->dimx(), sliceReader->dimy(),
                framesToOutput.size(), globalMinValue, globalMaxValue);
        for(int i = 0; i != framesToOutput.size(); i++)
        {
            // For each frame I write one slice into the output directory.
            LOGD << io::xprintf("Processing frame %d.", framesToOutput[i]);
            dicomWritter->writeSlice(sliceReader->readChunk2DAsItkImage(framesToOutput[i]), i);
        }
        break;
    }
    case io::DenSupportedType::double_:
    {
        std::shared_ptr<io::Chunk2DReaderItkI<double>> sliceReader
            = std::make_shared<io::DenChunk2DReaderItk<double>>(a.input_file);
        std::shared_ptr<io::AsyncImageWritterItkI<double>> dicomWritter
            = std::make_shared<io::DICOMAsyncWritterItk<double>>(
                a.output_dir, a.file_prefix, sliceReader->dimx(), sliceReader->dimy(),
                framesToOutput.size(), globalMinValue, globalMaxValue);
        for(int i = 0; i != framesToOutput.size(); i++)
        {
            // For each frame I write one slice into the output directory.
            LOGD << io::xprintf("Processing frame %d.", framesToOutput[i]);
            dicomWritter->writeSlice(sliceReader->readChunk2DAsItkImage(framesToOutput[i]), i);
        }
        break;
    }
    default:
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(dataType));
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
    }
}

int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Process different frames as color channels." };
    app.add_option("-f,--frames", frames,
                   "Specify only particular frames to process. You can input range i.e. 0-20 or "
                   "also individual comma separated frames i.e. 1,8,9. Order does matter. Accepts "
                   "end literal that means total number of slices of the input.");
    app.add_option("den_file", input_file, "File in a DEN format to process.")
        ->check(CLI::ExistingFile);
    app.add_option("-p,--file-prefix", file_prefix,
                   "Prefix of files to write, defaults to den_file.");
    app.add_option("output_DICOM_dir", output_dir,
                   "Directory where DICOM series data will be written")
        ->required();
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
    if(app.count("--file-prefix") == 0)
        file_prefix = input_file;
    if(!fs::is_directory(output_dir))
    {
        if(fs::exists(output_dir))
        {
            std::string errMsg = io::xprintf("Argument output directory %s exists in the path but "
                                             "does not point to a directory.",
                                             output_dir.c_str());
            LOGE << errMsg;
            return -1;
        } else
        {
            fs::create_directory(output_dir);
        }
    }

    return 0;
}
