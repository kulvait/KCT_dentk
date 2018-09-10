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
struct Args
{
    int parseArguments(int argc, char* argv[]);
    std::string input_file = "";
    std::string output_dir = "";
    std::string frames = "";
    std::string file_prefix = "";
    bool framesSpecified = false, windowingSpecified = false;
    bool stretchToRange = false;
    bool useSignedIntegers = false;
    float multiplyByFactor = 1.0, addToValues = 0.0;
    float windowMin, windowMax;
    int outputMin, outputMax;
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
    io::DenFileInfo di(a.input_file);
    io::DenSupportedType dataType = di.getDataType();
    std::vector<int> framesToOutput;
    framesToOutput = processResultingFrames(a.frames, di.getNumSlices());
    switch(dataType)
    {
    case io::DenSupportedType::uint16_t_:
    {
        uint16_t globalMinValue = di.getMinVal<uint16_t>();
        uint16_t globalMaxValue = di.getMaxVal<uint16_t>();
        LOGD << io::xprintf("For the file %s of the type %s min value is %d and max value is %d.",
                            a.input_file.c_str(), io::DenSupportedTypeToString(dataType).c_str(),
                            (int)globalMinValue, (int)globalMaxValue);
        uint16_t windowMin, windowMax;
        int outputMin, outputMax;
        if(a.stretchToRange)
        {
            windowMin = di.getMinVal<uint16_t>();
            windowMax = di.getMaxVal<uint16_t>();
            if(a.useSignedIntegers)
            {
                outputMin = -32768;
                outputMax = 32767;
            } else
            {
                outputMin = 0;
                outputMax = 65535;
            }
        } else
        {
            if(a.windowingSpecified)
            {
                windowMin = (uint16_t)a.windowMin;
                windowMax = (uint16_t)a.windowMax;
                outputMin = a.outputMin;
                outputMax = a.outputMax;
            } else
            {
                if(a.useSignedIntegers)
                {
                    windowMin = 0;
                    windowMax = 32767;
                    outputMin = 0;
                    outputMax = 32767;
                } else
                {
                    windowMin = 0;
                    windowMax = 65535;
                    outputMin = 0;
                    outputMax = 65535;
                }
            }
        }

        std::shared_ptr<io::Chunk2DReaderItkI<uint16_t>> sliceReader
            = std::make_shared<io::DenChunk2DReaderItk<uint16_t>>(a.input_file);
        std::shared_ptr<io::AsyncImageWritterItkI<uint16_t>> dicomWritter
            = std::make_shared<io::DICOMAsyncWritterItk<uint16_t>>(
                a.output_dir, a.file_prefix, sliceReader->dimx(), sliceReader->dimy(),
                framesToOutput.size(), windowMin, windowMax, outputMin, outputMax,
                a.useSignedIntegers, a.multiplyByFactor, a.addToValues);
        for(int i = 0; i != framesToOutput.size(); i++)
        {
            // For each frame I write one slice into the output directory.
            // LOGD << io::xprintf("Processing frame %d.", framesToOutput[i]);
            dicomWritter->writeSlice(sliceReader->readChunk2DAsItkImage(framesToOutput[i]), i);
        }
        break;
    }
    case io::DenSupportedType::float_:
    {
        float globalMinValue = di.getMinVal<float>();
        float globalMaxValue = di.getMaxVal<float>();
        LOGD << io::xprintf("For the file %s of the type %s min value is %f and max value is %f.",
                            a.input_file.c_str(), io::DenSupportedTypeToString(dataType).c_str(),
                            globalMinValue, globalMaxValue);
        float windowMin, windowMax;
        int outputMin, outputMax;
        if(a.stretchToRange)
        {
            windowMin = di.getMinVal<float>();
            windowMax = di.getMaxVal<float>();
            if(a.useSignedIntegers)
            {
                outputMin = -32768;
                outputMax = 32767;
            } else
            {
                outputMin = 0;
                outputMax = 65535;
            }
        } else
        {
            if(a.windowingSpecified)
            {
                windowMin = (float)a.windowMin;
                windowMax = (float)a.windowMax;
                outputMin = a.outputMin;
                outputMax = a.outputMax;
            } else
            {
                if(a.useSignedIntegers)
                {
                    windowMin = -32768.0;
                    windowMax = 32767.0;
                    outputMin = -32768;
                    outputMax = 32767;
                } else
                {
                    windowMin = 0.0;
                    windowMax = 65535.0;
                    outputMin = 0;
                    outputMax = 65535;
                }
            }
        }

        std::shared_ptr<io::Chunk2DReaderItkI<float>> sliceReader
            = std::make_shared<io::DenChunk2DReaderItk<float>>(a.input_file);
        std::shared_ptr<io::AsyncImageWritterItkI<float>> dicomWritter
            = std::make_shared<io::DICOMAsyncWritterItk<float>>(
                a.output_dir, a.file_prefix, sliceReader->dimx(), sliceReader->dimy(),
                framesToOutput.size(), windowMin, windowMax, outputMin, outputMax,
                a.useSignedIntegers, a.multiplyByFactor, a.addToValues);
        for(int i = 0; i != framesToOutput.size(); i++)
        {
            // For each frame I write one slice into the output directory.
            // LOGD << io::xprintf("Processing frame %d.", framesToOutput[i]);
            dicomWritter->writeSlice(sliceReader->readChunk2DAsItkImage(framesToOutput[i]), i);
        }
        break;
    }
    case io::DenSupportedType::double_:
    {
        double globalMinValue = di.getMinVal<double>();
        double globalMaxValue = di.getMaxVal<double>();
        LOGD << io::xprintf(
            "Processing the file %s of the type %s min value is %f and max value is %f.",
            a.input_file.c_str(), io::DenSupportedTypeToString(dataType).c_str(), globalMinValue,
            globalMaxValue);
        double windowMin, windowMax;
        int outputMin, outputMax;
        if(a.stretchToRange)
        {
            windowMin = di.getMinVal<double>();
            windowMax = di.getMaxVal<double>();
            if(a.useSignedIntegers)
            {
                outputMin = -32768;
                outputMax = 32767;
            } else
            {
                outputMin = 0;
                outputMax = 65535;
            }
        } else
        {
            if(a.windowingSpecified)
            {
                windowMin = (double)a.windowMin;
                windowMax = (double)a.windowMax;
                outputMin = a.outputMin;
                outputMax = a.outputMax;
            } else
            {
                if(a.useSignedIntegers)
                {
                    windowMin = -32768.0;
                    windowMax = 32767.0;
                    outputMin = -32768;
                    outputMax = 32767;
                } else
                {
                    windowMin = 0.0;
                    windowMax = 65535.0;
                    outputMin = 0;
                    outputMax = 65535;
                }
            }
        }
        std::shared_ptr<io::Chunk2DReaderItkI<double>> sliceReader
            = std::make_shared<io::DenChunk2DReaderItk<double>>(a.input_file);
        std::shared_ptr<io::AsyncImageWritterItkI<double>> dicomWritter
            = std::make_shared<io::DICOMAsyncWritterItk<double>>(
                a.output_dir, a.file_prefix, sliceReader->dimx(), sliceReader->dimy(),
                framesToOutput.size(), windowMin, windowMax, outputMin, outputMax,
                a.useSignedIntegers, a.multiplyByFactor, a.addToValues);
        for(int i = 0; i != framesToOutput.size(); i++)
        {
            // For each frame I write one slice into the output directory.
            // LOGD << io::xprintf("Processing frame %d.", framesToOutput[i]);
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
    CLI::App app{ "Convert den file into dicom. "
                  "By default after multiplication and addition values are truncated to the data "
                  "type size." };
    app.add_option("-f,--frames", frames,
                   "Specify only particular frames to process. You can input range i.e. 0-20 or "
                   "also individual comma separated frames i.e. 1,8,9. Order does matter. Accepts "
                   "end literal that means total number of slices of the input.");
    CLI::Option* xxx = app.add_flag(
        "-s,--stretch_to_range", stretchToRange,
        "Window the data between minimum and maximum and use the full range of output data type.");
    app.add_flag("-i,--use_signed_integers", useSignedIntegers,
                 "Output type should be int16 instead of default uint16.");
    app.add_option("-m,--multiply_by_factor", multiplyByFactor,
                   "Float value to multiply with the input values prior to further processing. "
                   "Multiplication prior to addition.");
    app.add_option("-a,--add_to_values", addToValues,
                   "Float value to add to the input values prior to further processing. "
                   "Multiplication prior to addition.");
    CLI::Option* wmn = app.add_option("--window-min", windowMin, "Min value of the window to use.");
    CLI::Option* wmx = app.add_option("--window-max", windowMax, "Max value of the window to use.");
    CLI::Option* omn
        = app.add_option("--output-min", outputMin, "Min value of the output to file.");
    CLI::Option* omx
        = app.add_option("--output-max", outputMax, "Max value of the output to file.");
    wmn->needs(wmx, omn, omx);
    wmx->needs(wmn, omn, omx);
    omn->needs(wmn, wmx, omx);
    omx->needs(wmn, wmx, omn);
    xxx->excludes(wmn, wmx, omn, omx);
    wmn->excludes(xxx);
    wmx->excludes(xxx);
    omn->excludes(xxx);
    omx->excludes(xxx);

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
    if(app.count("--window-min") > 0)
        windowingSpecified = true;
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
