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
#include "PROG/parseArgs.h"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "DENITK/AsyncFrame2DWritterItkI.hpp"
#include "DENITK/DICOMAsyncWritterItk.hpp"
#include "DENITK/DenFrame2DReaderItk.hpp"
#include "DENITK/itkop.h"
#include "Frame2DReaderI.hpp"

using namespace KCT;
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
    float spacing_x = 1.0, spacing_y = 1.0;
    int outputMin, outputMax;
};

std::string program_name = "";

int main(int argc, char* argv[])
{
    plog::Severity verbosityLevel
        = plog::debug; // Set to debug to see the debug messages, info messages
    std::string csvLogFile = "/tmp/imageRegistrationLog.csv"; // Set NULL to disable
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    LOGI << "dentk-todicom";
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
    //        tmpmin = (double)sliceReader->readFrame(framesToOutput[i])->minValue();
    //        tmpmax = (double)sliceReader->readFrame(framesToOutput[i])->maxValue();
    //        min = (min < tmpmin ? min : tmpmin);
    //        max = (max > tmpmax ? max : tmpmax);
    //    }
    //    LOGD << io::xprintf("Global min=%f, min=%f, global max=%f, max=%f", globalMinValue, min,
    //    globalMaxValue, max);
    io::DenFileInfo di(a.input_file);
    io::DenSupportedType dataType = di.getElementType();
    std::vector<int> framesToOutput = util::processFramesSpecification(a.frames, di.getNumSlices());
    switch(dataType)
    {
    case io::DenSupportedType::UINT16:
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

        std::shared_ptr<io::Frame2DReaderItkI<uint16_t>> sliceReader
            = std::make_shared<io::DenFrame2DReaderItk<uint16_t>>(a.input_file, a.spacing_x,
                                                                  a.spacing_y);
        std::shared_ptr<io::AsyncFrame2DWritterItkI<uint16_t>> dicomWritter
            = std::make_shared<io::DICOMAsyncWritterItk<uint16_t>>(
                a.output_dir, a.file_prefix, sliceReader->dimx(), sliceReader->dimy(),
                framesToOutput.size(), windowMin, windowMax, outputMin, outputMax,
                a.useSignedIntegers, a.multiplyByFactor, a.addToValues);
        for(std::size_t i = 0; i != framesToOutput.size(); i++)
        {
            // For each frame I write one slice into the output directory.
            // LOGD << io::xprintf("Processing frame %d.", framesToOutput[i]);
            dicomWritter->writeFrame(sliceReader->readChunk2DAsItkImage(framesToOutput[i]), i);
        }
        break;
    }
    case io::DenSupportedType::FLOAT32:
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

        std::shared_ptr<io::Frame2DReaderItkI<float>> sliceReader
            = std::make_shared<io::DenFrame2DReaderItk<float>>(a.input_file, a.spacing_x,
                                                               a.spacing_y);
        std::shared_ptr<io::AsyncFrame2DWritterItkI<float>> dicomWritter
            = std::make_shared<io::DICOMAsyncWritterItk<float>>(
                a.output_dir, a.file_prefix, sliceReader->dimx(), sliceReader->dimy(),
                framesToOutput.size(), windowMin, windowMax, outputMin, outputMax,
                a.useSignedIntegers, a.multiplyByFactor, a.addToValues);
        for(std::size_t i = 0; i != framesToOutput.size(); i++)
        {
            // For each frame I write one slice into the output directory.
            // LOGD << io::xprintf("Processing frame %d.", framesToOutput[i]);
            dicomWritter->writeFrame(sliceReader->readChunk2DAsItkImage(framesToOutput[i]), i);
        }
        break;
    }
    case io::DenSupportedType::FLOAT64:
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
        std::shared_ptr<io::Frame2DReaderItkI<double>> sliceReader
            = std::make_shared<io::DenFrame2DReaderItk<double>>(a.input_file, a.spacing_x,
                                                                a.spacing_y);
        std::shared_ptr<io::AsyncFrame2DWritterItkI<double>> dicomWritter
            = std::make_shared<io::DICOMAsyncWritterItk<double>>(
                a.output_dir, a.file_prefix, sliceReader->dimx(), sliceReader->dimy(),
                framesToOutput.size(), windowMin, windowMax, outputMin, outputMax,
                a.useSignedIntegers, a.multiplyByFactor, a.addToValues);
        for(std::size_t i = 0; i != framesToOutput.size(); i++)
        {
            // For each frame I write one slice into the output directory.
            // LOGD << io::xprintf("Processing frame %d.", framesToOutput[i]);
            dicomWritter->writeFrame(sliceReader->readChunk2DAsItkImage(framesToOutput[i]), i);
        }
        break;
    }
    default:
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(dataType).c_str());
        KCTERR(errMsg);
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
    app.add_option("--spacing_x", spacing_x, "Spacing in the x direction, defaults to 1.0.")
        ->check(CLI::Range(0.0, 100.0));
    app.add_option("--spacing_y", spacing_y, "Spacing in the y direction, defaults to 1.0.")
        ->check(CLI::Range(0.0, 100.0));
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
        file_prefix = fs::path(input_file).stem();
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
