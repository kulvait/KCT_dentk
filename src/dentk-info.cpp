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
#include "CLI/CLI.hpp"
#include "strtk.hpp"

// Internal libraries
#include "io/AsyncImageWritterI.hpp"
#include "io/Chunk2DReaderI.hpp"
#include "io/DenAsyncWritter.hpp"
#include "io/DenChunk2DReader.hpp"
#include "io/DenSupportedType.hpp"

using namespace CTL;

// class declarations
struct Args
{
    int parseArguments(int argc, char* argv[]);
    std::string input_file;
    std::string frames = "";
    bool framesSpecified = false;
};

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
    int elementSize = di.elementByteSize();
    io::DenSupportedType t = di.getDataType();
    std::string elm = io::DenSupportedTypeToString(t);
    std::cout << io::xprintf("The file of type %s has dimensions (x,y,z)=(cols,rows,slices)=(%d, "
                             "%d, %d), each cell has x*y=%d pixels.\n",
                             elm.c_str(), dimx, dimy, dimz, dimx * dimy);
    switch(t)
    {
    case io::DenSupportedType::uint16_t_:
    {
        uint16_t min = di.getMinVal<uint16_t>();
        uint16_t max = di.getMaxVal<uint16_t>();
        std::cout << io::xprintf("Global maximum and minimum values are (%d, %d).\n", (int)min,
                                 (int)max);
        break;
    }
    case io::DenSupportedType::float_:
    {
        float min = di.getMinVal<float>();
        float max = di.getMaxVal<float>();
        std::cout << io::xprintf("Global maximum and minimum values are (%f, %f).\n", (double)min,
                                 (double)max);
        break;
    }
    case io::DenSupportedType::double_:
    {
        double min = di.getMinVal<double>();
        double max = di.getMaxVal<double>();
        std::cout << io::xprintf("Global maximum and minimum values are (%f, %f).\n", min, max);
        break;
    }
    default:
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(t));
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
    }
    if(a.framesSpecified)
    {
        std::vector<int> framesToOutput;
        framesToOutput = processResultingFrames(a.frames, dimz);
        switch(t)
        {
        case io::DenSupportedType::uint16_t_:
        {
            std::shared_ptr<io::Chunk2DReaderI<uint16_t>> denSliceReader
                = std::make_shared<io::DenChunk2DReader<uint16_t>>(a.input_file);
            for(int i = 0; i != framesToOutput.size(); i++)
            {
                std::cout << io::xprintf("Statistic of %d-th frame:\n", framesToOutput[i]);
                auto slicePtr = denSliceReader->readSlice(framesToOutput[i]);
                float normSquared = (float)slicePtr->normSquare();
                float norm = std::sqrt(normSquared);
                float min = slicePtr->minValue();
                float max = slicePtr->maxValue();
                double avg = slicePtr->meanValue();
                std::cout << io::xprintf("Minimum, maximum, average values: %.3f, %0.3f, %0.3f.\n",
                                         min, max, avg);
                std::cout << io::xprintf("Euclidean 2-norm of the whole chunk: %.3f.\n",
                                         std::sqrt(normSquared));
                std::cout << io::xprintf("Average pixel square intensity: %.3f.\n\n",
                                         normSquared / (dimx * dimy));
            }
            break;
        }
        case io::DenSupportedType::float_:
        {

            std::shared_ptr<io::Chunk2DReaderI<float>> denSliceReader
                = std::make_shared<io::DenChunk2DReader<float>>(a.input_file);
            for(int i = 0; i != framesToOutput.size(); i++)
            {
                std::cout << io::xprintf("Statistic of %d-th frame:\n", framesToOutput[i]);
                auto slicePtr = denSliceReader->readSlice(framesToOutput[i]);
                float normSquared = (float)slicePtr->normSquare();
                float norm = std::sqrt(normSquared);
                float min = slicePtr->minValue();
                float max = slicePtr->maxValue();
                double avg = slicePtr->meanValue();
                std::cout << io::xprintf("Minimum, maximum, average values: %.3f, %0.3f, %0.3f.\n",
                                         min, max, avg);
                std::cout << io::xprintf("Euclidean 2-norm of the whole chunk: %.3f.\n",
                                         std::sqrt(normSquared));
                std::cout << io::xprintf("Average pixel square intensity: %.3f.\n\n",
                                         normSquared / (dimx * dimy));
            }
            break;
        }
        case io::DenSupportedType::double_:
        {

            std::shared_ptr<io::Chunk2DReaderI<double>> denSliceReader
                = std::make_shared<io::DenChunk2DReader<double>>(a.input_file);
            for(int i = 0; i != framesToOutput.size(); i++)
            {
                std::cout << io::xprintf("Statistic of %d-th frame:\n", framesToOutput[i]);
                auto slicePtr = denSliceReader->readSlice(framesToOutput[i]);
                float normSquared = (float)slicePtr->normSquare();
                float norm = std::sqrt(normSquared);
                float min = slicePtr->minValue();
                float max = slicePtr->maxValue();
                double avg = slicePtr->meanValue();
                std::cout << io::xprintf("Minimum, maximum, average values: %.3f, %0.3f, %0.3f.\n",
                                         min, max, avg);
                std::cout << io::xprintf("Euclidean 2-norm of the whole chunk: %.3f.\n",
                                         std::sqrt(normSquared));
                std::cout << io::xprintf("Average pixel square intensity: %.3f.\n\n",
                                         normSquared / (dimx * dimy));
            }
            break;
        }
        default:
            std::string errMsg = io::xprintf("Frame statistic for %s is unsupported.",
                                             io::DenSupportedTypeToString(t));
            LOGE << errMsg;
            throw std::runtime_error(errMsg);
        }
    }
}

int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Information about DEN file and its individual slices." };
    app.add_option("-f,--frames", frames,
                   "Specify only particular frames to process. You can input range i.e. 0-20 or "
                   "also individual comma separated frames i.e. 1,8,9. Order does matter. Accepts "
                   "end literal that means total number of slices of the input.");
    app.add_option("input_den_file", input_file, "File in a DEN format to process.")
        ->required()
        ->check(CLI::ExistingFile);

    try
    {
        app.parse(argc, argv);
        if(app.count("--frames") > 0)
        {
            framesSpecified = true;
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

    return 0;
}
