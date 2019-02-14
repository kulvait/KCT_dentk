// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <algorithm>
#include <cstdlib>
#include <ctype.h>
#include <iostream>
#include <regex>
#include <string>

// External libraries
#include "CLI/CLI.hpp" //Command line parser
#include "ctpl_stl.h" //Threadpool

// Internal libraries
#include "ARGPARSE/parseArgs.h"
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"

using namespace CTL;

template <typename T>
void writeFrame(int id,
                int fromId,
                std::shared_ptr<io::Frame2DReaderI<T>> denSliceReader,
                int toId,
                std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter)
{
    // LOGD << io::xprintf(
    //    "Writing %d th slice of file %s to %d th slice of file %s.", fromId,
    //    (std::dynamic_pointer_cast<io::DenFrame2DReader<float>>(denSliceReader))
    //        ->getFileName()
    //        .c_str(),
    //    toId,
    //    (std::dynamic_pointer_cast<io::DenAsyncFrame2DWritter<float>>(imagesWritter))
    //        ->getFileName()
    //        .c_str());
    imagesWritter->writeFrame(*(denSliceReader->readFrame(fromId)), toId);
}

struct Args
{
    bool interlacing = false;
    std::string frameSpecs = "";
    std::vector<int> frames;
    int eachkth = 1;
    int threads = 1;
    std::vector<std::string> inputFiles;
    std::string outputFile;
    bool force = false;
    int parseArguments(int argc, char* argv[]);
};

template <typename T>
void mergeFiles(Args a)
{
    std::vector<std::shared_ptr<io::Frame2DReaderI<T>>> denSliceReaders;
    LOGD << io::xprintf("Will merge file %s from specified files.", a.outputFile.c_str());
    for(const std::string& f : a.inputFiles)
    {
        denSliceReaders.push_back(std::make_shared<io::DenFrame2DReader<T>>(f));
    }
    uint16_t dimx = denSliceReaders[0]->dimx();
    uint16_t dimy = denSliceReaders[0]->dimy();
    ctpl::thread_pool* threadpool = new ctpl::thread_pool(a.threads);
    if(a.interlacing || a.eachkth || !a.frameSpecs.empty())
    {

        uint16_t dimz = denSliceReaders[0]->dimz();
        LOGD << io::xprintf("From each file will output %d frames.", a.frames.size());
        std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<T>>(
                a.outputFile, dimx, dimy, a.inputFiles.size() * a.frames.size());
        for(std::size_t i = 0; i != a.frames.size(); i++)
        {
            for(std::size_t j = 0; j != a.inputFiles.size(); j++)
            {
                if(a.interlacing)
                    threadpool->push(writeFrame<T>, a.frames[i], denSliceReaders[j],
                                     i * a.inputFiles.size() + j, imagesWritter);
                else
                    threadpool->push(writeFrame<T>, a.frames[i], denSliceReaders[j], j * dimz + i,
                                     imagesWritter);
            }
        }
    } else // just to merge files with potentially different z dimension
    {
        uint32_t totaldimz = 0;
        for(auto const& r : denSliceReaders)
        {
            totaldimz += r->dimz();
        }

        std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter
            = std::make_shared<io::DenAsyncFrame2DWritter<T>>(a.outputFile, dimx, dimy, totaldimz);
        uint32_t posoffset = 0;
        for(std::size_t i = 0; i != denSliceReaders.size(); i++)
        {
            auto r = denSliceReaders[i];
            uint16_t dimz = r->dimz();
            for(uint16_t j = 0; j != dimz; j++)
            {
                threadpool->push(writeFrame<T>, j, r, posoffset + j, imagesWritter);
            }
            posoffset += dimz;
        }
    }
    threadpool->stop(true);
    delete threadpool;
}

/**Argument parsing
 *
 */
int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Merge multiple DEN files together." };
    app.add_flag("-i,--interlacing", interlacing,
                 "First n frames in the output will be from the first n DEN files.");
    app.add_flag("--force", force, "Overwrite outputFile if it exists.");
    app.add_option("-f,--frames", frameSpecs,
                   "Specify only particular frames to process. You can input range i.e. 0-20 or "
                   "also individual comma separated frames i.e. 1,8,9. Order does matter. Accepts "
                   "end literal that means total number of slices of the input.");
    app.add_option("-k,--each-kth", eachkth,
                   "Process only each k-th frame specified by k to output. The frames to output "
                   "are then 1st specified, 1+kN, N=1...\\infty if such frame exists. Parameter k "
                   "must be positive integer.")
        ->check(CLI::Range(1, 65535));
    app.add_option("-j,--threads", threads, "Number of extra threads that application can use.")
        ->check(CLI::Range(0, 65535));
    app.add_option("output_den_file", outputFile, "File in a DEN format to output.")->required();
    app.add_option("input_den_file1 ... input_den_filen output_den_file", inputFiles,
                   "Files in a DEN format to process. These files should have the same x,y and z "
                   "dimension as the first file of input.")
        ->required()
        ->check(CLI::ExistingFile);
    try
    {
        app.parse(argc, argv);
	if(app.count("--help")>0)
	{
		return 1;
	}
        LOGD << io::xprintf(
            "Optional parameters: interlacing=%d, frames=%s, eachkth=%d, threads=%d "
            "and %d input files.",
            interlacing, frameSpecs.c_str(), eachkth, threads, inputFiles.size());
        // If force is not set, then check if output file does not exist
        if(!force)
        {
            if(io::fileExists(outputFile))
            {
                std::string msg
                    = "Error: output file already exists, use --force to force overwrite.";
                LOGE << msg;
                return 1;
            }
        }
        // How many projection matrices is there in total
        io::DenFileInfo di(inputFiles[0]);
        io::DenSupportedType dataType = di.getDataType();
        uint16_t dimx = di.dimx();
        uint16_t dimy = di.dimy();
        uint16_t dimz = di.dimz();
        for(std::string const& f : inputFiles)
        {
            io::DenFileInfo df(f);
            if(df.getDataType() != dataType)
            {
                io::throwerr("File %s and %s are of different element types.",
                             inputFiles[0].c_str(), f.c_str());
            }
            if(df.dimx() != dimx || df.dimy() != dimy)
            {
                io::throwerr("Files %s and %s do not have the same x and y dimensions.",
                             inputFiles[0].c_str(), f.c_str());
            }
            if(interlacing || eachkth || !frameSpecs.empty())
            {
                if(df.dimx() != dimx || df.dimy() != dimy || df.dimz() != dimz)
                {
                    io::throwerr("Files %s and %s do not have the same z dimensions. Since "
                                 "interlacing, eachkth or frame specification was given, this is "
                                 "important.",
                                 inputFiles[0].c_str(), f.c_str());
                }
            }
        }
        if(interlacing || eachkth || !frameSpecs.empty())
        {
            std::vector<int> f = util::processFramesSpecification(frameSpecs, di.getNumSlices());
            for(std::size_t i = 0; i != f.size(); i++)
            {
                if(i % eachkth == 0)
                {
                    frames.push_back(f[i]);
                }
            }
        }
    } catch(const CLI::ParseError& e)
    {
        int exitcode = app.exit(e);
        if(exitcode == 0) // Help message was printed
        {
            return 1;
        } else
        {
            LOGE << io::xprintf("There was perse error catched.\n %s", app.help().c_str());
            return -1;
        }
    }
    return 0;
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
    // Frames to process
    io::DenFileInfo inf(a.inputFiles[0]);
    io::DenSupportedType dataType = inf.getDataType();
    switch(dataType)
    {
    case io::DenSupportedType::uint16_t_:
    {
        mergeFiles<uint16_t>(a);
        break;
    }
    case io::DenSupportedType::float_:
    {
        mergeFiles<float>(a);
        break;
    }
    case io::DenSupportedType::double_:
    {
        mergeFiles<double>(a);
        break;
    }
    default:
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(dataType));
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
    }
    LOGI << io::xprintf("END %s", argv[0]);
}
