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
#include "PROG/parseArgs.h"
#include "AsyncFrame2DWritterI.hpp"
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "Frame2DI.hpp"

using namespace CTL;

template <typename T>
void writeFrame(int id,
                int fromId,
                std::vector<std::shared_ptr<io::Frame2DReaderI<T>>> denSliceReaders,
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
    uint32_t dimx = imagesWritter->dimx();
    uint32_t dimy = imagesWritter->dimy();
    std::unique_ptr<io::BufferedFrame2D<T>> f = std::make_unique<io::BufferedFrame2D<T>>(T(0), dimx, dimy);
    for(std::shared_ptr<io::Frame2DReaderI<T>> reader : denSliceReaders)
    {
	std::shared_ptr<io::Frame2DI<T>> curframe = reader->readFrame(fromId);
        for(uint32_t i = 0; i != dimx; i++)
        {
            for(uint32_t j = 0; j != dimy; j++)
            {
                T cur = f->get(i, j);
                cur += curframe->get(i, j);
                f->set(cur, i, j);
            }
        }
    }
    for(uint32_t i = 0; i != dimx; i++)
    {
        for(uint32_t j = 0; j != dimy; j++)
        {
            T cur = f->get(i, j);
            cur /= denSliceReaders.size();
            f->set(cur, i, j);
        }
    }
    imagesWritter->writeFrame(*f, toId);
}

struct Args
{
    std::string frameSpecs = "";
    std::vector<int> frames;
    uint32_t threads = 0;
    std::vector<std::string> inputFiles;
    std::string outputFile;
    bool force = false;
    int parseArguments(int argc, char* argv[]);
};

template <typename T>
void averageFiles(Args a)
{
    std::vector<std::shared_ptr<io::Frame2DReaderI<T>>> denSliceReaders;
    LOGD << io::xprintf("Will average file %s from specified files.", a.outputFile.c_str());
    for(const std::string& f : a.inputFiles)
    {
        denSliceReaders.push_back(std::make_shared<io::DenFrame2DReader<T>>(f));
    }
    uint16_t dimx = denSliceReaders[0]->dimx();
    uint16_t dimy = denSliceReaders[0]->dimy();
    ctpl::thread_pool* threadpool = nullptr;
    if(a.threads != 0)
    {
        threadpool = new ctpl::thread_pool(a.threads);
    }
    LOGD << io::xprintf("From each file will output %d frames.", a.frames.size());
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter
        = std::make_shared<io::DenAsyncFrame2DWritter<T>>(a.outputFile, dimx, dimy,
                                                          a.frames.size());
    for(std::size_t i = 0; i != a.frames.size(); i++)
    {
        if(threadpool != nullptr)
        {
            threadpool->push(writeFrame<T>, a.frames[i], denSliceReaders, i, imagesWritter);
        } else
        {
            writeFrame<T>(0, a.frames[i], denSliceReaders, i, imagesWritter);
        }
    }
    if(threadpool != nullptr)
    {
        threadpool->stop(true);
        delete threadpool;
    }
}

/**Argument parsing
 *
 */
int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Average input DEN files." };
    app.add_flag("--force", force, "Overwrite outputFile if it exists.");
    app.add_option("-f,--frames", frameSpecs,
                   "Specify only particular frames to process. You can input range i.e. 0-20 or "
                   "also individual comma separated frames i.e. 1,8,9. Order does matter. Accepts "
                   "end literal that means total number of slices of the input.");
    app.add_option("-j,--threads", threads,
                   "Number of extra threads that application can use, 0 no threading.")
        ->check(CLI::Range(0, 65535));
    app.add_option("output_den_file", outputFile, "File in a DEN format to output.")->required();
    app.add_option("input_den_file1 ... input_den_filen", inputFiles,
                   "Files in a DEN format to process. These files should have the same x,y and z "
                   "dimension as the first file of input.")
        ->required()
        ->check(CLI::ExistingFile);
    try
    {
        app.parse(argc, argv);
        if(app.count("--help") > 0)
        {
            return 1;
        }
        // If force is not set, then check if output file does not exist
        if(!force)
        {
            if(io::pathExists(outputFile))
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
            if(df.dimx() != dimx || df.dimy() != dimy || df.dimz() != dimz)
            {
                io::throwerr("Files %s and %s do not have the same dimensions.",
                             inputFiles[0].c_str(), f.c_str());
            }
        }
        frames = util::processFramesSpecification(frameSpecs, di.getNumSlices());
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
        averageFiles<uint16_t>(a);
        break;
    }
    case io::DenSupportedType::float_:
    {
        averageFiles<float>(a);
        break;
    }
    case io::DenSupportedType::double_:
    {
        averageFiles<double>(a);
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
