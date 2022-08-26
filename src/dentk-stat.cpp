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
#include "AsyncFrame2DWritterI.hpp"
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DI.hpp"
#include "Frame2DReaderI.hpp"
#include "FrameMemoryViewer2D.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/Program.hpp"
#include "PROG/parseArgs.h"

using namespace KCT;
using namespace KCT::util;

class Args : public ArgumentsFramespec, public ArgumentsThreading
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsFramespec(argc, argv, prgName)
        , ArgumentsThreading(argc, argv, prgName){};
    std::vector<std::string> inputFiles;
    std::string outputFile;
    bool force = false;
    bool variance = false;
    bool mean = false;
    bool standardDeviation = false;
};

void Args::defineArguments()
{
    cliApp->add_option("output_den_file", outputFile, "File in a DEN format to output.")
        ->required();
    cliApp
        ->add_option("input_den_file1 ... input_den_filen", inputFiles,
                     "Files in a DEN format to process. These files should have the same x,y and z "
                     "dimension as the first file of input.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_flag("--force", force, "Overwrite outputFile if it exists.");
    CLI::Option_group* op_clg
        = cliApp->add_option_group("Operation", "Mathematical operation to perform element-wise.");
    op_clg->add_flag("--mean", mean, "Compute mean of the files");
    // See http://mathworld.wolfram.com/SampleVariance.html
    op_clg->add_flag("--sample-variance", variance, "Compute sample variance");
    op_clg->add_flag("--sample-standard-deviation", standardDeviation,
                     "Compute sample standard deviation");
    op_clg->require_option(1);
    addFramespecArgs();
    addThreadingArgs();
}

int Args::postParse()
{
    // If force is not set, then check if output file does not exist
    std::string err;
    if(!force)
    {
        if(io::pathExists(outputFile))
        {
            err = "Error: output file already exists, use --force to force overwrite.";
            LOGE << err;
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
            err = io::xprintf("File %s and %s are of different element types.",
                              inputFiles[0].c_str(), f.c_str());
            LOGE << err;
            return 1;
        }
        if(df.dimx() != dimx || df.dimy() != dimy || df.dimz() != dimz)
        {
            err = io::xprintf("Files %s and %s do not have the same dimensions.",
                              inputFiles[0].c_str(), f.c_str());
            LOGE << err;
            return 1;
        }
    }
    fillFramesVector(dimz);
    return 0;
}

template <typename T>
void computeMean(uint32_t k,
                 std::vector<std::shared_ptr<io::Frame2DReaderI<T>>> denSliceReaders,
                 T* mean)
{
    T N = denSliceReaders.size();
    uint32_t dimx = denSliceReaders[0]->dimx();
    uint32_t dimy = denSliceReaders[0]->dimy();
    uint32_t frameSize = dimx * dimy;
    std::fill(mean, mean + frameSize, T(0));
    for(std::shared_ptr<io::Frame2DReaderI<T>> reader : denSliceReaders)
    {
        std::shared_ptr<io::Frame2DI<T>> curframe = reader->readFrame(k);
        for(uint32_t y = 0; y != dimy; y++)
        {
            for(uint32_t x = 0; x != dimx; x++)
            {
                mean[y * dimx + x] += curframe->get(x, y);
            }
        }
    }
    for(uint32_t i = 0; i != frameSize; i++)
    {
        mean[i] /= N;
    }
}

template <typename T>
void computeSampleVariance(uint32_t k,
                           std::vector<std::shared_ptr<io::Frame2DReaderI<T>>> denSliceReaders,
                           T* mean,
                           T* variance)
{
    T N_minus_1 = denSliceReaders.size() - 1;
    uint32_t dimx = denSliceReaders[0]->dimx();
    uint32_t dimy = denSliceReaders[0]->dimy();
    uint32_t frameSize = dimx * dimy;
    std::fill(variance, variance + frameSize, T(0));
    for(std::shared_ptr<io::Frame2DReaderI<T>> reader : denSliceReaders)
    {
        std::shared_ptr<io::Frame2DI<T>> curframe = reader->readFrame(k);
        for(uint32_t y = 0; y != dimy; y++)
        {
            for(uint32_t x = 0; x != dimx; x++)
            {
                T v = curframe->get(x, y);
                v = (v - mean[y * dimx + x]) * (v - mean[y * dimx + x]);
                variance[y * dimx + x] += v;
            }
        }
    }
    for(uint32_t i = 0; i != frameSize; i++)
    {
        variance[i] /= N_minus_1;
    }
}

template <typename T>
void writeMeanFrame(int id,
                    int fromId,
                    std::vector<std::shared_ptr<io::Frame2DReaderI<T>>> denSliceReaders,
                    int toId,
                    std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter)
{
    uint32_t dimx = imagesWritter->dimx();
    uint32_t dimy = imagesWritter->dimy();
    uint32_t frameSize = dimx * dimy;
    T* mean = new T[frameSize];
    computeMean(fromId, denSliceReaders, mean);
    std::unique_ptr<io::Frame2DI<T>> f
        = std::make_unique<io::FrameMemoryViewer2D<T>>(mean, dimx, dimy);
    imagesWritter->writeFrame(*f, toId);
    delete[] mean;
}

template <typename T>
void writeVarianceFrame(int id,
                        int fromId,
                        std::vector<std::shared_ptr<io::Frame2DReaderI<T>>> denSliceReaders,
                        int toId,
                        std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter)
{
    uint32_t dimx = imagesWritter->dimx();
    uint32_t dimy = imagesWritter->dimy();
    uint32_t frameSize = dimx * dimy;
    T* mean = new T[frameSize];
    T* variance = new T[frameSize]();
    computeMean(fromId, denSliceReaders, mean);
    computeSampleVariance(fromId, denSliceReaders, mean, variance);
    std::unique_ptr<io::Frame2DI<T>> f
        = std::make_unique<io::FrameMemoryViewer2D<T>>(variance, dimx, dimy);
    imagesWritter->writeFrame(*f, toId);
    delete[] mean;
    delete[] variance;
}

template <typename T>
void writeSampleStandardDeviationFrame(
    int id,
    int fromId,
    std::vector<std::shared_ptr<io::Frame2DReaderI<T>>> denSliceReaders,
    int toId,
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> imagesWritter)
{
    uint32_t dimx = imagesWritter->dimx();
    uint32_t dimy = imagesWritter->dimy();
    uint32_t frameSize = dimx * dimy;
    T* mean = new T[frameSize];
    T* variance = new T[frameSize]();
    computeMean(fromId, denSliceReaders, mean);
    computeSampleVariance(fromId, denSliceReaders, mean, variance);
    for(uint32_t i = 0; i != frameSize; i++)
    {
        variance[i] = std::pow(variance[i], 0.5);
    }
    std::unique_ptr<io::Frame2DI<T>> f
        = std::make_unique<io::FrameMemoryViewer2D<T>>(variance, dimx, dimy);
    imagesWritter->writeFrame(*f, toId);
    delete[] mean;
    delete[] variance;
}

template <typename T>
void elementWiseStatistics(Args a)
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
        if(a.mean)
        {
            if(threadpool != nullptr)
            {
                threadpool->push(writeMeanFrame<T>, a.frames[i], denSliceReaders, i, imagesWritter);
            } else
            {
                writeMeanFrame<T>(0, a.frames[i], denSliceReaders, i, imagesWritter);
            }
        }
        if(a.variance)
        {
            if(threadpool != nullptr)
            {
                threadpool->push(writeVarianceFrame<T>, a.frames[i], denSliceReaders, i,
                                 imagesWritter);
            } else
            {
                writeVarianceFrame<T>(0, a.frames[i], denSliceReaders, i, imagesWritter);
            }
        }
        if(a.standardDeviation)
        {
            if(threadpool != nullptr)
            {
                threadpool->push(writeSampleStandardDeviationFrame<T>, a.frames[i], denSliceReaders,
                                 i, imagesWritter);
            } else
            {
                writeSampleStandardDeviationFrame<T>(0, a.frames[i], denSliceReaders, i,
                                                     imagesWritter);
            }
        }
    }
    if(threadpool != nullptr)
    {
        threadpool->stop(true);
        delete threadpool;
    }
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    Args ARG(argc, argv, "Do a statistics from the mutiple DEN files.");
    // Argument parsing
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    // Frames to process
    io::DenFileInfo inf(ARG.inputFiles[0]);
    io::DenSupportedType dataType = inf.getDataType();
    switch(dataType)
    {
    case io::DenSupportedType::UINT16: {
        elementWiseStatistics<uint16_t>(ARG);
        break;
    }
    case io::DenSupportedType::FLOAT32: {
        elementWiseStatistics<float>(ARG);
        break;
    }
    case io::DenSupportedType::FLOAT64: {
        elementWiseStatistics<double>(ARG);
        break;
    }
    default:
        std::string errMsg = io::xprintf("Unsupported data type %s.",
                                         io::DenSupportedTypeToString(dataType).c_str());
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
    }
    LOGI << io::xprintf("END %s", argv[0]);
}
