// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <algorithm> // For std::copy
#include <thread>
#include <vector>

// Internal libraries
#include "DEN/DenAsyncFrame2DBufferedWritter.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "PROG/Arguments.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/Program.hpp"

using namespace KCT;
using namespace KCT::util;

// class declarations
class Args : public ArgumentsForce, public ArgumentsThreading
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName)
        , ArgumentsThreading(argc, argv, prgName){};
    int parseArguments(int argc, char* argv[]);
    std::string input_file;
    std::string output_file;
};

void Args::defineArguments()
{
    cliApp->add_option("input_den_file", input_file, "File that will be transposed.")
        ->check(CLI::ExistingFile)
        ->required();
    cliApp->add_option("output_den_file", output_file, "Transposed file in a DEN format to output.")
        ->required();
    addForceArgs();
    addThreadingArgs();
}

int Args::postParse()
{
    bool removeIfExists = true;
    int existFlag = handleFileExistence(output_file, force, removeIfExists);
    if(existFlag != 0)
    {
        return 1;
    }
    io::DenFileInfo di(input_file);
    if(di.getDimCount() != 3)
    {
        std::string ERR
            = io::xprintf("The file %s has %d dimensions!", input_file.c_str(), di.getDimCount());
        LOGE << ERR;
        return 1;
    }
    return 0;
}

// Function to swap axes in memory in parallel using std::copy
template <typename T>
void swapAxes(Args& ARG,
              std::shared_ptr<io::DenFrame2DReader<T>> inputReader,
              T* outputBuffer,
              uint32_t dimx,
              uint32_t dimy,
              uint32_t dimz)
{
    uint64_t frameSizeAfter = static_cast<uint64_t>(dimx) * static_cast<uint64_t>(dimz);
    uint64_t num_threads = std::max(ARG.threads, 1u);
    uint64_t frames_per_thread = (dimz + num_threads - 1) / num_threads; // ceil
    std::vector<std::thread> threads;
    for(uint64_t t = 0; t < num_threads; t++)
    {
        uint64_t start_frame = t * frames_per_thread;
        uint64_t end_frame = std::min((t + 1) * frames_per_thread, static_cast<uint64_t>(dimz));
        threads.emplace_back(
            [&inputReader, &outputBuffer, &dimx, &dimy, &frameSizeAfter, start_frame, end_frame]() {
                std::shared_ptr<io::BufferedFrame2D<T>> f;
                T* f_array;
                for(uint32_t k = start_frame; k < end_frame; k++)
                {
                    f = inputReader->readBufferedFrame(k);
                    f_array = f->getDataPointer();
                    for(uint32_t j = 0; j < dimy; j++)
                    {
                        std::copy(f_array + j * dimx, f_array + (j + 1) * dimx,
                                  outputBuffer + j * frameSizeAfter + k * dimx);
                    }
                }
            });
    }
    for(auto& t : threads)
    {
        t.join();
    }
}

// Function to write frames in parallel
template <typename T>
void writeFramesParallel(Args& ARG, T* outputBuffer, uint32_t dimx, uint32_t dimy, uint32_t dimz)
{
    std::string outputFile = ARG.output_file;
    uint64_t frameSize = static_cast<uint64_t>(dimx) * static_cast<uint64_t>(dimy);
    uint64_t num_threads = std::max(ARG.threads, 1u);
    uint64_t frames_per_thread = (dimz + num_threads - 1) / num_threads; // ceil
    std::vector<std::thread> threads;
    for(uint64_t t = 0; t < num_threads; t++)
    {
        uint64_t start_frame = t * frames_per_thread;
        uint64_t end_frame = std::min((t + 1) * frames_per_thread, static_cast<uint64_t>(dimz));
        threads.emplace_back(
            [&outputFile, &outputBuffer, frameSize, dimx, dimy, dimz, start_frame, end_frame]() {
                std::shared_ptr<io::DenAsyncFrame2DBufferedWritter<T>> outputWriter
                    = std::make_shared<io::DenAsyncFrame2DBufferedWritter<T>>(outputFile, dimx,
                                                                              dimy, dimz);
                T* f_array;
                for(uint32_t k = start_frame; k < end_frame; k++)
                {
                    f_array = outputBuffer + k * frameSize;
                    outputWriter->writeBuffer(f_array, k);
                }
            });
    }
    for(auto& t : threads)
    {
        t.join();
    }
}

template <typename T>
int process(Args& ARG, io::DenFileInfo& input_inf)
{
    uint32_t dimx = input_inf.dim(0);
    uint32_t dimy = input_inf.dim(1);
    uint32_t dimz = input_inf.dim(2);
    uint64_t frameSize = static_cast<uint64_t>(dimx) * static_cast<uint64_t>(dimy);
    uint64_t totalSize = frameSize * static_cast<uint64_t>(dimz);
    std::shared_ptr<io::DenFrame2DReader<T>> inputReader
        = std::make_shared<io::DenFrame2DReader<T>>(ARG.input_file, ARG.threads);
    LOGI << "Creating output buffer of size " << totalSize;
    T* outputBuffer = new T[totalSize];
    LOGI << "Swapping axes...";
    swapAxes<T>(ARG, inputReader, outputBuffer, dimx, dimy, dimz);
    LOGI << "Writing frames...";

    writeFramesParallel<T>(ARG, outputBuffer, dimx, dimz, dimy);
    LOGI << "Deleting output buffer...";
    delete[] outputBuffer;
    return 0;
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    const std::string prgInfo = "Swap axes of the DEN file.";
    Args ARG(argc, argv, prgInfo);
    int parseResult = ARG.parse(false);
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    io::DenFileInfo di(ARG.input_file);
    io::DenSupportedType dataType = di.getElementType();
    io::DenFileInfo::createEmpty3DDenFile(ARG.output_file, dataType, di.dim(0), di.dim(2),
                                          di.dim(1));
    LOGI << io::xprintf("Output file %s created to store swap result.", ARG.output_file.c_str());
    switch(dataType)
    {
    case io::DenSupportedType::UINT16:
        process<uint16_t>(ARG, di);
        break;
    case io::DenSupportedType::FLOAT32:
        process<float>(ARG, di);
        break;
    case io::DenSupportedType::FLOAT64:
        process<double>(ARG, di);
        break;
    default:
        std::string errMsg = io::xprintf("Unsupported data type %s.",
                                         io::DenSupportedTypeToString(dataType).c_str());
        KCTERR(errMsg);
    }
    PRG.endLog(true);
    return 0;
}

