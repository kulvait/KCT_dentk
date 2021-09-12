// Logging
#include "PLOG/PlogSetup.h"
// External libraries
#include "CLI/CLI.hpp" //Command line parser
// Program class
#include "PROG/Program.hpp"

//#include "FittingExecutor.hpp"
#include "AsyncFrame2DWritterI.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "FrameMemoryViewer2D.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "SVD/TikhonovInverse.hpp"
#include "frameop.h"
#include "stringFormatter.h"

#if DEBUG
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
#endif

using namespace KCT;

/// Arguments of the main function.
using namespace KCT;
using namespace KCT::util;

// class declarations
struct Args : public ArgumentsForce, public ArgumentsFramespec
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName)
        , ArgumentsFramespec(argc, argv, prgName){};

    /// Folder to which output merged DEN file
    std::string outputMatrix;

    /// Folder to which output merged DEN file
    std::string alphaFile;

    /// Static reconstructions in a DEN format to use for linear regression.
    std::vector<std::string> individualVolumeFiles;

    uint32_t dimx, dimy, dimz;
    uint32_t granularity;
    bool subtractMean = false;
    bool subtractFirstValue = false;
};

void Args::defineArguments()
{

    cliApp
        ->add_option("output_matrix", outputMatrix,
                     "Output matrix, which can be used as an input for SVD algorithm.")
        ->required();
    cliApp->add_option("volume_files", individualVolumeFiles, "Individual volume files.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("--alpha", alphaFile, "Alpha filtering.")->check(CLI::ExistingFile);
    CLI::Option_group* op_clg = cliApp->add_option_group(
        "Postprocessing operation", "Postprocessing operation to perform on temporal vectors.");
    op_clg->add_flag("--subtract-mean", subtractMean,
                     "Subtract mean of each vector from each element.");
    op_clg->add_flag("--subtract-first-value", subtractFirstValue,
                     "Subtract first value of each vector from each element.");
    op_clg->require_option(0, 1);
    addFramespecArgs();
    addForceArgs();
}

int Args::postParse()
{
    io::DenFileInfo di(individualVolumeFiles[0]);
    dimx = di.dimx();
    dimy = di.dimy();
    dimz = di.dimz();
    fillFramesVector(dimz);
    granularity = individualVolumeFiles.size();
    if(granularity < 2)
    {
        std::string err
            = io::xprintf("Small number of input files %d.", individualVolumeFiles.size());
        LOGE << err;
        return -1;
    }
    for(std::string f : individualVolumeFiles)
    {
        io::DenFileInfo df(f);
        if(dimx != df.dimx() || dimy != df.dimy() || dimz != df.dimz())
        {
            std::string err = io::xprintf("Dimension check for the file %s fails.", f.c_str());
            LOGE << err;
            return -1;
        }
    }
    if(!force)
    {
        if(io::pathExists(outputMatrix))
        {
            std::string msg = io::xprintf(
                "Error: output file %s already exists, use --force to force overwrite.",
                outputMatrix.c_str());
            LOGE << msg;
            return 1;
        }
    }
    return 0;
}

template <typename T>
void processFilesFast(Args ARG)
{

    io::createEmptyFile(ARG.outputMatrix, 0, true);
    // First compute dimensions of the output file
    uint64_t numberOfVectors = 0;
    std::shared_ptr<io::Frame2DReaderI<T>> alphaReader = nullptr;
    if(ARG.alphaFile == "")
    {
        numberOfVectors = uint64_t(ARG.dimx) * uint64_t(ARG.dimy) * uint64_t(ARG.frames.size());
    } else
    {
        alphaReader = std::make_shared<io::DenFrame2DReader<T>>(ARG.alphaFile);
        for(const int f : ARG.frames)
        {
            std::shared_ptr<io::Frame2DI<T>> framePtr = alphaReader->readFrame(f);
            numberOfVectors += io::sumNonzeroValues<T>(*framePtr);
        }
    }
    if(numberOfVectors > 4294967295)
    {
        std::string msg = io::xprintf(
            "Number of vectors is %d that is bigger than maximal representable value of uint_32!",
            numberOfVectors);
        LOGE << msg;
        throw std::runtime_error(msg);
    }
    uint8_t buf[18];
    util::putUint16(0, &buf[0]);
    util::putUint16(0, &buf[2]);
    util::putUint16(0, &buf[4]);
    util::putUint32(1, &buf[6]);
    util::putUint32(ARG.granularity, &buf[10]);
    util::putUint32(numberOfVectors, &buf[14]);
    io::appendBytes(ARG.outputMatrix, buf, 18);
    uint64_t pos = 0;
    uint64_t arrayPos = 0;
    uint64_t arrayVectors = 500;
    T* array = new T[arrayVectors * ARG.granularity];
    T* arrayHead = array;
    std::vector<T> vector;
    vector.resize(ARG.granularity);
    std::vector<std::shared_ptr<io::Frame2DReaderI<T>>> fileReaders;
    for(const std::string& vf : ARG.individualVolumeFiles)
    {
        fileReaders.emplace_back(std::make_shared<io::DenFrame2DReader<T>>(vf));
    }
    std::vector<std::shared_ptr<io::Frame2DI<T>>> currentFrames;
    std::shared_ptr<io::Frame2DI<T>> alphaFrame = nullptr;
    for(const uint32_t f : ARG.frames)
    {
        LOGI << io::xprintf("Processing frame %d", f);
        currentFrames.clear();
        for(uint32_t k = 0; k != fileReaders.size(); k++)
        {
            currentFrames.emplace_back(fileReaders[k]->readFrame(f));
        }
        if(alphaReader != nullptr)
        {
            alphaFrame = alphaReader->readFrame(f);
        }
        for(uint32_t i = 0; i != ARG.dimx; i++)
        {
            for(uint32_t j = 0; j != ARG.dimy; j++)
            {
                if(alphaReader == nullptr || alphaFrame->get(i, j) != T(0))
                {
                    for(uint32_t k = 0; k != ARG.granularity; k++)
                    {
                        vector[k] = currentFrames[k]->get(i, j);
                    }
                    if(ARG.subtractFirstValue)
                    {
                        T valToSubtract = vector[0];
                        for(uint32_t k = 0; k != ARG.granularity; k++)
                        {
                            vector[k] -= valToSubtract;
                        }
                    } else if(ARG.subtractMean)
                    {
                        double sum = std::accumulate(vector.begin(), vector.end(), 0.0);
                        double mean = sum / vector.size();
                        T valToSubtract = T(mean);
                        for(uint32_t k = 0; k != ARG.granularity; k++)
                        {
                            vector[k] -= valToSubtract;
                        }
                    }
                    std::copy(vector.begin(), vector.end(), arrayHead);
                    arrayHead += ARG.granularity;
                    pos++;
                    arrayPos++;
                    if(arrayPos == arrayVectors)
                    {
                        io::appendBytes(ARG.outputMatrix, (uint8_t*)array,
                                        sizeof(T) * arrayVectors * ARG.granularity);
                        arrayPos = 0;
                        arrayHead = array;
                    }
                }
            }
        }
    }
    if(arrayPos > 0)
    {
        io::appendBytes(ARG.outputMatrix, (uint8_t*)array, sizeof(T) * arrayPos * ARG.granularity);
        arrayPos = 0;
        arrayHead = array;
    }
    delete[] array;
}

template <typename T>
void processFiles(Args ARG)
{
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> volumeWritter;
    // First compute dimensions of the output file
    uint64_t numberOfVectors = 0;
    std::shared_ptr<io::Frame2DReaderI<T>> alphaReader = nullptr;
    if(ARG.alphaFile == "")
    {
        numberOfVectors = uint64_t(ARG.dimx) * uint64_t(ARG.dimy) * uint64_t(ARG.frames.size());
    } else
    {
        alphaReader = std::make_shared<io::DenFrame2DReader<T>>(ARG.alphaFile);
        for(const int f : ARG.frames)
        {
            std::shared_ptr<io::Frame2DI<T>> framePtr = alphaReader->readFrame(f);
            numberOfVectors += io::sumNonzeroValues<T>(*framePtr);
        }
    }
    if(numberOfVectors > 4294967295)
    {
        std::string msg = io::xprintf(
            "Number of vectors is %d that is bigger than maximal representable value of uint_32!",
            numberOfVectors);
        LOGE << msg;
        throw std::runtime_error(msg);
    }
    volumeWritter = std::make_shared<io::DenAsyncFrame2DWritter<T>>(
        ARG.outputMatrix, ARG.granularity, 1, numberOfVectors);
    uint64_t pos = 0;
    std::vector<T> vector;
    vector.resize(ARG.granularity);
    std::vector<std::shared_ptr<io::Frame2DReaderI<T>>> fileReaders;
    for(const std::string& vf : ARG.individualVolumeFiles)
    {
        fileReaders.emplace_back(std::make_shared<io::DenFrame2DReader<T>>(vf));
    }
    std::vector<std::shared_ptr<io::Frame2DI<T>>> currentFrames;
    std::shared_ptr<io::Frame2DI<T>> alphaFrame = nullptr;
    for(const uint32_t f : ARG.frames)
    {
        LOGI << io::xprintf("Processing frame %d", f);
        currentFrames.clear();
        for(uint32_t k = 0; k != fileReaders.size(); k++)
        {
            currentFrames.emplace_back(fileReaders[k]->readFrame(f));
        }
        if(alphaReader != nullptr)
        {
            alphaFrame = alphaReader->readFrame(f);
        }
        for(uint32_t i = 0; i != ARG.dimx; i++)
        {
            for(uint32_t j = 0; j != ARG.dimy; j++)
            {
                if(alphaReader == nullptr || alphaFrame->get(i, j) != T(0))
                {
                    for(uint32_t k = 0; k != ARG.granularity; k++)
                    {
                        vector[k] = currentFrames[k]->get(i, j);
                    }
                    if(ARG.subtractFirstValue)
                    {
                        T valToSubtract = vector[0];
                        for(uint32_t k = 0; k != ARG.granularity; k++)
                        {
                            vector[k] -= valToSubtract;
                        }
                    } else if(ARG.subtractMean)
                    {
                        double sum = std::accumulate(vector.begin(), vector.end(), 0.0);
                        double mean = sum / vector.size();
                        T valToSubtract = T(mean);
                        for(uint32_t k = 0; k != ARG.granularity; k++)
                        {
                            vector[k] -= valToSubtract;
                        }
                    }
                    std::unique_ptr<io::Frame2DI<T>> memoryViewer
                        = std::make_unique<io::FrameMemoryViewer2D<T>>(vector.data(),
                                                                       ARG.granularity, 0);
                    volumeWritter->writeFrame(*memoryViewer, pos);
                    pos++;
                }
            }
        }
    }
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    const std::string prgInfo
        = "From the series of the DEN files represented as volumes creates the SVD input matrix of "
          "the size T x 1 x N, where T is time granularity corresponding to number of input "
          "volumes and N is produced number of rows in the SVD matrix that might be dimx x dimy x "
          "dimz  of the volume but also less due to filtering of the frames or alpha filtering.";
    Args ARG(argc, argv, prgInfo);
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    io::DenFileInfo di(ARG.individualVolumeFiles[0]);
    io::DenSupportedType dataType = di.getDataType();
    switch(dataType)
    {
    case io::DenSupportedType::uint16_t_:
    {
        processFiles<uint16_t>(ARG);
        break;
    }
    case io::DenSupportedType::float_:
    {
        processFilesFast<float>(ARG);
        break;
    }
    case io::DenSupportedType::double_:
    {
        processFiles<double>(ARG);
        break;
    }
    default:
    {
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(dataType));
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
    }
    }
    PRG.endLog();
}
