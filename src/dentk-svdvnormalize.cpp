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
struct Args : public ArgumentsForce
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName){};

    /// Folder to which output merged DEN file
    std::string outputMatrix;

    /// Folder to which output merged DEN file
    std::string inputMatrix;

    uint32_t dimx, dimy, dimz;
    uint32_t granularity, numVectors;
    std::vector<int> flipAtPosition;
    std::vector<int> preserveAtPosition;
};

void Args::defineArguments()
{
    cliApp
        ->add_option(
            "input_matrix", inputMatrix,
            "Input matrix N x 1 x L, where N is a granularity and L is a number of vectors.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp
        ->add_option("output_matrix", outputMatrix,
                     "Output matrix, which can be used as an input for SVD algorithm.")
        ->required();
    cliApp->add_option("--flip", flipAtPosition,
                       "Specify intices to multiply vector by -1 at specified positions without "
                       "further processing.",
                       true);
    cliApp->add_option("--preserve", preserveAtPosition,
                       "Specify indices to preserve vector without further processing.", true);
    addForceArgs();
}

int Args::postParse()
{
    std::string ERR;
    io::DenFileInfo di(inputMatrix);
    dimx = di.dimx();
    dimy = di.dimy();
    dimz = di.dimz();
    if(dimy != 1)
    {
        ERR = io::xprintf("Y dimension of the supplied object is %d and shall be 1!", dimy);
        LOGE << ERR;
        return -1;
    }
    granularity = dimx;
    numVectors = dimz;
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
void processFiles(Args ARG)
{
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> volumeWritter;
    volumeWritter = std::make_shared<io::DenAsyncFrame2DWritter<T>>(
        ARG.outputMatrix, ARG.granularity, 1, ARG.numVectors);
    std::shared_ptr<io::Frame2DReaderI<T>> inputReader
        = std::make_shared<io::DenFrame2DReader<T>>(ARG.inputMatrix);
    std::shared_ptr<io::Frame2DI<T>> inputFrame;
    double min, max;
    for(uint64_t k = 0; k != ARG.dimz; k++)
    {
        inputFrame = inputReader->readFrame(k);
        min = (double)io::minFrameValue<T>(*inputFrame);
        max = (double)io::maxFrameValue<T>(*inputFrame);
        if(std::find(ARG.preserveAtPosition.begin(), ARG.preserveAtPosition.end(), k)
           != ARG.preserveAtPosition.end())
        {
            // Do nothing
        } else if(std::find(ARG.flipAtPosition.begin(), ARG.flipAtPosition.end(), k)
                  != ARG.flipAtPosition.end())
        {
            for(uint32_t i = 0; i != ARG.dimx; i++)
            {
                inputFrame->set(-inputFrame->get(i, 0), i, 0);
            }
        } else if(max < 0.0 || (max > 0 && min < 0 && std::abs(min) > max))
        {
            for(uint32_t i = 0; i != ARG.dimx; i++)
            {
                inputFrame->set(-inputFrame->get(i, 0), i, 0);
            }
        }
        volumeWritter->writeFrame(*inputFrame, k);
    }
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    const std::string prgInfo
        = "Normalize the engineered basis produced by SVD to be human readable.";
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
    io::DenFileInfo di(ARG.inputMatrix);
    io::DenSupportedType dataType = di.getDataType();
    switch(dataType)
    {
    case io::DenSupportedType::UINT16: {
        processFiles<uint16_t>(ARG);
        break;
    }
    case io::DenSupportedType::FLOAT32: {
        processFiles<float>(ARG);
        break;
    }
    case io::DenSupportedType::FLOAT64: {
        processFiles<double>(ARG);
        break;
    }
    default: {
        std::string errMsg = io::xprintf("Unsupported data type %s.",
                                         io::DenSupportedTypeToString(dataType).c_str());
        KCTERR(errMsg);
    }
    }
    PRG.endLog();
}
