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
    std::string inputFile1;
    std::string inputFile2;
    std::string alphaFile;

    bool ignoreZeros = false;
    uint32_t dimx, dimy, dimz;
};

void Args::defineArguments()
{
    cliApp->add_option("input_den_file1", inputFile1, "File1 in a DEN format to process.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp
        ->add_option("input_den_file2", inputFile2,
                     "File2 in a DEN format to process. This file should have the same x,y and z "
                     "dimension as the input_den_file1.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_flag("--ignore-zeros", ignoreZeros,
                     "Exclude zero values in the files from the computation.");
    cliApp->add_option("--alpha", alphaFile,
                     "Specify mask file.");
    addFramespecArgs();
}

int Args::postParse()
{
    // If force is not set, then check if output file does not exist
    std::string err;
    // How many projection matrices is there in total
    io::DenFileInfo di(inputFile1);
    io::DenSupportedType dataType = di.getDataType();
    dimx = di.dimx();
    dimy = di.dimy();
    dimz = di.dimz();
    std::string f = inputFile2;
    {
        io::DenFileInfo df(f);
        if(df.getDataType() != dataType)
        {
            err = io::xprintf("File %s and %s are of different element types.", inputFile1.c_str(),
                              f.c_str());
            LOGE << err;
            return 1;
        }
        if(df.dimx() != dimx || df.dimy() != dimy || df.dimz() != dimz)
        {
            err = io::xprintf("Files %s and %s do not have the same dimensions.",
                              inputFile1.c_str(), f.c_str());
            LOGE << err;
            return 1;
        }
    }
    fillFramesVector(dimz);
    return 0;
}

template <typename T>
void reportFrameStatistics(uint32_t frameID,
                           std::shared_ptr<io::Frame2DReaderI<T>> reader1,
                           std::shared_ptr<io::Frame2DReaderI<T>> reader2,
                           std::shared_ptr<io::Frame2DReaderI<T>> readerAlpha,
                           bool ignoreZeros,
                           double& sum1,
                           double& sum2,
                           double& sum12,
                           double& sumSquares1,
                           double& sumSquares2,
                           uint64_t& elementsIncluded)
{
    std::shared_ptr<io::Frame2DI<T>> f1 = reader1->readFrame(frameID);
    std::shared_ptr<io::Frame2DI<T>> f2 = reader2->readFrame(frameID);
    uint32_t dimx = f1->dimx();
    uint32_t dimy = f1->dimy();
    std::shared_ptr<io::Frame2DI<T>> fAlpha = nullptr;
    if(readerAlpha != nullptr)
    {
        fAlpha = readerAlpha->readFrame(frameID);
    }
    for(uint32_t y = 0; y != dimy; y++)
    {
        for(uint32_t x = 0; x != dimx; x++)
        {
            T a1 = f1->get(x, y);
            T a2 = f2->get(x, y);
            float alpha = 1.0f;
            if(readerAlpha != nullptr)
            {
                alpha = fAlpha->get(x, y);
            }
            if(alpha != 0)
            {
                if(ignoreZeros)
                {
                    if(a1 == 0.0 || a2 == 0.0)
                    {
                        continue;
                    }
                }
                elementsIncluded++;
                sum1 += a1;
                sum2 += a2;
                sum12 += a1 * a2;
                sumSquares1 += a1 * a1;
                sumSquares2 += a2 * a2;
            }
        }
    }
}

template <typename T>
double PearsonsCorrelation(Args a)
{

    std::shared_ptr<io::Frame2DReaderI<T>> reader1
        = std::make_shared<io::DenFrame2DReader<T>>(a.inputFile1);
    std::shared_ptr<io::Frame2DReaderI<T>> reader2
        = std::make_shared<io::DenFrame2DReader<T>>(a.inputFile2);
    std::shared_ptr<io::Frame2DReaderI<T>> readerAlpha = nullptr;
    if(a.alphaFile != "")
    {
        readerAlpha = std::make_shared<io::DenFrame2DReader<T>>(a.alphaFile);
    }
    double sum1 = 0.0, sum2 = 0.0, sum12 = 0.0, sumSquares1 = 0.0, sumSquares2 = 0.0;
    uint64_t elementsIncluded = 0;
    bool ignoreZeros = a.ignoreZeros;
    for(std::size_t i = 0; i != a.frames.size(); i++)
    {
        double sum1_loc = 0.0, sum2_loc = 0.0, sum12_loc = 0.0, sumSquares1_loc = 0.0,
               sumSquares2_loc = 0.0;
        uint64_t elementsIncluded_loc = 0;
        reportFrameStatistics<T>(a.frames[i], reader1, reader2, readerAlpha, ignoreZeros, sum1_loc,
                                 sum2_loc, sum12_loc, sumSquares1_loc, sumSquares2_loc,
                                 elementsIncluded_loc);
        sum1 += sum1_loc;
        sum2 += sum2_loc;
        sum12 += sum12_loc;
        sumSquares1 += sumSquares1_loc;
        sumSquares2 += sumSquares2_loc;
        elementsIncluded += elementsIncluded_loc;
    }
    double pcf = (elementsIncluded * sum12 - sum1 * sum2)
        / std::sqrt((elementsIncluded * sumSquares1 - sum1 * sum1)
                    * (elementsIncluded * sumSquares2 - sum2 * sum2));
    return pcf;
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    Args ARG(argc, argv,
             "Compute correlation coefficient between two DEN files possibly using a mask.");
    // Argument parsing
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    //PRG.startLog(true);
    // Frames to process
    io::DenFileInfo inf(ARG.inputFile1);
    io::DenSupportedType dataType = inf.getDataType();
    double cor;
    switch(dataType)
    {
    case io::DenSupportedType::uint16_t_:
    {
        cor = PearsonsCorrelation<uint16_t>(ARG);
        break;
    }
    case io::DenSupportedType::float_:
    {
        cor = PearsonsCorrelation<float>(ARG);
        break;
    }
    case io::DenSupportedType::double_:
    {
        cor = PearsonsCorrelation<double>(ARG);
        break;
    }
    default:
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(dataType));
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
    }
    std::cout << cor << std::endl;
    //LOGI << io::xprintf("END %s", argv[0]);
}
