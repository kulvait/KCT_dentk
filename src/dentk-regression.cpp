#include "PLOG/PlogSetup.h"

// External libraries
#include "CLI/CLI.hpp" //Command line parser
#include "ftpl.h" //Threadpool

// Internal libraries
#include "AsyncFrame2DWritterI.hpp"
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "DEN/DenSupportedType.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "PROG/Program.hpp"
#include "littleEndianAlignment.h"
#include "rawop.h"

using namespace KCT;
using namespace KCT::util;

class Args : public ArgumentsFramespec
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsFramespec(argc, argv, prgName){};
    std::string inputFileX;
    std::string inputFileY;
    std::string inputFileAlpha;

    bool noIntercept = false;
};

/**Argument parsing
 *
 */
void Args::defineArguments()
{
    cliApp
        ->add_option("input_file_x", inputFileX,
                     "Independent variable x in the equation y = a x + b.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp
        ->add_option("input_file_y", inputFileY,
                     "Dependent variable y in the equation y = a x + b .")
        ->required()
        ->check(CLI::ExistingFile);
    addFramespecArgs();
    cliApp->add_option("--alpha", inputFileAlpha,
                       "Alpha channel to consider, process only values for which alpha=1.");
    cliApp->add_flag("--no-intercept", noIntercept, "Use the equation y=ax instead of y=ax+b.");
}

int Args::postParse()
{
    std::string ERR;
    if(inputFileX == inputFileY)
    {
        LOGE << "Error: input files must differ!";
        return 1;
    }
    io::DenFileInfo ix(inputFileX);
    io::DenFileInfo iy(inputFileY);
    uint32_t dimx = ix.dimx();
    uint32_t dimy = ix.dimy();
    uint32_t dimz = ix.dimz();
    fillFramesVector(dimz);
    if(dimx != iy.dimx() || dimy != iy.dimy() || dimz != iy.dimz())
    {
        ERR = io::xprintf("Error: dimensions of files %s and %s differs", inputFileX.c_str(),
                          inputFileY.c_str());
        LOGE << ERR;
        return 1;
    }
    if(ix.getElementType() != iy.getElementType())
    {
        ERR = io::xprintf("Error: incompatible data types of files %s and %s.", inputFileX.c_str(),
                          inputFileY.c_str());
        LOGE << ERR;
        return 1;
    }
    if(!inputFileAlpha.empty())
    {
        io::DenFileInfo ia(inputFileAlpha);
        if(dimx != ia.dimx() || dimy != ia.dimy() || dimz != ia.dimz())
        {
            ERR = io::xprintf("Error: dimensions of files %s and %s differs", inputFileX.c_str(),
                              inputFileAlpha.c_str());
            LOGE << ERR;
            return 1;
        }
        if(ix.getElementType() != ia.getElementType())
        {
            ERR = io::xprintf("Error: incompatible data types of files %s and %s.",
                              inputFileX.c_str(), inputFileAlpha.c_str());
            LOGE << ERR;
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Generates \sum (x_i-offsetx)^2 / N, where N is given by frames and alpha
 *
 * @param X
 * @param xoffset
 * @param frames
 * @param alpha
 */
template <typename T>
double getSumOfSquares(std::shared_ptr<io::DenFileInfo> X,
                       double xoffset,
                       std::vector<uint64_t> frames,
                       std::shared_ptr<io::DenFileInfo> ALPHA)
{
    io::DenSupportedType dataType = X->getElementType();
    uint64_t dim_x = X->dimx();
    uint64_t dim_y = X->dimy();
    uint64_t offset = X->getOffset();
    uint64_t totalSize = 0;
    uint64_t currentPosition;
    uint32_t elementSize = X->getElementByteSize();
    double sum = 0.0;
    double valX;
    uint8_t* bufferX = new uint8_t[dim_x * dim_y * elementSize];
    uint8_t* bufferAlpha = nullptr;
    double valAlpha;
    if(ALPHA != nullptr)
    {
        bufferAlpha = new uint8_t[dim_x * dim_y * elementSize];
    }
    for(uint64_t k = 0; k != frames.size(); k++)
    {
        currentPosition = offset + frames[k] * dim_x * dim_y * elementSize;
        io::readBytesFrom(X->getFileName(), currentPosition, bufferX, dim_x * dim_y * elementSize);
        if(ALPHA == nullptr)
        {
            for(uint64_t pos = 0; pos != dim_y * dim_x; pos++)
            {
                valX = (double)util::getNextElement<T>(&bufferX[pos * elementSize], dataType);
                sum += (valX - xoffset) * (valX - xoffset);
                totalSize++;
            }
        } else
        {
            io::readBytesFrom(ALPHA->getFileName(), currentPosition, bufferAlpha,
                              dim_x * dim_y * elementSize);
            for(uint64_t pos = 0; pos != dim_y * dim_x; pos++)
            {
                valAlpha
                    = (double)util::getNextElement<T>(&bufferAlpha[pos * elementSize], dataType);
                if(valAlpha == 1.0)
                {
                    valX = (double)util::getNextElement<T>(&bufferX[pos * elementSize], dataType);
                    sum += (valX - xoffset) * (valX - xoffset);
                    totalSize++;
                }
            }
        }
    }
    delete[] bufferX;
    if(bufferAlpha != nullptr)
    {
        delete[] bufferAlpha;
    }
    return T(sum / totalSize);
}

/**
 * @brief Generates \sum (x_i-offsetx) (y_i-offsety) / N, where N is given by frames and alpha
 *
 * @param X
 * @param xoffset
 * @param Y
 * @param yoffset
 * @param frames
 * @param alpha
 */
template <typename T>
double getMeanOfOffsettedProducts(std::shared_ptr<io::DenFileInfo> X,
                                  double xoffset,
                                  std::shared_ptr<io::DenFileInfo> Y,
                                  double yoffset,
                                  std::vector<uint64_t> frames,
                                  std::shared_ptr<io::DenFileInfo> ALPHA)
{
    io::DenSupportedType dataType = X->getElementType();
    uint64_t dim_x = X->dimx();
    uint64_t dim_y = X->dimy();
    uint64_t offset = X->getOffset();
    uint64_t totalSize = 0;
    uint64_t currentPosition;
    uint32_t elementSize = X->getElementByteSize();
    double sum = 0.0;
    double valX, valY;
    uint8_t* bufferX = new uint8_t[dim_x * dim_y * elementSize];
    uint8_t* bufferY = new uint8_t[dim_x * dim_y * elementSize];
    uint8_t* bufferAlpha = nullptr;
    double alphaVal;
    if(ALPHA != nullptr)
    {
        bufferAlpha = new uint8_t[dim_x * dim_y * elementSize];
    }
    for(uint64_t k = 0; k != frames.size(); k++)
    {
        currentPosition = offset + frames[k] * dim_x * dim_y * elementSize;
        io::readBytesFrom(X->getFileName(), currentPosition, bufferX, dim_x * dim_y * elementSize);
        io::readBytesFrom(Y->getFileName(), currentPosition, bufferY, dim_x * dim_y * elementSize);
        if(ALPHA == nullptr)
        {
            for(uint64_t pos = 0; pos != dim_y * dim_x; pos++)
            {
                valX = (double)util::getNextElement<T>(&bufferX[pos * elementSize], dataType);
                valY = (double)util::getNextElement<T>(&bufferY[pos * elementSize], dataType);
                sum += (valX - xoffset) * (valY - yoffset);
                totalSize++;
            }
        } else
        {
            io::readBytesFrom(ALPHA->getFileName(), currentPosition, bufferAlpha,
                              dim_x * dim_y * elementSize);
            for(uint64_t pos = 0; pos != dim_y * dim_x; pos++)
            {
                alphaVal
                    = (double)util::getNextElement<T>(&bufferAlpha[pos * elementSize], dataType);
                if(alphaVal == 1.0)
                {
                    valX = (double)util::getNextElement<T>(&bufferX[pos * elementSize], dataType);
                    valY = (double)util::getNextElement<T>(&bufferY[pos * elementSize], dataType);
                    sum += (valX - xoffset) * (valY - yoffset);
                    totalSize++;
                }
            }
        }
    }
    delete[] bufferX;
    delete[] bufferY;
    if(bufferAlpha != nullptr)
    {
        delete[] bufferAlpha;
    }
    return T(sum / totalSize);
}

template <typename T>
double getMean(std::shared_ptr<io::DenFileInfo> X,
               std::vector<uint64_t> frames,
               std::shared_ptr<io::DenFileInfo> ALPHA)
{
    io::DenSupportedType dataType = X->getElementType();
    uint64_t dim_x = X->dimx();
    uint64_t dim_y = X->dimy();
    uint64_t offset = X->getOffset();
    uint64_t totalSize = 0;
    uint64_t currentPosition;
    uint32_t elementSize = X->getElementByteSize();
    double sum = 0.0;
    double val;
    uint8_t* buffer = new uint8_t[dim_x * dim_y * elementSize];
    uint8_t* bufferAlpha = nullptr;
    double valAlpha;
    if(ALPHA != nullptr)
    {
        bufferAlpha = new uint8_t[dim_x * dim_y * elementSize];
    }
    for(uint64_t k = 0; k != frames.size(); k++)
    {
        currentPosition = offset + frames[k] * dim_x * dim_y * elementSize;
        io::readBytesFrom(X->getFileName(), currentPosition, buffer, dim_x * dim_y * elementSize);
        if(ALPHA == nullptr)
        {
            for(uint64_t pos = 0; pos != dim_y * dim_x; pos++)
            {
                val = (double)util::getNextElement<T>(&buffer[pos * elementSize], dataType);
                sum += val;
                totalSize++;
            }
        } else
        {
            io::readBytesFrom(ALPHA->getFileName(), currentPosition, bufferAlpha,
                              dim_x * dim_y * elementSize);
            for(uint64_t pos = 0; pos != dim_y * dim_x; pos++)
            {
                valAlpha
                    = (double)util::getNextElement<T>(&bufferAlpha[pos * elementSize], dataType);
                if(valAlpha == 1)
                {
                    val = (double)util::getNextElement<T>(&buffer[pos * elementSize], dataType);
                    sum += val;
                    totalSize++;
                }
            }
        }
    }
    delete[] buffer;
    if(bufferAlpha != nullptr)
    {
        delete[] bufferAlpha;
    }
    return T(sum / totalSize);
}

template <typename T>
/**
 * @brief Compute (1/N) \sum (Y_i - a X_i - b)^2, where N is given by frames and alpha
 *
 * @param X
 * @param Y
 * @param a
 * @param b
 * @param frames
 * @param alpha
 *
 * @return
 */
double getMeanSquareResiduum(std::shared_ptr<io::DenFileInfo> X,
                             std::shared_ptr<io::DenFileInfo> Y,
                             double a,
                             double b,
                             std::vector<uint64_t> frames,
                             std::shared_ptr<io::DenFileInfo> ALPHA)
{
    io::DenSupportedType dataType = X->getElementType();
    uint64_t dim_x = X->dimx();
    uint64_t dim_y = X->dimy();
    uint64_t offset = X->getOffset();
    uint64_t totalSize = 0;
    uint64_t currentPosition;
    uint32_t elementSize = X->getElementByteSize();
    double sum = 0.0;
    double valX, valY, val;
    uint8_t* bufferX = new uint8_t[dim_x * dim_y * elementSize];
    uint8_t* bufferY = new uint8_t[dim_x * dim_y * elementSize];
    uint8_t* bufferAlpha = nullptr;
    double valAlpha;
    if(ALPHA != nullptr)
    {
        bufferAlpha = new uint8_t[dim_x * dim_y * elementSize];
    }
    for(uint64_t k = 0; k != frames.size(); k++)
    {
        currentPosition = offset + frames[k] * dim_x * dim_y * elementSize;
        io::readBytesFrom(X->getFileName(), currentPosition, bufferX, dim_x * dim_y * elementSize);
        io::readBytesFrom(Y->getFileName(), currentPosition, bufferY, dim_x * dim_y * elementSize);
        if(ALPHA == nullptr)
        {
            for(uint64_t pos = 0; pos != dim_y * dim_x; pos++)
            {
                valX = (double)util::getNextElement<T>(&bufferX[pos * elementSize], dataType);
                valY = (double)util::getNextElement<T>(&bufferY[pos * elementSize], dataType);
                val = valY - a * valX - b;
                sum += val * val;
                totalSize++;
            }
        } else
        {
            io::readBytesFrom(ALPHA->getFileName(), currentPosition, bufferAlpha,
                              dim_x * dim_y * elementSize);
            for(uint64_t pos = 0; pos != dim_y * dim_x; pos++)
            {
                valAlpha
                    = (double)util::getNextElement<T>(&bufferAlpha[pos * elementSize], dataType);
                if(valAlpha == 1)
                {
                    valX = (double)util::getNextElement<T>(&bufferX[pos * elementSize], dataType);
                    valY = (double)util::getNextElement<T>(&bufferY[pos * elementSize], dataType);
                    val = valY - a * valX - b;
                    sum += val * val;
                    totalSize++;
                }
            }
        }
    }
    delete[] bufferX;
    delete[] bufferY;
    if(bufferAlpha != nullptr)
    {
        delete[] bufferAlpha;
    }
    return T(sum / totalSize);
}

template <typename T>
void processFiles(Args ARG)
{
    std::shared_ptr<io::DenFileInfo> X = std::make_shared<io::DenFileInfo>(ARG.inputFileX);
    std::shared_ptr<io::DenFileInfo> Y = std::make_shared<io::DenFileInfo>(ARG.inputFileY);
    std::shared_ptr<io::DenFileInfo> A = nullptr;
    if(!ARG.inputFileAlpha.empty())
    {
        A = std::make_shared<io::DenFileInfo>(ARG.inputFileAlpha);
    }
    double a, b;
    double covariance, variance;
    double meanSquareResiduum, totalSumOfSquares, Rsquared;
    if(ARG.noIntercept)
    {
        covariance = getMeanOfOffsettedProducts<T>(X, 0, Y, 0, ARG.frames, A);
        variance = getSumOfSquares<T>(X, 0, ARG.frames, A);
        a = covariance / variance;
        b = 0.0;
        meanSquareResiduum = getMeanSquareResiduum<T>(X, Y, a, 0, ARG.frames, A);
        totalSumOfSquares = getSumOfSquares<T>(Y, 0.0, ARG.frames, A);
        LOGI << io::xprintf("The resulting equation Y=aX yields a=%f.", a);
    } else
    {
        double meanX = getMean<T>(X, ARG.frames, A);
        double meanY = getMean<T>(Y, ARG.frames, A);
        covariance = getMeanOfOffsettedProducts<T>(X, meanX, Y, meanY, ARG.frames, A);
        variance = getSumOfSquares<T>(X, meanX, ARG.frames, A);
        a = covariance / variance;
        b = meanY - a * meanX;
        meanSquareResiduum = getMeanSquareResiduum<T>(X, Y, a, b, ARG.frames, A);
        totalSumOfSquares = getSumOfSquares<T>(Y, meanY, ARG.frames, A);
        LOGI << io::xprintf("The resulting equation Y=aX + b yields a=%f, b=%f.", a, b);
    }
    Rsquared = 1 - meanSquareResiduum / totalSumOfSquares;
    LOGI << io::xprintf("R^2 is %f.", Rsquared);
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    Args ARG(argc, argv, "Linear regression of the two DEN files.");
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    io::DenFileInfo di(ARG.inputFileX);
    io::DenSupportedType dataType = di.getElementType();
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
