// Logging
#include "PLOG/PlogSetup.h"

// External libraries

// External libraries
#include "CLI/CLI.hpp"
#include "mkl.h"

// Internal libraries
#include "AsyncFrame2DWritterI.hpp"
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DBufferedWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "FUN/StepFunction.hpp"
#include "Frame2DI.hpp"
#include "MATRIX/Matrix.hpp"
#include "MATRIX/RQFactorization.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/Program.hpp"
#include "frameop.h"
#include "rawop.h"

using namespace KCT;

template <typename T>
using READER = io::DenFrame2DReader<T>;

template <typename T>
using READERPTR = std::shared_ptr<READER<T>>;

template <typename T>
using WRITER = io::DenAsyncFrame2DBufferedWritter<T>;

template <typename T>
using WRITERPTR = std::shared_ptr<WRITER<T>>;

template <typename T>
using FRAME = io::BufferedFrame2D<T>;

template <typename T>
using FRAMEPTR = std::shared_ptr<FRAME<T>>;

using namespace KCT::util;

// class declarations
class Args : public ArgumentsForce, public ArgumentsFramespec
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName)
        , ArgumentsFramespec(argc, argv, prgName){};
    std::string input_file = "";
    std::string output_file = "";
    uint32_t frameSize;
};

/**
 * Orthogonalization by RQ decomposition from the last vector. So we reorder.
 *
 * @param inputFile
 * @param outputFile
 */
template <typename T>
void process(Args ARG)
{
    READERPTR<T> frameReader = std::make_shared<READER<T>>(ARG.input_file);
    uint64_t functionCount = ARG.frames.size(); //baseSize
    uint64_t frameSize = ARG.frameSize; //granularity
    uint32_t dimx = frameReader->dimx();
    uint32_t dimy = frameReader->dimy();
    double* values = new double[functionCount * frameSize];
    FRAMEPTR<T> f;
    uint32_t IND = 0;
    for(uint64_t i = 0; i != functionCount; i++)
    {
        IND = ARG.frames
                  [ARG.frames.size() - 1
                   - i]; //First vector will be last in the matrix, and ortogonalized version will be last element in Q matrix
        f = frameReader->readBufferedFrame(IND);
        T* f_array = f->getDataPointer();
        std::copy(f_array, f_array + frameSize, values + i * frameSize);
    }
    matrix::RQFactorization rq;
    std::shared_ptr<matrix::Matrix> B
        = std::make_shared<matrix::Matrix>(functionCount, frameSize, values);
    rq.factorize(B);
    auto C = rq.getRMatrix(); // functionCount*functionCount
    auto Q = rq.getQMatrix(); // functionCount*frameSize
    double a = 0.0;
    std::vector<uint32_t> nonzeroVectorIndices;
    for(uint32_t i = 0; i != functionCount; i++)
    {
        IND = functionCount - 1 - i;
        a = 0.0;
        for(uint32_t j = 0; j != functionCount; j++)
        {
            a += C->get(j, IND) * C->get(j, IND);
        }
        if(std::sqrt(a) > 1e-10)
        {
            nonzeroVectorIndices.push_back(IND);
        }
    }
    WRITERPTR<T> w
        = std::make_shared<WRITER<T>>(ARG.output_file, dimx, dimy, nonzeroVectorIndices.size());
    FRAMEPTR<T> bf = std::make_shared<FRAME<T>>(T(0), dimx, dimy);
    T* bf_array = bf->getDataPointer();
    for(uint32_t i = 0; i != nonzeroVectorIndices.size(); i++)
    {
        IND = nonzeroVectorIndices[i];
        for(uint64_t k = 0; k != frameSize; k++)
        {
            bf_array[k] = Q->get(IND, k);
        }
        w->writeBufferedFrame(*bf, i);
    }
    delete[] values;
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    Args ARG(argc, argv,
             "Orthogonalization of the frames by means of QR algorithm, cut colinear vectors. We "
             "expect, that the z dimension of the file will correspond to different vectors and we "
             "orthogonalize dimx*dimy frames.");
    int parseResult = ARG.parse();
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
    switch(dataType)
    {
    case io::DenSupportedType::UINT16: {
        process<uint16_t>(ARG);
        break;
    }
    case io::DenSupportedType::FLOAT32: {
        process<float>(ARG);
        break;
    }
    case io::DenSupportedType::FLOAT64: {
        process<double>(ARG);
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

void Args::defineArguments()
{
    cliApp->add_option("input_file", input_file, "Input file.")
        ->check(CLI::ExistingFile)
        ->required();
    cliApp->add_option("output_file", output_file, "Output file.")->required();
    addForceArgs();
    addFramespecArgs();
}

int Args::postParse()
{
    bool removeIfExists = true;
    int existFlag = handleFileExistence(output_file, force, removeIfExists);
    if(existFlag != 0)
    {
        std::string msg
            = io::xprintf("Error: output file %s already exists, use --force to force overwrite.",
                          output_file.c_str());
        LOGE << msg;
        return 1;
    }
    io::DenFileInfo di(input_file);
    if(di.getDimCount() != 3)
    {
        std::string ERR
            = io::xprintf("The file %s has %d dimensions, three dimensional file expected.",
                          input_file.c_str(), di.getDimCount());
        LOGE << ERR;
        return 1;
    }
    fillFramesVector(di.getFrameCount());
    frameSize = di.getFrameSize();
    return 0;
}
