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
#include "DEN/DenFrame2DCachedReader.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "FrameMemoryViewer2D.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/ArgumentsThreading.hpp"
#include "SVD/TikhonovInverse.hpp"
#include "frameop.h"
#include "ftpl.h"
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
struct Args : public ArgumentsForce, public ArgumentsFramespec, public ArgumentsThreading
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName)
        , ArgumentsFramespec(argc, argv, prgName)
        , ArgumentsThreading(argc, argv, prgName){};

    /// DEN file with the basis
    std::string inputBasis;
    /// DEN file with the input images
    std::string inputImages;
    /// DEN file to write orthogonal projections
    std::string outputOrthogonalProjection;
    /// DEN file to write complements to the orthogonal projections
    std::string outputOrthogonalComplement;
    std::string scalarProductInfo = "";
    uint32_t dimx, dimy, dimz;
    uint32_t basis_dimz;
};

void Args::defineArguments()
{
    cliApp
        ->add_option("input_basis", inputBasis,
                     "Collection of the orthonormal basis vectors to be used for the "
                     "decomposition, can be represented by dimx x dimy images, which must match "
                     "with dimx x dimy dimensions of the projection images. Frame specification is "
                     "applied on the inputBasis to select vectors of the basis used for the direct "
                     "sum decomposition.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("input_images", inputImages, "Image data to be processed.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp
        ->add_option("output_orthogonal_projection", outputOrthogonalProjection,
                     "Output file with orthogonal projections.")
        ->required();
    cliApp
        ->add_option("output_orthogonal_complement", outputOrthogonalComplement,
                     "Output file with orthogonal complement.")
        ->required();
    cliApp->add_option("--scalar-product-info", scalarProductInfo,
                       "If specified, writes scalar products with selected components of the basis "
                       "into the file.");
    addForceArgs();
    addFramespecArgs();
    addThreadingArgs();
}

int handleFileExistence(std::string f, bool force, bool removeIfExist)
{
    std::string ERR;
    if(io::pathExists(f))
    {
        if(force)
        {
            if(removeIfExist)
            {
                LOGI << io::xprintf("Removing existing file %s", f.c_str());
                std::remove(f.c_str());
            }
        } else
        {
            ERR = io::xprintf(
                "Error: output file %s already exists, use --force to force overwrite.", f.c_str());
            LOGE << ERR;
            return 1;
        }
    }
    return 0;
}

int Args::postParse()
{
    std::string ERR;
    io::DenFileInfo inf_basis(inputBasis);
    io::DenFileInfo inf_img(inputImages);
    dimx = inf_basis.dimx();
    dimy = inf_basis.dimy();
    basis_dimz = inf_basis.dimz();
    fillFramesVector(basis_dimz);
    dimz = inf_img.dimz();
    if(dimx != inf_img.dimx() || dimy != inf_img.dimy())
    {
        ERR = io::xprintf("Dimensions (dimx, dimy)=(%d, %d) of the basis file %s and (dimx, "
                          "dimy)=(%d, %d) of the image file %s are incompatible!",
                          dimx, dimy, inputBasis.c_str(), inf_img.dimx(), inf_img.dimy(),
                          inputImages.c_str());
    }
    int e = handleFileExistence(outputOrthogonalProjection, force, force);
    if(e != 0)
    {
        return e;
    }
    e = handleFileExistence(outputOrthogonalComplement, force, force);
    if(e != 0)
    {
        return e;
    }
    if(scalarProductInfo != "")
    {
        e = handleFileExistence(scalarProductInfo, force, force);
        if(e != 0)
        {
            return e;
        }
    }
    return 0;
}

template <typename T>
T scalarProduct(std::shared_ptr<io::BufferedFrame2D<T>> v1,
                std::shared_ptr<io::BufferedFrame2D<T>> v2)
{
    uint64_t elmcount = (uint64_t)v1->dimx() * (uint64_t)v1->dimy();
    T* p1 = v1->getDataPointer();
    T* p2 = v2->getDataPointer();
    return std::inner_product(p1, p1 + elmcount, p2, T(0));
}

template <typename T>
void a_equals_a_plus_c_times_b(std::shared_ptr<io::BufferedFrame2D<T>> a,
                               std::shared_ptr<io::BufferedFrame2D<T>> b,
                               T c)
{
    uint64_t elmcount = (uint64_t)a->dimx() * (uint64_t)a->dimy();
    T* fa = a->getDataPointer();
    T* fb = b->getDataPointer();
    for(uint64_t i = 0; i != elmcount; i++)
    {
        *(fa + i) += c * (*(fb + i));
    }
}

template <typename T>
void processFrame(int _FTPLID,
                  Args ARG,
                  uint32_t k,
                  std::shared_ptr<io::DenFrame2DCachedReader<T>>& basisReader,
                  std::shared_ptr<io::DenFrame2DReader<T>>& imgReader,
                  std::shared_ptr<io::AsyncFrame2DWritterI<T>>& orthogonalProjectionWritter,
                  std::shared_ptr<io::AsyncFrame2DWritterI<T>>& orthogonalComplementWritter,
                  std::shared_ptr<io::AsyncFrame2DWritterI<T>>& infoWritter)
{
    std::shared_ptr<io::BufferedFrame2D<T>> imgFrame = imgReader->readBufferedFrame(k);
    std::shared_ptr<io::BufferedFrame2D<T>> orthogonalProjectionFrame
        = std::make_shared<io::BufferedFrame2D<T>>(T(0), ARG.dimx, ARG.dimy);
    std::shared_ptr<io::BufferedFrame2D<T>> orthogonalComplementFrame
        = std::make_shared<io::BufferedFrame2D<T>>(*imgFrame);
    std::shared_ptr<io::BufferedFrame2D<T>> infoFrame
        = std::make_shared<io::BufferedFrame2D<T>>(T(0), ARG.basis_dimz, 1);
    std::shared_ptr<io::BufferedFrame2D<T>> v;
    for(uint64_t ind = 0; ind != ARG.frames.size(); ind++)
    {
        v = basisReader->readBufferedFrame(ARG.frames[ind]);
        T product = scalarProduct<T>(v, imgFrame);
        infoFrame->set(product, ARG.frames[ind], 0);
        a_equals_a_plus_c_times_b<T>(orthogonalProjectionFrame, v, product);
    }
    a_equals_a_plus_c_times_b<T>(orthogonalComplementFrame, orthogonalProjectionFrame, T(-1));
    orthogonalProjectionWritter->writeFrame(*orthogonalProjectionFrame, k);
    orthogonalComplementWritter->writeFrame(*orthogonalComplementFrame, k);
    if(infoWritter != nullptr)
    {
        infoWritter->writeFrame(*infoFrame, k);
    }
}

template <typename T>
void processFiles(Args ARG)
{
    ftpl::thread_pool* threadpool = nullptr;
    if(ARG.threads > 0)
    {
        threadpool = new ftpl::thread_pool(ARG.threads);
    }
    std::shared_ptr<io::DenFrame2DCachedReader<T>> basisReader
        = std::make_shared<io::DenFrame2DCachedReader<T>>(ARG.inputBasis, 0, ARG.frames.size());
    //= std::make_shared<io::DenFrame2DCachedReader<T>>(ARG.inputBasis);
    // std::shared_ptr<io::DenFrame2DReader<T>> basisReader
    //    = std::make_shared<io::DenFrame2DReader<T>>(ARG.inputBasis);
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> orthogonalProjectionWritter;
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> orthogonalComplementWritter;
    orthogonalProjectionWritter = std::make_shared<io::DenAsyncFrame2DWritter<T>>(
        ARG.outputOrthogonalProjection, ARG.dimx, ARG.dimy, ARG.dimz);
    orthogonalComplementWritter = std::make_shared<io::DenAsyncFrame2DWritter<T>>(
        ARG.outputOrthogonalComplement, ARG.dimx, ARG.dimy, ARG.dimz);
    std::shared_ptr<io::DenFrame2DReader<T>> imgReader
        = std::make_shared<io::DenFrame2DReader<T>>(ARG.inputImages);
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> INFOWritter = nullptr;
    if(ARG.scalarProductInfo != "")
    {
        INFOWritter = std::make_shared<io::DenAsyncFrame2DWritter<T>>(ARG.scalarProductInfo,
                                                                      ARG.basis_dimz, 1, ARG.dimz);
    }
    // This will set up the cache
    for(uint64_t ind = 0; ind != ARG.frames.size(); ind++)
    {
        LOGD << io::xprintf("Reading frame %d from %d", ARG.frames[ind], ARG.basis_dimz);
        basisReader->readFrame(ARG.frames[ind]);
    }
    const int dummy_FTPLID = 0;
    for(uint64_t k = 0; k != ARG.dimz; k++)
    {
        if(threadpool)
        {
            threadpool->push(processFrame<T>, ARG, k, basisReader, imgReader,
                             orthogonalProjectionWritter, orthogonalComplementWritter, INFOWritter);
        } else
        {

            processFrame(dummy_FTPLID, ARG, k, basisReader, imgReader, orthogonalProjectionWritter,
                         orthogonalComplementWritter, INFOWritter);
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
    const std::string prgInfo = "Produce direct sum of given projections onto the orthogonal "
                                "projection and orthogonal complement of the orthogonal basis.";
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
    io::DenFileInfo di(ARG.inputBasis);
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
