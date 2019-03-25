// Logging
#include "PLOG/PlogSetup.h"

// External libraries

// External libraries
#include "CLI/CLI.hpp"
#include "mkl.h"

// Internal libraries
#include "AsyncFrame2DWritterI.hpp"
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "Frame2DI.hpp"
#include "frameop.h"
#include "rawop.h"

using namespace CTL;

// class declarations
struct Args
{
    int parseArguments(int argc, char* argv[]);
    std::string input_file;
    bool vizualize = false, info = false;
    std::string output_file = "";
    bool force;
    uint16_t resampledGranularity = 0;
    bool orthogonalize = false;
    int shiftBasis = 0;
};

template <typename T>
void printFrameStatistics(const io::Frame2DI<T>& f)
{
    double min = (double)io::minFrameValue<T>(f);
    double max = (double)io::maxFrameValue<T>(f);
    double avg = io::meanFrameValue<T>(f);
    double l2norm = io::normFrame<T>(f, 2);
    std::cout << io::xprintf("\tMinimum, maximum, average values: %.3f, %0.3f, %0.3f.\n", min, max,
                             avg);
    std::cout << io::xprintf("\tEuclidean 2-norm of the frame: %E.\n", l2norm);
    int nonFiniteCount = io::sumNonfiniteValues<T>(f);
    if(nonFiniteCount == 0)
    {
        std::cout << io::xprintf("\tNo NAN or not finite number.\n\n");
    } else
    {
        std::cout << io::xprintf("\tThere is %d non finite numbers. \tFrom that %d NAN.\n\n",
                                 io::sumNonfiniteValues<T>(f), io::sumNanValues<T>(f));
    }
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
    io::DenFileInfo di(a.input_file);
    int dimx = di.dimx();
    int dimy = di.dimy();
    int dimz = di.dimz();
    // int elementSize = di.elementByteSize();
    io::DenSupportedType t = di.getDataType();
    std::string elm = io::DenSupportedTypeToString(t);
    std::cout << io::xprintf(
        "The file %s of type %s has dimensions (x,y,z)=(cols,rows,slices)=(%d, "
        "%d, %d), each cell has x*y=%d pixels.\n",
        a.input_file.c_str(), elm.c_str(), dimx, dimy, dimz, dimx * dimy);
    switch(t)
    {
    case io::DenSupportedType::uint16_t_:
    {
        std::cout << io::xprintf("Not implemented yet.\n");
        break;
    }
    case io::DenSupportedType::float_:
    {
        uint64_t totalSize = dimx * dimy * dimz;
        if(totalSize * sizeof(float) > std::numeric_limits<uint32_t>::max())
        {
            io::throwerr("Uint32 is exceeded by the byte size of matrix that is %lu.",
                         totalSize * sizeof(float));
        }
        float* A = new float[totalSize]; // First going to read my matrix
        io::readBytesFrom(a.input_file, 6, (uint8_t*)A,
                          totalSize * sizeof(float)); // Quite hard to process
        float *U, *S, *V;
        U = nullptr;
        V = nullptr;
        S = nullptr;
        float* superb;
        char jobu = 'N', jobvt = 'N';
        int ldu = 0, ldvt = 0;
        bool storeU = !a.U_file.empty();
        bool storeV = !a.V_file.empty();
        uint32_t numvectors = std::min(dimx * dimy, dimz);
        S = new float[numvectors];
        superb = new float[numvectors - 1];
        ldu = numvectors;
        ldvt = dimx * dimy;
        if(storeU)
        {
            U = new float[dimz * numvectors];
            jobu = 'S';
        }
        if(storeV)
        {
            V = new float[dimx * dimy * numvectors];
            jobvt = 'S';
        }
        LOGD << io::xprintf("Calling LAPACKE_sgesvd with dimx*dimy=%d dimz=%d", dimx * dimy, dimz);
        int inf = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, jobu, jobvt, dimz, dimx * dimy, A, dimx * dimy,
                                 S, U, ldu, V, ldvt, superb);
        LOGI << "After LAPACKE_sgesvd.";
        if(inf == 0)
        {
            int numreturned = std::min(numvectors, a.max_s_vals);
            if(a.explain_procent < 100.0)
            {
                float explain = a.explain_procent / 100.0;
                float sinSum = 0.0;
                for(std::size_t i = 0; i != numvectors; i++)
                {
                    sinSum += S[i];
                }
                float sum = 0.0;
                int valCount = 0;
                for(std::size_t i = 0; i != numvectors; i++)
                {
                    sum += S[i];
                    valCount++;
                    if(sum / sinSum >= explain)
                    {
                        break;
                    }
                }
                numreturned = std::min(numreturned, valCount);
            }

            std::shared_ptr<io::AsyncFrame2DWritterI<float>> Sw
                = std::make_shared<io::DenAsyncFrame2DWritter<float>>(a.S_file, 1, 1, numreturned);
            for(int i = 0; i != numreturned; i++)
            {
                io::BufferedFrame2D<float> s(S[i], 1, 1);
                Sw->writeFrame(s, i);
            }
            if(storeU)
            {
                std::shared_ptr<io::AsyncFrame2DWritterI<float>> Uw
                    = std::make_shared<io::DenAsyncFrame2DWritter<float>>(a.U_file, dimz, 1,
                                                                          numreturned);
                for(int i = 0; i != numreturned; i++)
                {
                    io::BufferedFrame2D<float> s(float(0), dimz, 1);
                    for(int j = 0; j != dimz; j++)
                    {
                        s.set(U[numvectors * j + i], j, 0);
                    }
                    Uw->writeFrame(s, i);
                }
            }
            if(storeV)
            {
                std::shared_ptr<io::AsyncFrame2DWritterI<float>> Vw
                    = std::make_shared<io::DenAsyncFrame2DWritter<float>>(a.V_file, dimx * dimy, 1,
                                                                          numreturned);
                for(int i = 0; i != numreturned; i++)
                {
                    io::BufferedFrame2D<float> s(&V[dimx * dimy * i], 1, dimx * dimy);
                    Vw->writeFrame(s, i);
                }
            }

        } else
        {
            io::throwerr("There was a convergence problem catched inf=%d.", inf);
        }
        break;
    }
    case io::DenSupportedType::double_:
    {
        std::cout << io::xprintf("Not implemented yet.\n");
        break;
    }
    default:
        std::string errMsg
            = io::xprintf("Unsupported data type %s.", io::DenSupportedTypeToString(t));
        LOGE << errMsg;
        throw std::runtime_error(errMsg);
    }
}

int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Engineered bases processing. These bases are stored in a DEN file, where x is "
                  "granularity y is 1 and z is number of basis functions." };
    app.add_option("input_den_file", input_file, "File in a DEN format to process.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_flag("-v,--vizualize", vizualize, "Vizualize the basis.");
    app.add_flag("-i,--info", info, "Print information about the basis.");
    CLI::Option* _output = app.add_option("-o", output_file, "File to output.");
    CLI::Option* _force = app.add_flag("-f,--force", force, "Force overwrite output file.");
    _force->needs(_output);
    CLI::Option* _resample = app.add_option("--resample", resampledGranularity, "Resample discretization to respect new granularity size.");
    _resample->needs(_output);
    CLI::Option* _orthogonalize = app.add_flag("--orthogonalize", orthogonalize, "Print information about the basis.");
    _orthogonalize->needs(_output);
    CLI::Option* _shift = app.add_option("--shift", shift, "Shift basis and make it continuous at boundaries.");
    _shift->needs(_output);
    _shift->excludes(_resample);
    try
    {
        app.parse(argc, argv);
        if(!forceOverwrite)
        {
            if(io::fileExists(S_file))
            {
                std::string msg = io::xprintf(
                    "Error: output file S_file %s already exists, use -f to force overwrite.",
                    S_file.c_str());
                LOGE << msg;
                return 1;
            }
            if(!U_file.empty() && io::fileExists(U_file))
            {
                std::string msg = io::xprintf(
                    "Error: output file U_file %s already exists, use -f to force overwrite.",
                    U_file.c_str());
                LOGE << msg;
                return 1;
            }
            if(!V_file.empty() && io::fileExists(V_file))
            {
                std::string msg = io::xprintf(
                    "Error: output file V_file %s already exists, use -f to force overwrite.",
                    V_file.c_str());
                LOGE << msg;
                return 1;
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
            LOGE << "Parse error catched";
            // Negative value should be returned
            return -1;
        }
    }

    return 0;
}
