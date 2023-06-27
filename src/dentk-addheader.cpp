#include "CLI/CLI.hpp" //Command line parser
#include "PLOG/PlogSetup.h"

// Internal libraries
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenSupportedType.hpp"
#include "PROG/Arguments.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/parseArgs.h"
#include "PROG/Program.hpp"
#include "littleEndianAlignment.h"
#include "rawop.h"

using namespace KCT;
using namespace KCT::util;

class Args : public ArgumentsForce
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName){};

    std::string inputRawFile;
    std::string outputDenFile;
    uint16_t dimx, dimy, dimz;
    std::string type;
    io::DenSupportedType dataType;
    uint64_t inputFileSize;
    uint64_t inputDataSize;
    bool inputIsDen = false;
    uint64_t inputOffset = 0;
    int parseArguments(int argc, char* argv[]);
};

void Args::defineArguments()
{
    cliApp->add_option("dimx", dimx, "X dimension.")->required()->check(CLI::Range(1, 65535));
    cliApp->add_option("dimy", dimy, "Y dimension.")->required()->check(CLI::Range(1, 65535));
    cliApp->add_option("dimz", dimz, "Z dimension.")->required()->check(CLI::Range(1, 65535));
    cliApp->add_option("type", type, "Possible options are uint16_t, float or double")->required();
    cliApp->add_option("input_raw_file", inputRawFile, "Raw file.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("output_den_file", outputDenFile, "DEN output to write.")->required();
    addForceArgs();
}

int Args::postParse()
{
    cliApp->parse(argc, argv);
    if(!force)
    {
        if(io::pathExists(outputDenFile))
        {
            std::string msg = "Error: output file already exists, use --force to force overwrite.";
            LOGE << msg;
            return 1;
        }
    }
    if(inputRawFile == outputDenFile)
    {
        LOGE << "Error: input and output files must differ!";
        return 1;
    }
    uint64_t expectedSize = dimx * dimy * dimz;
    if(type == "uint16_t")
    {
        dataType = io::DenSupportedType::UINT16;
        expectedSize *= 2;
    } else if(type == "float")
    {
        dataType = io::DenSupportedType::FLOAT32;
        expectedSize *= 4;
    } else if(type == "double")
    {
        dataType = io::DenSupportedType::FLOAT64;
        expectedSize *= 8;
    } else
    {
        LOGE << io::xprintf("Error: unsupported data type %s!", type.c_str());
        return 1;
    }
    inputFileSize = io::getFileSize(inputRawFile);
    io::DenFileInfo di = io::DenFileInfo(inputRawFile, false);
    if(di.isValid())
    {
        inputIsDen = true;
        inputOffset = di.getOffset();
        inputDataSize = inputFileSize - inputOffset;
    } else
    {
        inputIsDen = false;
        inputOffset = 0;
        inputDataSize = inputFileSize;
    }
    if(inputDataSize != expectedSize)
    {
        LOGE << io::xprintf(
            "Error: input file %s has size %lu but the size of the den file with specified "
            "dimensions (dimx, dimy, dimz) = (%d, %d, %d) and type %s is %lu",
            inputRawFile.c_str(), inputFileSize, dimx, dimy, dimz, type.c_str(), expectedSize);
        return 1;
    }
    return 0;
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    const std::string prgInfo = "Insert DEN header to a raw file or DEN file with other alignment, size in the header needs to be compatible with data size.";
    Args ARG(argc, argv, prgInfo);
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog();
    // Simply add header to a new file
    std::array<uint32_t, 3> dim = { ARG.dimx, ARG.dimy, ARG.dimz };
    bool XMajorAlignment = true;
    io::DenFileInfo::createDenHeader(ARG.outputDenFile, ARG.dataType, 3, std::begin(dim), XMajorAlignment);
    uint32_t bufferSize = 1024 * 1024;
    uint8_t* buffer = new uint8_t[bufferSize];
    uint64_t inputFilePosition = ARG.inputOffset;
    while(inputFilePosition < ARG.inputFileSize)
    {
        uint32_t s;
        if(inputFilePosition + bufferSize < ARG.inputFileSize)
        {
            s = bufferSize;
        } else
        {
            s = uint32_t(ARG.inputFileSize - inputFilePosition);
        }
        io::readBytesFrom(ARG.inputRawFile, inputFilePosition, buffer, s);
        io::appendBytes(ARG.outputDenFile, buffer, s);
        inputFilePosition += s;
    }
    delete[] buffer;
    PRG.endLog();
}
