#include "CLI/CLI.hpp" //Command line parser
#include "PLOG/PlogSetup.h"

// Internal libraries
#include "PROG/parseArgs.h"
#include "DEN/DenSupportedType.hpp"
#include "littleEndianAlignment.h"
#include "rawop.h"

using namespace CTL;

struct Args
{
    std::string inputRawFile;
    std::string outputDenFile;
    uint16_t dimx, dimy, dimz;
    std::string type;
    io::DenSupportedType dataType;
    uint64_t inputFileSize;
    bool force = false;
    int parseArguments(int argc, char* argv[]);
};

/**Argument parsing
 *
 */
int Args::parseArguments(int argc, char* argv[])
{
    CLI::App app{ "Insert DEN header to a raw file." };
    app.add_flag("-f,--force", force, "Overwrite outputDenFile if it exists.");
    app.add_option("dimx", dimx, "X dimension.")->required()->check(CLI::Range(1, 65535));
    app.add_option("dimy", dimy, "Y dimension.")->required()->check(CLI::Range(1, 65535));
    app.add_option("dimz", dimz, "Z dimension.")->required()->check(CLI::Range(1, 65535));
    app.add_option("type", type, "Possible options are uint16_t, float or double")->required();
    app.add_option("input_raw_file", inputRawFile, "Raw file.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("output_den_file", outputDenFile, "DEN output to write.")->required();
    try
    {
        app.parse(argc, argv);
        if(!force)
        {
            if(io::pathExists(outputDenFile))
            {
                std::string msg
                    = "Error: output file already exists, use --force to force overwrite.";
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
            dataType = io::DenSupportedType::uint16_t_;
            expectedSize *= 2;
        } else if(type == "float")
        {
            dataType = io::DenSupportedType::float_;
            expectedSize *= 4;
        } else if(type == "double")
        {
            dataType = io::DenSupportedType::double_;
            expectedSize *= 8;
        } else
        {
            LOGE << io::xprintf("Error: unsupported data type %s!", type);
            return 1;
        }
        inputFileSize = io::getFileSize(inputRawFile);
        if(inputFileSize != expectedSize)
        {
            LOGE << io::xprintf(
                "Error: input file %s has size %lu but the size of the den file with specified "
                "dimensions (dimx, dimy, dimz) = (%d, %d, %d) and type %s is %lu",

                inputRawFile.c_str(), inputFileSize, dimx, dimy, dimz, type.c_str(), expectedSize);
            return 1;
        }
    } catch(const CLI::ParseError& e)
    {
        int exitcode = app.exit(e);
        if(exitcode == 0) // Help message was printed
        {
            return 1;
        } else
        {
            LOGE << io::xprintf("There was perse error catched.\n %s", app.help().c_str());
            return -1;
        }
    }
    return 0;
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
    LOGI << io::xprintf("START %s", argv[0]);
    // Simply add header to a new file
    uint8_t header[6];
    util::putUint16(a.dimy, header);
    util::putUint16(a.dimx, &header[2]);
    util::putUint16(a.dimz, &header[4]);
    io::createEmptyFile(a.outputDenFile, 0, true);
    io::appendBytes(a.outputDenFile, header, 6);
    uint32_t bufferSize = 1024 * 1024;
    uint8_t* buffer = new uint8_t[bufferSize];
    uint64_t inputFilePosition = 0;
    while(inputFilePosition < a.inputFileSize)
    {
        uint32_t s;
        if(inputFilePosition + bufferSize < a.inputFileSize)
        {
            s = bufferSize;
        } else
        {
            s = uint32_t(a.inputFileSize - inputFilePosition);
        }
        io::readBytesFrom(a.inputRawFile, inputFilePosition, buffer, s);
        io::appendBytes(a.outputDenFile, buffer, s);
        inputFilePosition += s;
    }
    delete[] buffer;
    LOGI << io::xprintf("END %s", argv[0]);
}
