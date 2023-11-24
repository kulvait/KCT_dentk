#include "CLI/CLI.hpp" //Command line parser
#include "PLOG/PlogSetup.h"

// Internal libraries
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenSupportedType.hpp"
#include "PROG/Arguments.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/Program.hpp"
#include "PROG/parseArgs.h"
#include "littleEndianAlignment.h"
#include "rawop.h"

using namespace KCT;
using namespace KCT::util;

class Args : public ArgumentsForce
{
    template <typename T>
    T remove_if(T beg, T end);
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName){};

    std::string input_file;
    std::string output_file;
    std::string dim_str;
    std::vector<uint32_t> dim;
    std::string type;
    io::DenSupportedType dataType;
    uint64_t inputFileSize;
    uint64_t inputDataSize;
    bool inputIsDen = false;
    bool outputFileExists = false;
    uint64_t inputOffset = 0;
    int parseArguments(int argc, char* argv[]);
};

void Args::defineArguments()
{
    cliApp->add_option("dimspec", dim_str,
                       "Dim specification in the format e.g. [dimx0, dimx1, dimx2]. Can handle "
                       "multidimensional DEN headers.");
    cliApp->add_option("type", type, "Possible options are uint16_t, float or double")->required();
    cliApp
        ->add_option("input_raw_file", input_file,
                     "Raw file to add header or DEN file to change header.")
        ->required()
        ->check(CLI::ExistingFile);
    cliApp->add_option("output_den_file", output_file, "DEN output to write.")->required();
    addForceArgs();
}

//Remove whitespaces https://stackoverflow.com/a/83538
template <typename T>
T Args::remove_if(T beg, T end)
{
    T dest = beg;
    for(T itr = beg; itr != end; ++itr)
        if(!std::isspace(*itr))
            *(dest++) = *itr;
    return dest;
}

int Args::postParse()
{
    if(input_file == output_file)
    {
        LOGE << "Error: input and output files must differ!";
        return 1;
    }
    int existFlag = handleFileExistence(output_file, force, input_file);
    if(existFlag == 1)
    {
        return 1;
    } else if(existFlag == -1)
    {
        outputFileExists = true;
    }
    uint64_t expectedSize = 1;
    uint64_t expectedByteSize;
    //Remove whitespaces https://stackoverflow.com/a/83538
    dim_str.erase(remove_if(dim_str.begin(), dim_str.end()), dim_str.end());
    int dim_str_len = dim_str.size();
    if(dim_str_len < 2)
    {
        std::string err = io::xprintf("Invalid dim_str=%s.", dim_str.c_str());
        KCTERR(err);
    }
    if(dim_str.at(0) != '[' && dim_str.at(dim_str_len - 1) != ']')
    {
        std::string err = io::xprintf("Invalid dim_str=%s.", dim_str.c_str());
        KCTERR(err);
    }
    dim_str_len -= 2;
    dim_str = dim_str.substr(1, dim_str_len);
    //See https://stackoverflow.com/a/10861816
    stringstream ss(dim_str);
    vector<string> values;
    while(ss.good())
    {
        string substr;
        getline(ss, substr, ',');
        int val = stoi(substr);
        if(val < 0)
        {
            std::string err
                = io::xprintf("Negative value in provided dim_str=[%s].", dim_str.c_str());
            KCTERR(err);
        }
        dim.push_back((uint32_t)val);
    }
    for(int i = 0; i != dim.size(); i++)
    {
        expectedSize *= dim[i];
    }
    if(type == "uint16_t")
    {
        dataType = io::DenSupportedType::UINT16;
        expectedByteSize = 2 * expectedSize;
    } else if(type == "float")
    {
        dataType = io::DenSupportedType::FLOAT32;
        expectedByteSize = 4 * expectedSize;
    } else if(type == "double")
    {
        dataType = io::DenSupportedType::FLOAT64;
        expectedByteSize = 8 * expectedSize;
    } else
    {
        LOGE << io::xprintf("Error: unsupported data type %s!", type.c_str());
        return 1;
    }
    inputFileSize = io::getFileSize(input_file);
    io::DenFileInfo di = io::DenFileInfo(input_file, false);
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
    if(inputDataSize != expectedByteSize)
    {
        LOGE << io::xprintf(
            "Error: input file %s has size %lu but the size of the den file with specified "
            "dimensions [%s] and type %s is %lu",
            input_file.c_str(), inputFileSize, dim_str.c_str(), type.c_str(), expectedSize);
        return 1;
    }
    return 0;
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    const std::string prgInfo = "Insert DEN header to a raw file or DEN file with other alignment, "
                                "size in the header needs to be compatible with data size.";
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
    bool XMajorAlignment = true;
    io::DenFileInfo::createDenHeader(ARG.output_file, ARG.dataType, ARG.dim.size(),
                                     &ARG.dim.front(), XMajorAlignment);
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
        io::readBytesFrom(ARG.input_file, inputFilePosition, buffer, s);
        io::appendBytes(ARG.output_file, buffer, s);
        inputFilePosition += s;
    }
    delete[] buffer;
    PRG.endLog();
}
