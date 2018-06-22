//Logging on the top
#include "io/rawop.h"

namespace CTL::io {

void readFirstBytes(std::string fileName, uint8_t* buffer, int numBytes)
{
    if (CHAR_BIT != 8) {
        std::stringstream errMsg;
        errMsg << "Can not use this platform since CHAR_BIT size is not 8, namely it is " << CHAR_BIT << ".";
        LOGE << errMsg.str();
        throw std::runtime_error(errMsg.str());
    }
    std::ifstream file(fileName, std::ifstream::binary | std::ios::in); //binary for input
    if (!file.is_open()) // cannot open file
    {
        std::stringstream errMsg;
        errMsg << "Can not open file " << fileName << ".";
        LOGE << errMsg.str();
        throw std::runtime_error(errMsg.str());
    }

    file.read((char*)buffer, numBytes);
    auto num = file.gcount();
    file.close();
    if (num != numBytes) {
        std::stringstream errMsg;
        errMsg << "Can not read first " << numBytes << "bytes from the file " << fileName << ".";
        LOGE << errMsg.str();
        throw std::runtime_error(errMsg.str());
    }
}

void readBytesFrom(std::string fileName, uint64_t fromPosition, uint8_t* buffer, int numBytes)
{
    if (CHAR_BIT != 8) {
        std::stringstream errMsg;
        errMsg << "Can not use this platform since CHAR_BIT size is not 8, namely it is " << CHAR_BIT << ".";
        LOGE << errMsg.str();
        throw std::runtime_error(errMsg.str());
    }
    std::ifstream file(fileName, std::ifstream::binary | std::ios::in); //binary for input
    if (!file.is_open()) // cannot open file
    {
        std::stringstream errMsg;
        errMsg << "Can not open file " << fileName << ".";
        LOGE << errMsg.str();
        throw std::runtime_error(errMsg.str());
    }
    file.seekg(fromPosition);
    file.read((char*)buffer, numBytes);
    auto num = file.gcount();
    file.close();
    if (num != numBytes) {
        std::stringstream errMsg;
        errMsg << "Can not read first " << numBytes << "bytes from the file " << fileName << ".";
        LOGE << errMsg.str();
        throw std::runtime_error(errMsg.str());
    }
}

void writeFirstBytes(std::string fileName, uint8_t* buffer, int numBytes)
{
    if (CHAR_BIT != 8) {
        std::stringstream errMsg;
        errMsg << "Can not use this platform since CHAR_BIT size is not 8, namely it is " << CHAR_BIT << ".";
        LOGE << errMsg.str();
        throw std::runtime_error(errMsg.str());
    }
    std::ofstream file(fileName, std::ios::binary | std::ios::out | std::ios::in); //Open binary, for output, for input
    if (!file.is_open()) // cannot open file
    {
        std::stringstream errMsg;
        errMsg << "Can not open file " << fileName << ".";
        LOGE << errMsg.str();
        throw std::runtime_error(errMsg.str());
    }

    file.write((char*)buffer, numBytes);
    auto num = file.tellp();
    file.close();
    if (num != numBytes) {
        std::stringstream errMsg;
        errMsg << "Can not read first " << numBytes << "bytes from the file " << fileName << ".";
        LOGE << errMsg.str();
        throw std::runtime_error(errMsg.str());
    }
}

void writeBytesFrom(std::string fileName, uint64_t fromPosition, uint8_t* buffer, int numBytes)
{
    if (CHAR_BIT != 8) {
        std::stringstream errMsg;
        errMsg << "Can not use this platform since CHAR_BIT size is not 8, namely it is " << CHAR_BIT << ".";
        LOGE << errMsg.str();
        throw std::runtime_error(errMsg.str());
    }
    std::ofstream file(fileName, std::ios::binary | std::ios::out | std::ios::in); //Open binary, for output, for input
    if (!file.is_open()) // cannot open file
    {
        std::stringstream errMsg;
        errMsg << "Can not open file " << fileName << ".";
        LOGE << errMsg.str();
        throw std::runtime_error(errMsg.str());
    }
    file.seekp(fromPosition); //put pointer
    std::streampos cur = file.tellp();
    file.write((char*)buffer, numBytes);
    auto pos = file.tellp();
    auto num = pos - cur;
    file.close();
    if (num != numBytes) {
        std::stringstream errMsg;
        errMsg << num << " bytes written from number of bytess that should be written " << numBytes << " to " << fileName << ".";
        LOGE << errMsg.str();
        throw std::runtime_error(errMsg.str());
    }
}

inline bool fileExists(std::string fileName)
{
    return (access(fileName.c_str(), F_OK) != -1);
}

void createEmptyFile(std::string fileName, int numBytes, bool overwrite)
{
    if (!fileExists(fileName) || overwrite) {
        std::ofstream ofs(fileName, std::ios::binary | std::ios::out | std::ios::trunc); //Open binary, for output, truncate when opening
        if (numBytes > 0) {
            ofs.seekp(numBytes - 1);
            ofs.write("", 1);
        }
        ofs.close();
    } else {
        std::stringstream errMsg;
        errMsg << "File " << fileName << " already exists and overwrite is no set.";
        LOGE << errMsg.str();
        throw std::runtime_error(errMsg.str());
    }
}

} //namespace CTL::io
