#include "io/DenFileInfo.hpp"

namespace CTL::io {

DenFileInfo::DenFileInfo(std::string fileName)
    : fileName(fileName)
{
}

int DenFileInfo::getNumRows()
{
    uint8_t buffer[2];
    readBytesFrom(this->fileName, 0, buffer, 2);
    return (util::nextUint16(buffer));
}
/**Y dimension*/

int DenFileInfo::getNumCols()
{
    uint8_t buffer[2];
    readBytesFrom(this->fileName, 2, buffer, 2);
    return (util::nextUint16(buffer));
}
/**X dimension*/

int DenFileInfo::getNumSlices()
{
    uint8_t buffer[2];
    readBytesFrom(this->fileName, 4, buffer, 2);
    return (util::nextUint16(buffer));
}
/**Z dimension*/

long DenFileInfo::getSize()
{
    std::ifstream ifs(this->fileName, std::ifstream::ate | std::ifstream::binary);
    long size = ifs.tellg();
    ifs.close();
    return size;
}

long DenFileInfo::getNumPixels()
{
    return (long)this->getNumCols() * (long)this->getNumRows() * (long)this->getNumSlices();
}

DenSupportedType DenFileInfo::getDataType()
{
    int elementByteSize = this->elementByteSize();
    switch(elementByteSize)
    {
    case 2:
        return DenSupportedType::uint16_t_;
    case 4:
        return DenSupportedType::float_;
    case 8:
        return DenSupportedType::double_;
    default:
        std::stringstream errMsg;
        errMsg << "File " << this->fileName
               << " is not valid DEN file because it has datatype of the length " << elementByteSize
               << ".";
        LOGE << errMsg.str();
        throw std::runtime_error(errMsg.str());
    }
}

int DenFileInfo::elementByteSize()
{
    long dataSize = this->getSize() - 6;
    long numPixels = this->getNumPixels();
    if(dataSize == 0)
    {
        if(numPixels == 0)
        {
            return 0;
        } else
        {
            std::stringstream errMsg;
            errMsg << "File " << this->fileName << " is not valid DEN file.";
            LOGE << errMsg.str();
            throw std::runtime_error(errMsg.str());
        }
    }
    if(dataSize < 0 || numPixels <= 0)
    {
        std::stringstream errMsg;
        errMsg << "File " << this->fileName
               << " is not valid DEN file because it is shorter than 6 bytes or number of pixels "
                  "is nonpositive.";
        LOGE << errMsg.str();
        throw std::runtime_error(errMsg.str());
    }
    if(dataSize % numPixels != 0)
    {
        std::stringstream errMsg;
        errMsg
            << "File " << this->fileName
            << " is not valid DEN file because its data are not aligned with a pixels represented.";
        LOGE << errMsg.str();
        throw std::runtime_error(errMsg.str());
    }
    switch(dataSize / numPixels)
    {
    case 2:
        return 2;
    case 4:
        return 4;
    case 8:
        return 8;
    default:
        std::stringstream errMsg;
        errMsg << "File " << this->fileName
               << " is not valid DEN file because it has datatype of the length "
               << dataSize / numPixels << ".";
        LOGE << errMsg.str();
        throw std::runtime_error(errMsg.str());
    }
}

} // namespace CTL::io
