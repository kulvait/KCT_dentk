#ifndef DENFILEINFO_HPP
#define DENFILEINFO_HPP
//Logging
#include <utils/PlogSetup.h>

//Standard libraries
#include <string>

//Internal libraries
#include "io/DenSupportedType.hpp"
#include "io/rawop.h" //To get number of rows...

namespace CTL::io {

class DenFileInfo {
public:
    DenFileInfo(std::string fileName);
    int getNumRows();
    int getNumCols();
    int getNumSlices();
    long getSize();
    long getNumPixels();
    DenSupportedType getDataType();
    int elementByteSize();

private:
    std::string fileName;
};

} //namespace CTL::io
#endif //DENFILEINFO_HPP
