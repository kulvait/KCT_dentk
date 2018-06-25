#ifndef RAWOP_H
#define RAWOP_H
// Raw reading of the part file into the uint8_t buffer

#include <plog/Log.h>

#include <fstream>
#include <iostream>
#include <limits.h>
#include <memory> // For std::unique_ptr
#include <sstream>
#include <stdarg.h> // For va_start, etc.
#include <string>
#include <sys/stat.h>

#include "utils/convertEndians.h"

namespace CTL::io {
void readFirstBytes(std::string fileName, uint8_t* buffer, int numBytes);
void readBytesFrom(std::string fileName, uint64_t fromPosition, uint8_t* buffer, int numBytes);
void writeFirstBytes(std::string fileName, uint8_t* buffer, int numBytes);
void writeBytesFrom(std::string fileName, uint64_t fromPosition, uint8_t* buffer, int numBytes);
inline bool fileExists(std::string fileName);
void createEmptyFile(std::string fileName, int numBytes, bool overwrite);
} // namespace CTL::io
#endif // RAWOP_H
