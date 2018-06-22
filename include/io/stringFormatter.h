#ifndef STRINGFORMATTER_H
#define STRINGFORMATTER_H
#include <plog/Log.h>
#include <utils/PlogSetup.h>

#include <cstdarg>
#include <string>

namespace CTL::io {
std::string xprintf(const std::string fmt_str, ...);
} //namespace CTL::io
#endif //STRINGFORMATTER_H
