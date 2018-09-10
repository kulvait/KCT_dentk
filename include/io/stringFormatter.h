#ifndef STRINGFORMATTER_H
#define STRINGFORMATTER_H
// Logging on the top
#include <plog/Log.h>
#include <utils/PlogSetup.h>

#include <cstdarg> //va_start, va_end
#include <string>

namespace CTL::io {
std::string xprintf(const std::string fmt_str, ...);
} // namespace CTL::io
#endif // STRINGFORMATTER_H
