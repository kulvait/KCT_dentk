#ifndef DENSUPPORTEDTYPE_HPP
#define DENSUPPORTEDTYPE_HPP

#include <string>

namespace CTL::io {

enum DenSupportedType { uint16_t_,
    float_,
    double_ };

inline std::string DenSupportedTypeToString(DenSupportedType dataType)
{
    switch (dataType) {
    case uint16_t_:
        return "uint16_t";
    case float_:
        return "float";
    case double_:
        return "double";
    default:
        return "[Unknown DenSupportedType]";
    }
}

} //namespace CTL::io
#endif //DENFILEINFO_HPP
