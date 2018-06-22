#include "utils/convertEndians.h"

namespace CTL::util {
//I decided to put data into the char* array instead of uint8_t* for the systems where byte size is not 8bit, where reading should work bite wise and alignment considerations raised by https://stackoverflow.com/questions/39668561/allocate-n-bytes-by-new-and-fill-it-with-any-type?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
uint8_t nextUint8(uint8_t* buffer)
{
    uint8_t num;
    num = buffer[0];
    return (num);
}

int16_t nextInt8(uint8_t* buffer)
{
    int8_t num;
    return (int8_t)nextUint8(buffer);
}

uint16_t nextUint16(uint8_t* buffer)
{
    uint16_t num;
    num = buffer[0] | buffer[1] << 8;
    return (num);
}

int16_t nextInt16(uint8_t* buffer)
{
    int16_t num;
    return (int16_t)nextUint16(buffer);
}

uint32_t nextUint32(uint8_t* buffer)
{
    uint32_t num;
    num = buffer[0] | buffer[1] << 8 | buffer[2] << 16 | buffer[3] << 24;
    //num = buffer[0] | buffer[1] << 8 | buffer[2] << 16 | 0x7f << 24;
    //num = 0xff | 0xff << 8 | 0xff << 16 | 0x7f << 24;
    //num = 2147483647;
    return (num);
}

uint64_t nextUint64(uint8_t* buffer)
{
    uint64_t num;
    num = (uint64_t)buffer[0] | (uint64_t)buffer[1] << 8 | (uint64_t)buffer[2] << 16 | (uint64_t)buffer[3] << 24 | (uint64_t)buffer[4] << 32 | (uint64_t)buffer[5] << 40 | (uint64_t)buffer[6] << 48 | (uint64_t)buffer[7] << 56;
    return (num);
}

int32_t nextInt32(uint8_t* buffer)
{
    int32_t num;
    return (int32_t)nextUint32(buffer);
}

int64_t nextInt64(uint8_t* buffer)
{
    return (int64_t)nextUint64(buffer);
}

/*This function might not preserve endianness.
*
*/
float nextFloat(uint8_t* buffer)
{
    uint32_t out_int;
    out_int = nextUint32(buffer);
    float out;
    std::memcpy(&out, &out_int, 4);
    //simpler but do not preserve endianness memcpy(&out, buffer, 4);
    return out;
}

/*This function might not preserve endianness.
*
*/
double nextDouble(uint8_t* buffer)
{
    uint64_t out_int;
    out_int = nextUint64(buffer);
    double out;
    std::memcpy(&out, &out_int, 8);
    //simpler but do not preserve endianness memcpy(&out, buffer, 8);
    return out;
}

uint8_t putUint8(uint8_t val, uint8_t* buffer)
{
    buffer[0] = val;
}

int16_t putInt8(int16_t val, uint8_t* buffer)
{
    //First convert to unsigned type and then write into the buffer
    putUint8((uint8_t)val, buffer);
}

uint16_t putUint16(uint16_t val, uint8_t* buffer)
{
    buffer[0] = val & 0xFF;
    buffer[1] = (val & 0xFF00) >> 8;
}

int16_t putInt16(int16_t val, uint8_t* buffer)
{
    putUint16((uint16_t)val, buffer);
}

uint32_t putUint32(uint32_t val, uint8_t* buffer)
{
    buffer[0] = val & 0xFF;
    buffer[1] = (val & 0xFF00) >> 8;
    buffer[2] = (val & 0xFF0000) >> 16;
    buffer[3] = (val & 0xFF000000) >> 24;
}

int32_t putInt32(int32_t val, uint8_t* buffer)
{
    putUint32((uint32_t)val, buffer);
}

uint64_t putUint64(uint64_t val, uint8_t* buffer)
{
    buffer[0] = val & 0xFF;
    buffer[1] = (val & 0xFF00) >> 8;
    buffer[2] = (val & 0xFF0000) >> 16;
    buffer[3] = (val & 0xFF000000) >> 24;
    buffer[4] = (val & 0xFF00000000) >> 32;
    buffer[5] = (val & 0xFF0000000000) >> 40;
    buffer[6] = (val & 0xFF000000000000) >> 48;
    buffer[7] = (val & 0xFF00000000000000) >> 56;
}
int32_t putInt64(int64_t val, uint8_t* buffer)
{
    putUint64((uint64_t)val, buffer);
}

float putFloat(float val, uint8_t* buffer)
{
    uint32_t tmp;
    std::memcpy(&tmp, &val, 4);
    putUint32(tmp, buffer);
}

double putDouble(double val, uint8_t* buffer)
{
    uint64_t tmp;
    std::memcpy(&tmp, &val, 8);
    putUint64(tmp, buffer);
}

} //namespace CTL::util
