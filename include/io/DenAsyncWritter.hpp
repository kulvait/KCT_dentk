#ifndef DENASYNCWRITTER_HPP
#define DENASYNCWRITTER_HPP

// External libraries
#include <string>

// Internal libraries
#include "io/AsyncImageWritterI.hpp"
#include "io/rawop.h"
#include "utils/convertEndians.h"

namespace CTL::io {
/**
Interface for writing images. It is not necessery to write matrices along them.
*/
template <typename T>
class DenAsyncWritter : public AsyncImageWritterI<T>
{
private:
    std::string projectionsFile;
    int sizex, sizey, sizez;

public:
    DenAsyncWritter(std::string projectionsFile, int dimx, int dimy, int dimz);
    /*Need to specify dimension first*/
    void writeSlice(const Chunk2DReadI<T>& s, int i) override;
    /**Writes i-th slice to the source.*/

    virtual unsigned int dimx() const override;
    /**Returns x dimension.*/

    virtual unsigned int dimy() const override;
    /**Returns y dimension.*/

    virtual unsigned int dimz() const override;
    /**Returns z dimension.*/

    std::string getFileName() const;
    /**Returns file name.**/
};

template <typename T>
DenAsyncWritter<T>::DenAsyncWritter(std::string projectionsFile, int dimx, int dimy, int dimz)
{
    if(dimx < 1 || dimy < 1 || dimz < 1)
    {
        std::string msg = io::xprintf("One of the dimensions is nonpositive x=%d, y=%d, z=%d.",
                                      dimx, dimy, dimz);
        LOGE << msg;
        throw std::runtime_error(msg);
    }
    this->projectionsFile = projectionsFile;
    this->sizex = dimx;
    this->sizey = dimy;
    this->sizez = dimz;
    long elementByteSize = sizeof(T);
    long totalFileSize = 6 + elementByteSize * dimx * dimy * dimz;
    if(io::fileExists(projectionsFile))
    {
        long fileSize = io::getFileSize(projectionsFile);
        if(fileSize != totalFileSize)
        {
            io::createEmptyFile(projectionsFile, totalFileSize, true);
            LOGD << io::xprintf(
                "Just overwritten the file %s with empty file of the size %ld bytes.",
                projectionsFile.c_str(), totalFileSize);
        }
        LOGD << io::xprintf(
            "Will be working on existing file %s ewith %ld bytes. Overwritten old file.",
            projectionsFile.c_str(), totalFileSize);
    } else
    {
        io::createEmptyFile(projectionsFile, totalFileSize, true);
        LOGD << io::xprintf("Just created a file %s with size %ld bytes.", projectionsFile.c_str(),
                            totalFileSize);
    }
    uint8_t buffer[6];
    util::putUint16((uint16_t)dimy, &buffer[0]);
    util::putUint16((uint16_t)dimx, &buffer[2]);
    util::putUint16((uint16_t)dimz, &buffer[4]);
    io::writeFirstBytes(projectionsFile, buffer, 6);
}

template <typename T>
std::string DenAsyncWritter<T>::getFileName() const
{
    return this->projectionsFile;
}

template <typename T>
unsigned int DenAsyncWritter<T>::dimx() const
{
    return sizex;
}

template <typename T>
unsigned int DenAsyncWritter<T>::dimy() const
{
    return sizey;
}

template <typename T>
unsigned int DenAsyncWritter<T>::dimz() const
{
    return sizez;
}

template <typename T>
void DenAsyncWritter<T>::writeSlice(const Chunk2DReadI<T>& s, int i)
{
    uint8_t buffer[sizeof(T) * this->dimx() * this->dimy()];
    for(int j = 0; j != this->dimy(); j++)
    {
        for(int k = 0; k != this->dimx(); k++)
        {
            util::setNextElement<T>(s(k, j), &buffer[(j * this->dimx() + k) * sizeof(T)]);
        }
    }
    uint64_t position = (uint64_t)6 + ((uint64_t)sizeof(T)) * i * this->dimx() * this->dimy();
    io::writeBytesFrom(projectionsFile, position, buffer, sizeof(T) * this->dimx() * this->dimy());
    return;
}

} // namespace CTL::io
#endif // DENASYNCWRITTER_HPP
