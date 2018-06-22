#ifndef ASYNCIMAGEWRITTERI_HPP
#define ASYNCIMAGEWRITTERI_HPP

//Internal libraries
#include "io/Chunk2DReadI.hpp"

namespace CTL::io {
/**
Interface for writing images. It is not necessery to write matrices along them.
*/
template <typename T>
class AsyncImageWritterI {
public:
    virtual void writeSlice(const Chunk2DReadI<T>& s, int i) = 0;
    /**Writes i-th slice to the source.*/

    virtual unsigned int dimx() const = 0;
    /**Returns x dimension.*/

    virtual unsigned int dimy() const = 0;
    /**Returns y dimension.*/

    virtual unsigned int dimz() const = 0;
    /**Returns y dimension.*/
};
} //namespace CTL::io
#endif //ASYNCIMAGEWRITTERI_HPP
