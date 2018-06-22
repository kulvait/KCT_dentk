#ifndef CHUNK2DREADERI_HPP
#define CHUNK2DREADERI_HPP

//External libraries

//Internal libraries
#include "io/Chunk2DReadI.hpp"

namespace CTL::io {
/**
*Interface for reading Chunk2D objects.
*
*The implementation might be "slim implementation" that access underlying source each time it is called.
*The implementation might be also "fat implementation" that holds in memory structure that source each call.
*/
template <typename T>
class Chunk2DReaderI {
public:
    virtual std::shared_ptr<io::Chunk2DReadI<T>> readSlice(int i) = 0;
    /*Returns i-th projection slice in the source.*/
    virtual unsigned int dimx() const = 0;
    /**Returns x dimension.*/
    virtual unsigned int dimy() const = 0;
    /**Returns y dimension.*/
    virtual unsigned int count() const = 0;
    /**Returns number of slices in the source, slices are indexed 0 <= i < count().*/
};
}
#endif //PROJECTIONREADERI_HPP
