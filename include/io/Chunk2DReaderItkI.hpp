#ifndef CHUNK2DREADERITKI_HPP
#define CHUNK2DREADERITKI_HPP

// External libraries
#include "itkImageFileReader.h"
#include "itkImageIOBase.h"
#include "itkRawImageIO.h"

// Internal libraries
#include "io/Chunk2DReaderI.hpp"

namespace CTL::io {
/**
 *Interface for reading frames from sources with a posibility to interpret them as ITK images.
 *
 *The implementation might be "slim implementation" that access underlying source each time it is
 *called. The implementation might be also "fat implementation" that holds in memory structure that
 *source each call.
 */
template <typename T>
class Chunk2DReaderItkI : virtual public Chunk2DReaderI<T>
// I don't want from implementer to implement all the behavior of ProjectionReaderI when its
// implemented in some base class which is itself inherited
{
public:
    virtual typename itk::Image<T, 2>::Pointer readChunk2DAsItkImage(int i) = 0;
    /*Returns i-th frame in the source as itk::Image<T,2>.*/
};
} // namespace CTL::io
#endif // CHUNK2DREADERITKI_HPP
