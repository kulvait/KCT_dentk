#ifndef ASYNCIMAGEWRITTERITKI_HPP
#define ASYNCIMAGEWRITTERITKI_HPP

// External libraries
#include "itkImage.h"
// Internal libraries

namespace CTL::io {
/**
Interface for writing images. It is not necessery to write matrices along them.
*/
template <typename T>
class AsyncImageWritterItkI
{
public:
    virtual void writeSlice(typename itk::Image<T, 2>::Pointer s, int i) = 0;
    /**Writes i-th slice to the source.*/

    virtual unsigned int dimx() const = 0;
    /**Returns x dimension.*/

    virtual unsigned int dimy() const = 0;
    /**Returns y dimension.*/

    virtual unsigned int dimz() const = 0;
    /**Returns y dimension.*/
};
} // namespace CTL::io
#endif // ASYNCIMAGEWRITTERITK_HPP
