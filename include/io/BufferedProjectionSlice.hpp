#ifndef BUFFEREDPROJECTIONSLICE_HPP
#define BUFFEREDPROJECTIONSLICE_HPP

// Internal
#include "io/Chunk2DReadI.hpp"

namespace CTL::io {
/// Implementation of ProjectionSlice that takes naked array of T elements and operates on them
/**
Interface for reading projection slice. T is a type of the data used, might be float, double or
uint16_t.
*/
template <typename T>
class BufferedProjectionSlice : public Chunk2DReadI<T>
{
public:
    BufferedProjectionSlice(T* slice, int sizex, int sizey)
    {
        this->slice = new T[sizex * sizey];
        memcpy(this->slice, slice, sizex * sizey * sizeof(T));
        this->sizex = sizex;
        this->sizey = sizey;
    }

    BufferedProjectionSlice(const BufferedProjectionSlice& b)
        : BufferedProjectionSlice(b.slice, b.sizex, b.sizey)
    {
    } // copy constructor

    ~BufferedProjectionSlice() { delete[] slice; } // destructor

    BufferedProjectionSlice& operator=(const BufferedProjectionSlice& b)
    {
        T* tmp;
        this->sizex = b.sizex;
        this->sizey = b.sizey;
        tmp = new T[sizex * sizey];
        memcpy(tmp, b.slice, sizex * sizey * sizeof(T));
        delete[] this->slice;
        this->slice = tmp;
        return *this;

    } // copy assignment, tmp is to solve situation when assigning to itself

    T get(unsigned int x, unsigned int y) const override { return slice[y * sizex + x]; }

    unsigned int dimx() const override { return sizex; }

    unsigned int dimy() const override { return sizey; }

    std::shared_ptr<io::Chunk2DReadI<T>> transpose()
    {
        T* tmp;
        int sizex_tmp = sizey;
        int sizey_tmp = sizex;
        tmp = new T[sizex * sizey];
        for(int x = 0; x != sizex_tmp; x++)
            for(int y = 0; y != sizey_tmp; y++)
            {
                tmp[y * sizex_tmp + x] = slice[x * sizex + y];
            }
        std::shared_ptr<io::Chunk2DReadI<T>> bps
            = std::make_shared<BufferedProjectionSlice<T>>(tmp, sizex_tmp, sizey_tmp);
        delete[] tmp;
        return bps;
    }

    /**
     * Function to get access to the data array. Very dangerous operation.
     * This array is destroyed after the slice is destroyed.
     */
    void* getDataPointer() { return (void*)slice; }

private:
    T* slice;
    int sizex, sizey;
};
} // namespace CTL::io
#endif // BUFFEREDPROJECTIONSLICE_HPP
