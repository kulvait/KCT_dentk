#ifndef BUFFEREDPROJECTIONSLICE_HPP
#define BUFFEREDPROJECTIONSLICE_HPP

//Internal
#include "io/Chunk2DReadI.hpp"

namespace CTL::io {
///Implementation of ProjectionSlice that takes naked array of T elements and operates on them
/**
Interface for reading projection slice. T is a type of the data used, might be float, double or uint16_t.
*/
template <typename T>
class BufferedProjectionSlice : public Chunk2DReadI<T> {
public:
    BufferedProjectionSlice(T* slice, int sizex, int sizey)
    {
        this->slice = new T[sizex * sizey];
        memcpy(this->slice, slice, sizex * sizey * sizeof(T));
        this->sizex = sizex;
        this->sizey = sizey;
    }

    BufferedProjectionSlice(const BufferedProjectionSlice& b)
        : BufferedProjectionSlice(b->slice, b->sizex, b->sizey)
    {
    } //copy constructor

    ~BufferedProjectionSlice()
    {
        delete[] slice;
    } //destructor

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

    } //copy assignment, tmp is to solve situation when assigning to itself

    T get(unsigned int x, unsigned int y) const override
    {
        return slice[y * sizex + x];
    }

    unsigned int dimx() const override
    {
        return sizex;
    }

    unsigned int dimy() const override
    {
        return sizey;
    }

private:
    T* slice;
    int sizex, sizey;
};
}
#endif //BUFFEREDPROJECTIONSLICE_HPP
