#ifndef CHUNK2DREADI_HPP
#define CHUNK2DREADI_HPP

namespace CTL::io {
///Interface to read one two dimensional slice of the multidimensional source data
/**
*Intention is to read the subarray of the three dimensional array T A[x, y, z], where one (typically z) dimendion is fixed.
*T is a type of the data used, might be float, double or uint16_t.
*/
template <typename T>
class Chunk2DReadI {
public:
    virtual T get(unsigned int x, unsigned int y) const = 0;
    /**Get the value at coordinates x, y.*/

    T operator()(unsigned int x, unsigned int y) const
    {
        return get(x, y);
    }
    /**Get the value at coordinates x, y. Calls get(x,y).*/

    virtual unsigned int dimx() const = 0;
    /**Returns x dimension.*/

    virtual unsigned int dimy() const = 0;
    /**Returns y dimension.*/

    T normSquare()
    {
        T sum = 0;
        T a;
        for (int i = 0; i != dimx(); i++)
            for (int j = 0; j != dimy(); j++) {
                a = get(i, j);
                sum += a * a;
            }
    }

    T minValue()
    {
        T a;
        T min = get(0, 0);
        for (int i = 0; i != dimx(); i++)
            for (int j = 0; j != dimy(); j++) {
                a = get(i, j);
                min = (a < min ? a : min);
            }
        return min;
    }

    T maxValue()
    {
        T a;
        T max = get(0, 0);
        for (int i = 0; i != dimx(); i++)
            for (int j = 0; j != dimy(); j++) {
                a = get(i, j);
                max = (a > max ? a : max);
            }
        return max;
    }

    double meanValue()
    {
        double sum = 0;
        for (int i = 0; i != dimx(); i++)
            for (int j = 0; j != dimy(); j++) {
                sum += (double)get(i, j);
            }

        return sum / (dimx() * dimy());
    }
};
}
#endif //CHUNK2DREADI_HPP
