#ifndef DENCHUNK2DREADER_HPP
#define DENCHUNK2DREADER_HPP

//External
#include <string>

//Internal
#include "io/BufferedProjectionSlice.hpp"
#include "io/Chunk2DReadI.hpp"
#include "io/Chunk2DReaderI.hpp"
#include "io/DenFileInfo.hpp"

namespace CTL {
namespace io {
    /**
* Implementation of the ProjectionReader for projections and projection matrices stored in the den files.
*/
    template <typename T>
    class DenChunk2DReader : virtual public Chunk2DReaderI<T>
    //Chunk2DReaderI<T> will be only once in the family tree
    {
    protected: //Visible in inheritance structure
        int sizex, sizey, sizez;
        std::string denFile;
        DenSupportedType dataType;
        int elementByteSize;

    public:
        DenChunk2DReader(std::string denFile);
        /**For the reader is necessery to provide den file with chunks.
	*
	*/
        std::shared_ptr<io::Chunk2DReadI<T>> readSlice(int i) override;
        unsigned int dimx() const override;
        unsigned int dimy() const override;
        unsigned int count() const override;
        std::string getFileName() const;
        /**Returns file name of the underlying DEN file.**/
    };

    template <typename T>
    DenChunk2DReader<T>::DenChunk2DReader(std::string denFile)
    {
        this->denFile = denFile;
        DenFileInfo pi = DenFileInfo(this->denFile);
        this->sizey = pi.getNumRows();
        this->sizex = pi.getNumCols();
        this->sizez = pi.getNumSlices();
        this->dataType = pi.getDataType();
        this->elementByteSize = pi.elementByteSize();
    }

    template <typename T>
    std::string DenChunk2DReader<T>::getFileName() const
    {
        return this->denFile;
    }

    template <typename T>
    unsigned int DenChunk2DReader<T>::dimx() const
    {
        return sizex;
    }

    template <typename T>
    unsigned int DenChunk2DReader<T>::dimy() const
    {
        return sizey;
    }

    template <typename T>
    unsigned int DenChunk2DReader<T>::count() const
    {
        return sizez;
    }

    template <typename T>
    std::shared_ptr<io::Chunk2DReadI<T>> DenChunk2DReader<T>::readSlice(int sliceNum)
    {
        uint8_t buffer[elementByteSize * sizex * sizey];
        uint64_t position = ((uint64_t)6) + ((uint64_t)sliceNum) * elementByteSize * sizex * sizey;
        io::readBytesFrom(this->denFile, position, buffer, elementByteSize * sizex * sizey);
        T* buffer_copy = new T[sizex * sizey]; //Object will be deleted during destruction of the BufferedProjectionSlice object
        for (int a = 0; a != sizex * sizey; a++) {
            buffer_copy[a] = util::getNextElement<T>(&buffer[a * elementByteSize], dataType);
        }
        std::shared_ptr<Chunk2DReadI<T>> ps = std::make_shared<BufferedProjectionSlice<T>>(buffer_copy, sizex, sizey);
        return ps;
    }
}
} //namespace CTL::io
#endif //DENPROJECTIONREADER_HPP
