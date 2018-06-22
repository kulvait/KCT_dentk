#ifndef DENCHUNK2DREADERITK_HPP
#define DENCHUNK2DREADERITK_HPP

//External
#include <string>

//Internal
#include "io/Chunk2DReaderItkI.hpp"

namespace CTL::io {
/**
* Chunk2D reader implementation that is also capable to read these chunks as ITK images.
*/
template <typename T>
class DenChunk2DReaderItk : public DenChunk2DReader<T>, public Chunk2DReaderItkI<T>
//Chunk2DReaderI<T> will be only once in the family tree
{
public:
    DenChunk2DReaderItk(std::string denFile)
        : DenChunk2DReader<T>(denFile)
    {
    }
    /**DEN file to read chunks from.
	*/
    typename itk::Image<T, 2>::Pointer readChunk2DAsItkImage(int i) override;
    /**Function to return projection slice as itk::Image<T, 2>*/
};

template <typename T>
typename itk::Image<T, 2>::Pointer DenChunk2DReaderItk<T>::readChunk2DAsItkImage(int i)
{
    //  LOGD << "Called readProjectionSliceToItkImage method, transpose???";
    typename itk::RawImageIO<T, 2>::Pointer rawImageIO = itk::RawImageIO<T, 2>::New();
    rawImageIO->SetFileName(this->denFile); //(1) ... this is probably unnecessery
    rawImageIO->SetFileTypeToBinary();
    rawImageIO->SetHeaderSize(6 + i * this->elementByteSize * (this->sizex * this->sizey));
    rawImageIO->SetFileDimensionality(2);

    rawImageIO->SetOrigin(0, 0.0); // origin in millimeters
    rawImageIO->SetOrigin(1, 0.0);
    //  LOGD << io::xprintf("Setting dimensions of %s to (x,y) = (%d, %d)", this->projectionsFile.c_str(), this->sizex, this->sizey);
    rawImageIO->SetDimensions(0, this->sizex); // size in pixels
    rawImageIO->SetDimensions(1, this->sizey);
    rawImageIO->SetSpacing(0, 1.0); // spacing in millimeters
    rawImageIO->SetSpacing(1, 1.0);

    rawImageIO->SetByteOrderToLittleEndian();
    rawImageIO->SetPixelType(itk::ImageIOBase::SCALAR);
    rawImageIO->SetNumberOfComponents(1);

    typename itk::ImageFileReader<itk::Image<T, 2>>::Pointer reader = itk::ImageFileReader<itk::Image<T, 2>>::New();
    reader->SetImageIO(rawImageIO);
    reader->SetFileName(this->denFile); //Is it necessary when I have (1)
    reader->Update();
    return reader->GetOutput();
}

} //namespace CTL::io
#endif //DENCHUNK2DREADERITK_HPP
