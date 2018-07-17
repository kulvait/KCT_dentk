#ifndef DICOMASYNCWRITTERITK_HPP
#define DICOMASYNCWRITTERITK_HPP

// External libraries
#include <string>

// ITK
#include "gdcmUIDGenerator.h"
#include "itkGDCMImageIO.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkIntensityWindowingImageFilter.h"
#include "itkMetaDataObject.h"

// Internal libraries
#include "io/AsyncImageWritterItkI.hpp"
#include "io/stringFormatter.h"

namespace CTL::io {
/**
Interface for writing images. It is not necessery to write matrices along them.
*/
template <typename T>
class DICOMAsyncWritterItk : public AsyncImageWritterItkI<T>
{
private:
    std::string dicomSeriesDir;
    std::string dicomSeriesPrefix;
    int sizex, sizey, sizez;
    T outputMin, outputMax;
    gdcm::UIDGenerator uid_generator;
    std::string seriesUID;
    std::string studyUID;
    std::string frameOfReferenceUID;

public:
    DICOMAsyncWritterItk(std::string dicomSeriesDir,
                         std::string dicomSeriesPrefix,
                         int dimx,
                         int dimy,
                         int dimz,
                         T outputMin,
                         T outputMax);
    /*Need to specify dimension first*/
    void writeSlice(typename itk::Image<T, 2>::Pointer s, int i) override;
    /**Writes i-th slice to the source.*/

    virtual unsigned int dimx() const override;
    /**Returns x dimension.*/

    virtual unsigned int dimy() const override;
    /**Returns y dimension.*/

    virtual unsigned int dimz() const override;
    /**Returns z dimension.*/

    std::string getSeriesDir() const;
    /**Directory to write to.**/

    std::string getSeriesPrefix() const;
    /**Series prefix.**/
};

template <typename T>
DICOMAsyncWritterItk<T>::DICOMAsyncWritterItk(std::string dicomSeriesDir,
                                              std::string dicomSeriesPrefix,
                                              int dimx,
                                              int dimy,
                                              int dimz,
                                              T outputMin,
                                              T outputMax)
{
    if(dimx < 1 || dimy < 1 || dimz < 1)
    {
        std::string msg = io::xprintf("One of the dimensions is nonpositive x=%d, y=%d, z=%d.",
                                      dimx, dimy, dimz);
        LOGE << msg;
        throw std::runtime_error(msg);
    }
    this->dicomSeriesDir = dicomSeriesDir;
    this->dicomSeriesPrefix = dicomSeriesPrefix;
    this->sizex = dimx;
    this->sizey = dimy;
    this->sizez = dimz;
    this->outputMin = outputMin;
    this->outputMax = outputMax;

    seriesUID = uid_generator.Generate();
    studyUID = uid_generator.Generate();
    frameOfReferenceUID
        = uid_generator.Generate(); // Coordinate system is the same for whole series
}

template <typename T>
std::string DICOMAsyncWritterItk<T>::getSeriesDir() const
{
    return this->dicomSeriesDir;
}

template <typename T>
std::string DICOMAsyncWritterItk<T>::getSeriesPrefix() const
{
    return this->dicomSeriesPrefix;
}

template <typename T>
unsigned int DICOMAsyncWritterItk<T>::dimx() const
{
    return sizex;
}

template <typename T>
unsigned int DICOMAsyncWritterItk<T>::dimy() const
{
    return sizey;
}

template <typename T>
unsigned int DICOMAsyncWritterItk<T>::dimz() const
{
    return sizez;
}

template <typename T>
void DICOMAsyncWritterItk<T>::writeSlice(typename itk::Image<T, 2>::Pointer s, int i)
{
    std::string fileName = xprintf("%s/%s_%03d.dcm", this->dicomSeriesDir.c_str(),
                                   this->dicomSeriesPrefix.c_str(), i);
    using OutputPixelType = uint16_t;
    using OutputImageType = itk::Image<OutputPixelType, 2>;
    using IntensityWindowingImageFilterType
        = itk::IntensityWindowingImageFilter<itk::Image<T, 2>, OutputImageType>;
    typename IntensityWindowingImageFilterType::Pointer filter
        = IntensityWindowingImageFilterType::New();
    filter->SetInput(s);
    filter->SetWindowMinimum(outputMin);
    filter->SetWindowMaximum(outputMax);
    LOGD << io::xprintf("Setting window to [%f, %f].", (double)outputMin, (double)outputMax);
    filter->SetOutputMinimum(0);
    filter->SetOutputMaximum(65535);
    //    filter->SetOutputMaximum((int)(outputMax*1000));//Only for Richard
    // filter->SetOutputMaximum(255);
    typename itk::GDCMImageIO::Pointer gdcmImageIO = itk::GDCMImageIO::New();
    gdcmImageIO->SetKeepOriginalUID(true); // Need to be set, otherwise it overwrites everything
    itk::ImageFileWriter<OutputImageType>::Pointer writer
        = itk::ImageFileWriter<OutputImageType>::New();
    writer->SetImageIO(gdcmImageIO);
    std::string tmpFile = std::tmpnam(nullptr);
    writer->SetFileName(tmpFile);
    writer->SetInput(filter->GetOutput());
    try
    {
        writer->Update();
    } catch(itk::ExceptionObject& e)
    {
        LOGE << xprintf("Exception thrown when writing file %s.", fileName.c_str());
        LOGE << xprintf("%s", e.GetDescription());
        throw(e);
    }
    itk::ImageFileReader<OutputImageType>::Pointer reader
        = itk::ImageFileReader<OutputImageType>::New();
    reader->SetFileName(tmpFile);
    gdcmImageIO = itk::GDCMImageIO::New();
    gdcmImageIO->SetKeepOriginalUID(true);
    reader->SetImageIO(gdcmImageIO);
    try
    {
        reader->Update();
    } catch(itk::ExceptionObject& e)
    {
        LOGE << xprintf("Exception thrown when reading file %s.", fileName.c_str());
        LOGE << xprintf("%s", e.GetDescription());
        throw(e);
    }
    OutputImageType::Pointer inputImage = reader->GetOutput();
    itk::MetaDataDictionary& dictionary = inputImage->GetMetaDataDictionary();

    typename itk::GDCMImageIO::Pointer gdcmImageIO2 = itk::GDCMImageIO::New();
    gdcmImageIO2->SetKeepOriginalUID(true);
    std::string sopClassUIDCTImageStorage = "1.2.840.10008.5.1.4.1.1.2";
    itk::EncapsulateMetaData<std::string>(dictionary, "0008|0016", sopClassUIDCTImageStorage);
    itk::EncapsulateMetaData<std::string>(dictionary, "0020|000d", studyUID);
    itk::EncapsulateMetaData<std::string>(dictionary, "0020|000e", seriesUID);
    itk::EncapsulateMetaData<std::string>(dictionary, "0020|0052", frameOfReferenceUID);
    // This should be unique for each slice
    itk::EncapsulateMetaData<std::string>(dictionary, "0008|0018", uid_generator.Generate());
    itk::EncapsulateMetaData<std::string>(dictionary, "(0008, 0060)", "CT");
    itk::EncapsulateMetaData<std::string>(dictionary, "0020|0013", std::to_string(i));
    writer = itk::ImageFileWriter<OutputImageType>::New();
    writer->SetInput(reader->GetOutput());
    writer->SetFileName(fileName);
    writer->SetImageIO(gdcmImageIO2);
    try
    {
        writer->Update();
    } catch(itk::ExceptionObject& e)
    {
        LOGE << xprintf("Exception thrown when writing file %s.", fileName.c_str());
        LOGE << xprintf("%s", e.GetDescription());
        throw(e);
    }
    return;
}

} // namespace CTL::io
#endif // DENASYNCWRITTER_HPP
