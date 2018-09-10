#ifndef DICOMASYNCWRITTERITK_HPP
#define DICOMASYNCWRITTERITK_HPP

// External libraries
#include <algorithm>
#include <string>

// ITK
#include "gdcmUIDGenerator.h"
#include "itkGDCMImageIO.h"
#include "itkImage.h"
#include "itkImageAdaptor.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkIntensityWindowingImageFilter.h"
#include "itkMetaDataObject.h"

// Internal libraries
#include "io/AsyncImageWritterItkI.hpp"
#include "io/stringFormatter.h"

namespace CTL::io {

template <typename T>
class MultiplyAddAccessor
{
public:
    using InternalType = T;
    using ExternalType = T;
    ExternalType Get(const InternalType& input) const
    {
        T summandx = (T)summand;
        T factorx = (T)factor;
        return (input * factorx) + summandx;
    }

    void setValues(float factor, float summand)
    {
        this->factor = factor;
        this->summand = summand;
    }

private:
    float factor, summand;
};

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
    T windowMin, windowMax;
    int outputMin, outputMax;
    bool useSignedIntegers;
    float multiplyByFactor;
    float addToValues;
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
                         T windowMin,
                         T windowMax,
                         int outputMin,
                         int outputMax,
                         bool useSignedIntegers,
                         float multiplyByFactor,
                         float addToValues);

    /*Need to specify dimension first*/
    void writeSlice(typename itk::Image<T, 2>::Pointer s, int i) override;
    /**Writes i-th slice to the source.*/

    void writeSignedSlice(
        typename itk::ImageAdaptor<itk::Image<T, 2>, MultiplyAddAccessor<T>>::Pointer s, int i);
    /**Writes slice in the int16_t format*/

    void writeUnsignedSlice(
        typename itk::ImageAdaptor<itk::Image<T, 2>, MultiplyAddAccessor<T>>::Pointer s, int i);
    /**Writes slice in the uint16_t format*/

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
                                              T windowMin,
                                              T windowMax,
                                              int outputMin,
                                              int outputMax,
                                              bool useSignedIntegers,
                                              float multiplyByFactor,
                                              float addToValues)
{
    if(dimx < 1 || dimy < 1 || dimz < 1)
    {
        std::string msg = io::xprintf("One of the dimensions is non positive x=%d, y=%d, z=%d.",
                                      dimx, dimy, dimz);
        LOGE << msg;
        throw std::runtime_error(msg);
    }
    this->dicomSeriesDir = dicomSeriesDir;
    this->dicomSeriesPrefix = dicomSeriesPrefix;
    this->sizex = dimx;
    this->sizey = dimy;
    this->sizez = dimz;
    this->windowMin = windowMin;
    this->windowMax = windowMax;
    this->outputMin = outputMin;
    this->outputMax = outputMax;
    this->useSignedIntegers = useSignedIntegers;
    this->multiplyByFactor = multiplyByFactor;
    this->addToValues = addToValues;
    if(useSignedIntegers && (outputMin < -32768 || outputMax > 32767 || outputMin >= outputMax))
    {
        std::string msg
            = io::xprintf("Values of output min %d or output max %d are inconsistent with int16.",
                          outputMin, outputMax);
        LOGE << msg;
        throw std::runtime_error(msg);
    } else if(!useSignedIntegers && (outputMin < 0 || outputMax > 65535 || outputMin >= outputMax))
    {
        std::string msg
            = io::xprintf("Values of output min %d or output max %d are inconsistent with uint16.",
                          outputMin, outputMax);
        LOGE << msg;
        throw std::runtime_error(msg);
    }
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
    typename itk::Image<T, 2>::Pointer x;
    MultiplyAddAccessor<T> a;
    a.setValues(multiplyByFactor, addToValues);
    using ImageAdaptorType = itk::ImageAdaptor<itk::Image<T, 2>, MultiplyAddAccessor<T>>;
    typename ImageAdaptorType::Pointer adaptor = ImageAdaptorType::New();
    adaptor->SetPixelAccessor(a);
    adaptor->SetImage(s);
    if(useSignedIntegers)
    {
        writeSignedSlice(adaptor, i);
    } else
    {
        writeUnsignedSlice(adaptor, i);
    }
    return;
}

template <typename T>
void DICOMAsyncWritterItk<T>::writeUnsignedSlice(
    typename itk::ImageAdaptor<itk::Image<T, 2>, MultiplyAddAccessor<T>>::Pointer s, int i)
{
    std::string fileName = xprintf("%s/%s_%03d.dcm", this->dicomSeriesDir.c_str(),
                                   this->dicomSeriesPrefix.c_str(), i);
    using OutputImageType = itk::Image<uint16_t, 2>;
    using IntensityWindowingImageFilterType = itk::IntensityWindowingImageFilter<
        itk::ImageAdaptor<itk::Image<T, 2>, MultiplyAddAccessor<T>>, OutputImageType>;
    typename IntensityWindowingImageFilterType::Pointer filter
        = IntensityWindowingImageFilterType::New();
    filter->SetInput(s);
    filter->SetWindowMinimum(windowMin);
    filter->SetWindowMaximum(windowMax);
    // LOGD << io::xprintf("Setting window to [%f, %f].", (double)outputMin, (double)outputMax);
    filter->SetOutputMinimum(outputMin);
    filter->SetOutputMaximum(outputMax);
    // filter->SetOutputMaximum((int)(std::min(outputMax*1000, 65535));//Only for Richard
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

template <typename T>
void DICOMAsyncWritterItk<T>::writeSignedSlice(
    typename itk::ImageAdaptor<itk::Image<T, 2>, MultiplyAddAccessor<T>>::Pointer s, int i)
{
    std::string fileName = xprintf("%s/%s_%03d.dcm", this->dicomSeriesDir.c_str(),
                                   this->dicomSeriesPrefix.c_str(), i);
    using OutputImageType = itk::Image<int16_t, 2>;
    using IntensityWindowingImageFilterType = itk::IntensityWindowingImageFilter<
        itk::ImageAdaptor<itk::Image<T, 2>, MultiplyAddAccessor<T>>, OutputImageType>;
    typename IntensityWindowingImageFilterType::Pointer filter
        = IntensityWindowingImageFilterType::New();
    filter->SetInput(s);
    filter->SetWindowMinimum(windowMin);
    filter->SetWindowMaximum(windowMax);
    // LOGD << io::xprintf("Setting window to [%f, %f].", (double)outputMin, (double)outputMax);
    filter->SetOutputMinimum(outputMin);
    filter->SetOutputMaximum(outputMax);
    // filter->SetOutputMaximum((int)(std::min(outputMax*1000, 65535));//Only for Richard
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
