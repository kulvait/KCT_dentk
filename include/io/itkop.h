#ifndef ITKOP_H
#define ITKOP_H

//External
#include "itkCastImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkIntensityWindowingImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include <cstdlib> //EXIT_SUCCESS

//Internal
#include "io/Chunk2DReadI.hpp"

namespace CTL::io {
template <typename T>
int writeImage(typename itk::Image<T, 2>::Pointer img, std::string outputFile);
template <typename T>
int writeCastedImage(typename itk::Image<T, 2>::Pointer img, std::string outputFile, double minValue = 0.0, double maxValue = 0.0);
template <typename T>
int writeChannelsJpg(std::shared_ptr<Chunk2DReadI<T>> img1, std::shared_ptr<Chunk2DReadI<T>> img2, std::string outputFile);
template <typename T>
int writeChannelsRGB(std::shared_ptr<Chunk2DReadI<T>> imgR, std::shared_ptr<Chunk2DReadI<T>> imgG, std::shared_ptr<Chunk2DReadI<T>> imgB, std::string outputFile, double minvalue = std::numeric_limits<double>::infinity(), double maxvalue = -std::numeric_limits<double>::infinity());
} //namespace CTL::io

#endif //ITKOP_H
