cmake_minimum_required(VERSION 3.19)
project(dentk LANGUAGES CXX)
#Conditionally enable CUDA
#before cmake
#module load cuda/11.3
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
	message(STATUS "CUDA compiler found")
	enable_language(CUDA)
else()
	message(STATUS "No CUDA compiler found")
endif()
find_package(CUDAToolkit 11.3)
#set(CUDAToolkit_ROOT "/software/cuda/cuda-11.4/include")
#IF(CUDAToolkit_FOUND)
#	set(CMAKE_CUDA_COMPILER "${CUDAToolkit_BIN_DIR}/nvcc")
#	set(CMAKE_CUDA_ARCHITECTURES "70")
#	message("CUDA enabled")
#ENDIF()

set(CUDA_ARCHITECTURES_NUM 70)
set(CUDA_ARCHITECTURES_NUM 35) #K40 but might be soon deprecated, basically however all the cards on Maxwell cluster will be working
#set(CUDA_ARCHITECTURES_NUM native)
set(CMAKE_INSTALL_PREFIX $ENV{HOME}/KCT_bin)
set(CMAKE_CXX_STANDARD 17)#Supported values are 98, 11, 17, 20, 23
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
#Default CMAKE_CXX_FLAGS are empty, default for CMAKE_CXX_FLAGS_DEBUG is -g and CMAKE_CXX_FLAGS_RELEASE are not empty
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAG} -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")
message("Current value of flags is ${CMAKE_CXX_FLAGS} and debug is ${CMAKE_CXX_FLAGS_DEBUG} and release is ${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_BUILD_TYPE Debug)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G cuda/diffractionPhysics.cu -G cuda/spectralMethod.cu")
endif()
message("Current value of CUDA_NVCC_FLAGS is ${CUDA_NVCC_FLAGS}")


set(BUILD_SHARED_LIBS False)

#Threads lib
find_package (Threads)

#ITK ... for producing jpg
find_package(ITK HINTS $ENV{HOME}/BIG/local/lib/cmake/ITK-5.4)
IF(ITK_FOUND)
#	message( "Including ${ITK_USE_FILE}")
#	message( "Include ${ITK_LIBRARIES}")
#	message( "Include ${ITK_INCLUDE_DIRS}")
	SET(ITKBULLSEYE_INCLUDE_DIR "/usr/include/ITK-4.13/")
	IF(EXISTS ${ITKBULLSEYE_INCLUDE_DIR})
		include_directories(${ITKBULLSEYE_INCLUDE_DIR})
	ELSE()
		include( ${ITK_USE_FILE} )
	ENDIF()
ENDIF()


#Header directories for the project
set(GLOBAL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
include_directories(${GLOBAL_INCLUDE_DIR})

#Includes relative to submodules directories
include_directories(${CMAKE_SOURCE_DIR}/submodules)

#Plog logger
include_directories(${CMAKE_SOURCE_DIR}/submodules/plog/include)

#C++ String Toolkit Library
#(2) If the Boost libraries  (random, regex,  lexical_cast etc) are not
#available  or  it  not  wished   they  be  used  then  the   following
#preprocessor directive needs defining, either in code before strtk.hpp
#is included or as a compiler switch:
#   (*) strtk_no_tr1_or_boost
add_definitions("-Dstrtk_no_tr1_or_boost")

#CLI11 comand line parser library
include_directories(${CMAKE_SOURCE_DIR}/submodules/CLI11/include)

#Thread pool management
include_directories(${CMAKE_SOURCE_DIR}/submodules/FTPL)

#Intel MKL
find_package(MKL)
include_directories(${MKL_INCLUDE_DIRS})

#Matplotlibcpp
include_directories(${CMAKE_SOURCE_DIR}/submodules/matplotlib-cpp)

#Python for matplotlib
#find_package(PythonLibs 2.7)
#include_directories(${PYTHON_INCLUDE_DIRS})
find_package (Python3 COMPONENTS Interpreter Development NumPy)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})

#CTIOL IO lib
IF(ITK_FOUND)
FILE( GLOB DENITK_SRC ${CMAKE_SOURCE_DIR}/submodules/CTIOL/src/DENITK/*.cpp )
add_library(ctiol_itk ${DENITK_SRC})
ENDIF()
 
include_directories("${CMAKE_SOURCE_DIR}/submodules/CTIOL/include")

FILE(GLOB CTIOL_SRC ${CMAKE_SOURCE_DIR}/submodules/CTIOL/src/*.cpp ${CMAKE_SOURCE_DIR}/submodules/CTIOL/src/PROG/*.cpp)
add_library(ctiol ${CTIOL_SRC})
target_link_libraries(ctiol stdc++fs)#include <experimental/filesystem>

#CTMAL mathematics
include_directories(${CMAKE_SOURCE_DIR}/submodules/CTMAL/include)
FILE( GLOB CTMAL_SRC ${CMAKE_SOURCE_DIR}/submodules/CTMAL/src/*.cpp )
add_library(ctmal ${CTMAL_SRC})
target_link_libraries(ctmal ctiol)
target_link_libraries(ctmal ${MKL_CORE_LIBRARY})
set_target_properties(
    ctmal
    PROPERTIES 
        OUTPUT_NAME "ctmal.so"
        SUFFIX ""
)

SET(DENTK_ALPHA_SOURCES src/dentk-alpha.cpp)
add_executable(dentk_alpha ${DENTK_ALPHA_SOURCES})
set_target_properties(
    dentk_alpha
    PROPERTIES 
        OUTPUT_NAME "dentk-alpha"
        SUFFIX ""
)
target_link_libraries(dentk_alpha ctiol)
target_link_libraries(dentk_alpha ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_alpha RUNTIME DESTINATION .)

SET(DENTK_FIND_SOURCES src/dentk-find.cpp)
add_executable(dentk_find ${DENTK_FIND_SOURCES})
set_target_properties(
    dentk_find
    PROPERTIES 
        OUTPUT_NAME "dentk-find"
        SUFFIX ""
)
target_link_libraries(dentk_find ctiol)
target_link_libraries(dentk_find ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_find RUNTIME DESTINATION .)

SET(DENTK_CORELATION_SOURCES src/dentk-corelation.cpp)
add_executable(dentk_corelation ${DENTK_CORELATION_SOURCES})
set_target_properties(
    dentk_corelation
    PROPERTIES 
        OUTPUT_NAME "dentk-corelation"
        SUFFIX ""
)
target_link_libraries(dentk_corelation ctiol)
target_link_libraries(dentk_corelation ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_corelation RUNTIME DESTINATION .)

SET(DENTK_ADDHEADER_SOURCES src/dentk-addheader.cpp)
add_executable(dentk_addheader ${DENTK_ADDHEADER_SOURCES})
set_target_properties(
    dentk_addheader 
    PROPERTIES 
        OUTPUT_NAME "dentk-addheader"
        SUFFIX ""
)
target_link_libraries(dentk_addheader ctiol)
install (TARGETS dentk_addheader RUNTIME DESTINATION .)

SET(DENTK_INFO_SOURCES src/dentk-info.cpp)
add_executable(dentk_info ${DENTK_INFO_SOURCES})
set_target_properties(
    dentk_info 
    PROPERTIES 
        OUTPUT_NAME "dentk-info"
        SUFFIX ""
)
target_link_libraries(dentk_info ctiol)
target_link_libraries(dentk_info ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_info RUNTIME DESTINATION .)

SET(DENTK_MATINFO_SOURCES src/dentk-matinfo.cpp)
add_executable(dentk_matinfo ${DENTK_MATINFO_SOURCES})
set_target_properties(
    dentk_matinfo 
    PROPERTIES 
        OUTPUT_NAME "dentk-matinfo"
        SUFFIX ""
)
target_link_libraries(dentk_matinfo ctiol)
target_link_libraries(dentk_matinfo ctmal)
install (TARGETS dentk_matinfo RUNTIME DESTINATION .)


SET(DENTK_GRAD_SOURCES src/dentk-grad.cpp)
add_executable(dentk_grad ${DENTK_GRAD_SOURCES})
set_target_properties(
    dentk_grad 
    PROPERTIES 
        OUTPUT_NAME "dentk-grad"
        SUFFIX ""
)
target_link_libraries(dentk_grad ctiol)
install (TARGETS dentk_grad RUNTIME DESTINATION .)

SET(DENTK_SVDMATRIX_SOURCES src/dentk-svdmatrix.cpp)
add_executable(dentk_svdmatrix ${DENTK_SVDMATRIX_SOURCES})
set_target_properties(
    dentk_svdmatrix 
    PROPERTIES 
        OUTPUT_NAME "dentk-svdmatrix"
        SUFFIX ""
)
target_link_libraries(dentk_svdmatrix ctiol)
target_link_libraries(dentk_svdmatrix ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_svdmatrix RUNTIME DESTINATION .)

SET(DENTK_CAT_SOURCES src/dentk-cat.cpp)
add_executable(dentk_cat ${DENTK_CAT_SOURCES})
set_target_properties(
    dentk_cat 
    PROPERTIES 
        OUTPUT_NAME "dentk-cat"
        SUFFIX ""
)
target_link_libraries(dentk_cat ctiol)
target_link_libraries(dentk_cat ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_cat RUNTIME DESTINATION .)

IF(TARGET CUDA::cufft)
	include_directories(CUDAToolkit_INCLUDE_DIRS)

	SET(CUDA_SPECTRAL_SOURCES cuda/spectralMethod.cu)
	add_library(cuda_spectral_method ${CUDA_SPECTRAL_SOURCES})
#	target_compile_features(cuda_poisson_equation PUBLIC cxx_std_11)
	set_target_properties(cuda_spectral_method 
		PROPERTIES
				CUDA_SEPARABLE_COMPILATION ON
				CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES_NUM})
	
	SET(CUDA_TOMOGRAPHIC_SOURCES cuda/tomographicFiltering.cu cuda/padding.cu)
	add_library(cuda_tomography ${CUDA_TOMOGRAPHIC_SOURCES})
#	target_compile_features(cuda_poisson_equation PUBLIC cxx_std_11)
	set_target_properties(cuda_tomography
		PROPERTIES
				CUDA_SEPARABLE_COMPILATION ON
				CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES_NUM})

	SET(DENTK_POISSON_SOURCES src/dentk-poisson.cpp)
	add_executable(dentk_poisson ${DENTK_POISSON_SOURCES})
	set_target_properties(
		dentk_poisson 
		PROPERTIES 
			OUTPUT_NAME "dentk-poisson"
			SUFFIX ""
			CUDA_SEPARABLE_COMPILATION ON
			CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES_NUM}
	)
	target_link_libraries(dentk_poisson ctiol)
	target_link_libraries(dentk_poisson cuda_spectral_method)
	target_link_libraries(dentk_poisson ${CMAKE_THREAD_LIBS_INIT})
	target_link_libraries(dentk_poisson CUDA::cudart)
	target_link_libraries(dentk_poisson CUDA::cufft)
	target_link_libraries(dentk_poisson CUDA::cuda_driver)
	install (TARGETS dentk_poisson RUNTIME DESTINATION .)
	
	SET(DENTK_GAUSS_SOURCES src/dentk-gauss.cpp)
	add_executable(dentk_gauss ${DENTK_GAUSS_SOURCES})
	set_target_properties(
		dentk_gauss 
		PROPERTIES 
			OUTPUT_NAME "dentk-gauss"
			SUFFIX ""
			CUDA_SEPARABLE_COMPILATION ON
			CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES_NUM}
	)
	target_link_libraries(dentk_gauss ctiol)
	target_link_libraries(dentk_gauss cuda_tomography)
	target_link_libraries(dentk_gauss ${CMAKE_THREAD_LIBS_INIT})
	target_link_libraries(dentk_gauss CUDA::cudart)
	target_link_libraries(dentk_gauss CUDA::cufft)
	target_link_libraries(dentk_gauss CUDA::cuda_driver)
	install (TARGETS dentk_gauss RUNTIME DESTINATION .)

	SET(DENTK_FILTER_SOURCES src/dentk-filter.cpp)
	add_executable(dentk_filter ${DENTK_FILTER_SOURCES})
	set_target_properties(
		dentk_filter 
		PROPERTIES 
			OUTPUT_NAME "dentk-filter"
			SUFFIX ""
			CUDA_SEPARABLE_COMPILATION ON
			CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES_NUM}
	)
	target_link_libraries(dentk_filter ctiol)
	target_link_libraries(dentk_filter cuda_tomography)
	target_link_libraries(dentk_filter ${CMAKE_THREAD_LIBS_INIT})
	target_link_libraries(dentk_filter CUDA::cudart)
	target_link_libraries(dentk_filter CUDA::cufft)
	target_link_libraries(dentk_filter CUDA::cuda_driver)
	install (TARGETS dentk_filter RUNTIME DESTINATION .)
	
	SET(DENTK_PARKER_SOURCES src/dentk-parker.cpp)
	add_executable(dentk_parker ${DENTK_PARKER_SOURCES})
	set_target_properties(
		dentk_parker 
		PROPERTIES 
			OUTPUT_NAME "dentk-parker"
			SUFFIX ""
			CUDA_SEPARABLE_COMPILATION ON
			CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES_NUM}
	)
	target_link_libraries(dentk_parker ctiol)
	target_link_libraries(dentk_parker ctmal)
	target_link_libraries(dentk_parker cuda_tomography)
	target_link_libraries(dentk_parker ${CMAKE_THREAD_LIBS_INIT})
	target_link_libraries(dentk_parker CUDA::cuda_driver)
	install (TARGETS dentk_parker RUNTIME DESTINATION .)

	SET(DENTK_LAPLACE_SOURCES src/dentk-laplace.cpp)
	add_executable(dentk_laplace ${DENTK_LAPLACE_SOURCES})
	set_target_properties(
		dentk_laplace 
		PROPERTIES 
			OUTPUT_NAME "dentk-laplace"
			SUFFIX ""
			CUDA_SEPARABLE_COMPILATION ON
			CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES_NUM}
	)
	target_link_libraries(dentk_laplace ctiol)
	target_link_libraries(dentk_laplace cuda_spectral_method)
	target_link_libraries(dentk_laplace ${CMAKE_THREAD_LIBS_INIT})
	target_link_libraries(dentk_laplace CUDA::cudart)
	target_link_libraries(dentk_laplace CUDA::cufft)
	target_link_libraries(dentk_laplace CUDA::cuda_driver)
	install (TARGETS dentk_laplace RUNTIME DESTINATION .)
	
	SET(CUDA_DIFFRACTION_SOURCES cuda/diffractionPhysics.cu)
	add_library(cuda_diffraction_physics ${CUDA_DIFFRACTION_SOURCES})
#	target_compile_features(cuda_poisson_equation PUBLIC cxx_std_11)
	set_target_properties(cuda_diffraction_physics 
		PROPERTIES
				CUDA_SEPARABLE_COMPILATION ON
				CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES_NUM})
	
	SET(DENTK_PROPAGATE_SOURCES src/dentk-propagate.cpp)
	add_executable(dentk_propagate ${DENTK_PROPAGATE_SOURCES})
	set_target_properties(
		dentk_propagate 
		PROPERTIES 
			OUTPUT_NAME "dentk-propagate"
			SUFFIX ""
			CUDA_SEPARABLE_COMPILATION ON
			CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES_NUM}
	)
	target_link_libraries(dentk_propagate ctiol)
	target_link_libraries(dentk_propagate cuda_diffraction_physics)
	target_link_libraries(dentk_propagate ${CMAKE_THREAD_LIBS_INIT})
	target_link_libraries(dentk_propagate CUDA::cudart)
	target_link_libraries(dentk_propagate CUDA::cufft)
	target_link_libraries(dentk_propagate CUDA::cuda_driver)
	install (TARGETS dentk_propagate RUNTIME DESTINATION .)

ELSE()
	message("TARGET CUDA::cufftw unknown, will not compile dentk-poisson and dentk-laplace")
ENDIF()

IF(ITK_FOUND)
	SET(DENTK_JPG_SOURCES src/dentk-jpg.cpp)
	add_executable(dentk_jpg ${DENTK_JPG_SOURCES})
	set_target_properties(
			dentk_jpg 
			PROPERTIES 
			OUTPUT_NAME "dentk-jpg"
			SUFFIX ""
	)
	target_link_libraries(dentk_jpg ctiol)
	target_link_libraries(dentk_jpg ctiol_itk)
	target_link_libraries(dentk_jpg ${ITK_LIBRARIES} )
	install (TARGETS dentk_jpg RUNTIME DESTINATION .)

SET(DENTK_TODICOM_SOURCES src/dentk-todicom.cpp)
add_executable(dentk_todicom ${DENTK_TODICOM_SOURCES})
set_target_properties(
    dentk_todicom 
    PROPERTIES 
        OUTPUT_NAME "dentk-todicom"
        SUFFIX ""
)
target_link_libraries(dentk_todicom ctiol)
target_link_libraries(dentk_todicom ${ITK_LIBRARIES} )
target_link_libraries(dentk_todicom stdc++fs)#<experimental/filesystem>
install (TARGETS dentk_todicom RUNTIME DESTINATION .)

ENDIF()

SET(DENTK_RESHAPE_SOURCES src/dentk-reshape.cpp)
add_executable(dentk_reshape ${DENTK_RESHAPE_SOURCES})
set_target_properties(
    dentk_reshape 
    PROPERTIES 
        OUTPUT_NAME "dentk-reshape"
        SUFFIX ""
)
target_link_libraries(dentk_reshape ctiol)
target_link_libraries(dentk_reshape stdc++fs)#<experimental/filesystem>
install (TARGETS dentk_reshape RUNTIME DESTINATION .)

SET(DENTK_PCOMP_SOURCES src/dentk-pcomp.cpp)
add_executable(dentk_pcomp ${DENTK_PCOMP_SOURCES})
set_target_properties(
    dentk_pcomp
    PROPERTIES 
        OUTPUT_NAME "dentk-pcomp"
        SUFFIX ""
)
target_link_libraries(dentk_pcomp ctiol)
target_link_libraries(dentk_pcomp ctmal)
target_link_libraries(dentk_pcomp ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(dentk_pcomp ${MKL_CORE_LIBRARY})
install (TARGETS dentk_pcomp RUNTIME DESTINATION .)

SET(DENTK_SVD_SOURCES src/dentk-svd.cpp)
add_executable(dentk_svd ${DENTK_SVD_SOURCES})
set_target_properties(
    dentk_svd
    PROPERTIES 
        OUTPUT_NAME "dentk-svd"
        SUFFIX ""
)
target_link_libraries(dentk_svd ctiol)
target_link_libraries(dentk_svd ctmal)
target_link_libraries(dentk_svd ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(dentk_svd ${MKL_CORE_LIBRARY})
install (TARGETS dentk_svd RUNTIME DESTINATION .)

SET(DENTK_DIRECTSUM_SOURCES src/dentk-directsum.cpp)
add_executable(dentk_directsum ${DENTK_DIRECTSUM_SOURCES})
set_target_properties(
    dentk_directsum
    PROPERTIES 
        OUTPUT_NAME "dentk-directsum"
        SUFFIX ""
)
target_link_libraries(dentk_directsum ctiol)
target_link_libraries(dentk_directsum ctmal)
target_link_libraries(dentk_directsum ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_directsum RUNTIME DESTINATION .)

SET(DENTK_SVDVNORMALIZE_SOURCES src/dentk-svdvnormalize.cpp)
add_executable(dentk_svdvnormalize ${DENTK_SVDVNORMALIZE_SOURCES})
set_target_properties(
    dentk_svdvnormalize
    PROPERTIES 
        OUTPUT_NAME "dentk-svdvnormalize"
        SUFFIX ""
)
target_link_libraries(dentk_svdvnormalize ctiol)
target_link_libraries(dentk_svdvnormalize ctmal)
target_link_libraries(dentk_svdvnormalize ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(dentk_svdvnormalize ${MKL_CORE_LIBRARY})
install (TARGETS dentk_svdvnormalize RUNTIME DESTINATION .)

SET(DENTK_BOUNDARYFILL_SOURCES src/dentk-boundaryfill.cpp)
add_executable(dentk_boundaryfill ${DENTK_BOUNDARYFILL_SOURCES})
set_target_properties(
    dentk_boundaryfill
    PROPERTIES 
        OUTPUT_NAME "dentk-boundaryfill"
        SUFFIX ""
)
target_link_libraries(dentk_boundaryfill ctiol)
target_link_libraries(dentk_boundaryfill ctmal)
target_link_libraries(dentk_boundaryfill ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(dentk_boundaryfill ${MKL_CORE_LIBRARY})
install (TARGETS dentk_boundaryfill RUNTIME DESTINATION .)

SET(DENTK_INSIDE_SOURCES src/dentk-inside.cpp)
add_executable(dentk_inside ${DENTK_INSIDE_SOURCES})
set_target_properties(
    dentk_inside
    PROPERTIES 
        OUTPUT_NAME "dentk-inside"
        SUFFIX ""
)
target_link_libraries(dentk_inside ctiol)
target_link_libraries(dentk_inside ctmal)
target_link_libraries(dentk_inside ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(dentk_inside ${MKL_CORE_LIBRARY})
install (TARGETS dentk_inside RUNTIME DESTINATION .)

SET(DENTK_BASIS_SOURCES src/dentk-basis.cpp)
add_executable(dentk_basis ${DENTK_BASIS_SOURCES})
set_target_properties(
    dentk_basis
    PROPERTIES 
        OUTPUT_NAME "dentk-basis"
        SUFFIX ""
)
target_link_libraries(dentk_basis ctiol)
target_link_libraries(dentk_basis ctmal)
target_link_libraries(dentk_basis ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(dentk_basis ${MKL_CORE_LIBRARY})
target_link_libraries(dentk_basis ${Python3_LIBRARIES})
install (TARGETS dentk_basis RUNTIME DESTINATION .)

SET(DENTK_NOISE_SOURCES src/dentk-noise.cpp)
add_executable(dentk_noise ${DENTK_NOISE_SOURCES})
set_target_properties(
    dentk_noise
    PROPERTIES 
        OUTPUT_NAME "dentk-noise"
        SUFFIX ""
)
target_link_libraries(dentk_noise ctiol)
target_link_libraries(dentk_noise ctmal)
target_link_libraries(dentk_noise ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_noise RUNTIME DESTINATION .)

SET(DENTK_NOISE_SOURCES src/dentk-regression.cpp)
add_executable(dentk_regression ${DENTK_NOISE_SOURCES})
set_target_properties(
    dentk_regression
    PROPERTIES 
        OUTPUT_NAME "dentk-regression"
        SUFFIX ""
)
target_link_libraries(dentk_regression ctiol)
target_link_libraries(dentk_regression ctmal)
target_link_libraries(dentk_regression ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_regression RUNTIME DESTINATION .)

SET(DENTK_ORTHOGONALIZE_SOURCES src/dentk-orthogonalize.cpp)
add_executable(dentk_orthogonalize ${DENTK_ORTHOGONALIZE_SOURCES})
set_target_properties(
    dentk_orthogonalize
    PROPERTIES 
        OUTPUT_NAME "dentk-orthogonalize"
        SUFFIX ""
)
target_link_libraries(dentk_orthogonalize ctiol)
target_link_libraries(dentk_orthogonalize ctmal)
target_link_libraries(dentk_orthogonalize ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(dentk_orthogonalize ${MKL_CORE_LIBRARY})
target_link_libraries(dentk_orthogonalize ${Python3_LIBRARIES})
install (TARGETS dentk_orthogonalize RUNTIME DESTINATION .)

SET(DENTK_TRUNCATE_SOURCES src/dentk-truncate.cpp)
add_executable(dentk_truncate ${DENTK_TRUNCATE_SOURCES})
set_target_properties(
    dentk_truncate 
    PROPERTIES 
        OUTPUT_NAME "dentk-truncate"
        SUFFIX ""
)
target_link_libraries(dentk_truncate ctiol)
target_link_libraries(dentk_truncate ctmal)
target_link_libraries(dentk_truncate ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_truncate RUNTIME DESTINATION .)

SET(DENTK_FEN2DEN_SOURCES src/dentk-fen2den.cpp)
add_executable(dentk_fen2den ${DENTK_FEN2DEN_SOURCES})
set_target_properties(
    dentk_fen2den
    PROPERTIES 
        OUTPUT_NAME "dentk-fen2den"
        SUFFIX ""
)
target_link_libraries(dentk_fen2den ctiol)
install (TARGETS dentk_fen2den RUNTIME DESTINATION .)

SET(DENTK_TRANSPOSE_SOURCES src/dentk-transpose.cpp)
add_executable(dentk_transpose ${DENTK_TRANSPOSE_SOURCES})
set_target_properties(
    dentk_transpose
    PROPERTIES 
        OUTPUT_NAME "dentk-transpose"
        SUFFIX ""
)
target_link_libraries(dentk_transpose ctiol)
target_link_libraries(dentk_transpose ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_transpose RUNTIME DESTINATION .)

SET(DENTK_SWAPAXES_SOURCES src/dentk-swapaxes.cpp)
add_executable(dentk_swapaxes ${DENTK_SWAPAXES_SOURCES})
set_target_properties(
    dentk_swapaxes
    PROPERTIES 
	OUTPUT_NAME "dentk-swapaxes"
	SUFFIX ""
)
target_link_libraries(dentk_swapaxes ctiol)
target_link_libraries(dentk_swapaxes ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_swapaxes RUNTIME DESTINATION .)

SET(DENTK_FLIP_SOURCES src/dentk-flip.cpp)
add_executable(dentk_flip ${DENTK_FLIP_SOURCES})
set_target_properties(
    dentk_flip
    PROPERTIES 
        OUTPUT_NAME "dentk-flip"
        SUFFIX ""
)
target_link_libraries(dentk_flip ctiol)
target_link_libraries(dentk_flip ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_flip RUNTIME DESTINATION .)

SET(DENTK_MERGE_SOURCES src/dentk-merge.cpp)
add_executable(dentk_merge ${DENTK_MERGE_SOURCES})
set_target_properties(
    dentk_merge 
    PROPERTIES 
        OUTPUT_NAME "dentk-merge"
        SUFFIX ""
)
target_link_libraries(dentk_merge ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(dentk_merge ctiol)
install (TARGETS dentk_merge RUNTIME DESTINATION .)

SET(DENTK_STAT_SOURCES src/dentk-stat.cpp)
add_executable(dentk_stat ${DENTK_STAT_SOURCES})
set_target_properties(
    dentk_stat 
    PROPERTIES 
        OUTPUT_NAME "dentk-stat"
        SUFFIX ""
)
target_link_libraries(dentk_stat ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(dentk_stat ctiol)
install (TARGETS dentk_stat RUNTIME DESTINATION .)

SET(DENTK_VALUE_SOURCES src/dentk-value.cpp)
add_executable(dentk_value ${DENTK_VALUE_SOURCES})
set_target_properties(
    dentk_value 
    PROPERTIES 
        OUTPUT_NAME "dentk-value"
        SUFFIX ""
)
target_link_libraries(dentk_value ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(dentk_value ctiol)
install (TARGETS dentk_value RUNTIME DESTINATION .)

SET(DENTK_EMPTY_SOURCES src/dentk-empty.cpp)
add_executable(dentk_empty ${DENTK_EMPTY_SOURCES})
set_target_properties(
    dentk_empty 
    PROPERTIES 
        OUTPUT_NAME "dentk-empty"
        SUFFIX ""
)
target_link_libraries(dentk_empty ctiol)
target_link_libraries(dentk_empty ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_empty RUNTIME DESTINATION .)

SET(DENTK_CALC_SOURCES src/dentk-calc.cpp)
add_executable(dentk_calc ${DENTK_CALC_SOURCES})
set_target_properties(
    dentk_calc
    PROPERTIES 
        OUTPUT_NAME "dentk-calc"
        SUFFIX ""
)
target_link_libraries(dentk_calc ctiol)
target_link_libraries(dentk_calc ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_calc RUNTIME DESTINATION .)

SET(DENTK_CALC_SOURCES src/dentk-calc1.cpp)
add_executable(dentk_calc1 ${DENTK_CALC_SOURCES})
set_target_properties(
    dentk_calc1
    PROPERTIES 
        OUTPUT_NAME "dentk-calc1"
        SUFFIX ""
)
target_link_libraries(dentk_calc1 ctiol)
target_link_libraries(dentk_calc1 ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_calc1 RUNTIME DESTINATION .)

SET(DENTK_FRAMECALC1_SOURCES src/dentk-framecalc1.cpp)
add_executable(dentk_framecalc1 ${DENTK_FRAMECALC1_SOURCES})
set_target_properties(
    dentk_framecalc1
    PROPERTIES 
        OUTPUT_NAME "dentk-framecalc1"
        SUFFIX ""
)
target_link_libraries(dentk_framecalc1 ctiol)
target_link_libraries(dentk_framecalc1 ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_framecalc1 RUNTIME DESTINATION .)

SET(DENTK_FRAMECALC_SOURCES src/dentk-framecalc.cpp)
add_executable(dentk_framecalc ${DENTK_FRAMECALC_SOURCES})
set_target_properties(
    dentk_framecalc
    PROPERTIES 
        OUTPUT_NAME "dentk-framecalc"
        SUFFIX ""
)
target_link_libraries(dentk_framecalc ctiol)
target_link_libraries(dentk_framecalc ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_framecalc RUNTIME DESTINATION .)

SET(DENTK_FRAMEPRODUCT_SOURCES src/dentk-frameproduct.cpp)
add_executable(dentk_frameproduct ${DENTK_FRAMEPRODUCT_SOURCES})
set_target_properties(
    dentk_frameproduct
    PROPERTIES 
        OUTPUT_NAME "dentk-frameproduct"
        SUFFIX ""
)
target_link_libraries(dentk_frameproduct ctiol)
target_link_libraries(dentk_frameproduct ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_frameproduct RUNTIME DESTINATION .)

SET(DENTK_FRAMENORMALIZE_SOURCES src/dentk-framenormalize.cpp)
add_executable(dentk_framenormalize ${DENTK_FRAMENORMALIZE_SOURCES})
set_target_properties(
    dentk_framenormalize
    PROPERTIES 
        OUTPUT_NAME "dentk-framenormalize"
        SUFFIX ""
)
target_link_libraries(dentk_framenormalize ctiol)
target_link_libraries(dentk_framenormalize ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_framenormalize RUNTIME DESTINATION .)

SET(DENTK_TOHU_SOURCES src/dentk-tohu.cpp)
add_executable(dentk_tohu ${DENTK_TOHU_SOURCES})
set_target_properties(
    dentk_tohu
    PROPERTIES 
        OUTPUT_NAME "dentk-tohu"
        SUFFIX ""
)
target_link_libraries(dentk_tohu ctiol)
install (TARGETS dentk_tohu RUNTIME DESTINATION .)

SET(DENTK_FROMHU_SOURCES src/dentk-fromhu.cpp)
add_executable(dentk_fromhu ${DENTK_FROMHU_SOURCES})
set_target_properties(
    dentk_fromhu
    PROPERTIES 
        OUTPUT_NAME "dentk-fromhu"
        SUFFIX ""
)
target_link_libraries(dentk_fromhu ctiol)
install (TARGETS dentk_fromhu RUNTIME DESTINATION .)

