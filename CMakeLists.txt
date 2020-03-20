cmake_minimum_required(VERSION 3.1)
project(ImageRegistration)

set(CMAKE_INSTALL_PREFIX $ENV{HOME}/bin)
set(CMAKE_CXX_STANDARD 14)#Supported values are 98, 11 and 14.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra")
message("Current value of flags is ${CMAKE_CXX_FLAGS} and debug is ${CMAKE_CXX_FLAGS_DEBUG} and release is ${CMAKE_CXX_FLAGS_RELEASE}")
#Default CMAKE_CXX_FLAGS are empty, default for CMAKE_CXX_FLAGS_DEBUG is -g and CMAKE_CXX_FLAGS_RELEASE are not empty
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_BUILD_TYPE Debug)
set(BUILD_SHARED_LIBS False)

#Threads lib
find_package (Threads)

#ITK ... for producing jpg
find_package(ITK)
IF(ITK_FOUND)
	include( ${ITK_USE_FILE} )
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

#Thread pool management lib ctpl from https://github.com/vit-vit/ctpl
include_directories(${CMAKE_SOURCE_DIR}/submodules/ctpl)

#Intel MKL
find_package(MKL)
include_directories(${MKL_INCLUDE_DIRS})

#Matplotlibcpp
include_directories(${CMAKE_SOURCE_DIR}/submodules/matplotlib-cpp)

#Python for matplotlib
find_package(PythonLibs 2.7)
include_directories(${PYTHON_INCLUDE_DIRS})

#CTIOL IO lib
FILE( GLOB DENITK_SRC ${CMAKE_SOURCE_DIR}/submodules/CTIOL/src/DENITK/*.cpp )
add_library(ctiol_itk ${DENITK_SRC})
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
target_link_libraries(dentk_basis ${PYTHON_LIBRARIES})
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
target_link_libraries(dentk_orthogonalize ${PYTHON_LIBRARIES})
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
install (TARGETS dentk_transpose RUNTIME DESTINATION .)

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
install (TARGETS dentk_calc RUNTIME DESTINATION .)

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

