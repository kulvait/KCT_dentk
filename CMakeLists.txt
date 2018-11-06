cmake_minimum_required(VERSION 3.1)
project(ImageRegistration)

set(CMAKE_INSTALL_PREFIX $ENV{HOME}/bin)
set(CMAKE_CXX_STANDARD 14)#Supported values are 98, 11 and 14.
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
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

#CTIOL IO lib
FILE( GLOB DENITK_SRC ${CMAKE_SOURCE_DIR}/submodules/CTIOL/src/DENITK/*.cpp )
add_library(ctiol_itk ${DENITK_SRC})
include_directories("${CMAKE_SOURCE_DIR}/submodules/CTIOL/include")

FILE(GLOB CTIOL_SRC ${CMAKE_SOURCE_DIR}/submodules/CTIOL/src/*.cpp ${CMAKE_SOURCE_DIR}/submodules/CTIOL/src/ARGPARSE/*.cpp)
add_library(ctiol ${CTIOL_SRC})

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

SET(DENTK_DIFF_SOURCES src/dentk-diff.cpp)
add_executable(dentk_diff ${DENTK_DIFF_SOURCES})
set_target_properties(
    dentk_diff
    PROPERTIES 
        OUTPUT_NAME "dentk-diff"
        SUFFIX ""
)
target_link_libraries(dentk_diff ctiol)
install (TARGETS dentk_diff RUNTIME DESTINATION .)

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

