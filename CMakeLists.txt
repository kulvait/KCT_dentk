cmake_minimum_required(VERSION 3.1)
project(ImageRegistration)

set(CMAKE_INSTALL_PREFIX /home/user/bin)
set(CMAKE_CXX_STANDARD 14)#Supported values are 98, 11 and 14.
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_BUILD_TYPE Debug)
set(BUILD_SHARED_LIBS False)

#Threads lib
find_package (Threads)

#ITK ... for producing jpg
find_package( ITK REQUIRED)
include( ${ITK_USE_FILE} )


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
include_directories(${CMAKE_SOURCE_DIR}/include/strtk)

#CLI11 comand line parser library
include_directories(${CMAKE_SOURCE_DIR}/submodules/CLI11/include)

#Thread pool management lib ctpl from https://github.com/vit-vit/ctpl
include_directories(${CMAKE_SOURCE_DIR}/submodules/ctpl)

FILE( GLOB UTILS_SRC ${CMAKE_SOURCE_DIR}/src/utils/*.cpp )
add_library(utils ${UTILS_SRC})

FILE( GLOB IO_SRC ${CMAKE_SOURCE_DIR}/src/io/*.cpp )
add_library(io ${IO_SRC})

SET(DENTK_CAT_SOURCES src/dentk-cat.cpp)
add_executable(dentk_cat ${DENTK_CAT_SOURCES})
set_target_properties(
    dentk_cat 
    PROPERTIES 
        OUTPUT_NAME "dentk-cat"
        SUFFIX ""
)
target_link_libraries(dentk_cat utils)
target_link_libraries(dentk_cat io)
target_link_libraries(dentk_cat ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries(dentk_cat pmc)
install (TARGETS dentk_cat RUNTIME DESTINATION .)

SET(DENTK_INFO_SOURCES src/dentk-info.cpp)
add_executable(dentk_info ${DENTK_INFO_SOURCES})
set_target_properties(
    dentk_info 
    PROPERTIES 
        OUTPUT_NAME "dentk-info"
        SUFFIX ""
)
target_link_libraries(dentk_info utils)
target_link_libraries(dentk_info io)
install (TARGETS dentk_info RUNTIME DESTINATION .)

SET(DENTK_JPG_SOURCES src/dentk-jpg.cpp)
add_executable(dentk_jpg ${DENTK_JPG_SOURCES})
set_target_properties(
    dentk_jpg 
    PROPERTIES 
        OUTPUT_NAME "dentk-jpg"
        SUFFIX ""
)
target_link_libraries(dentk_jpg utils)
target_link_libraries(dentk_jpg ${ITK_LIBRARIES} )
target_link_libraries(dentk_jpg io)
install (TARGETS dentk_jpg RUNTIME DESTINATION .)

SET(DENTK_MERGE_SOURCES src/dentk-merge.cpp)
add_executable(dentk_merge ${DENTK_MERGE_SOURCES})
set_target_properties(
    dentk_merge 
    PROPERTIES 
        OUTPUT_NAME "dentk-merge"
        SUFFIX ""
)
target_link_libraries(dentk_merge utils)
target_link_libraries(dentk_merge io)
target_link_libraries(dentk_merge ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS dentk_merge RUNTIME DESTINATION .)

SET(DENTK_EMPTY_SOURCES src/dentk-empty.cpp)
add_executable(dentk_empty ${DENTK_EMPTY_SOURCES})
set_target_properties(
    dentk_empty 
    PROPERTIES 
        OUTPUT_NAME "dentk-empty"
        SUFFIX ""
)
target_link_libraries(dentk_empty utils)
target_link_libraries(dentk_empty io)
install (TARGETS dentk_empty RUNTIME DESTINATION .)
