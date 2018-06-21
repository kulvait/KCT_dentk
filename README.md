# DENTK
Toolkit for manipulation with DEN files.

## Submodules
The project contains submodules in the subdirectory submodules. 

## Cloning repository:
Basic clone can be done via
```
git clone https://gitlab.stimulate.ovgu.de/sebastian.bannasch/Projektion_Interpolation.git
```
However to populate submodules directories it is then needed to issue
```
git submodule init
git submodule update
```

## How to build:
For build process, make and cmake utilities are required. Install them using
```
apt-get install make cmake
```


Please create separate directory and build there
```
mkdir build
cd build
cmake ..
make
```

## Used libraries
Documentation of the used version
https://itk.org/Doxygen410/html/index.html

## Using IDEs

# Eclipse
For eclipse to work, create out of source project that is not child or ancesor of the parent directory.
Run the following to generate eclipse project
```
cmake -G"Eclipse CDT4 - Unix Makefiles" -D CMAKE_BUILD_TYPE=Debug path_to_project
```
After each change in CMakeLists.txt remove eclipse project and regenerate it.
See more at https://gitlab.kitware.com/cmake/community/wikis/doc/editors/Eclipse-CDT4-Generator

# Tools included

## dentk-cat 
Is able to extract only specified frames from a den file.

## dentk-empty
Creates empty DEN file. Uses CLI command line parser.

## dentk-info
Prints info about DEN file. Can be used to specify different RGB channels. Intensity windowing is also possible.

## dentk-jpg
Creates jpg from particular slices of DEN file.

## dentk-merge
Merges multiple DEN files together. Can write interlacing. Uses CLI command line parser.
