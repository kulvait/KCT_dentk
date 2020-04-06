# DENTK
Toolkit for manipulation with DEN files.

## Cloning repository:
For developers that have ssh keys to access repository installed clone via
```
git clone ssh://git@gitlab.stimulate.ovgu.de:2200/vojtech.kulvait/dentk.git
```
For read only origin clone
```
git clone https://gitlab.stimulate.ovgu.de/vojtech.kulvait/dentk.git
```
Next populate submodules
```
git submodule init
git submodule update
```

## How to build:
For build process, make and cmake utilities are required. Install them using
```
apt-get install make cmake
```

Please create separate directory in the project folder then cd to the build directory and run the following commands
```
mkdir build
cd build
cmake ..
make
```

## Submodules
The project contains submodules in the subdirectory submodules. 

### [Plog](https://github.com/SergiusTheBest/plog) logger

Logger Plog is used for logging. It is licensed under the Mozilla Public License Version 2.0.

### [CTPL](https://github.com/vit-vit/ctpl)

Threadpool library. Licensed under Apache Version 2.0 License.

### [CTIOL](ssh://git@gitlab.stimulate.ovgu.de:2200/vojtech.kulvait/CTIOL.git)

Input output routines for asynchronous thread safe reading/writing CT data. The DEN format read/write is implemented.

### [CLI11](https://github.com/CLIUtils/CLI11)

Comand line parser CLI11. It is licensed under 3 Clause BSD License.


## Dependencies
Some of the tools, namely dentk-todicom and dentk-jpg use [ITK library](https://itk.org/Doxygen410/html/index.html). It is required to have this library in the system installed and headers reachable. On Debian it is sufficient to run

```
apt-get install libinsighttoolkit4-dev
```
# Tools included

## dentk-cat 
Is able to extract only specified frames from a den file.

## dentk-diff
Compute difference between two DEN files.

## dentk-empty
Creates empty DEN file. Uses CLI command line parser.

## dentk-fen2den
Fixes the format for the files produced by some conversion tools.

## dentk-fromhu
Convert den in HU to the unitless file.

## dentk-tohu
Convert unitless DEN file into HU.

## dentk-grad
Compute gradient of the DEN file in three dimensions.

## dentk-info
Prints info about DEN file. Can be used to specify different RGB channels. Intensity windowing is also possible.

## dentk-transpose
Transposes all frames in a DEN file and change the header accordingly.

## dentk-merge
Merges multiple DEN files together. Can write interlacing. Uses CLI command line parser.

## dentk-todicom
Converts DEN file into the DICOM format, requires ITK.

## dentk-jpg
Creates jpg from particular slices of DEN file, requires ITK.

# Using IDEs

## Eclipse
For eclipse to work, create out of source project that is not child or ancesor of the parent directory.
Run the following to generate eclipse project
```
cmake -G"Eclipse CDT4 - Unix Makefiles" -D CMAKE_BUILD_TYPE=Debug path_to_project
```
After each change in CMakeLists.txt remove eclipse project and regenerate it.
See more at https://gitlab.kitware.com/cmake/community/wikis/doc/editors/Eclipse-CDT4-Generator
