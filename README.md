# KCT DENTK
Toolkit for manipulation with DEN files.

## Cloning repository:
HTTPS
```
git clone https://github.com/kulvait/KCT_dentk.git
```

SSH
```
git clone git@github.com:kulvait/KCT_dentk.git
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

## Additional requirements
For `dentk-jpg` and `dentk-todicom` install insighttoolkit dev libs
Package `dentk-jpg` needs additionally `libfftw3-dev`

In Debian 12 install the packages
```
apt-get install libinsighttoolkit4-dev libfftw3-dev
```
It is safe to ignore cmake warnings of GDCMTargets.cmake

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
Some of the tools, namely dentk-todicom and dentk-jpg use [ITK library](https://itk.org/Doxygen410/html/index.html). 
In order to compile these tools it is recommended to have this library in the system installed and headers reachable. 
On Debian it is sufficient to run
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

# License

When there is no other licensing and/or copyright information in the source files of this project, the following apply for the source files in the directories include and src and for CMakeLists.txt file:

Copyright (C) 2018-2024 Vojtěch Kulvait

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.


This licensing applies to the direct source files in the directories include and src of this project and not for submodules.
