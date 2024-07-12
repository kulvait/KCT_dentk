# KCT DENTK Toolkit

KCT DENTK is a versatile toolkit designed for manipulating raw files in [DEN format](https://kulvait.github.io/KCT_doc/den-format.html), a raw file format with a header specifying dimensions and type of the on disk multidimensional arrays. These files typically represent 3D stacks used in tomographic data analysis, but the format is flexible enough to accommodate various other data structures. 

DENTK provides a set of command line programs to manipulate DEN files. It is designed for easily performing various manipulations, from simple frame extraction to complex mathematical operations.
The toolkit is designed to be used in the command line environment of Linux shell. It is written in C++ and uses the CMake build system. 
 
## Cloning the repository
To clone the repository, use either HTTPS or SSH

HTTPS
```
git clone https://github.com/kulvait/KCT_dentk.git
```

SSH
```
git clone git@github.com:kulvait/KCT_dentk.git
```

After cloning, before compiling DENTK, populate the submodules
```
git submodule init
git submodule update
```

## Dependencies

The DENTK project relies on several external libraries and tools. Below is a comprehensive list of dependencies and instructions on how to install them.

### Required Dependencies

1. **CMake (>= 3.19)**
   - CMake is a cross-platform build system generator.
   - Installation:
     ```bash
     sudo apt-get install cmake
     ```
2. **A C++ compiler that supports C++17**
    - GCC (>= 7.3.0) or Clang (>= 5.0.0) are recommended.
    - Installation:
        ```bash
        sudo apt-get install build-essential
        ```

3. **Threads Library**
   - Used for multi-threading support of virtually all the tools.
   - Installation:
     ```bash
     sudo apt-get install libpthread-stubs0-dev
     ```
4. **Intel MKL (Math Kernel Library)**
   - MKL is used for high-performance mathematical computations, it is a dependecy of the CTMAL submodule so that almost all tools are linked against it.
   - Installation:
     Follow the instructions on the [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html) download page.

5. **Python (>= 3.x)**
   - Python is required for `dentk-basis` and `dentk-orthogonalize`. It is due to the submodule `matplotlib-cpp`.
   - Installation:
     ```bash
     sudo apt-get install python3 python3-dev python3-numpy
     ```

### Optional Dependencies

1. **CUDA Toolkit (>= 11.3)**
   - CUDA is required for GPU acceleration of the tools `dentk-poisson`, `dentk-filter`, `dentk-parker`, `dentk-laplace`, `dentk-propagate`.
    - Without CUDA, these tools will not be built.
   - Installation:
     Follow the instructions on the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) download page.
    -To modify the CMakeLists.txt for different architectures or debug settings, you can uncomment or adjust the relevant sections. For example, to set CUDA architectures:
    ```cmake
    set(CUDA_ARCHITECTURES "70;35")
    ```


2. **ITK (Insight Segmentation and Registration Toolkit)**
   - ITK is required for the tools `dentk-jpg` and `dentk-todicom`.
    - Without ITK, these tools will not be built.
   - Installation:
     ```bash
     sudo apt-get install libinsighttoolkit4-dev
     ```
    - It is safe to ignore cmake warnings of GDCMTargets.cmake



### Submodules

The DENTK project uses the following submodules:

- **[CTIOL](https://github.com/kulvait/KCT_ctiol)**
  - A C++ library of asynchronous, thread-safe I/O routines for for reading and writing CT data in the DEN format.

- **[CTMAL](https://github.com/kulvait/KCT_ctmal)**
  - A C++ library of Mathematic/Algebraic algorithms for supporting CT data manipulation.

- **[Plog](https://github.com/SergiusTheBest/plog)**
   - External library used for logging. MIT license.
   
- **[CLI11](https://github.com/CLIUtils/CLI11)**
   - External library used for parsing command line arguments. CLI11 2.2 is a permissive open-source license that closely resembles the 3-clause BSD License or the MIT License. Originally licensed under 3-clause BSD License.
   
-  **[FTPL](https://github.com/kulvait/FTPL)**
   - Library used for thread pool, licensed under Apache License 2.0. It is fork of [CTPL](https://github.com/vit-vit/ctpl). Gradually replaced by using PROG::ThreadPool from CTIOL, which supports workers.
   
-  **matplotlib-cpp**
   - A C++ wrapper for Python’s matplotlib, MIT License. Used in `dentk-basis` and `dentk-orthogonalize` tools. It is becomming obsolete not supporting all Python 3 features.

To initialize and update the submodules, run:
```bash
git submodule init
git submodule update
```

## Building the Toolkit

The following steps describe how to build the DENTK toolkit out of source tree on Linux.

1. Clone the repository and initialize the submodules.
2. Ensure the dependencies of the tools required are satisfied.
3. Create a build directory in the git repository, navigate to it, and build the project.

```bash
mkdir build
cd build
cmake ..
make
```

# Tools Overview
KCT DENTK includes a variety of tools. Below is a brief overview of the tools and their functionalities. For more detailed information on each tool, refer to the respective tool's help message and the source code.

## Frequently Used Tools  

### dentk-info

Prints information about a DEN file, including dimensions and data format and basic statistic.

```
Usage: dentk-info [OPTIONS] input_den_file

Options:
  -h, --help           Print this help message and exit
  --l2norm             Print l2 norm of the frame specs.
  --dim                Return only the dimensions and quit.
  -f, --frames TEXT    Frames to process. Accepts ranges and individual frames.
  -r, --reverse-order  Output in reverse order.
  -k, --each-kth UINT  Process only each k-th frame.
  -j, --threads UINT   Number of extra threads, default is 32.
```

### dentk-framecalc

Performs mathematical operations between frames of two DEN files.

```
Usage: dentk-framecalc [OPTIONS] input_op1 input_op2 output

Options:
  -h, --help           Print this help message and exit
  --force              Overwrite output files if they exist.
  -f, --frames TEXT    Frames to process. Accepts ranges and individual frames.
  -r, --reverse-order  Output in reverse order.
  -k, --each-kth UINT  Process only each k-th frame.
  -j, --threads UINT   Number of extra threads, default is 0.
[Operation]
  --add, --subtract, --flipped-subtract, --multiply, --divide,
  --flipped-divide, --max, --min
```

### dentk-framecalc1

Aggregates data along the XY frame using efficient aggregation algorithms.
```
Usage: dentk-framecalc1 [OPTIONS] input_den output_den

Options:
  -h, --help           Print this help message and exit
  --force              Overwrite output files if they exist.
  -f, --frames TEXT    Frames to process. Accepts ranges and individual frames.
  -r, --reverse-order  Output in reverse order.
  -k, --each-kth UINT  Process only each k-th frame.
  -j, --threads UINT   Number of extra threads, default is 0.
[Operation]
  --sum, --avg, --variance, --sample-variance,
  --standard-deviation, --sample-standard-deviation,
  --max, --min, --median, --mad
```

### dentk-calc
Element-wise operations on two DEN files with the same dimensions.
```
Usage: dentk-calc [OPTIONS] input_op1 input_op2 output

Options:
  -h, --help           Print this help message and exit
  --force              Overwrite output files if they exist.
  --verbose            Increase verbosity.
  -f, --frames TEXT    Frames to process. Accepts ranges and individual frames.
  -r, --reverse-order  Output in reverse order.
  -k, --each-kth UINT  Process only each k-th frame.
  -j, --threads UINT   Number of extra threads, default is 0.
[Operation]
  --add, --subtract, --multiply, --divide,
  --inverse-divide, --max, --min
```

### dentk-calc1
Performs unary operations or transformations on DEN files.
```
Usage: dentk-calc1 [OPTIONS] input_den output_den

Options:
  -h, --help           Print this help message and exit
  --force              Overwrite output files if they exist.
  --verbose            Increase verbosity.
  -f, --frames TEXT    Frames to process. Accepts ranges and individual frames.
  -r, --reverse-order  Output in reverse order.
  -k, --each-kth UINT  Process only each k-th frame.
  -j, --threads UINT   Number of extra threads, default is 0.
[Operation]
  --log, --exp, --sqrt, --square, --abs,
  --inv, --nan-and-inf-to-zero, --multiply FLOAT,
  --add FLOAT, --min FLOAT, --max FLOAT
```


### dentk-cat 
Extracts and reorders particular frames from a DEN file.
```
Usage: dentk-cat [OPTIONS] input_den_file output_den_file

Options:
  -h, --help           Print this help message and exit
  -f, --frames TEXT    Frames to process. Accepts ranges and individual frames.
  -r, --reverse-order  Output in reverse order.
  -k, --each-kth UINT  Process only each k-th frame.
  -j, --threads UINT   Number of extra threads, default is 0.
```

### dentk-empty
Creates a DEN file with constant or noisy data.
```
Usage: dentk-empty [OPTIONS] dimx dimy dimz output_den_file

Options:
  -h, --help           Print this help message and exit
  -t, --type TEXT      Data type (float, double, uint16_t), default is float.
  --value FLOAT        Default value, default is 0.0
  --noise              Pseudorandom noise from [0,1).
  -j, --threads UINT   Number of extra threads, default is 0.
  --force              Overwrite output files if they exist.
```

### dentk-merge
Merges multiple DEN files together.
```
Usage: dentk-merge [OPTIONS] output_den_file input_den_file1 ... input_den_filen

Options:
  -h, --help           Print this help message and exit
  --force              Overwrite output files if they exist.
  -f, --frames TEXT    Frames to process. Accepts ranges and individual frames.
  -r, --reverse-order  Output in reverse order.
  -k, --each-kth UINT  Process only each k-th frame.
  -j, --threads UINT   Number of extra threads, default is 0.
  -i, --interlacing    First n frames in the output will be from the first n DEN files.
```

## Other Tools
The toolkit includes many other tools for specific operations such as gradient calculation, flipping frames, noise addition, statistical operations, and more. For a complete list of tools and their usage, refer to the source code or use the -h or --help option with each tool.

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

# Donations

If you find this software useful, you can support its development through a donation.

[![Thank you](https://img.shields.io/badge/donate-$15-blue.svg)](https://kulvait.github.io/donate/?amount=15&currency=USD)
