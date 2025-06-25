# Thrust Test and Integration Guide for QForte

This directory contains test files to help you integrate the Thrust library with QForte's Python build system. The tests demonstrate basic Thrust functionality and provide a path for integration with your existing code.

## Files Included

1. **thrust_test.cu** - Standalone C++ test for Thrust operations
2. **CMakeLists.txt** - CMake configuration for standalone test
3. **build_and_run.sh** - Script to build and run standalone test
4. **diagnose_cuda.sh** - Script to diagnose CUDA/Thrust installation
5. **thrust_test_py.cu** - Python extension using Thrust and pybind11
6. **CMakeLists_py.txt** - CMake configuration for Python extension
7. **setup.py** - Python setuptools script for building extension
8. **build_python_ext.sh** - Script to build the Python extension

## Prerequisites

- CUDA Toolkit (10.0 or later recommended)
- CMake (3.18 or later)
- Python 3.6 or later
- pybind11 (install with `pip install pybind11`)

## Diagnosing Your Environment

Before attempting to build, run the diagnostic script to verify your CUDA/Thrust installation:

```bash
./diagnose_cuda.sh
```

This will check for:
- CUDA compiler (nvcc)
- CUDA runtime headers
- Thrust headers
- Environment variables
- Python CUDA packages

## Building the Standalone Test

To build and run the standalone Thrust test:

```bash
./build_and_run.sh
```

This will:
1. Create a build directory
2. Configure with CMake
3. Build the test program
4. Run the test program if build succeeds

## Building the Python Extension

To build the Python extension with Thrust:

```bash
./build_python_ext.sh
```

This will:
1. Build the extension module using setuptools
2. Create a test script
3. Run the test script if build succeeds

## Integration with QForte

To integrate Thrust with QForte:

1. **Add Thrust Headers**: Make sure Thrust headers are included in your source files:
   ```cpp
   #include <thrust/device_vector.h>
   #include <thrust/host_vector.h>
   #include <thrust/transform.h>
   // etc.
   ```

2. **Update CMake Configuration**: Add CUDA as a language and find the CUDA package:
   ```cmake
   project(qforte LANGUAGES CXX CUDA)
   find_package(CUDA REQUIRED)
   include_directories(${CUDA_INCLUDE_DIRS})
   ```

3. **Update Build System**: Modify your setup.py to handle CUDA files:
   - Use the CMakeExtension and CMakeBuild classes from the example setup.py
   - Ensure CUDA files have the .cu extension
   - Make sure CMake is configured to compile CUDA files

4. **Gradual Migration**: Start by migrating simple operations to Thrust, then move to more complex ones.

## Troubleshooting

1. **CUDA Headers Not Found**: Make sure CUDA_HOME or CUDA_PATH environment variables are set correctly.

2. **Build Failures**: Check that your CUDA toolkit is compatible with your compiler version.

3. **Python Integration Issues**: Ensure pybind11 is correctly installed and found by CMake.

4. **Performance Issues**: If Thrust operations are slower than expected, check for unnecessary host-device transfers.

## Further Resources

- [Thrust Documentation](https://docs.nvidia.com/cuda/thrust/index.html)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CMake CUDA Guide](https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html)
