# Thrust Test CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(thrust_test CUDA CXX)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)

# Enable C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Add executable
add_executable(thrust_test thrust_test.cu)

# Set CUDA-specific compiler flags
set_target_properties(thrust_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link against CUDA libraries
target_link_libraries(thrust_test ${CUDA_LIBRARIES})

# Print some diagnostic information
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
