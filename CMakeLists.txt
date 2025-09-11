#cmake_minimum_required(VERSION 3.4)

cmake_minimum_required(VERSION 3.18...3.29)



# -------------------------------------------------------------------
# CUDA arch / Toolkit root
# -------------------------------------------------------------------
# Use the toolkit from the current conda env and target your GPU (RTX 3060 = 86)
# (FORCE clears stale cache values so old arch lists like 70/61/52 don't sneak back in.)
set(CUDAToolkit_ROOT "$ENV{CONDA_PREFIX}")
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "" FORCE)
message(STATUS "Using CUDA archs: ${CMAKE_CUDA_ARCHITECTURES}")

project(qforte LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)

# optionally use OpenMP, if so use OpenMP compatible compiler
OPTION(USE_OpenMP "Use OpenMP to enamble <omp.h>" OFF)

# -------------------------------------------------------------------
# BLAS / OpenBLAS config
# -------------------------------------------------------------------
cmake_policy(SET CMP0074 NEW)         # find_package() respects *_ROOT vars
set(BLA_VENDOR "Generic")             # bypass FindBLAS vendor picks

# (Nick) kept as-is so CMake finds headers/libs inside the env.
# set(CMAKE_PREFIX_PATH "/home/nstair/anaconda3/envs/cuqf_8_25_25_v2")
set(CMAKE_PREFIX_PATH "/home/zach_gonzales/anaconda3/envs/qforte")

set(BLAS_LIBRARIES "${CMAKE_PREFIX_PATH}/lib")
set(CBLAS_INCLUDE_DIR "${CMAKE_PREFIX_PATH}/include")         # need
set(OPENBLAS_EXE "${CMAKE_PREFIX_PATH}/lib/libopenblas.so")   # need

# -------------------------------------------------------------------
# CUDA toolkit & flags
# -------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# For all CUDA files, enable extended lambdas & suppress noisy diagnostics.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -diag-suppress 815,186 -Wno-deprecated-gpu-targets")

# -------------------------------------------------------------------
# Locate Thrust/CUB (handles both NVIDIA "cccl" layout and conda-forge)
# -------------------------------------------------------------------
set(_CUDA_TGT_INC "$ENV{CONDA_PREFIX}/targets/x86_64-linux/include")
set(_CUDA_INC     "$ENV{CONDA_PREFIX}/include")

if (EXISTS "${_CUDA_TGT_INC}/thrust/host_vector.h")
  set(THRUST_ROOT "${_CUDA_TGT_INC}")                # older NVIDIA layout
elseif (EXISTS "${_CUDA_TGT_INC}/cccl/thrust/host_vector.h")
  set(THRUST_ROOT "${_CUDA_TGT_INC}/cccl")           # newer NVIDIA (CCCL) layout
elseif (EXISTS "${_CUDA_INC}/thrust/host_vector.h")
  set(THRUST_ROOT "${_CUDA_INC}")                    # conda-forge layout
elseif (EXISTS "${_CUDA_INC}/cccl/thrust/host_vector.h")
  set(THRUST_ROOT "${_CUDA_INC}/cccl")
else()
  message(FATAL_ERROR
    "Thrust headers not found.\nChecked:\n"
    "  ${_CUDA_TGT_INC}/thrust/host_vector.h\n"
    "  ${_CUDA_TGT_INC}/cccl/thrust/host_vector.h\n"
    "  ${_CUDA_INC}/thrust/host_vector.h\n"
    "  ${_CUDA_INC}/cccl/thrust/host_vector.h\n"
    "Install cuda-cccl (NVIDIA) or thrust (conda-forge), then rebuild.")
endif()
message(STATUS "Using THRUST_ROOT = ${THRUST_ROOT}")

# -------------------------------------------------------------------
# Sources
# -------------------------------------------------------------------
set(SOURCE_DIR "src/qforte")

# CUDA sources (kernels + thrust-based .cu)
set(CUDA_SOURCES
    "${SOURCE_DIR}/tensor_gpu_kernels.cu"
    "${SOURCE_DIR}/fci_computer_gpu_kernels.cu"
    "${SOURCE_DIR}/fci_computer_gpu.cu"
    "${SOURCE_DIR}/tensor_thrust.cu"
    "${SOURCE_DIR}/fci_computer_thrust.cu"
    "${SOURCE_DIR}/fci_graph_thrust.cu"
    "${SOURCE_DIR}/fci_graph_thrust_kernels.cu"
    "${SOURCE_DIR}/sq_op_pool_thrust.cu"
)

# C++ sources
set(SOURCES
    "${SOURCE_DIR}/helpers.cc"
    "${SOURCE_DIR}/make_gate.cc"
    "${SOURCE_DIR}/qubit_basis.cc"
    "${SOURCE_DIR}/circuit.cc"
    "${SOURCE_DIR}/computer.cc"
    "${SOURCE_DIR}/fci_computer.cc"
    "${SOURCE_DIR}/fci_graph.cc"
    "${SOURCE_DIR}/tensor_einsum.cc"
    "${SOURCE_DIR}/tensor_operator.cc"
    "${SOURCE_DIR}/gate.cc"
    "${SOURCE_DIR}/blas_math.cc"
    "${SOURCE_DIR}/qubit_operator.cc"
    "${SOURCE_DIR}/qubit_op_pool.cc"
    "${SOURCE_DIR}/sq_operator.cc"
    "${SOURCE_DIR}/df_hamiltonian.cc"
    "${SOURCE_DIR}/sq_op_pool.cc"
    "${SOURCE_DIR}/sparse_tensor.cc"
    "${SOURCE_DIR}/timer.cc"
    "${SOURCE_DIR}/tensor_gpu.cc"
    "${SOURCE_DIR}/tensor.cc"
    "${SOURCE_DIR}/df_hamiltonian.cc")

# Headers also live in SOURCE_DIR
include_directories(${SOURCE_DIR})

# -------------------------------------------------------------------
# Third-party subdirs
# -------------------------------------------------------------------
include(cmake/PythonStubs.cmake)
add_subdirectory(lib/pybind11)
add_subdirectory(lib/fmt)

# -------------------------------------------------------------------
# Core library (SHARED): performs its own CUDA device-link
# -------------------------------------------------------------------
add_library(qforte_core SHARED ${SOURCES} ${CUDA_SOURCES})
set_target_properties(qforte_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  # Co-locate the shared lib next to the Python module (setup.py sets CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
target_include_directories(qforte_core PRIVATE
  ${CBLAS_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${THRUST_ROOT}
)
target_link_libraries(qforte_core PRIVATE
  CUDA::cudart
  fmt-header-only
  ${OPENBLAS_EXE}
)

# -------------------------------------------------------------------
# Pybind11 module: link against shared core
# -------------------------------------------------------------------
pybind11_add_module(qforte "${SOURCE_DIR}/bindings.cc")

target_include_directories(qforte PRIVATE
  ${CBLAS_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${THRUST_ROOT}
)

#target_link_libraries(qforte PRIVATE qforte_core)
target_link_libraries(qforte PRIVATE qforte_core fmt-header-only)

set_target_properties(qforte PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  # Ensure it can find libqforte_core.so at import time (next to the module)
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)

# Python type stub generation
add_pybind11_stubs(qforte)

# -------------------------------------------------------------------
# OpenMP (attach to core implementation, not the module glue)
# -------------------------------------------------------------------
if(USE_OpenMP)
  find_package(OpenMP REQUIRED)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(qforte_core PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------
set(TEST_DIR "tests")
set(TESTS
    "${TEST_DIR}/test_main.cc"
    "${TEST_DIR}/test_math.cc"
)

include_directories(${CBLAS_INCLUDE_DIR})
include_directories(lib/catch2/single_include/catch2)
include_directories(lib/fmt/include)

add_executable("${PROJECT_NAME}_test" ${TESTS})
# target_link_libraries("${PROJECT_NAME}_test" PRIVATE qforte_core)
target_link_libraries("${PROJECT_NAME}_test" PRIVATE qforte_core fmt-header-only)

# -------------------------------------------------------------------
# Benchmarks
# -------------------------------------------------------------------
add_executable("${PROJECT_NAME}_benchmarks" benchmarks/benchmarks.cc)
set_target_properties(qforte_benchmarks PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
target_include_directories(qforte_benchmarks PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${THRUST_ROOT})
# target_link_libraries(qforte_benchmarks PRIVATE qforte_core)
target_link_libraries(qforte_benchmarks PRIVATE qforte_core fmt-header-only)
