cmake_minimum_required(VERSION 3.4)

project(qforte LANGUAGES CXX C CUDA)

set (CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED OFF)

# optionally use OpenMP, if so use OpenMP compatible compiler
OPTION (USE_OpenMP "Use OpenMP to enamble <omp.h>" OFF)



# Need to check how much of this is actually needed!

# Find BLAS library
cmake_policy(SET CMP0074 NEW)

# Bypass FindBLAS module
set(BLA_VENDOR "Generic")

# (Nick) I need the following, may be elsewhere or unneeded on other os. 
# set(CMAKE_C_COMPILER "/usr/bin/gcc")
# set(CMAKE_CXX_COMPILER "/usr/bin/g++")

# Use these if installing to anaconda
set(CMAKE_PREFIX_PATH "/home/tylerpbrady/anaconda3/envs/tyler_gpu")
set(BLAS_LIBRARIES "${CMAKE_PREFIX_PATH}/lib") 
set(CBLAS_INCLUDE_DIR "${CMAKE_PREFIX_PATH}/include") #need
set(OPENBLAS_EXE "${CMAKE_PREFIX_PATH}/lib/libopenblas.so") #need, may be libopenblas.so for linux
set(CUDA_EXE "${CMAKE_PREFIX_PATH}/lib/libcudart.so")
# find_package(BLAS REQUIRED)

find_package(CUDAToolkit REQUIRED)

# Set source directory
set(SOURCE_DIR "src/qforte")
set(CUDA_SOURCES "${SOURCE_DIR}/tensor_gpu.cu")




# Tell CMake that headers are also in SOURCE_DIR
include_directories(${SOURCE_DIR})
#set(SOURCES "${SOURCE_DIR}/math.cc")

# Generate Python module
add_subdirectory(lib/pybind11)
add_subdirectory(lib/fmt)
pybind11_add_module(qforte ${SOURCES} "${SOURCE_DIR}/bindings.cc"
    "${SOURCE_DIR}/helpers.cc"
    "${SOURCE_DIR}/make_gate.cc"
    "${SOURCE_DIR}/qubit_basis.cc"
    "${SOURCE_DIR}/circuit.cc"
    "${SOURCE_DIR}/computer.cc"
    "${SOURCE_DIR}/fci_computer.cc"
    "${SOURCE_DIR}/fci_graph.cc"
    "${SOURCE_DIR}/tensor_gpu.cc"
    "${SOURCE_DIR}/tensor.cc"
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
    ${CUDA_SOURCES})

set_target_properties(qforte PROPERTIES CUDA_ARCHITECTURES "52;60;61;70;75")

link_directories(${BLAS_LIBRARIES})
target_include_directories(qforte PRIVATE ${CBLAS_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(qforte PRIVATE fmt-header-only CUDA::cudart ${OPENBLAS_EXE} ${CUDA_EXE})




# specify open MP flags
if(USE_OpenMP)
  find_package(OpenMP REQUIRED)
  if(OpenMP_CXX_FOUND)
      target_link_libraries(qforte PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif(USE_OpenMP)

SET(TEST_DIR "tests")
SET(TESTS ${SOURCES}
    "${TEST_DIR}/test_main.cc"
    "${TEST_DIR}/test_math.cc")

# Generate a test executable
include_directories(${CBLAS_INCLUDE_DIR})
include_directories(lib/catch2/single_include/catch2)
include_directories(lib/fmt/include)

add_executable("${PROJECT_NAME}_test" ${TESTS})

add_executable("${PROJECT_NAME}_benchmarks"
    benchmarks/benchmarks.cc
    "${SOURCE_DIR}/helpers.cc"
    "${SOURCE_DIR}/make_gate.cc"
    "${SOURCE_DIR}/qubit_basis.cc"
    "${SOURCE_DIR}/circuit.cc"
    "${SOURCE_DIR}/computer.cc"
    "${SOURCE_DIR}/fci_computer.cc"
    "${SOURCE_DIR}/fci_graph.cc"
    "${SOURCE_DIR}/tensor_gpu.cc"
    "${SOURCE_DIR}/tensor.cc"
    "${SOURCE_DIR}/tensor_einsum.cc"
    "${SOURCE_DIR}/tensor_operator.cc"
    "${SOURCE_DIR}/gate.cc"
    "${SOURCE_DIR}/blas_math.cc"
    "${SOURCE_DIR}/qubit_operator.cc"
    "${SOURCE_DIR}/sq_operator.cc"
    "${SOURCE_DIR}/df_hamiltonian.cc"
    "${SOURCE_DIR}/qubit_op_pool.cc"
    "${SOURCE_DIR}/sq_op_pool.cc"
    "${SOURCE_DIR}/sparse_tensor.cc"
    "${SOURCE_DIR}/timer.cc")

target_link_libraries(qforte_benchmarks PRIVATE fmt-header-only ${OPENBLAS_EXE})
