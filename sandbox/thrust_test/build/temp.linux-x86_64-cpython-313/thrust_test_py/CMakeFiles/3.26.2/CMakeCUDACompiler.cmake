set(CMAKE_CUDA_COMPILER "/home/zach_gonzales/anaconda3/envs/qforte_gpu/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/home/zach_gonzales/anaconda3/envs/qforte_gpu/bin/x86_64-conda-linux-gnu-c++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "12.9.86")
set(CMAKE_CUDA_DEVICE_LINKER "/home/zach_gonzales/anaconda3/envs/qforte_gpu/targets/x86_64-linux/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/home/zach_gonzales/anaconda3/envs/qforte_gpu/targets/x86_64-linux/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17;cuda_std_20")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "cuda_std_20")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "11.2")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/home/zach_gonzales/anaconda3/envs/qforte_gpu/targets/x86_64-linux")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/home/zach_gonzales/anaconda3/envs/qforte_gpu/targets/x86_64-linux")
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "12.9.86")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/home/zach_gonzales/anaconda3/envs/qforte_gpu/targets/x86_64-linux")

set(CMAKE_CUDA_ARCHITECTURES_ALL "50-real;52-real;53-real;60-real;61-real;62-real;70-real;72-real;75-real;80-real;86-real;87-real;89-real;90")
set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR "50-real;60-real;70-real;80-real;90")
set(CMAKE_CUDA_ARCHITECTURES_NATIVE "86-real")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/home/zach_gonzales/anaconda3/envs/qforte_gpu/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/home/zach_gonzales/anaconda3/envs/qforte_gpu/targets/x86_64-linux/lib/stubs;/home/zach_gonzales/anaconda3/envs/qforte_gpu/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/home/zach_gonzales/anaconda3/envs/qforte_gpu/x86_64-conda-linux-gnu/include/c++/11.2.0;/home/zach_gonzales/anaconda3/envs/qforte_gpu/x86_64-conda-linux-gnu/include/c++/11.2.0/x86_64-conda-linux-gnu;/home/zach_gonzales/anaconda3/envs/qforte_gpu/x86_64-conda-linux-gnu/include/c++/11.2.0/backward;/home/zach_gonzales/anaconda3/envs/qforte_gpu/lib/gcc/x86_64-conda-linux-gnu/11.2.0/include;/home/zach_gonzales/anaconda3/envs/qforte_gpu/lib/gcc/x86_64-conda-linux-gnu/11.2.0/include-fixed;/home/zach_gonzales/anaconda3/envs/qforte_gpu/x86_64-conda-linux-gnu/include;/home/zach_gonzales/anaconda3/envs/qforte_gpu/x86_64-conda-linux-gnu/sysroot/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/home/zach_gonzales/anaconda3/envs/qforte_gpu/lib;/home/zach_gonzales/anaconda3/envs/qforte_gpu/targets/x86_64-linux/lib;/home/zach_gonzales/anaconda3/envs/qforte_gpu/targets/x86_64-linux/lib/stubs;/home/zach_gonzales/anaconda3/envs/qforte_gpu/lib/gcc/x86_64-conda-linux-gnu/11.2.0;/home/zach_gonzales/anaconda3/envs/qforte_gpu/lib/gcc;/home/zach_gonzales/anaconda3/envs/qforte_gpu/x86_64-conda-linux-gnu/lib64;/home/zach_gonzales/anaconda3/envs/qforte_gpu/x86_64-conda-linux-gnu/sysroot/lib64;/home/zach_gonzales/anaconda3/envs/qforte_gpu/x86_64-conda-linux-gnu/sysroot/usr/lib64;/home/zach_gonzales/anaconda3/envs/qforte_gpu/x86_64-conda-linux-gnu/lib;/home/zach_gonzales/anaconda3/envs/qforte_gpu/x86_64-conda-linux-gnu/sysroot/lib;/home/zach_gonzales/anaconda3/envs/qforte_gpu/x86_64-conda-linux-gnu/sysroot/usr/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_AR "/app/bin/ar")
set(CMAKE_MT "")
