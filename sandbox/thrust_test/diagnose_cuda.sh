#!/bin/bash
# This script will help diagnose CUDA and Thrust availability in your environment

echo "=== CUDA Environment Diagnostics ==="

# Check CUDA compiler
echo -e "\n== CUDA Compiler =="
which nvcc
if [ $? -eq 0 ]; then
    nvcc --version
else
    echo "nvcc not found in PATH. Please make sure CUDA is properly installed."
fi

# Check CUDA runtime
echo -e "\n== CUDA Runtime =="
if [ -d "/usr/local/cuda" ]; then
    echo "CUDA found in /usr/local/cuda"
    ls -la /usr/local/cuda/include/cuda_runtime.h 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "cuda_runtime.h not found in expected location."
    fi
else
    echo "CUDA not found in standard location (/usr/local/cuda)."
    # Try to find CUDA
    find /usr -name cuda_runtime.h -type f 2>/dev/null
fi

# Check Thrust
echo -e "\n== Thrust Library =="
if [ -d "/usr/local/cuda" ]; then
    ls -la /usr/local/cuda/include/thrust/version.h 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Thrust version.h not found in expected location."
    else
        echo "Thrust found in /usr/local/cuda/include/thrust/"
    fi
else
    # Try to find Thrust
    find /usr -name "thrust" -type d 2>/dev/null
fi

# Check if Thrust headers can be found by the compiler
echo -e "\n== Thrust Headers Accessibility =="
cat > /tmp/thrust_test.cu << EOL
#include <thrust/version.h>
#include <stdio.h>

int main() {
    printf("Thrust version: %d.%d.%d\n", 
           THRUST_MAJOR_VERSION, 
           THRUST_MINOR_VERSION, 
           THRUST_SUBMINOR_VERSION);
    return 0;
}
EOL

nvcc -o /tmp/thrust_version /tmp/thrust_test.cu 2>/dev/null
if [ $? -eq 0 ]; then
    echo "Successfully compiled Thrust test program."
    /tmp/thrust_version
else
    echo "Failed to compile Thrust test program. NVCC output:"
    nvcc -o /tmp/thrust_version /tmp/thrust_test.cu
fi

# Display environment variables related to CUDA
echo -e "\n== Environment Variables =="
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_PATH: $CUDA_PATH"

# Check for Python CUDA packages
echo -e "\n== Python CUDA Packages =="
pip list | grep -E "cuda|thrust|pycuda|numba" || echo "No CUDA-related Python packages found."

echo -e "\n=== End of Diagnostics ==="
