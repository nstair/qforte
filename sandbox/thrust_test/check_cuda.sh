#!/bin/bash

echo "=== CUDA Installation Check ==="

# Check if nvcc is available
if command -v nvcc &> /dev/null; then
    echo "✓ nvcc found: $(which nvcc)"
    echo "CUDA version: $(nvcc --version | grep release)"
    
    # Get CUDA toolkit root directory
    CUDA_ROOT=$(dirname $(dirname $(which nvcc)))
    echo "CUDA root directory: $CUDA_ROOT"
    
    # Check for thrust headers
    if [ -f "$CUDA_ROOT/include/thrust/version.h" ]; then
        echo "✓ Thrust headers found in $CUDA_ROOT/include/thrust/"
        echo "Thrust version: $(grep 'THRUST_VERSION ' $CUDA_ROOT/include/thrust/version.h | head -1)"
    else
        echo "✗ Thrust headers not found in $CUDA_ROOT/include/thrust/"
    fi
else
    echo "✗ nvcc not found in PATH"
fi

# Check conda environment thrust
if [ ! -z "$CONDA_PREFIX" ]; then
    echo ""
    echo "=== Conda Environment Check ==="
    echo "Conda prefix: $CONDA_PREFIX"
    
    if [ -d "$CONDA_PREFIX/include/thrust" ]; then
        echo "✓ Thrust found in conda environment: $CONDA_PREFIX/include/thrust"
    else
        echo "✗ Thrust not found in conda environment"
    fi
    
    # Check for cuda-cccl package which includes thrust
    if [ -d "$CONDA_PREFIX/include/cuda/std" ] || [ -d "$CONDA_PREFIX/include/thrust" ]; then
        echo "✓ CUDA libraries found in conda environment"
    fi
fi

# Check system-wide installations
for cuda_dir in /usr/local/cuda /opt/cuda /usr/cuda; do
    if [ -d "$cuda_dir" ]; then
        echo ""
        echo "✓ CUDA directory found: $cuda_dir"
        
        if [ -f "$cuda_dir/include/thrust/version.h" ]; then
            echo "  ✓ Thrust headers found in $cuda_dir/include/thrust/"
        else
            echo "  ✗ Thrust headers not found in $cuda_dir/include/thrust/"
        fi
    fi
done

echo ""
echo "=== CMake Variable Suggestions ==="
if command -v nvcc &> /dev/null; then
    CUDA_ROOT=$(dirname $(dirname $(which nvcc)))
    echo "Add this to your CMakeLists.txt:"
    echo "set(CUDA_TOOLKIT_ROOT_DIR \"$CUDA_ROOT\")"
    echo "include_directories(\"\${CUDA_TOOLKIT_ROOT_DIR}/include\")"
fi