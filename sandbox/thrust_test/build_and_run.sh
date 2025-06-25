#!/bin/bash
# Script to build and run the Thrust test

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build
make -j4

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful. Running test..."
    ./thrust_test
else
    echo "Build failed. Please check the errors above."
fi
