#!/bin/bash
# Script to build the Python extension module

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Clean previous builds
rm -rf build dist *.egg-info

# Build the extension
python setup.py build_ext --inplace

# Test if the build was successful
if [ $? -eq 0 ]; then
    echo "Build successful. Creating test script..."
    
    # Create a test script
    cat > test_thrust_py.py << EOL
import numpy as np
import thrust_test_py
import time

def test_thrust():
    print("Testing Thrust/CUDA Python extension...")
    
    # Create test arrays
    size = 1000000
    a = np.array([complex(i * 0.1, i * 0.2) for i in range(size)], dtype=np.complex128)
    b = np.array([complex(i * 0.3, i * 0.4) for i in range(size)], dtype=np.complex128)
    
    # Test Python implementation
    start = time.time()
    cpu_result = a + b
    python_time = time.time() - start
    print(f"Python time: {python_time:.6f} seconds")
    
    # Test Thrust implementation
    start = time.time()
    thrust_result = thrust_test_py.add_complex_numpy(a, b)
    thrust_time = time.time() - start
    print(f"Thrust time: {thrust_time:.6f} seconds")
    
    # Verify results
    diff = np.abs(cpu_result - thrust_result).max()
    print(f"Maximum difference: {diff}")
    print(f"Results match: {diff < 1e-10}")
    
    # Show speedup
    print(f"Speedup: {python_time / thrust_time:.2f}x")
    
    # Test other operations
    print("\nTesting other operations...")
    # Convert small arrays for testing
    small_a = [complex(i * 0.1, i * 0.2) for i in range(5)]
    small_b = [complex(i * 0.3, i * 0.4) for i in range(5)]
    
    # Test multiply
    mult_result = thrust_test_py.multiply_complex(small_a, small_b)
    print("Multiplication result (first 5 elements):")
    for i in range(min(5, len(mult_result))):
        expected = small_a[i] * small_b[i]
        print(f"  {small_a[i]} * {small_b[i]} = {mult_result[i]} (Expected: {expected})")
    
    # Test scale
    scalar = complex(2.0, 1.0)
    scale_result = thrust_test_py.scale_complex(small_a, scalar)
    print(f"\nScaling result (scalar: {scalar}):")
    for i in range(min(5, len(scale_result))):
        expected = small_a[i] * scalar
        print(f"  {small_a[i]} * {scalar} = {scale_result[i]} (Expected: {expected})")

if __name__ == "__main__":
    test_thrust()
EOL
    
    echo "Running test script..."
    python test_thrust_py.py
else
    echo "Build failed. Please check the errors above."
fi
