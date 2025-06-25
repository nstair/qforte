import numpy as np
import sys
import os
import time

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Add the parent directory to the path (in case the module is built there)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    import thrust_test_py
    print("Successfully imported thrust_test_py module")
except ImportError as e:
    print(f"Error importing thrust_test_py: {e}")
    print("Current sys.path:", sys.path)
    print("Files in current directory:", os.listdir(current_dir))
    if os.path.exists(parent_dir):
        print("Files in parent directory:", os.listdir(parent_dir))
    sys.exit(1)

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
