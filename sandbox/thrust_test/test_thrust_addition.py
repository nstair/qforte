#!/usr/bin/env python

"""
This script tests the Thrust-based tensor addition implementation in QForte.
It creates tensors on the GPU, performs addition using Thrust, and verifies the results.
"""

import sys
import os
import time
import numpy as np
import qforte as qf

def main():
    print("Testing QForte Tensor GPU Thrust Implementation")
    print("==============================================")
    
    # Test 1: Basic verification of thrust addition
    test_basic_thrust_addition()
    
    # Test 2: Performance comparison between regular and thrust addition
    test_performance_comparison()

def test_basic_thrust_addition():
    print("\nTest 1: Basic Thrust Addition Verification")
    print("----------------------------------------")
    
    # Create test tensors
    size = 10  # Small size for verification
    shape = [size, size]
    
    # Create tensors
    tensor_a = qf.TensorGPU(shape, "A")
    tensor_b = qf.TensorGPU(shape, "B")
    
    # Fill with simple test pattern
    for i in range(size):
        for j in range(size):
            index = i * size + j
            tensor_a.h_data()[index] = complex(i + 1, j + 1)
            tensor_b.h_data()[index] = complex(j + 1, i + 1)
    
    # Make a copy of A for verification
    expected_result = np.array(tensor_a.h_data())
    b_values = np.array(tensor_b.h_data())
    expected_result = expected_result + b_values
    
    # Transfer to GPU
    tensor_a.to_gpu()
    tensor_b.to_gpu()
    
    # Perform addition using thrust
    tensor_a.addThrust(tensor_b)
    
    # Get result back
    tensor_a.to_cpu()
    result = np.array(tensor_a.h_data())
    
    # Verify
    is_correct = np.allclose(result, expected_result)
    
    if is_correct:
        print("✅ Thrust addition produces correct results!")
        
        # Print example values for verification
        print("\nSample values from the test:")
        for i in range(3):
            for j in range(3):
                index = i * size + j
                print(f"A[{i},{j}] + B[{i},{j}] = {expected_result[index]}")
    else:
        print("❌ Thrust addition test FAILED!")
        
        # Print differences
        diff = np.abs(result - expected_result)
        max_diff_idx = np.argmax(diff)
        i, j = max_diff_idx // size, max_diff_idx % size
        
        print(f"Maximum difference at [{i},{j}]:")
        print(f"Expected: {expected_result[max_diff_idx]}")
        print(f"Actual: {result[max_diff_idx]}")
        print(f"Difference: {diff[max_diff_idx]}")

def test_performance_comparison():
    print("\nTest 2: Performance Comparison")
    print("----------------------------")
    
    # Use larger tensors for performance testing
    sizes = [100, 500, 1000]
    
    print(f"{'Size':<10} {'Regular (ms)':<15} {'Thrust (ms)':<15} {'Speedup':<10}")
    print("-" * 50)
    
    for size in sizes:
        shape = [size, size]
        
        # Create tensors for both methods
        tensor_regular = qf.TensorGPU(shape, "Regular")
        tensor_thrust = qf.TensorGPU(shape, "Thrust")
        tensor_add = qf.TensorGPU(shape, "Add")
        
        # Fill with random data
        total_size = size * size
        random_data = np.random.random(total_size) + 1j * np.random.random(total_size)
        add_data = np.random.random(total_size) + 1j * np.random.random(total_size)
        
        for i in range(total_size):
            tensor_regular.h_data()[i] = random_data[i]
            tensor_thrust.h_data()[i] = random_data[i]
            tensor_add.h_data()[i] = add_data[i]
        
        # Transfer to GPU
        tensor_regular.to_gpu()
        tensor_thrust.to_gpu()
        tensor_add.to_gpu()
        
        # Time regular addition
        start_time = time.time()
        tensor_regular.add(tensor_add)
        regular_time = (time.time() - start_time) * 1000  # ms
        
        # Time thrust addition
        start_time = time.time()
        tensor_thrust.addThrust(tensor_add)
        thrust_time = (time.time() - start_time) * 1000  # ms
        
        # Calculate speedup
        speedup = regular_time / thrust_time if thrust_time > 0 else float('inf')
        
        print(f"{size*size:<10} {regular_time:<15.2f} {thrust_time:<15.2f} {speedup:<10.2f}x")
        
        # Verify results match
        tensor_regular.to_cpu()
        tensor_thrust.to_cpu()
        
        regular_result = np.array(tensor_regular.h_data())
        thrust_result = np.array(tensor_thrust.h_data())
        
        is_equal = np.allclose(regular_result, thrust_result)
        if not is_equal:
            print(f"⚠️ Warning: Results for size {size}x{size} do not match!")

if __name__ == "__main__":
    main()
