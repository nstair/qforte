import numpy as np
import qforte as qf

def test_tensor_gpu_thrust_addition():
    """
    Test the Thrust-based addition method in TensorGPU class.
    This test creates two tensors, transfers them to GPU,
    performs addition using Thrust, and verifies the result.
    """
    print("Testing TensorGPU Thrust Addition...")
    
    # Create two tensors with known values
    shape = [2, 3]
    tensor_a = qf.TensorGPU(shape, "A", False)
    tensor_b = qf.TensorGPU(shape, "B", False)
    
    # Fill tensors with test data
    a_data = np.array([
        [1.0 + 0.5j, 2.0 + 1.0j, 3.0 + 1.5j],
        [4.0 + 2.0j, 5.0 + 2.5j, 6.0 + 3.0j]
    ], dtype=np.complex128)
    
    b_data = np.array([
        [0.5 + 0.1j, 1.0 + 0.2j, 1.5 + 0.3j],
        [2.0 + 0.4j, 2.5 + 0.5j, 3.0 + 0.6j]
    ], dtype=np.complex128)
    
    # Expected result after addition
    expected_result = a_data + b_data
    
    # Fill tensors with data
    for i in range(shape[0]):
        for j in range(shape[1]):
            tensor_a.set([i,j], a_data[i, j])
            tensor_b.set([i,j], b_data[i, j])
    
    # Transfer tensors to GPU
    tensor_a.to_gpu()
    tensor_b.to_gpu()
    
    # Perform addition using Thrust
    tensor_a.addThrust(tensor_b)
    
    # Transfer result back to CPU
    tensor_a.to_cpu()
    
    # Check result
    result = np.array([[tensor_a.get([0,0]), tensor_a.get([0,1]), tensor_a.get([0,2])],
                      [tensor_a.get([1,0]) ,tensor_a.get([1,1]), tensor_a.get([1,2])]], dtype=np.complex128)
    
    # Verify results match expected values
    close_enough = np.allclose(result, expected_result)
    
    if close_enough:
        print("Thrust addition test PASSED!")
        print(f"Original tensor A: \n{a_data}")
        print(f"Original tensor B: \n{b_data}")
        print(f"Result after A.addThrust(B): \n{result}")
    else:
        print("Thrust addition test FAILED!")
        print(f"Original tensor A: \n{a_data}")
        print(f"Original tensor B: \n{b_data}")
        print(f"Expected result: \n{expected_result}")
        print(f"Actual result: \n{result}")
        print(f"Difference: \n{np.abs(result - expected_result)}")
    
    return close_enough

def test_compare_regular_vs_thrust():
    """
    Compare the performance and results of regular GPU addition vs Thrust-based addition.
    """
    print("\nComparing regular vs Thrust-based addition...")
    
    # Create larger tensors for a more meaningful performance test
    shape = [1000, 1000]  # 10x larger in each dimension = 100x more elements
    tensor_regular = qf.TensorGPU(shape, "Regular", False)
    tensor_thrust = qf.TensorGPU(shape, "Thrust", False)
    tensor_other = qf.TensorGPU(shape, "Other", False)
    
    # Fill tensors with random data
    random_data_a = (np.random.random(shape) + 1j * np.random.random(shape)).astype(np.complex128)
    random_data_b = (np.random.random(shape) + 1j * np.random.random(shape)).astype(np.complex128)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            tensor_regular.set([i, j], random_data_a[i, j])
            tensor_thrust.set([i, j], random_data_a[i, j])
            tensor_other.set([i, j], random_data_b[i, j])
    
    # Transfer tensors to GPU
    tensor_regular.to_gpu()
    tensor_thrust.to_gpu()
    tensor_other.to_gpu()
    
    # Warm-up runs to eliminate any initialization overhead
    print("Performing warm-up runs...")
    tensor_regular.add(tensor_other)
    tensor_thrust.addThrust(tensor_other)
    
    # Reset tensors after warm-up
    tensor_regular.to_cpu()
    tensor_thrust.to_cpu()
    tensor_regular.copy_in(qf.TensorGPU(shape, "Regular_reset", False))
    tensor_thrust.copy_in(qf.TensorGPU(shape, "Thrust_reset", False))
    
    # Re-fill tensors with the original data
    for i in range(shape[0]):
        for j in range(shape[1]):
            tensor_regular.set([i, j], random_data_a[i, j])
            tensor_thrust.set([i, j], random_data_a[i, j])
    
    # Transfer back to GPU
    tensor_regular.to_gpu()
    tensor_thrust.to_gpu()
    
    # Perform regular addition with timing
    import time
    num_repeats = 10  # Run multiple times for more reliable timing
    
    # Time CPU addition (using NumPy)
    cpu_times = []
    for _ in range(num_repeats):
        # Create copies of the data for CPU operation
        a_copy = random_data_a.copy()
        b_copy = random_data_b.copy()
        
        # Time the operation
        start_time = time.time()
        cpu_result = a_copy + b_copy  # NumPy addition
        end_time = time.time()
        cpu_times.append(end_time - start_time)
    
    cpu_time = sum(cpu_times) / len(cpu_times)
    
    # Time CPU addition using TensorGPU class (but keeping on CPU)
    cpu_tensor_times = []
    for _ in range(num_repeats):
        # Create tensors that stay on CPU
        tensor_cpu_a = qf.TensorGPU(shape, "CPU_A", False)
        tensor_cpu_b = qf.TensorGPU(shape, "CPU_B", False)
        
        # Fill with the same data
        for i in range(shape[0]):
            for j in range(shape[1]):
                tensor_cpu_a.set([i, j], random_data_a[i, j])
                tensor_cpu_b.set([i, j], random_data_b[i, j])
        
        # Time the CPU tensor operation
        start_time = time.time()
        tensor_cpu_a.add(tensor_cpu_b)  # This will use CPU implementation
        end_time = time.time()
        cpu_tensor_times.append(end_time - start_time)
    
    cpu_tensor_time = sum(cpu_tensor_times) / len(cpu_tensor_times)
    
    # Time regular addition
    regular_times = []

    for _ in range(num_repeats):
        # Create a copy for this iteration
        tensor_temp = qf.TensorGPU(shape, "Temp", True)
        tensor_temp.copy_in_gpu(tensor_regular)
        #tensor_temp.to_gpu()
        
        # Time the operation
        start_time = time.time()
        tensor_temp.add(tensor_other)
        end_time = time.time()
        regular_times.append(end_time - start_time)
    
    regular_time = sum(regular_times) / len(regular_times)
    
    # Time Thrust-based addition
    thrust_times = []
    for _ in range(num_repeats):
        # Create a copy for this iteration
        tensor_temp = qf.TensorGPU(shape, "Temp", True)
        tensor_temp.copy_in_gpu(tensor_thrust)
        #tensor_temp.to_gpu()
        
        # Time the operation
        start_time = time.time()
        tensor_temp.addThrust(tensor_other)
        end_time = time.time()
        thrust_times.append(end_time - start_time)
    
    thrust_time = sum(thrust_times) / len(thrust_times)
    
    # Transfer results back to CPU
    tensor_regular.to_cpu()
    tensor_thrust.to_cpu()
    
    # Compare results
    regular_result = np.empty(shape, dtype=np.complex128)
    thrust_result = np.empty(shape, dtype=np.complex128)
    for i in range(shape[0]):
        for j in range(shape[1]):
            regular_result[i, j] = tensor_regular.get([i, j])
            thrust_result[i, j] = tensor_thrust.get([i, j])
    
    are_equal = np.allclose(regular_result, thrust_result)
    
    print(f"Results are {'equal' if are_equal else 'different'}")
    print(f"Matrix size: {shape[0]}x{shape[1]} = {shape[0]*shape[1]} elements")
    print(f"NumPy CPU addition time: {cpu_time:.6f} seconds (min: {min(cpu_times):.6f}, max: {max(cpu_times):.6f})")
    print(f"TensorGPU CPU addition time: {cpu_tensor_time:.6f} seconds (min: {min(cpu_tensor_times):.6f}, max: {max(cpu_tensor_times):.6f})")
    print(f"Regular GPU addition time: {regular_time:.6f} seconds (min: {min(regular_times):.6f}, max: {max(regular_times):.6f})")
    print(f"Thrust GPU addition time: {thrust_time:.6f} seconds (min: {min(thrust_times):.6f}, max: {max(thrust_times):.6f})")
    
    # Calculate speedups relative to CPU
    numpy_to_regular_speedup = cpu_time / regular_time
    numpy_to_thrust_speedup = cpu_time / thrust_time
    tensor_cpu_to_regular_speedup = cpu_tensor_time / regular_time
    tensor_cpu_to_thrust_speedup = cpu_tensor_time / thrust_time
    
    print(f"\nSpeedup Comparisons:")
    print(f"Regular GPU vs NumPy CPU: {numpy_to_regular_speedup:.2f}x faster")
    print(f"Thrust GPU vs NumPy CPU: {numpy_to_thrust_speedup:.2f}x faster")
    print(f"Regular GPU vs TensorGPU CPU: {tensor_cpu_to_regular_speedup:.2f}x faster")
    print(f"Thrust GPU vs TensorGPU CPU: {tensor_cpu_to_thrust_speedup:.2f}x faster")
    
    # Compare Thrust vs Regular GPU
    if thrust_time < regular_time:
        speedup = regular_time / thrust_time
        print(f"Thrust is {speedup:.2f}x faster than regular GPU implementation!")
    else:
        slowdown = thrust_time / regular_time
        print(f"Thrust is {slowdown:.2f}x slower than regular GPU implementation.")
    
    return are_equal

if __name__ == "__main__":
    print("==== Testing TensorGPU Thrust Implementation ====")
    
    # Test basic thrust addition functionality
    basic_test_passed = test_tensor_gpu_thrust_addition()
    
    # Test and compare performance with regular addition
    comparison_test_passed = test_compare_regular_vs_thrust()
    
    # Overall test result
    if basic_test_passed and comparison_test_passed:
        print("\nAll tests PASSED! The Thrust implementation is working correctly.")
    else:
        print("\nSome tests FAILED. Please check the implementation.")
