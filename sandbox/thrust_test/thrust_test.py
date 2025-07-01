import qforte as qf
import numpy as np


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


def main():
    print("Testing QForte Tensor GPU Thrust Implementation")
    print("==============================================")
    
    # Test 1: Basic verification of thrust addition
    test_basic_thrust_addition()