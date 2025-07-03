import qforte as qf
import numpy as np
import pytest


def test_apply_tensor_spin_1bdy():
    complex_array = np.array([[1.0+0j, 2.0+0j, 5.0+0j, 7.0+0j], 
                              [3.0+0j, 4.0+0j, 6.0+0j, 8.0+0j], 
                              [9.0+0j, 10.0+0j, 11.0+0j, 12.0+0j], 
                              [13.0+0j, 14.0+0j, 15.0+0j, 16.0+0j]])

    fciThrust = qf.FCIComputerThrust(2, 0, 2, False)
    fciThrust.set_element([0, 0], np.complex128(1.0 + 0.0j))
    fciThrust.set_element([1, 0], np.complex128(2.0 + 0.0j))
    h1e = qf.TensorGPUThrust([4, 4], "h1e", False)
    h1e.fill_from_nparray(complex_array.flatten().tolist(), [4, 4])
    fciThrust.apply_tensor_spin_1bdy(h1e, 2)

    # Compare with GPU implementation
    fciGPU = qf.FCIComputerGPU(2, 0, 2)
    fciGPU.set_element([0, 0], np.complex128(1.0 + 0.0j))
    fciGPU.set_element([1, 0], np.complex128(2.0 + 0.0j))
    h1e_gpu = qf.TensorGPU([4, 4], "h1e_gpu", False)
    h1e_gpu.fill_from_nparray(complex_array.flatten().tolist(), [4, 4])
    fciGPU.apply_tensor_spin_1bdy(h1e_gpu, 2)

    print("Thrust results:")
    print(fciThrust)
    print("GPU results:")
    print(fciGPU)


if __name__ == "__main__":
    test_apply_tensor_spin_1bdy()