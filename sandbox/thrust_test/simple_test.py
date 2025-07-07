#!/usr/bin/env python3

import sys
import os
import numpy as np
import time

# Add the qforte build directory to the path
sys.path.insert(0, '/home/zach_gonzales/qforte/build')

try:
    import qforte
except ImportError as e:
    print(f"Error importing qforte: {e}")
    print("Make sure qforte is built and the path is correct.")
    sys.exit(1)

def simple_apply_sqop_test():
    """Simple test of apply_sqop functionality"""
    print("Simple FCIComputerThrust vs FCIComputerGPU Test")
    print("=" * 50)
    
    # Small test case
    nel = 2      # 2 electrons
    sz = 0       # singlet
    norb = 2     # 2 orbitals
    
    print(f"Parameters: nel={nel}, sz={sz}, norb={norb}")
    
    try:
        # Create computers
        print("\nCreating computers...")
        fci_gpu = qforte.FCIComputerGPU(nel, sz, norb)
        fci_thrust = qforte.FCIComputerThrust(nel, sz, norb)
        
        # Initialize to HF
        print("Initializing to Hartree-Fock...")
        fci_gpu.hartree_fock()
        fci_thrust.hartree_fock()
        
        # Create simple operator: a†_0 a_1
        print("Creating simple SQOperator...")
        sqop = qforte.SQOperator()
        sqop.add_term(0.1, [0], [1])  # Single excitation
        
        print(f"Operator: {sqop.str()}")
        
        # Apply operator
        print("\nApplying operator...")
        start_time = time.time()
        fci_gpu.apply_sqop(sqop)
        gpu_time = time.time() - start_time
        
        start_time = time.time()
        fci_thrust.apply_sqop(sqop)
        thrust_time = time.time() - start_time
        
        print(f"GPU time: {gpu_time:.6f} seconds")
        print(f"Thrust time: {thrust_time:.6f} seconds")
        
        # Get states
        state_gpu = fci_gpu.get_state()
        state_thrust = fci_thrust.get_state()
        
        print(f"\nGPU state norm: {state_gpu.norm():.6f}")
        print(f"Thrust state norm: {state_thrust.norm():.6f}")
        
        # Compare states
        gpu_data = state_gpu.data()
        thrust_data = state_thrust.data()
        
        print(f"\nState comparison:")
        print(f"GPU state size: {len(gpu_data)}")
        print(f"Thrust state size: {len(thrust_data)}")
        
        if len(gpu_data) == len(thrust_data):
            max_diff = 0.0
            for i in range(len(gpu_data)):
                diff = abs(gpu_data[i] - thrust_data[i])
                if diff > max_diff:
                    max_diff = diff
                    
            print(f"Maximum difference: {max_diff:.2e}")
            
            if max_diff < 1e-10:
                print("✓ Test PASSED: States match!")
                return True
            else:
                print("✗ Test FAILED: States differ!")
                return False
        else:
            print("✗ Test FAILED: State sizes differ!")
            return False
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def performance_test():
    """Performance comparison test"""
    print("\n" + "=" * 50)
    print("Performance Test")
    print("=" * 50)
    
    # Larger test case
    nel = 4
    sz = 0
    norb = 4
    
    print(f"Parameters: nel={nel}, sz={sz}, norb={norb}")
    
    try:
        # Create computers
        fci_gpu = qforte.FCIComputerGPU(nel, sz, norb)
        fci_thrust = qforte.FCIComputerThrust(nel, sz, norb)
        
        # Create more complex operator
        sqop = qforte.SQOperator()
        sqop.add_term(0.1, [0], [1])
        sqop.add_term(-0.1, [1], [0])
        sqop.add_term(0.05, [0, 2], [1, 3])
        sqop.add_term(-0.05, [1, 3], [0, 2])
        
        # Run multiple iterations
        n_iterations = 10
        gpu_times = []
        thrust_times = []
        
        print(f"\nRunning {n_iterations} iterations...")
        
        for i in range(n_iterations):
            # Reset states
            fci_gpu.hartree_fock()
            fci_thrust.hartree_fock()
            
            # Time GPU
            start_time = time.time()
            fci_gpu.apply_sqop(sqop)
            gpu_times.append(time.time() - start_time)
            
            # Time Thrust
            start_time = time.time()
            fci_thrust.apply_sqop(sqop)
            thrust_times.append(time.time() - start_time)
            
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{n_iterations} iterations")
        
        # Calculate statistics
        avg_gpu_time = sum(gpu_times) / len(gpu_times)
        avg_thrust_time = sum(thrust_times) / len(thrust_times)
        
        print(f"\nPerformance Results:")
        print(f"Average GPU time: {avg_gpu_time:.6f} seconds")
        print(f"Average Thrust time: {avg_thrust_time:.6f} seconds")
        print(f"Speedup: {avg_gpu_time/avg_thrust_time:.2f}x")
        
        # Verify final states still match
        state_gpu = fci_gpu.get_state()
        state_thrust = fci_thrust.get_state()
        
        gpu_data = state_gpu.data()
        thrust_data = state_thrust.data()
        
        max_diff = max(abs(gpu_data[i] - thrust_data[i]) for i in range(len(gpu_data)))
        print(f"Final state difference: {max_diff:.2e}")
        
        return max_diff < 1e-10
        
    except Exception as e:
        print(f"Error during performance test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("FCIComputerThrust Test Suite")
    print("=" * 50)
    
    success = True
    
    # Run simple test
    if not simple_apply_sqop_test():
        success = False
    
    # Run performance test
    if not performance_test():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print("=" * 50)

if __name__ == "__main__":
    main()
