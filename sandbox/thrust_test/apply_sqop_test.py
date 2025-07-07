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

def create_test_sqoperator(norb=4):
    """Create a simple test SQOperator for testing"""
    sqop = qforte.SQOperator()
    
    # Add some single excitation terms
    # a†_0 a_1 - a†_1 a_0 (single excitation)
    sqop.add_term(0.1, [0], [1])
    sqop.add_term(-0.1, [1], [0])
    
    # Add some double excitation terms
    # a†_0 a†_2 a_3 a_1 - a†_1 a†_3 a_2 a_0 (double excitation)
    sqop.add_term(0.05, [0, 2], [1, 3])
    sqop.add_term(-0.05, [1, 3], [0, 2])
    
    return sqop

def test_fci_computer_basic():
    """Test basic functionality of FCIComputerThrust vs FCIComputerGPU"""
    print("=" * 60)
    print("Testing FCIComputerThrust vs FCIComputerGPU")
    print("=" * 60)
    
    # Test parameters
    nel = 4      # 4 electrons
    sz = 0       # singlet
    norb = 4     # 4 orbitals
    
    print(f"Test parameters: nel={nel}, sz={sz}, norb={norb}")
    
    # Create both computers
    print("\nCreating FCIComputerGPU...")
    fci_gpu = qforte.FCIComputerGPU(nel, sz, norb)
    
    print("Creating FCIComputerThrust...")
    fci_thrust = qforte.FCIComputerThrust(nel, sz, norb)
    
    # Initialize both to Hartree-Fock state
    print("\nInitializing to Hartree-Fock state...")
    fci_gpu.hartree_fock()
    fci_thrust.hartree_fock()
    
    # Get initial states
    state_gpu_initial = fci_gpu.get_state()
    state_thrust_initial = fci_thrust.get_state()
    
    print(f"Initial GPU state norm: {state_gpu_initial.norm():.6f}")
    print(f"Initial Thrust state norm: {state_thrust_initial.norm():.6f}")
    
    # Create test operator
    print("\nCreating test SQOperator...")
    sqop = create_test_sqoperator(norb)
    print(f"SQOperator: {sqop.str()}")
    
    return fci_gpu, fci_thrust, sqop

def test_apply_sqop_comparison():
    """Test apply_sqop method and compare results"""
    print("\n" + "=" * 60)
    print("Testing apply_sqop comparison")
    print("=" * 60)
    
    # Setup
    fci_gpu, fci_thrust, sqop = test_fci_computer_basic()
    
    # Test 1: Apply SQOperator and compare results
    print("\nApplying SQOperator to both computers...")
    
    # Time GPU version
    start_time = time.time()
    fci_gpu.apply_sqop(sqop)
    gpu_time = time.time() - start_time
    
    # Time Thrust version  
    start_time = time.time()
    fci_thrust.apply_sqop(sqop)
    thrust_time = time.time() - start_time
    
    print(f"GPU apply_sqop time: {gpu_time:.6f} seconds")
    print(f"Thrust apply_sqop time: {thrust_time:.6f} seconds")
    print(f"Speedup: {gpu_time/thrust_time:.2f}x")
    
    # Get final states
    state_gpu_final = fci_gpu.get_state()
    state_thrust_final = fci_thrust.get_state()
    
    print(f"\nFinal GPU state norm: {state_gpu_final.norm():.6f}")
    print(f"Final Thrust state norm: {state_thrust_final.norm():.6f}")
    
    # Compare states by computing their difference
    print("\nComparing final states...")
    
    # Convert to numpy arrays for comparison
    try:
        # Get the state data (this might require copying to CPU first)
        gpu_data = state_gpu_final.data()
        thrust_data = state_thrust_final.data()
        
        # Calculate difference
        max_diff = 0.0
        for i in range(len(gpu_data)):
            diff = abs(gpu_data[i] - thrust_data[i])
            if diff > max_diff:
                max_diff = diff
        
        print(f"Maximum difference between states: {max_diff:.2e}")
        
        # Check if states are essentially equal
        tolerance = 1e-10
        if max_diff < tolerance:
            print("✓ States match within tolerance!")
            return True
        else:
            print("✗ States differ significantly!")
            return False
            
    except Exception as e:
        print(f"Error comparing states: {e}")
        return False

def test_timing_detailed():
    """Test detailed timing using internal timers"""
    print("\n" + "=" * 60)
    print("Testing detailed timing")
    print("=" * 60)
    
    # Setup
    fci_gpu, fci_thrust, sqop = test_fci_computer_basic()
    
    # Clear any existing timings
    fci_gpu.clear_timings()
    fci_thrust.clear_timings()
    
    # Apply operators multiple times for better timing statistics
    n_iterations = 5
    print(f"\nRunning {n_iterations} iterations for timing...")
    
    for i in range(n_iterations):
        print(f"Iteration {i+1}/{n_iterations}")
        
        # Reset to HF state before each iteration
        fci_gpu.hartree_fock()
        fci_thrust.hartree_fock()
        
        # Apply operator
        fci_gpu.apply_sqop(sqop)
        fci_thrust.apply_sqop(sqop)
    
    # Get timing results
    print("\nTiming results:")
    print("-" * 40)
    
    try:
        gpu_timings = fci_gpu.get_timings()
        thrust_timings = fci_thrust.get_timings()
        
        print("GPU timings:")
        for timing in gpu_timings:
            print(f"  {timing[0]}: {timing[1]:.6f} seconds")
        
        print("\nThrust timings:")
        for timing in thrust_timings:
            print(f"  {timing[0]}: {timing[1]:.6f} seconds")
            
    except Exception as e:
        print(f"Error getting detailed timings: {e}")

def test_different_operators():
    """Test with different types of operators"""
    print("\n" + "=" * 60)
    print("Testing different operator types")
    print("=" * 60)
    
    nel, sz, norb = 4, 0, 4
    
    # Test different operators
    operators = []
    
    # 1. Single excitation only
    sqop1 = qforte.SQOperator()
    sqop1.add_term(0.1, [0], [1])
    operators.append(("Single excitation", sqop1))
    
    # 2. Double excitation only
    sqop2 = qforte.SQOperator()
    sqop2.add_term(0.05, [0, 2], [1, 3])
    operators.append(("Double excitation", sqop2))
    
    # 3. Mixed terms
    sqop3 = qforte.SQOperator()
    sqop3.add_term(0.1, [0], [1])
    sqop3.add_term(0.05, [0, 2], [1, 3])
    sqop3.add_term(-0.03, [1, 3], [0, 2])
    operators.append(("Mixed terms", sqop3))
    
    results = []
    
    for name, sqop in operators:
        print(f"\nTesting {name}...")
        
        # Create fresh computers
        fci_gpu = qforte.FCIComputerGPU(nel, sz, norb)
        fci_thrust = qforte.FCIComputerThrust(nel, sz, norb)
        
        # Initialize to HF
        fci_gpu.hartree_fock()
        fci_thrust.hartree_fock()
        
        # Apply operator
        fci_gpu.apply_sqop(sqop)
        fci_thrust.apply_sqop(sqop)
        
        # Compare results
        state_gpu = fci_gpu.get_state()
        state_thrust = fci_thrust.get_state()
        
        gpu_data = state_gpu.data()
        thrust_data = state_thrust.data()
        
        max_diff = max(abs(gpu_data[i] - thrust_data[i]) for i in range(len(gpu_data)))
        
        print(f"  Max difference: {max_diff:.2e}")
        results.append((name, max_diff))
    
    print(f"\nSummary of operator tests:")
    for name, diff in results:
        status = "✓" if diff < 1e-10 else "✗"
        print(f"  {status} {name}: {diff:.2e}")

def main():
    """Main test function"""
    print("FCIComputerThrust vs FCIComputerGPU Test Suite")
    print("=" * 60)
    
    try:
        # Run all tests
        success = True
        
        # Basic functionality test
        if not test_apply_sqop_comparison():
            success = False
        
        # Detailed timing test
        test_timing_detailed()
        
        # Different operator types
        test_different_operators()
        
        print("\n" + "=" * 60)
        if success:
            print("All tests completed successfully! ✓")
        else:
            print("Some tests failed! ✗")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
