#!/usr/bin/env python3

import sys
import os
import time

# Add the qforte build directory to the path
sys.path.insert(0, '/home/zach_gonzales/qforte/build')

try:
    import qforte
except ImportError as e:
    print(f"Error importing qforte: {e}")
    print("Make sure qforte is built and the path is correct.")
    sys.exit(1)

def minimal_test():
    """Minimal test to verify FCIComputerThrust works"""
    print("Minimal FCIComputerThrust Test")
    print("=" * 30)
    
    # Very simple test case
    nel = 2
    sz = 0  
    norb = 2
    
    print(f"Creating FCIComputerThrust({nel}, {sz}, {norb})...")
    
    try:
        # Create Thrust computer
        fci_thrust = qforte.FCIComputerThrust(nel, sz, norb)
        print("✓ FCIComputerThrust created successfully")
        
        # Test hartree_fock
        print("Testing hartree_fock()...")
        fci_thrust.hartree_fock()
        print("✓ hartree_fock() completed")
        
        # Test get_state
        print("Testing get_state()...")
        state = fci_thrust.get_state()
        print(f"✓ get_state() returned state with norm: {state.norm():.6f}")
        
        # Test str method
        print("Testing str() method...")
        try:
            state_str = fci_thrust.str()
            print(f"✓ str() method works, length: {len(state_str)}")
        except Exception as e:
            print(f"✗ str() method failed: {e}")
        
        # Test get_state_deep
        print("Testing get_state_deep()...")
        try:
            state_deep = fci_thrust.get_state_deep()
            print(f"✓ get_state_deep() returned state with norm: {state_deep.norm():.6f}")
        except Exception as e:
            print(f"✗ get_state_deep() failed: {e}")
        
        # Test simple operator
        print("Testing simple apply_sqop()...")
        sqop = qforte.SQOperator()
        sqop.add_term(0.1, [0], [1])
        
        start_time = time.time()
        fci_thrust.apply_sqop(sqop)
        elapsed = time.time() - start_time
        
        print(f"✓ apply_sqop() completed in {elapsed:.6f} seconds")
        
        # Check final state
        final_state = fci_thrust.get_state()
        print(f"✓ Final state norm: {final_state.norm():.6f}")
        
        print("\n✓ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def comparison_test():
    """Compare FCIComputerThrust with FCIComputerGPU"""
    print("\nComparison Test")
    print("=" * 30)
    
    nel, sz, norb = 2, 0, 2
    
    try:
        # Create both computers
        print("Creating computers...")
        fci_gpu = qforte.FCIComputerGPU(nel, sz, norb)
        fci_thrust = qforte.FCIComputerThrust(nel, sz, norb)
        
        # Initialize both
        fci_gpu.hartree_fock()
        fci_thrust.hartree_fock()
        
        # Create operator
        sqop = qforte.SQOperator()
        sqop.add_term(0.1, [0], [1])
        
        # Apply to both
        print("Applying operator to both...")
        
        start_time = time.time()
        fci_gpu.apply_sqop(sqop)
        gpu_time = time.time() - start_time
        
        start_time = time.time()
        fci_thrust.apply_sqop(sqop)
        thrust_time = time.time() - start_time
        
        print(f"GPU time: {gpu_time:.6f} seconds")
        print(f"Thrust time: {thrust_time:.6f} seconds")
        
        # Compare results
        state_gpu = fci_gpu.get_state()
        state_thrust = fci_thrust.get_state()
        
        gpu_data = state_gpu.data()
        thrust_data = state_thrust.data()
        
        max_diff = 0.0
        for i in range(len(gpu_data)):
            diff = abs(gpu_data[i] - thrust_data[i])
            if diff > max_diff:
                max_diff = diff
        
        print(f"Maximum difference: {max_diff:.2e}")
        
        if max_diff < 1e-10:
            print("✓ Results match!")
            return True
        else:
            print("✗ Results differ!")
            return False
            
    except Exception as e:
        print(f"✗ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test"""
    success = True
    
    if not minimal_test():
        success = False
    
    if not comparison_test():
        success = False
    
    print("\n" + "=" * 30)
    if success:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print("=" * 30)

if __name__ == "__main__":
    main()
