import qforte
import numpy as np

def create_test_sqoperator(norb=4):
    """Create a simple test SQOperator for testing—terms always have matching alpha/beta creators/annihilators."""
    sqop = qforte.SQOperator()

    # Alpha-only single excitation: a†_2 a_0 - a†_0 a_2
    sqop.add_term(0.1, [2], [0])
    sqop.add_term(-0.1, [0], [2])

    # Beta-only single excitation: a†_3 a_1 - a†_1 a_3
    sqop.add_term(0.1, [3], [1])
    sqop.add_term(-0.1, [1], [3])

    # Alpha-beta double excitation: a†_2 a†_3 a_0 a_1 - a†_0 a†_1 a_2 a_3
    sqop.add_term(0.05, [2, 3], [0, 1])
    sqop.add_term(-0.05, [0, 1], [2, 3])
    
    return sqop

if __name__ == "__main__":
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

    # Create test operator
    print("\nCreating test SQOperator...")
    sqop = create_test_sqoperator(norb)
    print(f"SQOperator: {sqop.str()}")

    fci_gpu.to_gpu()
    fci_thrust.to_gpu()

    fci_thrust.apply_sqop(sqop)
    fci_gpu.apply_sqop(sqop)
    