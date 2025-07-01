import qforte as qf
import numpy as np

def main():
    """
    Main function to run the Thrust addition test.
    This will create two tensors, perform addition using Thrust,
    and print the results.
    """
    print(qf.thrust_square([1.0,2.0,3.0]))
    thrust_tensor = qf.TensorGPUThrust(3, 1)

if __name__ == '__main__':
    main()
