import qforte as qf
import numpy as np
import time

# Try to import matplotlib, but continue if it's not available
try:
    import matplotlib.pyplot as plt
    has_matplotlib = True
except ImportError:
    has_matplotlib = False
    print("Warning: matplotlib not found. No plots will be generated.")

def test_tensor_add(tensor_type, size, num_iterations=100):
    """
    Test the speed of tensor addition for a specific tensor type.
    
    Args:
        tensor_type (str): Type of tensor to test ('GPU' or 'GPUThrust')
        size (int): Size of the tensor (number of elements)
        num_iterations (int): Number of iterations for timing
    
    Returns:
        float: Average time per addition operation in milliseconds
    """
    # Create tensors based on type
    if tensor_type == 'GPU':
        tensor1 = qf.TensorGPU([size, 1], "tensor1", False)
        tensor2 = qf.TensorGPU([size, 1], "tensor2", False)
    else:  # GPUThrust
        tensor1 = qf.TensorGPUThrust([size, 1])
        tensor2 = qf.TensorGPUThrust([size, 1])
    
    # Fill tensors with random data
    data1 = np.random.rand(size).astype(np.float32)
    data2 = np.random.rand(size).astype(np.float32)
    
    tensor1.fill_from_nparray(data1, [size, 1])
    tensor2.fill_from_nparray(data2, [size, 1])
    
    # Move to GPU
    tensor1.to_gpu()
    tensor2.to_gpu()
    
    # Warm-up
    for _ in range(5):
        if tensor_type == 'GPU':
            tensor1.add2(tensor2)
        else:
            tensor1.add(tensor2)
    
    # Timing
    start_time = time.time()
    for _ in range(num_iterations):
        if tensor_type == 'GPU':
            tensor1.add2(tensor2)
        else:
            tensor1.add(tensor2)
    end_time = time.time()
    
    # Move back to CPU for validation
    tensor1.to_cpu()
    
    # Calculate average time in milliseconds
    avg_time_ms = (end_time - start_time) * 1000 / num_iterations
    
    return avg_time_ms

def main():
    """
    Main function to run the comparison test between TensorGPU and TensorGPUThrust.
    """
    print("Comparing TensorGPU vs TensorGPUThrust for add function")
    print("=" * 60)
    
    # Test with different sizes
    sizes = [10, 100, 1000, 10000, 100000, 1000000]
    iterations = 100
    
    # Results storage
    gpu_times = []
    thrust_times = []
    
    for size in sizes:
        print(f"\nTesting with tensor size: {size}")
        
        # Test TensorGPU
        gpu_time = test_tensor_add('GPU', size, iterations)
        gpu_times.append(gpu_time)
        print(f"TensorGPU average time: {gpu_time:.4f} ms")
        
        # Test TensorGPUThrust
        thrust_time = test_tensor_add('GPUThrust', size, iterations)
        thrust_times.append(thrust_time)
        print(f"TensorGPUThrust average time: {thrust_time:.4f} ms")
        
        # Performance comparison
        speedup = gpu_time / thrust_time if thrust_time > 0 else float('inf')
        print(f"Speedup (TensorGPU / TensorGPUThrust): {speedup:.2f}x")
    
    # Plot results if matplotlib is available
    if has_matplotlib:
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, gpu_times, 'o-', label='TensorGPU')
        plt.plot(sizes, thrust_times, 's-', label='TensorGPUThrust')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Tensor Size (elements)')
        plt.ylabel('Average Time per Addition (ms)')
        plt.title('Performance Comparison: TensorGPU vs TensorGPUThrust')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig('tensor_performance_comparison.png')
        print("\nPerformance plot saved as 'tensor_performance_comparison.png'")
    else:
        # Print tabular results instead
        print("\nSummary of Results:")
        print(f"{'Size':<10} {'TensorGPU (ms)':<15} {'TensorGPUThrust (ms)':<20} {'Speedup':<10}")
        print("-" * 55)
        for i, size in enumerate(sizes):
            print(f"{size:<10} {gpu_times[i]:<15.4f} {thrust_times[i]:<20.4f} {gpu_times[i]/thrust_times[i]:<10.2f}x")

if __name__ == '__main__':
    main()
