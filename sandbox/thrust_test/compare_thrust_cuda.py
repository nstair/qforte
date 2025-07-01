#!/usr/bin/env python3

import qforte
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

def time_operation(func, iterations=10):
    """Time a function execution over multiple iterations and return average time."""
    times = []
    for _ in range(iterations):
        start = time.time()
        func()
        end = time.time()
        times.append(end - start)
    return np.mean(times), np.std(times)

def benchmark_addition(sizes, iterations=10):
    """Benchmark tensor addition across different implementations and sizes."""
    results = {
        'Size': [],
        'CPU Time (ms)': [],
        'CPU Std (ms)': [],
        'GPU Legacy Time (ms)': [],
        'GPU Legacy Std (ms)': [],
        'GPU Thrust Time (ms)': [],
        'GPU Thrust Std (ms)': []
    }
    
    for size in sizes:
        print(f"Benchmarking addition for size {size}...")
        results['Size'].append(size)
        
        # Create test data
        shape = [size, size]  # Square matrices for simplicity
        data1 = np.random.random((size, size)) + 1j * np.random.random((size, size))
        data2 = np.random.random((size, size)) + 1j * np.random.random((size, size))
        
        # CPU Tensor benchmark
        tensor1 = qforte.Tensor(shape)
        tensor2 = qforte.Tensor(shape)
        
        # Fill with test data
        tensor1.set_data(data1.reshape(-1))
        tensor2.set_data(data2.reshape(-1))
        
        # Time CPU addition
        def cpu_add():
            result = tensor1.clone()
            result.add(tensor2)
            return result
        
        cpu_time, cpu_std = time_operation(cpu_add, iterations)
        results['CPU Time (ms)'].append(cpu_time * 1000)  # Convert to ms
        results['CPU Std (ms)'].append(cpu_std * 1000)
        
        # GPU Legacy benchmark
        gpu_tensor1 = qforte.TensorGPU(shape)
        gpu_tensor2 = qforte.TensorGPU(shape)
        
        # Fill with test data
        gpu_tensor1.fill_from_nparray(data1.reshape(-1), shape)
        gpu_tensor2.fill_from_nparray(data2.reshape(-1), shape)
        
        # Move to GPU
        gpu_tensor1.to_gpu()
        gpu_tensor2.to_gpu()
        
        # Time GPU legacy addition
        def gpu_legacy_add():
            result = gpu_tensor1.clone()
            result.add2(gpu_tensor2)
            result.to_cpu()  # Include transfer back to ensure fair comparison
            return result
        
        gpu_legacy_time, gpu_legacy_std = time_operation(gpu_legacy_add, iterations)
        results['GPU Legacy Time (ms)'].append(gpu_legacy_time * 1000)
        results['GPU Legacy Std (ms)'].append(gpu_legacy_std * 1000)
        
        # Time GPU Thrust addition
        def gpu_thrust_add():
            result = gpu_tensor1.clone()
            result.addThrust(gpu_tensor2)
            result.to_cpu()  # Include transfer back to ensure fair comparison
            return result
        
        gpu_thrust_time, gpu_thrust_std = time_operation(gpu_thrust_add, iterations)
        results['GPU Thrust Time (ms)'].append(gpu_thrust_time * 1000)
        results['GPU Thrust Std (ms)'].append(gpu_thrust_std * 1000)
    
    return results

def benchmark_subtraction(sizes, iterations=10):
    """Benchmark tensor subtraction across different implementations and sizes."""
    results = {
        'Size': [],
        'CPU Time (ms)': [],
        'CPU Std (ms)': [],
        'GPU Legacy Time (ms)': [],
        'GPU Legacy Std (ms)': [],
        'GPU Thrust Time (ms)': [],
        'GPU Thrust Std (ms)': []
    }
    
    for size in sizes:
        print(f"Benchmarking subtraction for size {size}...")
        results['Size'].append(size)
        
        # Create test data
        shape = [size, size]  # Square matrices for simplicity
        data1 = np.random.random((size, size)) + 1j * np.random.random((size, size))
        data2 = np.random.random((size, size)) + 1j * np.random.random((size, size))
        
        # CPU Tensor benchmark
        tensor1 = qforte.Tensor(shape)
        tensor2 = qforte.Tensor(shape)
        
        # Fill with test data
        tensor1.set_data(data1.reshape(-1))
        tensor2.set_data(data2.reshape(-1))
        
        # Time CPU subtraction
        def cpu_subtract():
            result = tensor1.clone()
            result.subtract(tensor2)
            return result
        
        cpu_time, cpu_std = time_operation(cpu_subtract, iterations)
        results['CPU Time (ms)'].append(cpu_time * 1000)
        results['CPU Std (ms)'].append(cpu_std * 1000)
        
        # GPU Legacy benchmark
        gpu_tensor1 = qforte.TensorGPU(shape)
        gpu_tensor2 = qforte.TensorGPU(shape)
        
        # Fill with test data
        gpu_tensor1.fill_from_nparray(data1.reshape(-1), shape)
        gpu_tensor2.fill_from_nparray(data2.reshape(-1), shape)
        
        # Move to GPU
        gpu_tensor1.to_gpu()
        gpu_tensor2.to_gpu()
        
        # Time GPU legacy subtraction
        def gpu_legacy_subtract():
            # First, bring tensors back to CPU for legacy subtraction
            temp1 = gpu_tensor1.clone()
            temp2 = gpu_tensor2.clone()
            temp1.to_cpu()
            temp2.to_cpu()
            temp1.subtract(temp2)
            temp1.to_gpu()  # Move back to GPU
            temp1.to_cpu()  # Include transfer back to ensure fair comparison
            return temp1
        
        gpu_legacy_time, gpu_legacy_std = time_operation(gpu_legacy_subtract, iterations)
        results['GPU Legacy Time (ms)'].append(gpu_legacy_time * 1000)
        results['GPU Legacy Std (ms)'].append(gpu_legacy_std * 1000)
        
        # Time GPU Thrust subtraction
        def gpu_thrust_subtract():
            result = gpu_tensor1.clone()
            result.subtractThrust(gpu_tensor2)
            result.to_cpu()  # Include transfer back to ensure fair comparison
            return result
        
        gpu_thrust_time, gpu_thrust_std = time_operation(gpu_thrust_subtract, iterations)
        results['GPU Thrust Time (ms)'].append(gpu_thrust_time * 1000)
        results['GPU Thrust Std (ms)'].append(gpu_thrust_std * 1000)
    
    return results

def benchmark_scale(sizes, iterations=10):
    """Benchmark tensor scaling across different implementations and sizes."""
    results = {
        'Size': [],
        'CPU Time (ms)': [],
        'CPU Std (ms)': [],
        'GPU Legacy Time (ms)': [],
        'GPU Legacy Std (ms)': [],
        'GPU Thrust Time (ms)': [],
        'GPU Thrust Std (ms)': []
    }
    
    for size in sizes:
        print(f"Benchmarking scaling for size {size}...")
        results['Size'].append(size)
        
        # Create test data
        shape = [size, size]  # Square matrices for simplicity
        data = np.random.random((size, size)) + 1j * np.random.random((size, size))
        scale_factor = complex(1.5, 0.5)
        
        # CPU Tensor benchmark
        tensor = qforte.Tensor(shape)
        
        # Fill with test data
        tensor.set_data(data.reshape(-1))
        
        # Time CPU scaling
        def cpu_scale():
            result = tensor.clone()
            result.scale(scale_factor)
            return result
        
        cpu_time, cpu_std = time_operation(cpu_scale, iterations)
        results['CPU Time (ms)'].append(cpu_time * 1000)
        results['CPU Std (ms)'].append(cpu_std * 1000)
        
        # GPU Legacy benchmark
        gpu_tensor = qforte.TensorGPU(shape)
        
        # Fill with test data
        gpu_tensor.fill_from_nparray(data.reshape(-1), shape)
        
        # Time GPU legacy scaling (on CPU then transfer)
        def gpu_legacy_scale():
            result = gpu_tensor.clone()
            result.to_cpu()
            result.scale(scale_factor)
            result.to_gpu()
            result.to_cpu()  # Include transfer back to ensure fair comparison
            return result
        
        gpu_legacy_time, gpu_legacy_std = time_operation(gpu_legacy_scale, iterations)
        results['GPU Legacy Time (ms)'].append(gpu_legacy_time * 1000)
        results['GPU Legacy Std (ms)'].append(gpu_legacy_std * 1000)
        
        # GPU Thrust benchmark
        gpu_tensor.to_gpu()
        
        # Time GPU Thrust scaling
        def gpu_thrust_scale():
            result = gpu_tensor.clone()
            result.scaleThrust(scale_factor)
            result.to_cpu()  # Include transfer back to ensure fair comparison
            return result
        
        gpu_thrust_time, gpu_thrust_std = time_operation(gpu_thrust_scale, iterations)
        results['GPU Thrust Time (ms)'].append(gpu_thrust_time * 1000)
        results['GPU Thrust Std (ms)'].append(gpu_thrust_std * 1000)
    
    return results

def plot_results(add_results, sub_results, scale_results, output_file="tensor_benchmark_results.png"):
    """Plot benchmark results."""
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    
    # Addition plot
    df_add = pd.DataFrame(add_results)
    axs[0].errorbar(df_add['Size'], df_add['CPU Time (ms)'], yerr=df_add['CPU Std (ms)'], 
                 label='CPU', marker='o', linestyle='-')
    axs[0].errorbar(df_add['Size'], df_add['GPU Legacy Time (ms)'], yerr=df_add['GPU Legacy Std (ms)'], 
                 label='GPU Legacy', marker='s', linestyle='--')
    axs[0].errorbar(df_add['Size'], df_add['GPU Thrust Time (ms)'], yerr=df_add['GPU Thrust Std (ms)'], 
                 label='GPU Thrust', marker='^', linestyle='-.')
    axs[0].set_title('Tensor Addition Performance')
    axs[0].set_xlabel('Matrix Size (N×N)')
    axs[0].set_ylabel('Time (ms)')
    axs[0].set_xscale('log2')
    axs[0].set_yscale('log')
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[0].legend()
    
    # Subtraction plot
    df_sub = pd.DataFrame(sub_results)
    axs[1].errorbar(df_sub['Size'], df_sub['CPU Time (ms)'], yerr=df_sub['CPU Std (ms)'], 
                 label='CPU', marker='o', linestyle='-')
    axs[1].errorbar(df_sub['Size'], df_sub['GPU Legacy Time (ms)'], yerr=df_sub['GPU Legacy Std (ms)'], 
                 label='GPU Legacy', marker='s', linestyle='--')
    axs[1].errorbar(df_sub['Size'], df_sub['GPU Thrust Time (ms)'], yerr=df_sub['GPU Thrust Std (ms)'], 
                 label='GPU Thrust', marker='^', linestyle='-.')
    axs[1].set_title('Tensor Subtraction Performance')
    axs[1].set_xlabel('Matrix Size (N×N)')
    axs[1].set_ylabel('Time (ms)')
    axs[1].set_xscale('log2')
    axs[1].set_yscale('log')
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[1].legend()
    
    # Scaling plot
    df_scale = pd.DataFrame(scale_results)
    axs[2].errorbar(df_scale['Size'], df_scale['CPU Time (ms)'], yerr=df_scale['CPU Std (ms)'], 
                 label='CPU', marker='o', linestyle='-')
    axs[2].errorbar(df_scale['Size'], df_scale['GPU Legacy Time (ms)'], yerr=df_scale['GPU Legacy Std (ms)'], 
                 label='GPU Legacy', marker='s', linestyle='--')
    axs[2].errorbar(df_scale['Size'], df_scale['GPU Thrust Time (ms)'], yerr=df_scale['GPU Thrust Std (ms)'], 
                 label='GPU Thrust', marker='^', linestyle='-.')
    axs[2].set_title('Tensor Scaling Performance')
    axs[2].set_xlabel('Matrix Size (N×N)')
    axs[2].set_ylabel('Time (ms)')
    axs[2].set_xscale('log2')
    axs[2].set_yscale('log')
    axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    
def print_speedup_tables(add_results, sub_results, scale_results):
    """Print tables showing speedup factors."""
    operations = ["Addition", "Subtraction", "Scaling"]
    all_results = [add_results, sub_results, scale_results]
    
    for op_name, results in zip(operations, all_results):
        df = pd.DataFrame(results)
        
        # Calculate speedups
        df['GPU Legacy vs CPU'] = df['CPU Time (ms)'] / df['GPU Legacy Time (ms)']
        df['GPU Thrust vs CPU'] = df['CPU Time (ms)'] / df['GPU Thrust Time (ms)']
        df['GPU Thrust vs Legacy'] = df['GPU Legacy Time (ms)'] / df['GPU Thrust Time (ms)']
        
        # Select columns for display
        display_df = df[['Size', 'GPU Legacy vs CPU', 'GPU Thrust vs CPU', 'GPU Thrust vs Legacy']]
        
        print(f"\n{op_name} Speedup Factors (higher is better):")
        print(tabulate(display_df, headers='keys', tablefmt='grid', floatfmt='.2f'))

def main():
    # Define matrix sizes to test (powers of 2)
    sizes = [32, 64, 128, 256, 512, 1024, 2048]
    iterations = 5  # Number of iterations for each test
    
    # Run benchmarks
    add_results = benchmark_addition(sizes, iterations)
    sub_results = benchmark_subtraction(sizes, iterations)
    scale_results = benchmark_scale(sizes, iterations)
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs("results", exist_ok=True)
    
    # Save raw results
    pd.DataFrame(add_results).to_csv("results/addition_benchmark.csv", index=False)
    pd.DataFrame(sub_results).to_csv("results/subtraction_benchmark.csv", index=False)
    pd.DataFrame(scale_results).to_csv("results/scaling_benchmark.csv", index=False)
    
    # Plot results
    plot_results(add_results, sub_results, scale_results, "results/tensor_benchmark_results.png")
    
    # Print speedup tables
    print_speedup_tables(add_results, sub_results, scale_results)
    
    print("\nBenchmarking complete! Results saved to the 'results' directory.")

if __name__ == "__main__":
    main()