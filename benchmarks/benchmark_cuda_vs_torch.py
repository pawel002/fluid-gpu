"""
Benchmark script to compare CUDA C++ vs PyTorch implementations
of the 2D Burgers' equation solver.
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from fluidgpu import solver_steps as solver_steps_cuda
from fluidgpu_torch import solver_steps as solver_steps_torch


def generate_test_data(nx, ny):
    """Generate test data for benchmarking"""
    u = np.zeros((ny, nx), dtype=np.float32)
    v = np.zeros((ny, nx), dtype=np.float32)
    
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="xy")
    
    u += np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01).astype(np.float32)
    v += 0.5 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02).astype(np.float32)
    
    return u, v


def benchmark_implementation(name, solver_func, u, v, nu, dt, dx, dy, steps, warmup_runs=2, benchmark_runs=10):
    """Benchmark a solver implementation"""
    # Warmup runs
    u_warmup = u.copy()
    v_warmup = v.copy()
    for _ in range(warmup_runs):
        solver_func(u=u_warmup, v=v_warmup, nu=nu, dt=dt, dx=dx, dy=dy, steps=steps)
    
    # Benchmark runs
    times = []
    results = []
    
    for run in range(benchmark_runs):
        u_run = u.copy()
        v_run = v.copy()
        
        start = time.perf_counter()
        solver_func(u=u_run, v=v_run, nu=nu, dt=dt, dx=dx, dy=dy, steps=steps)
        end = time.perf_counter()
        
        elapsed = end - start
        times.append(elapsed)
        results.append((u_run.copy(), v_run.copy()))
    
    # Statistics
    times = np.array(times)
    mean_time = np.mean(times)
    
    print(f"  {name}: {mean_time*1000:.2f} ms ({mean_time/steps*1000:.4f} ms/step)")
    
    # Return mean result and timing
    mean_result = (np.mean([r[0] for r in results], axis=0), 
                   np.mean([r[1] for r in results], axis=0))
    
    return mean_result, mean_time, times


def compare_results(result1, result2, name1, name2, tolerance=1e-5):
    u1, v1 = result1
    u2, v2 = result2
    
    # Ensure numpy arrays
    u1 = np.asarray(u1, dtype=np.float32)
    v1 = np.asarray(v1, dtype=np.float32)
    u2 = np.asarray(u2, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    
    # Check for NaN or Inf
    if np.any(np.isnan(u1)) or np.any(np.isnan(v1)):
        print(f"  X {name1} contains NaN values")
        return np.nan, np.nan
    if np.any(np.isnan(u2)) or np.any(np.isnan(v2)):
        print(f"  X {name2} contains NaN values")
        return np.nan, np.nan
    
    u_diff = np.abs(u1 - u2)
    v_diff = np.abs(v1 - v2)
    
    u_max_diff = np.max(u_diff)
    v_max_diff = np.max(v_diff)
    
    match = u_max_diff < tolerance and v_max_diff < tolerance
    status = "✓" if match else "X"
    print(f"  {status} Max diff: U={u_max_diff:.2e}, V={v_max_diff:.2e}")
    
    return u_max_diff, v_max_diff


def main():
    print("2D Burgers' Equation Solver: CUDA vs PyTorch Benchmark\n")
    
    # Test parameters (similar to visualize_solver.py)
    test_cases = [
        {"nx": 128, "ny": 128, "steps": 1500, "name": "Small (128x128)"},
        {"nx": 256, "ny": 256, "steps": 1500, "name": "Medium (256x256)"},
        {"nx": 512, "ny": 512, "steps": 1500, "name": "Large (512x512)"},
    ]
    
    nu = 0.005  # Similar to Gaussian_Diffusion scenario
    dt = 0.0005  # Same as visualize_solver.py
    
    for test_case in test_cases:
        nx = test_case["nx"]
        ny = test_case["ny"]
        steps = test_case["steps"]
        name = test_case["name"]
        
        dx = 1.0 / (nx - 1)
        dy = 1.0 / (ny - 1)
        
        print(f"{name} ({nx}x{ny}, {steps} steps):")
        
        # Generate test data
        u_base, v_base = generate_test_data(nx, ny)
        
        # Benchmark CUDA
        result_cuda, time_cuda, times_cuda = benchmark_implementation(
            "CUDA C++", solver_steps_cuda, u_base.copy(), v_base.copy(),
            nu, dt, dx, dy, steps
        )
        
        # Benchmark PyTorch
        result_torch, time_torch, times_torch = benchmark_implementation(
            "PyTorch", solver_steps_torch, u_base.copy(), v_base.copy(),
            nu, dt, dx, dy, steps
        )
        
        # Compare results
        compare_results(result_cuda, result_torch, "CUDA", "PyTorch")
        
        # Performance comparison
        speedup = time_torch / time_cuda
        faster = "CUDA" if speedup > 1 else "PyTorch"
        ratio = speedup if speedup > 1 else 1/speedup
        print(f"  {faster} is {ratio:.2f}x faster\n")


if __name__ == "__main__":
    main()

