import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("benchmarks/benchmark_results.csv", sep=";")
df.columns = df.columns.str.strip()  


tests = df["Test Id"]
sizes = df["Size"]
cuda_time = df["CUDA Time"]
pytorch_time = df["PyTorch Time"]
steps_value = df["Steps"].iloc[0]  


plt.figure(figsize=(10,6))
plt.plot(sizes, cuda_time, marker='o', label='CUDA Kernel Time (ms)')
plt.plot(sizes, pytorch_time, marker='s', label='PyTorch Time (ms)')
plt.yscale("log")
plt.xlabel("Grid Size")
plt.ylabel("Time (ms)")
plt.title(f"Performance Comparison over {steps_value} Steps")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("benchmarks/performance_comparison.png", dpi=300)
plt.close()


speedup = pytorch_time / cuda_time  

plt.figure(figsize=(10,6))
bars = plt.bar(sizes, speedup, color='skyblue')
plt.ylabel("Speedup (PyTorch / CUDA)")
plt.title("PyTorch vs CUDA Speedup")


for bar, val in zip(bars, speedup):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.05*max(speedup),
             f"{val:.1f}", ha='center', va='bottom', fontsize=10)

plt.grid(axis='y', linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("benchmarks/speedup_plot.png", dpi=300)
plt.close()