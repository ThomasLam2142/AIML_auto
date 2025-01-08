# Credit to SemiAnalysis / semianalysis.com

import torch
import triton

# Tensor size (1GB)
tensor_size = 2 * 1024**3
dtype = torch.float32
num_elements = tensor_size

# Allocate tensors on the device
a = torch.randn(num_elements, device="cuda", dtype=dtype)
b = torch.empty_like(a)

# Function to benchmark
def copy_tensor():
    b.copy_(a)

# Benchmark using Triton
time_ms = triton.testing.do_bench(copy_tensor, warmup=30, rep=200)

# Calculate bandwidth
bandwidth_gbps = (tensor_size * 2) / (time_ms * 1e-3) / 1e9 #multiply by 2 for read + write

# Print results
print(f"Copy Bandwidth: {bandwidth_gbps:.2f} GB/s (time: {time_ms:.2f} ms)")