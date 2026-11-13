import torch, time

for size in [500, 2000, 4000, 8000]:
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    a @ b                                   # warm up the CPU
    start = time.time(); a @ b; cpu = time.time() - start

    a_gpu, b_gpu = a.to("mps"), b.to("mps")
    a_gpu @ b_gpu; torch.mps.synchronize()  # warm up the GPU
    start = time.time(); a_gpu @ b_gpu; torch.mps.synchronize(); gpu = time.time() - start

    print(f"{size}x{size}: CPU {cpu:.4f}s | GPU {gpu:.4f}s | GPU is {cpu/gpu:.1f}x the speed of CPU")
