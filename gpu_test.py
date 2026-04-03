import sys
import time

import torch

print("Python executable:", sys.executable)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch CUDA build:", torch.version.cuda)

    device = torch.device("cuda")
    x = torch.randn(5000, 5000, device=device)
    y = torch.randn(5000, 5000, device=device)

    torch.cuda.synchronize()
    t1 = time.time()

    z = x @ y

    torch.cuda.synchronize()
    t2 = time.time()

    print("Result device:", z.device)
    print("Elapsed time:", t2 - t1, "seconds")
    print("Allocated memory (GB):", torch.cuda.memory_allocated() / 1e9)
else:
    print("GPU not available")
