#!/usr/bin/env python3
"""Sustained dense BF16 GEMM (fp32 accumulation) TFLOPS per GPU, to check whether gpu-unlock is active.
Run inside the image:  docker run --rm --gpus all --entrypoint python3 -v $PWD/tools:/t:ro IMAGE /t/gemm_tflops.py
Stock GeForce RTX 3090 runs bf16-with-fp32-accumulate tensor math at half rate (spec: 71 TFLOPS dense at boost);
with the unlock this doubles."""
import time, torch
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
for d in range(torch.cuda.device_count()):
    torch.cuda.set_device(d); n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16); b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(20): a @ b
    torch.cuda.synchronize(); t = time.time(); it = 0
    while time.time() - t < 10:
        for _ in range(20): a @ b
        it += 20; torch.cuda.synchronize()
    dt = time.time() - t
    print(f"GPU{d} {torch.cuda.get_device_name(d)}: {2 * n**3 * it / dt / 1e12:.1f} TFLOPS bf16 (10 s sustained)")
