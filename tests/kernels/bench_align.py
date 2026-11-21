"""Time the three align implementations at decode and prefill route counts (GPU free)."""
import sys, os
sys.path.insert(0, "/p")
import torch, triton
import moe_align_block_size as M

dev = torch.device("cuda", 0); torch.manual_seed(0)
E, BS = 128, 16
ITERS = 100

def buf(n):
    cap = n + E * (BS - 1)
    return (torch.empty(cap, device=dev, dtype=torch.int32),
            torch.empty((cap + BS - 1) // BS, device=dev, dtype=torch.int32),
            torch.empty(1, device=dev, dtype=torch.int32))

def timeit(fn, n):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
    for _ in range(ITERS): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / ITERS

for n in (2048, 3072, 4096, 6144):
    ids = torch.randint(0, E, (n,), device=dev, dtype=torch.int32)
    rows = [(("torch", M._moe_align_block_size_portable),
             ("single", M._moe_align_block_size_triton),
             ("parallel", M._moe_align_block_size_parallel))]
    t = {}
    for name, fn in rows[0]:
        if name == "single" and n > M._TRITON_ALIGN_MAX_ROUTES:
            continue
        s, e, c = buf(n)
        t[name] = timeit(lambda fn=fn, s=s, e=e, c=c: fn(ids, E, BS, s, e, c, None, None), n)
    base = t.get("torch")
    print(f"n={n:5d} " + "  ".join(f"{k} {v:7.1f} us" + (f" ({base/v:4.1f}x)" if k != "torch" and base else "") for k, v in t.items()), flush=True)
