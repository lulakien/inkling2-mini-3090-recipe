"""Exactness + timing: 3-pass `_attn_res_kernel` vs the sm8x 2-pass prefill kernel."""
import os, sys
sys.path.insert(0, "/p")  # patches/ mounted at /p (see tests/kernels/run.sh)
import torch
import attn_res as A

dev = torch.device("cuda", 0)
torch.manual_seed(0)
D = 2304
bad = 0

def run(inputs, normalize, two_pass):
    stack, last, add, queries, norm = inputs
    os.environ["VLLM_ATTN_RES_2PASS"] = "1" if two_pass else "0"
    return A.attn_res(stack, last, add, queries, norm, 1e-5, normalize)

for t in (1024, 1536, 2048):
    for q in (1, 2, 4):
        for ns in (1, 2, 4):
            for addend in (False, True):
                for normalize in (False, True):
                    for apply_norm in (False, True):
                        stack = torch.randn(ns, t, D, device=dev, dtype=torch.bfloat16) * 0.1
                        last = torch.randn(t, D, device=dev, dtype=torch.bfloat16) * 0.1
                        add = torch.randn(t, D, device=dev, dtype=torch.bfloat16) * 0.1 if addend else None
                        queries = torch.randn(q, D, device=dev, dtype=torch.float32) * 0.05
                        norm = torch.randn(q, D, device=dev, dtype=torch.float32) * 0.1 if apply_norm else None
                        a = run((stack, last, add, queries, norm), normalize, False)
                        b = run((stack, last, add, queries, norm), normalize, True)
                        ok = torch.equal(a, b)
                        if not ok:
                            d = (a.float() - b.float()).abs().max().item()
                            print(f"t={t} q={q} ns={ns} add={addend} norm={normalize} apply={apply_norm} FAIL maxdiff={d}")
                        bad += not ok
print("bit-identical in all 72 cases" if bad == 0 else f"{bad} FAILURES")

def bench(two_pass, t=1024, q=2, ns=2):
    stack = torch.randn(ns, t, D, device=dev, dtype=torch.bfloat16) * 0.1
    last = torch.randn(t, D, device=dev, dtype=torch.bfloat16) * 0.1
    queries = torch.randn(q, D, device=dev, dtype=torch.float32) * 0.05
    norm = torch.randn(q, D, device=dev, dtype=torch.float32) * 0.1
    os.environ["VLLM_ATTN_RES_2PASS"] = "1" if two_pass else "0"
    for _ in range(5):
        A.attn_res(stack, last, None, queries, norm, 1e-5, True)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(50):
        A.attn_res(stack, last, None, queries, norm, 1e-5, True)
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / 50

for t, q, ns in ((1024, 2, 2), (1024, 1, 4), (2048, 2, 2)):
    x, y = bench(False, t, q, ns), bench(True, t, q, ns)
    print(f"T={t} Q={q} N_STACK={ns}: 3-pass {x:6.1f} us  2-pass {y:6.1f} us  ({x/y:4.2f}x)")
