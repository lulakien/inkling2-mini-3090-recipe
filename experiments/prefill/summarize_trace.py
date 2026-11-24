#!/usr/bin/env python3
"""Sum GPU kernel time by kernel name in torch-profiler traces (one file per rank), grouped into coarse buckets.
  summarize_trace.py DIR"""
import collections, glob, json, os, re, sys

BUCKETS = [  # (bucket, regex on kernel name), first match wins
    ("full attention (Inkling prefill kernel)", r"fa4|_attention_kernel|attn_fwd|splitk|_rel_attention|combine"),
    ("experts / MoE GEMM", r"humming|moe|marlin|expert|fused_moe|align"),
    ("dense GEMM (cuBLAS / W8)", r"gemm|sm80_xmma|cutlass|ampere_.*gemm|_w8|skinny|gemv"),
    ("communication (all-reduce / copies)", r"nccl|allreduce|all_reduce|hostar|cear|memcpy|Memcpy|copy"),
    ("short conv / mix / attn_res", r"sconv|conv|attn_res|mix|blend"),
    ("norms / elementwise", r"norm|elementwise|vectorized|reduce|softcap|silu|gelu|act"),
]

files = sorted(glob.glob(os.path.join(sys.argv[1], "**", "*.json"), recursive=True))
if not files:
    sys.exit(f"no trace json under {sys.argv[1]}")
for f in files:
    ev = json.load(open(f))
    ev = ev.get("traceEvents", ev)
    k = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not k:
        continue
    span = (max(e["ts"] + e["dur"] for e in k) - min(e["ts"] for e in k)) / 1e3
    busy = sum(e["dur"] for e in k) / 1e3
    by = collections.Counter(); cnt = collections.Counter()
    for e in k:
        by[e["name"]] += e["dur"] / 1e3; cnt[e["name"]] += 1
    bucket = collections.Counter()
    for name, ms in by.items():
        b = next((b for b, rx in BUCKETS if re.search(rx, name, re.I)), "other")
        bucket[b] += ms
    print(f"== {os.path.basename(f)}: {len(k)} GPU ops, wall span {span:.0f} ms, GPU busy {busy:.0f} ms")
    for b, ms in bucket.most_common():
        print(f"  {ms:9.1f} ms {100 * ms / busy:5.1f}%  {b}")
    print("  top kernels:")
    for name, ms in by.most_common(12):
        print(f"  {ms:9.1f} ms {100 * ms / busy:5.1f}%  x{cnt[name]:<5} {name[:110]}")
