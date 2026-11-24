"""Microbenchmark of the prefill split-KV attention call at depth: one 2048-token chunk at past=PAST keys, 12/4 heads,
d=128, bf16 paged KV (64-token pages, the real (blocks, kv_heads, page, 2*D) layout), rel extent 1024, MODE_PARTIAL,
5 splits (what the server ran in the 500K trace). Shipped kernel vs the variant with tuning knobs; checks outputs."""
import sys, json, torch
sys.path.insert(0, "/x/experiments/prefill")
from relfar import attention_sm90 as V
sys.path.insert(0, "/x")
from patches import attention_sm90 as S  # the overlay (far path off unless INKLING2_FAR_ATTN=1)
from relfar import orig_sm90 as O  # the shipped kernel before the patch
torch.manual_seed(0); dev = "cuda"
T, H, HK, D, E, PG = 2048, 12, 4, 128, 1024, 64
PAST = int(sys.argv[1]) if len(sys.argv) > 1 else 483328
SPL = int(sys.argv[2]) if len(sys.argv) > 2 else 5
L = PAST + T; nb = (L + PG - 1) // PG
kv = torch.randn(nb, HK, PG, 2 * D, device=dev, dtype=torch.bfloat16)
import os
FP8 = os.environ.get("FP8") == "1"
if FP8:
    PG = 128; nb = (L + PG - 1) // PG
    kv = (torch.randint(0, 0x40, (nb, HK, PG, 2 * D), device=dev, dtype=torch.uint8) | (torch.randint(0, 2, (nb, HK, PG, 2 * D), device=dev, dtype=torch.uint8) << 7))
q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
rel = torch.randn(T + 128, H, E + 256, device=dev, dtype=torch.bfloat16) * 0.5
bt = torch.randperm(nb, device=dev, dtype=torch.int32)[None]
kv2 = torch.empty_like(kv); kv2[bt[0].long()] = kv  # the page table points into a shuffled pool
assert torch.equal(kv2[bt[0].long()], kv)
k2, v2 = kv2.transpose(1, 2).split(D, dim=-1)
sl = torch.tensor([L], device=dev, dtype=torch.int32); cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
def run(mod, tune=None):
    part = torch.empty(SPL, T, H, D, device=dev); lse = torch.empty(SPL, T, H, device=dev)
    kw = dict(block_table=bt, cache_seqlens=sl, cu_seqlens_q=cu, max_query_len=T, softmax_scale=D ** -0.5,
              window=0, num_splits=SPL, mode=S.MODE_PARTIAL)
    if tune is not None: kw["tune"] = tune
    f = lambda: mod.rel_attention(q, k2, v2, part, lse, rel, **kw)
    f(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for _ in range(3): f()
    e.record(); torch.cuda.synchronize()
    assert not part.isnan().any() and not lse.isnan().any(), "NaN in kernel output"
    m = lse.max(0).values; w = torch.exp2(lse - m); o = (part * w[..., None]).sum(0) / w.sum(0)[..., None]
    return s.elapsed_time(e) / 3, o
flops = 4 * T * L * H * D
base_ms, ref = run(O)
print(f"FP8={FP8} PAST={PAST} splits={SPL} shipped: {base_ms:.2f} ms {flops / base_ms / 1e9:.1f} TFLOPS", flush=True)
off_ms, off = run(S)
print(f"overlay, far path off: {off_ms:.2f} ms, bit-identical to shipped: {torch.equal(off, ref)}", flush=True)
cfgs = json.loads(sys.argv[3]) if len(sys.argv) > 3 else [
    {"FAR": True}, {"FAR": True, "S": 3}, {"FAR": True, "BN": 128}, {"FAR": True, "BN": 128, "S": 3},
    {"FAR": True, "W": 4}, {"FAR": True, "BN": 32, "S": 3}, {"FAR": True, "BM": 64, "W": 4},
    {"FAR": True, "BM": 64, "W": 4, "BN": 128}, {"FAR": True, "BM": 64, "W": 4, "S": 3}]
for c in cfgs:
    try:
        ms, o = run(V, c)
        err = (o - ref).abs().max().item(); rel_err = err / ref.abs().max().item()
        print(f"{json.dumps(c):48s} {ms:7.2f} ms {flops / ms / 1e9:6.1f} TFLOPS  x{base_ms / ms:.3f}  maxdiff {err:.2e} (rel {rel_err:.1e}), bit-identical {torch.equal(o, ref)}", flush=True)
    except Exception as ex:
        print(f"{json.dumps(c):48s} FAIL {type(ex).__name__}: {str(ex)[:120]}", flush=True)
