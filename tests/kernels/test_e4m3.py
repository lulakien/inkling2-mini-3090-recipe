"""Exactness harness for patches/e4m3.py (software FP8 E4M3 on sm_86) against torch.float8_e4m3fn.
Run with tests/kernels/run.sh. Encode: 10M random values over many scales + every boundary/tie case;
decode: all 256 codes. Saturation: |x| > 448 must give +-448 (torch gives NaN there, so compared on the clamp)."""
import sys
sys.path.insert(0, "/p")  # patches/ mounted at /p
import torch
import triton
import triton.language as tl
from e4m3 import e4m3_decode, e4m3_encode


@triton.jit
def _enc(X, Y, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + o, e4m3_encode(tl.load(X + o, mask=o < n)), mask=o < n)


@triton.jit
def _dec(X, Y, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + o, e4m3_decode(tl.load(X + o, mask=o < n)), mask=o < n)


def enc(x):
    y = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
    _enc[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), BLOCK=1024)
    return y


def dec(b):
    y = torch.empty(b.shape, dtype=torch.float32, device=b.device)
    _dec[(triton.cdiv(b.numel(), 1024),)](b, y, b.numel(), BLOCK=1024)
    return y


torch.manual_seed(0)
dev = "cuda"
fails = 0
# decode: all 256 codes (NaN codes compared as NaN)
codes = torch.arange(256, dtype=torch.uint8, device=dev)
ref = codes.view(torch.float8_e4m3fn).float()
got = dec(codes)
ok = torch.equal(torch.nan_to_num(ref, nan=1e9), torch.nan_to_num(got, nan=1e9))
if not ok:
    for i in range(256):
        a_, b_ = ref[i].item(), got[i].item()
        if not (a_ == b_ or (a_ != a_ and b_ != b_)):
            print(f"  code {i:#04x}: torch {a_!r} ours {b_!r}")
print("decode 256 codes:", "OK" if ok else "FAIL"); fails += not ok
# encode: random values at many scales, plus exact representable values, midpoints (ties) and neighbours
xs = [torch.randn(10_000_000, device=dev) * s for s in (1e-3, 0.03, 1.0, 10.0, 100.0)]
vals = ref[~ref.isnan()]
vals = torch.unique(vals)
mids = (vals[1:] + vals[:-1]) / 2
xs += [vals, mids, torch.nextafter(mids, torch.full_like(mids, 1e9)), torch.nextafter(mids, torch.full_like(mids, -1e9)),
       torch.tensor([0.0, -0.0, 448.0, -448.0, 2.0**-9, 2.0**-10, 3 * 2.0**-11, 2.0**-6 - 2.0**-12], device=dev)]
for x in xs:
    xc = x.clamp(-448.0, 448.0)
    r = xc.to(torch.float8_e4m3fn).view(torch.uint8)
    g = enc(x.contiguous())
    bad = (r != g).sum().item()
    fails += bad > 0
    if bad:
        i = (r != g).nonzero()[0].item()
        print(f"encode n={x.numel()}: {bad} mismatches, e.g. x={x[i].item()!r} torch={r[i].item():#04x} ours={g[i].item():#04x}")
    else:
        print(f"encode n={x.numel():>10}: OK")
sat = enc(torch.tensor([449.0, 1e6, -1e30, float("inf")], device=dev))
ok = sat.tolist() == [0x7E, 0x7E, 0xFE, 0x7E]
print("saturation:", "OK" if ok else f"FAIL {sat.tolist()}"); fails += not ok
print("ALL EXACT" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
