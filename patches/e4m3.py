# SPDX-License-Identifier: Apache-2.0
"""Software FP8 E4M3 (e4m3fn) conversion for GPUs without the sm_89+ cvt instructions (sm_86 RTX 3090).

Encode: fp32 -> e4m3 byte, round to nearest even, saturating to +-448 (= cvt.rn.satfinite.e4m3x2.f32).
Decode: e4m3 byte -> fp32. The byte's sign/exponent/mantissa shifted into an fp16 bit pattern is exactly
value * 2^-8 (normals and subnormals alike: the e4m3 bias 7 vs fp16 bias 15 differ by 8), so decode is a shift,
a bitcast and an exact multiply by 256 (idea from HyperInkling, Pento95, Apache-2.0).
Both are bit-exact against torch.float8_e4m3fn (tests/kernels/test_e4m3.py)."""
import triton
import triton.language as tl


@triton.jit
def e4m3_encode(x):
    """fp32 tensor -> uint8 e4m3 codes (RNE, satfinite; NaN -> 0x7F)."""
    bits = x.to(tl.uint32, bitcast=True)
    sign = ((bits >> 31) & 1).to(tl.uint32)
    a = tl.abs(x)
    a = tl.minimum(a, 448.0)
    abits = a.to(tl.uint32, bitcast=True)
    # normal range (a >= 2^-6): round the fp32 mantissa to 3 bits, nearest even
    lsb = (abits >> 20) & 1
    r = abits + 0x7FFFF + lsb
    exp = (r >> 23).to(tl.int32) - 127 + 7
    man = (r >> 20) & 7
    normal = (exp.to(tl.uint32) << 3) | man
    # subnormal range (a < 2^-6): code = round_half_even(a * 2^9), 8 becomes the smallest normal (0x08)
    big = 8388608.0
    q = ((a * 512.0 + big) - big).to(tl.uint32)
    code = tl.where(a >= 0.015625, normal, q)
    code = tl.where(x != x, 0x7F, code)
    return ((sign << 7) | code).to(tl.uint8)


@triton.jit
def e4m3_decode(b):
    """uint8/uint16/uint32 tensor holding e4m3 codes in its low byte -> fp32 (exact)."""
    c = b.to(tl.uint32) & 0xFF
    h = (((c & 0x80) << 8) | ((c & 0x7F) << 7)).to(tl.uint16)
    v = h.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
    return tl.where((c & 0x7F) == 0x7F, float("nan"), v)
