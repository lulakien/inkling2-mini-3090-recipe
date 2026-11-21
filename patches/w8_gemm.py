"""W8A16 skinny GEMM for decode / spec-verify batches on Ampere: out[M, N] = (x[M, K] @ q[N, K]^T) * scale[N] (+ bias).
q int8 symmetric per output channel (scale = amax/127), exactly representable in bf16; fp32 accumulation; split-K
partials (already scaled) summed in a fixed order (deterministic). M <= 64."""
import functools

import torch
import triton
import triton.language as tl


@triton.jit
def _w8_kernel(x_ptr, w_ptr, s_ptr, b_ptr, o_ptr, p_ptr, M, N, K, sxm, swn, som,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT_K: tl.constexpr,
               HAS_BIAS: tl.constexpr, EVEN_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    k_per = tl.cdiv(K, SPLIT_K * BK) * BK
    k_lo = pid_k * k_per
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    m_mask = rm[:, None] < M
    n_mask = rn[:, None] < N
    for k0 in range(k_lo, tl.minimum(k_lo + k_per, K), BK):
        kk = k0 + rk
        if EVEN_K:
            x = tl.load(x_ptr + rm[:, None] * sxm + kk[None, :], mask=m_mask, other=0.0)
            w = tl.load(w_ptr + rn[:, None] * swn + kk[None, :], mask=n_mask, other=0)
        else:
            km = kk[None, :] < K
            x = tl.load(x_ptr + rm[:, None] * sxm + kk[None, :], mask=m_mask & km, other=0.0)
            w = tl.load(w_ptr + rn[:, None] * swn + kk[None, :], mask=n_mask & km, other=0)
        acc = tl.dot(x, tl.trans(w.to(tl.bfloat16)), acc)
    acc = acc * tl.load(s_ptr + rn, mask=rn < N, other=0.0)[None, :]
    omask = m_mask & (rn[None, :] < N)
    if SPLIT_K == 1:
        if HAS_BIAS:
            acc += tl.load(b_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
        tl.store(o_ptr + rm[:, None] * som + rn[None, :], acc.to(o_ptr.dtype.element_ty), mask=omask)
    else:
        tl.store(p_ptr + (pid_k * M + rm[:, None]) * N + rn[None, :], acc, mask=omask)


@triton.jit
def _reduce(p_ptr, b_ptr, o_ptr, M, N, som, SPLIT_K: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < M * N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(p_ptr + s * M * N + idx, mask=mask, other=0.0)
    n = idx % N
    if HAS_BIAS:
        acc += tl.load(b_ptr + n, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptr + (idx // N) * som + n, acc.to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _dequant_kernel(q_ptr, s_ptr, o_ptr, N, K, BN: tl.constexpr, BK: tl.constexpr):
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    rk = tl.program_id(1) * BK + tl.arange(0, BK)
    mask = (rn[:, None] < N) & (rk[None, :] < K)
    q = tl.load(q_ptr + rn[:, None] * K + rk[None, :], mask=mask, other=0).to(tl.float32)
    sc = tl.load(s_ptr + rn, mask=rn < N, other=0.0)
    tl.store(o_ptr + rn[:, None] * K + rk[None, :], (q * sc[:, None]).to(tl.bfloat16), mask=mask)


def dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """int8 [N, K] (row-contiguous) * scale[N] -> bf16 [N, K] in one pass (same rounding as the fp32 product)."""
    n, k = q.shape
    out = torch.empty((n, k), dtype=torch.bfloat16, device=q.device)
    _dequant_kernel[(triton.cdiv(n, 32), triton.cdiv(k, 256))](q, s, out, n, k, BN=32, BK=256, num_warps=4)
    return out


@functools.cache
def _num_sms(i: int) -> int:
    return torch.cuda.get_device_properties(i).multi_processor_count


@functools.cache
def _config(M: int, N: int, K: int, dev: int, bn: int | None, bk: int | None):
    bn = bn or (64 if N >= 64 else 32)
    bk = bk or (256 if K >= 256 else max(32, triton.next_power_of_2(K)))
    n_prog = triton.cdiv(N, bn)
    split = 1
    while n_prog * split < 2 * _num_sms(dev) and split < 16 and K // (split * 2) >= bk:
        split *= 2
    return bn, bk, split


MAX_M = 64


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [N, K] -> (int8 [N, K], fp32 scale [N]); symmetric per output channel, round to nearest."""
    q = torch.empty(w.shape, dtype=torch.int8, device=w.device)
    s = torch.empty(w.shape[0], dtype=torch.float32, device=w.device)
    for i in range(0, w.shape[0], 4096):  # row chunks: small temporaries (the memory profile sees the peak)
        wf = w[i:i + 4096].float()
        si = (wf.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
        q[i:i + 4096] = torch.round(wf / si[:, None]).clamp_(-127, 127).to(torch.int8)
        s[i:i + 4096] = si
    return q, s


def w8_gemm(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, bias: torch.Tensor | None = None,
            out_dtype: torch.dtype | None = None, bn: int | None = None, bk: int | None = None,
            warps: int = 4, stages: int = 3, out: torch.Tensor | None = None) -> torch.Tensor:
    assert x.dim() == 2 and 1 <= x.shape[0] <= MAX_M and x.stride(-1) == 1 and x.dtype == torch.bfloat16
    M, K = x.shape
    N = q.shape[0]
    dev = x.device.index if x.device.index is not None else torch.cuda.current_device()
    bn, bk, split = _config(M, N, K, dev, bn, bk)
    if out is None:
        out = torch.empty((M, N), dtype=out_dtype or x.dtype, device=x.device)
    part = torch.empty((split, M, N), dtype=torch.float32, device=x.device) if split > 1 else out
    _w8_kernel[(triton.cdiv(N, bn), split)](
        x, q, s, bias if bias is not None else out, out, part, M, N, K, x.stride(0), q.stride(0), out.stride(0),
        BM=max(16, triton.next_power_of_2(M)), BN=bn, BK=bk, SPLIT_K=split, HAS_BIAS=bias is not None and split == 1,
        EVEN_K=K % (split * bk) == 0, num_warps=warps, num_stages=stages)
    if split > 1:
        _reduce[(triton.cdiv(M * N, 1024),)](part, bias if bias is not None else out, out, M, N, out.stride(0),
                                             SPLIT_K=split, HAS_BIAS=bias is not None, BLOCK=1024, num_warps=4)
    return out


# --- sm8x patch: lazily quantized module weights (VLLM_INKLING_W8="attn,head") -------------------------------
import os as _os

W8 = {p.strip() for p in _os.environ.get("VLLM_INKLING_W8", "").split(",") if p.strip()}


def get_w8(module: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(q, scale) of `module.weight`, quantized on first eager use; the bf16 weight is then freed.
    Returns None while capturing a CUDA graph before quantization (the caller keeps the bf16 path)."""
    w8 = getattr(module, "_w8", None)
    if w8 is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        w = module.weight.data
        q, s = quantize(w)
        module._w8 = w8 = (q, s)
        module._w8_shape = tuple(w.shape)
        module.weight.data = w.new_empty((0, w.shape[1]))
        torch.cuda.empty_cache()
    return w8


def w8_linear(x: torch.Tensor, w8: tuple[torch.Tensor, torch.Tensor], r0: int = 0, r1: int | None = None,
              out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """x @ dequant(q[r0:r1])^T: skinny int8 kernel up to MAX_M rows, else dequantize the rows to bf16 and use cuBLAS."""
    q, s = w8
    q, s = q[r0:r1], s[r0:r1]
    m = x.shape[0]
    if m <= MAX_M:
        return w8_gemm(x.contiguous(), q, s, out_dtype=out_dtype)
    if q.shape[0] > 16384:  # LM head: no full-size dequantized temporary (it would show in the memory profile)
        x = x.contiguous()
        out = torch.empty((m, q.shape[0]), dtype=out_dtype or x.dtype, device=x.device)
        for i in range(0, m, MAX_M):  # each chunk writes its rows in place (no per-chunk temporaries + cat)
            w8_gemm(x[i:i + MAX_M], q, s, out=out[i:i + MAX_M])
        return out
    wd = dequant(q, s)
    if out_dtype == torch.float32:
        return torch.mm(x, wd.t(), out_dtype=torch.float32)
    return torch.nn.functional.linear(x, wd)
