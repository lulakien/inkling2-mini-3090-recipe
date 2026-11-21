"""Skinny BF16 GEMM for decode / spec-verify batches on Ampere (sm_8x), where cuBLAS picks slow kernels for
M = 1..16 rows with small N (e.g. a 6-CTA cutlass wmma kernel for the loop controllers, 30 us for 1.7 MB of weights).
out[B, M, N] = x[B, M, K] @ w[B, N, K]^T (+ bias), fp32 accumulation, any output dtype.
Weight-streaming: each program owns BLOCK_N output columns of one batch; when that leaves too few programs to fill
the GPU, K is split into SPLIT_K slices whose fp32 partials are summed in a fixed order (deterministic)."""
import functools

import torch
import triton
import triton.language as tl


@triton.jit
def _skinny_kernel(x_ptr, w_ptr, b_ptr, o_ptr, p_ptr, M, N, K, sxb, sxm, swb, swn, sob, som,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT_K: tl.constexpr,
                   HAS_BIAS: tl.constexpr, EVEN_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_b = tl.program_id(2)
    x_ptr += pid_b * sxb
    w_ptr += pid_b * swb
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
            w = tl.load(w_ptr + rn[:, None] * swn + kk[None, :], mask=n_mask, other=0.0)
        else:
            km = kk[None, :] < K
            x = tl.load(x_ptr + rm[:, None] * sxm + kk[None, :], mask=m_mask & km, other=0.0)
            w = tl.load(w_ptr + rn[:, None] * swn + kk[None, :], mask=n_mask & km, other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    omask = m_mask & (rn[None, :] < N)
    if SPLIT_K == 1:
        if HAS_BIAS:
            acc += tl.load(b_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
        tl.store(o_ptr + pid_b * sob + rm[:, None] * som + rn[None, :], acc.to(o_ptr.dtype.element_ty), mask=omask)
    else:
        tl.store(p_ptr + ((pid_b * SPLIT_K + pid_k) * M + rm[:, None]) * N + rn[None, :], acc, mask=omask)


@triton.jit
def _splitk_reduce(p_ptr, b_ptr, o_ptr, M, N, sob, som, SPLIT_K: tl.constexpr, HAS_BIAS: tl.constexpr,
                   BLOCK: tl.constexpr):
    pid_b = tl.program_id(1)
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < M * N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(p_ptr + (pid_b * SPLIT_K + s) * M * N + idx, mask=mask, other=0.0)
    n = idx % N
    if HAS_BIAS:
        acc += tl.load(b_ptr + n, mask=mask, other=0.0).to(tl.float32)
    m = idx // N
    tl.store(o_ptr + pid_b * sob + m * som + n, acc.to(o_ptr.dtype.element_ty), mask=mask)


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _config(B: int, M: int, N: int, K: int, device_index: int):
    bm = 16
    bn = 64 if N >= 64 else (32 if N > 16 else 16)
    bk = 256 if K >= 256 else max(16, triton.next_power_of_2(K))
    n_prog = B * triton.cdiv(N, bn)
    target = 2 * _num_sms(device_index)
    split = 1
    while n_prog * split < target and split < 16 and K // (split * 2) >= bk:
        split *= 2
    return bm, bn, bk, split


MAX_M = 16


def eligible(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """2-D x [M, K] / weight [N, K] or 3-D x [B, M, K] / weight [B, N, K], bf16, unit stride in K."""
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or not x.is_cuda:
        return False
    if x.dim() != weight.dim() or x.dim() not in (2, 3) or x.stride(-1) != 1 or weight.stride(-1) != 1:
        return False
    if x.dim() == 3 and (x.shape[0] != weight.shape[0]):
        return False
    return 1 <= x.shape[-2] <= MAX_M and weight.shape[-1] == x.shape[-1] and x.shape[-1] % 8 == 0


def skinny_gemm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
                out_dtype: torch.dtype | None = None) -> torch.Tensor:
    squeeze = x.dim() == 2
    x3 = x.unsqueeze(0) if squeeze else x
    w3 = weight.unsqueeze(0) if squeeze else weight
    B, M, K = x3.shape
    N = w3.shape[1]
    dev = x.device.index if x.device.index is not None else torch.cuda.current_device()
    bm, bn, bk, split = _config(B, M, N, K, dev)
    out = torch.empty((B, M, N), dtype=out_dtype or x.dtype, device=x.device)
    partials = torch.empty((B, split, M, N), dtype=torch.float32, device=x.device) if split > 1 else out
    _skinny_kernel[(triton.cdiv(N, bn), split, B)](
        x3, w3, bias if bias is not None else out, out, partials, M, N, K,
        x3.stride(0), x3.stride(1), w3.stride(0), w3.stride(1), out.stride(0), out.stride(1),
        BM=bm, BN=bn, BK=bk, SPLIT_K=split, HAS_BIAS=bias is not None and split == 1,
        EVEN_K=K % (split * bk) == 0, num_warps=4, num_stages=3)
    if split > 1:
        block = 1024
        _splitk_reduce[(triton.cdiv(M * N, block), B)](
            partials, bias if bias is not None else out, out, M, N, out.stride(0), out.stride(1),
            SPLIT_K=split, HAS_BIAS=bias is not None, BLOCK=block, num_warps=4)
    return out[0] if squeeze else out
