# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, overload

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up


@overload
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    *,
    return_scatter_idx: Literal[False] = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@overload
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    *,
    return_scatter_idx: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...


@overload
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    *,
    return_scatter_idx: bool,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
): ...


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    return_scatter_idx: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Note: In the case of expert_parallel, moe_align_block_size initially
    considers all experts as valid and aligns all tokens appropriately.
    Before the function returns it marks the experts_ids that are not in
    the current GPU rank as -1 so the MoE matmuls could skip those blocks.
    This requires the num_experts input arg to be the num global experts.

    Args:
        topk_ids: A tensor of shape [total_tokens, top_k] representing the
            top-k expert indices for each token.
        block_size: The block size used in block matrix multiplication.
        num_experts: The total number of experts.
        expert_map: A tensor of shape [num_experts] that maps the expert index
            from the global space to the local index space of the current
            expert parallel shard. If the expert is not in the current expert
            parallel shard, the mapping is set to -1.
        pad_sorted_ids: A flag indicating whether the sorted_token_ids length
            should be padded to a multiple of block_size,
        ignore_invalid_experts: A flag indicating whether to ignore invalid
            experts. When False, all expert_ids in topk_ids will participate in
            counting and ranking, but invalid experts in expert_ids will be marked
            as -1. When True, all invalid expert_ids in topk_ids will be ignored
            and will not participate in counting or ranking, and there will be no
            -1 in expert_ids.
        return_scatter_idx: Whether to additionally return a masked identity
            mapping for the original routed rows.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.
    - scatter_idx: Only returned when return_scatter_idx=True. An int32 tensor
        shaped [topk_ids.numel(), 1], containing the original routed row index
        for valid routes and -1 otherwise. With expert_map, this requires
        ignore_invalid_experts=True.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.

    """
    if return_scatter_idx and expert_map is not None and not ignore_invalid_experts:
        raise ValueError("scatter_idx with expert_map requires ignore_invalid_experts")
    scatter_idx = None
    if return_scatter_idx:
        scatter_idx = torch.empty(
            (topk_ids.numel(), 1),
            dtype=torch.int32,
            device=topk_ids.device,
        )
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    if _needs_portable_align(topk_ids.device):
        import os as _os

        # sm8x: VLLM_ALIGN_MODE=legacy reproduces the pre-kernel behaviour (single program up to 4096
        # routes, torch above); auto uses the single program below 2048 routes (measured crossover:
        # 39.9 vs 44.9 us at 1024, 71.3 vs 45.3 at 2048) and the parallel path above it.
        _mode = _os.environ.get("VLLM_ALIGN_MODE", "auto")
        _n = topk_ids.numel()
        if _mode == "legacy":
            _align = (
                _moe_align_block_size_triton
                if _n <= _TRITON_ALIGN_MAX_ROUTES
                else _moe_align_block_size_portable
            )
        elif _mode == "torch":
            _align = _moe_align_block_size_portable
        elif _mode == "parallel":
            _align = _moe_align_block_size_parallel
        else:
            _align = (
                _moe_align_block_size_parallel
                if _n >= 2048
                else _moe_align_block_size_triton
            )
        _align(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            expert_map if ignore_invalid_experts else None,
            scatter_idx,
        )
    else:
        ops.moe_align_block_size(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            expert_map if ignore_invalid_experts else None,
            scatter_idx,
        )

    if expert_map is not None and not ignore_invalid_experts:
        expert_ids = expert_map[expert_ids]

    if scatter_idx is not None:
        return sorted_ids, expert_ids, num_tokens_post_pad, scatter_idx
    return sorted_ids, expert_ids, num_tokens_post_pad


def batched_moe_align_block_size(
    max_tokens_per_batch: int, block_size: int, expert_num_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Given num_batches, max_tokens_per_batch, block_size and the number of
    valid-tokens in each batch, prepare sorted_token_ids, expert_ids and
    num_tokens_post_pad. sorted_token_ids, expert_ids and num_tokens_post_pad
    have the same semantics as in moe_align_block_size.

    This function is intended to be a drop in replacement for
    moe_align_batch_size for the batched case.

    Args:
        max_tokens_per_batch (int): Number of tokens in each batch (both
            valid and invalid).
        block_size (int): block_size to align the data to.
        expert_num_tokens (torch.Tensor): expert_num_tokens[i], indicates
            the number of valid tokens in batch i.

    Returns:
    - sorted_token_ids (torch.Tensor): Torch tensor of size
        (num_batches * max_tokens_per_batch) indicating the token indices for
        that block.
    - expert_ids (torch.Tensor): Torch tensor of size
        ceil((num_batches * max_tokens_per_batch) / block_size) indicating
        what expert to use for each block.
    - num_tokens_post_pad (torch.Tensor): Torch tensor of size 1
        indicating the number of valid blocks with actual data to
        process. This is represented in terms of num tokens.

    Example:
    Let num_batches=5, max_tokens_per_batch=8, block_size=4, and
    expert_num_tokens=[2, 3, 0, 6, 8]. This expert_num_tokens tensor
    indicates that,
     - The first 2 tokens in the 0th batch are valid and the rest 6 are
     invalid (i.e. in the 2D hidden_states tensor of shape,
     [num_batches * max_tokens_per_batch, K], indices 0, 1 are valid)
     - The first 3 tokens in the 1st batch are valid. i.e. indices 8, 9, 10
     - 0 tokens in the 2nd batch are valid
     - first 6 tokens in the  3rd batch are valid. i.e. indices,
     24, 25, 26, 27, 28, 29
     - so on ...

     In this case,
      sorted_token_ids will be [0, 1, 40, 40,
                                8, 9, 10, 40,
                                24, 25, 26, 27,
                                28, 29, 40, 40,
                                32, 33, 34, 35,
                                36, 37, 38, 39,
                                40, 40, 40, 40,
                                (rest all 40, 40, 40, 40)
                                ...]
      Here, 40 represents an invalid index. as there is no token index 40.
      The gemm kernel using this sorted_token_ids is expected to skip the
      gemm computation when it encounters this invalid index.

      expert_ids will be [0, 1, 3, 3, 4, 5, 5, -1, -1, (rest all -1) ...]
      Here, -1 represents an invalid expert. The gemm kernel using this
      expert_ids is expected to skip the gemm computation when it encounters
      an expert of id -1.

      num_tokens_post_pad will be 24 as sorted_token_ids has valid entries
      until 24.

    """
    B = expert_num_tokens.size(0)
    device = expert_num_tokens.device

    # Round up so each batch can be split to blocks evenly.
    max_num_tokens_padded = B * round_up(max_tokens_per_batch, block_size)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    assert max_num_tokens_padded % block_size == 0
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=device)

    ops.batched_moe_align_block_size(
        max_tokens_per_batch,
        block_size,
        expert_num_tokens,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    return sorted_ids, expert_ids, num_tokens_post_pad


# sm8x patch: the image's _moe_C moe_align_block_size kernel is built for sm_100 only; on
# Ampere its launch fails (cudaErrorNoKernelImageForDevice, unchecked) and leaves the outputs
# unset, which breaks every MoE GEMM after it. Same contract in plain torch ops: fixed shapes,
# no host syncs (CUDA-graph safe). Routes are ordered by (expert, original row), i.e. stable.
_PORTABLE_ALIGN: dict[int, bool] = {}


def _needs_portable_align(device: torch.device) -> bool:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _PORTABLE_ALIGN:
        _PORTABLE_ALIGN[index] = torch.cuda.get_device_capability(index)[0] < 9
    return _PORTABLE_ALIGN[index]


def _moe_align_block_size_portable(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None,
    scatter_idx: torch.Tensor | None,
) -> None:
    device = topk_ids.device
    flat = topk_ids.reshape(-1).to(torch.int64)
    n = flat.numel()
    in_range = (flat >= 0) & (flat < num_experts)
    expert = torch.where(in_range, flat, 0)
    if expert_map is not None:
        local = expert_map.to(torch.int64)[expert]
        in_range = in_range & (local >= 0)
        expert = torch.where(in_range, local, 0)
    # Invalid routes get key num_experts: counted in a dummy bin, sorted last, never placed.
    key = torch.where(in_range, expert, num_experts)
    counts = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    counts.scatter_add_(0, key, torch.ones_like(key))
    counts = counts[:num_experts]
    padded = (counts + block_size - 1) // block_size * block_size
    ends = torch.cumsum(padded, 0)
    starts = ends - padded
    group_start = torch.cumsum(counts, 0) - counts
    order = torch.argsort(key, stable=True)
    sorted_key = key[order]
    valid_sorted = sorted_key < num_experts
    safe_key = torch.where(valid_sorted, sorted_key, 0)
    rank = torch.arange(n, device=device) - group_start[safe_key]
    cap = sorted_ids.numel()
    dest = torch.where(valid_sorted, starts[safe_key] + rank, cap)
    staging = torch.full((cap + 1,), n, dtype=torch.int32, device=device)
    staging.scatter_(0, dest, order.to(torch.int32))
    sorted_ids.copy_(staging[:cap])
    total = ends[-1] if num_experts > 0 else torch.zeros((), dtype=torch.int64, device=device)
    block_starts = torch.arange(expert_ids.numel(), device=device, dtype=torch.int64) * block_size
    block_expert = torch.searchsorted(ends, block_starts, right=True)
    # Blocks past the padded total point at expert 0 (valid memory); their rows are all
    # the sentinel n, so the GEMMs skip them.
    expert_ids.copy_(torch.where(block_starts < total, block_expert, 0).to(torch.int32))
    num_tokens_post_pad.copy_(total.reshape(1).to(torch.int32))
    if scatter_idx is not None:
        rows = torch.arange(n, device=device, dtype=torch.int32)
        scatter_idx.copy_(torch.where(in_range, rows, -1).view(-1, 1))


# sm8x patch, parallel path (prefill / large batches): the torch version above launches ~15 tiny kernels and
# an 8-pass radix sort per MoE visit; a single-program kernel does not scale past ~256 routes. Three parallel
# Triton passes reproduce its exact contract: per-chunk expert histograms, an exclusive prefix over chunks
# per expert (so routes keep the (expert, original row) stable order), then a per-chunk scatter. Deterministic,
# no atomics; sorted_ids/expert_ids/count are bit-identical to _moe_align_block_size_portable.
_ALIGN_CHUNK = 128
_ALIGN_BLOCK_B = 128


@triton.jit
def _align_fill_kernel(sorted_ptr, cap, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(sorted_ptr + i, tl.zeros((BLOCK,), tl.int32) + n, mask=i < cap)


@triton.jit
def _align_hist_kernel(ids_ptr, n, map_ptr, hist_ptr, E: tl.constexpr, BLOCK_E: tl.constexpr,
                       CHUNK: tl.constexpr, HAS_MAP: tl.constexpr):
    c = tl.program_id(0)
    idx = c * CHUNK + tl.arange(0, CHUNK)
    m = idx < n
    e = tl.load(ids_ptr + idx, mask=m, other=-1).to(tl.int32)
    ok = (e >= 0) & (e < E) & m
    if HAS_MAP:
        loc = tl.load(map_ptr + tl.where(ok, e, 0), mask=ok, other=-1).to(tl.int32)
        ok = ok & (loc >= 0)
        e = loc
    key = tl.where(ok, e, E)
    tl.store(hist_ptr + c * BLOCK_E + tl.arange(0, BLOCK_E), tl.histogram(key, BLOCK_E))


@triton.jit
def _align_prefix_kernel(hist_ptr, starts_ptr, ends_ptr, npad_ptr, nchunks,
                         E: tl.constexpr, BLOCK_E: tl.constexpr, BS: tl.constexpr):
    offs_e = tl.arange(0, BLOCK_E)
    total = tl.zeros((BLOCK_E,), tl.int32)
    for c in range(0, nchunks):
        total += tl.load(hist_ptr + c * BLOCK_E + offs_e)
    counts = tl.where(offs_e < E, total, 0)
    padded = ((counts + BS - 1) // BS) * BS
    ends = tl.cumsum(padded, 0)
    starts = ends - padded
    tl.store(starts_ptr + offs_e, starts, mask=offs_e < E)
    tl.store(ends_ptr + offs_e, ends, mask=offs_e < E)
    tl.store(npad_ptr, tl.sum(padded, 0))


@triton.jit
def _align_base_kernel(hist_ptr, starts_ptr, base_ptr, nchunks, BLOCK_E: tl.constexpr):
    e = tl.program_id(0)
    run = tl.load(starts_ptr + e)
    for c in range(0, nchunks):
        tl.store(base_ptr + c * BLOCK_E + e, run)
        run += tl.load(hist_ptr + c * BLOCK_E + e)


@triton.jit
def _align_blocks_kernel(ends_ptr, expert_ids_ptr, num_blocks, E: tl.constexpr,
                         BLOCK_E: tl.constexpr, BS: tl.constexpr, BLOCK_B: tl.constexpr):
    bi = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_e = tl.arange(0, BLOCK_E)
    ends = tl.load(ends_ptr + offs_e, mask=offs_e < E, other=0)
    le = (ends[None, :] <= (bi * BS)[:, None]) & (offs_e[None, :] < E)
    ex = tl.sum(le.to(tl.int32), 1)
    ex = tl.where(bi * BS < tl.load(ends_ptr + E - 1), ex, 0)
    tl.store(expert_ids_ptr + bi, ex, mask=bi < num_blocks)


@triton.jit
def _align_place_kernel(ids_ptr, n, map_ptr, base_ptr, sorted_ptr, cap, scatter_ptr,
                        E: tl.constexpr, BLOCK_E: tl.constexpr, CHUNK: tl.constexpr,
                        HAS_MAP: tl.constexpr, HAS_SCATTER: tl.constexpr):
    c = tl.program_id(0)
    lane = tl.arange(0, CHUNK)
    idx = c * CHUNK + lane
    m = idx < n
    e = tl.load(ids_ptr + idx, mask=m, other=-1).to(tl.int32)
    ok = (e >= 0) & (e < E) & m
    if HAS_MAP:
        loc = tl.load(map_ptr + tl.where(ok, e, 0), mask=ok, other=-1).to(tl.int32)
        ok = ok & (loc >= 0)
        e = loc
    key = tl.where(ok, e, E)
    same = (key[:, None] == key[None, :]) & (lane[None, :] < lane[:, None])
    rank = tl.sum(same.to(tl.int32), 1)
    dest = tl.load(base_ptr + c * BLOCK_E + tl.where(ok, key, 0), mask=ok, other=0) + rank
    tl.store(sorted_ptr + dest, idx.to(tl.int32), mask=ok & (dest < cap))
    if HAS_SCATTER:
        tl.store(scatter_ptr + idx, tl.where(ok, idx, -1).to(tl.int32), mask=m)


def _moe_align_block_size_parallel(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None,
    scatter_idx: torch.Tensor | None,
) -> None:
    dev = topk_ids.device
    flat = topk_ids.reshape(-1)
    n = flat.numel()
    chunk = _ALIGN_CHUNK
    nchunks = triton.cdiv(n, chunk)
    block_e = triton.next_power_of_2(num_experts + 1)
    hist = torch.empty((nchunks, block_e), dtype=torch.int32, device=dev)
    base = torch.empty((nchunks, block_e), dtype=torch.int32, device=dev)
    starts = torch.empty((block_e,), dtype=torch.int32, device=dev)
    ends = torch.empty((block_e,), dtype=torch.int32, device=dev)
    map_arg = expert_map if expert_map is not None else flat
    _align_fill_kernel[(triton.cdiv(sorted_ids.numel(), 2048),)](
        sorted_ids, sorted_ids.numel(), n, BLOCK=2048, num_warps=4
    )
    _align_hist_kernel[(nchunks,)](
        flat, n, map_arg, hist, E=num_experts, BLOCK_E=block_e, CHUNK=chunk,
        HAS_MAP=expert_map is not None, num_warps=4,
    )
    _align_prefix_kernel[(1,)](
        hist, starts, ends, num_tokens_post_pad, nchunks,
        E=num_experts, BLOCK_E=block_e, BS=block_size, num_warps=4,
    )
    _align_base_kernel[(num_experts,)](hist, starts, base, nchunks, BLOCK_E=block_e, num_warps=1)
    _align_blocks_kernel[(triton.cdiv(expert_ids.numel(), _ALIGN_BLOCK_B),)](
        ends, expert_ids, expert_ids.numel(), E=num_experts, BLOCK_E=block_e,
        BS=block_size, BLOCK_B=_ALIGN_BLOCK_B, num_warps=4,
    )
    _align_place_kernel[(nchunks,)](
        flat, n, map_arg, base, sorted_ids, sorted_ids.numel(),
        scatter_idx if scatter_idx is not None else sorted_ids,
        E=num_experts, BLOCK_E=block_e, CHUNK=chunk,
        HAS_MAP=expert_map is not None, HAS_SCATTER=scatter_idx is not None, num_warps=4,
    )


# sm8x patch, fast path: the torch version above launches ~15 tiny kernels; at 48 MoE visits per
# token that was ~2 ms of a ~12 ms decode step. Decode-sized batches use this single-program
# Triton kernel instead (same contract; routes ordered by (expert, original row), deterministic).
_TRITON_ALIGN_MAX_ROUTES = 4096


@triton.jit
def _moe_align_kernel(
    ids_ptr, n, expert_map_ptr, scratch_ptr, sorted_ptr, cap, expert_ids_ptr, num_blocks,
    npad_ptr, scatter_ptr,
    E: tl.constexpr, BLOCK_E: tl.constexpr, BS: tl.constexpr, CHUNK: tl.constexpr,
    HAS_MAP: tl.constexpr, HAS_SCATTER: tl.constexpr, FILL: tl.constexpr,
):
    offs_e = tl.arange(0, BLOCK_E)
    offs_c = tl.arange(0, CHUNK)
    # 1) per-expert counts (invalid routes go to bin E)
    counts = tl.zeros([BLOCK_E], dtype=tl.int32)
    for c0 in range(0, n, CHUNK):
        idx = c0 + offs_c
        m = idx < n
        e = tl.load(ids_ptr + idx, mask=m, other=-1).to(tl.int32)
        ok = (e >= 0) & (e < E) & m
        if HAS_MAP:
            loc = tl.load(expert_map_ptr + tl.where(ok, e, 0), mask=ok, other=-1).to(tl.int32)
            ok = ok & (loc >= 0)
            e = loc
        key = tl.where(ok, e, E)
        counts += tl.histogram(tl.where(m, key, E), BLOCK_E)
    counts = tl.where(offs_e < E, counts, 0)
    padded = (counts + BS - 1) // BS * BS
    ends = tl.cumsum(padded, 0)
    starts = ends - padded
    total = tl.sum(padded, 0)
    tl.store(npad_ptr, total)
    # scratch[0:BLOCK_E] = running write offset per expert
    tl.store(scratch_ptr + offs_e, starts)
    # 2) sentinel fill
    for f0 in range(0, cap, FILL):
        fi = f0 + tl.arange(0, FILL)
        tl.store(sorted_ptr + fi, tl.zeros([FILL], dtype=tl.int32) + n, mask=fi < cap)
    tl.debug_barrier()
    # 3) place routes: rank within chunk via pairwise compare, plus running offset
    lower = offs_c[None, :] < offs_c[:, None]
    for c0 in range(0, n, CHUNK):
        idx = c0 + offs_c
        m = idx < n
        e = tl.load(ids_ptr + idx, mask=m, other=-1).to(tl.int32)
        ok = (e >= 0) & (e < E) & m
        if HAS_MAP:
            loc = tl.load(expert_map_ptr + tl.where(ok, e, 0), mask=ok, other=-1).to(tl.int32)
            ok = ok & (loc >= 0)
            e = loc
        key = tl.where(ok, e, E)
        same = (key[:, None] == key[None, :]) & lower
        rank = tl.sum(same.to(tl.int32), 1)
        base = tl.load(scratch_ptr + tl.where(ok, key, 0), mask=ok, other=0)
        tl.store(sorted_ptr + base + rank, idx.to(tl.int32), mask=ok)
        if HAS_SCATTER:
            tl.store(scatter_ptr + idx, tl.where(ok, idx, -1).to(tl.int32), mask=m)
        tl.debug_barrier()
        hist = tl.histogram(key, BLOCK_E)
        cur = tl.load(scratch_ptr + offs_e)
        tl.debug_barrier()
        tl.store(scratch_ptr + offs_e, cur + tl.where(offs_e < E, hist, 0))
        tl.debug_barrier()
    # 4) expert of each block: number of experts whose padded range ends at or before the block start
    for b0 in range(0, num_blocks, CHUNK):
        bi = b0 + offs_c
        bstart = bi * BS
        le = (ends[None, :] <= bstart[:, None]) & (offs_e[None, :] < E)
        ex = tl.sum(le.to(tl.int32), 1)
        ex = tl.where(bstart < total, ex, 0)
        tl.store(expert_ids_ptr + bi, ex, mask=bi < num_blocks)


def _moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None,
    scatter_idx: torch.Tensor | None,
) -> None:
    import triton.language as tl  # noqa: F401  (kernel globals)

    flat = topk_ids.reshape(-1)
    n = flat.numel()
    block_e = triton.next_power_of_2(num_experts + 1)
    scratch = torch.empty(block_e, dtype=torch.int32, device=flat.device)
    _moe_align_kernel[(1,)](
        flat, n, expert_map if expert_map is not None else flat, scratch,
        sorted_ids, sorted_ids.numel(), expert_ids, expert_ids.numel(),
        num_tokens_post_pad, scatter_idx if scatter_idx is not None else flat,
        E=num_experts, BLOCK_E=block_e, BS=block_size, CHUNK=128,
        HAS_MAP=expert_map is not None, HAS_SCATTER=scatter_idx is not None, FILL=1024,
        num_warps=4,
    )
