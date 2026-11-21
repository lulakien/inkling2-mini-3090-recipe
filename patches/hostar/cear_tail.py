# SPDX-License-Identifier: Apache-2.0
"""Inkling2 (sm8x) prefill: the row-wise layer tail as one pipelined op on the copy engines.

One loop visit's tail starts at the attention output (row-parallel `wo_ud`, no
reduce) and ends at the AttnRes mix that produces the next visit's inputs:

    attn_partial --AR--> conv(attn stream) + mix --MoE--> combine --AR--> conv(mlp stream) + mix(+residual)

Both ARs are TP2 sums of one bf16 ``(T, 2304)`` tensor (9.4 MB at T=2048) and
~43% of prefill time is spent in them (96 ARs per 2048-token chunk). This op
cuts the rows into P pieces (default 2 x 1024 rows) and runs the DMAs of piece i
on the copy engines (cear.py) while the SMs run piece i+1's conv + MoE expert
GEMMs, so only piece 0's attention AR and the last piece's MoE AR stay exposed.

Row independence: every op in the region is per-token *except* the hidden-stream
short conv, whose tap at lag 1..3 comes from a token that may sit in the
previous piece. That case is exactly what the conv-state cache is for (the
kernel already writes every token's input to its slot and reads lag taps
outside the current forward from the paged cache), so a piece boundary behaves
like a past-forward boundary. The one thing that breaks is `query_start`, which
the kernel compares against *its own* row offsets: pass it piece-relative
(``max(query_start - r0, 0)``) or the "does this tile need cache taps" test
(`tile_reads_cache`) misfires and the conv silently drops the lag taps of the
piece's first rows. Everything else is handled: see NOTES.md.

Numerics vs the unpipelined path, with P pieces:
  * attention AR / MoE AR: bit-identical (fp32 sum of two bf16 values, one
    rounding -- what NCCL's TP2 sum does; same as hostar/cear).
  * short conv, AttnRes mix, RMSNorm, residual updates: bit-identical (the
    per-token math and its order are unchanged; lag taps come from the cache
    instead of `x` but hold the same bf16 values).
  * MoE: the expert GEMM runs at T/P rows, so the Humming kernel's tiling /
    reduction order changes (same class of change as VLLM_CEAR=1 and as
    changing the prefill chunk size). Bit-identical only if the expert GEMM is
    row-count invariant.

Fallback: decode batches, CUDA-graph capture, small M, batches without a
conv-state cache, and runs with routed-expert capture use the stock path.

Runtime knobs (all read at import):
  VLLM_CEAR_TAIL=1             enable the patch (default 0; 1 = on)
  VLLM_CEAR_TAIL_STAGE=2       1 = attention AR only, 2 = also the MoE AR
  VLLM_CEAR_TAIL_PIECE_ROWS=1024
  VLLM_CEAR_TAIL_MAX_PIECES=2
  VLLM_CEAR_TAIL_MIN_TOKENS=1536   below this, no pipelining
  VLLM_CEAR_TAIL_MIN_PIECE=512     keep short_conv/attn_res on their prefill kernels
"""

import os
import sys

import torch

ENABLED = os.environ.get("VLLM_CEAR_TAIL", "0") == "1"
STAGE = int(os.environ.get("VLLM_CEAR_TAIL_STAGE", "2"))
PIECE_ROWS = int(os.environ.get("VLLM_CEAR_TAIL_PIECE_ROWS", "1024"))
MAX_PIECES = int(os.environ.get("VLLM_CEAR_TAIL_MAX_PIECES", "2"))
MIN_TOKENS = int(os.environ.get("VLLM_CEAR_TAIL_MIN_TOKENS", "1536"))
MIN_PIECE = int(os.environ.get("VLLM_CEAR_TAIL_MIN_PIECE", "512"))

_LOOP = "vllm.models.inkling2.nvidia.loop"

# (id(metadata), row_start) -> (metadata, piece-relative query_start); the
# strong ref keeps `id` unique, and the `is` check catches a recycled id.
_QS: dict = {}


# ----------------------------------------------------------------------------- protocol
def _ce():
    from vllm.distributed import get_tp_group

    return getattr(get_tp_group().device_communicator, "cear", None)


def spans(n: int, row_bytes: int) -> tuple:
    """Row pieces of an `n`-row AR, or () when the AR cannot be pipelined.

    Both ARs of a visit place their pieces at distinct byte offsets of the
    cear slot (`r0 * row_bytes` and `n * row_bytes + r0 * row_bytes`) because
    the received pieces land in one device tmp buffer, so 2 * n * row_bytes
    must fit the slot.
    """
    pieces = min(max(-(-n // PIECE_ROWS), 1), MAX_PIECES)
    if pieces < 2:
        return ()
    bounds = [n * i // pieces for i in range(pieces + 1)]
    if min(hi - lo for lo, hi in zip(bounds[:-1], bounds[1:])) < MIN_PIECE:
        return ()
    ce = _ce()
    if ce is None or 2 * n * row_bytes > ce.cap:
        return ()
    return tuple(zip(bounds[:-1], bounds[1:]))


def _qs_for(m, r0: int, r1: int) -> torch.Tensor:
    key = (id(m), r0)
    hit = _QS.get(key)
    if hit is not None and hit[0] is m:
        return hit[1]
    qs = torch.sub(m.query_start[r0:r1], r0).clamp_(min=0)
    if len(_QS) > 256:
        _QS.clear()
    _QS[key] = (m, qs)
    return qs


# ----------------------------------------------------------------------------- the op
def tail(
    layer,
    owner,
    positions: torch.Tensor,
    attn_partial: torch.Tensor,
    attn_out: torch.Tensor,
    stack1: torch.Tensor,
    last1: torch.Tensor | None,
    queries1: torch.Tensor,
    norm_w1: torch.Tensor | None,
    norm_eps1: float,
    stack2: torch.Tensor,
    out2: torch.Tensor,
    queries2: torch.Tensor,
    norm_w2: torch.Tensor | None,
    norm_eps2: float,
    residual2: torch.Tensor | None,
    addend2: torch.Tensor | None,
    messages: torch.Tensor | None,
    pass_idx: int,
) -> torch.Tensor:
    """Pipelined tail of one visit: `attn_partial` (this rank's, pre-AR) ->
    AR -> conv(attn)+mix -> MoE -> combine -> AR -> conv(mlp)+mix(+residual).
    Returns the `(Q2, T, D)` mixes (bf16 with `norm_w2`, else fp32) exactly as
    the two `conv_mix` calls + `layer.mlp` would."""
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.models.inkling2.nvidia.ops.conv_mix import conv_mix
    from vllm.models.inkling2.nvidia.ops.moe import combine
    from vllm.models.inkling2.nvidia.sconv import conv_metadata

    ce = _ce()
    n, d = attn_partial.shape
    row = d * attn_partial.element_size()
    pieces = spans(n, row)
    assert pieces and ce is not None
    m = conv_metadata(owner)
    assert m is not None
    off_a, width_a = owner.stream_ranges[layer.attn_sconv.stream]
    off_m, width_m = owner.stream_ranges[layer.mlp_sconv.stream]
    cw_a, cw_m = layer.attn_sconv.conv_weight, layer.mlp_sconv.conv_weight
    cache = owner.kv_cache
    tensor_bytes = n * row
    outs = torch.empty(
        (queries2.shape[0], n, d),
        dtype=attn_partial.dtype if norm_w2 is not None else torch.float32,
        device=attn_partial.device,
    )

    def conv1(x, r0, r1):
        mix = conv_mix(
            x, cw_a, cache, positions[r0:r1], m.block_table,
            m.seq_idx[r0:r1], m.slot_mapping[r0:r1], _qs_for(m, r0, r1),
            off_a, width_a, attn_out[r0:r1], None, None,
            stack1[:, r0:r1], None if last1 is None else last1[r0:r1],
            queries1, norm_w1, norm_eps1,
        )
        if messages is not None:
            mix = layer.blend(
                slice(3, 4), mix, messages[3:, r0:r1],
                layer.mlp_norm.weight, layer.mlp_norm.eps,
            )
        return mix

    def conv2(x, r0, r1):
        mix = conv_mix(
            x, cw_m, cache, positions[r0:r1], m.block_table,
            m.seq_idx[r0:r1], m.slot_mapping[r0:r1], _qs_for(m, r0, r1),
            off_m, width_m, out2[r0:r1],
            None if residual2 is None else residual2[r0:r1],
            None if addend2 is None else addend2[r0:r1],
            stack2[:, r0:r1], None, queries2, norm_w2, norm_eps2,
        )
        outs[:, r0:r1].copy_(mix)

    def moe(x, r0, r1):
        """This rank's combined expert rows for piece rows [r0, r1)."""
        from vllm.forward_context import get_forward_context, is_forward_context_available

        ctx = get_forward_context() if is_forward_context_available() else None
        pad = getattr(ctx, "is_padding", None) if ctx is not None else None
        if pad is not None:
            ctx.is_padding = pad[r0:r1]
        try:
            down, src2dst = layer.mlp(x, reduce=False, pass_idx=pass_idx)
        finally:
            if pad is not None:
                ctx.is_padding = pad
        # One bf16 rounding per token, before the AR (as layer.mlp does).
        return combine(down, src2dst, layer.mlp.experts.topk, x.dtype).contiguous()

    ce.begin()
    try:
        # Every piece of the attention AR is queued up front: each rank's D2H
        # then streams while its SMs run piece 0, so the peer's H2D for piece i
        # is already under way when this rank consumes piece i.
        for r0, r1 in pieces:
            ce.send(attn_partial[r0:r1], r0 * row)
        pend = None  # (r0, r1, combined MoE rows) waiting for its AR + conv2
        for r0, r1 in pieces:
            peer = ce.recv(attn_partial[r0:r1], r0 * row)
            red = torch.add(attn_partial[r0:r1], peer)
            x = conv1(red, r0, r1)[0]
            p = moe(x, r0, r1)
            if STAGE >= 2:
                ce.send(p, tensor_bytes + r0 * row)
                # Depth-1 lag: the recv (and its stream wait) lands *after*
                # this piece's expert GEMMs are queued, so the previous piece's
                # MoE AR transfer is hidden behind them.
                prev, pend = pend, (r0, r1, p)
            else:
                prev, pend = (r0, r1, tensor_model_parallel_all_reduce(p)), None
            if prev is not None:
                r0p, r1p, pp = prev
                conv2(torch.add(pp, ce.recv(pp, tensor_bytes + r0p * row))
                      if STAGE >= 2 else pp, r0p, r1p)
        if pend is not None:
            r0p, r1p, pp = pend
            conv2(torch.add(pp, ce.recv(pp, tensor_bytes + r0p * row)), r0p, r1p)
    finally:
        ce.end()
    return outs


# ----------------------------------------------------------------------------- dispatch
def ready(layer, owner, pass_idx: int, attn_partial: torch.Tensor) -> bool:
    if not ENABLED or torch.cuda.is_current_stream_capturing():
        return False
    n, d = attn_partial.shape
    if n < MIN_TOKENS or attn_partial.dtype != torch.bfloat16 or not attn_partial.is_contiguous():
        return False
    if not spans(n, d * attn_partial.element_size()):
        return False
    experts = layer.mlp.experts
    if experts.sm100 or experts.sm120 or experts.humming is None:
        return False
    if layer.mlp.routing_captures[pass_idx].capture_fn is not None:
        return False
    from vllm.models.inkling2.nvidia.sconv import conv_metadata

    return conv_metadata(owner) is not None and owner.kv_cache.numel() > 0


def _run_layer_piped(
    self, i, pass_idx, stack, num_blocks, inputs, bank, next_query, next_norm,
    positions, tau, plan,
):
    """`Inkling2LoopModel._run_layer` with the row-wise tail replaced by
    `tail()` when it can be pipelined; the stock path otherwise."""
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.models.inkling2.nvidia.loop import Inkling2LoopDecoderLayer
    from vllm.models.inkling2.nvidia.ops.attn_res import attn_res

    layer = self.layers[i]
    partial = stack[num_blocks]
    owner = layer.conv_states[pass_idx]
    (attn_fused, attn_two_shot), (moe_fused, moe_two_shot) = plan
    messages = None
    if bank is not None:
        assert isinstance(layer, Inkling2LoopDecoderLayer) and inputs is not None
        messages = attn_res(
            bank[:1], bank[1], None,
            self.message_queries[i - self.loop_core.start], None, 0.0, True,
        )
        norm = layer.attn_norm
        dests, norm_weight = inputs, norm.weight
        if layer.is_local:
            dests = dests.expand(3, -1, -1)
            norm_weight = norm_weight.expand(3, -1)
        attn_inputs = layer.blend(
            slice(0, 3), dests, messages[:3], norm_weight, norm.eps
        )
    elif inputs is None:
        assert num_blocks == 0
        attn_inputs = [
            layer.attn_norm(partial, j) for j in range(layer.num_attn_inputs)
        ]
    else:
        attn_inputs = inputs
    attn_partial = layer.attention(positions, attn_inputs, tau, pass_idx=pass_idx)

    first = i % self.block_size == 0
    attn_out = torch.empty_like(partial)
    residual = None if first else partial
    if residual is None:
        num_blocks += 1
    if not attn_fused and not moe_fused and ready(layer, owner, pass_idx, attn_partial):
        outputs = tail(
            layer, owner, positions, attn_partial, attn_out,
            stack[:num_blocks], None if first else partial,
            self.attn_res.mlp_queries[i],
            None if messages is not None else layer.mlp_norm.weight,
            0.0 if messages is not None else layer.mlp_norm.eps,
            stack[:num_blocks], stack[num_blocks], next_query,
            None if next_norm is None else next_norm.weight,
            0.0 if next_norm is None else next_norm.eps,
            residual, attn_out, messages, pass_idx,
        )
    else:
        mix_args = (
            positions, attn_out, stack[:num_blocks],
            None if first else partial, self.attn_res.mlp_queries[i],
            None if messages is not None else layer.mlp_norm,
        )
        if attn_fused:
            assert self.allreduce_workspace is not None
            mlp_in = layer.attn_sconv.all_reduce_mix(
                self.allreduce_workspace, *mix_args,
                x=attn_partial, two_shot=attn_two_shot, owner=owner,
            )
        else:
            mlp_in = layer.attn_sconv.conv_mix(
                tensor_model_parallel_all_reduce(attn_partial), *mix_args, owner=owner
            )
        if messages is not None:
            mlp_in = layer.blend(
                slice(3, 4), mlp_in, messages[3:],
                layer.mlp_norm.weight, layer.mlp_norm.eps,
            )
        mix_args = (
            positions, stack[num_blocks], stack[:num_blocks], None,
            next_query, next_norm,
        )
        if moe_fused:
            assert self.allreduce_workspace is not None
            outputs = layer.mlp_sconv.all_reduce_mix(
                self.allreduce_workspace, *mix_args,
                combine=layer.mlp(mlp_in[0], reduce=False, pass_idx=pass_idx),
                addend=attn_out, residual=residual,
                two_shot=moe_two_shot, owner=owner,
            )
        else:
            # The MoE output is all-reduced (in bf16) inside layer.mlp.
            # VLLM_CEAR=1's MoE-only row piecing stays the fallback (huge
            # batches, profile runs without a conv-state cache).
            legacy = sys.modules.get("cear_moe")
            check = None
            if legacy is not None:
                check = getattr(legacy, "legacy_eligible", None) or legacy.eligible
            if check is not None and check(mlp_in[0]):
                moe_out = legacy.moe_pieced(layer.mlp, mlp_in[0], pass_idx)
            else:
                moe_out = layer.mlp(mlp_in[0], pass_idx=pass_idx)
            outputs = layer.mlp_sconv.conv_mix(
                moe_out, *mix_args,
                addend=attn_out, residual=residual, owner=owner,
            )
    return num_blocks, outputs


def install() -> bool:
    """Patch `Inkling2LoopModel._run_layer` (idempotent)."""
    if not ENABLED:
        return False
    import importlib

    mod = sys.modules.get(_LOOP)
    if mod is None:
        try:
            mod = importlib.import_module(_LOOP)
        except Exception:
            return False
    cls = getattr(mod, "Inkling2LoopModel", None)
    if cls is None or getattr(cls, "_cear_tail", False):
        return cls is not None
    cls._run_layer = _run_layer_piped
    cls._cear_tail = True
    print(
        f"[cear_tail] prefill tail pipeline installed: stage {STAGE}, "
        f"{MIN_TOKENS}+ tokens, {PIECE_ROWS}-row pieces (max {MAX_PIECES})",
        flush=True,
    )
    return True
