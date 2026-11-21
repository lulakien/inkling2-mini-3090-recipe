"""Shadow of `hostar/cear_moe.py` for the prefill-tail experiment (work/prefill-pipe).

Keeps the VLLM_CEAR=1 MoE-only row piecing (so `CEAR=1` alone still reproduces
today's baseline) and installs the pipelined layer tail (`cear_tail.py`) when
`VLLM_CEAR_TAIL=1`.

Why this file: `patches/loop.py` imports `cear_moe` from `/hostar` at import
time, before `Inkling2LoopModel` exists, so installation is retried on the
first `eligible()` call (which `_run_layer` makes on the first layer visit of
the process; from the second visit on the pipelined `_run_layer` is in place).

Mount (nested inside the launcher's `-v .../patches/hostar:/hostar:ro`):
  -v <this dir>/cear_moe.py:/hostar/cear_moe.py:ro
  -v <this dir>/cear_tail.py:/hostar/cear_tail.py:ro
with CEAR=1 CEAR_TAIL=1.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cear_tail  # noqa: E402

PIECES = int(os.environ.get("VLLM_CEAR_MOE_PIECES", "2"))
MIN_TOKENS = int(os.environ.get("VLLM_CEAR_MIN_TOKENS", "1024"))


def _eligible(x) -> bool:
    from vllm.distributed import get_tp_group

    if x.shape[0] < MIN_TOKENS or torch_capturing():
        return False
    ce = getattr(get_tp_group().device_communicator, "cear", None)
    return ce is not None and x.shape[0] * x.shape[1] * x.element_size() <= ce.cap


def torch_capturing() -> bool:
    import torch

    return torch.cuda.is_current_stream_capturing()


def eligible(x) -> bool:
    # First call of the process: install the pipelined _run_layer and let this
    # one visit run the old tail (its attention AR already happened).
    if cear_tail.ENABLED and cear_tail.install():
        return False
    return _eligible(x)


# The pipelined `_run_layer` falls back to the MoE-only piecing through this
# name (it must not call `eligible`, which would retry the install).
legacy_eligible = _eligible


def moe_pieced(mlp, x, pass_idx: int):
    """VLLM_CEAR=1: the MoE all-reduce in PIECES row pieces (unchanged)."""
    import torch
    from vllm.distributed import get_tp_group
    from vllm.forward_context import get_forward_context, is_forward_context_available
    from vllm.models.inkling2.nvidia.ops.moe import combine

    ce = get_tp_group().device_communicator.cear
    x = x.contiguous()
    t = x.shape[0]
    bounds = [t * i // PIECES for i in range(PIECES + 1)]
    out = torch.empty_like(x)
    row_bytes = x.shape[1] * x.element_size()
    ctx = get_forward_context() if is_forward_context_available() else None
    pad = getattr(ctx, "is_padding", None) if ctx is not None else None
    ce.begin()
    parts = []
    try:
        for r0, r1 in zip(bounds[:-1], bounds[1:]):
            if pad is not None:
                ctx.is_padding = pad[r0:r1]
            down, src2dst = mlp(x[r0:r1], reduce=False, pass_idx=pass_idx)
            p = combine(down, src2dst, mlp.experts.topk, x.dtype).contiguous()
            ce.send(p, r0 * row_bytes)
            parts.append((r0, r1, p))
    finally:
        if pad is not None:
            ctx.is_padding = pad
    for r0, r1, p in parts:
        torch.add(p, ce.recv(p, r0 * row_bytes), out=out[r0:r1])
    ce.end()
    return out

