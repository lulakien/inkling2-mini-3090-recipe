#!/usr/bin/env bash
# Bit-exactness harnesses for the two new prefill kernels, run inside the pinned image on one GPU.
# test_align.py:    parallel MoE align vs the portable torch contract (35 cases, torch.equal)
# test_attn_res.py: 2-pass attn_res vs the stock 3-pass kernel (72 cases, torch.equal)
# test_e4m3.py:     software FP8 E4M3 encode/decode (KV_DTYPE=fp8 on sm_86) vs torch.float8_e4m3fn, bit-exact
set -euo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); . "$R/launch/image.env"
TR=/usr/local/lib/python3.14/dist-packages/triton   # sm_86: the two Triton overlays (no griddepcontrol / PDL)
for t in test_align.py test_attn_res.py test_e4m3.py; do
  docker run --rm --gpus "device=${GPU:-0}" --ipc=host -v "$R/patches:/p:ro" -v "$R/tests/kernels:/k:ro" \
    -v "$R/patches/gdc.py:$TR/language/extra/cuda/gdc.py:ro" -v "$R/patches/triton_driver.py:$TR/backends/nvidia/driver.py:ro" \
    --entrypoint python3 "$IMAGE" "/k/$t"
done
