#!/usr/bin/env bash
# bash experiments/prefill/relfar/run.sh [PAST] [SPLITS] [CFG_JSON]  (GPU 1, pinned image, triton overlays)
R=$(cd "$(dirname "$0")/../../.." && pwd); . "$R/launch/image.env"; P=$R/patches
TR=/usr/local/lib/python3.14/dist-packages/triton
docker run --rm -e FP8 --gpus device=1 --ipc=host -v "$R":/x:ro -v "$HOME/.cache/inkling2-vllm":/root/.cache \
  -v $P/gdc.py:$TR/language/extra/cuda/gdc.py:ro -v $P/triton_driver.py:$TR/backends/nvidia/driver.py:ro \
  --entrypoint python3 "$IMAGE" /x/experiments/prefill/relfar/bench.py "$@"
