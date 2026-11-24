#!/usr/bin/env bash
# Round 4: PROFILE=1m (FP8 KV): MNBT 1024 (shipped) vs 2048 (CEAR tail pipeline engages at >=1536 tokens) vs 2048+FAR.
# KV pool and a cold 250K needle each.
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); cd "$R"; M=$HOME/ai/models/inkling2-mini
for cfg in base:1024: mnbt2048:2048: mnbt2048_far:2048:-e+INKLING2_FAR_ATTN=1; do
  IFS=: read L N X <<< "$cfg"; O=$R/experiments/prefill/results/1m_$L; mkdir -p $O; echo "== $L"
  PROFILE=1m MNBT=$N MODEL_DIR=$M DOCKER_EXTRA="${X//+/ }" bash launch/2gpu.sh > $O/boot.log 2>&1 || { echo "BOOT FAIL $L"; tail -5 $O/boot.log; docker stop inkling2 >/dev/null 2>&1; continue; }
  docker logs inkling2 2>&1 | grep -E "GPU KV cache size|max-num-batched|cear|CEAR" | tail -4
  python3 tests/needle.py 250000 --depth 0.5 | tail -1 | tee $O/needle_250k.json
  docker stop inkling2 >/dev/null
done; echo "== done4"
