#!/usr/bin/env bash
# Round 3: FAR_ATTN + CEAR tail pipeline in 4 pieces of 512 rows (default 2 x 1024): cold 500K and 16K.
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); cd "$R"; O=$R/experiments/prefill/results/far_p4_512k; mkdir -p $O
PROFILE=512k MODEL_DIR=$HOME/ai/models/inkling2-mini DOCKER_EXTRA="-e INKLING2_FAR_ATTN=1 -e VLLM_CEAR_TAIL_PIECE_ROWS=512 -e VLLM_CEAR_TAIL_MAX_PIECES=4" \
  bash launch/2gpu.sh > $O/boot.log 2>&1 || { echo "BOOT FAIL"; tail -5 $O/boot.log; exit 1; }
docker exec inkling2 env | grep -E "FAR|PIECE"
python3 tests/needle.py 16384 --depth 0.5 | tail -1 | tee $O/needle_16k.json
python3 tests/needle.py 500000 --depth 0.5 | tail -1 | tee $O/needle_500k.json
docker stop inkling2 >/dev/null; echo "== done3"
