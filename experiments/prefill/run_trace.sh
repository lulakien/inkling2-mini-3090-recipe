#!/usr/bin/env bash
# Torch-profiler trace of two prefill steps ~485K deep in a cold 500K prompt (PROFILE=512k, bf16), plus the
# per-kernel GPU time summary. Usage: bash experiments/prefill/run_trace.sh [LABEL]
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); cd "$R"
LABEL=${1:-trace_512k}; O=$R/experiments/prefill/results/$LABEL; mkdir -p "$O"
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/inkling2-vllm}; PD=/root/.cache/prof_$LABEL
mkdir -p "$CACHE_DIR/prof_$LABEL"
PROF="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PD\",\"torch_profiler_with_stack\":false,\"ignore_frontend\":true,\"delay_iterations\":${DELAY:-236},\"max_iterations\":2,\"torch_profiler_use_gzip\":false}"
PROFILE=512k MODEL_DIR=${MODEL_DIR:-$HOME/ai/models/inkling2-mini} EXTRA="--profiler-config $PROF ${EXTRA_ARGS:-}" \
  bash launch/2gpu.sh > "$O/boot.log" 2>&1 || { echo "BOOT FAIL"; tail -20 "$O/boot.log"; exit 1; }
curl -s -X POST http://127.0.0.1:8000/start_profile > /dev/null
python3 tests/needle.py 500000 --depth 0.5 | tail -1 | tee "$O/needle_500k.json"
curl -s -X POST http://127.0.0.1:8000/stop_profile > /dev/null; sleep 5
docker logs inkling2 > "$O/server.log" 2>&1
cp -r "$CACHE_DIR/prof_$LABEL" "$O/raw" 2>/dev/null
docker stop inkling2 > /dev/null
python3 -I "$R/experiments/prefill/summarize_trace.py" "$O/raw" | tee "$O/kernels.txt"
