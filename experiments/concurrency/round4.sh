#!/usr/bin/env bash
# Round 4: warm-up JIT check, 4096-token prefill chunks (with/without SCONV_TAIL), 1-GPU draft schedule.
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); cd "$R"; E=experiments/concurrency; M=$HOME/ai/models/inkling2-mini
jit() { docker logs "$1" 2>&1 | grep -c 'JIT compilation during inference'; }

O=$E/results/warm2; mkdir -p "$O"
MODEL_DIR=$M bash launch/2gpu.sh 2>&1 | tail -1
a=$(jit inkling2); python3 tests/concurrency.py "$O/c.json" --levels 1,2,3,4,6,8 >/dev/null 2>&1; b=$(jit inkling2)
echo "warm2 JIT after warm-up $a, after concurrency $b"
docker stop inkling2 >/dev/null

prefill() {  # label, then env assignments
  local lab=$1; shift; local O=$E/results/$lab; mkdir -p "$O"
  if env "$@" MODEL_DIR=$M bash launch/2gpu.sh > "$O/boot.log" 2>&1; then
    docker logs inkling2 2>&1 | grep -E "KV cache size" | sed -E 's/^.*\] //' | sed "s/^/$lab /"
    python3 tests/prefill_ttft.py "$O/prefill.json" --sizes 16384,65536,253952 --reps 1 | tail -4 | sed "s/^/$lab /"
  else
    echo "$lab BOOT FAIL: $(docker logs inkling2 2>&1 | grep -E 'Error|estimated' | tail -1 | cut -c1-250)"
  fi
  docker stop inkling2 >/dev/null 2>&1
}
prefill mnbt4096_tail MNBT=4096 SCONV_TAIL=1
prefill mnbt2048_tail SCONV_TAIL=1
prefill mnbt4096 MNBT=4096

one() {  # label, SPEC_SCHED value ("default" keeps serve.sh's default)
  local lab=$1; local O=$E/results/1gpu_$lab; mkdir -p "$O"
  if [ "$2" = default ]; then env -u SPEC_SCHED MODEL_DIR=$M bash launch/1gpu.sh > "$O/boot.log" 2>&1
  else SPEC_SCHED="$2" MODEL_DIR=$M bash launch/1gpu.sh > "$O/boot.log" 2>&1; fi || { echo "1gpu $lab BOOT FAIL"; return; }
  python3 tests/concurrency.py "$O/c.json" --levels 1,2,3,4 >/dev/null 2>&1
  python3 -I -c 'import json,sys;d=json.load(open(sys.argv[1]));print("1gpu",sys.argv[2],{k:(v["aggregate_tok_s"],v["mean_ttft_s"]) for k,v in d.items()})' "$O/c.json" "$lab"
  docker stop inkling2-1gpu >/dev/null 2>&1
}
one dyn default
one k7 ""
