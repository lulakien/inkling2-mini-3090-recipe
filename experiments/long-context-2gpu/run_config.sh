#!/usr/bin/env bash
# One config of the varloop / shared-KV matrix on the 2-GPU profile. Boots, records the KV pool, runs:
# decode (spec_bench), agent turns at 64K and ~250K (cold / warm / multi-needle / post-compaction), MMLU, then stops.
#   LABEL=fixed MODEL_DIR=... [HF='{"text_config":{...}}'] [SPEC=0] [KV_DTYPE=fp8] [SCONV_TAIL=1] [MAXLEN=262144]
#   [SKIP=mmlu,250k] bash experiments/long-context-2gpu/run_config.sh
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); T=$R/tests
LABEL=${LABEL:?}; O=$R/experiments/long-context-2gpu/results/$LABEL; mkdir -p "$O"
export MODEL_DIR NAME=${NAME:-inkling2} PORT=${PORT:-8000} WARMUP=${WARMUP:-1}
[ -n "${HF:-}" ] && export EXTRA="--hf-overrides $HF"
env | grep -E '^(MODEL_DIR|HF|EXTRA|SPEC|KV_DTYPE|SCONV_TAIL|ASYNC|MAXLEN|MNBT|MAXSEQS|GMU|CG_MAX)=' | sort > "$O/config.env"
echo "bash launch/2gpu.sh  # with config.env" > "$O/command.txt"
t0=$(date +%s)
bash "$R/launch/2gpu.sh" > "$O/boot.log" 2>&1 || { echo "BOOT FAILED: $LABEL"; docker logs --tail 80 "$NAME" > "$O/boot_fail.log" 2>&1; docker rm -f "$NAME" >/dev/null 2>&1; exit 1; }
echo "boot_s=$(( $(date +%s) - t0 ))" >> "$O/config.env"
docker logs "$NAME" 2>&1 | grep -E "GPU KV cache size|Model loading took|Available KV|Extended KV|Graph capturing|loop|Speculative" \
  | sed -E 's/^.*\] //' | sort -u > "$O/server_log_excerpt.txt"
docker inspect -f '{{json .Args}}' "$NAME" > "$O/vllm_args.json"
nvidia-smi --query-gpu=timestamp,index,power.draw,temperature.gpu,memory.used,clocks.sm,utilization.gpu \
  --format=csv -l 1 > "$O/gpu.csv" & MON=$!
trap 'kill $MON 2>/dev/null' EXIT
skip() { case ",${SKIP:-}," in *",$1,"*) return 0;; esac; return 1; }
echo "== $LABEL decode";  python3 "$T/spec_bench.py" "$O/decode.json" --port "$PORT" | tail -4
echo "== $LABEL agent 64K";  python3 "$T/agent_turns.py" "$O/agent_64k.json" --ctx 65536 --port "$PORT"
skip 250k || { echo "== $LABEL agent 250K"; python3 "$T/agent_turns.py" "$O/agent_250k.json" --ctx 250000 --port "$PORT"; }
skip mmlu || { echo "== $LABEL MMLU"; python3 "$T/mmlu_letters.py" "$O/mmlu.json" --workers 1 --port "$PORT" | tail -1; }
docker logs "$NAME" > "$O/server.log" 2>&1
[ "${KEEP:-0}" = 1 ] || docker rm -f "$NAME" >/dev/null 2>&1
echo "done $LABEL in $(( $(date +%s) - t0 )) s -> $O"
