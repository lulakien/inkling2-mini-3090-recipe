#!/usr/bin/env bash
# Full public test suite against a running server. Writes results/<LABEL>/*.json + gpu.csv (power/temp/VRAM, 1 Hz).
#   LABEL=2gpu MAXCTX=262144 PORT=8000 bash tests/run_suite.sh
# Takes ~10 min (1 GPU, 128k), ~15 min (2 GPUs), ~25 min (1 GPU, 262k). Needs tests/data/haystack (tools/fetch_haystack.sh).
set -uo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); T=$R/tests
LABEL=${LABEL:?set LABEL}; MAXCTX=${MAXCTX:?set MAXCTX (the server max-model-len)}; PORT=${PORT:-8000}
O=$R/results/$LABEL; mkdir -p "$O"
[ -d "$T/data/haystack" ] || bash "$R/tools/fetch_haystack.sh"
nvidia-smi --query-gpu=timestamp,index,power.draw,power.limit,temperature.gpu,memory.used,clocks.sm,utilization.gpu \
  --format=csv -l 1 > "$O/gpu.csv" & MON=$!
trap 'kill $MON 2>/dev/null' EXIT
step() { echo; echo "=== $1"; }
NEEDLE=$((MAXCTX - 3000)); [ $NEEDLE -gt 255000 ] && NEEDLE=255000
SIZES=2048,8192,16384,32768; [ $MAXCTX -ge 70000 ] && SIZES=$SIZES,65536; [ $MAXCTX -ge 131072 ] && SIZES=$SIZES,126976
[ $MAXCTX -ge 262144 ] && SIZES=$SIZES,253952; [ $MAXCTX -lt 70000 ] && SIZES=$SIZES,61440
step "server";    curl -s "http://127.0.0.1:$PORT/v1/models" | tee "$O/models.json" | head -c 300; echo
step "api smoke"; python3 "$T/api_smoke.py" "$O/api_smoke.json" --port "$PORT" ${IMAGE:+--image $IMAGE}
step "decode";    python3 "$T/spec_bench.py" "$O/decode.json" --port "$PORT"
step "prefill";   python3 "$T/prefill_ttft.py" "$O/prefill.json" --port "$PORT" --sizes $SIZES --reps 2
step "agent turn (prefix cache)"; python3 "$T/prefix_midprobe.py" "$O/prefix.json" --port "$PORT"
step "concurrency"; python3 "$T/concurrency.py" "$O/concurrency.json" --port "$PORT" --levels 1,2,4$([ $MAXCTX -ge 262144 ] && echo ,8)
step "needle $NEEDLE @5%";  python3 "$T/needle.py" $NEEDLE --port "$PORT" | tail -1 | tee "$O/needle_05.json"
step "needle $NEEDLE @90%"; python3 "$T/needle.py" $NEEDLE --depth 0.9 --port "$PORT" | tail -1 | tee "$O/needle_90.json"
step "greedy determinism"; python3 "$T/greedy_check.py" "$O/greedy_a.json" --port "$PORT" >/dev/null && \
  python3 "$T/greedy_check.py" "$O/greedy_b.json" --ref "$O/greedy_a.json" --port "$PORT" | tail -3
step "MMLU (57 subjects x 20, sequential)"; python3 "$T/mmlu_letters.py" "$O/mmlu.json" --workers 1 --port "$PORT" | tail -1
docker logs "${NAME:-inkling2}" 2>&1 | grep -E "GPU KV cache size|Model loading took|Available KV|Extended KV|Graph capturing" \
  | sed -E 's/^.*\] //' > "$O/server_log_excerpt.txt" 
echo; echo "results in $O"
