#!/usr/bin/env bash
# One speculation config on the 2gpu profile: boot, single-stream decode (spec_bench), 1/2/4/8 streams (concurrency).
#   LABEL=dyn_a SCHED='[[1,1,7],[2,2,5],[3,4,3],[5,8,1]]' bash experiments/concurrency/run_spec.sh
#   LABEL=nospec SPEC=0 bash ...   |   LABEL=k7 bash ...  (stock k=7)
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); O=$R/experiments/concurrency/results/$LABEL; mkdir -p "$O"
export MODEL_DIR=${MODEL_DIR:-$HOME/ai/models/inkling2-mini} WARMUP=${WARMUP:-1}
if [ -n "${SCHED:-}" ]; then
  export SPEC=0 EXTRA="--speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"probabilistic\",\"num_speculative_tokens_per_batch_size\":$SCHED}"
fi
env | grep -E '^(SCHED|SPEC|EXTRA|PROFILE|MAXSEQS|MNBT|ASYNC|SCONV_TAIL|KV_DTYPE)=' | sort > "$O/config.env"
bash "$R/launch/2gpu.sh" > "$O/boot.log" 2>&1 || { echo "BOOT FAIL $LABEL"; docker logs --tail 60 inkling2 > "$O/boot_fail.log" 2>&1; docker rm -f inkling2 >/dev/null; exit 1; }
python3 "$R/tests/spec_bench.py" "$O/decode.json" | grep "0.0" | sed "s/^/$LABEL /"
for rep in $(seq 1 "${REPEAT:-1}"); do
  f=$O/concurrency$([ "$rep" = 1 ] || echo "_r$rep").json
  python3 "$R/tests/concurrency.py" "$f" --levels "${LEVELS:-1,2,4,8}" > "$O/concurrency_r$rep.log" 2>&1
  python3 -I -c 'import json,sys;d=json.load(open(sys.argv[1]));print(sys.argv[2],"streams",{k:(v["aggregate_tok_s"],v["mean_ttft_s"]) for k,v in d.items()})' "$f" "$LABEL r$rep"
done
docker logs inkling2 > "$O/server.log" 2>&1; docker rm -f inkling2 >/dev/null
