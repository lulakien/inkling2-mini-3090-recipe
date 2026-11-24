#!/usr/bin/env bash
# Second agent joining a running one (tests/second_stream.py) on the 2gpu profile with the dynamic draft schedule,
# for one --long-prefill-token-threshold value (0 = off).
#   LABEL=lpt512 LPT=512 bash experiments/concurrency/run_ttft.sh
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); O=$R/experiments/concurrency/results/$LABEL; mkdir -p "$O"
SCHED=${SCHED:-'[[1,1,7],[2,2,5],[3,4,3],[5,8,1]]'}
export MODEL_DIR=${MODEL_DIR:-$HOME/ai/models/inkling2-mini} SPEC=0
EXTRA="--speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"probabilistic\",\"num_speculative_tokens_per_batch_size\":$SCHED}"
[ "${LPT:-0}" != 0 ] && EXTRA="$EXTRA --long-prefill-token-threshold $LPT"
export EXTRA
env | grep -E '^(LPT|SCHED|EXTRA|PROFILE|MNBT|ASYNC|SCONV_TAIL)=' | sort > "$O/config.env"
bash "$R/launch/2gpu.sh" > "$O/boot.log" 2>&1 || { echo "BOOT FAIL $LABEL"; docker logs --tail 60 inkling2 > "$O/boot_fail.log" 2>&1; docker rm -f inkling2 >/dev/null; exit 1; }
python3 "$R/tests/second_stream.py" "$O/second_stream.json" --ctx-a "${CA:-200000}" --ctx-b "${CB:-65536}" | sed "s/^/$LABEL /"
docker logs inkling2 > "$O/server.log" 2>&1; docker rm -f inkling2 >/dev/null
