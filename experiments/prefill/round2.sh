#!/usr/bin/env bash
# Round 2: FAR_ATTN end-to-end on cold 500K (512k profile), then k=7 vs k=4 on warm 64K agent turns (default profile).
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); cd "$R"; O=$R/experiments/prefill/results; mkdir -p $O/far_512k $O/../../concurrency/results/warm_k
M=$HOME/ai/models/inkling2-mini
echo "== far 512k"
PROFILE=512k MODEL_DIR=$M DOCKER_EXTRA="-e INKLING2_FAR_ATTN=1" bash launch/2gpu.sh > $O/far_512k/boot.log 2>&1 || { echo "BOOT FAIL"; tail -5 $O/far_512k/boot.log; }
docker exec inkling2 env | grep FAR
for d in 0.5 0.1; do python3 tests/needle.py 500000 --depth $d | tail -1 | tee $O/far_512k/needle_500k_$d.json; done
python3 tests/needle.py 16384 --depth 0.5 | tail -1 | tee $O/far_512k/needle_16k.json
docker logs inkling2 2>&1 | grep -iE "KV cache|maximum concurrency" | tail -3
docker stop inkling2 >/dev/null
for k in 7 4; do
  echo "== warm decode k=$k"
  S='[[1,1,7],[2,2,4],[3,4,2],[5,256,1]]'; [ $k = 4 ] && S='[[1,1,4],[2,2,4],[3,4,2],[5,256,1]]'
  SPEC_SCHED="$S" MODEL_DIR=$M bash launch/2gpu.sh > $O/../../concurrency/results/warm_k/boot_k$k.log 2>&1 || { echo "BOOT FAIL"; continue; }
  for r in 1 2; do python3 experiments/concurrency/warm_decode.py $O/../../concurrency/results/warm_k/k${k}_r$r.json --ctx 65536 --effort medium; done
  docker stop inkling2 >/dev/null
done
echo "== done"
