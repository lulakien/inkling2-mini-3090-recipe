#!/usr/bin/env bash
# Two RTX 3090s, tensor parallel 2. PROFILE picks the context (results: README and experiments/long-context-2gpu):
#   262k (default): 262,144 tokens, bf16 KV (pool ~302K), up to 8 concurrent requests. Unchanged reference profile.
#   512k: 524,288 tokens, still bf16 KV. SCONV_TAIL=1 frees the short-conv state reserve and --no-async-scheduling
#         its in-flight copy (pool ~535K). Same decode and prefill speed as 262k; 500K cold prompt ~167 s.
#   1m:   1,048,576 tokens with FP8 KV + SCONV_TAIL and the single-GPU memory knobs (pool ~1.33M, 2 sequences).
#         Software FP8 on Ampere makes long prefill slow: a 1.04M-token cold prompt took 19.5 min; 250K 102 s vs 59 s.
# Any serve.sh variable can be overridden, e.g. MAXLEN=131072 PORT=8001 bash launch/2gpu.sh
set -euo pipefail
export TP=2 GPUS="${GPUS:-all}"
case "${PROFILE:-262k}" in
  262k) export MAXLEN="${MAXLEN:-262144}" MAXSEQS="${MAXSEQS:-8}" GMU="${GMU:-0.97}" CG_MAX="${CG_MAX:-64}" ;;
  512k) export MAXLEN="${MAXLEN:-524288}" MAXSEQS="${MAXSEQS:-4}" GMU="${GMU:-0.97}" CG_MAX="${CG_MAX:-64}" \
               SCONV_TAIL="${SCONV_TAIL:-1}" ASYNC="${ASYNC:-0}" ;;
  1m)   export MAXLEN="${MAXLEN:-1048576}" MAXSEQS="${MAXSEQS:-2}" GMU="${GMU:-1.0}" CG_MAX="${CG_MAX:-16}" \
               SCONV_TAIL="${SCONV_TAIL:-1}" ASYNC="${ASYNC:-0}" KV_DTYPE="${KV_DTYPE:-fp8}" MNBT="${MNBT:-1024}" \
               KV_MARGIN_MB="${KV_MARGIN_MB:-512}" PLP_CHUNK="${PLP_CHUNK:-64}" NONTORCH_MB="${NONTORCH_MB:-256}" \
               EMBED_HOST="${EMBED_HOST:-1}" ;;
  *) echo "PROFILE must be 262k, 512k or 1m" >&2; exit 1 ;;
esac
exec bash "$(dirname "$0")/serve.sh"
