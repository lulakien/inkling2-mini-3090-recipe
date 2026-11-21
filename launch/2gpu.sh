#!/usr/bin/env bash
# Two RTX 3090s, tensor parallel 2: 262,144-token context (KV pool ~302K tokens), up to 8 concurrent requests.
# Any serve.sh variable can be overridden, e.g. MAXLEN=131072 PORT=8001 bash launch/2gpu.sh
set -euo pipefail
export TP=2 GPUS="${GPUS:-all}"
export MAXLEN="${MAXLEN:-262144}" MAXSEQS="${MAXSEQS:-8}" GMU="${GMU:-0.97}" CG_MAX="${CG_MAX:-64}"
exec bash "$(dirname "$0")/serve.sh"
