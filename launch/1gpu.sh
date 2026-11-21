#!/usr/bin/env bash
# One RTX 3090 (24 GB). PROFILE=64k (default) or PROFILE=128k. GPU=0|1 picks the card.
#   64k : max context 65,536, 1024-token prefill chunks (full prefill speed)
#   128k: max context 131,072, 512-token prefill chunks (~15% slower prefill, same decode)
# How it fits (see docs/NOTES.md, "Single GPU"):
#   W8 weights quantized at load; GMU 1.0 with a 512 MiB KV margin (the extensible KV cache is sized after warm-up);
#   prompt-logprob chunk 128; the 0.86 GiB input embedding lives in pinned host memory (EMBED_HOST, ~1% decode);
#   --no-async-scheduling: every short-conv / sliding-window cache reserves window + in-flight tokens per request
#   and async scheduling doubles in-flight, so turning it off frees ~1 GiB (A/B: no measurable speed change).
# Do not lower KV_MARGIN_MB below 512: 256 boots at 131K with 768-token chunks but OOMs on the first long prefill.
set -euo pipefail
case "${PROFILE:-64k}" in
  64k)  D_MAXLEN=65536;  D_MNBT=1024; D_CG=64 ;;
  128k) D_MAXLEN=131072; D_MNBT=512;  D_CG=32 ;;
  *) echo "PROFILE must be 64k or 128k" >&2; exit 1 ;;
esac
export TP=1 GPUS="device=${GPU:-0}" NAME="${NAME:-inkling2-1gpu}"
export MAXLEN="${MAXLEN:-$D_MAXLEN}" MNBT="${MNBT:-$D_MNBT}" CG_MAX="${CG_MAX:-$D_CG}" MAXSEQS="${MAXSEQS:-4}"
export GMU="${GMU:-1.0}" KV_MARGIN_MB="${KV_MARGIN_MB:-512}" PLP_CHUNK="${PLP_CHUNK:-128}"
export EMBED_HOST="${EMBED_HOST:-1}" ASYNC="${ASYNC:-0}"
exec bash "$(dirname "$0")/serve.sh"
