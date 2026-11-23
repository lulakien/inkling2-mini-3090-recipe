#!/usr/bin/env bash
# One RTX 3090 (24 GB). PROFILE=128k (default) or PROFILE=262k. GPU=0|1 picks the card.
#   128k: max context 131,072, bf16 KV
#   262k: max context 262,144, FP8 E4M3 KV converted in software (decode step +3-5%, prefill -12% at 16K and -30% at
#         127K vs 128k; a precision change: temp-0 output diverges from bf16 within the first few hundred characters)
# How it fits (see docs/NOTES.md, "Single GPU"):
#   W8 weights quantized at load; GMU 1.0 with a 512 MiB KV margin (the extensible KV cache is sized after warm-up);
#   prompt-logprob warm-up in small chunks; the 0.86 GiB input embedding lives in pinned host memory (EMBED_HOST, ~1%
#   decode); --no-async-scheduling and SCONV_TAIL=1 shrink the per-request short-conv / sliding-window reserves.
#   262k also: CUDA graphs up to 16 tokens, 64-row prompt-logprob chunks (less allocator fragmentation in the
#   sizing warm-up) and a 256 MiB non-torch reserve (raises torch's allocator cap; the KV size does not change).
# Do not lower KV_MARGIN_MB below 512: it lowers torch's allocator cap below the measured warm-up growth, and the
# first long prefill runs out of memory.
set -euo pipefail
case "${PROFILE:-128k}" in
  128k) D_MAXLEN=131072; D_KV=bfloat16; D_CG=32; D_PLP=128; D_NT= ;;
  262k) D_MAXLEN=262144; D_KV=fp8;      D_CG=16; D_PLP=64;  D_NT=256 ;;
  *) echo "PROFILE must be 128k or 262k" >&2; exit 1 ;;
esac
export TP=1 GPUS="device=${GPU:-0}" NAME="${NAME:-inkling2-1gpu}"
export MAXLEN="${MAXLEN:-$D_MAXLEN}" KV_DTYPE="${KV_DTYPE:-$D_KV}" CG_MAX="${CG_MAX:-$D_CG}" PLP_CHUNK="${PLP_CHUNK:-$D_PLP}"
export NONTORCH_MB="${NONTORCH_MB:-$D_NT}" MNBT="${MNBT:-1024}" MAXSEQS="${MAXSEQS:-4}" SCONV_TAIL="${SCONV_TAIL:-1}"
export GMU="${GMU:-1.0}" KV_MARGIN_MB="${KV_MARGIN_MB:-512}" EMBED_HOST="${EMBED_HOST:-1}" ASYNC="${ASYNC:-0}"
exec bash "$(dirname "$0")/serve.sh"
