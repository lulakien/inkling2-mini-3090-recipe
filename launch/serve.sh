#!/usr/bin/env bash
# Inkling 2 Mini (W4A16 experts on the shipped NVFP4 bytes) on RTX 3090s (sm_86) with the vLLM inkling-2-mini image.
# Use the profile wrappers (launch/2gpu.sh, launch/1gpu.sh); they set TP/context/memory knobs and exec this file.
# Every patch is a whole file mounted read-only over the image; nothing inside the image is modified.
#
# Environment (defaults in brackets):
#   MODEL_DIR   checkpoint directory (thinkingmachines/Inkling-2-Mini)                [$HOME/models/Inkling-2-Mini]
#   NAME / PORT container name / host port on 127.0.0.1                             [inkling2 / 8000]
#   SERVED_NAME model name in the API                                                 [inkling-2-mini]
#   CACHE_DIR   persistent Triton/inductor/host-allreduce build cache                 [$HOME/.cache/inkling2-vllm]
#   GPUS        docker --gpus value                                                   [all]
#   TP          tensor parallel size (2 or 1)                                         [2]
#   MAXLEN MAXSEQS GMU CG_MAX MNBT                                                    [profile]
#   SPEC=0      disable DSpark speculative decoding (k=7)                             [1]
#   SPEC_SCHED  DSpark draft length per batch size, [[from,to,k],...]; empty = k=7 always [[[1,1,7],[2,2,4],[3,4,2],[5,256,1]]]
#   W8=         int8 per-channel weights for attn qkvr, LM head, Markov head, drafter [attn,head,markov,draft]
#   HOSTAR=0    TP2 small all-reduce/all-gather through mapped host memory off        [1]
#   CEAR=0      TP2 prefill all-reduce on the copy engines off                        [1]
#   ATTN_SPLITK full-attention prefill split-KV (0 = shipped kernel)                  [5]
#   ALIGN_MODE  MoE align: auto (parallel kernel for >=2048 routes) | legacy          [auto]
#   ATTN_RES_2PASS=0  stock 3-pass attn_res prefill kernel                            [1]
#   EMBED_HOST=1      input embedding table in pinned host memory (1-GPU profiles)    [0]
#   KV_MARGIN_MB, PLP_CHUNK  KV margin floor / prompt-logprob chunk (1-GPU profiles)  [1024 MiB / 1024]
#   NONTORCH_MB  memory kept free outside torch's allocator cap after KV sizing (1-GPU 262k) [512 MiB]
#   KV_DTYPE=fp8 unit-scale E4M3 KV, converted in software on sm_86 (half the KV bytes)  [bfloat16]
#   SCONV_TAIL=1 short-conv state cache backs only each step's tail (frees KV; see docs/NOTES.md) [0]
#   LOOP_SHARE=1 varloop checkpoints only (TML-d24-varloop-research): one KV cache for all loop passes  [0]
#   LPT=512     cap a long prompt's prefill chunk while other requests run: running streams keep ~2.4x more
#               speed during a newcomer's prefill, the newcomer's TTFT grows ~50% (see experiments/concurrency) [off]
#   ASYNC=0     --no-async-scheduling (frees sliding/conv in-flight reserve, 1-GPU)   [1]
#   MM=1        keep the vision tower (untested beyond a smoke test, see README)      [0]
#   EAGER=1     no CUDA graphs (debugging)                                            [0]
#   WARMUP=0    skip the post-boot warm-up requests                                   [1]
#   EXTRA       extra vLLM args;  DOCKER_EXTRA  extra docker args
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); . "$R/launch/image.env"
P=$R/patches
NAME=${NAME:-inkling2}; PORT=${PORT:-8000}; SERVED_NAME=${SERVED_NAME:-inkling-2-mini}
MODEL_DIR=${MODEL_DIR:-$HOME/models/Inkling-2-Mini}; CACHE_DIR=${CACHE_DIR:-$HOME/.cache/inkling2-vllm}
TP=${TP:-2}
[ -f "$MODEL_DIR/config.json" ] || { echo "MODEL_DIR=$MODEL_DIR has no config.json (download thinkingmachines/Inkling-2-Mini)" >&2; exit 1; }
caps=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1 | sort -u | tr '\n' ' ')
[ "$caps" = "280 " ] || echo "note: power limits are '$caps' W; the published numbers are at 280 W" >&2
mkdir -p "$CACHE_DIR"

PY=/usr/local/lib/python3.14/dist-packages
VM=$PY/vllm/models/inkling2; TR=$PY/triton; VF=$PY/vllm/model_executor/layers/fused_moe
VL=$PY/vllm/model_executor/layers; VB=$PY/vllm/v1/attention/backends; VD=$PY/vllm/distributed/device_communicators

# FP8 KV doubles the attention block (128 tokens per page), and the retention interval must be a multiple of it
PREFIX_RET=${PREFIX_RET:-$(case "${KV_DTYPE:-bfloat16}" in fp8*) echo 128;; *) echo 64;; esac)}
ARGS=(/model --served-model-name "$SERVED_NAME" --tensor-parallel-size "$TP"
  --max-model-len "${MAXLEN:-262144}" --max-num-seqs "${MAXSEQS:-8}" --kv-cache-dtype "${KV_DTYPE:-bfloat16}"
  --gpu-memory-utilization "${GMU:-0.97}"
  --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice
  --compilation-config "{\"max_cudagraph_capture_size\": ${CG_MAX:-64}}"
  --prefix-cache-retention-interval "$PREFIX_RET" --enable-prompt-tokens-details)
[ "$TP" = 2 ] && ARGS+=(--enable-batch-sharded-sampling)
[ -n "${MNBT:-}" ] && ARGS+=(--max-num-batched-tokens "$MNBT")
[ "${ASYNC:-1}" = 0 ] && ARGS+=(--no-async-scheduling)
[ -n "${LPT:-}" ] && ARGS+=(--long-prefill-token-threshold "$LPT")
[ "${MM:-0}" = 1 ] || ARGS+=(--language-model-only)
[ "${LOOP_SHARE:-0}" = 1 ] && ARGS+=(--hf-overrides '{"text_config":{"loop_share_cache":true}}')
[ "${EAGER:-0}" = 1 ] && ARGS+=(--enforce-eager)
# Draft length by batch size: k=7 for one stream, shorter drafts as more streams share a step (verify cost grows
# with batch x k while acceptance does not). SPEC_SCHED= (empty) keeps k=7 at every batch size.
SPEC_SCHED=${SPEC_SCHED-'[[1,1,7],[2,2,4],[3,4,2],[5,256,1]]'}
SPEC_JSON='{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"'
[ -n "$SPEC_SCHED" ] && SPEC_JSON="$SPEC_JSON,\"num_speculative_tokens_per_batch_size\":$SPEC_SCHED"
[ "${SPEC:-1}" = 1 ] && ARGS+=(--speculative-config "$SPEC_JSON}")
CEAR=${CEAR:-1}; [ "$TP" = 1 ] && CEAR=0

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus "${GPUS:-all}" --ipc=host \
  -v "$MODEL_DIR":/model:ro -v "$CACHE_DIR":/root/.cache \
  -v $P/quant_config.py:$VM/quant_config.py:ro \
  -v $P/inkling2_init.py:$VM/__init__.py:ro \
  -v $P/vision.py:$VM/nvidia/vision.py:ro \
  -v $P/conv_mix.py:$VM/nvidia/ops/conv_mix.py:ro \
  -v $P/linear.py:$VM/nvidia/ops/linear.py:ro \
  -v $P/attn_res.py:$VM/nvidia/ops/attn_res.py:ro \
  -v $P/skinny_gemm.py:$VM/nvidia/ops/skinny_gemm.py:ro \
  -v $P/sconv_ops.py:$VM/nvidia/ops/sconv.py:ro \
  -v $P/attention_sm90.py:$VM/nvidia/ops/attention_sm90.py:ro -v $P/e4m3.py:$VM/nvidia/ops/e4m3.py:ro \
  -v $P/sconv_swa_attn.py:$PY/vllm/models/inkling/nvidia/sconv_swa_attn.py:ro \
  -v $P/inkling_sconv_ops.py:$PY/vllm/models/inkling/nvidia/ops/sconv.py:ro \
  -v $P/w8_gemm.py:$VM/nvidia/ops/w8_gemm.py:ro \
  -v $P/loop.py:$VM/nvidia/loop.py:ro -v $P/sconv.py:$VM/nvidia/sconv.py:ro \
  -v $P/inkling2_attention.py:$VM/nvidia/attention.py:ro \
  -v $P/inkling2_model.py:$VM/nvidia/model.py:ro \
  -v $P/inkling2_dspark.py:$VM/nvidia/dspark.py:ro \
  -v $P/dspark_boot.py:$VM/nvidia/dspark_boot.py:ro \
  -v $P/moe_align_block_size.py:$VF/moe_align_block_size.py:ro \
  -v $P/vocab_parallel_embedding.py:$VL/vocab_parallel_embedding.py:ro \
  -v $P/fa_utils.py:$VB/fa_utils.py:ro \
  -v $P/cuda_communicator.py:$VD/cuda_communicator.py:ro -v $P/hostar:/hostar:ro \
  -v $P/prompt_logprob.py:$PY/vllm/v1/worker/gpu/sample/prompt_logprob.py:ro \
  -v $P/extensible_kv_cache.py:$PY/vllm/v1/worker/extensible_kv_cache.py:ro \
  -v $P/gdc.py:$TR/language/extra/cuda/gdc.py:ro \
  -v $P/triton_driver.py:$TR/backends/nvidia/driver.py:ro \
  -e VLLM_INKLING_W8="${W8-attn,head,markov,draft}" \
  -e VLLM_HOST_ALLREDUCE="${HOSTAR:-1}" -e HOSTAR_BUILD_DIR=/root/.cache/hostar-build \
  -e VLLM_CEAR="$CEAR" -e VLLM_CEAR_TAIL="$CEAR" -e VLLM_CEAR_TAIL_STAGE=2 \
  -e VLLM_ATTN_SPLITK="${ATTN_SPLITK:-5}" -e VLLM_ALIGN_MODE="${ALIGN_MODE:-auto}" \
  -e VLLM_ATTN_RES_2PASS="${ATTN_RES_2PASS:-1}" -e VLLM_SCONV_TAIL="${SCONV_TAIL:-0}" -e VLLM_INKLING_EMBED_HOST="${EMBED_HOST:-0}" \
  -e VLLM_PROMPT_LOGPROB_CHUNK="${PLP_CHUNK:-1024}" ${KV_MARGIN_MB:+-e VLLM_KV_MARGIN_FLOOR_MB=$KV_MARGIN_MB} \
  ${NONTORCH_MB:+-e VLLM_KV_NON_TORCH_RESERVE_MB=$NONTORCH_MB} \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor -e TRITON_CACHE_DIR=/root/.cache/triton \
  ${DOCKER_EXTRA:-} -p 127.0.0.1:$PORT:8000 "$IMAGE" "${ARGS[@]}" ${EXTRA:-} >/dev/null

echo "container $NAME starting (logs: docker logs -f $NAME)"
for i in $(seq 1 240); do
  if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; then
    echo "BOOT_OK after ~$((i * 5)) s, serving $SERVED_NAME on http://127.0.0.1:$PORT/v1"
    [ "${WARMUP:-1}" = 1 ] && INKLING_MODEL="$SERVED_NAME" python3 "$R/tests/warmup.py" --port "$PORT"
    exit 0
  fi
  [ "$(docker inspect -f '{{.State.Status}}' "$NAME")" = running ] || { echo BOOT_FAIL; docker logs --tail 40 "$NAME" 2>&1; exit 1; }
  sleep 5
done
echo BOOT_TIMEOUT; exit 1
