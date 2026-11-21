# Engineering notes

How the recipe got here, what each patch does, what was measured and rejected. All numbers: RTX 3090s at 280 W,
gpu-unlock active (see README), batch 1 unless stated.

## 1. Making the image run on sm_86 at all

`vllm/vllm-openai:inkling-2-mini` targets Hopper/Blackwell. Its Hopper code path (Triton relative attention, Humming
W4A16 expert GEMMs with bf16 activations) uses no sm_90-only instructions on the routes this model takes, so the port
is mostly about the places that refuse Ampere or call kernels that were only compiled for sm_100:

| file | change |
|---|---|
| `quant_config.py` | minimum compute capability 90 -> 80 |
| `vision.py` | vision attention FA3 -> FA2 off Hopper |
| `conv_mix.py` | sm8x takes the SM12x Triton fallback (short_conv + attn_res); the `_C` conv_mix op has no sm_86 build |
| `linear.py` | `k_group_partials`: sm8x skips the TMA kernel like SM12x |
| `moe_align_block_size.py` | the `_moe_C` align kernel is sm_100-only **and fails silently** (error 209 left in cudart, routing tables unset -> garbage MoE output, the next torch op gets blamed). Replaced by Triton kernels, verified against brute force and under CUDA graphs |
| `vocab_parallel_embedding.py` | TP>1: unfused embedding (the fused `_C` op is sm_100-only) |
| `fa_utils.py` | `reshape_and_cache_flash` for the DSpark drafter KV writes: portable torch (`_C_cache_ops` is sm_100-only) |
| `gdc.py`, `triton_driver.py` | `griddepcontrol.wait/launch_dependents` (sm_90 PTX) -> no-op, and PDL launch forced off so the no-op waits stay correct |

KV cache stays bf16: the image's FP8-KV Triton loader uses `cvt.e4m3x2` (sm_89+).
The silent-failure class is the dangerous one. A debugging aid that checks `cudaGetLastError` after every op found
all three (`moe_align`, the fused embedding, `conv_mix`).

## 2. Decode (single stream)

Decode on these cards is memory-bandwidth bound. Per DSpark step (k=7, 8 tokens verified), per GPU at TP2:

| piece | time | notes |
|---|---|---|
| experts (Humming W4A16) | ~4.6 ms | verifying 8 tokens touches ~20 distinct experts/GPU x 48 visits; at bandwidth |
| dense GEMMs | ~1.9 ms | int8 now (qkvr, LM head, Markov head, drafter); `wo_ud` stays bf16 |
| TP2 all-reduces | ~1.8 ms | ~100 per step through mapped host memory (11.7 us each vs 31 us NCCL) |
| logits all-gather | ~0.5 ms | fp32 verify logits, PCIe bound |
| rest | ~2 ms | attention 0.7, attn_res/blend/norms 0.8, moe_align 0.4, sconv 0.3, drafter 0.3 |

What moved the step from ~20 ms to 14.0 ms (TP2):
- **moe_align as one Triton kernel** (the portable torch version was 15 kernels x 48 visits, ~2 ms/token).
- **Host-memory all-reduce / all-gather** (`cuda_communicator.py` + `hostar/hostar.py`): GeForce cards have no P2P,
  so NCCL bounces through the host anyway. A kernel that writes/reads mapped pinned host memory does tensors up to
  256 KB (all-reduce) / 512 KB (all-gather) in ~12 us, bit-identical to NCCL, CUDA-graph safe. `HOSTAR=0` disables.
- **Split-K skinny GEMM** (`skinny_gemm.py`) for the 3-stream loop-controller bmm and dynamic sconv taps at 2-16
  rows: cuBLAS picked a 6-CTA kernel (28/13 us -> 6/4 us). Big projections stay on cuBLAS (already at bandwidth).
- **W8: int8 per-channel weights** (`w8_gemm.py`, own split-K Triton kernel with fused dequant) for attention qkvr,
  the LM head (shared by target and drafter), the drafter's Markov fold and the drafter layers. Quantized at load,
  so the bf16 copies are freed. Quality gate: MMLU equal or better than bf16 (see README). Experts and `wo_ud` are
  untouched. `W8=` turns it off.
- **Batch-sharded verify-logits sampling** (`--enable-batch-sharded-sampling`, TP2 only) and in-place softcap.

Tokens per step depend on the text (DSpark acceptance): ~4.4 on code, ~2.4 on technical prose, ~1.75 on stories.
Plain decode without speculation is ~130 tok/s at TP2.

Single GPU: the same step is ~18.4 ms. Experts are 8.5 ms of it at ~0.9 TB/s, the LM head runs twice (target and
drafter read different hidden states, so the logits cannot be shared). Nothing large is left except
quantizing `wo_ud` (~3%) or the LM head to int4 (~3%, needs a logits quality gate).

## 3. Prefill

TP2 prefill is PCIe-bound: before these patches NCCL all-reduce was 41% of each 2048-token chunk
(97 all-reduces x 1.36 ms at x8/x8 link speed), Humming 22%, attention 13%.

- **Row-piece layer tail on the copy engines** (`CEAR`, `hostar/cear_tail.py`, `cear_moe.py`): the row-wise tail of
  each layer (attention-output all-reduce -> conv/mix -> MoE -> MoE all-reduce -> conv/mix) runs as a 2 x 1024-row
  pipeline; piece *i* is DMA'd through a shared pinned buffer (stream write/wait-value signalling, no SMs) while
  piece *i+1* computes. +5.9% at 16K, -9% time at 255K.
- **Split-KV full attention** (`ATTN_SPLITK=5`, in `inkling2_attention.py`): long prompts had 196 CTAs = 2.4 waves
  on 82 SMs. -9% time at 255K, no change at 16-32K.
- **Parallel MoE align** (`ALIGN_MODE=auto`): 193 -> 42 us per align at 8192 routes; 35/35 cases `torch.equal`.
- **2-pass attn_res** (`ATTN_RES_2PASS=1`): sources read twice instead of three times; 72/72 cases `torch.equal`.
  The last two together: +3.4% (1 GPU), +7.5% (2 GPUs, bigger chunks).

## 4. Prefix cache for agents

`--prefix-cache-retention-interval 64` keeps sliding-window/conv state every 64 tokens, so a request that shares a
long prefix with an earlier one but diverges mid-message still hits (13K shared system prompt + a different user
message: 0 hits / 2.05 s -> 12,608 cached / 0.14 s). Plain chat/agent turns that extend the previous request hit
anyway. Cost: more retained blocks per cached prefix.

## 5. Single GPU: how 24 GB holds 64K / 128K

Weights take 16.5 GiB at TP1. What it took to fit (3.7K tokens -> 128K):

1. **Quantize W8 at load.** Lazy quantization kept the bf16 originals alive through the memory profile, which counted
   them as a 1.27 GiB activation peak.
2. **Let the extensible KV cache have the memory.** It is sized after warm-up; GMU 1.0, a 512 MiB margin floor
   (`extensible_kv_cache.py` reads `VLLM_KV_MARGIN_FLOOR_MB`, stock is 1 GiB) and prompt-logprob warm-up in
   128-row chunks (`prompt_logprob.py`, stock 1024 rows x 201K vocab fp32 = 0.78 GiB).
3. **Input embedding in pinned host memory** (`EMBED_HOST=1`, UVA view): +0.86 GiB of KV for ~1% decode and ~2.5%
   prefill (one gather per token over PCIe).
4. **The fixed per-request reserves.** vLLM's KV planner showed (TP1, 64K): every request reserved 1,075 blocks
   (2.36 GiB) for the six short-conv groups (130 blocks each), five sliding-window groups (49 each) and the drafter's
   sliding window (50), against 1,024 blocks for 64K of full attention. Sliding/conv managers reserve
   `window - 1 + in_flight_tokens`, and with async scheduling (depth 2) in-flight is 2 x `max_num_batched_tokens`.
   The short-conv window is ~16 tokens, so that 1.7 GiB was almost all in-flight room.
   `--no-async-scheduling` halves it; A/B with CUDA graphs + DSpark: decode 243 vs 242 tok/s, 16K prefill
   5.28K vs 5.29K tok/s, i.e. free. Smaller prefill chunks shrink it further at a prefill cost.

| 1 GPU, async off | max context | prefill 16K |
|---|---|---|
| 1024-token chunks | ~114K | 5.29K tok/s |
| 768 + 256 MiB margin | 131K boots, **OOM on first long prefill** | - |
| 512-token chunks (`PROFILE=128k`) | 131K (KV 136K tokens) | 4.48K tok/s |

## 6. Tried and rejected (measured)

- Adaptive DSpark verify: needs varlen-graph overlays; break-even at batch 1 and temp-0 output not reproducible.
- Full-attention BLOCK_M=64 (0.84x), tail pipeline with 3 pieces (-2%), host all-reduce launch-config sweep (shipped
  4x256 already optimal), dense Marlin (no sm_86 build in this image), skinny GEMM on big projections.
- `--max-num-batched-tokens 4096` at TP2: the startup KV check fails at 262K.
- `draft_lm_head_quantization`: vLLM only allows it for DeepSeek drafters.
- `expandable_segments`: worse fragmentation for the single-GPU fit.
- Sequence-parallel prefill path in the model: needs NVLink symmetric memory.
- `KV_MARGIN_MB=256` on one GPU: boots, then OOMs under a real long prefill.

## 7. Numerics

Speculative decoding at temp 0 is not token-identical to plain greedy decoding (bf16 verify numerics differ by batch
shape), so losslessness is judged by task metrics: sequential MMLU letter accuracy (57 subjects x 20), long-context
needle retrieval and temp-0 run-to-run determinism. Host all-reduce/all-gather: 0/1140 MMLU answer differences on vs
off. The two prefill kernels are bit-identical to the stock ones (`tests/kernels`).
