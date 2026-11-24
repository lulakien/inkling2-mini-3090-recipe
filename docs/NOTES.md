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

The image's FP8-KV Triton loader uses `cvt.e4m3x2` (sm_89+); `KV_DTYPE=fp8` replaces it with a software conversion
(section 6). The default KV cache is bf16.
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

### Several streams

DSpark's draft length now follows the batch size (`SPEC_SCHED`, default k=7 / 4 / 2 / 1 for 1 / 2 / 3-4 / 5+
streams; vLLM's `num_speculative_tokens_per_batch_size`). With a fixed k=7, 8 streams verify 64 tokens per step and
lost to plain decoding (429 vs 545 tok/s); with the schedule 8 streams give ~575 tok/s and one stream is unchanged.
The 1-GPU 262k profile keeps k=7: the extra CUDA graphs per draft length leave it 0.03 GiB short of 262K.
`tests/warmup.py` also covers batched decode/sampling shapes and multi-prompt prefills, so no Triton kernel compiles
during real requests (it was 14 on the first burst). Measurements: `experiments/concurrency/RESULTS.md`.

## 3. Prefill

TP2 prefill is PCIe-bound: before these patches NCCL all-reduce was 41% of each 2048-token chunk
(97 all-reduces x 1.36 ms at x8/x8 link speed), Humming 22%, attention 13%.

- **Row-piece layer tail on the copy engines** (`CEAR`, `hostar/cear_tail.py`, `cear_moe.py`): the row-wise tail of
  each layer (attention-output all-reduce -> conv/mix -> MoE -> MoE all-reduce -> conv/mix) runs as a 2 x 1024-row
  pipeline; piece *i* is DMA'd through a shared pinned buffer (stream write/wait-value signalling, no SMs) while
  piece *i+1* computes. +5.9% at 16K, -9% time at 255K.
- **Split-KV full attention** (`ATTN_SPLITK=5`, in `inkling2_attention.py`): long prompts had 196 CTAs = 2.4 waves
  on 82 SMs. -9% time at 255K, no change at 16-32K.
- **Far-block fast path** (`FAR_ATTN=1`, opt-in, `attention_sm90.py`): at 485K depth the full-attention prefill
  kernel was 71% of GPU time (93.5 ms per layer visit, ~64 TFLOPS). The relative bias only reaches 1,024 keys back,
  but the kernel ran the bias load and distance masks on every key block. A first loop now covers the blocks wholly
  before every row's bias extent (no bias, no masks), then the unchanged loop runs the rest; same splits and online
  softmax, no second kernel or merge. x1.25-1.27 bf16 (= FA2 on the same shape), x1.13-1.15 FP8 (bound by the
  software E4M3 decode); 500K cold 167 -> ~145 s (4 runs, 144-146 s). Output differs from the shipped kernel at fp32 rounding level
  (max relative 1.4e-4); with the switch off the kernel is bit-identical to the shipped one.
- **Parallel MoE align** (`ALIGN_MODE=auto`): 193 -> 42 us per align at 8192 routes; 35/35 cases `torch.equal`.
- **2-pass attn_res** (`ATTN_RES_2PASS=1`): sources read twice instead of three times; 72/72 cases `torch.equal`.
  The last two together: +3.4% (1 GPU), +7.5% (2 GPUs, bigger chunks).

## 4. Prefix cache for agents

`--prefix-cache-retention-interval 64` keeps sliding-window/conv state every 64 tokens, so a request that shares a
long prefix with an earlier one but diverges mid-message still hits (13K shared system prompt + a different user
message: 0 hits / 2.05 s -> 12,608 cached / 0.14 s). Plain chat/agent turns that extend the previous request hit
anyway. Cost: more retained blocks per cached prefix.

## 5. Single GPU: how 24 GB holds 128K / 262K

Weights take 16.5 GiB at TP1. What it took to fit (3.7K tokens -> 262K):

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
   5.28K vs 5.29K tok/s, i.e. free.
5. **Short-conv tail manager** (`SCONV_TAIL=1`, issue #1, adapted from HyperInkling). The conv kernel reads cached
   taps only for the first `window - 1` tokens of a step and writes the state of the last ones, so a step needs real
   cache blocks only for its tail. `SconvStateManager` (`sconv_swa_attn.py`) reserves
   `window + k x in-flight steps + one prefix-hit alignment` tokens per conv group instead of the whole prefill
   chunk; every earlier slot of the chunk maps to block 0, the null block, and the store in `sconv_ops.py` is masked
   off for it. The alignment term keeps prefix-cache hits working: a hit resumes at a scheduler block boundary, so the
   tail must hold the state at that boundary (64 tokens with bf16 KV, 2 x 64 with FP8, whose attention pages hold 128
   tokens). `inkling_sconv_ops.py` fixes a metadata bug the tail exposes: padding rows got `query_start = 0`, so a
   >256-token tile mixing real and padding rows skipped its cached taps. With the tail, 1024-token prefill chunks fit
   at 131K again (the old 128K profile needed 512-token chunks): 16K prefill 4.48K -> 5.28K tok/s, KV 135K tokens.
   Side effect: conv state now exists only at step ends, so a prompt that diverges mid-message from a cached one hits
   at the last 1024-token chunk boundary instead of the last 64-token one (13K agent probe: 12,288 instead of 12,608
   tokens cached, 0.19 vs 0.17 s). Prompts that extend the previous one hit as before.

| 1 GPU, async off | max context | prefill 16K |
|---|---|---|
| 1024-token chunks | ~114K | 5.29K tok/s |
| 768 + 256 MiB margin | 131K boots, **OOM on first long prefill** | - |
| 512-token chunks (old 128K profile) | 131K (KV 136K tokens) | 4.48K tok/s |
| 1024-token chunks + `SCONV_TAIL=1` (`PROFILE=128k`) | 131K (KV 135K tokens) | 5.28K tok/s |
| same + `KV_DTYPE=fp8` + the knobs below (`PROFILE=262k`) | 262K (KV 263K tokens) | 4.64K tok/s |

### The 262K memory budget

With FP8 KV (section 6) and the tail manager, 1024-token chunks and the 128K profile's other knobs, the planner
estimated a 250,048-token maximum: 0.23 GiB short of 262,144. The sizing after warm-up reads (one GPU, 23.58 GiB):

| | GiB |
|---|---|
| weights + CUDA context + non-torch | 16.71 |
| CUDA graphs (capture sizes up to 32 / 16 tokens) | 0.22 / 0.17 |
| free after warm-up | 6.25 / 6.30 |
| margin floor (`KV_MARGIN_MB`) | 0.50 |
| kept for transients: max(allocated peak 0.24, torch reservation growth) | 0.54 -> 0.35 |
| KV cache | 5.21 -> **5.45** (22,336 blocks, 262,755 tokens; 262,144 need 5.44) |

The transient reserve was the lever. Its size is the largest growth of torch's reservation over the sizing
warm-up steps, fragmentation included, and the prompt-logprob step fragmented most. A/B at 262,144, one boot each:

| change from the 128K knobs | reservation growth | KV fit | planner max |
|---|---|---|---|
| none (CUDA graphs <= 32, prompt-logprob chunk 128, 4 seqs) | 0.54 | 5.21 | 250,048 |
| prompt-logprob chunk 64 (32: same) | 0.35 | 5.40 | 260,224 |
| CUDA graphs <= 16 tokens | 0.54 | 5.26 | 252,608 |
| `MAXSEQS=1` | 0.48 | 5.34 | 257,024 |
| `MNBT=768` | 0.42 | 5.33 | 257,664 (and slower prefill) |
| **chunk 64 + graphs <= 16** (shipped) | 0.35 | 5.45 | **boots, 262,755 tokens** |
| same + `MAXSEQS=2` | 0.28 | 5.56 | boots, 267,743, then OOM in the post-boot warm-up |

The `MAXSEQS=2` row shows the catch. After sizing, `extensible_kv_cache.py` caps torch's allocator at its current
reservation + free memory - new KV - a 512 MiB non-torch reserve. Since the free memory left after the KV is
margin + transient reserve, torch may grow by exactly `margin + reserve - 512 MiB`. A smaller measured reserve
therefore means a lower cap. The same arithmetic explains the old `KV_MARGIN_MB=256` OOM: it put the cap 256 MiB
below the growth warm-up had measured.

With the shipped knobs and the stock 512 MiB non-torch reserve, a 262,000-token prompt completed, but 678 prefill
chunks fell back from split-KV attention (its 5-way partial buffers could not be allocated under the cap), 4.5%
slower. `NONTORCH_MB=256` (`VLLM_KV_NON_TORCH_RESERVE_MB`) raises the cap by 256 MiB without changing the KV size:
no fallbacks, and at least 502 MiB of the card stayed physically free throughout that prefill (`nvidia-smi` every
200 ms), so nothing came close to needing the reserve that was given up.

Decode at 262K is 18.8-19.9 ms per step, the same as FP8 at 131K. A 240K development run had 21-23 ms in its first
two cells; that did not reproduce. The decode attention split count depends only on the batch, not on `MAXLEN`.

## 6. FP8 KV cache on sm_86 (`KV_DTYPE=fp8`)

The model is built for a unit-scale E4M3 KV cache (no per-tensor scales), and the image already has the FP8 path:
FP8 attention blocks hold 128 tokens per page instead of 64, and the Triton attention loader converts pairs with
`cvt.rn.f16x2.e4m3x2`, which needs sm_89+. Issue #2 replaced the conversion only:

- `e4m3.py`: Triton `e4m3_encode` (fp32 -> byte, round to nearest even, saturating at +-448, NaN -> 0x7F) and
  `e4m3_decode`. Decode moves sign/exponent/mantissa into an fp16 bit pattern, which is exactly value x 2^-8 for
  normals and subnormals alike (the biases differ by 8), then multiplies by 256 (idea from HyperInkling).
  `tests/kernels/test_e4m3.py` checks both against `torch.float8_e4m3fn`, bit-exact: decode on all 256 codes, encode
  on 10M random values over many scales plus every boundary and tie case.
- `attention_sm90.py`: the loader decodes in software. `inkling2_attention.py` accepts `fp8` below sm_89 and views
  the cache as `uint8` for those kernels. `sconv_ops.py` encodes K/V on the cache write.
- `serve.sh` sets `--prefix-cache-retention-interval 128` with FP8 (it must be a multiple of the 128-token block),
  and the conv tail covers 2 x 64 tokens of prefix-hit alignment.

Cost on one GPU (`PROFILE=262k` vs `PROFILE=128k`, same chunk size): decode step +3-5% (18.4-19.0 vs 17.8-18.6 ms);
prefill -7% at 2K, -12% at 16K, -23% at 64K and -30% at 127K. The gap grows with context, as expected if the
software decode of every cached key the attention reads is the cost (not profiled). It is a precision change: temp-0
output diverges from bf16 within the first 18-584 characters (8 greedy prompts, median ~330; in the earliest cases
the reasoning rewords its first sentence). Task metrics do not move (MMLU 911 vs 912, needles at 255K and 262K found at 5% and 90%, 8/8
deterministic).

### On 2 GPUs (`PROFILE=512k` / `1m`)

`SCONV_TAIL=1` is worth more than FP8 KV on two cards: at MAXLEN 262144 the pool grows 302K -> 483K with no speed
change, against 302K -> 440K and a ~45% slower long prefill for FP8. With async scheduling off as well, 524,288 fits
in bf16 (pool 535,422) at the default GMU 0.97. For 1M, FP8 + the tail + the single-GPU knobs (GMU 1.0, 512 MiB
margin, 2 sequences, CUDA graphs to 16, 1024-token chunks, 64-row prompt logprobs, `NONTORCH_MB=256`, host
embedding) give 1,325,310 tokens; without the knobs vLLM estimated 906K-949K. Full matrix, including the varloop
checkpoint with shared KV and the second-agent test: `experiments/long-context-2gpu/RESULTS.md`.

## 7. Tried and rejected (measured)

- Adaptive DSpark verify: needs varlen-graph overlays; break-even at batch 1 and temp-0 output not reproducible.
- Full-attention BLOCK_M=64 (0.84x), tail pipeline with 3 pieces (-2%), host all-reduce launch-config sweep (shipped
  4x256 already optimal), dense Marlin (no sm_86 build in this image), skinny GEMM on big projections.
- `--max-num-batched-tokens 4096` at TP2: the startup KV check fails at 262K.
- Tail pipeline in 4 x 512-row pieces with `FAR_ATTN=1`: 500K cold 147 s vs 144-146 s (MoE GEMMs lose more at 512 rows).
- Prefill attention tile sweep for the far path (BLOCK_N 32/128, 4 warps, BLOCK_M 64, 3 stages): all slower or over
  the 99 KB shared-memory limit; 128 x 64, 8 warps, 2 stages stays.
- DSpark k=4 at one stream (warm 64K agent turns): 227 vs 245 tok/s for k=7 (steps 7% cheaper, 14% fewer tokens
  accepted; code turns -15%).
- `draft_lm_head_quantization`: vLLM only allows it for DeepSeek drafters.
- `expandable_segments`: worse fragmentation for the single-GPU fit.
- Sequence-parallel prefill path in the model: needs NVLink symmetric memory.
- `KV_MARGIN_MB=256` on one GPU: boots, then OOMs under a real long prefill (see "The 262K memory budget" for why).
- `MAXSEQS=2` for the 262K profile: more KV, but a lower torch cap, OOM in the post-boot warm-up.

## 8. Numerics

Speculative decoding at temp 0 is not token-identical to plain greedy decoding (bf16 verify numerics differ by batch
shape), so losslessness is judged by task metrics: sequential MMLU letter accuracy (57 subjects x 20), long-context
needle retrieval and temp-0 run-to-run determinism. Host all-reduce/all-gather: 0/1140 MMLU answer differences on vs
off. The two prefill kernels are bit-identical to the stock ones (`tests/kernels`). The opt-in `FAR_ATTN=1` is not:
it differs from the shipped attention kernel at fp32 rounding level (`experiments/prefill/relfar/bench.py`).
