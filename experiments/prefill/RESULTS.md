# Cold long prefill and warm decode on 2x RTX 3090 (2026-11-24)

Same machine and pinned image as the README. Everything here is opt-in; defaults are unchanged.
Scripts next to this file (`run_trace.sh`, `summarize_trace.py`, `relfar/`, `round2-4.sh`), raw output in `results/`.
The switch is `FAR_ATTN=1` in `launch/serve.sh` (the experiment scripts set `INKLING2_FAR_ATTN=1` directly).

## 1. Where 500K cold prefill goes (`PROFILE=512k`, bf16 KV)

Torch-profiler trace of two 2048-token chunks ~485K deep (`results/trace_512k/kernels.txt`); cold 500K needle 166.8 s.

| share of GPU time | what |
|---|---|
| 70.6% | `_rel_attention_kernel` (full attention): 8 launches per chunk (4 full layers x 3 loop passes = 8 visits), 93.5 ms each, 63.6 TFLOPS |
| 13.3% | host-staged TP2 all-reduce copies (DtoH + HtoD, 96 all-reduces per chunk); ~67 ms per chunk not hidden behind compute |
| 11.0% | MoE experts (Humming) |
| ~5% | conv, AttnRes mix, norms, dense GEMMs |

The 5 sliding-window launches per layer visit are 0.3 ms each and do not matter here.

## 2. Far-block fast path in the prefill kernel (opt-in `INKLING2_FAR_ATTN=1`)

The relative bias only reaches 1024 keys back. The shipped kernel still runs the bias load, distance masks and
`-inf` handling on every key block. The patch adds a first loop over the key blocks that lie wholly before every
row's relative extent (all visible, all valid, no bias), then runs the unchanged loop on the rest. Same split
interleaving, same online softmax, same output layout. No second kernel, no LSE merge, no cache copy; this replaces
the FA2-far + Inkling-near plan, since it already reaches FA2's speed on the same shape.

Microbenchmark (`relfar/bench.py`, output `relfar/bench_fixed.txt`; GPU 1, one 2048-token chunk, 12/4 heads,
paged KV in the real layout behind a shuffled page table, 5 splits). Reference = the shipped kernel file before the
patch; the overlay with the switch off is `torch.equal` to it at every row below.

| KV, depth | shipped | far path | max abs / relative diff vs shipped |
|---|---|---|---|
| bf16, 64K | 12.56 ms | 9.87 ms (x1.27) | 4.8e-6 / 1.4e-4 |
| bf16, 254K | 49.74 ms | 39.92 ms (x1.25) | 2.4e-6 / 1.3e-4 |
| bf16, 485K | 94.53 ms | 74.81 ms (x1.26) | 1.4e-6 / 9.6e-5 |
| fp8, 64K | 25.96 ms | 23.08 ms (x1.13) | 1.1e-6 / 7.3e-5 |
| fp8, 254K | 99.33 ms | 86.48 ms (x1.15) | 4.0e-7 / 5.9e-5 |
| fp8, 485K | 188.57 ms | 163.97 ms (x1.15) | 2.4e-7 / 4.9e-5 |

For reference, FA2 (vLLM's sm_80 build) on only the far prefix of the 485K shape (no bias, no near tail): 73.8 ms
paged (`fa2_far_bench.py`). Outputs are finite everywhere and differ from the shipped kernel at fp32 rounding level,
~30x below bf16 resolution; they are not bit-identical. (A first version of the benchmark left the shuffled KV pool
unfilled after an edit, which produced NaN in the FP8 runs and wrong "bit-identical" claims; the table above is the
corrected rerun.)

FP8 gains less: that kernel is bound by the software E4M3 decode. Tile sweeps (BLOCK_N 32/128, 4 warps, BLOCK_M 64,
3 stages) were all slower or did not fit in shared memory; 128x64 tiles, 8 warps, 2 stages stays.

End to end (`round2.sh`, 512k profile, cold needles):

| prompt | shipped | far path |
|---|---|---|
| 500K, needle at 50% | 166.8 s | **144.3 s (-13.5%)**, found |
| 500K, needle at 10% | | 145.7 s, found |
| 16K | 2.3 s (7,140 tok/s, on the 262k profile) | 2.4 s (6,916 tok/s), found |
| 500K agent turns (`tests/agent_turns.py --ctx 500000`), repeat on a fresh boot | cold TTFT 166.8 s (`long-context-2gpu/results/fixed_tail_512k`) | **145.6 s**; cold / warm / 8-fact multi / post-compaction all found, as before |
| 500K, needle at 50%, same boot | | 145.8 s, found |

KV pool unchanged (kernel-only change, verified on the 1m profile: 1,290,023 tokens with and without).
Decode is untouched: decode/verify run `rel_decode_attention`, and the draft path (DRAFT) skips the fast loop.

Four cold 500K runs: 144.3-145.8 s (-13%). Short of the 140 s target. The rest is mostly the exposed all-reduce copies:
- CEAR tail in 4 x 512-row pieces (default 2 x 1024), with the far path: 500K **147.3 s** (slower than 144.3 s),
  16K 2.4 s. The MoE GEMMs lose more at 512 rows than the extra overlap hides. Dropped.

## 3. 1m profile (fp8 KV): MNBT 2048 so the CEAR tail pipeline runs (`round4.sh`)

The 1m profile ships `MNBT=1024`, below the tail pipeline's 1536-token minimum, so it never pipelined.

| config | KV pool | cold 250K needle |
|---|---|---|
| shipped (MNBT 1024) | 1,324,318 | 100.6 s, found |
| MNBT 2048 | 1,290,023 (still 1.23x of 1M) | 85.5 s (-15%), found |
| MNBT 2048 + far path | 1,290,023 | **79.1 s (-21%)**, found |

1.04M cold needle (90% depth) on MNBT 2048 + far path: **896.8 s (14.9 min) vs 19.5 min shipped (-23%)**, found, no OOM.

## 4. Warm decode: k=4 vs k=7 at one stream (`../concurrency/warm_decode.py`)

64K agent transcript (prefilled once, every turn ~65K cached), reasoning / code / edit questions, temp 0, medium
effort, 2,500 tokens, two runs each, default 262k profile. Outputs differ between runs (not batch-invariant), so
compare means.

| k at 1 stream | mean decode | tokens per step | ms per step | code turns |
|---|---|---|---|---|
| 7 (default) | **245.5 tok/s** | 3.82 | 15.52 | 299 / 281 |
| 4 | 227.1 tok/s | 3.27 | 14.40 | 251 / 250 |

k=4 steps are 7% cheaper but accept 14% fewer tokens; reasoning turns tie, code loses ~15%. k=7 stays.
