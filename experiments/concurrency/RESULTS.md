# Concurrency and TTFT on 2x / 1x RTX 3090 (2026-11-24)

Same machine as the README (PCIe, no P2P, 280 W, gpu-unlock), pinned image, Inkling 2 Mini. Raw JSON and server logs
in `results/<label>/`, console output in `round*.out`, the scripts that produced them next to this file.
`tests/concurrency.py`: N streams, different coding/explanation prompts, temperature 0.6, top-p 0.95, 512 tokens each
(ignore EOS), total tok/s. At temperature 0.6 the single-stream number moves a lot between runs (161-232 tok/s for
the same config), so compare multi-stream columns and repeated runs.

## 1. Draft length by batch size (landed, default)

DSpark verifies k=7 drafts per stream every step. At one stream that is the win (2.4x on code); with 8 streams a
step verifies 64 tokens while acceptance per stream stays the same, so the verify cost eats the gain and plain
decoding (`SPEC=0`) was faster at 8 streams. This vLLM build already has `num_speculative_tokens_per_batch_size`;
`serve.sh` now passes `SPEC_SCHED='[[1,1,7],[2,2,4],[3,4,2],[5,256,1]]'` (k=7 alone, 4 at two streams, 2 at
three-four, 1 beyond). `SPEC_SCHED=` restores k=7 everywhere.

2 GPUs, total tok/s at 1 / 2 / 3 / 4 / 6 / 8 streams:

| config | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| k=7 always (old default), round 1 | 167 | 227 | | 296 | | 429 |
| `SPEC=0`, round 1 | 127 | 205 | | 349 | | 545 |
| schedule A 7/5/3/1, round 1 | 169 | 239 | | 400 | | 551 |
| schedule B 7/7/3/0, round 1 | 168 | 227 | | 417 | | 540 |
| schedule A, mean of 2 | 198 | 244 | 351 | 413 | 486 | 524 |
| schedule C 7/5/4/3/2/1, mean of 2 | 200 | 251 | 322 | 410 | 497 | 552 |
| **schedule D 7/4/2/1**, mean of 2 | 200 | 253 | 350 | 416 | 489 | 554 |
| D as default, gate boot (2 runs) | 161, 221 | 267, 258 | 366, 350 | 411, 419 | 495, 494 | **578, 573** |
| D as default, final warm-up, fresh boot | 196 | 250 | 359 | 426 | 493 | **580** |

1 GPU (`PROFILE=128k`), total tok/s at 1 / 2 / 3 / 4 streams: k=7 151 / 191 / 223 / 265, schedule D
151 / 211 / 267 / 317 (+11% / +20% / +20%).

Single stream is untouched: same tokens, same acceptance, same step time as k=7 (code 312 tok/s, 14.2 ms/step).
Gate on the default 2-GPU profile: API 9/9, temp-0 greedy identical 8/8 to the old k=7 run, MMLU 911.
The extra CUDA graphs per draft length cost 0.1-0.4% of the KV pool on the 2-GPU 512k/1m profiles (533,475 and
1,324,318 tokens, both boot) but push the 1-GPU 262k profile 0.03 GiB short, so that profile keeps k=7
(`1gpu.sh` sets `SPEC_SCHED=` for it; it holds one 262K request anyway).

## 2. Warm-up covers batched shapes (landed)

After the old warm-up, the first multi-stream burst still compiled 14 Triton kernels (batched decode, the sampler,
MoE align variants) during real requests: one 1.25 s TTFT at 8 streams, and a slow first batch. `tests/warmup.py`
now also sends concurrent bursts at 2/3/4/5/6/8 streams (greedy and top-p sampling), one burst of eight 300-token
prompts (parallel MoE align path) and small odd-length concurrent prefills (Triton specializes integer arguments
on divisibility by 16). Fresh boot: 14 compiles during warm-up, **0 after two full concurrency runs**. Warm-up takes
~13 s instead of ~6 s.

## 3. A second agent joining (TTFT trade-off, opt-in `LPT`)

`tests/second_stream.py`, 2 GPUs, default profile with schedule D: agent A streams with a 200K context, agent B joins
with a cold 64K prompt. `--long-prefill-token-threshold` (`LPT=` in `serve.sh`) caps B's prefill chunk only while
another request is running.

| LPT | A alone | A during B's prefill | A while both decode | B's TTFT (alone: 10.6 s) | A's longest gap |
|---|---|---|---|---|---|
| off (default) | 181 tok/s | 11.7 tok/s | 142 | 11.0 s | 0.38 s |
| 2048 | 183 | 12.0 | 141 | 11.0 s | 0.38 s |
| 1024 | 179 | 16.5 | 139 | 14.1 s | 0.68 s |
| 512 | 185 | **28.3** | 145 | **16.2 s** | 0.24 s |

It trades the two agents' time almost linearly, so it stays off by default; `LPT=512` suits "keep the main agent
responsive, the second one can wait".

## 4. Bigger prefill chunks: no gain (not landed)

`MNBT=4096` failed to boot at 262K before; with `SCONV_TAIL=1` it now boots (pool 417,049 vs 479,073 at 2048).
Prefill 16K / 64K / 254K: 7,163 tok/s / 6,337 tok/s / 59.1 s, the same as 2048-token chunks (7,140 / 6,280 / 59.4 s).
Prefill is bound by per-token compute and PCIe all-reduce, not by per-chunk overhead.

## Notes

- A request that generates a single token leaves no reusable prefix cache when async scheduling is on (the
  second-stream test's 1-token pre-warm did not cache; with `ASYNC=0` it did). Normal agent turns (more than one
  output token) cache as before.
- Not tried yet from issue #4: more decode-attention splits, sm_86 expert-GEMM configs, `MAXSEQS` above 8.
