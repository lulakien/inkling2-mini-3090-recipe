# Long context on 2x RTX 3090: FP8 KV, SCONV_TAIL, varloop shared KV, a second agent (2026-11-24)

Question: what gives one fast long-context agent the most room on 2x3090 (PCIe, no P2P/NVLink, 280 W, gpu-unlock),
and what does a second agent cost the first? Same image pin, DSpark k=7 everywhere (it ran on every config,
including varloop with shared KV), TP2, bf16 KV unless stated. Raw output of every run is in `results/<config>/`,
the exact settings in each `config.env` / `command.txt` / `vllm_args.json`, and the console logs in `*.out`.

## What landed

- `PROFILE=512k bash launch/2gpu.sh`: **524,288 tokens in plain bf16** (SCONV_TAIL=1, async scheduling off,
  4 sequences; pool 535,422). Same decode as the 262k profile, 500K cold agent prompt in 167 s, 8/8 multi-needle at
  500K, MMLU 911, API 9/9, greedy 8/8 identical run to run.
- `PROFILE=1m bash launch/2gpu.sh`: 1,048,576 tokens with FP8 KV + SCONV_TAIL + the single-GPU memory knobs
  (pool 1,325,310). A 1,040,000-token needle at 90% depth is found, but the cold prompt took **19.5 min**; 250K
  cold 102 s vs 59 s. Use it only for one very long session that stays warm.
- `LOOP_SHARE=1` in `serve.sh`: `loop_share_cache` for the varloop research checkpoint (vLLM rejects it on fixed Mini).
- `tests/agent_turns.py` (agent-style transcript with planted facts: cold / warm / multi-needle / after compaction)
  and `tests/second_stream.py` (second agent joining mid-run).
- The 262k default is unchanged.

## Results (2 GPUs, MAXLEN 262144 unless the config name says otherwise)

`python3 -I summarize.py` regenerates this table from the JSON.

| config | KV pool | decode code/explain/story | 64K cold TTFT s | 250K cold TTFT s | 250K warm TTFT s | agent turns found (64K c/w/m/x, 250K c/w/m/x; * = no final answer) | MMLU /1140 |
|---|---|---|---|---|---|---|---|
| fixed_bf16 (reference; 1,536-token answer budget) | 302,534 | 315 / 166 / 126 | 11.45 | 58.94 | 1.1 | 1/1 1/1 8/8 1/1 1/1 1/1 0/8 * 1/1 | 911 |
| fixed_bf16_t4096 (same, 4,096 budget) | 302,534 | 317 / 167 / 126 | 10.54 | 58.79 | 1.1 | 1/1 1/1 8/8 1/1 1/1 1/1 8/8 1/1 | |
| fixed_fp8 | 439,820 | 198 / 175 / 129 | 13.79 | 85.65 | 1.79 | 1/1 1/1 8/8 1/1 1/1 1/1 8/8 1/1 | 913 |
| fixed_sconvtail | 482,577 | 312 / 165 / 125 | 10.7 | 59.32 | 1.13 | 1/1 1/1 8/8 1/1 1/1 1/1 7/8 * 1/1 | 911 |
| fixed_tail_512k (MAXLEN 524288, async off) | 535,422 | 333 / 171 / 127 (t=0.6) | | 500K: 166.8 | 500K: 1.98 | 500K: 1/1 1/1 8/8 1/1 | 911 |
| fixed_fp8_tail_1m (MAXLEN 1048576) | 1,325,310 | 286 / 159 / 118 | 15.89 | 102.25 | 2.03 | 1/1 1/1 8/8 1/1 1/1 1/1 0/8 * 1/1 | 912 |
| varloop3_native | 302,534 | 288 / 145 / 125 | 10.69 | 59.33 | 1.13 | 1/1 1/1 8/8 1/1 1/1 1/1 8/8 1/1 | 892 |
| varloop3_shared | 432,780 | 300 / 130 / 121 | 10.66 | 59.03 | 1.12 | 1/1 1/1 8/8 1/1 1/1 1/1 8/8 1/1 | 891 |
| varloop3_shared_fp8 | 544,546 | 261 / 132 / 122 | 12.76 | 85.42 | 1.73 | 1/1 1/1 8/8 1/1 1/1 1/1 8/8 1/1 | 894 |
| varloop3_shared_tail | 855,417 | 304 / 132 / 123 | 10.55 | 58.73 | 1.1 | 1/1 1/1 8/8 1/1 1/1 1/1 8/8 1/1 | 893 |
| varloop3_shared_tail_1m (MAXLEN 1048576) | 1,342,976 | 304 / 132 / 122 | | 1.0M: 625 | 1.0M: 3.42 | 1.0M, 16 needles: 1/1 1/1 0/16 * (16/16 in reasoning) 1/1 | |

Decode is the second run of `spec_bench.py` at temperature 0 (the code prompt's FP8 number is a different greedy
text with lower acceptance, not a slower kernel). TTFT is time to the first streamed token, reasoning included.

## Findings

1. **SCONV_TAIL is the big lever on 2 GPUs, and it is free.** 302K -> 483K pool at 262K MAXLEN with unchanged decode,
   prefill and MMLU. With async scheduling off it holds 524,288 in bf16 (the `512k` profile).
2. **FP8 KV buys less on 2 GPUs than the tail and costs prefill.** 302K -> 440K pool, 250K cold TTFT 59 s -> 86 s
   (software E4M3 decode in the attention loop). Only worth it to go past ~530K.
3. **Varloop shared KV works and costs nothing measurable over native varloop** (MMLU 891 vs 892, all agent turns
   pass, DSpark still runs), pool +43%. Shared + SCONV_TAIL reaches 855K at 262K MAXLEN and 1.34M at 1M in bf16,
   and a 1.0M cold prompt took 10.4 min against fixed Mini's 19.5 min (FP8). **But the varloop checkpoint itself is
   19-20 MMLU questions below fixed Mini** (891-894 vs 911-913) and DSpark accepts less on prose (explain 130-145 vs
   166 tok/s). Its MRCR advantage (512K-1M) did not show in our tests because fixed Mini also passed every retrieval
   we ran up to 1.04M; our test is easier than 8-needle MRCR. Not landed as a profile; `LOOP_SHARE=1` is there to try.
4. **Retrieval holds to the edge.** Every config found every planted fact; the only misses are turns that spent the
   whole 4,096-token budget reasoning (listing files one by one) before writing the answer. Six seeds of the 250K
   multi-needle turn: SCONV_TAIL 3 full answers + one 7/8 cut-off, bf16 5 full answers; the failures were the same
   kind in both (`results/tail_seeds`, `results/bf16_seeds`). SCONV_TAIL output is not bit-identical to bf16 at 250K
   (it drifts late in long answers), but the drift does not favour either side.
5. **A second agent costs the first one a stall while it prefills, then about a quarter of its speed.**
   SCONV_TAIL at 262K (pool 483K): agent A (200K warm context) 182 tok/s alone, **11 tok/s during B's 11 s prefill
   of 64K**, 134 tok/s while both decode (B 196 tok/s), back to 180 after. B's TTFT 11.3 s vs 10.6 s alone.
   On the 1m profile: 129 -> 11.5 -> 88 tok/s, B TTFT 17.8 s vs 15.9 s. The longest gap between A's chunks was
   0.4-0.7 s, so A is slowed, not frozen.
6. **After compaction the next turn is cold.** The compacted prompt shares only the system prompt, and with the
   hybrid cache's retention points the hit was 0-2,688 tokens, so its TTFT is a fresh prefill of the summary
   (0.5-1.1 s for ~5.4K tokens here).

## Not run / caveats

- 2DSD (SqueezeAILab, `feat/inkling-dspark` @ 962874f): built in its own env (~/ai/2dsd, vLLM 0.26 fork, Humming
  0.1.20). On one 3090 the packed-expert path runs out of memory converting the last MXFP8 expert layer, ~90 MB
  short even after two local patches that free the raw expert bytes after conversion. The runner holds its dense
  weights in FP32 (5.45 GiB before the experts) and has no tensor parallelism, so on 24 GB it needs a real port
  (BF16 dense storage or the drafter on the second card). No 2DSD speed numbers from this machine.
- Varloop at 4 passes: not run (DSpark needs 3, and our retrieval tests did not separate 3-pass varloop from fixed).
- Fixed Mini 1M multi-needle: not run (only the 1.04M single needle); the 1M profile's cold prefill is the blocker.
- The varloop 1m second-stream run is invalid (agent A finished before B's first token); the test was changed to
  join after 50 chunks with an 8,000-token answer afterwards.
- The "warm" turn appends a small file in these transcripts (~100-2,000 tokens), so warm TTFT is the cached-prefix
  case, not a big new tool result.
- `agent_turns.py` forces `tool_choice: none` and tells the model to answer from context; with tools allowed the
  model chose to grep instead of answering.
- The 1-token warm-up of `second_stream.py` left no reusable cache under SCONV_TAIL with async scheduling on, so
  that run's agent A started cold (a_cached 0); with async off (the 512k/1m profiles) it was cached.
