# 2x NVIDIA A40 (48 GB, sm_86): community report

Reported by [buzz](https://github.com/buzz) in issue #3 (2026-11-23):
[first report](https://github.com/lulakien/inkling2-mini-3090-recipe/issues/3#issuecomment-7109679182),
[512K needle](https://github.com/lulakien/inkling2-mini-3090-recipe/issues/3#issuecomment-7116338455).
Numbers are as reported; they were not re-run here.

## Setup (different from this repo's Docker path)

- 2x NVIDIA A40 (48 GB each, sm_86), tensor parallel 2, torch 2.14.0+cu130.
- Upstream vLLM built from source at the Inkling-2-Mini commit (`ab18334a4f`), with this repo's overlays applied to
  site-packages (the diffs plus the new modules), `W8` int8 on attn/head/markov/draft, DSpark k=7.
- `HOSTAR=0`, `CEAR=0`: A40s have P2P, so the host-memory all-reduce meant for GeForce PCIe rigs is not needed.
- GMU 0.97, bf16 KV, `MAXLEN=262144` (the `2gpu` profile). No power cap.

## Results

| | 2x A40, `2gpu` profile (bf16 KV) | 2x A40, 512K with FP8 KV |
|---|---|---|
| KV pool | 1,042,673 tokens (~3.98x a full 262K request), 32.81 GiB per rank, ~3 GiB free per card | |
| decode, prose | ~96-100 tok/s (mean acceptance ~1.67) | |
| decode, code | ~250-290 tok/s (mean acceptance ~5.1) | |
| prefill | ~8.8K tok/s peak (short prompts) | 524,032-token prompt in 281 s (1,862 tok/s) |
| needle at 90% depth | | found (`7429-ORCHID`, `tests/needle.py`) |

Caveats from the report: single stream, short and medium prompts for the speed numbers, 48 GB cards with less memory
bandwidth than a 3090 and no power cap.

Raw needle output:

```json
{"prompt_tokens": 524032, "needle_at": 471583, "seconds": 281.4, "prefill_tok_s": 1862, "answer": " 7429-ORCHID.\n\n# ====", "found": true}
```
