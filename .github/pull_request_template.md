## What and why

<!-- One or two sentences. Link the issue if there is one. -->

## Measurements (A/B, same machine, same image)

| | before | after |
|---|---|---|
| decode tok/s code / explanation / story | | |
| prefill tok/s at 16K | | |
| max context / KV pool | | |

Hardware, profile and power limit:

## Quality gate

- [ ] `tests/api_smoke.py` passes
- [ ] MMLU (`tests/mmlu_letters.py --workers 1`) within ±3 of the README number for that profile
- [ ] needle near max context, 5% and 90% depth, found
- [ ] temp-0 determinism (`tests/greedy_check.py` twice) identical
- [ ] kernel change: bit-exactness harness in `tests/kernels/` (or a numerics argument why it can't be bit-exact)

## Housekeeping

- [ ] new switch defaults to the old behaviour unless the gate above passes, and is documented in the `launch/serve.sh` header
- [ ] `bash tools/make_diffs.sh` run, `patches/diffs/` updated
- [ ] README / docs/NOTES.md updated if numbers or behaviour changed
