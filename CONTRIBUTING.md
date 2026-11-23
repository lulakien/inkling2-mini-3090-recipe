# Contributing

Help is very welcome: benchmark reports from other cards, bug reports, optimization ideas, and patches.
The open problems are listed in the README's [Roadmap](README.md#roadmap--help-wanted).

## The cheapest useful contribution: numbers from your machine

Start a profile, run `tests/run_suite.sh` (or only `spec_bench.py` and `prefill_ttft.py`), and open a
**Benchmark report** issue with the output of `bash tools/env_report.sh`. Other GPUs (3090 Ti, A5000, A6000,
4090), NVLink pairs, stock cards without gpu-unlock, and other power limits are all missing today.

## How the repo is put together

- `launch/serve.sh` mounts each file in `patches/` read-only over the pinned image (`launch/image.env`).
  Overlays replace **whole files**, so they only fit that exact image build.
- Every behaviour change sits behind an environment switch in `serve.sh`, documented in its header.
- `patches/diffs/` holds each overlay's diff against the stock file. Regenerate with `bash tools/make_diffs.sh`.
- `tests/` has everything the README numbers come from. `results/` holds the raw JSON.

## Ground rules for patches

1. **The checkpoint stays as shipped.** No requantized experts or edited weights in the default path. On-the-fly
   transforms such as the int8 per-channel weights are fine if they pass the quality gate and have a switch.
2. **Measure A/B on the same machine and image**, ideally the same boot (switch on/off) or back-to-back boots.
   Report decode (`spec_bench.py`, use the second run of a boot) and prefill (`prefill_ttft.py`). Run-to-run noise is
   about 1-2%; smaller gains need repeated runs.
3. **Quality gate** (the PR template has the checklist): API smoke test, sequential MMLU subset within ±3 of the
   README number, needle near max context at 5% and 90% depth, temp-0 determinism. Kernel rewrites should come with
   a bit-exactness harness like the ones in `tests/kernels/`.
4. **New switches default to the old behaviour** until they pass the gate on both profiles they touch.
5. Keep changes small and in the style of the file they touch. Prefer one PR per idea.

## Debugging tips

- The stock image calls a few `_C` kernels that exist only for sm_100. On Ampere they **fail silently**: the error
  is left in the CUDA runtime and the next torch op gets blamed, or the output is just wrong. If output turns to
  garbage after a change, suspect this first.
- Delete `~/.cache/inkling2-vllm` after changing a Triton kernel if you see stale behaviour.
- `docs/NOTES.md` has the per-step time breakdown and the list of things already tried and rejected.

## License

By contributing you agree your contribution is licensed under Apache-2.0, like the rest of the repo.
