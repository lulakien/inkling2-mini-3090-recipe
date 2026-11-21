"""Exactness harness: parallel Triton MoE align vs the torch portable contract.
Run in the image with a free GPU (it allocates a few MB):
  bash tests/kernels/run.sh   (or by hand:  docker run --rm --gpus device=0 --ipc=host -v $PWD/patches:/p:ro \
    -v $PWD/tests/kernels:/k:ro --entrypoint python3 \
    vllm/vllm-openai:inkling-2-mini /k/test_align.py)
"""
import sys
sys.path.insert(0, "/p")  # patches/ mounted at /p (see tests/kernels/run.sh)
import torch
import moe_align_block_size as M

torch.manual_seed(0)
dev = torch.device("cuda", 0)


def make_case(n, E, BS, topk=8, invalid=0.05, use_map=False, use_scatter=True, pad=False):
    ids = torch.randint(0, E, (n,), device=dev, dtype=torch.int32)
    if invalid:
        mask = torch.rand(n, device=dev) < invalid
        ids[mask] = -1
    # ids arrive as [tokens, topk] in the real call; flattening is what the op does
    topk_ids = ids.view(max(1, n // topk), -1) if n % topk == 0 else ids.view(1, -1)
    cap = n + E * (BS - 1)
    if pad:
        from vllm.utils.math_utils import round_up
        cap = round_up(cap, BS)
    if n < E:
        cap = min(n * BS, cap)
    emap = None
    if use_map:
        emap = torch.arange(E, device=dev, dtype=torch.int32)
        emap[::2] = -1
    outs = {}
    for tag, fn in (("torch", M._moe_align_block_size_portable),
                    ("parallel", M._moe_align_block_size_parallel)):
        sorted_ids = torch.empty(cap, device=dev, dtype=torch.int32)
        expert_ids = torch.empty((cap + BS - 1) // BS, device=dev, dtype=torch.int32)
        npad = torch.empty(1, device=dev, dtype=torch.int32)
        scat = torch.empty((n, 1), device=dev, dtype=torch.int32) if use_scatter else None
        fn(topk_ids, E, BS, sorted_ids, expert_ids, npad, emap, scat)
        outs[tag] = (sorted_ids.clone(), expert_ids.clone(), npad.clone(),
                     scat.clone() if scat is not None else None)
    ok = all((a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
             for a, b in zip(outs["torch"], outs["parallel"]))
    if not ok:
        for i, name in enumerate(("sorted_ids", "expert_ids", "npad", "scatter")):
            a, b = outs["torch"][i], outs["parallel"][i]
            if a is None:
                continue
            if not torch.equal(a, b):
                idx = (a != b).nonzero()[:6].flatten().tolist()
                print(f"  MISMATCH {name} at {idx}: torch={a.flatten()[idx].tolist()} par={b.flatten()[idx].tolist()}")
    return ok


cases = []
for n in (64, 256, 512, 1024, 2048, 8192, 16384):
    for BS in (16, 32, 64):
        cases.append((n, 128, BS, dict()))
cases += [
    (8192, 128, 16, dict(use_map=True)),
    (8192, 128, 16, dict(use_map=True, use_scatter=False)),
    (8192, 128, 16, dict(use_scatter=False)),
    (4096, 128, 16, dict(invalid=0.3)),
    (8192, 128, 16, dict(invalid=0.0)),
    (8192, 64, 16, dict()),          # TP2-like local expert count
    (8192, 64, 16, dict(use_map=True)),
    (64, 128, 16, dict()),           # n < E
    (64, 128, 16, dict(pad=True)),
    (1024, 256, 32, dict()),
]
bad = 0
for n, E, BS, kw in cases:
    ok = make_case(n, E, BS, **kw)
    print(f"n={n:5d} E={E:3d} BS={BS:2d} {str(kw):40s} {'OK' if ok else 'FAIL'}", flush=True)
    bad += not ok
print("ALL EXACT" if bad == 0 else f"{bad} FAILURES")
sys.exit(1 if bad else 0)
