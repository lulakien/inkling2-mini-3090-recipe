#!/usr/bin/env python3
"""Greedy determinism/losslessness check. greedy_check.py OUT.json [--ref REF.json] [--port 8000] [--n 256]
8 prompts, effort none, temp 0, /v1/completions on the rendered chat template. With --ref, reports the first
differing character per prompt (spec decoding at temp 0 should track plain greedy until a near-tie)."""
import argparse, json, urllib.request, os
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--ref"); ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--n", type=int, default=256); ap.add_argument("--workers", type=int, default=1); a = ap.parse_args()
M = os.environ.get("INKLING_MODEL", "inkling-2-mini")
P = ["Write a Python function that merges two sorted lists, with a docstring and tests.",
     "Explain how TCP congestion control works.",
     "Write a short story about a lighthouse keeper who finds a message in a bottle.",
     "List the planets of the solar system with one fact about each.",
     "What causes the seasons on Earth? Answer in three paragraphs.",
     "Translate to Turkish: The weather is nice today and we are going to the beach.",
     "Write a bash script that backs up a directory into a timestamped tar.gz file.",
     "Summarize the plot of Romeo and Juliet."]
def post(path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{a.port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))
def run(p):
    ids = post("/tokenize", {"model": M, "messages": [{"role": "user", "content": p}], "add_generation_prompt": True,
                             "chat_template_kwargs": {"reasoning_effort": "none"}})["tokens"]
    r = post("/v1/completions", {"model": M, "prompt": ids, "max_tokens": a.n, "temperature": 0, "ignore_eos": True})
    return r["choices"][0]["text"]
with ThreadPoolExecutor(a.workers) as ex: out = list(ex.map(run, P))
json.dump(out, open(a.out, "w"), indent=1)
if a.ref:
    ref = json.load(open(a.ref)); same = 0
    for i, (x, y) in enumerate(zip(out, ref)):
        j = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), None)
        same += j is None
        print(i, "identical" if j is None else f"diverges at char {j}/{len(y)}: {x[max(0,j-30):j+30]!r} vs {y[max(0,j-30):j+30]!r}")
    print(f"identical {same}/{len(P)}")
