#!/usr/bin/env python3
"""Throughput with N concurrent streams (the recipe is tuned for single-stream latency; this shows what batching gives).
  concurrency.py OUT.json [--port 8000] [--levels 1,2,4,8] [--tokens 512]
Each stream: different coding/explanation prompt, temp 0.6, ignore_eos so every stream generates exactly --tokens.
Reports aggregate tok/s, mean per-stream decode tok/s (after the first token) and mean time to first token
(reasoning or content, streaming). Prompts are short, so TTFT here is queueing + prefill of ~30 tokens."""
import argparse, json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--levels", default="1,2,4,8"); ap.add_argument("--tokens", type=int, default=512); a = ap.parse_args()
B = f"http://127.0.0.1:{a.port}"; M = os.environ.get("INKLING_MODEL", "inkling-2-mini")
TOPICS = ["an LRU cache in Python with tests", "how TCP congestion control works", "a Rust CLI that counts words in files",
          "how a B-tree index works in a database", "a JavaScript debounce and throttle implementation",
          "how public-key cryptography works", "a C function that parses CSV lines", "how garbage collection works in Go"]
def one(i):
    body = {"model": M, "stream": True, "max_tokens": a.tokens, "ignore_eos": True, "temperature": 0.6, "top_p": 0.95,
            "messages": [{"role": "user", "content": f"Write about {TOPICS[i % len(TOPICS)]} in detail."}],
            "chat_template_kwargs": {"reasoning_effort": "none"}, "stream_options": {"include_usage": True}}
    r = urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; n = 0
    for line in urllib.request.urlopen(r, timeout=1800):
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:])
        dl = d["choices"][0].get("delta", {}) if d.get("choices") else {}
        if ttft is None and (dl.get("content") or dl.get("reasoning") or dl.get("reasoning_content")):
            ttft = time.time() - t0
        if d.get("usage"):
            n = d["usage"]["completion_tokens"]
    dt = time.time() - t0
    return {"tokens": n, "seconds": dt, "ttft": ttft, "tok_s": n / (dt - (ttft or 0))}
one(0)  # warm
res = {}
for c in [int(x) for x in a.levels.split(",")]:
    t = time.time()
    with ThreadPoolExecutor(c) as ex:
        rows = list(ex.map(one, range(c)))
    wall = time.time() - t
    agg = sum(r["tokens"] for r in rows) / wall
    res[c] = {"aggregate_tok_s": round(agg, 1), "per_stream_tok_s": round(sum(r["tok_s"] for r in rows) / c, 1),
              "mean_ttft_s": round(sum(r["ttft"] or 0 for r in rows) / c, 3), "wall_s": round(wall, 2)}
    print(c, res[c], flush=True)
json.dump(res, open(a.out, "w"), indent=1)
