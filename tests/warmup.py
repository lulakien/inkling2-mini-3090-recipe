#!/usr/bin/env python3
"""Post-boot warm-up: trigger the one-time Triton JITs (long-prompt split-KV attention, prefix-hit chunk shapes,
DSpark decode) so the first real request does not pay them. warmup.py [--port 8000]"""
import argparse, json, os, random, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8000); ap.add_argument("--tokens", type=int, default=34000)
a = ap.parse_args()
URL = f"http://127.0.0.1:{a.port}/v1/completions"
MODEL = os.environ.get("INKLING_MODEL", "inkling-2-mini")
def req(prompt, n):
    t = time.time()
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": n, "temperature": 0}).encode()
    urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=600).read()
    return time.time() - t
rng = random.Random(1234)
ids = [rng.randrange(1000, 100000) for _ in range(a.tokens)]
t0 = time.time()
s = [req(ids, 1), req(ids + [rng.randrange(1000, 100000) for _ in range(2500)], 1),
     req("Write a Python function that merges two sorted lists, with tests.", 200)]
print("WARMUP done in %.1f s (long %.1f, extend %.1f, decode %.1f)" % (time.time() - t0, *s))
