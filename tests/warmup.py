#!/usr/bin/env python3
"""Post-boot warm-up: trigger the one-time Triton JITs (long-prompt split-KV attention, prefix-hit chunk shapes,
DSpark decode, batched decode and sampling at 2-8 streams) so the first real request does not pay them.
warmup.py [--port 8000]"""
import argparse, json, os, random, threading, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8000); ap.add_argument("--tokens", type=int, default=34000)
a = ap.parse_args()
URL = f"http://127.0.0.1:{a.port}/v1/completions"
MODEL = os.environ.get("INKLING_MODEL", "inkling-2-mini")
def req(prompt, n, temperature=0):
    t = time.time()
    body = {"model": MODEL, "prompt": prompt, "max_tokens": n, "temperature": temperature}
    if temperature:
        body["top_p"] = 0.95
    body = json.dumps(body).encode()
    urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=600).read()
    return time.time() - t
rng = random.Random(1234)
ids = [rng.randrange(1000, 100000) for _ in range(a.tokens)]
t0 = time.time()
s = [req(ids, 1), req(ids + [rng.randrange(1000, 100000) for _ in range(2500)], 1),
     req("Write a Python function that merges two sorted lists, with tests.", 200)]
def burst(n, temperature):
    t = time.time()
    th = [threading.Thread(target=req, args=(f"Explain topic number {i} of computer networking in detail.", 48, temperature))
          for i in range(n)]
    [x.start() for x in th]; [x.join() for x in th]
    return time.time() - t
tb = time.time()
for n in (2, 3, 4, 5, 6, 8):
    for temp in (0, 0.6):
        burst(n, temp)
# several prompts prefilling in one step (>= 2048 MoE routes): the parallel MoE align kernels
long_prompts = [[rng.randrange(1000, 100000) for _ in range(300)] for _ in range(8)]
th = [threading.Thread(target=req, args=(p_, 8)) for p_ in long_prompts]
[x.start() for x in th]; [x.join() for x in th]
# and small concurrent prefills whose total token count is odd (Triton specializes on divisibility by 16)
for n, length in ((3, 101), (5, 67), (2, 33)):
    th = [threading.Thread(target=req, args=([rng.randrange(1000, 100000) for _ in range(length)], 4)) for _ in range(n)]
    [x.start() for x in th]; [x.join() for x in th]
s.append(time.time() - tb)
print("WARMUP done in %.1f s (long %.1f, extend %.1f, decode %.1f, batched %.1f)" % (time.time() - t0, *s))
