#!/usr/bin/env python3
"""Cold prefill / TTFT ladder with zero prefix-cache reuse.

All prompts are unique random token ids, so a rising hit rate is impossible. Reports
wall time, derive tok/s, and (with --stream) time-to-first-token separately from the
prefill+1-token total.

  prefill_ttft.py OUT.json [--port 8000] [--sizes 1024,2048,4096,8192,16384,32768] [--reps 2]
  prefill_ttft.py OUT.json --extend 20000,2000   # agent-turn TTFT: 20K prompt, then +2K more
"""
import argparse, json, os, random, time, urllib.request, os

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--sizes", default="1024,2048,4096,8192,16384,32768")
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--extend", default="")
ap.add_argument("--vocab", type=int, default=199000)
ap.add_argument("--stream", action="store_true", help="measure TTFT with stream=true")
a = ap.parse_args()
M = os.environ.get("INKLING_MODEL", "inkling-2-mini")
B = f"http://127.0.0.1:{a.port}"


def post(path, body):
    r = urllib.request.Request(B + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))


def stream_ttft(ids):
    """POST with stream=true, return (ttft_s, total_s, completion_tokens)."""
    body = json.dumps({"model": M, "prompt": ids, "max_tokens": 2, "temperature": 0,
                       "ignore_eos": True, "stream": True}).encode()
    r = urllib.request.Request(B + "/v1/completions", body, {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; n = 0
    with urllib.request.urlopen(r, timeout=3600) as resp:
        for line in resp:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            if ttft is None:
                ttft = time.time() - t0
            n += 1
    return ttft, time.time() - t0, n


def rand_ids(n, seed):
    rng = random.Random(seed)
    return [rng.randrange(1000, a.vocab) for _ in range(n)]


post("/v1/completions", {"model": M, "prompt": [1, 2, 3], "max_tokens": 8, "temperature": 0})
out = {}
sizes = [int(x) for x in a.sizes.split(",") if x]
for n in sizes:
    row = []
    for rep in range(a.reps):
        ids = rand_ids(n, 100_000_000 + n * 10 + rep)
        t = time.time(); r = post("/v1/completions", {"model": M, "prompt": ids, "max_tokens": 1,
                                                     "temperature": 0}); dt = time.time() - t
        e = {"seconds": round(dt, 4), "tok_s": round(n / dt)}
        if a.stream:
            ids2 = rand_ids(n, 200_000_000 + n * 10 + rep)
            ttft, tot, ntok = stream_ttft(ids2)
            e.update(ttft_s=round(ttft, 4), total_s=round(tot, 4), completion_tokens=ntok,
                     ttft_tok_s=round(n / ttft))
        row.append(e)
        print(n, rep, e, flush=True)
    out[f"prefill_{n}"] = row
if a.extend:
    p, x = (int(v) for v in a.extend.split(","))
    ids = rand_ids(p, 777); ids2 = ids + rand_ids(x, 778)
    t = time.time(); post("/v1/completions", {"model": M, "prompt": ids, "max_tokens": 4, "temperature": 0}); first = time.time() - t
    t = time.time(); post("/v1/completions", {"model": M, "prompt": ids2, "max_tokens": 4, "temperature": 0}); second = time.time() - t
    out["extend"] = {"first_s": round(first, 4), "second_s": round(second, 4), "new_tokens": x,
                     "second_tok_s": round(x / second)}
    print("extend", out["extend"], flush=True)
json.dump(out, open(a.out, "w"), indent=1)
print("wrote", a.out)
