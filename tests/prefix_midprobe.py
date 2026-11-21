#!/usr/bin/env python3
"""Mid-request prefix divergence probe: long shared system prompt, long *different* user messages
(a shared prefix that ends mid-request, the pattern the handoff says gets 0 hits).
Prints hit/query deltas and wall time per request, then JSON.
  prefix_midprobe.py OUT.json [--port 8000] [--sys-words 700] [--user-words 120]"""
import json, random, sys, time, urllib.request, os

out_path = sys.argv[1]
port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
SYSN = int(sys.argv[sys.argv.index("--sys-words") + 1]) if "--sys-words" in sys.argv else 700
USERN = int(sys.argv[sys.argv.index("--user-words") + 1]) if "--user-words" in sys.argv else 120
B = f"http://127.0.0.1:{port}"; M = os.environ.get("INKLING_MODEL", "inkling-2-mini")


def post(p, b):
    r = urllib.request.Request(B + p, json.dumps(b).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=1200))


def counters():
    t = urllib.request.urlopen(B + "/metrics").read().decode()
    q = h = 0.0
    for line in t.splitlines():
        if line.startswith("vllm:prefix_cache_queries_total"):
            q = float(line.split()[-1])
        elif line.startswith("vllm:prefix_cache_hits_total"):
            h = float(line.split()[-1])
    return q, h


SYS = ("The quick brown fox jumps over the lazy dog while the system prompt keeps going. ") * SYSN
WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike".split()


def user(seed):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(USERN * 3))


def req(name, u):
    q0, h0 = counters(); t = time.time()
    r = post("/v1/chat/completions", {"model": M, "messages": [
        {"role": "system", "content": SYS}, {"role": "user", "content": u}],
        "max_tokens": 1, "temperature": 0})
    dt = time.time() - t; q1, h1 = counters()
    det = (r["usage"].get("prompt_tokens_details") or {})
    row = {"prompt": r["usage"]["prompt_tokens"], "cached": det.get("cached_tokens"),
           "wall_s": round(dt, 3), "hit_delta": int(h1 - h0)}
    print(name, row, flush=True)
    return row


res = {"sys_words": SYSN, "user_words": USERN}
res["A_cold"] = req("A_cold", user(1))
req("A2_same", user(1))
res["B_divergent"] = req("B_divergent", user(2))
res["C_divergent"] = req("C_divergent", user(3))
json.dump(res, open(out_path, "w"), indent=1)
print("wrote", out_path)
