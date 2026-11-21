#!/usr/bin/env python3
"""Single-stream decode speed + DSpark acceptance on realistic prompts. spec_bench.py OUT.json [--port 8000]
Speed = completion tokens / wall time of a non-streamed request (prefill is tiny here). Acceptance from /metrics deltas."""
import json, re, sys, time, urllib.request, os
port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
IGNORE_EOS = "--ignore-eos" in sys.argv
M = os.environ.get("INKLING_MODEL", "inkling-2-mini"); B = f"http://127.0.0.1:{port}"
PROMPTS = {
 "code": "Write a complete Python implementation of an LRU cache class with get/put in O(1), with docstrings and a few unit tests.",
 "explain": "Explain how TCP congestion control works (slow start, congestion avoidance, fast retransmit, fast recovery) in detail.",
 "story": "Write a long story about a lighthouse keeper who discovers a message in a bottle.",
}
def post(path, body):
    r = urllib.request.Request(B + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=1800))
def metrics():
    txt = urllib.request.urlopen(B + "/metrics").read().decode(); out = {}
    for k in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        m = re.findall(rf"^vllm:spec_decode_{k}_total\{{[^}}]*\}} ([0-9.e+]+)", txt, re.M)
        out[k] = sum(float(x) for x in m)
    return out
post("/v1/chat/completions", {"model": M, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
res = {}
for temp in (0.0, 0.6):
    for name, p in PROMPTS.items():
        m0 = metrics(); t = time.time()
        r = post("/v1/chat/completions", {"model": M, "messages": [{"role": "user", "content": p}], "max_tokens": 1500,
                                          "temperature": temp, "top_p": 0.95, "top_k": 20, "ignore_eos": IGNORE_EOS,
                                          "chat_template_kwargs": {"reasoning_effort": "low"}})
        dt = time.time() - t; m1 = metrics(); n = r["usage"]["completion_tokens"]
        d = {k: m1[k] - m0[k] for k in m0}
        row = {"tok_s": round(n / dt, 1), "tokens": n, "text": r["choices"][0]["message"].get("content") or ""}
        if d["num_drafts"]:
            row["mean_accepted_per_step"] = round(d["num_accepted_tokens"] / d["num_drafts"], 2)
            row["tokens_per_step"] = round(1 + d["num_accepted_tokens"] / d["num_drafts"], 2)
            row["ms_per_step"] = round(dt * 1000 / d["num_drafts"], 2)
        res[f"{name}_t{temp}"] = row; print(name, temp, {k: v for k, v in row.items() if k != "text"}, flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
