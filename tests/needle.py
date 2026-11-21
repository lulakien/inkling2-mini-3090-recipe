#!/usr/bin/env python3
"""Long-context fill test: hide a needle fact early in an N-token haystack (repo source code), ask for it at the end.
  needle.py N [--depth 0.05] [--port 8000]"""
import json, sys, time, urllib.request, os
N = int(sys.argv[1]); depth = float(sys.argv[sys.argv.index("--depth") + 1]) if "--depth" in sys.argv else 0.05
port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
B = f"http://127.0.0.1:{port}"; M = os.environ.get("INKLING_MODEL", "inkling-2-mini")
def post(p, b):
    r = urllib.request.Request(B + p, json.dumps(b).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))
import glob
srcs = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "haystack", "**", "*.py"), recursive=True))
hay = ""
for f in srcs * 4:
    hay += f"\n# ==== file {f} ====\n" + open(f).read()
    if len(hay) > N * 4.2: break
ids = post("/tokenize", {"model": M, "prompt": hay})["tokens"]
needle = post("/tokenize", {"model": M, "prompt": "\n# NOTE: the secret launch code for project Lighthouse is 7429-ORCHID.\n", "add_special_tokens": False})["tokens"]
q = post("/tokenize", {"model": M, "prompt": "\n\nQuestion: according to the NOTE comment above, what is the secret launch code for project Lighthouse? Answer: The code is", "add_special_tokens": False})["tokens"]
body_len = N - len(needle) - len(q)
assert len(ids) >= body_len, (len(ids), body_len)
pos = int(body_len * depth)
prompt = ids[:pos] + needle + ids[pos:body_len] + q
t = time.time(); r = post("/v1/completions", {"model": M, "prompt": prompt, "max_tokens": 12, "temperature": 0}); dt = time.time() - t
ans = r["choices"][0]["text"]
print(json.dumps({"prompt_tokens": r["usage"]["prompt_tokens"], "needle_at": pos, "seconds": round(dt, 1),
                  "prefill_tok_s": round(r["usage"]["prompt_tokens"] / dt), "answer": ans, "found": "7429-ORCHID" in ans}))
