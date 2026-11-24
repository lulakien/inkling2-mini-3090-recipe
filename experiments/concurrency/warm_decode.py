#!/usr/bin/env python3
"""Decode on warm agent turns: an agent transcript of --ctx tokens (agent_turns.py, prefilled once so later turns hit
the prefix cache), then reasoning and coding questions at temp 0. Per turn: decode tok/s (first streamed token to
last), tokens per DSpark step, ms per step (from /metrics deltas), reasoning and answer lengths.
  warm_decode.py OUT.json [--ctx 65536] [--port 8000] [--effort medium]"""
import json, os, re, subprocess, sys, time, urllib.request

a = sys.argv
def arg(k, d): return type(d)(a[a.index(k) + 1]) if k in a else d
OUT = a[1]; CTX = arg("--ctx", 65536); PORT = arg("--port", 8000); EFFORT = arg("--effort", "medium")
B = f"http://127.0.0.1:{PORT}"; M = os.environ.get("INKLING_MODEL", "inkling-2-mini")
HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, "..", "..", "tests")


def metrics():
    txt = urllib.request.urlopen(B + "/metrics").read().decode(); out = {}
    for k in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        m = re.findall(rf"^vllm:spec_decode_{k}_total\{{[^}}]*\}} ([0-9.e+]+)", txt, re.M)
        out[k] = sum(float(x) for x in m)
    return out


TOOLS = [{"type": "function", "function": {"name": n, "description": "see system prompt",
          "parameters": {"type": "object", "properties": {}, "additionalProperties": True}}}
         for n in ["read", "bash", "edit", "grep", "write", "todo"]]
NOTOOLS = " Answer directly; do not call any tool."


def chat(messages, max_tokens):
    body = {"model": M, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "reasoning_effort": EFFORT, "tools": TOOLS, "tool_choice": "none"}
    req = urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = None; text = reason = ""; usage = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                dl = ch.get("delta", {})
                piece = (dl.get("content") or "") + (dl.get("reasoning_content") or dl.get("reasoning") or "")
                if piece and first is None:
                    first = time.time()
                text += dl.get("content") or ""; reason += dl.get("reasoning_content") or dl.get("reasoning") or ""
    return t0, first, time.time(), usage, text, reason


msgs = json.loads(subprocess.run([sys.executable, os.path.join(TESTS, "agent_turns.py"), "/dev/null", "--ctx", str(CTX),
                                  "--port", str(PORT), "--seed", "7", "--dump-messages"],
                                 capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
chat(msgs + [{"role": "user", "content": "Reply with OK."}], 8)  # prefill once; the turns below hit the cache
QUESTIONS = {
    "reason": "Based on the files you read, explain step by step how a request flows from the tokenizer through the "
              "model's forward pass to sampling, and where the code you saw could become a bottleneck.",
    "code": "Write a complete Python module that implements a thread-safe LRU cache with TTL expiry, type hints, "
            "docstrings and pytest tests. Output only the code.",
    "edit": "Pick one function from the files you read and rewrite it to be faster. Show the full new function and "
            "a short explanation of the change.",
}
res = {"ctx": CTX, "effort": EFFORT}
for name, q in QUESTIONS.items():
    m0 = metrics()
    t0, first, t1, usage, text, reason = chat(msgs + [{"role": "user", "content": q + NOTOOLS}], 2500)
    m1 = metrics(); d = {k: m1[k] - m0[k] for k in m0}; n = usage.get("completion_tokens", 0)
    dec = t1 - (first or t0)
    row = {"tokens": n, "ttft_s": round((first or t1) - t0, 2), "decode_tok_s": round((n - 1) / dec, 1) if dec > 0 else None,
           "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
           "reasoning_chars": len(reason), "answer_chars": len(text)}
    if d["num_drafts"]:
        row["tokens_per_step"] = round(1 + d["num_accepted_tokens"] / d["num_drafts"], 2)
        row["ms_per_step"] = round(dec * 1000 / d["num_drafts"], 2)
    res[name] = row; print(name, json.dumps(row), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
