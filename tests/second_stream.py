#!/usr/bin/env python3
"""What a second agent costs the first one. Agent A has a long warm context (agent_turns-style transcript of
--ctx-a tokens, prefilled first and cached) and streams a long answer; agent B joins after A has produced
--join-after tokens with its own cold transcript of --ctx-b tokens.

Reports for A: tok/s before B joins, during B's prefill (until B's first token), after (both decoding), and the
longest gap between two A chunks. For B: TTFT while A runs vs B alone (measured first, same prompt length,
different seed so nothing is cached), and B's decode tok/s.
  second_stream.py OUT.json [--ctx-a 200000] [--ctx-b 65536] [--join-after 50] [--port 8000]
Needs KV for ctx-a + ctx-b (+ both answers) at once."""
import json, os, subprocess, sys, threading, time, urllib.request

a = sys.argv
def arg(k, d): return type(d)(a[a.index(k) + 1]) if k in a else d
OUT = a[1]; CA = arg("--ctx-a", 200000); CB = arg("--ctx-b", 65536); JOIN = arg("--join-after", 50); PORT = arg("--port", 8000)
B = f"http://127.0.0.1:{PORT}"; M = os.environ.get("INKLING_MODEL", "inkling-2-mini")
HERE = os.path.dirname(os.path.abspath(__file__))
LONG_Q = ("Write a detailed design review of the code you have read: for each of ten different files, explain what it "
          "does, its main classes and functions, and one possible improvement. Be thorough; at least 1500 words.")


def build(ctx, seed):
    """Reuse agent_turns.py's transcript builder by running it in 'dump' mode."""
    p = subprocess.run([sys.executable, os.path.join(HERE, "agent_turns.py"), "/dev/null", "--ctx", str(ctx), "--port", str(PORT),
                        "--seed", str(seed), "--dump-messages"], capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1])


def stream(messages, max_tokens, log, ignore_eos=False):
    body = {"model": M, "messages": messages + [{"role": "user", "content": LONG_Q}], "temperature": 0,
            "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}, "reasoning_effort": "none"}
    if ignore_eos:
        body["ignore_eos"] = True
    req = urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    log["t0"] = time.time(); log["chunks"] = []
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                log["usage"] = ev["usage"]
            for ch in ev.get("choices", []):
                dl = ch.get("delta", {})
                piece = (dl.get("content") or "") + (dl.get("reasoning_content") or dl.get("reasoning") or "")
                if piece:
                    log["chunks"].append(time.time())
    log["t1"] = time.time()


def rate(ts, lo, hi):
    """chunks per second between two times (DSpark emits several tokens per chunk; reported as chunks and
    scaled to tokens with the run's tokens/chunk ratio)."""
    n = sum(1 for t in ts if lo <= t < hi)
    return n / (hi - lo) if hi > lo else None


ma = build(CA, 11); mb = build(CB, 22); mb_alone = build(CB, 33)
res = {"ctx_a": CA, "ctx_b": CB, "join_after_chunks": JOIN}
# 1) prefill A once so its context is warm (what a running agent has)
la = {}; stream(ma, 1, la); res["a_cold_prefill_s"] = round(la["t1"] - la["t0"], 1)
# 2) B alone (cold) for reference
lb0 = {}; stream(mb_alone, 64, lb0); res["b_alone_ttft_s"] = round(lb0["chunks"][0] - lb0["t0"], 2)
# 3) A streams; B joins after JOIN chunks
la = {}; lb = {}
ta = threading.Thread(target=stream, args=(ma, 8000, la, True)); ta.start()
while len(la.get("chunks", [])) < JOIN:
    time.sleep(0.01)
tb = threading.Thread(target=stream, args=(mb, 800, lb, True)); tb.start()
tb.join(); ta.join()
A = la["chunks"]; Bc = lb["chunks"]
a_tpc = la["usage"]["completion_tokens"] / max(1, len(A)); b_tpc = lb["usage"]["completion_tokens"] / max(1, len(Bc))
join_t = lb["t0"]; b_first = Bc[0]; b_end = lb["t1"]
gaps = [y - x for x, y in zip(A, A[1:])]
res.update({
    "a_warm_ttft_s": round(A[0] - la["t0"], 2),
    "a_tok_s_before_join": round(rate(A, A[0], join_t) * a_tpc, 1),
    "a_tok_s_during_b_prefill": round((rate(A, join_t, b_first) or 0) * a_tpc, 1),
    "a_tok_s_both_decoding": round((rate(A, b_first, min(b_end, la["t1"])) or 0) * a_tpc, 1),
    "a_tok_s_after_b_done": round((rate(A, b_end, la["t1"]) or 0) * a_tpc, 1) if la["t1"] > b_end + 1 else None,
    "a_longest_gap_s": round(max(gaps), 2), "a_longest_gap_at_s_after_join": round(A[gaps.index(max(gaps))] - join_t, 2),
    "b_ttft_with_a_running_s": round(b_first - join_t, 2),
    "b_tok_s": round(lb["usage"]["completion_tokens"] / (b_end - b_first), 1),
    "a_completion_tokens": la["usage"]["completion_tokens"], "b_completion_tokens": lb["usage"]["completion_tokens"],
    "a_prompt_tokens": la["usage"]["prompt_tokens"], "a_cached": (la["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"),
    "b_prompt_tokens": lb["usage"]["prompt_tokens"],
})
print(json.dumps(res))
json.dump(res, open(OUT, "w"), indent=1)
