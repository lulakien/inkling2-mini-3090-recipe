#!/usr/bin/env python3
"""Markdown tables from experiments/long-context-2gpu/results/*: KV pool, decode, agent turns, MMLU, second stream."""
import glob, json, os, re

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def pool(d):
    try:
        t = open(os.path.join(d, "server_log_excerpt.txt")).read()
        m = re.search(r"GPU KV cache size: ([\d,]+) tokens", t)
        return m.group(1) if m else ""
    except OSError:
        return ""


def dec(d):
    j = load(os.path.join(d, "decode.json")) or {}
    v = [j.get(k, {}).get("tok_s") for k in ("code_t0.0", "explain_t0.0", "story_t0.0")]
    return " / ".join(str(round(x)) for x in v) if all(v) else ""


def turns(d, name):
    j = load(os.path.join(d, name))
    if not j:
        return {}
    return {t["turn"]: t for t in j["turns"]}


rows = []
for d in sorted(glob.glob(os.path.join(R, "*"))):
    lab = os.path.basename(d)
    t64, t250 = turns(d, "agent_64k.json"), turns(d, "agent_250k.json")
    m = load(os.path.join(d, "mmlu.json"))
    def f(t, k):
        x = t.get(k)
        return "" if not x else (x["found"] + ("" if x["success"] else " *"))
    rows.append([lab, pool(d), dec(d),
                 str(t64.get("cold", {}).get("ttft_s", "")), str(t250.get("cold", {}).get("ttft_s", "")),
                 str(t250.get("warm", {}).get("ttft_s", "")),
                 " ".join(f(t, k) for t in (t64, t250) for k in ("cold", "warm", "multi", "compact") if t.get(k)),
                 str(m["summary"]["correct"]) if m else ""])
hdr = ["config", "KV pool", "decode code/explain/story", "64K cold TTFT s", "250K cold TTFT s", "250K warm TTFT s",
       "agent turns found (64K c/w/m/x, 250K c/w/m/x; * = no final answer)", "MMLU /1140"]
print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
for r in rows:
    print("| " + " | ".join(r) + " |")
print()
for d in sorted(glob.glob(os.path.join(R, "*"))):
    s = load(os.path.join(d, "second_stream.json"))
    if s:
        print(f"- second stream, {os.path.basename(d)}: A alone {s['a_tok_s_before_join']} tok/s, during B prefill "
              f"{s['a_tok_s_during_b_prefill']}, both decoding {s['a_tok_s_both_decoding']} (B {s['b_tok_s']}); "
              f"B TTFT {s['b_ttft_with_a_running_s']} s vs {s['b_alone_ttft_s']} s alone; A longest gap {s['a_longest_gap_s']} s")
