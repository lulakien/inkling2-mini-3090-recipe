#!/usr/bin/env python3
"""Agent-style long-context turns (the shape of an OMP / coding-agent session) with planted facts.

The history is a chat transcript: a ~3K-token agent system prompt, then assistant tool calls and tool results that
read real Python files (tests/data/haystack, see tools/fetch_haystack.sh) until the prompt reaches --ctx tokens.
K facts ("service <name> listens on port <n>") are planted in tool results at evenly spread depths.

Turns (all temperature 0, streamed; tools stay in the prompt but tool_choice is "none", so each turn must answer):
  cold     first request at full length: one fact question                       (cold TTFT, no prefix hit)
  warm     previous turn + its answer + a new ~2K-token tool result + new question (warm TTFT, cached tokens)
  multi    warm again, asks for all K facts at once                                (multi-needle score K/K)
  compact  system prompt + a ~4K-token summary + new question (what an agent sends right after compaction:
           only the system prompt is still cached)
Per turn: TTFT, cached prompt tokens, completion tokens, decode tok/s (after first token), wall, success.
Peak VRAM per GPU is sampled with nvidia-smi during the run.
  agent_turns.py OUT.json --ctx 65536 [--port 8000] [--needles 8] [--effort none] [--max-tokens 4096] [--seed 0]
"""
import glob, json, os, random, subprocess, sys, threading, time, urllib.request

a = sys.argv
def arg(k, d): return type(d)(a[a.index(k) + 1]) if k in a else d
OUT = a[1]; CTX = arg("--ctx", 65536); PORT = arg("--port", 8000); K = arg("--needles", 8)
EFFORT = arg("--effort", "none"); MAXT = arg("--max-tokens", 4096); SEED = arg("--seed", 0)
B = f"http://127.0.0.1:{PORT}"; M = os.environ.get("INKLING_MODEL", "inkling-2-mini")
rng = random.Random(SEED * 1000003 + CTX)  # different transcript per size, so runs never share a prefix


def post(p, b, timeout=7200):
    r = urllib.request.Request(B + p, json.dumps(b).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def ntok(text):
    return len(post("/tokenize", {"model": M, "prompt": text, "add_special_tokens": False})["tokens"])


# ---- peak VRAM sampler
peak = {}
stop = threading.Event()
def sampler():
    while not stop.is_set():
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10).stdout
            for line in o.strip().splitlines():
                i, m = (x.strip() for x in line.split(","))
                peak[i] = max(peak.get(i, 0), int(m))
        except Exception:
            pass
        stop.wait(0.5)
threading.Thread(target=sampler, daemon=True).start()

# ---- build the transcript
TOOLS = "\n".join(
    f"- {n}: {d}" for n, d in [
        ("read", "Read a file. Args: path (str), offset (int, optional), limit (int, optional). Returns the file text."),
        ("bash", "Run a shell command in the workspace. Args: command (str), timeout (int seconds). Returns stdout+stderr."),
        ("edit", "Replace an exact string in a file. Args: path, old_string, new_string. Fails if not unique."),
        ("grep", "Search files with a regex. Args: pattern, path, glob. Returns matching lines with line numbers."),
        ("write", "Write a whole file. Args: path, content."),
        ("todo", "Update the task list. Args: items (list of {content, status})."),
    ])
SYS_RULES = " ".join([
    "You are a coding agent working in the user's repository. Prefer reading code over guessing.",
    "Use the tools to inspect files, run commands, and make edits. Keep answers short and exact.",
    "When asked about configuration values, quote them exactly as they appear in the files you read.",
] * 12)
SYSTEM = f"# Session {rng.randrange(10**9)}\n# Agent instructions\n{SYS_RULES}\n\n# Tools\n{TOOLS}\n\n# Environment\nworkspace: /repo\nshell: bash\n" * 3

hay_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "haystack")
files = sorted(glob.glob(os.path.join(hay_dir, "**", "*.py"), recursive=True))
assert files, "run tools/fetch_haystack.sh first"
rng.shuffle(files)
names = rng.sample(["orchid", "lantern", "basalt", "meridian", "quill", "saffron", "tundra", "cobalt", "juniper", "falcon",
                    "harbor", "ember", "glacier", "nimbus", "obsidian", "willow"], K)
ports = rng.sample(range(10000, 60000), K)
facts = [f"# DEPLOY NOTE: service {n} listens on port {p}.\n" for n, p in zip(names, ports)]

# budget in characters first (code ~3.3 chars/token), then trim by tokens
sys_tok = ntok(SYSTEM)
budget = CTX - sys_tok - 1200          # room for the question, template tokens and the warm-turn growth
msgs = [{"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Audit the model sources under /repo/models: read the files and report anything odd."}]
results_tok = []
fi = 0
used = 0
while used < budget:
    f = files[fi % len(files)]; fi += 1
    text = open(f).read()[:24000]
    if not text.strip():
        continue
    rel = os.path.relpath(f, hay_dir)
    t = ntok(text)
    if used + t > budget:
        text = text[: max(0, int((budget - used) * 3.0))]
        t = ntok(text)
        if t <= 0:
            break
    cid = f"call_{fi}"
    msgs.append({"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": "read", "arguments": json.dumps({"path": f"/repo/models/{rel}"})}}]})
    msgs.append({"role": "tool", "tool_call_id": cid, "content": text})
    results_tok.append((len(msgs) - 1, t))
    used += t + 30
# plant facts at evenly spread tool results (depth 3% .. 95%)
tool_idx = [i for i, _ in results_tok]
for j, fact in enumerate(facts):
    d = 0.03 + 0.92 * j / max(1, K - 1)
    i = tool_idx[min(len(tool_idx) - 1, int(d * len(tool_idx)))]
    body = msgs[i]["content"]
    cut = body.find("\n", len(body) // 2)
    msgs[i]["content"] = body[:cut + 1] + fact + body[cut + 1:]

if "--dump-messages" in a:  # used by second_stream.py
    stop.set(); print(json.dumps(msgs)); sys.exit(0)

tools = [{"type": "function", "function": {"name": n, "description": "see system prompt",
          "parameters": {"type": "object", "properties": {}, "additionalProperties": True}}}
         for n in ["read", "bash", "edit", "grep", "write", "todo"]]


NOTOOLS = " Answer directly from what you have already read in this conversation; do not call any tool."


def turn(label, messages, expect):
    messages = messages[:-1] + [{"role": "user", "content": messages[-1]["content"] + NOTOOLS}]
    body = {"model": M, "messages": messages, "tools": tools, "temperature": 0, "max_tokens": MAXT, "stream": True,
            "stream_options": {"include_usage": True}, "reasoning_effort": EFFORT, "tool_choice": "none"}
    req = urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = None; text = ""; reason = ""; usage = {}; calls = ""
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                dl = ch.get("delta", {})
                for tc in dl.get("tool_calls") or []:
                    calls += json.dumps(tc.get("function", {}))
                piece = (dl.get("content") or "") + (dl.get("reasoning_content") or dl.get("reasoning") or "") + calls
                if piece and first is None:
                    first = time.time()
                text += dl.get("content") or ""
                reason += dl.get("reasoning_content") or dl.get("reasoning") or ""
    t1 = time.time()
    ct = usage.get("completion_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    ok = [e for e in expect if e in text]
    ok_r = [e for e in expect if e in text or e in reason]
    res = {"turn": label, "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": cached,
           "ttft_s": round((first or t1) - t0, 2), "completion_tokens": ct,
           "decode_tok_s": round((ct - 1) / (t1 - first), 1) if first and ct > 1 and t1 > first else None,
           "wall_s": round(t1 - t0, 2), "found": f"{len(ok)}/{len(expect)}", "success": len(ok) == len(expect), "found_incl_reasoning": f"{len(ok_r)}/{len(expect)}",
           "answer": text[-400:], "reasoning_tail": reason[-300:], "tool_calls": calls[:300]}
    print(json.dumps({k: v for k, v in res.items() if k not in ("answer", "reasoning_tail", "tool_calls")}), flush=True)
    return res, text


out = {"ctx_target": CTX, "needles": K, "effort": EFFORT, "seed": SEED, "model": M, "turns": []}
T0 = time.time()
q1 = f"Question: according to the DEPLOY NOTE comments in the files you read, which port does service {names[0]} listen on? Reply with just the port number."
m1 = msgs + [{"role": "user", "content": q1}]
r, ans = turn("cold", m1, [str(ports[0])]); out["turns"].append(r)

extra = open(files[(fi + 7) % len(files)]).read()[:7000]
m2 = m1 + [{"role": "assistant", "content": ans},
           {"role": "assistant", "content": "", "tool_calls": [{"id": "call_w", "type": "function", "function": {
               "name": "read", "arguments": json.dumps({"path": "/repo/models/extra.py"})}}]},
           {"role": "tool", "tool_call_id": "call_w", "content": extra},
           {"role": "user", "content": f"Question: which port does service {names[K // 2]} listen on? Reply with just the port number."}]
r, ans2 = turn("warm", m2, [str(ports[K // 2])]); out["turns"].append(r)

m3 = m2 + [{"role": "assistant", "content": ans2},
           {"role": "user", "content": "List every service named in a DEPLOY NOTE comment with its port, one per line as name: port."}]
r, _ = turn("multi", m3, [str(p) for p in ports]); out["turns"].append(r)

summary_facts = "".join(f"- service {n} listens on port {p}\n" for n, p in zip(names, ports))
summary = ("## Summary of the session so far\nThe user asked for an audit of /repo/models. I read most model files. "
           "Findings worth keeping:\n" + summary_facts + "\n" + ("Notes: " + SYS_RULES[:3000] + "\n") * 4)
m4 = [msgs[0], {"role": "user", "content": summary},
      {"role": "assistant", "content": "Understood. Continuing from the summary."},
      {"role": "user", "content": f"Question: which port does service {names[-1]} listen on? Reply with just the port number."}]
r, _ = turn("compact", m4, [str(ports[-1])]); out["turns"].append(r)

stop.set()
out["total_wall_s"] = round(time.time() - T0, 1)
out["peak_vram_mib"] = peak
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({"total_wall_s": out["total_wall_s"], "peak_vram_mib": peak}))
