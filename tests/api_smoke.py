#!/usr/bin/env python3
"""OpenAI-API feature smoke test: the things an agent/client depends on, each PASS/FAIL with evidence.
  api_smoke.py OUT.json [--port 8000] [--image PATH.png]   (--image needs a server started with MM=1)
Note: Inkling 2 always reasons a little, even at reasoning_effort "none" (model card), so every check leaves room
for a short reasoning block before the answer and reads `content` and `reasoning` separately.
Checks: chat, streaming (incl. usage), reasoning split (reasoning_effort), tool call parsing + tool-result turn,
JSON-schema structured output, /v1/completions with logprobs, prefix-cache hit on a repeated long prefix,
temp-0 determinism, and stop sequences."""
import argparse, base64, json, os, sys, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--image"); a = ap.parse_args()
B = f"http://127.0.0.1:{a.port}"; M = os.environ.get("INKLING_MODEL", "inkling-2-mini")
def post(path, body, stream=False):
    r = urllib.request.Request(B + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    f = urllib.request.urlopen(r, timeout=600)
    return f if stream else json.load(f)
def chat(msgs, **kw):
    return post("/v1/chat/completions", {"model": M, "messages": msgs, **kw})
res = {}
def check(name, fn):
    t = time.time()
    try:
        ok, ev = fn()
    except Exception as e:  # noqa: BLE001
        ok, ev = False, f"{type(e).__name__}: {e}"
    res[name] = {"pass": bool(ok), "evidence": ev, "seconds": round(time.time() - t, 2)}
    print(("PASS " if ok else "FAIL ") + name, "|", str(ev)[:160], flush=True)

def t_chat():
    r = chat([{"role": "user", "content": "What is the capital of France? One word."}], max_tokens=512, temperature=0,
             chat_template_kwargs={"reasoning_effort": "none"})
    c = r["choices"][0]["message"]["content"] or ""
    return "paris" in c.lower(), c.strip()
def t_stream():
    f = post("/v1/chat/completions", {"model": M, "stream": True, "stream_options": {"include_usage": True},
             "messages": [{"role": "user", "content": "Count from 1 to 20, comma separated."}], "max_tokens": 512,
             "temperature": 0, "chat_template_kwargs": {"reasoning_effort": "none"}}, stream=True)
    chunks, text, reasoning, usage = 0, "", "", None
    for line in f:
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:]); chunks += 1
        if d.get("usage"): usage = d["usage"]
        for ch in d.get("choices", []):
            text += ch.get("delta", {}).get("content") or ""
            reasoning += ch.get("delta", {}).get("reasoning") or ch.get("delta", {}).get("reasoning_content") or ""
    return chunks > 3 and "20" in text and usage is not None, \
        {"chunks": chunks, "reasoning_chars": len(reasoning), "content": text.strip()[:80], "completion_tokens": (usage or {}).get("completion_tokens")}
def t_reasoning():
    r = chat([{"role": "user", "content": "A train leaves at 14:35 and arrives at 17:10. How long is the trip?"}],
             max_tokens=2048, temperature=0, chat_template_kwargs={"reasoning_effort": "high"})
    m = r["choices"][0]["message"]; rc = m.get("reasoning_content") or m.get("reasoning") or ""
    c = m.get("content") or ""
    return len(rc) > 20 and ("2 h" in c or "2h" in c or "35" in c or "two" in c.lower()), \
        {"reasoning_chars": len(rc), "content": c.strip()[:120]}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                         "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}]
def t_tools():
    msgs = [{"role": "user", "content": "What's the weather in Istanbul right now, in celsius?"}]
    r = chat(msgs, tools=TOOLS, tool_choice="auto", max_tokens=512, temperature=0)
    m = r["choices"][0]["message"]; tc = m.get("tool_calls") or []
    if not tc:
        return False, {"finish": r["choices"][0]["finish_reason"], "content": (m.get("content") or "")[:200]}
    args = json.loads(tc[0]["function"]["arguments"])
    ok1 = tc[0]["function"]["name"] == "get_weather" and "istanbul" in args.get("city", "").lower()
    msgs += [{"role": "assistant", "content": m.get("content") or "", "tool_calls": tc},
             {"role": "tool", "tool_call_id": tc[0]["id"], "content": json.dumps({"temp_c": 18, "sky": "cloudy"})}]
    r2 = chat(msgs, tools=TOOLS, max_tokens=256, temperature=0)
    c2 = r2["choices"][0]["message"].get("content") or ""
    return ok1 and "18" in c2, {"call": tc[0]["function"], "finish": r["choices"][0]["finish_reason"], "answer": c2.strip()[:120]}
def t_json_schema():
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "year": {"type": "integer"},
              "languages": {"type": "array", "items": {"type": "string"}}}, "required": ["name", "year", "languages"],
              "additionalProperties": False}
    r = chat([{"role": "user", "content": "Describe the Python programming language as JSON (name, first release year, 2 influencing languages)."}],
             max_tokens=1024, temperature=0, chat_template_kwargs={"reasoning_effort": "none"},
             response_format={"type": "json_schema", "json_schema": {"name": "lang", "schema": schema}})
    c = r["choices"][0]["message"]["content"]; d = json.loads(c)
    return set(d) == {"name", "year", "languages"} and isinstance(d["year"], int), d
def t_completions_logprobs():
    r = post("/v1/completions", {"model": M, "prompt": "The quick brown fox", "max_tokens": 8, "temperature": 0, "logprobs": 3})
    lp = r["choices"][0]["logprobs"]
    return len(lp["tokens"]) == 8 and len(lp["top_logprobs"][0]) >= 3, {"text": r["choices"][0]["text"], "lp0": lp["token_logprobs"][0]}
def t_prefix_cache():
    sysmsg = " ".join(f"rule{i}: always be precise about item {i}." for i in range(600))
    q = lambda s: chat([{"role": "system", "content": sysmsg}, {"role": "user", "content": s}], max_tokens=8,
                       temperature=0, chat_template_kwargs={"reasoning_effort": "none"})
    t = time.time(); r1 = q("Say OK."); t1 = time.time() - t
    t = time.time(); r2 = q("Say YES."); t2 = time.time() - t
    c = (r2["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    p = r2["usage"]["prompt_tokens"]
    return c >= p * 0.9, {"prompt_tokens": p, "cached_tokens": c, "cold_s": round(t1, 3), "warm_s": round(t2, 3)}
def t_determinism():
    body = dict(max_tokens=800, temperature=0, chat_template_kwargs={"reasoning_effort": "none"})
    msgs = [{"role": "user", "content": "Write a haiku about GPUs, then explain it in two sentences."}]
    o = [chat(msgs, **body)["choices"][0]["message"] for _ in range(3)]
    o = [((m.get("reasoning") or ""), (m.get("content") or "")) for m in o]
    return o[0] == o[1] == o[2] and o[0][1], {"identical_runs": sum(x == o[0] for x in o), "sample": o[0][1][:80]}
def t_stop():
    r = chat([{"role": "user", "content": "List the numbers 1 to 10, one per line."}], max_tokens=600, temperature=0,
             stop=["5"], chat_template_kwargs={"reasoning_effort": "none"})
    c = r["choices"][0]["message"]["content"] or ""
    return "5" not in c and "4" in c and r["choices"][0]["finish_reason"] == "stop", c.strip()
def t_image():
    b64 = base64.b64encode(open(a.image, "rb").read()).decode()
    r = chat([{"role": "user", "content": [{"type": "text", "text": "Which two colors does this image show? Answer in one short sentence."},
              {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}], max_tokens=600, temperature=0,
             chat_template_kwargs={"reasoning_effort": "none"})
    c = (r["choices"][0]["message"]["content"] or "").lower()
    return "red" in c and "blue" in c, c.strip()[:200]

for n, f in [("chat", t_chat), ("stream", t_stream), ("reasoning_split", t_reasoning), ("tool_call", t_tools),
             ("json_schema", t_json_schema), ("completions_logprobs", t_completions_logprobs),
             ("prefix_cache_hit", t_prefix_cache), ("temp0_determinism", t_determinism), ("stop_sequence", t_stop)]:
    check(n, f)
if a.image:
    check("image_input", t_image)
print(f"{sum(v['pass'] for v in res.values())}/{len(res)} passed")
json.dump(res, open(a.out, "w"), indent=1)
sys.exit(0 if all(v["pass"] for v in res.values()) else 1)
