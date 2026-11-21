#!/usr/bin/env python3
"""MMLU letter-logit accuracy against the local Inkling server (model card harness: 57 subjects x 20, letter logits).
  mmlu_letters.py OUT.json [--port 8000] [--n 1140] [--workers 8]
Each question: chat template at effort 0 plus a prefilled closed thinking block and answer header\n(token ids), max_tokens 1, top-20 logprobs; the prediction is the most likely
of A/B/C/D among those (a question whose top 20 contains no letter counts as wrong and is reported as 'no_letter')."""
import argparse, json, math, sys, time, urllib.request, os
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--n", type=int, default=1140)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mmlu_57x20.jsonl"))
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.data)][: a.n]
L = "ABCD"


def prompt(r):
    subj = r["subject"].replace("_", " ")
    opts = "\n".join(f"{L[i]}. {c}" for i, c in enumerate(r["choices"]))
    return (f"The following is a multiple choice question about {subj}.\n\n{r['question']}\n{opts}\n\n"
            "Answer with the letter of the correct option only.")


SUFFIX = None  # token ids: closed brief thinking block + answer header (template's own special tokens)


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def ask(r):
    global SUFFIX
    M = os.environ.get("INKLING_MODEL", "inkling-2-mini")
    for attempt in range(3):
        try:
            if SUFFIX is None:
                SUFFIX = post("/tokenize", {"model": M, "add_special_tokens": False, "prompt":
                              "<|message_model|><|content_thinking|>Answer directly.<|end_message|>"
                              "<|message_model|><|content_text|>"})["tokens"]
            head = post("/tokenize", {"model": M, "messages": [{"role": "user", "content": prompt(r)}],
                                      "add_generation_prompt": True,
                                      "chat_template_kwargs": {"reasoning_effort": "none"}})["tokens"]
            res = post("/v1/completions", {"model": M, "prompt": head + SUFFIX, "max_tokens": 1,
                                           "temperature": 0, "logprobs": 20})
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                return {"err": str(e)}
            time.sleep(2)
    lp = res["choices"][0]["logprobs"]["top_logprobs"][0]
    best = {}
    for tok, v in lp.items():
        k = tok.strip()
        if k in L and (k not in best or v > best[k]):
            best[k] = v
    pred = max(best, key=best.get) if best else None
    return {"pred": pred, "p": {k: math.exp(v) for k, v in best.items()}, "top": max(lp, key=lp.get)}


t0 = time.time()
with ThreadPoolExecutor(a.workers) as ex:
    res = list(ex.map(ask, rows))
ok = sum(1 for r, o in zip(rows, res) if o.get("pred") == L[r["answer"]])
errs = sum(1 for o in res if "err" in o)
nolet = sum(1 for o in res if "err" not in o and o["pred"] is None)
subj = {}
for r, o in zip(rows, res):
    s = subj.setdefault(r["subject"], [0, 0])
    s[0] += o.get("pred") == L[r["answer"]]
    s[1] += 1
summary = {"accuracy": ok / len(rows), "correct": ok, "n": len(rows), "errors": errs, "no_letter": nolet,
           "seconds": round(time.time() - t0, 1)}
json.dump({"summary": summary, "per_subject": subj, "results": res}, open(a.out, "w"))
print(json.dumps(summary))
