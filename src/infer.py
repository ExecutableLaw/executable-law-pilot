"""vLLM inference + exact-match scoring.

python infer.py --model PATH_OR_ID --data test.jsonl --out res.jsonl [--shots shots.jsonl] [--n 1 --temp 0]
Writes per-item results and prints a JSON summary line (also written to <out>.summary.json).
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from prompts import messages
from scoring import score


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def ece(confs, zs, bins=10):
    if not confs:
        return None
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if idx:
            tot += len(idx) / len(confs) * abs(sum(zs[i] for i in idx) / len(idx) - sum(confs[i] for i in idx) / len(idx))
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shots", default=None)
    ap.add_argument("--nshots", type=int, default=0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--abstain", action="store_true")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--lora", default=None, help="adapter dir applied at runtime on top of --model")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    data = [json.loads(l) for l in open(a.data)]
    if a.limit:
        data = data[:a.limit]
    shots = []
    if a.shots and a.nshots:
        pool = [json.loads(l) for l in open(a.shots)]
        random.Random(0).shuffle(pool)
        shots = pool[:a.nshots]
    kw = dict(enable_lora=True, max_lora_rank=64) if a.lora else {}
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len,
              enable_prefix_caching=True, seed=0, **kw)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(messages(ex, shots, a.abstain), tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for ex in data]
    sp = SamplingParams(n=a.n, temperature=a.temp, top_p=1.0, max_tokens=a.max_tokens, seed=0)
    t0 = time.time()
    if a.lora:
        from vllm.lora.request import LoRARequest
        outs = llm.generate(prompts, sp, lora_request=LoRARequest("a", 1, a.lora))
    else:
        outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    n_in = sum(len(o.prompt_token_ids) for o in outs)
    n_out = sum(len(c.token_ids) for o in outs for c in o.outputs)
    rows, zs, valid, confs, czs, abst = [], [], 0, [], [], 0
    for ex, o in zip(data, outs):
        ss = [score(c.text, ex) for c in o.outputs]
        rows.append({"id": ex["id"], "z": [s["z"] for s in ss], "valid": [s["valid"] for s in ss],
                     "abstain": [s["abstain"] for s in ss], "conf": [s["conf"] for s in ss],
                     "text": [c.text for c in o.outputs][:2]})
        s0 = ss[0]
        zs.append(s0["z"]); valid += s0["valid"]; abst += s0["abstain"]
        if s0["conf"] is not None and not s0["abstain"]:  # invalid outputs count as wrong
            confs.append(s0["conf"]); czs.append(s0["z"])
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    k, n = sum(zs), len(zs)
    by_year = {}
    for ex, z in zip(data, zs):
        y = ex["date"][:4]; by_year.setdefault(y, []).append(z)
    summ = {"model": a.model, "lora": a.lora, "em_by_year": {y: (sum(v) / len(v), len(v)) for y, v in sorted(by_year.items())}, "data": os.path.basename(a.data), "nshots": a.nshots, "n_items": n,
            "em": k / max(n, 1), "em_ci95": wilson(k, n), "valid_rate": valid / max(n, 1), "abstain_rate": abst / max(n, 1),
            "mean_conf": (sum(confs) / len(confs)) if confs else None, "ece": ece(confs, czs),
            "pass_any@n": sum(any(r["z"]) for r in rows) / max(n, 1) if a.n > 1 else None,
            "gen_seconds": dt, "prompt_tokens": n_in, "output_tokens": n_out,
            "prefill_tok_per_s": n_in / dt, "decode_tok_per_s": n_out / dt}
    json.dump(summ, open(a.out + ".summary.json", "w"), indent=1)
    print("SUMMARY", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
