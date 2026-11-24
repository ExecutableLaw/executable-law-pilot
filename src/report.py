"""Aggregate pilot results. python report.py --res /data/res --splits /data/splits --models /data/models"""
import argparse
import glob
import json
import os
import random


def load_rows(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)}


def paired(a, b, iters=10000):
    ids = sorted(set(a) & set(b))
    za = [a[i]["z"][0] for i in ids]
    zb = [b[i]["z"][0] for i in ids]
    d = sum(zb) / len(ids) - sum(za) / len(ids)
    rng = random.Random(0)
    bs = []
    for _ in range(iters):
        idx = [rng.randrange(len(ids)) for _ in ids]
        bs.append(sum(zb[i] - za[i] for i in idx) / len(ids))
    bs.sort()
    b01 = sum(1 for x, y in zip(za, zb) if x == 0 and y == 1)
    b10 = sum(1 for x, y in zip(za, zb) if x == 1 and y == 0)
    return {"n": len(ids), "delta": d, "ci95": (bs[int(0.025 * iters)], bs[int(0.975 * iters)]), "wins": b01, "losses": b10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--funnel", default=None)
    a = ap.parse_args()
    out = {"summaries": {}, "paired": {}}
    for p in sorted(glob.glob(os.path.join(a.res, "*.summary.json"))):
        out["summaries"][os.path.basename(p).replace(".jsonl.summary.json", "")] = json.load(open(p))
    rows = {os.path.basename(p).replace(".jsonl", ""): load_rows(p) for p in glob.glob(os.path.join(a.res, "test_*.jsonl"))}
    pairs = [("test_base1.7b_3shot", "test_sft"), ("test_base8b_3shot", "test_sft"), ("test_base14b_3shot", "test_sft"),
             ("test_sft", "test_grpo_R1"), ("test_sft_abstainprompt", "test_grpo_R3"), ("test_sft", "test_grpo_R3"),
             ("test_grpo_R1", "test_grpo_R3"), ("test_sft_all", "test_grpo_R1"), ("test_sft", "test_sft_all")]
    for x, y in pairs:
        if x in rows and y in rows:
            out["paired"][f"{y} - {x}"] = paired(rows[x], rows[y])
    out["split_stats"] = json.load(open(os.path.join(a.splits, "split_stats.json")))
    if a.funnel and os.path.exists(a.funnel):
        out["funnel"] = json.load(open(a.funnel))
    for name in ("sft", "sft_all", "grpo_R1", "grpo_R3"):
        for f in ("sft_stats.json", "grpo_stats.json"):
            p = os.path.join(a.models, name, f)
            if os.path.exists(p):
                out[f"train_{name}"] = json.load(open(p))
    json.dump(out, open(os.path.join(a.res, "REPORT.json"), "w"), indent=1)
    print(f"{'run':32s} {'EM':>6s} {'95% CI':>15s} {'valid':>6s} {'ECE':>6s} {'abst':>5s}")
    for k, s in out["summaries"].items():
        ci = s["em_ci95"]
        print(f"{k:32s} {s['em']*100:6.1f} [{ci[0]*100:5.1f},{ci[1]*100:5.1f}] {s['valid_rate']*100:6.1f} "
              f"{'' if s['ece'] is None else f'{s['ece']:.3f}':>6s} {s['abstain_rate']*100:5.1f}")
    for k, v in out["paired"].items():
        print(f"{k:40s} delta {v['delta']*100:+.1f} pts CI [{v['ci95'][0]*100:+.1f},{v['ci95'][1]*100:+.1f}] wins {v['wins']} losses {v['losses']}")


if __name__ == "__main__":
    main()
