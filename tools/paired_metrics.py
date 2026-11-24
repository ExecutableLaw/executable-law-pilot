"""Recompute per-run and paired metrics from per-item outputs (standard library only, Python 3.9+).

Written after the pilot run and not part of the executed code. It reads the per-item files that src/infer.py
writes and repeats the computations of src/infer.py (EM, Wilson CI, valid rate, ECE, EM by year) and
src/report.py (paired bootstrap with the same seed and resampling order), then adds checks the pilot report
does not have: a bootstrap that resamples whole Federal Register documents (items from one rule are not
independent), an exact McNemar test on discordant items, and paired deltas per year.

  # recompute the 47 CFR pilot and check every number against results/*.summary.json and results/REPORT.json
  python tools/paired_metrics.py --res results --data data/splits/test.jsonl --check-report results/REPORT.json

  # held-out title (same file names, separate directory); write the extended report as JSON
  python tools/paired_metrics.py --res results/title21 --data heldout/t21/heldout.jsonl --out /tmp/heldout-report.json

  # compare a rerun against the original per-item outputs (rebuild check)
  python tools/paired_metrics.py --compare results/test_sft.jsonl rerun/test_sft.jsonl
"""
import argparse
import glob
import json
import math
import os
import random
import sys

# the pairs src/report.py reports, in the same orientation (second minus first)
PAIRS = [("test_base1.7b_3shot", "test_sft"), ("test_base8b_3shot", "test_sft"), ("test_base14b_3shot", "test_sft"),
         ("test_sft", "test_grpo_R1"), ("test_sft_abstainprompt", "test_grpo_R3"), ("test_sft", "test_grpo_R3"),
         ("test_grpo_R1", "test_grpo_R3"), ("test_sft_all", "test_grpo_R1"), ("test_sft", "test_sft_all"),
         # not in report.py, added for the held-out title
         ("test_base14b_3shot", "test_grpo_R1"), ("test_base1.7b_0shot", "test_sft")]


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p))}


def wilson(k, n, z=1.96):  # as src/infer.py
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def ece(confs, zs, bins=10):  # as src/infer.py
    if not confs:
        return None
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if idx:
            tot += len(idx) / len(confs) * abs(sum(zs[i] for i in idx) / len(idx) - sum(confs[i] for i in idx) / len(idx))
    return tot


def summary(rows, data):
    """Per-run summary over the items of data, in data order (as src/infer.py)."""
    zs, valid, abst, confs, czs, by_year = [], 0, 0, [], [], {}
    for ex in data:
        r = rows[ex["id"]]
        z, v, a, c = r["z"][0], r["valid"][0], r["abstain"][0], r["conf"][0]
        zs.append(z); valid += v; abst += a
        if c is not None and not a:
            confs.append(c); czs.append(z)
        by_year.setdefault(ex["date"][:4], []).append(z)
    n, k = len(zs), sum(zs)
    return {"n_items": n, "em": k / max(n, 1), "em_ci95": list(wilson(k, n)), "valid_rate": valid / max(n, 1),
            "abstain_rate": abst / max(n, 1), "mean_conf": (sum(confs) / len(confs)) if confs else None,
            "ece": ece(confs, czs), "em_by_year": {y: [sum(v) / len(v), len(v)] for y, v in sorted(by_year.items())}}


def paired_item(a, b, iters=10000):
    """Identical to src/report.py paired()."""
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
    return {"n": len(ids), "delta": d, "ci95": [bs[int(0.025 * iters)], bs[int(0.975 * iters)]], "wins": b01, "losses": b10}


def paired_cluster(a, b, doc_of, iters=10000):
    """Bootstrap that resamples Federal Register documents, keeping all items of a resampled document."""
    ids = sorted(set(a) & set(b))
    groups = {}
    for i in ids:
        groups.setdefault(doc_of[i], []).append(a[i]["z"][0] - b[i]["z"][0])
    g = [(-sum(v), len(v)) for _, v in sorted(groups.items())]  # (sum of zb - za, size)
    rng = random.Random(0)
    bs = []
    for _ in range(iters):
        s = n = 0
        for _ in g:
            x = g[rng.randrange(len(g))]
            s += x[0]; n += x[1]
        bs.append(s / n)
    bs.sort()
    return {"n_docs": len(g), "ci95": [bs[int(0.025 * iters)], bs[int(0.975 * iters)]]}


def mcnemar_exact(w, l):
    """Two-sided exact binomial test on discordant pairs."""
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def run(res, data_path, check_report=None, out=None, tol=1e-9):
    data = [json.loads(l) for l in open(data_path)]
    ids = [e["id"] for e in data]
    assert len(ids) == len(set(ids)), "duplicate ids in data"
    doc_of = {e["id"]: e["doc"] for e in data}
    year_of = {e["id"]: e["date"][:4] for e in data}
    runs = {}
    for p in sorted(glob.glob(os.path.join(res, "test_*.jsonl"))):
        name = os.path.basename(p)[:-len(".jsonl")]
        rows = load(p)
        if not set(ids) <= set(rows):
            print(f"WARNING {name}: {len(set(ids) - set(rows))} items of {data_path} have no output; run skipped")
            continue
        if len(rows) > len(ids):  # outputs cover a superset (e.g. heldout_all); restrict to the items of --data
            print(f"note {name}: restricted to the {len(ids)} items of {data_path} (file has {len(rows)})")
            rows = {i: rows[i] for i in ids}
        runs[name] = rows
    problems = []
    out_j = {"data": data_path, "n_items": len(data), "n_docs": len(set(doc_of.values())), "summaries": {}, "paired": {}}
    print(f"{len(data)} items from {out_j['n_docs']} Federal Register documents; runs: {', '.join(runs)}")
    print(f"{'run':26s} {'EM':>6s} {'95% CI':>13s} {'valid':>6s} {'ECE':>6s} {'abst':>5s}  EM by year")
    for name, rows in runs.items():
        s = summary(rows, data)
        out_j["summaries"][name] = s
        yr = " ".join(f"{y}:{v[0]*100:.1f}(n={v[1]})" for y, v in s["em_by_year"].items())
        print(f"{name:26s} {s['em']*100:6.1f} {s['em_ci95'][0]*100:5.1f}-{s['em_ci95'][1]*100:5.1f} "
              f"{s['valid_rate']*100:6.1f} {'' if s['ece'] is None else format(s['ece'], '.3f'):>6s} "
              f"{s['abstain_rate']*100:5.1f}  {yr}")
        sp = os.path.join(res, name + ".jsonl.summary.json")
        if os.path.exists(sp) and json.load(open(sp)).get("n_items") == len(ids):  # compare with what infer.py wrote
            ref = json.load(open(sp))
            for k in ("n_items", "em", "valid_rate", "abstain_rate", "mean_conf", "ece"):
                x, y = s[k], ref.get(k)
                if (x is None) != (y is None) or (x is not None and abs(x - y) > tol):
                    problems.append(f"{name}.{k}: recomputed {x} vs summary {y}")
            for k in ("em_ci95",):
                if any(abs(u - v) > tol for u, v in zip(s[k], ref[k])):
                    problems.append(f"{name}.{k}: recomputed {s[k]} vs summary {ref[k]}")
            if {y: list(v) for y, v in ref.get("em_by_year", {}).items()} != s["em_by_year"]:
                problems.append(f"{name}.em_by_year differs")
    print()
    print(f"{'comparison (second minus first)':44s} {'delta':>6s} {'item-bootstrap CI':>17s} {'doc-bootstrap CI':>17s} "
          f"{'fix/brk':>7s} {'McNemar p':>9s}  delta by year")
    for x, y in PAIRS:
        if x not in runs or y not in runs:
            continue
        pi = paired_item(runs[x], runs[y])
        pc = paired_cluster(runs[x], runs[y], doc_of)
        byy = {}
        for yy in sorted(set(year_of.values())):
            sub = [i for i in ids if year_of[i] == yy]
            byy[yy] = sum(runs[y][i]["z"][0] - runs[x][i]["z"][0] for i in sub) / len(sub)
        p = mcnemar_exact(pi["wins"], pi["losses"])
        key = f"{y} - {x}"
        out_j["paired"][key] = dict(pi, ci95_doc_cluster=pc["ci95"], n_docs=pc["n_docs"], mcnemar_p=p, delta_by_year=byy)
        print(f"{key:44s} {pi['delta']*100:+6.1f} {pi['ci95'][0]*100:+7.1f}..{pi['ci95'][1]*100:+6.1f} "
              f"{pc['ci95'][0]*100:+7.1f}..{pc['ci95'][1]*100:+6.1f} {pi['wins']:3d}/{pi['losses']:<3d} {p:9.2g}  "
              + " ".join(f"{k}:{v*100:+.1f}" for k, v in byy.items()))
    if check_report:
        rep = json.load(open(check_report))
        for key, v in rep.get("paired", {}).items():
            mine = out_j["paired"].get(key)
            if mine is None:
                problems.append(f"paired {key} in report but not recomputed"); continue
            for k in ("n", "wins", "losses"):
                if mine[k] != v[k]:
                    problems.append(f"paired {key}.{k}: {mine[k]} vs report {v[k]}")
            if abs(mine["delta"] - v["delta"]) > tol or any(abs(u - w) > tol for u, w in zip(mine["ci95"], v["ci95"])):
                problems.append(f"paired {key}: delta/ci {mine['delta']} {mine['ci95']} vs report {v['delta']} {v['ci95']}")
        print(f"\nchecked {len(rep.get('paired', {}))} paired comparisons in {check_report}")
    if out:
        json.dump(out_j, open(out, "w"), indent=1)
        print("wrote", out)
    if problems:
        print("\nMISMATCHES:\n  " + "\n  ".join(problems))
        return 1
    print("\nall recomputed summary and paired values match the stored files" if (check_report or runs) else "")
    return 0


def compare(orig, new):
    """Per-item agreement between an original output file and a rerun (e.g. after rebuilding models/sft)."""
    a, b = load(orig), load(new)
    ids = sorted(set(a) & set(b))
    same_z = sum(a[i]["z"][0] == b[i]["z"][0] for i in ids)
    same_txt = sum(a[i]["text"][0] == b[i]["text"][0] for i in ids)
    ka, kb = sum(a[i]["z"][0] for i in ids), sum(b[i]["z"][0] for i in ids)
    print(f"items in both {len(ids)} (only original {len(set(a) - set(b))}, only rerun {len(set(b) - set(a))})")
    print(f"EM original {ka}/{len(ids)} = {ka/len(ids)*100:.1f}%, rerun {kb}/{len(ids)} = {kb/len(ids)*100:.1f}%")
    print(f"identical z on {same_z}/{len(ids)} items, byte-identical generated text on {same_txt}/{len(ids)} items")
    flips = [i for i in ids if a[i]["z"][0] != b[i]["z"][0]]
    for i in flips[:20]:
        print(f"  z {a[i]['z'][0]} -> {b[i]['z'][0]}  {i}")
    ok = len(ids) == len(a) == len(b) and same_z == len(ids)
    print("REBUILD_CHECK", "PASS (all exact-match outcomes identical)" if ok else "DIFFERS (report the flips above)")
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res")
    ap.add_argument("--data")
    ap.add_argument("--check-report")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("ORIGINAL", "RERUN"))
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare))
    if not (a.res and a.data):
        ap.error("--res and --data are required unless --compare is given")
    sys.exit(run(a.res, a.data, a.check_report, a.out))


if __name__ == "__main__":
    main()
