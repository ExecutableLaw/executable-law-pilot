"""Filter aligned examples and split by time. python make_splits.py --in examples.jsonl --out DIR"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from prompts import messages, target
from el_core import attested, parse_fr_rule, render_instruction

_fr_cache = {}


def refresh_fr(ex):
    """Re-parse the rule XML with the current parser (sub-instructions carried over) and re-attest strictly."""
    import requests, time
    doc = ex["doc"]
    if doc not in _fr_cache:
        y, m, d = ex.get("pub_date", "0000-00-00").split("-") if ex.get("pub_date") else (None, None, None)
        meta = requests.get(f"https://www.federalregister.gov/api/v1/documents/{doc}.json",
                            params={"fields[]": ["full_text_xml_url"]}, timeout=60).json()
        time.sleep(0.2)
        x = requests.get(meta["full_text_xml_url"], timeout=60).content if meta.get("full_text_xml_url") else None
        time.sleep(0.2)
        _fr_cache[doc] = parse_fr_rule(x, ex["title"]) if x else {}
    fr = _fr_cache[doc].get(ex["section"])
    if not fr or not fr["amdpars"]:
        return False, "fr_reparse_failed"
    ok, why = attested(ex["before"], ex["after"], fr)
    if not ok:
        return False, "strict_" + why
    ex["amdpars"], ex["supplied"], ex["heading"] = fr["amdpars"], fr["supplied"], fr.get("heading")
    ex["instruction"] = render_instruction(fr)
    return True, "ok"

EXCLUDE_SECTIONS = {"73.202", "73.622", "73.606", "73.504"}  # FM/TV allotment tables (formulaic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tok", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--max-before", type=int, default=1500)
    ap.add_argument("--max-target", type=int, default=1200)
    ap.add_argument("--max-prompt", type=int, default=5000)
    ap.add_argument("--test-cap", type=int, default=1000)
    ap.add_argument("--sft-frac", type=float, default=0.6)
    ap.add_argument("--nshots", type=int, default=3)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)
    os.makedirs(a.out, exist_ok=True)
    # collisions: several rules touching the same section on the same date (from examples AND rejects)
    docs_per_key = collections.defaultdict(set)
    for path in a.inp:
        title = None
        for f in (path, os.path.join(os.path.dirname(path), "rejects.jsonl")):
            if not os.path.exists(f):
                continue
            for l in open(f):
                r = json.loads(l)
                if r.get("section") and r.get("date"):
                    docs_per_key[(os.path.dirname(path), r["section"], r["date"])].add(r["doc"])
    seen, kept, drop = set(), [], collections.Counter()
    for path in a.inp:
        for l in open(path):
            ex = json.loads(l)
            key = (ex["title"], ex["section"], ex["date"])
            if key in seen:
                drop["duplicate"] += 1; continue
            seen.add(key)
            if len(docs_per_key[(os.path.dirname(path), ex["section"], ex["date"])]) > 1:
                drop["same_day_collision"] += 1; continue
            if ex["flags"].get("other_tags"):
                drop["untracked_xml_tags"] += 1; continue
            if ex["flags"].get("table"):
                drop["table"] += 1; continue
            if ex["section"] in EXCLUDE_SECTIONS and str(ex["title"]) == "47":
                drop["formulaic"] += 1; continue
            ok, why = refresh_fr(ex)
            if not ok:
                drop[why] += 1; continue
            nb = len(tok("\n".join(ex["before"]))["input_ids"])
            if nb > a.max_before:
                drop["before_too_long"] += 1; continue
            nt = len(tok(target(ex))["input_ids"])
            if nt > a.max_target:
                drop["target_too_long"] += 1; continue
            np_ = len(tok(tok.apply_chat_template(messages(ex), tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False), add_special_tokens=False)["input_ids"])
            if np_ > a.max_prompt:
                drop["prompt_too_long"] += 1; continue
            ex["id"] = f"{ex['title']}-{ex['section']}-{ex['date']}-{ex['doc']}"
            ex["n_before_tok"], ex["n_target_tok"], ex["n_prompt_tok"] = nb, nt, np_
            ex["ops"] = sorted({o["op"] for o in ex["ref_program"]})
            kept.append(ex)
    train = [e for e in kept if e["date"] < "2024-01-01"]
    test = [e for e in kept if "2024-01-01" <= e["date"] <= "2025-12-31"]
    rng = random.Random(0)
    rng.shuffle(train)
    rng.shuffle(test)
    test_all = len(test)
    test = test[:a.test_cap]
    k = int(len(train) * a.sft_frac)
    sft, rl = train[:k], train[k:]
    # few-shot pool: short SFT examples covering the main op types
    short = sorted([e for e in sft if e["n_prompt_tok"] < 900], key=lambda e: e["n_prompt_tok"])
    shots, have = [], set()
    for want in (["replace"], ["set"], ["insert_after"], ["delete"]):
        for e in short:
            if e["ops"] == want and e["id"] not in {s["id"] for s in shots}:
                shots.append(e); break
    assert len(shots) >= a.nshots, f"only {len(shots)} shots"
    for name, rows in (("sft", sft), ("rl", rl), ("test", test), ("shots", shots)):
        with open(os.path.join(a.out, f"{name}.jsonl"), "w") as f:
            for e in rows:
                f.write(json.dumps(e) + "\n")
    stats = {"kept": len(kept), "dropped": drop, "train": len(train), "sft": len(sft), "rl": len(rl),
             "test": len(test), "test_available": test_all, "shots": [s["id"] for s in shots],
             "op_mix_test": collections.Counter("+".join(e["ops"]) for e in test).most_common(8),
             "years": collections.Counter(e["date"][:4] for e in kept),
             "titles": collections.Counter(str(e["title"]) for e in kept)}
    json.dump(stats, open(os.path.join(a.out, "split_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
