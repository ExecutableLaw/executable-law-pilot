"""CPU-only verification of the released benchmark, predictions and scorer (no GPU, no network, no model weights).

  python tools/verify_cpu.py

Checks, in order:
  1. src/*.py match the SHA-256 hashes of the code that ran (configs/run_code_sha256.txt).
  2. The frozen Title 21 inputs match heldout/t21/manifest.json.
  3. Every reference program in the splits and the Title 21 sets, run by the edit-program interpreter
     (src/el_core.py), reproduces the official amended text exactly.
  4. Re-scoring every stored model output with src/scoring.py reproduces the stored exact-match, validity,
     abstention and confidence values item by item.
Exit status 0 if everything matches, 1 otherwise. Needs lxml (imported by src/el_core.py).
"""
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from el_core import canon, execute  # noqa: E402
from scoring import score  # noqa: E402

problems = []


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def jsonl(p):
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


# 1. code hashes
n = 0
for line in open("configs/run_code_sha256.txt"):
    if line.strip():
        h, name = line.split()
        n += 1
        if sha(os.path.join("src", name)) != h:
            problems.append(f"src/{name} differs from configs/run_code_sha256.txt")
print(f"[1] code hashes: {n} files checked")

# 2. frozen Title 21 inputs
man = json.load(open("heldout/t21/manifest.json"))
for name, h in man["sha256_outputs"].items():
    if sha(os.path.join("heldout/t21", name)) != h:
        problems.append(f"heldout/t21/{name} differs from manifest.json")
print(f"[2] frozen Title 21 inputs: {len(man['sha256_outputs'])} files checked")

# 3. reference programs reproduce the official text
sets = {p: jsonl(p) for p in sorted(glob.glob("data/splits/*.jsonl")) + ["heldout/t21/heldout_all.jsonl"]}
for p, rows in sets.items():
    bad = 0
    for ex in rows:
        try:
            ok = canon(execute(ex["ref_program"], ex["before"])) == canon(ex["after"])
        except Exception:
            ok = False
        bad += not ok
    if bad:
        problems.append(f"{p}: {bad} reference programs do not reproduce the official text")
    print(f"[3] {p}: {len(rows) - bad}/{len(rows)} reference programs reproduce the official text")

# 4. re-score stored predictions
items = {}
for rows in sets.values():
    for ex in rows:
        items[ex["id"]] = ex
runs = sorted(glob.glob("results/*.jsonl") + glob.glob("results/title21/*.jsonl") + glob.glob("results/title21/title47_rerun/*.jsonl"))
total = 0
for p in runs:
    mism = 0
    rows = jsonl(p)
    for r in rows:
        ex = items[r["id"]]
        for k, text in enumerate(r["text"]):  # outputs keep up to 2 generated texts per item
            s = score(text, ex)
            stored = {"z": r["z"][k], "valid": r["valid"][k], "abstain": r["abstain"][k], "conf": r["conf"][k]}
            if s != stored:
                mism += 1
            total += 1
    if mism:
        problems.append(f"{p}: {mism} outputs re-score differently")
    print(f"[4] {p}: {len(rows)} items re-scored, {mism} mismatches")

print(f"\n{total} stored outputs re-scored")
if problems:
    print("FAILED:\n  " + "\n  ".join(problems))
    sys.exit(1)
print("ALL CHECKS PASSED")
