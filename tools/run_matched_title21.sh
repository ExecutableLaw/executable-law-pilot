#!/usr/bin/env bash
# Title 21 held-out comparison: original GRPO R1 versus the data-matched SFT control (772 examples).
# Run from the pilot repository root in the original CUDA environment (configs/environment.json).
#
# Usage:
#   CONTROL_PROVENANCE=original|retrained bash tools/run_matched_title21.sh ADAPTER_DIR CONTROL NEW_OUTPUT_DIR
#   PREFLIGHT_ONLY=1 CONTROL_PROVENANCE=... bash tools/run_matched_title21.sh ADAPTER_DIR CONTROL NEW_OUTPUT_DIR
#
# ADAPTER_DIR  holds the pilot's sft_adapter/ and grpo_R1_adapter/ (expected SHA-256 in results/title21/provenance.json)
# CONTROL     the data-matched control, either a merged model directory (config.json + *.safetensors, e.g.
#              models/sft_all from src/sft.py) or a LoRA adapter directory (adapter_config.json +
#              adapter_model.safetensors, e.g. models/sft_all_adapter), which is merged onto the pinned
#              Qwen/Qwen3-1.7B with tools/rebuild_sft_merged.py exactly like the ordinary SFT parent
# NEW_OUTPUT_DIR must not exist
# CONTROL_PROVENANCE (required) original = the 2026-11-23 pilot checkpoint, retrained = a new training run
# PREFLIGHT_ONLY=1  CPU-only checks; creates nothing and imports no GPU library
# PYTHON            interpreter (default: python)
#
# The preflight rejects the ordinary SFT model offered as the control (identical adapter weight hashes, training
# stats with 463 examples, or, after the run, Title 47 outputs identical to the ordinary SFT outputs), checks the
# frozen Title 21 inputs and the scoring code against their recorded hashes, and records provenance.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo "Usage: CONTROL_PROVENANCE=original|retrained [PREFLIGHT_ONLY=1] $0 ADAPTER_DIR CONTROL NEW_OUTPUT_DIR" >&2
  exit 2
fi
adapters=$1
control=$2
out=$3
data=heldout/t21
py=${PYTHON:-python}
provenance=${CONTROL_PROVENANCE:-}
preflight_only=${PREFLIGHT_ONLY:-0}
if [[ "$provenance" != original && "$provenance" != retrained ]]; then
  echo "Set CONTROL_PROVENANCE=original (the pilot's own sft_all checkpoint) or CONTROL_PROVENANCE=retrained" >&2
  exit 2
fi
for file in "$adapters/sft_adapter/adapter_model.safetensors" "$adapters/grpo_R1_adapter/adapter_model.safetensors" "$data/heldout_all.jsonl" "$data/heldout.jsonl" "$data/manifest.json"; do
  [[ -s "$file" ]] || { echo "Missing input: $file" >&2; exit 1; }
done
if [[ -e "$out" ]]; then
  echo "Output directory already exists: $out (choose a new one)" >&2
  exit 1
fi
if [[ "$preflight_only" == 1 ]]; then
  record=-
else
  mkdir "$out"
  mkdir "$out/title47" "$out/title21"
  record="$out/provenance.json"
fi

"$py" - "$adapters" "$control" "$data" "$provenance" "$record" <<'PY_PREFLIGHT'
"""CPU-only checks (standard library only). Exit 1 on any failure; write provenance JSON to argv[5] ('-' = stdout)."""
import datetime, glob, hashlib, json, os, shutil, subprocess, sys
from importlib import metadata

adapters, control, data, provenance, record = sys.argv[1:6]
errors, warnings = [], []


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    with open(p) as f:
        return json.load(f)


# frozen Title 21 inputs and the code that prompts and scores
man = load(os.path.join(data, "manifest.json"))
for name in ("heldout.jsonl", "heldout_all.jsonl"):
    if sha(os.path.join(data, name)) != man["sha256_outputs"][name]:
        errors.append(f"frozen input changed: {data}/{name}")
code = {}
for line in open("configs/run_code_sha256.txt"):
    if line.strip():
        h, n = line.split()
        code[n] = h
for n in ("prompts.py", "scoring.py", "el_core.py", "infer.py", "sft.py"):
    if sha(os.path.join("src", n)) != code[n]:
        errors.append(f"src/{n} differs from configs/run_code_sha256.txt")
for n in ("paired_metrics.py", "rebuild_sft_merged.py"):
    if not os.path.exists(os.path.join("tools", n)):
        errors.append(f"missing tools/{n}")

# released adapters
rel = {}
for n in ("sft_adapter", "grpo_R1_adapter"):
    d = os.path.join(adapters, n)
    w, c = os.path.join(d, "adapter_model.safetensors"), os.path.join(d, "adapter_config.json")
    if not os.path.exists(c):
        errors.append(f"missing {c}"); continue
    # base_model_name_or_path records a training-machine path; compare every other field
    got = {k: v for k, v in load(c).items() if k != "base_model_name_or_path"}
    ref = {k: v for k, v in load(os.path.join("configs/lora", n + ".json")).items() if k != "base_model_name_or_path"}
    if got != ref:
        errors.append(f"{c} differs from configs/lora/{n}.json")
    rel[n] = sha(w)
if len(set(rel.values())) < len(rel):
    errors.append("sft_adapter and grpo_R1_adapter have identical weights")

# the control
orig_stats = load("results/REPORT.json")["train_sft_all"]
sft_stats = load("results/REPORT.json")["train_sft"]
has_adapter = os.path.exists(os.path.join(control, "adapter_config.json"))
has_merged = os.path.exists(os.path.join(control, "config.json"))
ctl = {"path": os.path.normpath(control), "provenance": provenance}
if has_adapter and has_merged:
    errors.append(f"{control} has both adapter_config.json and config.json; pass the adapter or the merged directory")
elif has_adapter:
    ctl["kind"] = "adapter (merged here onto Qwen/Qwen3-1.7B@70d244cc with tools/rebuild_sft_merged.py)"
    w = os.path.join(control, "adapter_model.safetensors")
    if not os.path.exists(w):
        errors.append(f"missing {w}")
    else:
        ctl["weights_sha256"] = {"adapter_model.safetensors": sha(w)}
        h = ctl["weights_sha256"]["adapter_model.safetensors"]
        if h == rel.get("sft_adapter"):
            errors.append("control adapter weights are identical to the ordinary sft_adapter (463 examples), not the matched control")
        if h == rel.get("grpo_R1_adapter"):
            errors.append("control adapter weights are identical to grpo_R1_adapter")
    cfg = load(os.path.join(control, "adapter_config.json"))
    want = load("configs/lora/sft_adapter.json")  # the matched control used the same LoRA settings as SFT
    for k in ("r", "lora_alpha", "lora_dropout", "peft_type", "task_type"):
        if cfg.get(k) != want.get(k):
            errors.append(f"control adapter {k}={cfg.get(k)} but the matched SFT used {want.get(k)}")
    if sorted(cfg.get("target_modules") or []) != sorted(want["target_modules"]):
        errors.append("control adapter target_modules differ from the SFT settings")
    base = str(cfg.get("base_model_name_or_path"))
    if "Qwen3-1.7B" not in base:
        warnings.append(f"control adapter base_model_name_or_path is {base}; it will be merged onto pinned Qwen/Qwen3-1.7B")
    c_ = control.rstrip("/")
    stats_dirs = [c_, c_[:-len("_adapter")] if c_.endswith("_adapter") else c_]
elif has_merged:
    ctl["kind"] = "merged model"
    ws = sorted(glob.glob(os.path.join(control, "*.safetensors")))
    if not ws:
        errors.append(f"no *.safetensors in {control}")
    ctl["weights_sha256"] = {os.path.basename(p): sha(p) for p in ws}
    stats_dirs = [control]
else:
    errors.append(f"{control} is neither a merged model (config.json) nor a LoRA adapter (adapter_config.json)")
    stats_dirs = []

# training stats written by src/sft.py next to the merged model
st = None
for d in stats_dirs:
    p = os.path.join(d.rstrip("/"), "sft_stats.json")
    if os.path.exists(p):
        st = load(p); ctl["sft_stats"] = st; ctl["sft_stats_path"] = os.path.normpath(p); break
key = ("examples", "tokens_per_epoch", "opt_steps", "final_loss")
if st is not None:
    if st.get("examples") == sft_stats["examples"] or all(st.get(k) == sft_stats[k] for k in key):
        errors.append(f"control training stats show {st.get('examples')} examples: this is the ordinary SFT model, not the matched control")
    elif st.get("examples") != orig_stats["examples"] or st.get("tokens_per_epoch") != orig_stats["tokens_per_epoch"]:
        errors.append(f"control was trained on {st.get('examples')} examples / {st.get('tokens_per_epoch')} tokens per epoch; "
                      f"the matched control uses data/splits/sft_all.jsonl ({orig_stats['examples']} / {orig_stats['tokens_per_epoch']})")
    same_as_orig = all(st.get(k) == orig_stats[k] for k in key)
    if provenance == "original" and not same_as_orig:
        errors.append("CONTROL_PROVENANCE=original but sft_stats.json differs from the pilot's sft_all run (results/REPORT.json train_sft_all); use retrained")
    if provenance == "retrained" and same_as_orig:
        errors.append("CONTROL_PROVENANCE=retrained but sft_stats.json is identical to the pilot's original sft_all run; use original")
elif provenance == "retrained":
    errors.append("retrained control has no sft_stats.json (src/sft.py writes it into the --out directory); keep it next to the model")
else:
    warnings.append("no sft_stats.json for the original control; provenance rests on CONTROL_PROVENANCE and the Title 47 comparison")

# environment (versions read from package metadata; nothing is imported)
want_v = load("configs/environment.json")["versions_reported_on_box"]
env = {}
for m in ("torch", "vllm", "transformers", "peft", "trl"):
    try:
        env[m] = metadata.version(m)
    except Exception:
        env[m] = None
    if env[m] != want_v[m]:
        warnings.append(f"{m} {env[m]} (pilot {want_v[m]})")
env["python"] = sys.version.split()[0]
if shutil.which("nvidia-smi"):
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True)
    env["gpu"] = q.stdout.strip() or q.stderr.strip()
else:
    env["gpu"] = None
    warnings.append("nvidia-smi not found (fine for the preflight; the run needs a CUDA GPU)")
try:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
except Exception:
    head = None

rec = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
       "repo_commit": head, "control": ctl, "released_adapter_sha256": rel,
       "frozen_inputs_sha256": {n: man["sha256_outputs"][n] for n in ("heldout.jsonl", "heldout_all.jsonl")},
       "environment": env, "warnings": warnings, "errors": errors}
txt = json.dumps(rec, indent=1)
if record == "-":
    print(txt)
else:
    with open(record, "w") as f:
        f.write(txt + "\n")
    print(f"provenance written to {record}")
for w_ in warnings:
    print("WARNING", w_, file=sys.stderr)
for e in errors:
    print("ERROR", e, file=sys.stderr)
print("PREFLIGHT", "FAILED" if errors else "OK", f"(control: {ctl.get('kind')}, {provenance})", file=sys.stderr)
sys.exit(1 if errors else 0)
PY_PREFLIGHT

if [[ "$preflight_only" == 1 ]]; then
  exit 0
fi

"$py" tools/rebuild_sft_merged.py --adapter "$adapters/sft_adapter" --out "$out/sft"
if [[ -s "$control/adapter_config.json" ]]; then
  "$py" tools/rebuild_sft_merged.py --adapter "$control" --out "$out/sft_all_merged"
  control_model="$out/sft_all_merged"
else
  control_model="$control"
fi
"$py" - "$out/sft" "$control_model" <<'PY'
import glob, hashlib, os, sys
def digest(d):
    return {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest()
            for p in sorted(glob.glob(os.path.join(d, "*.safetensors")))}
if digest(sys.argv[1]) == digest(sys.argv[2]):
    raise SystemExit("The control's merged weights are identical to the ordinary SFT parent: not the matched control")
PY
for split in title47 title21; do
  if [[ "$split" == title47 ]]; then
    items=data/splits/test.jsonl
  else
    items="$data/heldout_all.jsonl"
  fi
  "$py" src/infer.py --model "$out/sft" --lora "$adapters/grpo_R1_adapter" --data "$items" --out "$out/$split/test_grpo_R1.jsonl"
  "$py" src/infer.py --model "$control_model" --data "$items" --out "$out/$split/test_sft_all.jsonl"
  "$py" - "$items" "$out/$split" <<'PY'
import json, sys
from pathlib import Path
def ids(path):
    return [json.loads(line)["id"] for line in Path(path).read_text().splitlines()]
expected = ids(sys.argv[1])
for name in ("test_sft_all.jsonl", "test_grpo_R1.jsonl"):
    if ids(Path(sys.argv[2]) / name) != expected:
        raise SystemExit(f"Incomplete or misaligned predictions: {name}")
PY
  if [[ "$split" == title47 ]]; then
    # behavioural guard: a control whose Title 47 answers are byte-identical to the ordinary SFT model is that model
    "$py" - "$out/title47/test_sft_all.jsonl" <<'PY'
import json, sys
load = lambda p: {r["id"]: r["text"][0] for r in map(json.loads, open(p))}
new, sft, orig = load(sys.argv[1]), load("results/test_sft.jsonl"), load("results/test_sft_all.jsonl")
same_sft = sum(new[i] == sft[i] for i in new)
same_orig = sum(new[i] == orig[i] for i in new)
print(f"control Title 47 text identical to ordinary SFT on {same_sft}/{len(new)}, to original sft_all on {same_orig}/{len(new)}")
if same_sft == len(new) and same_orig < len(new):
    raise SystemExit("Control reproduces the ordinary SFT outputs exactly: not the matched control")
PY
  fi
done
for model in grpo_R1 sft_all; do
  # Model rebuilds and independent retrains can change individual answers.
  # Preserve both sets of predictions and surface those differences explicitly.
  comparison_status=0
  "$py" tools/paired_metrics.py --compare "results/test_$model.jsonl" "$out/title47/test_$model.jsonl" > "$out/title47/$model-comparison.txt" || comparison_status=$?
  cat "$out/title47/$model-comparison.txt"
  if [[ "$comparison_status" != 0 && "$comparison_status" != 2 ]]; then exit "$comparison_status"; fi
done
for subset in heldout heldout_all; do
  "$py" tools/paired_metrics.py --res "$out/title21" --data "$data/$subset.jsonl" --out "$out/title21/$subset-report.json"
done
echo "Done. Control provenance: $provenance (see $out/provenance.json)"
