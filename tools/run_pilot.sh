#!/usr/bin/env bash
# Title 47 pilot stages on one CUDA GPU (pilot: 1x L40S 48 GB). Run from the repository root.
#
#   bash tools/run_pilot.sh <stage> [<stage> ...]
#   stages: baselines sft prefilter grpo_R1 grpo_R3 sft_all report   (or: all)
#
# Environment:
#   SPLITS   input splits             (default data/splits)
#   OUT      models, outputs, logs    (default runs/pilot)
#   PYTHON   interpreter              (default python)
#   MINUTES  GRPO wall-clock budget   (default 50, as in the pilot)
#
# Commands and settings are those of configs/*.json. This replaces the machine-specific stage scripts of the
# original run; the Python entry points in src/ are the exact files that ran (configs/run_code_sha256.txt).
set -euo pipefail
[[ $# -ge 1 ]] || { sed -n 2,12p "$0"; exit 2; }
SP=${SPLITS:-data/splits}
OUT=${OUT:-runs/pilot}
PY=${PYTHON:-python}
M=$OUT/models R=$OUT/results L=$OUT/logs
mkdir -p "$M" "$R" "$L"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=${MASTER_PORT:-29511}

stage() {
  case "$1" in
    baselines)
      $PY src/infer.py --model Qwen/Qwen3-1.7B --data "$SP/test.jsonl" --out "$R/test_base1.7b_0shot.jsonl"
      for m in 1.7B 8B 14B; do
        $PY src/infer.py --model "Qwen/Qwen3-$m" --data "$SP/test.jsonl" --shots "$SP/shots.jsonl" --nshots 3 \
          --out "$R/test_base$(echo "$m" | tr 'B' 'b')_3shot.jsonl"
      done ;;
    sft)
      $PY src/sft.py --data "$SP/sft.jsonl" --out "$M/sft" --bs 1 --accum 16 --epochs 3
      $PY src/infer.py --model "$M/sft" --data "$SP/test.jsonl" --out "$R/test_sft.jsonl"
      $PY src/infer.py --model "$M/sft" --data "$SP/test.jsonl" --abstain --out "$R/test_sft_abstainprompt.jsonl" ;;
    prefilter)
      # 8 samples per RL-pool prompt from the SFT model; keep prompts with 1 to 7 correct (pilot: 175 of 309)
      $PY src/infer.py --model "$M/sft" --data "$SP/rl.jsonl" --n 8 --temp 1.0 --max-tokens 1280 --out "$R/rl_prefilter.jsonl"
      $PY - "$R/rl_prefilter.jsonl" "$SP/rl.jsonl" "$OUT/rl_prompts.jsonl" <<'PY'
import collections, json, sys
pre, rl, out = sys.argv[1:4]
k = {r["id"]: sum(r["z"]) for r in map(json.loads, open(pre))}
keep = [l for l in open(rl) if 0 < k.get(json.loads(l)["id"], 0) < 8]
open(out, "w").writelines(keep)
print("pass-count histogram", sorted(collections.Counter(k.values()).items()), "kept", len(keep))
PY
      ;;
    grpo_R1|grpo_R3)
      V=${1#grpo_}
      P="$OUT/rl_prompts.jsonl"; [[ -s "$P" ]] || P="$SP/rl_prompts.jsonl"   # fall back to the pilot's prompt set
      $PY src/grpo.py --model "$M/sft" --data "$P" --variant "$V" --out "$M/grpo_$V" --micro 4 --minutes "${MINUTES:-50}"
      if [[ $V == R3 ]]; then
        $PY src/infer.py --model "$M/sft" --lora "$M/grpo_${V}_adapter" --data "$SP/test.jsonl" --abstain --out "$R/test_grpo_$V.jsonl"
      else
        $PY src/infer.py --model "$M/sft" --lora "$M/grpo_${V}_adapter" --data "$SP/test.jsonl" --out "$R/test_grpo_$V.jsonl"
      fi ;;
    sft_all)
      # data-matched control: SFT on the SFT split plus the whole RL pool (772 examples)
      cat "$SP/sft.jsonl" "$SP/rl.jsonl" > "$OUT/sft_all.jsonl"
      $PY src/sft.py --data "$OUT/sft_all.jsonl" --out "$M/sft_all" --bs 1 --accum 16 --epochs 3
      $PY src/infer.py --model "$M/sft_all" --data "$SP/test.jsonl" --out "$R/test_sft_all.jsonl" ;;
    report)
      $PY src/report.py --res "$R" --splits "$SP" --models "$M" --funnel "$SP/funnel_all.json" ;;
    *) echo "unknown stage $1" >&2; exit 2 ;;
  esac
}

stages=("$@")
[[ "${stages[0]}" == all ]] && stages=(baselines sft prefilter grpo_R1 grpo_R3 sft_all report)
for s in "${stages[@]}"; do
  echo "== $(date -u +%H:%M:%S) begin $s"
  stage "$s" 2>&1 | tee "$L/$s.log"
  echo "== $(date -u +%H:%M:%S) end $s"
done
