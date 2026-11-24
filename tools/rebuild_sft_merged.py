"""Rebuild models/sft (Qwen3-1.7B with sft_adapter merged in bf16), the start point and eval base of the GRPO adapters.

This repeats the last steps of src/sft.py. It was written after the run and is not part of the executed code.
Use the versions in configs/environment.json on a GPU, then check the rebuild by re-running
    python src/infer.py --model models/sft --data data/splits/test.jsonl --out /tmp/test_sft.jsonl
and comparing against results/test_sft.jsonl (49.2% exact match).

python tools/rebuild_sft_merged.py --adapter sft_adapter --out models/sft
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
model = AutoModelForCausalLM.from_pretrained(BASE, revision=REV, torch_dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.save_pretrained(a.out, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE, revision=REV).save_pretrained(a.out)
print("saved", a.out)
