"""GRPO from the SFT checkpoint with execution-verified rewards R1 or R3 (TRL, vLLM colocate).
python grpo.py --model /data/models/sft --data rl_prompts.jsonl --variant R1 --out /data/models/grpo_R1
"""
import argparse
import dataclasses
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from prompts import messages
from scoring import reward, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", choices=["R1", "R3"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--vllm-mem", type=float, default=0.30)
    a = ap.parse_args()
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    tok = AutoTokenizer.from_pretrained(a.model)
    abstain = a.variant == "R3"
    rows = [json.loads(l) for l in open(a.data)]
    ds = Dataset.from_list([{"prompt": tok.apply_chat_template(messages(ex, (), abstain), tokenize=False,
                                                                add_generation_prompt=True, enable_thinking=False),
                             "ex": json.dumps({"before": ex["before"], "after": ex["after"]})} for ex in rows])
    log = []

    def reward_fn(prompts, completions, ex, **kw):
        out = []
        for c, e in zip(completions, ex):
            e = json.loads(e)
            text = c if isinstance(c, str) else c[0]["content"]
            r = reward(text, e, a.variant)
            s = score(text, e)
            log.append({"r": r, "z": s["z"], "valid": s["valid"], "abstain": s["abstain"], "conf": s["conf"]})
            out.append(r)
        return out

    total_completions = a.prompts_per_step * a.G
    cfg = dict(output_dir=a.out + "_ckpt", learning_rate=a.lr, lr_scheduler_type="constant_with_warmup", warmup_steps=5,
               per_device_train_batch_size=a.micro, gradient_accumulation_steps=total_completions // a.micro,
               num_generations=a.G, max_steps=a.steps, max_prompt_length=None, max_completion_length=1280,
               temperature=1.0, top_p=1.0, beta=0.0, num_iterations=1, bf16=True, gradient_checkpointing=True,
               gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, save_strategy="no",
               report_to="none", use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_mem,
               vllm_max_model_length=8192, mask_truncated_completions=True, seed=0, shuffle_dataset=True,
               remove_unused_columns=False, log_completions=False, model_init_kwargs={"dtype": torch.bfloat16})
    fields = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = [k for k in cfg if k not in fields]
    print("GRPOConfig: dropping unsupported", dropped, flush=True)
    for req in ("use_vllm", "vllm_mode", "num_generations", "beta", "max_completion_length", "gradient_checkpointing",
                "vllm_max_model_length", "per_device_train_batch_size", "gradient_accumulation_steps"):
        assert req in fields, f"GRPOConfig lacks {req}"
    args = GRPOConfig(**{k: v for k, v in cfg.items() if k in fields})

    t_start = time.time()

    class Clock(TrainerCallback):
        def on_step_end(self, args_, state, control, **kw):
            el = (time.time() - t_start) / 60
            recent = log[-total_completions:]
            if recent:
                print(f"STEP {state.global_step} {el:.1f}min mean_r={sum(x['r'] for x in recent)/len(recent):.3f} "
                      f"z={sum(x['z'] for x in recent)/len(recent):.3f} valid={sum(x['valid'] for x in recent)/len(recent):.3f} "
                      f"abst={sum(x['abstain'] for x in recent)/len(recent):.3f}", flush=True)
            if el > a.minutes:
                print("WALL_CLOCK_STOP", flush=True)
                control.should_training_stop = True

    trainer = GRPOTrainer(model=a.model, reward_funcs=reward_fn, args=args, train_dataset=ds,
                          peft_config=LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
                          callbacks=[Clock()])
    trainer.train()
    dt = time.time() - t_start
    trainer.model.save_pretrained(a.out + "_adapter")  # evaluated with runtime LoRA (no bf16 merge rounding)
    tok.save_pretrained(a.out + "_adapter")
    # diagnostic: share of weights whose merged bf16 value differs from the SFT weight
    vis = {}
    try:
        for name, mod in trainer.model.named_modules():
            if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}) and len(vis) < 14:
                W = mod.base_layer.weight.detach().float()
                dW = mod.get_delta_weight("default").detach().float()
                vis[name] = float(((W + dW).to(torch.bfloat16) != W.to(torch.bfloat16)).float().mean())
    except Exception as e:
        vis["error"] = repr(e)
    json.dump(vis, open(os.path.join(a.out + "_adapter", "bf16_visibility.json"), "w"), indent=1)
    print("BF16_VISIBILITY", json.dumps(vis), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(log, open(os.path.join(a.out, "reward_log.json"), "w"))
    json.dump(trainer.state.log_history, open(os.path.join(a.out, "grpo_log.json"), "w"))
    stats = {"variant": a.variant, "steps": trainer.state.global_step, "seconds": dt, "completions": len(log),
             "lr": a.lr, "rank": a.rank, "G": a.G, "prompts_per_step": a.prompts_per_step, "dropped_cfg": dropped}
    json.dump(stats, open(os.path.join(a.out, "grpo_stats.json"), "w"), indent=1)
    print("GRPO_DONE", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
