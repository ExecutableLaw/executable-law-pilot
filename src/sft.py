"""LoRA SFT of Qwen3 on (instruction, before) -> reference program. Completion-only loss.
python sft.py --data sft.jsonl --out /data/models/sft [--base Qwen/Qwen3-1.7B]
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from prompts import messages, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    a = ap.parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                              TrainingArguments)
    tok = AutoTokenizer.from_pretrained(a.base)
    rows = [json.loads(l) for l in open(a.data)]
    feats = []
    for ex in rows:
        p = tok.apply_chat_template(messages(ex), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        # the program and the confidence tags are learned; the confidence VALUE is masked (no constant prior)
        head, val, tail = target(ex).rsplit("<confidence>", 1)[0] + "<confidence>", "0.5", "</confidence><|im_end|>"
        pi = tok(p, add_special_tokens=False)["input_ids"]
        hi = tok(head, add_special_tokens=False)["input_ids"]
        vi = tok(val, add_special_tokens=False)["input_ids"]
        ti = tok(tail, add_special_tokens=False)["input_ids"]
        ids = pi + hi + vi + ti
        labels = [-100] * len(pi) + hi + [-100] * len(vi) + ti
        feats.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    ntok = sum(len(f["input_ids"]) for f in feats)
    print(f"{len(feats)} examples, {ntok} tokens, max {max(len(f['input_ids']) for f in feats)}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    import dataclasses, math
    total_steps = math.ceil(len(feats) / (a.bs * a.accum) * a.epochs)
    cfg = dict(output_dir=a.out + "_ckpt", per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum,
               num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
               warmup_steps=max(1, int(0.03 * total_steps)), bf16=True, logging_steps=5, save_strategy="no",
               report_to="none", train_sampling_strategy="group_by_length", group_by_length=True,
               dataloader_num_workers=2, remove_unused_columns=False, seed=0)
    fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    print("TrainingArguments: dropping unsupported", [k for k in cfg if k not in fields], flush=True)
    for req in ("per_device_train_batch_size", "learning_rate", "warmup_steps", "bf16"):
        assert req in fields, req
    args = TrainingArguments(**{k: v for k, v in cfg.items() if k in fields})
    from datasets import Dataset

    class CompletionOnlyTrainer(Trainer):
        """Batch size 1: compute logits only where the next token is supervised (avoids full-vocab fp32 logits)."""

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            assert labels.shape[0] == 1, "use --bs 1"
            nxt = labels[0, 1:]
            keep = torch.nonzero(nxt != -100).squeeze(-1)
            out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], logits_to_keep=keep)
            logits = out.logits[0].float()
            assert logits.shape[0] == keep.shape[0], (logits.shape, keep.shape)
            loss = torch.nn.functional.cross_entropy(logits, nxt[keep])
            return (loss, out) if return_outputs else loss

    tr = CompletionOnlyTrainer(model=model, args=args, train_dataset=Dataset.from_list(feats),
                 data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    t0 = time.time()
    tr.train()
    dt = time.time() - t0
    model.save_pretrained(a.out + "_adapter")
    model = model.merge_and_unload()
    model.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    stats = {"examples": len(feats), "tokens_per_epoch": ntok, "epochs": a.epochs, "seconds": dt,
             "train_tok_per_s": ntok * a.epochs / dt, "final_loss": [h for h in tr.state.log_history if "loss" in h][-1:],
             "lr": a.lr, "rank": a.rank, "opt_steps": tr.state.global_step}
    json.dump(stats, open(os.path.join(a.out, "sft_stats.json"), "w"), indent=1)
    json.dump(tr.state.log_history, open(os.path.join(a.out, "sft_log.json"), "w"))
    print("SFT_DONE", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
