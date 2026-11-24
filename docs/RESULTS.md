# Results

Two experiments, both scored by executing the model's edit program and comparing the result with the official
eCFR text (exact match after canonicalisation, `src/scoring.py`).

1. **Title 47 pilot (2026-11-23).** Qwen3-1.7B trained on 47 CFR (FCC) amendments effective 2017 to 2023 and
   tested on amendments effective 2024 to 2025. One NVIDIA L40S, 3.8 GPU-hours end to end.
2. **Title 21 transfer (2026-11-24).** The Title 47 GRPO R1 model and a data-matched SFT control, evaluated
   without further training on frozen 21 CFR (FDA/DEA) amendments.

Every number below is recomputed from the per-item files by `tools/paired_metrics.py` (see the README).

## 1. Data and alignment (Title 47)

Final rules affecting 47 CFR with effective dates from 2017-01-20 to 2025-12-31 (925 rules) were paired with eCFR
point-in-time section text on the day before and the day of each effective date.

| Stage | Section-level instructions |
|---|---|
| All section instructions parsed from rule XML | 3,448 |
| No eCFR version on the stated effective date | 712 |
| Instruction parsed but no amendatory text for the section | 258 |
| No text change in eCFR | 218 |
| Section removed / newly added (out of scope) | 122 / 66 |
| Diff not attested by the instruction text | 617 |
| **Aligned and reproduced exactly by a reference program** | **1,452** (42% of all, 52% of amendments to existing sections with a text change) |
| After filters (tables, untracked XML, same-day collisions, section > 1,500 tokens, strict re-attestation) | 951 |

Splits are by time (`data/splits/split_stats.json`). SFT gets 463 examples (2017 to 2023), the RL pool 309
(2017 to 2023, disjoint from SFT) and test 179 (88 from 2024, 91 from 2025). Every reference program reproduces the
official text exactly (`tools/verify_cpu.py` re-checks all of them). Any paragraph-level diff can be written in the
edit language, so it covers 100% of aligned amendments by construction.

## 2. Title 47 test set (n = 179, greedy, non-thinking mode)

| Model | EM | 95% CI (Wilson) | Valid programs | ECE | EM, 2025 only |
|---|---|---|---|---|---|
| Qwen3-1.7B, 0-shot | 2.8 | 1.2 to 6.4 | 26% | 0.961 | 2.2 |
| Qwen3-1.7B, 3-shot | 5.6 | 3.1 to 10.0 | 74% | 0.463 | 6.6 |
| Qwen3-8B, 3-shot | 33.5 | 27.0 to 40.7 | 85% | 0.166 | 33.0 |
| Qwen3-14B, 3-shot | 46.4 | 39.2 to 53.7 | 91% | 0.075 | 46.2 |
| 1.7B LoRA SFT (463 ex.) | 49.2 | 41.9 to 56.4 | 94% | 0.503 | 44.0 |
| 1.7B LoRA SFT, abstention prompt | 50.3 | 43.0 to 57.5 | 93% | 0.492 | 44.0 |
| 1.7B LoRA SFT on SFT + RL pool (772 ex., data-matched control) | 54.2 | 46.9 to 61.3 | 93% | 0.452 | 48.4 |
| **1.7B SFT + GRPO R1 (72 steps)** | **69.3** | **62.2 to 75.6** | 96% | 0.253 | **64.8** |
| 1.7B SFT + GRPO R3 (69 steps) | 60.9 | 53.6 to 67.7 | 92% | 0.388 | 58.2 |

R1 rewards 1 for an exact match, -0.5 for a wrong program and -1 for an invalid one. R3 adds a Brier term on the
stated confidence and allows abstention (`src/scoring.py`).

Paired differences on the same 179 items. The item bootstrap is the one in `src/report.py`; the document bootstrap
resamples whole Federal Register rules, since items from one rule are not independent.

| Comparison | Delta (points) | Item bootstrap 95% | Rule bootstrap 95% | Fixed / broken | McNemar p |
|---|---|---|---|---|---|
| GRPO R1 minus SFT | +20.1 | +13.4 to +27.4 | +12.7 to +27.7 | 42 / 6 | 1e-07 |
| GRPO R1 minus data-matched SFT | +15.1 | +8.4 to +21.8 | +8.6 to +21.5 | 35 / 8 | 4e-05 |
| GRPO R1 minus 14B 3-shot | +22.9 | +15.6 to +30.2 | +15.6 to +29.9 | 48 / 7 | 1e-08 |
| SFT minus 1.7B 3-shot | +43.6 | +36.3 to +51.4 | +35.3 to +52.2 | 79 / 1 | 1e-22 |
| SFT minus 8B 3-shot | +15.6 | +7.3 to +24.0 | +7.9 to +23.6 | 46 / 18 | 0.0006 |
| SFT minus 14B 3-shot | +2.8 | -5.0 to +10.6 | -6.5 to +11.8 | 29 / 24 | 0.58 |
| Data-matched SFT minus SFT | +5.0 | -0.6 to +10.6 | -0.5 to +11.3 | 17 / 8 | 0.11 |
| GRPO R3 minus GRPO R1 | -8.4 | -14.0 to -3.4 | -14.0 to -2.6 | 5 / 20 | 0.004 |

### Pre-specified pilot criteria

| Criterion | Result | Verdict |
|---|---|---|
| At least 70% of instructions align | 42% of all, 52% of in-scope amendments; the largest loss is effective-date mismatch (21%) | Not met |
| Edit language covers at least 95% of aligned amendments | 100% by construction | Met (trivially) |
| SFT clearly beats the base model and a few-shot 8B to 14B model | Beats 1.7B base (+43.6) and 8B (+15.6); ties 14B (+2.8, CI includes 0) | Partly met |
| GRPO beats SFT by at least 5 points | +20.1 vs SFT, +15.1 vs data-matched SFT | Met |
| R3 lowers calibration error vs R1 without losing accuracy | ECE 0.39 (R3) vs 0.25 (R1); EM 8.4 points lower | Not met |

### Measured throughput on one L40S

| Workload | Measured |
|---|---|
| LoRA SFT, 1.7B, completion-only loss | 3,540 training tokens/s |
| GRPO, 1.7B, 64 completions per step, vLLM colocated | 48 s/step (about 75 steps per GPU-hour) |
| vLLM generation, 1.7B, 8 samples per prompt | 6,056 decode tokens/s |

## 3. Title 21 transfer

The frozen sets in `heldout/t21/` were built with the same builder, filters and prompt, restricted to rules effective
2024 to 2025 (`manifest.json`). No Federal Register document is shared with the Title 47 data, no item's section,
before text or after text appears in training, and no instruction is identical or has a 5-gram Jaccard of 0.5 or
more with a training instruction (`overlap.json`). 41 items share a formulaic amendatory sentence with training once
section numbers are masked, and 2 have a reference program seen in training up to paragraph indices. The primary set
excludes parts 500 to 599 (formulaic animal-drug updates); the sensitivity set keeps them.

The original data-matched control checkpoint was not kept after the pilot, so it was retrained once with
`src/sft.py` on the same 772 examples and settings (957,723 tokens per epoch, 147 optimizer steps, final loss 0.122;
early losses track the original run to about 0.01). GRPO R1 was rebuilt from the pilot's adapters.

| Set | Items (rules) | GRPO R1 | Retrained control | Delta (points) | Item bootstrap 95% | Rule bootstrap 95% | Fixed / broken | McNemar p |
|---|---|---|---|---|---|---|---|---|
| Primary (`heldout.jsonl`) | 34 (17) | 76.5% (60.0 to 87.6) | 52.9% (36.7 to 68.5) | +23.5 | +5.9 to +41.2 | +8.1 to +37.9 | 10 / 2 | 0.039 |
| Sensitivity, with parts 500 to 599 (`heldout_all.jsonl`) | 89 (28) | 73.0% (63.0 to 81.2) | 46.1% (36.1 to 56.4) | +27.0 | +15.7 to +38.2 | +17.1 to +36.2 | 29 / 5 | 4e-05 |

The sign is the same in both years in both sets (primary: +28.6 in 2024, +20.0 in 2025).

### Rebuild check on Title 47 (`results/title21/title47_rerun/`)

| Model | Pilot | Rerun | Items changed |
|---|---|---|---|
| GRPO R1 (rebuilt SFT parent plus the pilot adapter) | 124/179 = 69.3% | 124/179 = 69.3% | 2 flips that cancel; generated text byte-identical on 173 of 179 |
| Data-matched control | 97/179 = 54.2% (original) | 93/179 = 52.0% (retrained) | 16 flips (6 gained, 10 lost) |

The GRPO rebuild reproduces the pilot. The retrained control lands 2.2 points below the original, within
run-to-run variation for one SFT seed; against it, GRPO R1 gains +17.3 points on Title 47 (37 fixed, 6 broken).

## Caveats

- One seed per model; 179 Title 47 test items (95% intervals about plus or minus 7 points) and 34 primary Title 21
  items (interval about +6 to +41). The direction on Title 21 is clear; the size is not.
- GRPO stopped on a 50-minute wall clock at 72 (R1) and 69 (R3) of 120 planned steps, with LoRA rank 32,
  learning rate 5e-5, 8 prompts by 8 samples per step and no KL penalty (beta = 0).
- GRPO adapters were evaluated as runtime LoRA on the merged SFT model, without a bf16 merge
  (`results/bf16_visibility_R*.json` measures how much of the update a bf16 merge would keep).
- Qwen3 was released in April 2025, so 2024 test items may appear in its pretraining data. The 2025-only column shows
  the same ordering (R1 64.8 vs SFT 44.0).
- R3 did not calibrate: the SFT model states confidence 1.0 almost deterministically, so within-group confidence
  barely varied and the Brier term added penalty without useful gradient. Abstention was never sampled. Seeding varied
  confidence and abstention before RL (for example SFT targets carrying empirical pass rates) is the obvious fix.
- Few-shot baselines used non-thinking mode, and their demonstrations show confidence 0.5, which anchors baseline ECE.
- The Title 21 control is a retrained checkpoint, not the pilot's original (Title 47 score within 2.2 points).
