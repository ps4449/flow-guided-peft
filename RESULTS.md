# Results

LLaVA-1.5-7B on two datasets: [GQA](#gqa) (already in LLaVA's training data) and [ScienceQA](#scienceqa) (new to LLaVA).

**Batch** is microbatch × gradient accumulation (16 questions per update either way). **4 × 4 ⚠️** runs used the shared
trainer (`peft_lora/train_lora.py`), which did not pass `modalities` to LLaVA-NeXT. In the installed version, a batch of 4
without it keeps only the first sample for the loss, so these runs may have learned from ~1 of every 4 training questions.
Treat them as provisional until rerun with the fix; comparisons between these affected runs and the **1 × 16** runs (AdaLoRA) are not fair yet.
**4 × 4 ✅** runs used the fixed trainer (it passes `modalities` for every sample); see [docs/MULTIMODAL_BATCH_FIX.md](docs/MULTIMODAL_BATCH_FIX.md).
Zero-shot and information-flow results do not train, so they are not affected.

## GQA

GQA testdev-balanced (12,578 questions).

Training answers are either lowercase (GQA's original answers) or capitalised (first letter uppercase, matching
LLaVA's own style). Capitalised training scores higher, so the two are kept in separate tables and only rows in the
same table should be compared. Using capitalised training for configurations moving forward.
See [docs/CAPITALIZATION_FIX.md](docs/CAPITALIZATION_FIX.md) for the fix, how to check a run, and the AdaLoRA comparison.

### Baseline

| Configuration | Accuracy (%) | Batch (micro × accum) | Notes |
|---|---|---|---|
| Zero-shot (no fine-tuning) | 61.91 | – | 7,787 / 12,578 correct; paper reports 62.0 |

### Capitalised training answers

| Configuration | Accuracy (%) | Batch (micro × accum) | Notes |
|---|---|---|---|
| Uniform LoRA (rank 8), seed 0 | 61.94 | 4 × 4 ⚠️ | 7,791 / 12,578 correct; `uniform/default/seed0` (rerun on capitalised answers) |
| Uniform LoRA (rank 8), seed 1 | 61.99 | 4 × 4 ⚠️ | 7,797 / 12,578 correct; [outputs](output/peft/uniform/default/seed1/eval/metrics.json) |
| Uniform LoRA (rank 8), seed 2 | 61.93 | 4 × 4 ⚠️ | 7,789 / 12,578 correct; [outputs](output/peft/uniform/default/seed2/eval/metrics.json) |
| **Uniform LoRA (rank 8), mean of seeds 0–2** | **61.95 ± 0.03** | 4 × 4 ⚠️ | sample std over 3 seeds; +0.04 vs zero-shot (61.91) |
| AdaLoRA (init rank 12 → avg 8), seed 0 | 62.56 | 1 × 16 | 7,869 / 12,578 correct; [outputs](output/peft/adalora_capfirst/seed0/eval/metrics.json); +1.12 vs lowercase seed 0 |
| AdaLoRA (init rank 12 → avg 8), seed 1 | 62.45 | 1 × 16 | 7,855 / 12,578 correct; [outputs](output/peft/adalora_capfirst/seed1/eval/metrics.json) |
| AdaLoRA (init rank 12 → avg 8), seed 2 | 62.47 | 1 × 16 | 7,857 / 12,578 correct; [outputs](output/peft/adalora_capfirst/seed2/eval/metrics.json) |
| **AdaLoRA, mean ± sample std of seeds 0–2** | **62.49 ± 0.06** | 1 × 16 | Calculated from exact correct counts; std in percentage points, ddof=1; [summary](output/peft/adalora_capfirst/three_seed_summary.json) |
| Flow-guided LoRA, seed 0 | 61.77 | 4 × 4 ⚠️ | 7,770 / 12,578 correct; `flow_guided/default/seed0`; log shows target 'Right</s>' |
| Flow-guided LoRA, seed 1 | 62.03 | 4 × 4 ⚠️ | 7,802 / 12,578 correct; `flow_guided/default/seed1` |
| Flow-guided LoRA, seed 2 | 61.57 | 4 × 4 ⚠️ | 7,744 / 12,578 correct; `flow_guided/default/seed2` |
| **Flow-guided LoRA, mean of seeds 0–2** | **61.79 ± 0.23** | 4 × 4 ⚠️ | sample std over 3 seeds; −0.12 vs zero-shot (61.91), within seed-to-seed variation |
| Inverse-flow LoRA, seed 0 | 62.72 | 4 × 4 ✅ | 7,889 / 12,578 correct; [outputs](output/peft/inverse_flow/default/seed0/eval/metrics.json); rerun with the batch fix; capitalized targets; rank budget 512 |
| Inverse-flow LoRA, seed 1 | 62.73 | 4 × 4 ✅ | 7,890 / 12,578 correct; [outputs](output/peft/inverse_flow/default/seed1/eval/metrics.json); rerun with the batch fix; capitalized targets; rank budget 512 |
| Inverse-flow LoRA, seed 2 | 62.79 | 4 × 4 ✅ | 7,898 / 12,578 correct; [outputs](output/peft/inverse_flow/default/seed2/eval/metrics.json); rerun with the batch fix; capitalized targets; rank budget 512 |
| **Inverse-flow LoRA, mean ± sample std of seeds 0–2** | **62.75 ± 0.04** | 4 × 4 ✅ | Calculated from exact correct counts; std in percentage points, ddof=1; +0.84 vs zero-shot (61.91) |

### Variants

Capitalised training answers, with a change to the training data or setup.

| Configuration | Accuracy (%) | Batch (micro × accum) | Notes |
|---|---|---|---|
| Uniform LoRA (rank 8), trained on mistakes, seed 0 | 61.63 | 4 × 4 ⚠️ | 7,752 / 12,578 correct; `uniform/train_mistakes_llava/seed0`; 5,753 zero-shot-wrong + 5,753 right training questions |

### Lowercase training answers

| Configuration | Accuracy (%) | Batch (micro × accum) | Notes |
|---|---|---|---|
| Uniform LoRA (rank 8), seed 0 | 61.27 | 4 × 4 ⚠️ | 7,706 / 12,578 correct; `configs/ranks/uniform.json` |
| Uniform LoRA (rank 8), seed 0, rerun | 61.36 | 4 × 4 ⚠️ | 7,718 / 12,578 correct; `uniform/default/seed0`; meant to be capitalised, but the cluster still had the lowercase training file |
| AdaLoRA (init rank 12 → avg 8), seed 0 | 61.44 | 1 × 16 | 7,728 / 12,578 correct; [outputs](output/peft/adalora/seed0/eval/metrics.json); final ranks in `adalora_seed0_ranks.csv` |
| AdaLoRA (init rank 12 → avg 8), seed 1 | 61.35 | 1 × 16 | 7,717 / 12,578 correct; [outputs](output/peft/adalora/seed1/eval/metrics.json) |
| Flow-guided LoRA, seed 0 | 61.02 | 4 × 4 ⚠️ | 7,675 / 12,578 correct; ranks averaged over 6 question types (`configs/ranks/flow_guided.json`) |
| Flow-guided LoRA, seed 0, rerun | 61.24 | 4 × 4 ⚠️ | 7,703 / 12,578 correct; same setup as 61.02 (log shows 'right</s>'); shows run-to-run variation of ~0.2 |
| Flow-guided LoRA, seed 1 | 61.07 | 4 × 4 ⚠️ | 7,681 / 12,578 correct; `flow_guided/lowercase/seed1` |
| Flow-guided LoRA, LR 1e-4, seed 0 | 61.06 | 4 × 4 ⚠️ | 7,680 / 12,578 correct; LR 2e-4 → 1e-4 made no difference (61.02) |
| Flow-protected LoRA, seed 0 | 61.54 | 4 × 4 ⚠️ | 7,740 / 12,578 correct; `configs/ranks/flow_protected.json`; no LoRA on layers 9–15 (critical flow layers), rank 10–11 elsewhere; trained before the cluster's training file was capitalised |

## ScienceQA

ScienceQA-IMG test (2,017 multiple-choice questions with a picture); trained on the 6,218 training questions with a picture
(`datasets/scienceqa/train_llava.json`). Answers are option letters (A, B, C, ...). Same LoRA budget and settings as GQA.
Run folders: `output/peft/<ranks>/scienceqa_train_llava/seed<k>/eval_scienceqa/`.

### Baseline

| Configuration | Accuracy (%) | Batch (micro × accum) | Notes |
|---|---|---|---|
| Zero-shot (no fine-tuning) | 69.16 | – | 1,395 / 2,017 correct; LLaVA-1.5 paper reports 66.8 (prompt details differ); `zero_shot/eval_scienceqa`; by subject: natural 67.41, social 72.12, language 65.91 |

### LoRA methods

| Configuration | Accuracy (%) | Batch (micro × accum) | Notes |
|---|---|---|---|
| Uniform LoRA (rank 8), seed 0 | 80.86 | 4 × 4 ✅ | 1,631 / 2,017 correct; batch fix, all 6,208 training rows checked; `uniform/scienceqa_modalities_fixed/seed0`; merged adapter (unmerged: 81.01); +11.70 vs zero-shot |
| Uniform LoRA (rank 8), seed 0, before the batch fix | 77.64 | 4 × 4 ⚠️ | 1,566 / 2,017 correct; trained on 1,552 of 6,208 rows (1 of 4 per microbatch); kept for reference |
| Uniform LoRA (rank 8), seed 1 | 81.66 | 4 × 4 ✅ | 1,647 / 2,017 correct; [outputs](output/peft/uniform/scienceqa_modalities_fixed/seed1/eval_scienceqa/metrics.json) |
| Uniform LoRA (rank 8), seed 2 | 81.11 | 4 × 4 ✅ | 1,636 / 2,017 correct; [outputs](output/peft/uniform/scienceqa_modalities_fixed/seed2/eval_scienceqa/metrics.json) |
| **Uniform LoRA, mean ± sample std of seeds 0–2** | **81.21 ± 0.41** | 4 × 4 ✅ | +12.05 vs zero-shot (69.16) |
| AdaLoRA (regularization 0.5; init rank 12 → avg 8), seed 0 | 71.74 | 1 × 16 | 1,447 / 2,017 correct |
| AdaLoRA (regularization 0.5; init rank 12 → avg 8), seed 1 | 71.10 | 1 × 16 | 1,434 / 2,017 correct |
| AdaLoRA (regularization 0.5; init rank 12 → avg 8), seed 2 | 71.24 | 1 × 16 | 1,437 / 2,017 correct |
| **AdaLoRA (regularization 0.5), mean ± sample std of seeds 0–2** | **71.36 ± 0.34** | 1 × 16 | +2.20 vs zero-shot (69.16) |
| AdaLoRA (regularization off), seed 0 | 76.05 | 1 × 16 | 1,534 / 2,017 correct; [outputs](output/peft/adalora/scienceqa_regoff_confirm/seed0/eval_test/metrics.json); selected after exploratory seed-0 analysis |
| AdaLoRA (regularization off), seed 1 | 76.50 | 1 × 16 | 1,543 / 2,017 correct; [outputs](output/peft/adalora/scienceqa_regoff_confirm/seed1/eval_test/metrics.json) |
| AdaLoRA (regularization off), seed 2 | 76.70 | 1 × 16 | 1,547 / 2,017 correct; [outputs](output/peft/adalora/scienceqa_regoff_confirm/seed2/eval_test/metrics.json) |
| **AdaLoRA (regularization off), mean ± sample std of seeds 0–2** | **76.42 ± 0.33** | 1 × 16 | Pruning remains enabled; final rank sum 512. [Exact-count summary](output/peft/adalora/scienceqa_regoff_confirm/summary.json). Post-hoc configuration; original regularization-on results preserved. |
| Flow-guided LoRA, seed 0 | 81.36 | 4 × 4 ✅ | 1,641 / 2,017 correct; `configs/ranks/flow_guided_scienceqa.json` (from the ScienceQA `train_correct` Image→Question and Question→Last graphs: rank 27–28 in layers 0–4, 5–9 in 5–16, 2–4 in 17–31); [outputs](output/peft/flow_guided_scienceqa/scienceqa_modalities_fixed/seed0/eval_scienceqa/metrics.json) |
| Flow-guided LoRA, seed 1 | 82.15 | 4 × 4 ✅ | 1,657 / 2,017 correct; [outputs](output/peft/flow_guided_scienceqa/scienceqa_modalities_fixed/seed1/eval_scienceqa/metrics.json) |
| Flow-guided LoRA, seed 2 | 80.61 | 4 × 4 ✅ | 1,626 / 2,017 correct; [outputs](output/peft/flow_guided_scienceqa/scienceqa_modalities_fixed/seed2/eval_scienceqa/metrics.json) |
| **Flow-guided LoRA, mean ± sample std of seeds 0–2** | **81.37 ± 0.77** | 4 × 4 ✅ | +0.16 vs uniform (81.21 ± 0.41), within seed-to-seed variation; paired over the 3 seeds, flow-guided alone right on 98 questions vs uniform alone on 88 (exact McNemar p = 0.51) |
| Inverse-flow LoRA, seed 0 | 79.13 | 4 × 4 ✅ | 1,596 / 2,017 correct; `configs/ranks/inverse_flow_scienceqa.json` (`make_ranks.py --inverse`: same ranks, the least important layer gets the largest); [outputs](output/peft/inverse_flow_scienceqa/scienceqa_train_llava/seed0/eval_scienceqa/metrics.json) |
| Inverse-flow LoRA, seed 1 | 79.08 | 4 × 4 ✅ | 1,595 / 2,017 correct; [outputs](output/peft/inverse_flow_scienceqa/scienceqa_train_llava/seed1/eval_scienceqa/metrics.json) |
| Inverse-flow LoRA, seed 2 | 79.42 | 4 × 4 ✅ | 1,602 / 2,017 correct; [outputs](output/peft/inverse_flow_scienceqa/scienceqa_train_llava/seed2/eval_scienceqa/metrics.json) |
| **Inverse-flow LoRA, mean ± sample std of seeds 0–2** | **79.21 ± 0.19** | 4 × 4 ✅ | −2.00 vs uniform (81.21 ± 0.41), −2.16 vs flow-guided; worse on every seed (McNemar p < 0.001 vs uniform each seed; paired over the 3 seeds, inverse alone right on 61 questions vs uniform alone on 182); largest drops in grades 4–8 and social science |

### Pruned LoRA (half the parameters)

LoRA rank 8 on 16 of the 32 layers and none on the others: 2,097,152 trainable parameters (half of the 4,194,304 above).
The 16 layers come from the same ScienceQA `train_correct` Image→Question and Question→Last importance
(`make_ranks.py --keep 16`, with `--inverse` or `--random 0`). Seed 0 only; McNemar tests are paired on the same questions.

| Configuration | Layers with LoRA | Accuracy (%) | Batch (micro × accum) | Notes |
|---|---|---|---|---|
| Flow-pruned LoRA, seed 0 | 0–15 (most important) | **81.11** | 4 × 4 ✅ | 1,636 / 2,017 correct; `configs/ranks/flow_pruned16_scienceqa.json`; same as uniform seed 0 (80.86; 21 vs 16 questions, p = 0.51); final loss 0.237 (uniform 0.239); [outputs](output/peft/flow_pruned16_scienceqa/scienceqa_train_llava/seed0/eval_scienceqa/metrics.json) |
| Random-pruned LoRA, seed 0 | 1, 4, 6, 8, 9, 11, 12, 13, 15, 16, 18, 24, 25, 28, 30, 31 (9 shared with flow) | 79.38 | 4 × 4 ✅ | 1,601 / 2,017 correct; `configs/ranks/random_pruned16_scienceqa.json`; −1.73 vs flow-pruned (56 vs 21 questions, p = 8e-5); final loss 0.246; [outputs](output/peft/random_pruned16_scienceqa/scienceqa_train_llava/seed0/eval_scienceqa/metrics.json) |
| Inverse-pruned LoRA, seed 0 | 16–31 (least important) | 72.43 | 4 × 4 ✅ | 1,461 / 2,017 correct; `configs/ranks/inverse_pruned16_scienceqa.json`; −8.68 vs flow-pruned (228 vs 53 questions, p = 5e-27), only +3.27 over zero-shot; final loss 0.346; natural science 68.7, social 78.1; [outputs](output/peft/inverse_pruned16_scienceqa/scienceqa_train_llava/seed0/eval_scienceqa/metrics.json) |

Ordered flow > random > inverse, matching the knockout importance. With half the parameters, flow-pruned matches
full uniform LoRA. Same story as the full-budget runs (inverse-flow 79.21 vs uniform 81.21): the early flow layers must
be tuned, the late ones add little. Seeds 1–2 are needed before this is final.

### AdaLoRA diagnostics (seed 0)

Why AdaLoRA trails uniform on ScienceQA: each of its two extra parts switched off in turn, same merged evaluation.
Regularization-off seeds 1–2 are complete. The setting was chosen after inspecting seed 0 test results (validation showed the same pattern); the three-seed results above are explicitly labelled as a post-hoc configuration, not an untouched-test confirmatory study. All three planned diagnostic conditions are complete; no further diagnostic training is queued. [Full diagnostic summary](output/ablations/adalora/scienceqa_summary.json).

| Orthogonal regularization (weight 0.5) | Rank pruning (12 → avg 8) | Accuracy (%) |
|---|---|---|
| on | on (original) | 71.74 |
| off | on | 76.05 |
| on | off (rank 12 kept, larger budget) | 71.59 |
| off | off | 76.35 |
| Uniform LoRA in the same training loop as AdaLoRA (1 × 16, seed 0) | – | 81.21 |

Turning the regularization off gains 4.3–4.8 points; turning pruning off changes −0.15 to +0.30. About 5 points to uniform remain unexplained.

## AdaLoRA reproducibility and report notes

- [Frozen training implementation and TC2 instructions](experiments/adalora_reproduction/README.md): the exact source used for these runs, added without replacing the shared trainer.
- [AdaLoRA report and presentation notes](docs/adalora/report_notes.md): method, parameters, results, ranks, compute and limitations.
- Final per-layer q/v ranks and masks are stored with each capitalised run. AdaLoRA does not consume information-flow graphs.

### Matched-loop Uniform control (ScienceQA, seed 0)

Uniform rank 8 through the same engine/microbatch 1 × accumulation 16 as AdaLoRA scored **81.21% (1638/2017) merged** and **81.31% (1640/2017) unmerged**. All 6208 training rows reached the language model and fresh adapter reload passed. Training took 66.2 minutes. This is a single-seed pipeline control, separate from the 4 × 4 Uniform three-seed benchmark above. [Artifacts](output/peft/uniform/scienceqa_matched_engine/job46525/seed0/eval_scienceqa/metrics.json).

The seed-0 ablation improved merged test accuracy by 4.31–4.76 pp when removing regularization; disabling pruning alone changed scores by −0.15 to +0.30 pp. Fixed-rank-12 diagnostic arms use a larger final budget. These observations implicate the original regularization setting in part of the deficit but do not fully explain the remaining gap to Uniform, establish statistical significance, or generalize beyond the tested settings.
