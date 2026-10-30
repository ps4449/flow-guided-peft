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
| Uniform LoRA (rank 8), seed 0 | | 4 × 4 ✅ | running (rerun with the batch fix) |
| Uniform LoRA (rank 8), seed 1 | | 4 × 4 ✅ | |
| Uniform LoRA (rank 8), seed 2 | | 4 × 4 ✅ | |
| **Uniform LoRA (rank 8), mean ± sample std of seeds 0–2** | | 4 × 4 ✅ | |
| AdaLoRA (init rank 12 → avg 8), seed 0 | 62.56 | 1 × 16 | 7,869 / 12,578 correct; [outputs](output/peft/adalora_capfirst/seed0/eval/metrics.json); +1.12 vs lowercase seed 0 |
| AdaLoRA (init rank 12 → avg 8), seed 1 | 62.45 | 1 × 16 | 7,855 / 12,578 correct; [outputs](output/peft/adalora_capfirst/seed1/eval/metrics.json) |
| AdaLoRA (init rank 12 → avg 8), seed 2 | 62.47 | 1 × 16 | 7,857 / 12,578 correct; [outputs](output/peft/adalora_capfirst/seed2/eval/metrics.json) |
| **AdaLoRA, mean ± sample std of seeds 0–2** | **62.49 ± 0.06** | 1 × 16 | Calculated from exact correct counts; std in percentage points, ddof=1; [summary](output/peft/adalora_capfirst/three_seed_summary.json) |
| Flow-guided LoRA, seed 0 | 62.38 | 4 × 4 ✅ | 7,846 / 12,578 correct; `configs/ranks/flow_guided.json`; [outputs](output/peft/flow_guided/modalities_fixed/seed0/eval/metrics.json) |
| Flow-guided LoRA, seed 1 | 62.67 | 4 × 4 ✅ | 7,883 / 12,578 correct; [outputs](output/peft/flow_guided/modalities_fixed/seed1/eval/metrics.json) |
| Flow-guided LoRA, seed 2 | 62.82 | 4 × 4 ✅ | 7,901 / 12,578 correct; [outputs](output/peft/flow_guided/modalities_fixed/seed2/eval/metrics.json) |
| **Flow-guided LoRA, mean ± sample std of seeds 0–2** | **62.62 ± 0.22** | 4 × 4 ✅ | +0.71 vs zero-shot (61.91; better on every seed, McNemar p = 0.04 / 7e-4 / 4e-5); −0.13 vs inverse-flow (paired over the 3 seeds, flow alone right on 543 questions vs inverse alone on 590, not significant); +0.13 vs AdaLoRA (883 vs 834, not significant); +0.83 vs the pre-fix runs (61.79) |
| Inverse-flow LoRA, seed 0 | 62.72 | 4 × 4 ✅ | 7,889 / 12,578 correct; [outputs](output/peft/inverse_flow/default/seed0/eval/metrics.json); rerun with the batch fix; capitalized targets; rank budget 512 |
| Inverse-flow LoRA, seed 1 | 62.73 | 4 × 4 ✅ | 7,890 / 12,578 correct; [outputs](output/peft/inverse_flow/default/seed1/eval/metrics.json); rerun with the batch fix; capitalized targets; rank budget 512 |
| Inverse-flow LoRA, seed 2 | 62.79 | 4 × 4 ✅ | 7,898 / 12,578 correct; [outputs](output/peft/inverse_flow/default/seed2/eval/metrics.json); rerun with the batch fix; capitalized targets; rank budget 512 |
| **Inverse-flow LoRA, mean ± sample std of seeds 0–2** | **62.75 ± 0.04** | 4 × 4 ✅ | Calculated from exact correct counts; std in percentage points, ddof=1; +0.84 vs zero-shot (61.91) |

### Variants: before the batch fix

Capitalised training answers, trained before the batch fix (⚠️: may have learned from ~1 of every 4 training questions).
Kept for reference; the table above replaces them.

| Configuration | Accuracy (%) | Batch (micro × accum) | Notes |
|---|---|---|---|
| Uniform LoRA (rank 8), seed 0 | 61.94 | 4 × 4 ⚠️ | 7,791 / 12,578 correct; `uniform/default/seed0` (rerun on capitalised answers) |
| Uniform LoRA (rank 8), seed 1 | 61.99 | 4 × 4 ⚠️ | 7,797 / 12,578 correct; [outputs](output/peft/uniform/default/seed1/eval/metrics.json) |
| Uniform LoRA (rank 8), seed 2 | 61.93 | 4 × 4 ⚠️ | 7,789 / 12,578 correct; [outputs](output/peft/uniform/default/seed2/eval/metrics.json) |
| **Uniform LoRA (rank 8), mean of seeds 0–2** | **61.95 ± 0.03** | 4 × 4 ⚠️ | sample std over 3 seeds; +0.04 vs zero-shot (61.91) |
| Flow-guided LoRA, seed 0 | 61.77 | 4 × 4 ⚠️ | 7,770 / 12,578 correct; `flow_guided/default/seed0`; log shows target 'Right</s>' |
| Flow-guided LoRA, seed 1 | 62.03 | 4 × 4 ⚠️ | 7,802 / 12,578 correct; `flow_guided/default/seed1` |
| Flow-guided LoRA, seed 2 | 61.57 | 4 × 4 ⚠️ | 7,744 / 12,578 correct; `flow_guided/default/seed2` |
| **Flow-guided LoRA, mean of seeds 0–2** | **61.79 ± 0.23** | 4 × 4 ⚠️ | sample std over 3 seeds; −0.12 vs zero-shot (61.91), within seed-to-seed variation |
| Inverse-flow LoRA, seed 0 | 62.13 | 4 × 4 ⚠️ | 7,815 / 12,578 correct; [outputs](output/peft/inverse_flow/seed0/eval/metrics.json); rank budget 512 |
| Inverse-flow LoRA, seed 1 | 62.16 | 4 × 4 ⚠️ | 7,819 / 12,578 correct; outputs overwritten by the fixed rerun in `inverse_flow/default/seed1` (in git history before 8113224) |
| Inverse-flow LoRA, seed 2 | 62.16 | 4 × 4 ⚠️ | 7,819 / 12,578 correct; outputs overwritten by the fixed rerun in `inverse_flow/default/seed2` (in git history before 8113224) |
| **Inverse-flow LoRA, mean of seeds 0–2** | **62.15 ± 0.02** | 4 × 4 ⚠️ | sample std over 3 seeds |
| Uniform LoRA (rank 8), trained on mistakes, seed 0 | 61.63 | 4 × 4 ⚠️ | 7,752 / 12,578 correct; `uniform/train_mistakes_llava/seed0`; 5,753 zero-shot-wrong + 5,753 right training questions |

### Lowercase training answers

| Configuration | Accuracy (%) | Batch (micro × accum) | Notes |
|---|---|---|---|
| Uniform LoRA (rank 8), seed 0 | 61.27 | 4 × 4 ⚠️ | 7,706 / 12,578 correct; `configs/ranks/uniform.json` |
| Uniform LoRA (rank 8), seed 0, rerun | 61.36 | 4 × 4 ⚠️ | 7,718 / 12,578 correct; `uniform/default/seed0`; meant to be capitalised, but the cluster still had the lowercase training file |
| AdaLoRA (init rank 12 → avg 8), seed 0 | 61.44 | 1 × 16 | 7,728 / 12,578 correct; [outputs](output/peft/adalora/seed0/eval/metrics.json); final ranks in [`configs/adalora_seed0_ranks_gqa.csv`](configs/adalora_seed0_ranks_gqa.csv) |
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
| AdaLoRA (regularization 0.5; init rank 12 → avg 8), seed 0 | 71.74 | 1 × 16 | 1,447 / 2,017 correct; [outputs](output/peft/adalora/scienceqa_train_llava/seed0/eval_scienceqa/metrics.json) |
| AdaLoRA (regularization 0.5; init rank 12 → avg 8), seed 1 | 71.10 | 1 × 16 | 1,434 / 2,017 correct; [outputs](output/peft/adalora/scienceqa_train_llava/seed1/eval_scienceqa/metrics.json) |
| AdaLoRA (regularization 0.5; init rank 12 → avg 8), seed 2 | 71.24 | 1 × 16 | 1,437 / 2,017 correct; [outputs](output/peft/adalora/scienceqa_train_llava/seed2/eval_scienceqa/metrics.json) |
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
(`make_ranks.py --keep 16`, with `--inverse` or `--random 0`; the random layers are the same for every seed).
McNemar tests are paired on the same questions and seed; "pooled" adds the counts over seeds 0–2.

| Configuration | Layers with LoRA | Accuracy (%) | Batch (micro × accum) | Notes |
|---|---|---|---|---|
| Flow-pruned LoRA, seed 0 | 0–15 (most important) | 81.11 | 4 × 4 ✅ | 1,636 / 2,017 correct; `configs/ranks/flow_pruned16_scienceqa.json`; same as uniform seed 0 (80.86; 21 vs 16 questions, p = 0.51); final loss 0.237 (uniform 0.239); [outputs](output/peft/flow_pruned16_scienceqa/scienceqa_train_llava/seed0/eval_scienceqa/metrics.json) |
| Flow-pruned LoRA, seed 1 | 0–15 | 81.71 | 4 × 4 ✅ | 1,648 / 2,017 correct; [outputs](output/peft/flow_pruned16_scienceqa/scienceqa_train_llava/seed1/eval_scienceqa/metrics.json) |
| Flow-pruned LoRA, seed 2 | 0–15 | 81.26 | 4 × 4 ✅ | 1,639 / 2,017 correct; [outputs](output/peft/flow_pruned16_scienceqa/scienceqa_train_llava/seed2/eval_scienceqa/metrics.json) |
| **Flow-pruned LoRA, mean ± sample std of seeds 0–2** | 0–15 | **81.36 ± 0.31** | 4 × 4 ✅ | +0.15 vs uniform with half the parameters (81.21 ± 0.41; pooled 64 vs 55 questions, not significant); same as flow-guided at full budget (81.37) |
| Random-pruned LoRA, seed 0 | 1, 4, 6, 8, 9, 11, 12, 13, 15, 16, 18, 24, 25, 28, 30, 31 (9 shared with flow) | 79.38 | 4 × 4 ✅ | 1,601 / 2,017 correct; `configs/ranks/random_pruned16_scienceqa.json`; −1.73 vs flow-pruned (56 vs 21 questions, p = 8e-5); final loss 0.246; [outputs](output/peft/random_pruned16_scienceqa/scienceqa_train_llava/seed0/eval_scienceqa/metrics.json) |
| Random-pruned LoRA, seed 1 | same as seed 0 | 80.61 | 4 × 4 ✅ | 1,626 / 2,017 correct; [outputs](output/peft/random_pruned16_scienceqa/scienceqa_train_llava/seed1/eval_scienceqa/metrics.json) |
| Random-pruned LoRA, seed 2 | same as seed 0 | 79.47 | 4 × 4 ✅ | 1,603 / 2,017 correct; [outputs](output/peft/random_pruned16_scienceqa/scienceqa_train_llava/seed2/eval_scienceqa/metrics.json) |
| **Random-pruned LoRA, mean ± sample std of seeds 0–2** | same as seed 0 | **79.82 ± 0.69** | 4 × 4 ✅ | −1.54 vs flow-pruned (worse on every seed, p = 8e-5 / 0.02 / 7e-5; pooled 72 vs 165 questions); −1.39 vs uniform (worse on every seed; pooled 66 vs 150) |
| Inverse-pruned LoRA, seed 0 | 16–31 (least important) | 72.43 | 4 × 4 ✅ | 1,461 / 2,017 correct; `configs/ranks/inverse_pruned16_scienceqa.json`; −8.68 vs flow-pruned (228 vs 53 questions, p = 5e-27), only +3.27 over zero-shot; final loss 0.346; natural science 68.7, social 78.1; [outputs](output/peft/inverse_pruned16_scienceqa/scienceqa_train_llava/seed0/eval_scienceqa/metrics.json) |
| Inverse-pruned LoRA, seed 1 | 16–31 | 72.88 | 4 × 4 ✅ | 1,470 / 2,017 correct; [outputs](output/peft/inverse_pruned16_scienceqa/scienceqa_train_llava/seed1/eval_scienceqa/metrics.json) |
| Inverse-pruned LoRA, seed 2 | 16–31 | 73.18 | 4 × 4 ✅ | 1,476 / 2,017 correct; [outputs](output/peft/inverse_pruned16_scienceqa/scienceqa_train_llava/seed2/eval_scienceqa/metrics.json) |
| **Inverse-pruned LoRA, mean ± sample std of seeds 0–2** | 16–31 | **72.83 ± 0.37** | 4 × 4 ✅ | −8.53 vs flow-pruned (worse on every seed, p < 1e-22; pooled 187 vs 703 questions); +3.67 over zero-shot |

Ordered flow > random > inverse on every seed, matching the knockout importance. With half the parameters, flow-pruned
matches full uniform LoRA, while a random half loses 1.4 and the opposite half loses 8.4. Same story as the full-budget
runs (inverse-flow 79.21 vs uniform 81.21): the early flow layers must be tuned, the late ones add little.

#### Budget-matched uniform rank-4 control (30 October 2026)

Uniform rank 4 on q_proj/v_proj in all 32 layers uses **2,097,152 trainable parameters**, the same as rank 8 on 16 layers. These are new corrected-batch runs from source commit `1138b73`, one epoch and microbatch 4 × accumulation 4, evaluated with merged adapters on all 2,017 ScienceQA image test questions.

| Configuration | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD (%) |
|---|---:|---:|---:|---:|
| Uniform rank 4, all 32 layers | 80.12 (1616) | 81.06 (1635) | 80.37 (1621) | **80.52 ± 0.49** |
| Flow-pruned rank 8, layers 0–15 | 81.11 (1636) | 81.71 (1648) | 81.26 (1639) | **81.36 ± 0.31** |

Parentheses give correct counts out of 2,017. The flow-pruned control is higher on all three seeds, by **0.84 percentage points on average** (calculated from exact counts). This supports concentrating the budget in these selected layers rather than uniformly spreading a smaller rank in this setting. It does **not** isolate a unique benefit of flow: the selected 16 layers are exactly the first half of the model. No formal significance or equivalence conclusion is asserted here.

[Exact-count summary](output/peft/uniform_rank4/scienceqa_20261030/summary.json); per-seed predictions, metrics, training configurations, training logs, and validation records are alongside it. [Run provenance](docs/UNIFORM_RANK4_SCIENCEQA.md). SLURM job 49074 completed with exit 0 in 01:46:11 on TC2N04; this timing must not be used as a hardware-matched speed comparison against older runs.

#### Fewer layers (seed 0)

The top 8 and top 4 layers by the same importance (`make_ranks.py --keep 8` / `--keep 4`), rank 8 each.

| Configuration | Layers with LoRA | Parameters | Accuracy (%) | Batch (micro × accum) | Notes |
|---|---|---|---|---|---|
| Flow-pruned LoRA, 16 layers, seed 0 | 0–15 | 2,097,152 | 81.11 | 4 × 4 ✅ | from the table above |
| Flow-pruned LoRA, 8 layers, seed 0 | 0, 1, 2, 3, 4, 9, 10, 11 | 1,048,576 | 79.57 | 4 × 4 ✅ | 1,605 / 2,017 correct; `configs/ranks/flow_pruned8_scienceqa.json`; −1.54 vs 16 layers (35 vs 66 questions, p = 0.003), −1.29 vs uniform seed 0 (p = 0.01); same as random-pruned 16 layers seed 0 (79.38, p = 0.75) with half its parameters; final loss 0.239; [outputs](output/peft/flow_pruned8_scienceqa/scienceqa_train_llava/seed0/eval_scienceqa/metrics.json) |
| Flow-pruned LoRA, 4 layers, seed 0 | 0, 2, 3, 4 | 524,288 | 77.99 | 4 × 4 ✅ | 1,573 / 2,017 correct; `configs/ranks/flow_pruned4_scienceqa.json`; −3.12 vs 16 layers (45 vs 108 questions, p = 4e-7); +8.83 over zero-shot and +5.56 over inverse-pruned 16 layers seed 0, with 1/4 of its parameters; final loss 0.258; [outputs](output/peft/flow_pruned4_scienceqa/scienceqa_train_llava/seed0/eval_scienceqa/metrics.json) |

Accuracy rises with the number of flow layers and reaches full LoRA at 16; fewer layers lose accuracy, but flow layers
still give more per parameter than random or late layers.

#### Top-8 layer-placement ablations (Experiments #4 and #5)

All methods use 1,048,576 trainable adapter parameters and matched ScienceQA training/evaluation on the same 2,017 image test questions; values are accuracy percentages, with mean ± sample SD over seeds 0–2 where available.

| Method (layers) | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD (%) |
|---|---:|---:|---:|---:|
| [Combined flow (0–4, 9–11)](configs/ranks/flow_pruned8_scienceqa.json) | [79.57](output/peft/flow_pruned8_scienceqa/scienceqa_train_llava/seed0/eval_scienceqa/metrics.json) | [80.12](output/peft/flow_pruned8_scienceqa/scienceqa_train_llava/seed1/eval_scienceqa/metrics.json) | [79.92](output/peft/flow_pruned8_scienceqa/scienceqa_train_llava/seed2/eval_scienceqa/metrics.json) | 79.87 ± 0.28 |
| [Question→Last (9–16)](configs/ranks/question_last_pruned8_scienceqa.json) | [77.94](output/peft/question_last_pruned8_scienceqa/scienceqa_train_llava/seed0/eval_scienceqa/metrics.json) | [79.03](output/peft/question_last_pruned8_scienceqa/scienceqa_train_llava/seed1/eval_scienceqa/metrics.json) | [79.57](output/peft/question_last_pruned8_scienceqa/scienceqa_train_llava/seed2/eval_scienceqa/metrics.json) | 78.85 ± 0.83 |
| [Fixed middle (12–19)](configs/ranks/middle_pruned8_scienceqa.json) | [75.46](output/peft/middle_pruned8_scienceqa/scienceqa_train_llava/seed0/eval_scienceqa/metrics.json) | [76.25](output/peft/middle_pruned8_scienceqa/scienceqa_train_llava/seed1/eval_scienceqa/metrics.json) | [76.65](output/peft/middle_pruned8_scienceqa/scienceqa_train_llava/seed2/eval_scienceqa/metrics.json) | 76.12 ± 0.61 |
| [Fixed early (0–7)](configs/ranks/early_pruned8_scienceqa.json) | [79.57](output/peft/early_pruned8_scienceqa/scienceqa_train_llava/seed0/eval_scienceqa/metrics.json) | [79.57](output/peft/early_pruned8_scienceqa/scienceqa_train_llava/seed1/eval_scienceqa/metrics.json) | [79.18](output/peft/early_pruned8_scienceqa/scienceqa_train_llava/seed2/eval_scienceqa/metrics.json) | 79.44 ± 0.23 |

Combined-flow selection outperformed Question→Last and fixed middle placement in all three matched seeds. Fixed early-layer selection matched combined flow in seed 0 and trailed it in seeds 1 and 2, giving combined flow a 0.43-point higher three-seed mean; this small comparison does not by itself establish a unique advantage over a simple early-layer rule.

#### Other ways to choose the 16 layers (seed 0)

Layer-selection methods from the literature, scored without the knockout by
`peft_lora/methods/layer_baselines/make_ranks.py` (scores in `output/layer_baselines/scienceqa/scores.json`;
grad and similarity use 512 random training questions), then rank 8 on the top 16 layers as above.

| Configuration | Layers with LoRA | Accuracy (%) | Batch (micro × accum) | Notes |
|---|---|---|---|---|
| Gradient size (IST, EMNLP'24 Findings; LISA, NeurIPS'24) | 0–15 | = flow-pruned (81.36 ± 0.31) | — | not trained: `configs/ranks/grad_pruned16_scienceqa.json` is identical to `flow_pruned16_scienceqa.json`; mean ‖grad‖ of q/v is highest in layers 0–11 and below 1 in layers 21–30 |
| Weight spectrum (AlphaLoRA, EMNLP'24) | 16–31 | = inverse-pruned (72.83 ± 0.37) | — | not trained: `configs/ranks/spectrum_pruned16_scienceqa.json` is identical to `inverse_pruned16_scienceqa.json`; the heavy-tail α rises with depth (1.49 → 1.99), so it picks the deep layers |
| Layer similarity (Gromov et al., ICLR'25), seed 0 | 0–8, 13–17, 30, 31 (12 shared with flow) | 80.61 | 4 × 4 ✅ | 1,626 / 2,017 correct; `configs/ranks/similarity_pruned16_scienceqa.json`; same as flow-pruned seed 0 (81.11; 24 vs 34 questions, p = 0.24) and uniform seed 0 (80.86; 30 vs 35, p = 0.62); +1.23 vs random-pruned seed 0 (52 vs 27, p = 0.007); final loss 0.252; [outputs](output/peft/similarity_pruned16_scienceqa/scienceqa_train_llava/seed0/eval_scienceqa/metrics.json) |

The methods that pick early layers (flow, gradient size, similarity) all match full LoRA; the one that picks the deep
layers (spectrum) fails like inverse-pruned. Flow ties gradient size by construction (same layers) and similarity on one seed.

#### LoRA variants on the flow layers

Same 16 flow layers and rank 8 (`flow_pruned16_scienceqa.json`), with the adapter changed by `train_lora.py --variant`
(or the learning rate by `--lr`). The uniform rows use all 32 layers (`uniform.json`) as the control.

| Configuration | Parameters | Accuracy (%) | Batch (micro × accum) | Notes |
|---|---|---|---|---|
| Flow-pruned LoRA, seed 0 | 2,097,152 | 81.11 | 4 × 4 ✅ | from the table above |
| Flow-pruned DoRA (ICML'24), seed 0 | 2,228,224 | 81.26 | 4 × 4 ✅ | 1,639 / 2,017 correct; same as plain LoRA (12 vs 9 questions, p = 0.66) and uniform seed 0 (80.86; 24 vs 16, p = 0.27); final loss 0.240 (LoRA 0.237); 75 min training vs 57; evaluated with the merged weights cast back to bf16 (peft 0.12 leaves DoRA merges in float32); [outputs](output/peft/flow_pruned16_scienceqa/scienceqa_train_llava_dora/seed0/eval_scienceqa/metrics.json) |
| Flow-pruned PiSSA (NeurIPS'24), seed 0 | 2,097,152 | 83.19 | 4 × 4 ✅ | 1,678 / 2,017 correct; +2.08 vs plain LoRA (76 vs 34 questions, p = 8e-5); +2.33 vs uniform seed 0 (78 vs 31, p = 8e-6) with half the parameters; gains mostly in grades 6–8 (grade 7: 72.2 → 77.9); mean loss over steps 300–388 0.202 (LoRA 0.204); saved converted to a plain rank-16 LoRA; resumed from step 200 after a full-disk crash; [outputs](output/peft/flow_pruned16_scienceqa/scienceqa_train_llava_pissa/seed0/eval_scienceqa/metrics.json) |
| Flow-pruned LoRA+ (ICML'24), seed 0 | 2,097,152 | 83.59 | 4 × 4 ✅ | 1,686 / 2,017 correct; B learns 16× faster than A; +2.48 vs plain LoRA (92 vs 42 questions, p = 2e-5); +2.73 vs uniform seed 0 (94 vs 39, p = 2e-6); same as PiSSA (54 vs 46, p = 0.48); mean loss over steps 300–388 0.186 (LoRA 0.204); 57 min training; [outputs](output/peft/flow_pruned16_scienceqa/scienceqa_train_llava_loraplus/seed0/eval_scienceqa/metrics.json) |
| Uniform PiSSA (all 32 layers), seed 0 | 4,194,304 | 82.00 | 4 × 4 ✅ | 1,654 / 2,017 correct; −1.19 vs flow-pruned PiSSA with twice the parameters (43 vs 67 questions, p = 0.03); +1.14 vs uniform LoRA seed 0 (76 vs 53, p = 0.05); flow-pruned PiSSA is ahead mostly in grades 7–8 (77.9 vs 73.3, 75.8 vs 73.8) and language science (81.8 vs 77.3); mean loss over steps 300–388 0.202; 63 min training; [outputs](output/peft/uniform/scienceqa_train_llava_pissa/seed0/eval_scienceqa/metrics.json) |
| Flow-pruned PiSSA, seed 1 | 2,097,152 | 83.04 | 4 × 4 ✅ | 1,675 / 2,017 correct; [outputs](output/peft/flow_pruned16_scienceqa/scienceqa_train_llava_pissa/seed1/eval_scienceqa/metrics.json) |
| Flow-pruned PiSSA, seed 2 | 2,097,152 | 82.94 | 4 × 4 ✅ | 1,673 / 2,017 correct; [outputs](output/peft/flow_pruned16_scienceqa/scienceqa_train_llava_pissa/seed2/eval_scienceqa/metrics.json) |
| **Flow-pruned PiSSA, mean ± sample std of seeds 0–2** | 2,097,152 | **83.06 ± 0.12** | 4 × 4 ✅ | +1.70 vs flow-pruned LoRA (81.36 ± 0.31; better on every seed, pooled 270 vs 167 questions, p = 1e-6); +1.85 vs uniform LoRA (81.21 ± 0.41; pooled 281 vs 169, p = 2e-7) |
| Uniform PiSSA, seed 1 | 4,194,304 | 83.49 | 4 × 4 ✅ | 1,684 / 2,017 correct; [outputs](output/peft/uniform/scienceqa_train_llava_pissa/seed1/eval_scienceqa/metrics.json) |
| Uniform PiSSA, seed 2 | 4,194,304 | 81.61 | 4 × 4 ✅ | 1,646 / 2,017 correct; [outputs](output/peft/uniform/scienceqa_train_llava_pissa/seed2/eval_scienceqa/metrics.json) |
| **Uniform PiSSA, mean ± sample std of seeds 0–2** | 4,194,304 | **82.37 ± 0.99** | 4 × 4 ✅ | −0.69 vs flow-pruned PiSSA with twice the parameters (flow ahead on seeds 0 and 2, p = 0.03 / 0.05; behind on seed 1, 41 vs 50, p = 0.40; pooled 171 vs 213, p = 0.04); +1.16 vs uniform LoRA (pooled 243 vs 173, p = 7e-4) |
| Uniform LoRA+, seed 0 | 4,194,304 | 82.75 | 4 × 4 ✅ | 1,669 / 2,017 correct; −0.84 vs flow-pruned LoRA+ (64 vs 81 questions, p = 0.18, not significant); +1.89 vs uniform LoRA seed 0 (92 vs 54, p = 0.002); mean loss over steps 300–388 0.187; 60 min training; [outputs](output/peft/uniform/scienceqa_train_llava_loraplus/seed0/eval_scienceqa/metrics.json) |
| Flow-pruned LoRA, lr 8e-4 (4×), seed 0 | 2,097,152 | 81.41 | 4 × 4 ✅ | 1,642 / 2,017 correct; same as lr 2e-4 (82 vs 76 questions, p = 0.69); below PiSSA (52 vs 88, p = 0.003) and LoRA+ (41 vs 85, p = 1e-4); mean loss over steps 300–388 0.192 (lr 2e-4: 0.204); [outputs](output/peft/flow_pruned16_scienceqa/lr8e-4_scienceqa_train_llava/seed0/eval_scienceqa/metrics.json) |

PiSSA and LoRA+ add about 2–2.5 points on the flow layers, while DoRA adds nothing. The gain is not just a larger step:
plain LoRA at 4× the learning rate stays at 81.4. Over three seeds, flow-pruned PiSSA (83.06 ± 0.12) matches or beats
uniform PiSSA (82.37 ± 0.99) with half the parameters and is far more stable across seeds; uniform PiSSA has the single
best seed (83.49). LoRA+ shows the same direction on seed 0 (83.59 vs 82.75) but is not significant on one seed.

### AdaLoRA diagnostics (seed 0)

Why AdaLoRA trails uniform on ScienceQA: each of its two extra parts switched off in turn, same merged evaluation.
The regularization-off setting above was chosen from these seed-0 test results (post-hoc). Configs:
`AdaLoRA-experiments/scienceqa_adalora/configs/`; run outputs are in the Git tag `archive/nop-experiments-2026-10-26`.

| Orthogonal regularization (weight 0.5) | Rank pruning (12 → avg 8) | Accuracy (%) |
|---|---|---|
| on | on (original) | 71.74 |
| off | on | 76.05 |
| on | off (rank 12 kept, larger budget) | 71.59 |
| off | off | 76.35 |
| Uniform LoRA in the same training loop as AdaLoRA (1 × 16, seed 0) | – | 81.21 |

Turning the regularization off gains 4.3–4.8 points; turning pruning off changes −0.15 to +0.30. About 5 points to uniform remain unexplained.

How to rerun AdaLoRA (GQA and ScienceQA): [AdaLoRA-experiments/README.md](AdaLoRA-experiments/README.md). Final per-layer q/v ranks of the
capitalised GQA runs: `output/peft/adalora_capfirst/seed<k>/final_ranks.csv`.
