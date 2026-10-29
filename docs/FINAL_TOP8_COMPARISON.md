# Final ScienceQA top-8 comparison

**Prespecified primary comparison: combined-flow versus fixed early layers 0–7.**

All four fixed checkpoints use seed 0, rank-8 q_proj/v_proj LoRA, alpha 16, dropout 0.05, and 1,048,576 trainable parameters.
All 32 base layers remain. The matched protocol uses the original LLaVA-1.5-7B, corrected multimodal batching, uppercase option-letter answers,
vicuna_v1, one epoch, batch 4 × accumulation 4, LR 2e-4, 3% warmup, bf16 AdamW and cosine scheduling for 388 updates.

| Fixed checkpoint | Layers | Correct / total | Accuracy | Trainer minutes |
| --- | --- | --- | --- | --- |
| Combined-flow | [0, 1, 2, 3, 4, 9, 10, 11] | 1605/2017 | 79.57% | 55.6 |
| Question→Last | [9, 10, 11, 12, 13, 14, 15, 16] | 1572/2017 | 77.94% | 55.7 |
| Fixed middle | [12, 13, 14, 15, 16, 17, 18, 19] | 1522/2017 | 75.46% | 55.3 |
| Fixed early | [0, 1, 2, 3, 4, 5, 6, 7] | 1605/2017 | 79.57% | 55.5 |

## Primary paired result

Combined − early: **+0.000 percentage points**, 95% paired-question bootstrap interval **[-0.892, +0.892]**; exact two-sided McNemar p = **1**.
Both correct: **1563**; both wrong: **370**; Combined-only correct: **42**; early-only correct: **42**.

## All paired contrasts

Positive differences favor A. Primary contrast was declared before early training; the others are secondary/exploratory.

| A − B | Both correct | Both wrong | A-only | B-only | Δ pp [95% CI] | Exact McNemar p | Holm p (all six) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| combined − early | 1563 | 370 | 42 | 42 | +0.000 [-0.892, +0.892] | 1 | 1 |
| combined − question_last | 1513 | 353 | 92 | 59 | +1.636 [+0.446, +2.876] | 0.00898662 | 0.0269598 |
| combined − middle | 1461 | 351 | 144 | 61 | +4.115 [+2.727, +5.553] | 6.40493e-09 | 3.84296e-08 |
| question_last − middle | 1485 | 408 | 87 | 37 | +2.479 [+1.388, +3.570] | 8.27416e-06 | 3.30966e-05 |
| question_last − early | 1510 | 350 | 62 | 95 | -1.636 [-2.876, -0.397] | 0.010423 | 0.0269598 |
| middle − early | 1461 | 351 | 61 | 144 | -4.115 [-5.503, -2.727] | 6.40493e-09 | 3.84296e-08 |

Predictions are aligned by the same 2,017 unique question IDs and exact gold labels. The original answer normalization is unchanged.
Intervals use 20,000 paired question resamples with seed 0 and are marginal 95% percentile intervals, not simultaneous intervals.
McNemar uses the exact two-sided binomial test on discordant questions; all-six Holm values are additionally reported.

## Interpretation and limits

Combined-flow and fixed early layers 0–7 each answered 1,605 questions correctly. Each model uniquely answered 42 questions correctly, so equal aggregate accuracy does not mean identical predictions. There is no observed accuracy advantage for combined-flow placement over this simple early-layer control in seed 0. The tie does not establish statistical equivalence.
These tests describe evaluation-question uncertainty conditional on these trained checkpoints, under an i.i.d.-question sampling approximation.
They do **not** measure training-seed variation or establish reliable method superiority across training runs. One seed is exploratory.
A position-control comparison can inform this project hypothesis without proving that flow rankings identify isolated causal layers.

## Artifacts and provenance

Source: `dev` at `8734991cd5c80eff45b09eb954d6d849ef495d58`; exact source-file hashes and package versions are in the new experiment manifest.
Uncommitted launcher/helper files are recorded by hash. The historical baseline's exact runtime revision was not recorded, as documented in the previous paired audit.
Early training / evaluation job IDs: **48704 / 48705**. GPU work ran exclusively inside Slurm allocations with the unchanged TC2 resources.
Final adapter: `output/peft/early_pruned8_scienceqa/scienceqa_train_llava/seed0/adapter/`; resumable `checkpoint.pt` is removed by the unchanged trainer after success.
Predictions and metrics: `output/peft/early_pruned8_scienceqa/scienceqa_train_llava/seed0/eval_scienceqa/`. Historical artifacts were read only and passed preservation hashes.

[Summary CSV](../output/pathway_top8_seed0/final_summary.csv) · [Paired comparisons](../output/pathway_top8_seed0/final_paired_comparison.csv) · [Accuracy chart](../output/pathway_top8_seed0/final_accuracy.png) · [Selected layers](../output/pathway_top8_seed0/final_selected_layers.png) · [Full analysis](../output/pathway_top8_seed0/final_comparison.json) · [Submission manifest](../output/pathway_top8_seed0/early8_submission_manifest.json)

The completed early control follows the repository layout: rank map in [configs/ranks/early_pruned8_scienceqa.json](../configs/ranks/early_pruned8_scienceqa.json), training/evaluation artifacts under `output/peft/early_pruned8_scienceqa/scienceqa_train_llava/seed0/`, shared comparison artifacts under `output/pathway_top8_seed0/`, and original Slurm logs under `slurm/logs/sqa_final_early8_s0_{train,eval}_{jobid}.{out,err}`. The manifest preserves submitted paths and hashes alongside relocation mappings. Reorganization changed path metadata only; it did not rerun training or evaluation.

## Completed jobs and branch provenance

Original Slurm accounting verified both jobs **COMPLETED, exit 0:0** on 29 October 2026 (Asia/Singapore): training **48704**, 12:34:11–13:29:57 (**55m 46s**); evaluation **48705**, 13:29:57–13:37:36 (**7m 39s**). These are allocation durations; the table's minutes are the trainer's numerical log and include a different timing boundary.

The historical source revision above describes the actual run. This experimental branch preserves newer teammate changes from `dev` revision `920b2e29d5d048fac2ee142fa2283a929574365e`; current training/evaluation files are not asserted byte-identical to the run revision. No source code, models or GPU jobs were changed during finalization.

The completed evidence uses the current repository paths above. The earlier [precompletion manifest](../outputs/final_early8_seed0/submission_manifest.json) and [submitted rank snapshot](../outputs/final_early8_seed0/rank_config.json) are retained as historical evidence; their old paths and job states describe that earlier snapshot, not current completion status. The canonical rank map is byte-identical to the submitted snapshot. The completed manifest preserves original submitted commands, source fingerprints, package versions and path-relocation records. The checkpoint remains in the main TC2 checkout; no model weights, resume checkpoints or Slurm/scratch logs are included here. The short `train_log.jsonl` is numerical evidence for the reported training duration and final step.

For CPU reproduction, the existing [comparison analysis](../peft_lora/methods/pathway_specific/compare_top8.py) expects a manifest inside its early-run input directory. Assemble a fresh temporary directory containing the retained early `train_config.json`, numerical `train_log.jsonl`, and `eval_scienceqa/` files; copy the completed manifest into that temporary directory as `submission_manifest.json`. Pass this staging directory with `--early-run`, plus fresh output/report destinations within this worktree. This preserves the canonical saved artifacts and requires no source changes. The analysis uses the original answer normalization and recomputes all 20,000 paired-question resamples. Its defaults refer to the historical precompletion location and should not be used for this completed comparison. Checkpoint paths in manifests and numerical summaries record the original TC2 storage; the corresponding weights are deliberately not part of this branch.

Finalization independently rescored all four retained prediction sets and regenerated all six paired contrasts with 20,000 bootstrap draws. Counts, confidence intervals, exact tests and multiplicity adjustments reproduced the completed numerical report exactly. Only documentation and result artifacts changed; analysis and training/evaluation source files remained unchanged.
