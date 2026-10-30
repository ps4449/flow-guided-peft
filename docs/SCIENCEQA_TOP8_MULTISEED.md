# ScienceQA top-8: three-seed results

Combined flow, Question→Last and fixed middle have completed seeds 0–2. Fixed early has **seed 0 only** and is discussed separately in the [final early-layer comparison](FINAL_TOP8_COMPARISON.md). The [team update](FLOW_PLACEMENT_EXPERIMENTS_UPDATE.md) explains how these experiments developed.

## Results

Accuracy is a percentage; parentheses give correct counts out of the same 2,017 held-out ScienceQA image questions. Means and sample SDs are calculated from exact counts across seeds 0–2, with ddof=1.

| Allocation | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |
| --- | ---: | ---: | ---: | ---: |
| [Combined flow](../configs/ranks/flow_pruned8_scienceqa.json) | [79.57 (1605)](../output/peft/flow_pruned8_scienceqa/scienceqa_train_llava/seed0/eval_scienceqa/metrics.json) | [80.12 (1616)](../output/peft/flow_pruned8_scienceqa/scienceqa_train_llava/seed1/eval_scienceqa/metrics.json) | [79.92 (1612)](../output/peft/flow_pruned8_scienceqa/scienceqa_train_llava/seed2/eval_scienceqa/metrics.json) | **79.87 ± 0.28%** |
| [Question→Last](../configs/ranks/question_last_pruned8_scienceqa.json) | [77.94 (1572)](../output/peft/question_last_pruned8_scienceqa/scienceqa_train_llava/seed0/eval_scienceqa/metrics.json) | [79.03 (1594)](../output/peft/question_last_pruned8_scienceqa/scienceqa_train_llava/seed1/eval_scienceqa/metrics.json) | [79.57 (1605)](../output/peft/question_last_pruned8_scienceqa/scienceqa_train_llava/seed2/eval_scienceqa/metrics.json) | **78.85 ± 0.83%** |
| [Fixed middle](../configs/ranks/middle_pruned8_scienceqa.json) | [75.46 (1522)](../output/peft/middle_pruned8_scienceqa/scienceqa_train_llava/seed0/eval_scienceqa/metrics.json) | [76.25 (1538)](../output/peft/middle_pruned8_scienceqa/scienceqa_train_llava/seed1/eval_scienceqa/metrics.json) | [76.65 (1546)](../output/peft/middle_pruned8_scienceqa/scienceqa_train_llava/seed2/eval_scienceqa/metrics.json) | **76.12 ± 0.61%** |

Combined flow outperformed both alternatives in every matched seed. Three seeds give limited evidence about training variability. This comparison does not establish a unique benefit over fixed early layers, which were tested only in seed 0.

## Matched protocol

All allocations use LLaVA-1.5-7B with rank-8 LoRA on q_proj/v_proj, alpha 16, dropout 0.05 and **1,048,576 trainable adapter parameters**. Selected layers are 0–4 and 9–11 for combined flow, 9–16 for Question→Last, and 12–19 for fixed middle. The frozen protocol uses corrected multimodal batching, uppercase option-letter targets, vicuna_v1, one epoch, microbatch 4 × accumulation 4, LR 2e-4, 3% warmup and 388 updates. Predictions align to identical unique question IDs and exact gold labels using the original answer normalization.

Canonical runs are under `output/peft/<rank-stem>/scienceqa_train_llava/seed<k>/`. Each contains `train_config.json`, `train_log.jsonl` and `eval_scienceqa/{metrics.json,predictions.jsonl}`. The linked rank maps define the allocations.

## Saved evidence

The [completed replication manifest](../output/pathway_top8_multiseed/submission_manifest.json) records actual jobs, source hashes and execution status; [seed results](../output/pathway_top8_multiseed/seed_results.csv), [three-seed summary](../output/pathway_top8_multiseed/three_seed_summary.csv) and [matched seed differences](../output/pathway_top8_multiseed/matched_seed_differences.csv) retain the numerical exports. The [preflight](../output/pathway_top8_multiseed/preflight.json) and [job plan](../output/pathway_top8_multiseed/job_plan.json) are historical snapshots, not current run status.

Replication uses frozen runtime revision `8734991`; result commit `13acdce` preserves completed outputs. The original combined-flow seed-0 runtime revision was not recorded exactly, so its result commit must not be treated as runtime provenance. Paired-question uncertainty for fixed checkpoints is separate from variability across training seeds.
