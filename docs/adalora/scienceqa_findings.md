# ScienceQA AdaLoRA findings

## Verified protocol

All original and regularization-off AdaLoRA runs used ScienceQA train_llava.json (SHA256 8cc00fb37a7e642e05b0063e2554a5e4a0be3a3e8bb13e49dbba20045ea1c4f9), image-only 6218 training rows, batch1/accumulation16, one epoch and388 updates (6208 shuffled examples). They did not suffer the multi-sample batch-truncation defect. Prompt/label checks, complete logs, final masks, prediction/reference IDs and adapter reload checks passed.

Pruning-enabled arms start at rank12 per q/v matrix and target retained rank sum512. Orthogonal regularization is .5 in the original configuration and0 in the post-hoc alternative. Original results remain preserved.

| Configuration | Test seed0 | Test seed1 | Test seed2 | Mean +/- sample SD (pp) |
| --- | --- | --- | --- | --- |
| Original, regularization .5 |71.74|71.10|71.24|71.36 +/- .34|
| Regularization off |76.05|76.50|76.70|76.42 +/- .33|

These are merged-adapter accuracies on2017 questions. Every score/mean is recomputed from exact correct counts; raw results and their links are in [RESULTS.md](../../RESULTS.md). Reg-off validation is77.05 +/- .24%; unmerged test is76.35 +/- .30%. The alternative was selected after exploratory seed0/test inspection, then seeds1–2 were requested; disclose that sequence rather than presenting an untouched-test confirmatory study.

## Matched-loop control

The original batch-affected Uniform seed0 scored77.64%, but only one of four microbatch rows reached the language model. After correction, Uniform4x4 scored80.86% (61.7min training). Static Uniform through AdaLoRA's engine and microbatch1/accumulation16 scored81.21% merged and81.31% unmerged (66.2min training). Thus pipeline/microbatch differences alone do not explain the roughly71–72% original AdaLoRA outcome. The earlier18.4min batch-truncated Uniform is not a valid efficiency reference. AdaLoRA training took about66–71min depending on condition.

## Seed-0 mechanism ablations

| Regularization | Pruning | Validation | Merged test | Final rank sum |
| --- | --- | --- | --- | --- |
| .5 | on |71.24|71.74|512|
| 0 | on |77.16|76.05|512|
| .5 | off |71.15|71.59|768|
| 0 | off |77.44|76.35|768|

Removing regularization improved merged test by4.31–4.76pp. Disabling pruning changed it by-.15 to+.30pp. Validation shows the same direction. The original regularization setting contributes to part of the deficit in this tested setup, but the residual difference to Uniform remains unexplained. Fixed-rank12 controls have a larger budget; do not call them equal-budget competitors or isolate pruning location from retained capacity. No statistical significance or universal cause is established by a single diagnostic seed. The three-condition scope is complete; no further experiments were automatically authorized.

## Numerical evaluation and raw evidence

Merged and unmerged computations are not exactly interchangeable in BF16. Original AdaLoRA seed0 is71.74% merged versus72.38% unmerged; this numerical difference does not explain the whole gap to Uniform. Use the same mode across methods. The saved unmerged reload check verifies save/load, not merge equivalence.

Final results are under output/peft/adalora/scienceqa_regoff_confirm/seed0–2; original reg-on under output/peft/adalora/scienceqa_train_llava/. Rank12 controls and the complete mechanism summary are under output/ablations/adalora/. Matched-loop Uniform stays under output/peft/uniform/scienceqa_matched_engine/job46525/seed0/. See [historical recovery](historical_runs.md) and result_path_map.json for exact paths.
