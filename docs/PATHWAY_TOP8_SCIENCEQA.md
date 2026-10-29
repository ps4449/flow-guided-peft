# Matched ScienceQA top-8 seed-0 experiments

Git: `dev` at `8734991cd5c80eff45b09eb954d6d849ef495d58`. Experiment state: **completed**.
Both allocations use rank-8 q_proj/v_proj LoRA, alpha 16, dropout 0.05, and 1,048,576 trainable parameters.
They match the combined-flow baseline's training settings and evaluate the same 2,017 ScienceQA image questions.

| Experiment | Seed | Layers | Correct / total | Accuracy | Δ pp | Parameters | Trainer minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| combined_flow_top8 | 0 | [0, 1, 2, 3, 4, 9, 10, 11] | 1605/2017 | 79.57% | +0.0000 | 1,048,576 | 55.6 |
| question_to_last | 0 | [9, 10, 11, 12, 13, 14, 15, 16] | 1572/2017 | 77.94% | -1.6361 | 1,048,576 | 55.7 |
| middle_control | 0 | [12, 13, 14, 15, 16, 17, 18, 19] | 1522/2017 | 75.46% | -4.1150 | 1,048,576 | 55.3 |

Durations are rounded trainer-log minutes, including model setup. Exact new-run wall durations are in the manifest.

| Experiment | Rank configuration | Training / evaluation jobs | Run artifacts |
| --- | --- | --- | --- |
| question_to_last | [question_last_pruned8_scienceqa.json](../configs/ranks/question_last_pruned8_scienceqa.json) | 48610 / 48619 | [output/peft/question_last_pruned8_scienceqa/scienceqa_train_llava/seed0](../output/peft/question_last_pruned8_scienceqa/scienceqa_train_llava/seed0/) |
| middle_control | [middle_pruned8_scienceqa.json](../configs/ranks/middle_pruned8_scienceqa.json) | 48611 / 48623 | [output/peft/middle_pruned8_scienceqa/scienceqa_train_llava/seed0](../output/peft/middle_pruned8_scienceqa/scienceqa_train_llava/seed0/) |

On TC2, trained adapters remain in each original run's `adapter/` (weights are excluded from this branch); evaluation artifacts are `eval_scienceqa/predictions.jsonl` and `metrics.json`.
The [submission manifest](../output/pathway_top8_seed0/submission_manifest.json) preserves validation, execution records, source hashes and path relocations.
The [comparison](../output/pathway_top8_seed0/comparison.json) contains exact percentage-point differences and artifact paths. Slurm logs are under `slurm/logs/`.

Question→Last scored above the middle control, but both scored below the combined-flow baseline in seed 0.
This single seed is exploratory; no superiority across training seeds is established. The [paired audit](PATHWAY_TOP8_PAIRED_ERRORS.md) quantifies question-sampling uncertainty for these fixed checkpoints. Historical results and uniform-LoRA configurations are unchanged.

## Curated source and evidence

This branch preserves each completed run's exact predictions, metrics, training configuration and short numerical training log, alongside rank maps and the original submission/comparison manifests. The full manifest records optimizer/scheduler settings, LoRA tensor validation, package versions, runtime argv, source fingerprints and original paths. Absolute TC2 paths are historical provenance, not portable destinations.

Training used recorded source revision `8734991cd5c80eff45b09eb954d6d849ef495d58`. The curated branch is based on newer teammate revision `920b2e29d5d048fac2ee142fa2283a929574365e`; its shared training/evaluation code is preserved. Reproducing the historical protocol requires that recorded revision and the saved configs, using the established training/evaluation entry points. No experiments were rerun during curation.

Completed-run submission coordinators and their thin shell wrappers are excluded: they do not define the scoring or training objective. The active early-control deployment remains intact in the original worktree. The CPU comparison source is retained separately for scientific reproducibility.

The branch also excludes duplicate PDF plots, repeated generated Markdown summaries, regenerable bootstrap-score arrays, failed window partials, redundant copied baseline artifacts and never-used region training plans. Complete measurement CSVs, frozen prompts/IDs, predictions, bootstrap overlap counts, validation/provenance JSON and key PNG figures remain. Historical results and other teammates' files were not removed.

Curation validation recomputed the region selection, all matched window scores/stability, all 5,000 pathway bootstrap draws, and all 20,000 paired-error bootstrap draws in fresh temporary destinations. Saved numerical comparisons, overlap tables, subgroup results, examples and tests reproduced exactly. Main three-seed counts and all local documentation links were checked independently. The original protected training sources, rank maps and historical artifacts passed preservation checks. No GPU work or new jobs were submitted.
