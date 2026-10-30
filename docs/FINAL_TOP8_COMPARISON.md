# ScienceQA: final fixed early-layer comparison

The primary control asked whether combined-flow top-eight placement improves on simply training the first eight layers. This comparison uses **seed 0 only**. Fixed early selects layers 0–7; combined flow selects 0–4 and 9–11. Both use 1,048,576 adapter parameters and the [matched top-eight training/evaluation protocol](SCIENCEQA_TOP8_MULTISEED.md#matched-protocol).

## Result and interpretation

Both checkpoints answered **1605/2017 questions correctly (79.57%)**. They were both correct on 1,563 questions and both wrong on 370; each uniquely answered 42 correctly. Equal accuracy therefore does not mean identical predictions.

Combined minus early was **0.000 percentage points**, with a 95% paired-question bootstrap interval **[−0.892, +0.892]** and exact two-sided McNemar **p = 1.0**. The interval uses 20,000 paired question resamples with seed 0. Predictions were aligned by unique question ID and exact gold labels, with unchanged `strip().rstrip('.').lower()` normalization.

There is no observed combined-flow advantage over this simple early-layer control in seed 0. **The tie does not establish equivalence.** Fixed early seeds 1/2 were not run. These paired tests describe evaluation-question uncertainty conditional on fixed trained checkpoints, not training-seed variability or general method superiority. No unique benefit of flow over simple early-layer selection has been established.

The [three-seed comparison](SCIENCEQA_TOP8_MULTISEED.md#results) covers combined flow, Question→Last and fixed middle. Its numerical table is separate from this single-seed control.

## Evidence and provenance

- [Final comparison JSON](../output/pathway_top8_seed0/final_comparison.json) and [all six paired contrasts](../output/pathway_top8_seed0/final_paired_comparison.csv) preserve counts, bootstrap intervals, exact tests and Holm corrections. The non-primary contrasts are exploratory.
- [Checkpoint summary](../output/pathway_top8_seed0/final_summary.csv), [accuracy figure](../output/pathway_top8_seed0/final_accuracy.png) and [selected-layer figure](../output/pathway_top8_seed0/final_selected_layers.png) retain the saved exports.
- The [completed early-control manifest](../output/pathway_top8_seed0/early8_submission_manifest.json) records submitted commands/configuration, source hashes, package versions, rank validation and path relocations. Training/evaluation jobs 48704/48705 completed with exit 0:0 on 29 October 2026. The recorded runtime source is `8734991`; current shared code includes later teammate changes.
- The [early rank map](../configs/ranks/early_pruned8_scienceqa.json) defines layers 0–7. Its canonical saved run is `output/peft/early_pruned8_scienceqa/scienceqa_train_llava/seed0/`, containing the original training config/log and `eval_scienceqa/{metrics.json,predictions.jsonl}`. Weights and scheduler logs are not tracked. The older precompletion rank/manifest snapshots remain in Git history at `7c63b20:outputs/final_early8_seed0/`.

The retained [CPU comparison code](../peft_lora/methods/pathway_specific/compare_top8.py) expects a manifest within its early-run input directory. For scientific reproduction, stage copies of the saved early config/log/evaluation files and the completed manifest (named `submission_manifest.json`) in a fresh temporary input directory, then supply `--early-run` and fresh output/report destinations. Its historical default path is not the completed result location. This does not require changing the saved evidence or training code.
