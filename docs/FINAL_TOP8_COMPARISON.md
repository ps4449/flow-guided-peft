# Final ScienceQA top-8 comparison — result pending

**Prespecified primary comparison: combined flow versus fixed early layers 0–7.**
The early control is the final planned training experiment. This branch is a snapshot, not a live status page.

At **29 October 2026, 12:52 SGT**, original Slurm accounting showed training **48704 RUNNING**, started **12:34:11 SGT** on TC2N01, and evaluation **48705 PENDING** with `afterok:48704`. Completed early evaluation metrics/predictions were not available. **No early accuracy, paired result or completion is claimed.**

| Fixed seed-0 allocation | Layers | Correct / total | Accuracy |
| --- | --- | --- | --- |
| Combined flow | 0–4, 9–11 | 1605/2017 | 79.57% |
| Question→Last | 9–16 | 1572/2017 | 77.94% |
| Fixed middle | 12–19 | 1522/2017 | 75.46% |
| Fixed early | 0–7 | pending | pending |

Every allocation uses rank-8 LoRA on `q_proj`/`v_proj`, **1,048,576 trainable parameters**, alpha 16 and dropout 0.05; all 32 base transformer layers remain. The early rank map was validated against the actual model architecture and CPU adapter tensors before submission. Full validation, packages and source hashes are preserved in the [submission manifest](../outputs/final_early8_seed0/submission_manifest.json); the [rank map](../outputs/final_early8_seed0/rank_config.json) is the exact submitted file.

## Matched protocol and provenance

The source worktree remains `dev` at `8734991cd5c80eff45b09eb954d6d849ef495d58`. The new curated branch instead starts at remote `dev` revision `920b2e29d5d048fac2ee142fa2283a929574365e` to preserve later teammate changes. The historical trainer/evaluator should be read at the recorded run revision when reproducing the original protocol; current files are not asserted byte-identical.

The early configuration matches the previous runs: `liuhaotian/llava-v1.5-7b`, corrected multimodal batching, uppercase option-letter targets, `vicuna_v1`, seed 0, one epoch, batch 4 × accumulation 4, LR 2e-4, warmup 0.03, 388 optimizer steps, bf16, AdamW with zero weight decay, cosine scheduling and gradient clipping at 1.0. Evaluation uses the same 2,017 held-out image questions and the original normalization `strip().rstrip('.').lower()`.

The manifest records torch 2.1.2, transformers 4.40.0.dev0, PEFT 0.12.0, NumPy 1.26.1 and SciPy 1.15.3, plus the remaining package versions. Both jobs use the established MGPU-TC2 partition, normal QoS, one GPU, 30 GB and six-hour limit; training requests eight CPUs, evaluation four. This curation submitted no jobs.

Original outputs remain under `/home/msai/swarangi001/flow-guided-peft/outputs/final_early8_seed0/`: logs in `logs/`, model adapter in `adapter/`, and eventual predictions/metrics in `eval_scienceqa/`. These active files and all deployment helpers remain untouched. Only the frozen rank map and a submission-manifest snapshot are copied here; no partial training log, checkpoint or model weights are committed.

## Analysis readiness and scope

The retained CPU-only [comparison source](../peft_lora/methods/pathway_specific/compare_top8.py) extracts the scientific comparison from the deployed helper, excluding Slurm coordination. It requires completed evaluation artifacts, exact 2,017 unique IDs/gold labels, matching saved metrics/configurations and all four rank maps before writing any result. Fresh output/report destinations are required; it cannot submit jobs or update the active manifest. The deployed helper in the original worktree still owns the automatic final report when evaluation completes.

The analysis uses 20,000 paired-question bootstrap draws, seed 0, exact two-sided McNemar tests and secondary Holm adjustment across all six comparisons. Combined versus early is primary. The final accuracy and selected-layer figures will accompany verified evaluation; no placeholder result charts are supplied.

These comparisons estimate question-sampling uncertainty conditional on fixed seed-0 checkpoints. They do not establish training-seed reliability or flow's unique superiority over gradient/depth selection. See the [existing paired audit](PATHWAY_TOP8_PAIRED_ERRORS.md) for the baseline runtime-provenance limitation.
