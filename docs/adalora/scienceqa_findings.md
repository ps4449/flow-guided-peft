# ScienceQA AdaLoRA audit — 24 October 2026

## Conclusion and limits

The three completed AdaLoRA runs used the correct ScienceQA data and completed the intended updates. No wrong-dataset, missing-seed, invalid-answer-format, incomplete-pruning or failed-reload explanation was found in the artifact checks. **The cause of the remaining performance gap is not isolated.** Algorithm-specific regularization, pruning/importance dynamics and parameterization remain hypotheses; a lower AdaLoRA score alone is not evidence of a bug.

## Evidence from actual saved runs

The CPU audit is reproducible from the repository root using `python experiments/scienceqa_adalora/audit_saved_runs.py` once the published artifacts are present. Its report is `output/diagnostics/scienceqa_investigation/saved_run_audit.json`.

- Dataset: `datasets/scienceqa/train_llava.json`, SHA256 `8cc00fb37a7e642e05b0063e2554a5e4a0be3a3e8bb13e49dbba20045ea1c4f9`, 6,218 rows. All targets are uppercase option letters. Each AdaLoRA run's saved environment manifest records this exact hash. The image directory is `datasets/scienceqa/images`.
- On TC2, `validate_data.py` passed: the conversation text, image names and labels match the shared question objects, splits are disjoint, and 10,332 image files exist.
- All three AdaLoRA configurations/logs confirm one epoch, batch 1 / accumulation 16, LR 2e-4, warmup .03 and all 388 optimizer steps. The CPU partial-batch regression passed: both loops select the same shuffled first 6,208 examples for the same seed; final ten rows are dropped.
- AdaLoRA ranks reach 512 across 64 matrices in every seed. Saved mask/reload checks pass; task losses and gradient norms are finite. A prior tensor audit found finite weights and zero singular values in masked slots.
- Every evaluated set contains all 2,017 unique test IDs with matching reference labels. All predictions normalize to valid option letters. Scores were recomputed from predictions, not accepted from metrics alone.
- Published Python/shell/JSON sources were compared byte-for-byte against the files on TC2. Hashes are in `output/diagnostics/scienceqa_investigation/tc2_source_hashes.json` (the newly added audit script and documentation are separate).

| Run | Merged accuracy | Unmerged accuracy | Training minutes |
| --- | --- | --- | --- |
| Corrected Uniform seed 0 | 80.86% (1631/2017) | 81.01% (1634/2017) | 61.7 |
| AdaLoRA seed 0 | 71.74% (1447/2017) | 72.38% (1460/2017) | 70.9 |
| AdaLoRA seed 1 | 71.10% (1434/2017) | Not run | 70.7 |
| AdaLoRA seed 2 | 71.24% (1437/2017) | Not run | 70.9 |

The matched seed-0 gap is 9.12 percentage points merged and 8.63 points unmerged. Merging contributes a numerical difference but does not explain the full gap. These test evaluations are exploratory diagnostics, not a basis for selecting hyperparameters. Statistical significance has not been established by these checks.

## Training differences and next discriminating control

Corrected Uniform and AdaLoRA share the intended batch 16, examples, update count, LR and task objective. Saved adapters for both are FP32. The implementations still differ: Uniform uses microbatch 4 / accumulation 4, while AdaLoRA uses 1/16, a separate guarded engine, SVD parameterization, importance-based pruning, and orthogonality regularization (weight .5). Uniform alpha is 16 at rank 8; AdaLoRA alpha is 24 at initial rank 12. These nominal values should not be interpreted as equivalent effective update magnitudes throughout pruning.

In AdaLoRA seed 0's first 40 updates, mean task loss is about .413 and added regularization about .575. The regularizer is active, not accidentally omitted. Its size alone does not prove that it caused worse accuracy. The historical tensor audit showed smaller AdaLoRA weight updates, but compared against the OLD batch-affected Uniform; it cannot establish the corrected comparison's mechanism.

The earlier 18.4-minute Uniform training was batch-truncated and is not a valid efficiency reference. Corrected Uniform took 61.7 minutes versus AdaLoRA about 71 minutes. This is observed wall time, not an isolated measure of AdaLoRA overhead.

A useful next control is static Uniform rank 8 through **the same AdaLoRA engine**, microbatch 1 / accumulation 16, in a new output directory. That tests pipeline/microbatch differences as a group before changing AdaLoRA. If the deficit persists, use a predeclared validation-based regularization/pruning ablation to isolate those effects. No new GPU training or hyperparameter changes were launched for this audit.

## Reproduction entry points

Code lives in `experiments/scienceqa_adalora/`; the unchanged GQA engine dependencies live in `experiments/adalora_reproduction/`. These ScienceQA wrappers were used with Python 3.10, torch 2.1.2+cu121, transformers 4.40.0.dev0, PEFT 0.12.0, LLaVA 1.7.0.dev0 and LLaVA-NeXT commit `bce12e479bc4dfee2b9c50c88137b01ff51bd483`.

The dataset preparation instructions remain in `SETUP.md`. Run from the repository root, with the ScienceQA images downloaded and the existing environment/model cache prepared:

```bash
python experiments/scienceqa_adalora/validate_data.py
# Historical full-size regression; CPU only, imports existing ML dependencies:
CUDA_VISIBLE_DEVICES='' python experiments/scienceqa_adalora/check_partial_batch_fix.py
```

Inside a TC2 GPU allocation, a direct AdaLoRA reproduction command is:

```bash
# Replace 0 with the desired seed and choose a NEW output directory.
python -u experiments/scienceqa_adalora/train_adalora.py \
  --ranks experiments/adalora_reproduction/adalora.json --seed 0 \
  --train-file datasets/scienceqa/train_llava.json --image-folder datasets/scienceqa/images \
  --epochs 1 --batch-size 1 --grad-accum 16 --num-workers 0 --lr 2e-4 \
  --warmup-ratio .03 --max-length 2048 --save-every 200 --max-hours 5.3 \
  --out-dir output/peft/adalora/scienceqa_reproduction/seed0
```

`tc2_full.sh` is the actual historical SLURM wrapper, including the environment and output validators. It uses a user-specific working directory and historical smoke-marker/output paths; adapt those before reuse. Its **Uniform branch and `team_train_lora.py` preserve the obsolete batch-affected baseline**. Do not restart completed login controllers or use that Uniform branch for new experiments. `tc2_uniform_fixed.sh`, `train_uniform_modalities_fixed.py` and `batch_guard.py` record the corrected Uniform run; the shared `peft_lora/train_lora.py` also has the modalities fix.

For a fair inference comparison, use `eval_scienceqa.py` (merged) for both methods or `eval_scienceqa_unmerged.py` for both, with the same question file and new output folder. `check_shared_reload.py` checks unmerged save/load equivalence; it is not a merged/unmerged equivalence test.

Full model/adapter weight binaries are not committed. Configurations, logs, masks, reload reports, predictions, metrics, source manifests and diagnostic reports are sufficient to inspect the recorded process and rerun it with the cached base model.
