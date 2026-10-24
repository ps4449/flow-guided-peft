> **Historical reproduction sources and current audit (24 October 2026).** These are the actual ScienceQA wrappers/engine, not a replacement for the shared trainer. `team_train_lora.py` and the Uniform branch of `tc2_full.sh` preserve the OLD batch-affected baseline; do not use them for new multi-sample training. Use the fixed shared trainer or `train_uniform_modalities_fixed.py` for corrected Uniform. The completed controllers must not be restarted. See `docs/SCIENCEQA_ADALORA_AUDIT.md` for current conclusions and direct reproduction commands.

# ScienceQA: Uniform seed 0 and AdaLoRA seeds 0–2

This adapts the proven AdaLoRA engine to the team's ScienceQA-IMG protocol, without changing the frozen GQA reproduction source or existing GQA results. Team ScienceQA source: `a3060ab` (see `input_manifest.json`); `run_manifest.json` records the exact staged code/data hashes.

## Data and scoring

- Download `derek-thomas/ScienceQA` parquet files, extract images, and compare regenerated question objects with the team's committed files before use.
- Image-only splits: 6,218 train, 2,097 validation, 2,017 test; 10,332 images. Training/validation/test identifiers are disjoint.
- All training prompts are checked for answer-boundary alignment and the 576-token image expansion before training. Maximum measured expanded training length: 1,170 tokens, under 2,048.
- Targets are uppercase option letters, already capitalised. Use the shared team prompt and exact case-normalised matching (`strip().rstrip('.').lower()`), not a new multiple-choice extraction rule. An explanatory phrase is not scored as a standalone letter.
- Test predictions are never used to choose hyperparameters. The 20-question smoke evaluation checks generation plumbing only.

## Training protocol

Both methods were configured for LLaVA-1.5-7B, one epoch, intended effective batch 16, LR 2e-4, cosine warmup 0.03, no weight decay, and the prepared `llava_shared` environment (PEFT 0.12.0). Images are loaded from `datasets/scienceqa/images`.

- Uniform: rank 8 on q/v, 4,194,304 trainable parameters, the team's ScienceQA shared training source, microbatch 4 / accumulation 4. Worker count is 0 for these runs, recorded in the configuration.
- AdaLoRA: verified GQA training engine with the ScienceQA partial-batch guard adaptation, microbatch 1 / accumulation 16, base BF16 / adapter FP32, alpha 24, dropout .05, orthogonality regularizer .5, init rank 12 → target average rank 8. No flow graph input. Final effective A/B/E budget is 4,194,816; initial/allocated budget is 6,292,224.
- The original loops yield **388 optimizer updates**. They process 6,208 questions and omit the final 10 after shuffling, consistently with the shared effective-batch convention. This is recorded rather than silently changing either loop.
- AdaLoRA computes its schedule from 388 updates: tinit=38, tfinal=38, deltaT=10. It does not reuse the GQA 1,250-update schedule.
- The tiny pilot trains each method for 20 updates on the first 320 questions. Its shorter schedule is for mechanics validation, not a final result. AdaLoRA must reach the correct mask budget, export, reload and generate successfully before the full pipeline is released.

The corrected effective batch does not erase all implementation differences: microbatch size, adapter precision, resume handling and AdaLoRA's extra regularizer/state should be reported alongside compute and accuracy. The preserved shared trainer remains the team's uniform baseline; this module does not silently retrofit other methods.

## TC2 execution

From the repository root, after the download completes:

```bash
sbatch experiments/scienceqa_adalora/tc2_prepare_smoke.sh
# Start the login-only controller with the returned smoke job ID:
nohup /home/msai/c250274/.conda/envs/llava_shared/bin/python -u \
  experiments/scienceqa_adalora/login_pipeline.py <smoke_job_id> \
  > output/peft/pipeline_scienceqa.log 2>&1 < /dev/null &
```

The authorized current controller already exists; **do not start a second one**. Original state is `output/peft/pipeline_scienceqa.json`; after the guard failure, the active recovery state is `output/peft/pipeline_scienceqa_recovery.json`. Stages: validate smoke → Uniform seed0 train/eval → AdaLoRA seeds0/1/2 train/eval → mean/sample std. All subsequent submissions happen on the login node. Errors stop the sequence; exit3 checkpoint resumes are bounded. Existing output folders are checked to avoid duplicates.

Outputs are separate from GQA:

- `output/peft/uniform/scienceqa_train_llava/seed0/`
- `output/peft/adalora/scienceqa_train_llava/seed{0,1,2}/`
- Each run's predictions and metrics: `eval_scienceqa/`
- AdaLoRA final summary: `output/peft/adalora/scienceqa_train_llava/three_seed_summary.json`

Full training and evaluation are artifact-validated at each stage. No zero-shot, flow-guided, inverse-flow or information-flow jobs are included: these are owned elsewhere in the team.

## Full-size regression and recovery

The 320-question pilot passed, but the first full AdaLoRA job (46207) stopped before any model training: the frozen GQA entry point rejected 6,218 because it is not divisible by effective batch 16. Uniform had already completed and was not rerun. The original failed run folder contains only its environment record and is preserved with a job-specific suffix.

`adalora_train_engine.py` differs from the frozen GQA engine only by removal of this divisibility guard; the existing order slicing, full-update count, losses, optimizer, scheduler, budget checks, export and resume mechanics are unchanged. It therefore processes the same shuffled first 6,208 questions as Uniform, without changing the shared data. `check_partial_batch_fix.py` reproduces the original rejection with the real full-size JSON, verifies the adapted engine reaches the model-loading boundary, and checks the identical selected sample order for seeds 0–2 without loading the model or using a GPU.

The recovery controller uses `--resume-after-uniform`, validates the completed Uniform artifacts, and runs only AdaLoRA seeds 0–2. Job 46208 is the first corrected full-size attempt. GQA reproduction files remain byte-identical.
