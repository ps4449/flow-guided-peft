# AdaLoRA: exact training source and TC2 reproduction

This directory publishes the implementation used for the completed capitalised AdaLoRA seeds 0, 1 and 2. It is an additional, frozen entry point: it does not replace the team's current `peft_lora/train_lora.py`, information-flow code, or other methods. The six Python source files and `adalora.json` are copied byte-for-byte from the TC2 checkout used for these results. `source_manifest.json` can be checked against each published run's `environment.json`.

## What the method does

AdaLoRA does **not** read ChooseAttr or other information-flow graphs. During training, PEFT estimates the importance of adapter components from adapter parameters and gradients, then reallocates its rank budget across 64 language-model matrices (q_proj and v_proj in all 32 blocks).

It starts at rank 12 per matrix (rank sum 768) and ends at rank sum 512, an average of 8 per matrix; individual final ranks can differ. Initial/final effective adapter parameter counts are 6,292,224 / 4,194,816 including the singular values. The static rank-8 LoRA reference has 4,194,304 A/B parameters. Initial allocation, optimizer state and physical padded tensors stay larger: matching the final effective budget is not matching training compute or allocated memory.

The runtime includes the AdaLoRA orthogonality regularizer via the tuner forward path, allocates after the optimizer update while gradients are still present, saves allocator/optimizer/scheduler/RNG state for resume, and exports full-shaped masked adapters. These details are part of the completed-run implementation; substituting the newer shared loop is a separate experiment.

## Requirements

Use the model, images and TC2 setup in `SETUP.md`. The recorded environment is Python 3.10, torch 2.1.2+cu121, transformers 4.40.0.dev0, PEFT 0.12.0 and llava 1.7.0.dev0. LLaVA-NeXT source commit: `bce12e479bc4dfee2b9c50c88137b01ff51bd483`; model revision used: `4481d270cc22fd5c4d1bb5df129622006ccd9234`. The loader in the historical source uses the model name rather than a revision argument: use the matching cached model and offline flags when reproducing. Package version strings alone do not guarantee matching source revisions.

On the login node, if `llava_shared` does not already exist, create it separately from a working `llava` environment:

```bash
module load anaconda
eval "$(conda shell.bash hook)"
conda create --name llava_shared --clone llava --offline --override-channels -c conda-forge -y
conda activate llava_shared
python -m pip install --no-deps 'peft==0.12.0'
```

If the environment already exists, activate and inspect it; do not overwrite it. This is the recorded PEFT override to LLaVA's older package pin. The frozen-source and runtime tests below validate this integration.

## Checks and data preparation

From the repository root, in the prepared environment:

```bash
python experiments/adalora_reproduction/verify_protocol.py --check-env
python experiments/adalora_reproduction/prepare_data.py
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s experiments/adalora_reproduction/tests -v
```

`prepare_data.py` changes only the first answer character, writes a separate JSON, verifies the exact completed-run SHA-256 and refuses to overwrite different data. It accepts the repository's current already-capitalised source if it produces the same recorded data. It does not change shared training data.

The CPU tests check budget/target validation and a tiny PEFT model's continuous versus interrupted/resumed updates and adapter save/reload. The actual MLLM reload checks from all three completed runs are also published. There is no need to spend GPU time to use the existing results.

## Optional new run

These commands **start a new reproduction**, so run only if another experiment is needed. They intentionally write to `output/peft/adalora_reproduction/seed<k>`, separate from the completed results. Submit from the repository root, substituting seed 0, 1 or 2:

```bash
mkdir -p slurm/logs
sbatch experiments/adalora_reproduction/run_tc2.sh train 0
# After training and its reload check complete successfully:
sbatch experiments/adalora_reproduction/run_tc2.sh eval 0
```

The shell wrapper uses `llava_shared`, cached/offline model access, batch 1, gradient accumulation 16, one epoch, LR 2e-4, warmup 0.03, max length 2048, checkpoint every 200 steps and the original 1,250-step schedule. The final 10% of updates hold the target rank budget fixed.

Exit code 3 from training means a time-budget checkpoint was saved: resubmit the same **train** command before evaluating. No dependent jobs are submitted from compute nodes. The historical training capture wrapper is restored before export; it hides the forward signature during training, so do not add generation/evaluation hooks inside that loop without the separate validated signature fix. Fresh-process evaluation is supported and was used for the reported results.

The fullrun script checks reload before evaluation. Frozen inference is used first; the current shared evaluator then adds per-type metrics from the already-complete predictions without loading another model.

## Results and artifacts

See `output/peft/adalora_capfirst/three_seed_summary.json`, each seed's `eval/`, `train_config.json`, `train_log.jsonl`, `reload_check.json`, `final_rank_mask.json`, and `final_ranks.csv`. Original predictions are unchanged; per-type metrics were recomputed with the current shared evaluator. The mean is calculated from exact correct counts; standard deviation uses ddof=1.

Weights, binary checkpoints, model cache and images are not included. Published predictions are sufficient to recompute the reported scores and compare methods, but not to run fresh inference with the trained adapters. The training source and instructions are now available to rerun the method.
