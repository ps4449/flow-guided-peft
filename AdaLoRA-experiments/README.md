# AdaLoRA reproduction

AdaLoRA runs use their own frozen trainers, separate from the shared `peft_lora/train_lora.py`. Results are in
[RESULTS.md](../RESULTS.md). Removed files can be recovered from Git: the diagnostic run outputs, probes and original
scripts from the tag `archive/nop-experiments-2026-10-26`, and the later AdaLoRA notes (`docs/adalora/`) and tests from
commit `6efc769` (`git checkout 6efc769 -- docs/adalora`).

**Method.** LoRA-style adapters on q_proj and v_proj of all 32 LLM layers, starting at rank 12 per matrix (rank sum 768)
and pruned by PEFT's importance scores to rank sum 512 (average 8, the shared budget). AdaLoRA does not read the
information-flow graphs.

**Environment.** Python 3.10, torch 2.1.2+cu121, transformers 4.40.0.dev0, PEFT 0.12.0, LLaVA 1.7.0.dev0, LLaVA-NeXT
commit `bce12e479bc4dfee2b9c50c88137b01ff51bd483`, model `liuhaotian/llava-v1.5-7b` revision
`4481d270cc22fd5c4d1bb5df129622006ccd9234`. Use a separate conda environment `llava_shared`, cloned from `llava`, with
`pip install --no-deps 'peft==0.12.0'`. Data setup as in [SETUP.md](../SETUP.md).

All runs: microbatch 1 × accumulation 16, LR 2e-4, warmup 0.03, one epoch, max length 2048, BF16 base with FP32 adapters,
alpha 24, dropout 0.05. Each command writes to a new folder and never overwrites the published runs.
Submit from the repository root after `mkdir -p slurm/logs`.

## GQA (`adalora_reproduction/`)

Published runs: `output/peft/adalora_capfirst/seed0-2` (capitalised answers, the command below) and
`output/peft/adalora/seed0-1` (lowercase answers: same trainer with `--train-file datasets/gqa/train_20000_llava.json`
before it was capitalised). 1,250 optimizer steps on the 20,000-question training sample.

```bash
sbatch AdaLoRA-experiments/adalora_reproduction/run_tc2.sh train 0   # seed 0, 1 or 2; resubmit on exit code 3
sbatch AdaLoRA-experiments/adalora_reproduction/run_tc2.sh eval 0    # after training and its reload check
```

`run_tc2.sh` checks the environment (`verify_protocol.py`), builds the capitalised training file and checks its hash
against `source_manifest.json` (`prepare_data.py`), trains, checks adapter reload, and evaluates on testdev.
Output: `output/peft/adalora_reproduction/seed<k>`.

## ScienceQA (`scienceqa_adalora/`)

Published runs: `output/peft/adalora/scienceqa_train_llava/seed0-2` (regularization 0.5) and
`output/peft/adalora/scienceqa_regoff_confirm/seed0-2` (regularization off). 6,218 training rows, 388 optimizer steps.
The trainer imports the model loader from `adalora_reproduction/`.

| Config (`scienceqa_adalora/configs/`) | Regularization | Pruning |
|---|---|---|
| `regularization_on.json` | 0.5 | 12 → avg 8 (original) |
| `regularization_off.json` | off | 12 → avg 8 (main ScienceQA setting, chosen after the seed-0 diagnostics) |
| `no_pruning.json` | 0.5 | off, rank 12 kept (larger budget, diagnostic only) |
| `neither.json` | off | off, rank 12 kept (larger budget, diagnostic only) |

```bash
sbatch AdaLoRA-experiments/scienceqa_adalora/tc2_run.sh regularization_off 0   # config name, then seed 0, 1 or 2
```

`tc2_run.sh` checks the data hashes (`validate_data.py`), trains, checks reload, validates the run
(`validate_run.py`), and evaluates validation and merged/unmerged test with `evaluate.py`.
Output: `output/peft/adalora/scienceqa_reproduction/<config>/seed<k>_job<id>`.
