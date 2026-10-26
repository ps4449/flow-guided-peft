# ScienceQA AdaLoRA reproduction

The supported ScienceQA interface and final configs are documented here; the frozen GQA interface stays separate in [gqa_reproduction.md](gqa_reproduction.md). Use the pre-cleanup tag for byte-exact historical scripts; current code preserves the training operations and adds a clearer interface.

Environment: Python 3.10; torch 2.1.2+cu121; PEFT 0.12.0; transformers 4.40.0.dev0; LLaVA 1.7.0.dev0 and LLaVA-NeXT commit bce12e479bc4dfee2b9c50c88137b01ff51bd483. Model: liuhaotian/llava-v1.5-7b, revision 4481d270cc22fd5c4d1bb5df129622006ccd9234. Prepare ScienceQA images following the team's [SETUP.md](../../SETUP.md). No cleanup command submits training.

All full ScienceQA arms: 6218 training rows, first 6208 shuffled examples per epoch, 388 optimizer updates, seed0/1/2, microbatch1 / accumulation16, LR2e-4, warmup .03, BF16 base and FP32 adapters, alpha24/dropout.05. Validation has2097 image questions and test2017. Targets are uppercase option letters. The evaluator normalizes via strip().rstrip('.').lower(), without a new answer extractor.

Configs: regularization_off.json (main post-hoc ScienceQA configuration), regularization_on.json (original reference), no_pruning.json (regularizer .5 with rank12 retained), neither.json (no regularizer, rank12 retained). Fixed-rank-12 arms use a larger budget and are mechanism controls only. All begin at rank12; pruning-enabled arms end at total rank512.

Use a new output directory for a new experiment; completed artifacts are never overwritten. The cleanup itself submits no GPU jobs.

## CPU checks (no training)

```bash
python experiments/adalora_reproduction/verify_protocol.py
python experiments/scienceqa_adalora/validate_data.py
python -m unittest discover -s experiments/scienceqa_adalora/tests -p 'test_*.py' -v
# In the existing ML environment, tiny CPU model only:
CUDA_VISIBLE_DEVICES='' python experiments/scienceqa_adalora/tests/check_mechanics.py
python experiments/scienceqa_adalora/summarize.py
```

The cleanup unit suite recomputes published predictions and checks that evaluator normalization/metrics remain unchanged. Tiny-model mechanics cover regularization/pruning switches and reload. The cleaned engine's function/class ASTs were also checked against the archived engine: identical except CLI microbatch/accumulation defaults. An import/runtime repeat on TC2 at cleanup time was blocked by the expired SSH authentication session; no new real-model/GPU run is claimed.

## Explicit reproduction commands (only when a new run is intended)

From the repository root, in an allocated GPU environment:

```bash
python experiments/scienceqa_adalora/train.py \
  --ranks experiments/scienceqa_adalora/configs/regularization_off.json --seed 0 \
  --train-file datasets/scienceqa/train_llava.json --image-folder datasets/scienceqa/images \
  --epochs 1 --batch-size 1 --grad-accum 16 --num-workers 0 --lr 2e-4 \
  --warmup-ratio .03 --max-length 2048 --out-dir output/peft/adalora/new_reproduction_seed0
python experiments/adalora_reproduction/check_shared_reload.py --run output/peft/adalora/new_reproduction_seed0
```

For rank12 controls (no_pruning/neither), add `--no-pruning`; keep these controls separate from the rank512 benchmark. The one TC2 wrapper handles these switches automatically:

```bash
mkdir -p slurm/logs
sbatch experiments/scienceqa_adalora/tc2_run.sh regularization_off 0
# Second argument may be seed 0, 1 or 2. The environment defaults to llava_shared.
```

The wrapper records source/data hashes, trains once into a fresh job-specific folder, checks export/reload and evaluates validation plus merged/unmerged test. It does not enqueue extra runs, search configurations, restart finished pipelines, or overwrite historical outputs. It stops on error. The direct engine retains historical checkpoint semantics; exact recovery workflows and row-counter resume limitations should be inspected before resuming a partial run.

## Evaluation and analysis

```bash
python experiments/scienceqa_adalora/evaluate.py \
  --adapter <run>/adapter --questions datasets/scienceqa/test_questions.jsonl --out-dir <run>/eval_test
# For unmerged, use --unmerged AND a separate --out-dir:
python experiments/scienceqa_adalora/evaluate.py \
  --adapter <run>/adapter --questions datasets/scienceqa/test_questions.jsonl --unmerged --out-dir <run>/eval_test_unmerged
python experiments/scienceqa_adalora/validate_run.py --run <run>
python experiments/scienceqa_adalora/analysis/rank_flow_alignment.py
```

The model/prompt helper is deliberately imported from the frozen GQA reproduction dependency, preserving the exact loader used for these runs and not switching to the team's newer flow-gated loader. Both evaluator modes are explicit; never choose different modes per method. `summarize.py` is read-only by default and recomputes scores from predictions. Its optional `--write` refuses overwrite. Historical metadata paths inside saved JSON files are not changed; current artifact locations are in result_path_map.json.
