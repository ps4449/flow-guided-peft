# ScienceQA uniform rank-4 budget control

Completed 30 October 2026; SLURM 49074, TC2N04, exit 0, elapsed 01:46:11. Seeds trained/evaluated sequentially from 00:11 to 01:57 Singapore time. No new hypotheses or additional diagnostic runs were added.

## Fixed protocol

Source: `1138b73` shared `peft_lora/train_lora.py`, `common.py`, `eval_gqa.py`, copied unchanged into an isolated remote execution directory. Exact source hashes are recorded in `output/peft/uniform_rank4/scienceqa_20261030/source_manifest.json` and were rechecked against that Git revision before publication. Historical absolute remote paths in raw configs are retained.

Rank 4 in all 64 q/v matrices: alpha 8, dropout .05, 2,097,152 trainable parameters; base frozen. Plain LoRA, no flow gates. LR 2e-4, cosine scheduler, warmup .03, 1 epoch, microbatch 4, accumulation 4, num_workers 0. Train file has 6,218 questions; 388 full updates cover 6,208 examples. Batch collator passes a modality per row. Train data SHA256: `8cc00fb37a7e642e05b0063e2554a5e4a0be3a3e8bb13e49dbba20045ea1c4f9` (checked on TC2 before training).

Environment: existing llava_shared. Model is liuhaotian/llava-v1.5-7b, same shared loader and merged evaluator as controls. No new GPU peak-memory measurement was collected; it remains unavailable. Adapter export/reload is exercised by the separate evaluation processes; no logits-equivalence claim is made.

## Reproduction

From the source revision above, use `configs/ranks/uniform_rank4_scienceqa.json` from this result commit. For each seed 0, 1, 2 (in an environment with PEFT 0.12 and installed LLaVA):

```bash
python peft_lora/train_lora.py --ranks configs/ranks/uniform_rank4_scienceqa.json --seed 0 --train-file datasets/scienceqa/train_llava.json --image-folder datasets/scienceqa/images --out-dir output/peft/uniform_rank4/reproduction/seed0 --batch-size 4 --grad-accum 4 --num-workers 0
python peft_lora/eval_gqa.py --adapter output/peft/uniform_rank4/reproduction/seed0/adapter --questions datasets/scienceqa/test_questions.jsonl --image-folder datasets/scienceqa/images --out-dir output/peft/uniform_rank4/reproduction/seed0/eval
```

Change both seed and output paths for additional seeds. Do not reuse a completed output directory for a fresh run. Shared budget.py's default checker expects rank sum512; this half-budget control intentionally has sum256, so validate the actual expected count instead of modifying the team's full-budget constant.

## Verification and interpretation

All seeds have exactly 2,017 unique test IDs, matching the test split and saved references. Scores were recomputed from normalized predictions and matched saved metrics; local rechecks also confirmed 388 steps and the expected parameter count. The three saved adapter_config.json files were also checked on TC2: rank4, alpha8 and all64 rank-pattern entries equal4. The shared trainer checks attached ranks at runtime. Raw predictions, metrics and logs are published without model checkpoints.

Mean accuracy80.5156172533%, sample SD0.4882924047 percentage points. Against existing flow-pruned exact counts [1636,1648,1639], all matched seeds favor the early16 rank8 configuration. Both arms have identical trainable A/B budgets. This establishes neither flow's superiority over a fixed early-layer rule nor general robustness beyond the studied setup.
