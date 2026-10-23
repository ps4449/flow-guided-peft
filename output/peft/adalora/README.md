# Completed AdaLoRA runs

Updated on 24 October 2026:

| Training answers | Seed | Correct / total | Accuracy | Location |
|---|---:|---:|---:|---|
| Original lowercase | 0 | 7,728 / 12,578 | 61.44% | `output/peft/adalora/seed0` |
| Original lowercase | 1 | 7,717 / 12,578 | 61.35% | `output/peft/adalora/seed1` |
| First answer character uppercase | 0 | 7,869 / 12,578 | 62.56% | `output/peft/adalora_capfirst/seed0` |
| First answer character uppercase | 1 | 7,855 / 12,578 | 62.45% | `output/peft/adalora_capfirst/seed1` |
| First answer character uppercase | 2 | 7,857 / 12,578 | 62.47% | `output/peft/adalora_capfirst/seed2` |

Each run includes all 12,578 unique testdev predictions, overall and per-type metrics, all 1,250 training updates, training configuration, budget, environment/source hashes, and the successful adapter reload check. Weights and resume checkpoints are excluded. The legacy run paths are retained to match the recorded configuration.

The runs used the pinned local AdaLoRA integration based on `b2959d0ad6b79296075d78a898a804b58a8d330f`, with PEFT 0.12.0. That checkout included uncommitted integration changes: `environment.json` records the exact training source hashes. The exact training source is now published in `experiments/adalora_reproduction/` with hash verification and TC2 instructions; the newer shared trainer on `dev` is unchanged.

The experiment used LLaVA-1.5-7B, 20,000 GQA training questions, one epoch, seed-specific initialization and data order, batch size 1, gradient accumulation 16, LR 2e-4, cosine schedule, warmup 0.03, and frozen base/vision/projector weights. AdaLoRA starts at rank 12 for each of the 64 q/v matrices (sum 768), then prunes to sum 512. Final effective adapter budget is 4,194,816 parameters including singular values; allocated tensors remain larger.

Predictions are the original TC2 output bytes. Only `eval/metrics.json` was recomputed locally with the current shared `peft_lora/eval_gqa.py` and annotated testdev questions to add structural/semantic breakdowns; the overall correct counts were checked against the original metrics. No inference or training was rerun for this publication. `results_manifest.json` records file hashes.

Capitalised seeds 0–2 completed on 23 October 2026. Mean accuracy is 62.49%, sample standard deviation 0.06 percentage points. Lowercase seed 2 was not run. Do not treat the two lowercase seeds plus the capitalised seed as one three-seed experiment.
