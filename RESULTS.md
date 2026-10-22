# Results

GQA testdev-balanced (12,578 questions), LLaVA-1.5-7B.

| Configuration | Accuracy (%) | Notes |
|---|---|---|
| Zero-shot (no fine-tuning) | 61.91 | 7,787 / 12,578 correct; paper reports 62.0 |
| Uniform LoRA (rank 8), seed 0 | 61.27 | 7,706 / 12,578 correct; `configs/ranks/uniform.json` |
| Flow-guided LoRA, seed 0 | 61.02 | 7,675 / 12,578 correct; ranks averaged over 6 question types (`configs/ranks/flow_guided.json`) |
| Flow-guided LoRA, LR 1e-4, seed 0 | 61.06 | 7,680 / 12,578 correct; lowercase training answers; LR 2e-4 → 1e-4 made no difference (61.02) |
| AdaLoRA (init rank 12 → avg 8), seed 0 | 61.44 | 7,728 / 12,578 correct; final ranks in `adalora_seed0_ranks.csv` |
