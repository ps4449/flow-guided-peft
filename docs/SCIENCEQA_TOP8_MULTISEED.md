# ScienceQA top-8 replication: pre-submission comparison

**Audited 2026-10-29T21:17:41 SGT on `experiments/flow-placement-ablations`. No GPU jobs submitted.** Shared `dev`, historical results and checkpoints are preserved.

| Allocation | Layers | Seed 0, correct / 2,017 | Accuracy | Seed 1 | Seed 2 |
| --- | --- | ---: | ---: | --- | --- |
| Combined flow | 0–4, 9–11 | 1,605 | 79.57% | missing | missing |
| Question→Last | 9–16 | 1,572 | 77.94% | missing | missing |
| Fixed middle | 12–19 | 1,522 | 75.46% | missing | missing |
| Fixed early, optional | 0–7 | 1,605 | 79.57% | missing | missing |

The audit scanned **113 saved training configurations across both worktrees**, matching actual rank-map contents, seed, dataset and adapter type. Only seed-0 allocations matched; no corresponding seed-1/2 outputs, adapters or resume checkpoints were found. All four seed-0 prediction sets reproduce their saved counts and share 2,017 unique IDs, exact gold labels and original scoring normalization. Uniform and sixteen-layer seeds are different allocations and do not count.

The queue was empty at the recorded query. No matching seed-1/2 jobs appeared in **51 available accounting records queried since 1 October**. Older combined seed-0 jobs were not returned; their verified accounting remains in the earlier paired-analysis artifact. This describes checked storage/accounting, not deleted or externally stored runs.

## Planned jobs and resources

**Six required training runs:** combined, Question→Last and middle × seeds 1 and 2. Each receives a separate successful-training (`afterok`) evaluation: **12 jobs total**. Fixed early's two additional seeds are excluded pending team approval.

| Stage | Partition / QoS | GPUs | CPUs | RAM | Wall limit |
| --- | --- | ---: | ---: | --- | --- |
| Training | MGPU-TC2 / normal | 1 | 8 | 30 GB | 6 hours |
| Evaluation | MGPU-TC2 / normal | 1 | 4 | 30 GB | 6 hours |

These are unchanged repository requests. Normal QoS allows two submitted jobs, but aggregate limits of one GPU, ten CPUs and 30 GB require sequential GPU execution. Dispatch one train/evaluation pair at a time. Resume exit code 3 continues the same experiment; evaluation must depend on its successful continuation. Seed-0 timings suggest **about 6.3 GPU-hours**, excluding queueing, for all six pairs.

Outputs follow `output/peft/<rank-map-stem>/scienceqa_train_llava/seed{1,2}/`; maps stay in `configs/ranks/`, logs in `slurm/logs/`, and cross-seed records in `output/pathway_top8_multiseed/`.

## Matching and prerequisites

All three maps passed actual CPU adapter-tensor validation: **1,048,576 trainable parameters**, rank-8 q_proj/v_proj only, alpha 16, dropout 0.05, all 32 base layers preserved. Packages and frozen LLaVA runtime files match seed 0; all train/test images exist.

Keep the original checkpoint, corrected multimodal batching, `vicuna_v1`, uppercase targets, one epoch, batch 4 × accumulation 4, LR 2e-4, warmup 0.03, 388 steps, bf16 AdamW/cosine, and unchanged greedy evaluation. The original loop supervises 6,208 of 6,218 training rows.

Before launch, place the isolated experimental worktree/output root on shared persistent `/home` storage; current `/tmp` is local. The plan proposes `/home/msai/swarangi001/flow-placement-ablations`. Export the seed-0 source/scripts from `8734991cd5c80eff45b09eb954d6d849ef495d58` into an isolated runtime to avoid subsequent optional-variant/dtype-cast changes. Shared source files remain untouched. Staging/dispatch are planned, not yet materialized; recheck duplicates immediately before submission.

## Current conclusion

Seed 0 orders **combined > Question→Last > middle**. Matched differences are **+1.636 pp** (combined−Question→Last), **+4.115 pp** (combined−middle), and **+2.479 pp** (Question→Last−middle). Fixed early ties combined.

**Cross-seed consistency remains unknown; three-seed mean ± sample SD cannot yet be calculated.** After completion, verify matching IDs/labels/protocol, collect exact counts, calculate mean accuracy and sample SD (`ddof=1`) over seeds 0–2, and compare accuracies within each seed. Question-bootstrap intervals for fixed checkpoints do not measure training-seed variability. Even three seeds provide limited precision.

[Audit](../output/pathway_top8_multiseed/preflight.json) · [exact job plan](../output/pathway_top8_multiseed/job_plan.json) · [CPU validation](../output/pathway_top8_multiseed/adapter_validation.json) · [Slurm snapshot](../output/pathway_top8_multiseed/slurm_audit.json) · [seed-0 comparison](FINAL_TOP8_COMPARISON.md).
