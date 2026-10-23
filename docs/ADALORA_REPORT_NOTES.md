# AdaLoRA: report and presentation notes

Verified completed runs, updated 24 October 2026. These notes describe the measured implementation, not proposed extensions.

## Research role and method

AdaLoRA is the importance-based adaptive allocation baseline for the team's question: can visual-information-flow-guided LoRA allocation improve adaptation under a comparable final effective parameter budget? It does not use ChooseAttr or any other flow graph. Instead, PEFT uses adapter parameters and their gradients to estimate importance during training and prune low-importance rank components. Allocation is per matrix: each of 32 language-model blocks has independent q_proj and v_proj adapters.

All 64 matrices begin at rank 12 (sum 768). The schedule has 125 initial updates, allocation updates every 10 steps, and a 125-update final phase at the target budget. Final rank sum is 512, equivalent to an average rank of 8. A rank of zero disables only that adapter contribution, not the frozen base projection.

## Protocol and budget

- Model: LLaVA-1.5-7B, original model revision `4481d270cc22fd5c4d1bb5df129622006ccd9234`; PEFT 0.12.0. Base, vision encoder and multimodal projector are frozen. Base BF16; adapters FP32.
- Data: 20,000 shared GQA training questions. Only the first character of each training answer is uppercased. Published data SHA-256: `a7976b64c609217d0a8118fbee2c5b87ef9cacf2c67f8e9cd5bab7f88d971402`.
- Training: one epoch, 1,250 optimizer updates, microbatch 1, accumulation 16 (effective batch 16), AdamW LR 2e-4, zero weight decay, cosine schedule, warmup 0.03, gradient clipping 1.0, max length 2048, adapter dropout 0.05, alpha 24, orthogonality regularizer weight 0.5. Initialization/data order vary with seed.
- Evaluation: all 12,578 GQA testdev-balanced questions, greedy generation, maximum 16 new tokens. Exact matching after stripping whitespace, trailing periods and lowercasing; no synonym or explanatory-phrase matching.
- Fairness: static rank-8 q/v LoRA has 4,194,304 A/B parameters. AdaLoRA starts with 6,292,224 allocated A/B/E parameters and ends at 4,194,816 effective parameters. This matches the final effective budget approximately, not the initial budget or compute. Pruned slots remain physically allocated.

## Completed results

| Capitalised run | Correct / total | Accuracy | Training time | Peak allocated GPU memory |
|---|---:|---:|---:|---:|
| Seed 0 | 7,869 / 12,578 | 62.56% | 201.0 min | 14.04 GiB |
| Seed 1 | 7,855 / 12,578 | 62.45% | 204.1 min | 14.04 GiB |
| Seed 2 | 7,857 / 12,578 | 62.47% | 200.5 min | 14.04 GiB |
| Mean ± sample std | — | **62.49 ± 0.06%** | 201.9 min mean | — |

The standard deviation is across three seed accuracies, ddof=1, and is expressed in percentage points. Means use unrounded correct counts. Each run's evaluation took approximately 43–44 minutes. Memory is PyTorch's peak allocated training memory, not all device/process memory; peak reserved memory was 14.87–14.95 GiB. All runs completed, reached rank sum 512 and passed fresh-process adapter reload checks (reported maximum absolute logit difference 0.0 on the fixed diagnostic input).

The shared zero-shot result is 61.91% (7,787 correct). The three-seed AdaLoRA mean is about 0.58 percentage points higher. This is descriptive: it is not a significance claim or a completed comparison against every PEFT method. Use capitalised results only when comparing adapted methods. The latest team table notes that uniform seed 0 was inadvertently lowercase and still needs a capitalised rerun; do not compare its earlier mixed-casing three-seed mean as a controlled baseline.

## Final rank allocation

| Seed | Sum of q_proj ranks | Sum of v_proj ranks | Total |
|---|---:|---:|---:|
| 0 | 225 | 287 | 512 |
| 1 | 211 | 301 | 512 |
| 2 | 211 | 301 | 512 |

Each run's `final_ranks.csv` provides all 32 layers and `final_rank_mask.json` preserves the underlying 64 component masks. In these runs, more capacity was retained in v_proj overall. This is a run-specific observation, not evidence that v_proj is universally more important, that individual layers learned identical functions across seeds, or that these ranks measure visual information flow.

## Capitalization ablation

Original lowercase seed 0 scored 61.44%; its capitalised counterpart scored 62.56% (+1.12 percentage points, 141 net additional correct). Original lowercase seed 1 scored 61.35%; capitalised seed 1 scored 62.45% (+1.10 percentage points, 138 net additional correct). Lowercase seed 2 was not run, so there is no paired three-seed capitalization estimate.

For seed 0, the fixed exploratory 512-question short comparison found similar accuracy at equal updates: step 20 was 60.74% lowercase versus 60.35% capitalised; step 100 was 61.13% versus 60.94%. Median steady update time was 9.60 versus 9.51 seconds. These measurements do not demonstrate faster learning or meaningful throughput improvement. The subset was not used to select hyperparameters.

The initial frozen-model loss probe showed that target case explains much of the high initial task loss in this setup. In full seed 0 training, mean task loss over the first 20 updates was 5.524 lowercase versus 0.336 capitalised. Over the final 250 updates, mean task loss was 0.383 versus 0.342; per-update standard deviation was 0.142 versus 0.127. These exclude regularization. Target case can change tokenization and the supervised objective; lower loss alone does not establish semantic improvement. The final accuracy measurements provide separate evidence.

## Implementation and limitations

The exact completed-run sources, manifest, environment checks and TC2 commands are in `experiments/adalora_reproduction/`. They are published separately from the latest shared trainer. The implementation preserves AdaLoRA's orthogonality term, full allocator/optimizer/scheduler/RNG resume state and padded export scaling. The small CPU test verifies continuous versus resumed updates and export/reload on a toy model; all actual MLLM runs also have reload checks. Do not treat the shared commit identifier alone as sufficient provenance: these runs originally included local integration changes, now preserved as exact hashed sources.

No model weights or checkpoints are uploaded. Published predictions support metric recomputation; fresh inference requires the original adapter files or rerunning training. No testdev-driven hyperparameter search was performed here. Three seeds characterize variability for this setup, not robustness across datasets/models. Compare target matrices, precision, effective microbatch, training loop and compute explicitly when combining results from other methods.

## Suggested presentation sequence

1. **Method and budget:** importance-based allocation, independent q/v ranks, 768 → 512, larger initial allocation than static LoRA.
2. **Results and stability:** show 62.56 / 62.45 / 62.47 and 62.49 ± 0.06; report A40 time/memory alongside accuracy.
3. **What we learned:** explain capitalization using the full loss plot and final accuracy, while distinguishing lower initial loss from faster learning. Show the per-layer allocation table/plot if comparing mechanisms with flow-guided ranks.
4. **Limits and comparison:** one dataset/model, approximate final-budget match, no complete paired three-seed casing experiment, and wait for consistently capitalised controls before claiming method superiority.
