# ScienceQA seed-0 paired error analysis

All three files contain the same **2,017 unique held-out question IDs and exact gold labels**.
Predictions were aligned by ID and rescored with the unchanged evaluator's
`answer.strip().rstrip('.').lower()` rule; no option extraction or rescoring change was introduced.
Verified totals: Combined **1,605**, Question→Last **1,572**, middle control **1,522** correct.

## Paired comparison

A and B follow the order in the first column. Positive differences favor A.

| A − B | Both correct | Both wrong | A-only | B-only | Δ pp [95% paired CI] | Exact McNemar p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Combined − Question→Last | 1513 | 353 | 92 | 59 | +1.636 [+0.446, +2.876] | 0.00898662 | 0.00898662 |
| Combined − Middle control | 1461 | 351 | 144 | 61 | +4.115 [+2.727, +5.553] | 6.40493e-09 | 1.92148e-08 |
| Question→Last − Middle control | 1485 | 408 | 87 | 37 | +2.479 [+1.388, +3.570] | 8.27416e-06 | 1.65483e-05 |

Intervals use **20,000 ordinary paired question-bootstrap draws**, seed 0;
each draw resamples 2,017 question IDs with replacement, using the same indices across models.
Reported endpoints are percentile 2.5/97.5; they are marginal 95% intervals, not simultaneous family-wise intervals.
Both-wrong includes cases with different incorrect predictions.

Exact two-sided McNemar tests use a Binomial(discordant count, 0.5) null on A-only/B-only outcomes;
no chi-square approximation or continuity correction is used. See the
[exact-test documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html) and
[SciPy binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).
Holm adjustment covers the three overall pairwise tests only.

## Subject and grade differences

Percentages and signed differences below are descriptive; tiny groups should not be generalized.

### Subject

| Group | n | Combined % | Question→Last % | Middle % | C−Q pp | C−M pp | Q−M pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| language science | 44 | 79.55 | 84.09 | 81.82 | -4.55 | -2.27 | +2.27 |
| natural science | 1209 | 74.86 | 73.61 | 71.46 | +1.24 | +3.39 | +2.15 |
| social science | 764 | 87.04 | 84.42 | 81.41 | +2.62 | +5.63 | +3.01 |

### Grade

| Group | n | Combined % | Question→Last % | Middle % | C−Q pp | C−M pp | Q−M pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grade1 | 2 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |
| grade2 | 117 | 83.76 | 83.76 | 80.34 | +0.00 | +3.42 | +3.42 |
| grade3 | 342 | 88.01 | 88.30 | 87.43 | -0.29 | +0.58 | +0.88 |
| grade4 | 361 | 84.21 | 81.16 | 79.22 | +3.05 | +4.99 | +1.94 |
| grade5 | 336 | 84.52 | 81.85 | 80.95 | +2.68 | +3.57 | +0.89 |
| grade6 | 271 | 71.59 | 70.11 | 66.79 | +1.48 | +4.80 | +3.32 |
| grade7 | 281 | 70.46 | 69.40 | 66.90 | +1.07 | +3.56 | +2.49 |
| grade8 | 302 | 72.52 | 70.20 | 64.57 | +2.32 | +7.95 | +5.63 |
| grade12 | 5 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | +0.00 |

## Illustrative disagreements

For each direction, examples were selected by a fixed SHA-256 rule within the largest disagreement-subject strata.
The table shows the first example per direction; full prompts, images, and additional subject examples are in `analysis.json`.
These illustrate observed disagreements, not an unbiased estimate of error mechanisms.

| A / B | Who succeeds | Question / subject / grade | Gold; A output; B output |
| --- | --- | --- | --- |
| Combined / Question→Last | A | test_3430: What is the direction of this push? (natural science, grade2) | B; B; A |
| Combined / Question→Last | B | test_1188: Which of these organisms contains matter that was once part of the bear sedge? (natural science, grade5) | A; B; A |
| Combined / Middle control | A | test_1589: Which better describes the tide pool ecosystems in Little Corona Beach? (natural science, grade3) | B; B; A |
| Combined / Middle control | B | test_4060: What is the probability that a cockatiel produced by this cross will have bright orange cheeks? (natural science, grade8) | A; C; A |
| Question→Last / Middle control | A | test_3341: Select the mammal below. (natural science, grade3) | B; B; A |
| Question→Last / Middle control | B | test_1889: Which of these organisms contains matter that was once part of the lichen? (natural science, grade7) | E; D; E |

## Original Slurm and configuration provenance

Slurm accounting timestamps are **Asia/Singapore (UTC+08:00)**, not file modification times.

| Model | Stage / job | Start SGT | End SGT | Elapsed | State / exit |
| --- | --- | --- | --- | --- | --- |
| Combined | train / 47417 | 2026-10-27 04:21:51 | 2026-10-27 05:17:47 | 00:55:56 | COMPLETED / 0:0 |
| Combined | eval / 47421 | 2026-10-27 06:12:36 | 2026-10-27 06:20:36 | 00:08:00 | COMPLETED / 0:0 |
| Question→Last | train / 48610 | 2026-10-29 03:33:47 | 2026-10-29 04:29:44 | 00:55:57 | COMPLETED / 0:0 |
| Question→Last | eval / 48619 | 2026-10-29 05:25:20 | 2026-10-29 05:32:59 | 00:07:39 | COMPLETED / 0:0 |
| Middle control | train / 48611 | 2026-10-29 04:29:45 | 2026-10-29 05:25:20 | 00:55:35 | COMPLETED / 0:0 |
| Middle control | eval / 48623 | 2026-10-29 05:32:59 | 2026-10-29 05:40:31 | 00:07:32 | COMPLETED / 0:0 |

All six jobs ran on TC2N01 with one GPU and 30 GB RAM; training requested eight CPUs and evaluation four.
The baseline's original submission commands explicitly identify `flow_pruned8_scienceqa`, seed 0 and the ScienceQA files.
Shared saved training settings match: LLaVA-1.5-7B, vicuna_v1, one epoch, microbatch 4 × accumulation 4,
lr 2e-4, 3% warmup, 388 steps, and 1,048,576 trainable parameters. New adapters verify rank 8, alpha 16, dropout 0.05.
At the original audit, the frozen new-run trainer, evaluator, shared loader and datasets were byte-identical to baseline artifact-introduction commit
`6efc769401506c0a2b5c5416578f8f0f05602891`; new runs' frozen source hashes also matched.
The curated branch includes later teammate changes. Offline reproduction verifies the recorded training revision
from Git history rather than assuming that the latest trainer/evaluator is the historical implementation.
This supports protocol consistency, but the baseline's exact runtime commit/package fingerprint was not recorded:
its checkpoint is absent, and its config omits the newer inactive flow-gate fields. The metadata-addition commit
is not proof of the checked-out source revision at training time. New runs record commit
`8734991cd5c80eff45b09eb954d6d849ef495d58`. Original submitted paths are retained alongside relocation mappings in the existing manifest.

## Uncertainty and scope

These are comparisons of **three fixed seed-0 trained artifacts**. Bootstrap intervals and McNemar p-values describe
evaluation-question uncertainty under the paired/i.i.d.-question sampling model. They do not measure training-seed
variation, certify performance on another population, or establish pathway-level statistical superiority across seeds.
Correlated questions would weaken the i.i.d. approximation. No subgroup significance tests or causal error explanations are claimed.
No inference, training, checkpoint changes, or GPU jobs were performed. Existing experiment artifacts passed hash-preservation checks.

[Comparison table](../output/pathway_specific/scienceqa/paired_errors_seed0_20261029/comparison.csv) · [Subject/grade details](../output/pathway_specific/scienceqa/paired_errors_seed0_20261029/subgroup_differences.csv) · [Paired-error visualization](../output/pathway_specific/scienceqa/paired_errors_seed0_20261029/paired_errors.png) · [Per-question audit](../output/pathway_specific/scienceqa/paired_errors_seed0_20261029/paired_correctness.csv) · [Full results and provenance](../output/pathway_specific/scienceqa/paired_errors_seed0_20261029/analysis.json)

## CPU reproduction and preserved provenance

The [paired analysis](../peft_lora/methods/pathway_specific/paired_errors.py) accepts fresh `--output-dir` and `--report` destinations. Without `--slurm-audit` it reuses the original read-only accounting snapshot embedded in the retained `analysis.json`; it does not contact Slurm or submit jobs. It verifies the frozen training source against Git revision `8734991cd5c80eff45b09eb954d6d849ef495d58`, keeps the original normalization, and recomputes all 20,000 bootstrap draws. Existing saved analyses and predictions remain unchanged.
