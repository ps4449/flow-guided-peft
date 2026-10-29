# ScienceQA pathway-specific top-8 selection

**Question→Last selects layers 9–16: a genuinely different, highly stable
calibration-based allocation worth a controlled training comparison.** It shares
only three layers with combined-flow top-8. This establishes a distinct selection
hypothesis, not better downstream performance. No model was trained or evaluated during the selection analysis. The subsequent matched training results are reported separately in [PATHWAY_TOP8_SCIENCEQA.md](PATHWAY_TOP8_SCIENCEQA.md).

The analysis uses all **989 correctly answered training-calibration examples**
in both existing window-9 CSVs: 31,648 observations per pathway. It calls the
unchanged original `flow_curve` and `layer_importance` implementation, averaging
relative changes before clipping negative means into positive importance.

| Ranking | Top-8 sorted indices | Selected score sum | All-32 score sum | Combined overlap | Jaccard |
| --- | --- | --- | --- | --- | --- |
| Image→Question | 0, 1, 2, 3, 4, 7, 8, 9 | 64.888137 | 73.710623 | 6/8 | 0.600000 |
| Question→Last | 9, 10, 11, 12, 13, 14, 15, 16 | 12.598068 | 16.954030 | 3/8 | 0.230769 |
| Combined | 0, 1, 2, 3, 4, 9, 10, 11 | 69.216979 | 90.664653 | 8/8 | 1.000000 |

Score totals use each ranking's own objective and relative-change percentage-point
units; they are not model accuracies or evidence that one pathway is better.

Position controls were fixed without consulting the scores:

| Control | Sorted indices | Image→Question sum | Question→Last sum | Combined sum | Combined overlap / Jaccard |
| --- | --- | --- | --- | --- | --- |
| Early eight | 0, 1, 2, 3, 4, 5, 6, 7 | 64.758162 | 0.984186 | 65.742347 | 5/8; 0.454545 |
| Centered middle eight | 12, 13, 14, 15, 16, 17, 18, 19 | 3.269431 | 9.192261 | 12.461692 | 0/8; 0.000000 |

The **5,000 paired question-bootstrap draws**, seed 0, resample all 989 examples
with replacement and recompute all three rankings. Two fixed disjoint halves
contain 494 and 495 questions. Fixed position controls are not reselected.

| Ranking | Median overlap with own full top-8 | Fifth percentile | Exact-set frequency | Half overlaps |
| --- | --- | --- | --- | --- |
| Image→Question | 7/8 | 6/8 | 24.66% | 8/8, 7/8 |
| Question→Last | 8/8 | 8/8 | 99.84% | 8/8, 8/8 |
| Combined | 8/8 | 6/8 | 50.06% | 7/8, 8/8 |

Nonzero overlap-distribution counts, written as `shared layers: draws`:

- Image→Question: `5: 4, 6: 1414, 7: 2349, 8: 1233`.
- Question→Last: `7: 8, 8: 4992`.
- Combined: `5: 40, 6: 352, 7: 2105, 8: 2503`.

Image→Question's eighth/ninth layers are **9/6**, with score gap **0.012221**
and bootstrap gap interval **[−0.358266, 0.403042]**. Layers 6, 8, and 9 each
enter about 46–50% of draws. Its two-layer departure from combined selection
is therefore a weaker experimental distinction. Combined's **11/12** cutoff
is also ambiguous: gap **0.029466**, interval **[−0.185901, 0.240529]**.
Question→Last's **14/17** cutoff has a wider gap, **0.440058**, with interval
**[0.204379, 0.664462]**. The Question→Last/combined overlap remains exactly
3/8 in **99.48%** of paired resampled-ranking comparisons.

The screen declared before ranking/bootstrap requires at least three replacements
versus combined and stability of median ≥7/8, fifth percentile ≥6/8, and both
halves ≥7/8. Only Question→Last passes both difference and stability. These
are practical screening choices, not a significance test.

Interpretation remains conditional on this calibration cohort and overlapping
nine-layer windows. Scores are indexed by window center, not isolated-layer
causal effects. Image→Question layers 16–31 have tied clipped zero scores;
their lower-index-first tail order is a deterministic convention. Window smoothing
and position effects remain reasons to retain the fixed controls in any later
training comparison. Neither pathway's downstream superiority is established.

[All 32 scores/ranks](../output/pathway_specific/scienceqa/train_correct_window9_n989_bootstrap5000_seed0_20261029/layer_rankings.csv) · [importance overlay](../output/pathway_specific/scienceqa/train_correct_window9_n989_bootstrap5000_seed0_20261029/importance_overlay.png) · [selected layers](../output/pathway_specific/scienceqa/train_correct_window9_n989_bootstrap5000_seed0_20261029/layer_selection_comparison.png) · [bootstrap stability](../output/pathway_specific/scienceqa/train_correct_window9_n989_bootstrap5000_seed0_20261029/bootstrap_stability.png) · [overlap counts](../output/pathway_specific/scienceqa/train_correct_window9_n989_bootstrap5000_seed0_20261029/overlap_distributions.csv) · [full numerical results](../output/pathway_specific/scienceqa/train_correct_window9_n989_bootstrap5000_seed0_20261029/analysis.json).

The original analysis ran on `dev` at `8734991cd5c80eff45b09eb954d6d849ef495d58`. Its [manifest](../output/pathway_specific/scienceqa/train_correct_window9_n989_bootstrap5000_seed0_20261029/manifest.json) records calibration IDs, input hashes, scoring implementation and screening rules. Original numerical artifacts are preserved; duplicate generated reports/PDFs and the regenerable bootstrap-score binary are excluded from this branch.

The CPU-only [analysis](../peft_lora/methods/pathway_specific/analyze.py) accepts a fresh `--output-dir` and recomputes all 5,000 draws from the retained full flow CSVs. Fixed bootstrap seed is 0; ties use lower layer indices. Calibration selection uncertainty does not estimate training-seed variation.
