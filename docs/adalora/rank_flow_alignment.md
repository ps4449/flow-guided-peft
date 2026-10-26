# Exploratory ScienceQA rank/flow alignment

Reproduce with `python experiments/scienceqa_adalora/analysis/rank_flow_alignment.py`; numerical output is `output/analysis/adalora/rank_flow_alignment.json`.

The available ScienceQA flow CSVs measure **Question -> Last**, with centered/clipped window 9, on train_correct and train_wrong subsets (989 and 977 retained question IDs). Scores are negative mean relative probability change: larger means stronger probability drop under blocking. AdaLoRA ranks come from final masks for seeds 0–2. Neither ranks nor scores were fitted to maximize this relationship.

Spearman correlations using rank at each window's central layer:

| Subset | Adapter ranks | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- | --- |
| Correct | q | .662 | .663 | .528 |
| Correct | v | .426 | .519 | .399 |
| Correct | q+v | .622 | .609 | .548 |
| Wrong | q | .601 | .623 | .398 |
| Wrong | v | .480 | .496 | .312 |
| Wrong | q+v | .596 | .591 | .447 |

A same-window rank-sum aggregation is also supplied as a sensitivity description. Do not select whichever produces the largest correlation. Adjacent window scores overlap and are not independent isolated-layer effects; no naive 32-independent-layer significance claim is made. Shared broad depth trends could explain association. This is an exploratory comparison after viewing the results.

These results support the narrower statement that independently trained AdaLoRA ranks align positively with the available Question-to-Last blocking profile in all three seeds. They do not prove the image flow direction is correct, establish isolated causal layer importance, or show that assigning ranks according to this profile improves task accuracy. Image-to-Question measurements, when available, need a separate explicit analysis.

The team's already planned corrected flow-guided versus uniform and inverse controls across seeds are the appropriate test of whether using the signal actually helps allocation. A stronger association claim could additionally use an autocorrelation-aware null and checks against a generic depth trend; those tests have not been run here. No extra GPU training was needed for this correlation audit.

## Matched-loop control

The completed matched-loop Uniform job46525 reached81.21% merged /81.31% unmerged. It is a separate pipeline control, not evidence that flow-derived ranks improve accuracy. See [ScienceQA findings](scienceqa_findings.md). The unsuccessful first pilot46523 and its guard repair remain recoverable in the historical snapshot; no additional run is requested by this analysis.
