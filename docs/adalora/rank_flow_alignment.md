# Exploratory ScienceQA rank/flow alignment

Reproduce with `python experiments/scienceqa_adalora/check_rank_flow_alignment.py`; numerical output is `output/analysis/adalora/rank_flow_alignment.json`.

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

Job 46525 runs Uniform rank 8, seed 0 through the unchanged AdaLoRA training engine with microbatch 1 / accumulation 16, FP32 adapters and the same training settings/data. A two-update pilot validates row counting, export, fresh reload and four-question generation before the 388-update run and both evaluations can start. Outputs are `output/peft/uniform/scienceqa_matched_engine/job46525/`.

Job 46523 was an unsuccessful pilot: the added diagnostic guard rejected the engine's unlabeled reload-reference forward after two optimizer steps. No full training started. The guard was corrected to inspect bound positional arguments and exclude unlabeled reference forwards, and a CPU regression passed. Historical adapters and training engines were not overwritten.
