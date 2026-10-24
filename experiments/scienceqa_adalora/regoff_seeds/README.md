# Regularization-off ScienceQA seeds 1 and 2

Explicitly requested after the completed seed-0 diagnostic investigation. This is a separate scope: exactly two additional seeds, not an automatic extension of the stopped B/C/D controller.

Reuses B's unchanged engine/config (regularizer 0, pruning 12 -> average 8, alpha24, dropout.05, same data/LR/one epoch/microbatch1 accumulation16). Source/data hashes must match seed0 before training begins. Technical pilot and full run of the exact configuration already passed on seed0; no duplicate short pilots are required for a seed change.

Each job trains 388 steps / 6208 examples, checks row counts, final mask512, zero regularizer and reload, then evaluates validation and merged/unmerged test. The seed2 summary combines exact counts with existing B/seed0 and new seeds1/2. Original regularization-on results are preserved. Main-table method choice should be agreed by the team; disclose that regularizer-off was selected after exploratory seed0/test inspection. Do not present this as guaranteed better than Uniform or flow-guided.

Outputs: output/peft/adalora/scienceqa_regoff_confirm/seed1 and seed2. Seed0 reference: output/diagnostics/scienceqa_closure_seed0_20261024/B/seed0. summary.json reports exact-score mean and sample SD in percentage points.

Submit seed1 as a single 3-hour job and seed2 with afterok dependency and kill-on-invalid-dep. Only these two jobs; no background scientific controller or later training.
