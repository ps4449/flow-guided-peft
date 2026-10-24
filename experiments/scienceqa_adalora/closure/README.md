# Bounded ScienceQA seed-0 mechanism investigation

Authorized scope: **three new full training conditions, B/C/D, then stop**. No search over learning rates, epochs, seeds or additional methods. Existing AdaLoRA seed 0 is A. Existing matched-loop Uniform is a separate reference. This is an exploratory mechanism diagnosis, not a replacement for the matched-budget main benchmark.

| Arm | Orthogonal regularization | Rank pruning | Initial / final rank sum |
| --- | --- | --- | --- |
| A (existing) | .5 | enabled | 768 / 512 |
| B | 0 | enabled | 768 / 512 |
| C | .5 | disabled | 768 / 768 |
| D | 0 | disabled | 768 / 768 |

All start with the same rank-12 SVD parameterization, alpha 24, dropout .05, seed0, model/data, optimizer settings and microbatch1 / accumulation16. C/D retain rank12: they have a larger final budget and must not be presented as equal-budget competitors. They test removal of the pruning/capacity-reduction package, not placement alone.

## Predeclared checks and stop rule

1. CPU mechanics regression on a tiny model with 32 q/v layers checks all four switch combinations, identical initial adapter tensors, active/zero regularizer, final rank sums, finite loss and full-shape export/reload. It passed before submission.
2. Each GPU arm must pass a 20-update pilot on 320 examples: regularizer switch, rank schedule, input/label counts, fresh-process adapter reload and generation on four validation questions. No pilot score determines whether an arm is selected; passing means technical correctness only.
3. Each full arm trains one epoch: 388 updates, 6,208 of 6,218 examples in the same shuffled order, dropping the last ten as in the existing protocol. Runtime row/token counts must match. Checkpoint/source/config hashes are preserved.
4. Evaluate final validation (2,097) and test (2,017), with merged and unmerged test evaluations for every arm. A's missing validation evaluation is added without retraining. Validation is the primary diagnostic comparison; previously inspected test results remain exploratory. No best-arm selection or new tuning follows these results.
5. The finite login-node controller submits one arm at a time and stops after B/C/D complete, or stops on the first failed job. It does not launch extra seeds or follow-up experiments. Repairs may be needed for an execution failure, but no new scientific condition is authorized by this pipeline.
6. At completion, `final_summary.json` contains all outcomes and paired descriptive contrasts (removing regularization with/without pruning, and removing pruning with/without regularization). Report what these contrasts resolve, and leave any unresolved parameterization/scaling or seed-dependence as limitations. A single seed cannot establish a universal cause or statistical significance. **No further experiment is automatically required to close this diagnostic scope.**

## Implementation details

`train_engine.py` is a separate copy of the published ScienceQA engine. Historical/GQA engines are unchanged. A diff shows only diagnostic switch validation, allowing target rank12 for the explicit larger-budget controls, bypassing the PEFT regularizer when off, skipping the allocator when pruning is disabled, and row-count instrumentation. With pruning disabled, rank masks remain all true and all SVD components continue training.

PEFT 0.12.0 raises for `orth_reg_weight=0` in its tuner forward. Off arms therefore call the same underlying model's task-loss forward, still capture task loss and still use the allocator when pruning is enabled. CPU regression verifies this is zero added regularization and export/reload is equivalent. It is not a switch to static A/B LoRA.

`check_mechanics.py` runs on CPU; `validate_arm.py` checks completed artifacts. Standard package dependencies are those of the existing llava_shared environment. tc2_arm.sh has the original TC2 working directory/environment; adapt those paths for another account.

## Actual pipeline

Output root: `output/diagnostics/scienceqa_closure_seed0_20261024`

First arm B: job **46572**. C/D are submitted by the login-node controller only after the previous arm completes successfully (TC2 permits only two submitted jobs and one GPU per user).

Controller log: `slurm/logs/scienceqa_closure_controller.log`.

State file: `<output root>/pipeline.json`.

Do not start another controller for the same output root. At completion status is `complete`; an error is `stopped`. No live data/state files or model weights are embedded in the source publication.
