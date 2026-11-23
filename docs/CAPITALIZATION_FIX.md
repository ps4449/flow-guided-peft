# Capitalised training answers (GQA)

## The problem

GQA's answers are lowercase (`right`, `yes`, `man`). LLaVA-1.5 answers with a capital first letter: every zero-shot
prediction on GQA testdev that starts with a letter starts with a capital (`Right`, `Yes`, `Man`). The first training
files used GQA's lowercase answers as targets, so fine-tuning first had to teach the model a new letter case:

- The loss at step 10 was about **5.6–6.4** with lowercase targets, against about **1** once the case matches.
- The early updates went into changing the case instead of answering better.

The evaluation already lowercases both the prediction and the answer before comparing them
(`normalize()` in `peft_lora/eval_gqa.py`), so the letter case never changed the score directly. It changed
what the model was trained to produce.

## The fix

`scripts/prepare_gqa_splits.py` now capitalises the first letter of each training answer, to match LLaVA's style:

```python
def capitalize(answer):
    return answer[:1].upper() + answer[1:]
```

`datasets/gqa/train_20000_llava.json` was regenerated with it (commit `f88f43b`, 22 October 2026). The questions,
images and every other setting are the same; only the first letter of the answers changed. Testdev and the
evaluation are unchanged. ScienceQA needs no fix: its answers are option letters (`A`, `B`, ...), which are already capitals.

## How to check a run used it

Line 2 of every training log prints the first training target:

```
Target of sample 0: 'Right</s>'    <- capitalised (fixed)
Target of sample 0: 'right</s>'    <- lowercase (old file)
```

The step-10 loss tells too: about 1 when capitalised, about 5–6 when lowercase.

## Pitfall: the file has to reach the cluster

The code change alone is not enough: the cluster trains on its own copy of `train_20000_llava.json`. Until the
regenerated file was copied there, runs meant to be capitalised still trained on lowercase answers
(the flow-guided seed 0 rerun logged `'right</s>'`). That is why `RESULTS.md` lists the uniform seed 0 rerun (61.36),
the flow-guided seed 0 rerun (61.24) and flow-protected (61.54) under lowercase. After `git pull` or rsync,
check the file: `head -c 300 datasets/gqa/train_20000_llava.json` should show `"value": "Right"`.

## Effect on accuracy (GQA testdev)

| Method | Lowercase | Capitalised | Change |
|---|---|---|---|
| Uniform LoRA | 61.27 (seed 0), 61.36 (seed 0 rerun) | 61.95 ± 0.03 (seeds 0–2) | about +0.6 |
| Flow-guided LoRA | 61.11 (mean of seed 0, seed 0 rerun, seed 1) | 61.79 ± 0.23 (seeds 0–2) | about +0.7 |
| AdaLoRA | 61.44 (seed 0), 61.35 (seed 1) | 62.56 (seed 0), 62.45 (seed 1) | +1.12, +1.10 (same seeds) |

Capitalising helps every method by 0.6–1.1 points, more than most differences between methods, so lowercase and
capitalised runs are kept in separate tables in `RESULTS.md` and only rows in the same table are compared.
The uniform and flow-guided rows used the shared trainer with microbatch 4, which had the batching bug in
[MULTIMODAL_BATCH_FIX.md](MULTIMODAL_BATCH_FIX.md); their exact sizes may change once rerun with that fix.

## AdaLoRA capitalization comparison

The seed 0 comparison changes only the first character of each training answer to uppercase; the remaining data, training settings and evaluation are unchanged. Evaluation lowercases predictions and references before exact matching, so the improvement is not direct credit for matching answer case. The detailed loss and short-step comparisons are for seed 0. Capitalised seeds 0–2 are now complete (62.49 ± 0.06%, sample std); the improvement versus lowercase is not a three-seed paired estimate because lowercase seed 2 was not run. Lowercase seed 1 scored 61.35% versus 62.45% capitalised.

- Full training: 205.4 minutes lowercase versus 201.0 minutes capitalised; this small timing difference does not establish a speed improvement.
- On the fixed exploratory 512-question subset, step 100 accuracy was 61.13% lowercase versus 60.94% capitalised (one question difference). The short comparison provides no clear early learning advantage.
- Over the final 250 training updates, answer/task loss averaged 0.383 lowercase versus 0.342 capitalised, with per-update standard deviations 0.142 versus 0.127. These exclude AdaLoRA regularization and compare different target casing/tokenization; loss alone is not a measure of task accuracy.

The comparison files and loss plot are in the Git tag `archive/nop-experiments-2026-10-26`
(`output/peft/adalora_capfirst/seed0/capitalization_comparison.json`, `output/analysis/adalora/capitalization/`).
How to rerun AdaLoRA: [AdaLoRA-experiments/README.md](../AdaLoRA-experiments/README.md).
