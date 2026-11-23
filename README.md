# Information-Flow-Guided Fine-Tuning for Multimodal LLMs

Course project for AI6130 Large Language Models, Nanyang Technological University.

LoRA is usually added to every layer of a model. We ask whether the model's internal flow of information can show where
adaptation is needed. Using attention knockout on LLaVA-1.5-7B, we measure in which layers the image is mixed into the
question, and in which layers the question passes it on to the answer. We then place LoRA only on the layers this map
ranks highest.

## Main results (ScienceQA-IMG test, 2,017 questions, mean of 3 seeds)

| Method | Layers | Trainable parameters | Accuracy (%) |
|---|---|---|---|
| Zero-shot LLaVA-1.5-7B | – | – | 69.16 |
| LoRA, all layers (rank 8) | 0–31 | 4.19M | 81.21 ± 0.41 |
| **Flow-pruned LoRA** | 0–15 (flow layers) | 2.10M | **81.36 ± 0.31** |
| Uniform LoRA, rank 4 (same budget) | 0–31 | 2.10M | 80.52 ± 0.49 |
| Random-pruned LoRA | 16 random | 2.10M | 79.82 ± 0.69 |
| Inverse-pruned LoRA | 16–31 | 2.10M | 72.83 ± 0.37 |
| PiSSA, all layers | 0–31 | 4.19M | 82.37 ± 0.99 |
| **Flow-pruned PiSSA** | 0–15 | 2.10M | **83.06 ± 0.12** |

- **Flow-pruned LoRA matches LoRA on all layers with half the trainable parameters.** Other ways of spending the same
  budget all do worse, on every seed.
- **The image enters the question almost entirely in layer 0.** The question then carries it to the answer through
  the middle layers.
- **The same layers work with PiSSA,** which is both more accurate and more stable across seeds on the flow layers.
- **On GQA, all placements tie.** LLaVA already saw GQA during training, so every rank allocation stays within 1 point
  of the untuned model.

Every run, with per-seed numbers and significance tests, is in [RESULTS.md](RESULTS.md).

## Repository layout

| Path | Contents |
|---|---|
| `InformationFlow.py`, `methods.py`, `utils.py` | Attention knockout (from the upstream project, extended to ScienceQA and single-layer windows) |
| `peft_lora/train_lora.py` | Shared LoRA trainer (`--variant lora/dora/pissa/loraplus`, `--lora-tokens text/image`) |
| `peft_lora/eval_gqa.py` | Evaluation on GQA testdev or ScienceQA test |
| `peft_lora/methods/flow_guided/make_ranks.py` | Turns the knockout results into per-layer ranks (`--keep 16` for flow-pruned, `--inverse`, `--random`) |
| `peft_lora/methods/layer_baselines/make_ranks.py` | Other layer choices (gradient, similarity, spectrum) |
| `configs/ranks/` | Rank files used for every run |
| `AdaLoRA-experiments/` | AdaLoRA runs with their own trainer ([README](AdaLoRA-experiments/README.md)) |
| `scripts/` | Data preparation and TC2 Slurm job scripts |
| `datasets/` | Question files for GQA and ScienceQA (images are downloaded separately) |
| `output/` | Knockout results, training logs, predictions and metrics (adapter weights are not tracked) |
| `vil/` | Plotting scripts |
| `docs/` | Notes on fixes and experiments; `docs/slides/` has the presentation, script and figures |
| `tests/` | Test for the multimodal batch fix |

## Setup and running

[SETUP.md](SETUP.md) has the step-by-step setup for the NTU TC2 GPU cluster: environment, model download, GQA and
ScienceQA data, and the Slurm commands for the knockout, training and evaluation. In short:

```bash
# 1. Knockout on ScienceQA training questions the base model answers correctly
sbatch --export=ALL,DATASET=datasets/SQA_train_correct_ScienceQA.csv,IMAGES=datasets/scienceqa/images,BLOCK="Image->Question" scripts/tc2_informationFlow.sh
sbatch --export=ALL,DATASET=datasets/SQA_train_correct_ScienceQA.csv,IMAGES=datasets/scienceqa/images,BLOCK="Question->Last" scripts/tc2_informationFlow.sh

# 2. Keep the 16 highest-ranked layers
python peft_lora/methods/flow_guided/make_ranks.py --types ScienceQA --split train_correct \
  --flows Image___Question Question___Last --keep 16 --out configs/ranks/flow_pruned16_scienceqa.json

# 3. Train and evaluate
sbatch --export=ALL,RANKS=configs/ranks/flow_pruned16_scienceqa.json,SEED=0,EXTRA="--train-file datasets/scienceqa/train_llava.json" scripts/tc2_train.sh
sbatch --export=ALL,ADAPTER=output/peft/flow_pruned16_scienceqa/scienceqa_train_llava/seed0/adapter,EXTRA="--questions datasets/scienceqa/test_questions.jsonl" scripts/tc2_eval.sh
```

Add `--variant pissa` to `EXTRA` in step 3 for flow-pruned PiSSA.

## Team

Nop Polboon, Priyanka Srinivas, Ananya Tirumala, Swarangi Satpute.

## Acknowledgement

This repository started as a fork of
[Cross-modal Information Flow in Multimodal Large Language Models](https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM)
(Zhang et al., CVPR 2025), whose attention-knockout code we build on. That code in turn builds on
[dissecting_factual_predictions](https://github.com/google-research/google-research/tree/master/dissecting_factual_predictions)
and [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT). The datasets are [GQA](https://cs.stanford.edu/people/dorarad/gqa/index.html)
and [ScienceQA](https://scienceqa.github.io/).

```bibtex
@article{zhang2024cross,
  title={Cross-modal Information Flow in Multimodal Large Language Models},
  author={Zhang, Zhi and Yadav, Srishti and Han, Fengze and Shutova, Ekaterina},
  journal={arXiv preprint arXiv:2411.18620},
  year={2024}
}
```

## License

MIT, as in the upstream project. See [LICENSE](LICENSE).
