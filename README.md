# Flow-guided PEFT — AI6130 project

We study whether attention-knockout information flow can guide where to place LoRA adapters in LLaVA-1.5-7B, using ScienceQA and GQA.

[Zhang et al., *Cross-modal Information Flow in Multimodal Large Language Models*](https://arxiv.org/abs/2411.18620) motivate measuring how visual information reaches question tokens and the answer position. This project extends their [analysis code](https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM).
[Zhou et al., *An Empirical Study on Parameter-Efficient Fine-Tuning for MultiModal Large Language Models*](https://aclanthology.org/2024.findings-acl.598/) study PEFT choices, including module placement, in multimodal models. Together these motivate our placement question; neither paper establishes that knockout importance predicts the best LoRA allocation.

**Flow-guided LoRA** assigns adapter ranks using flow importance. **Flow-pruned LoRA** assigns rank 8 to selected layers and no adapters elsewhere. “Pruned” refers to adapter placement: all 32 original transformer layers remain. Both use `q_proj` and `v_proj`.

Our main corrected ScienceQA comparison uses seeds 0–2 and the same 2,017 held-out image questions:

| Allocation | Trainable adapter parameters | Accuracy, mean ± sample SD |
| --- | ---: | ---: |
| Uniform rank 8, all 32 layers | 4,194,304 | 81.21 ± 0.41% |
| Flow-pruned rank 8, layers 0–15 | 2,097,152 | 81.36 ± 0.31% |

This gives approximately the same observed accuracy with half the adapter parameters (2.10M versus 4.19M). Gradient-based selection chose the same layers 0–15, so this comparison does not distinguish flow from gradient-based or simple early-layer selection. It does not establish equivalence, faster inference, or a halving of total model size. See the [full results ledger](RESULTS.md) for main experiments, other methods, and historical runs affected by earlier batching/answer-format issues.

**The follow-up placement ablations are complete.** The [short team update](docs/FLOW_PLACEMENT_EXPERIMENTS_UPDATE.md) follows the region, window-size, pathway and position-control studies. At the eight-layer seed-0 budget of 1,048,576 adapter parameters, combined flow (layers 0–4, 9–11) and fixed early (0–7) each reached **79.57% (1605/2017)**; Question→Last reached **77.94%**, and fixed middle **75.46%**. The [final paired comparison](docs/FINAL_TOP8_COMPARISON.md) found no observed combined-flow advantage over this early control: difference 0.000 percentage points, 95% paired-question interval [−0.892, +0.892], exact McNemar p = 1.0.

These single-seed follow-ups are separate from the three-seed main result. The tie does not establish statistical equivalence or reliability across training seeds. Flow's unique advantage over gradient-based or simple depth selection remains unestablished.

Detailed evidence: [regions](docs/REGION_GUIDED_PILOT.md), [window sensitivity](docs/FLOW_WINDOW_SENSITIVITY.md), [pathway rankings](docs/PATHWAY_SPECIFIC_SELECTION.md), [matched training](docs/PATHWAY_TOP8_SCIENCEQA.md), [paired errors](docs/PATHWAY_TOP8_PAIRED_ERRORS.md), and [final four-model comparison](docs/FINAL_TOP8_COMPARISON.md). [SETUP.md](SETUP.md) contains environment and reproduction details. Other teammates' experiments and historical results are preserved.

The upstream analysis also builds on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) and Google's [dissecting factual predictions](https://github.com/google-research/google-research/tree/master/dissecting_factual_predictions).
