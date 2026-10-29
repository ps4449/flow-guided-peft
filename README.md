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

This gives approximately the same observed accuracy with half the adapter parameters. It does not establish equivalence, faster inference, or a halving of total model size. See the [full results ledger](RESULTS.md) for main experiments, other methods, and historical runs affected by earlier batching/answer-format issues.

The [exploratory placement update](docs/FLOW_PLACEMENT_EXPERIMENTS_UPDATE.md) explains the subsequent region, window-size, pathway and position-control studies. Stable Question→Last selection did not improve seed-0 accuracy over combined flow. Fixed early-eight evaluation remains pending in this branch snapshot. These follow-ups test the placement hypothesis; they are separate from the three-seed main result. Flow's unique advantage over gradient-based or simple depth selection remains unestablished.

Detailed evidence: [regions](docs/REGION_GUIDED_PILOT.md), [window sensitivity](docs/FLOW_WINDOW_SENSITIVITY.md), [pathway rankings](docs/PATHWAY_SPECIFIC_SELECTION.md), [matched training](docs/PATHWAY_TOP8_SCIENCEQA.md), [paired errors](docs/PATHWAY_TOP8_PAIRED_ERRORS.md), and [early control status](docs/FINAL_TOP8_COMPARISON.md). [SETUP.md](SETUP.md) contains environment and reproduction details. Other teammates' experiments and historical results are preserved.

The upstream analysis also builds on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) and Google's [dissecting factual predictions](https://github.com/google-research/google-research/tree/master/dissecting_factual_predictions).
