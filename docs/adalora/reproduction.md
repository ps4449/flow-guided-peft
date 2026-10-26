# ScienceQA AdaLoRA reproduction

The supported ScienceQA interface and final configs are documented here; the frozen GQA interface stays separate in [gqa_reproduction.md](gqa_reproduction.md). Use the pre-cleanup tag for byte-exact historical scripts; current code preserves the training operations and adds a clearer interface.

Environment: Python 3.10; torch 2.1.2+cu121; PEFT 0.12.0; transformers 4.40.0.dev0; LLaVA 1.7.0.dev0 and LLaVA-NeXT commit bce12e479bc4dfee2b9c50c88137b01ff51bd483. Model: liuhaotian/llava-v1.5-7b, revision 4481d270cc22fd5c4d1bb5df129622006ccd9234. Prepare ScienceQA images following the team's [SETUP.md](../../SETUP.md). No cleanup command submits training.

All full ScienceQA arms: 6218 training rows, first 6208 shuffled examples per epoch, 388 optimizer updates, seed0/1/2, microbatch1 / accumulation16, LR2e-4, warmup .03, BF16 base and FP32 adapters, alpha24/dropout.05. Validation has2097 image questions and test2017. Targets are uppercase option letters. The evaluator normalizes via strip().rstrip('.').lower(), without a new answer extractor.

Configs: regularization_off.json (main post-hoc ScienceQA configuration), regularization_on.json (original reference), no_pruning.json (regularizer .5 with rank12 retained), neither.json (no regularizer, rank12 retained). Fixed-rank-12 arms use a larger budget and are mechanism controls only. All begin at rank12; pruning-enabled arms end at total rank512.

Run commands, environment checks and CPU tests are listed below by the final cleanup commit. Use a new output directory for any new experiment; do not overwrite completed artifacts. No new GPU experiments are needed to use the published results.
