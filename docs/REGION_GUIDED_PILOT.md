# Region-guided flow LoRA pilot

The best two separated eight-layer regions were **0–7 and 9–16**. They share **15/16** layers with existing flow-pruned layers 0–15 (Jaccard **15/17 = 0.882353**), removing layer 8 and adding layer 16. Training was **skipped**: this fails the practical difference screen, which flags overlap of at least 14/16.

| Allocation | Combined importance sum |
| --- | ---: |
| Existing layers 0–15 | 86.3500521601 |
| Two regions | 85.7927326148 |

The region constraint retains 99.3546% of the existing score but does not create a substantially different experiment. It exhaustively compares all **136** ordered region pairs with at least one unselected layer between the two eight-layer blocks; exact objective ties use the smallest pair of starting indices. Independent enumeration and tie/invalid-gap checks are retained in `analysis_validation.json`.

## Scoring and interpretation

The unchanged [original aggregation](../peft_lora/methods/flow_guided/make_ranks.py) uses the full ScienceQA `train_correct` window-9 Image→Question and Question→Last CSVs, with **989 observations at every center**. Raw relative change is `100 * (knockout_probability - base_probability) / base_probability`.

```text
importance[layer] = max(0, -mean(Image→Question relative change))
                  + max(0, -mean(Question→Last relative change))
```

The mean precedes clipping. Higher importance means a larger average probability drop. These are overlapping nine-layer interventions indexed by their centers, not isolated-layer causal effects. The analysis reproduces existing flow-pruned top-16 exactly; no alternative scores or post-hoc region constraints were introduced.

## Adapter validation and artifacts

The [rank map](../configs/ranks/region_pruned16_scienceqa.json) assigns rank 8 to `q_proj` and `v_proj` in the selected layers, zero elsewhere. Zero entries are filtered before constructing adapters; all 32 original base layers remain.

Validation instantiated the full checkpoint architecture on the meta device and initialized actual PEFT adapter tensors on CPU. It verified **2,097,152 trainable parameters**, 32 projection modules / 64 A/B tensors, rank 8, alpha 16, and no trainable base weights or adapters elsewhere. Checkpoint configuration snapshot: `4481d270cc22fd5c4d1bb5df129622006ccd9234`. No pretrained weights, forward pass or training were needed for this count.

[Scores](../output/region_pilot/scienceqa/flow_scores.csv) · [region selection](../output/region_pilot/scienceqa/region_selection.json) · [figure](../output/region_pilot/scienceqa/flow_regions_vs_top16.png) · [input provenance](../output/region_pilot/scienceqa/provenance.json) · [actual tensor validation](../output/region_pilot/scienceqa/adapter_validation.json) · [independent selection checks](../output/region_pilot/scienceqa/analysis_validation.json).

The pilot was prepared on `dev` at `8734991cd5c80eff45b09eb954d6d849ef495d58` on 28 October 2026. Its saved comparison keeps pilot accuracy and difference null because no pilot was trained. Baseline seed 0 was rescored as 1636/2017, with exact IDs/labels matching the held-out questions. The nominal training file has 6,218 rows; the matched existing loop uses 388 full updates over 6,208 rows and drops ten rows.

The historical ledger's 69.16% test zero-shot figure had no verifiable test prediction artifact in that audit. It is not a result produced by this pilot. Duplicate baseline copies, the never-used planned training configuration, archived scratch source, PDF duplicate and unused submission wrappers are excluded here; original active-worktree files remain untouched.

## Offline reproduction

The [analysis script](../peft_lora/methods/region_guided/make_ranks.py) contains the region search and CPU adapter validation. Use fresh output/rank destinations to preserve saved evidence; inspect `--help` for `--output-dir`, `--rank-output`, and `--validate-adapter`. Validation requires the LLaVA environment and cached checkpoint configuration described in [SETUP.md](../SETUP.md). The script never submits training jobs.
