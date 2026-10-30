# Flow placement: five completed experiments

Our earlier sixteen-layer result suggested that sparse LoRA placement can save adapter parameters. It also selected the early half of the model, so we asked whether information flow offers guidance beyond a simple early-layer rule. These follow-ups change adapter placement; all 32 base transformer layers remain.

1. **Region-guided selection.** We tried choosing two separated eight-layer regions instead of individual layers. The selected regions, 0–7 and 9–16, shared 15/16 layers with the existing allocation. We skipped training because this supplied almost the same comparison. [Detailed selection evidence](REGION_GUIDED_PILOT.md).

2. **Knockout-window sensitivity.** We tested whether the original rankings depended on blocking nine layers at a time. Widths three and nine on the same 200 calibration questions changed the rankings (Spearman 0.686), but the narrower-window top sixteen still shared 15/16 historical early layers and failed the stability screen. Window scores describe grouped interventions, not individual layers. This did not justify another near-identical training run. [Detailed window evidence](FLOW_WINDOW_SENSITIVITY.md).

3. **Pathway-specific selection.** We separated Image→Question from Question→Last using 989 calibration examples. Question→Last selected layers 9–16, a distinct allocation that recurred in 99.84% of 5,000 paired bootstrap draws. This gave us a stable candidate to test; calibration stability alone did not predict adaptation quality. [Detailed pathway evidence](PATHWAY_SPECIFIC_SELECTION.md).

4. **Matched pathway training.** We compared Question→Last (9–16) and fixed middle layers (12–19) with the existing combined-flow top-eight baseline (0–4, 9–11), using seeds 0–2 and 1,048,576 adapter parameters each. Combined flow scored higher than both alternatives in every matched seed. The [three-seed results and protocol](SCIENCEQA_TOP8_MULTISEED.md) contain the numerical table. Three seeds provide limited evidence about training variability.

5. **Fixed early-layer control.** To test whether early placement explained the combined-flow result, we trained fixed layers 0–7 for seed 0. Both allocations reached **1605/2017 (79.57%)**, but each uniquely answered 42 questions correctly. Fixed early seeds 1/2 were not run. The [final paired comparison](FINAL_TOP8_COMPARISON.md) explains why this tie does not establish equivalence.

These experiments support sparse adapter placement in this setting. They establish no unique advantage of flow over simple early-layer selection. Paired-question tests describe uncertainty for fixed checkpoints; they do not measure training-seed reliability.
