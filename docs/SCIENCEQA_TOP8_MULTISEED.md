# ScienceQA top-8: seeds 0–2 comparison

Experiment state: **completed**. Required runs only; fixed-early seeds 1/2 remain excluded.

| Allocation | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |
| --- | --- | --- | --- | --- |
| combined | 1605/2017 (79.57%) | 1616/2017 (80.12%) | 1612/2017 (79.92%) | 79.87 ± 0.28% |
| question_last | 1572/2017 (77.94%) | 1594/2017 (79.03%) | 1605/2017 (79.57%) | 78.85 ± 0.83% |
| middle | 1522/2017 (75.46%) | 1538/2017 (76.25%) | 1546/2017 (76.65%) | 76.12 ± 0.61% |

| Seed | Combined − Question→Last pp | Combined − middle pp | Question→Last − middle pp |
| --- | --- | --- | --- |
| 0 | +1.636 | +4.115 | +2.479 |
| 1 | +1.091 | +3.867 | +2.776 |
| 2 | +0.347 | +3.272 | +2.925 |

The strict seed-0 ordering (combined > Question→Last > middle) is preserved in all three seeds.

All runs use the same original checkpoint, rank-8 q_proj/v_proj, 1,048,576 trainable parameters and the exact seed-0 protocol; only seeds/output paths differ.
All predictions are aligned to the same 2,017 unique IDs and exact gold labels with original normalization. Sample SD uses ddof=1 across seeds 0–2.
Three seeds describe training variation with limited precision. Fixed-checkpoint question-bootstrap significance is not training-seed reliability; no statistical superiority is claimed.
Fixed early seed 0 tied combined flow at 1605/2017; its optional additional seeds were not submitted.

Outputs follow `output/peft/<rank-stem>/scienceqa_train_llava/seed<k>/`; logs remain in `slurm/logs/`. Shared dev and historical artifacts are preserved.
[Actual jobs/provenance](../output/pathway_top8_multiseed/submission_manifest.json) · [Seed results](../output/pathway_top8_multiseed/seed_results.csv) · [Three-seed summary](../output/pathway_top8_multiseed/three_seed_summary.csv) · [Matched differences](../output/pathway_top8_multiseed/matched_seed_differences.csv) · [Original audit](../output/pathway_top8_multiseed/preflight.json)
