# AdaLoRA experiment guide

- [Reproduce current ScienceQA runs](reproduction.md): one supported entry point, explicit configs and seed.
- [GQA reproduction](gqa_reproduction.md): frozen implementation, separate from current team methods.
- [ScienceQA findings](scienceqa_findings.md): data/protocol audit, matched-loop control, regularization/pruning ablations and three-seed results.
- [Rank/flow association](rank_flow_alignment.md): exploratory correlation and its limits.
- [Report notes](report_notes.md): method and reporting details.
- [Historical files and recovery](historical_runs.md): pre-cleanup snapshot, provenance and source/result path mappings.

Shared comparison table: [RESULTS.md](../../RESULTS.md). All raw result files remain under output/. Cleaning folders does not change scores or scientific settings. The regularization-off configuration was chosen after exploratory seed-0 analysis; it is labelled accordingly.
