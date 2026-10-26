# Historical files and recovery

Pre-cleanup Git tag: `archive/nop-experiments-2026-10-26` (commit `21b2d6fac1feb6a2e633c5cf1e46af7a71fa6d77`). It contains the complete historical scripts, intermediate probes and original paths. Files removed from the current branch remain recoverable there; no Git history was rewritten.

```bash
git fetch origin tag archive/nop-experiments-2026-10-26
git worktree add ../flow-guided-peft-historical archive/nop-experiments-2026-10-26
```

Use the historical checkout for byte-exact reproduction of past runs and manifest validation. Never replace a completed run's source hash, original output path or configuration to pretend it was produced by the reorganized code. Historical manifests are evidence about the old checkout, not a requirement that removed scripts remain on the current branch. New runs record the new sources.

The frozen GQA source files in experiments/adalora_reproduction remain unchanged. ScienceQA originally had separate engines/wrappers, including batch-affected Uniform, matched-loop Uniform, a three-condition ablation controller and reg-off confirmation wrappers. The current interface consolidates them. Obsolete batch/merge/gradient probe executables and finished login controllers are recovered from this tag if needed. The batch-integrity regression, adapter-reload check and CPU mechanics regression remain supported.

Local untracked scratch files from the publishing checkout were moved outside it to `../cleanup-local-backup-20261026/`; they were never part of the shared branch. They are not required for published results.

See source_path_map.json and result_path_map.json for the explicit old-to-current locations after cleanup. Unchanged source manifests deliberately retain historical paths/hashes. Fields such as adapter/out_dir inside saved artifacts are historical metadata, not live pointers; consult the result map for current storage.
