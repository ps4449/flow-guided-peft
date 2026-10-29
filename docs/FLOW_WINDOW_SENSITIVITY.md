# Information-flow window-size sensitivity

The original experiment ran on `dev` at commit
`8734991cd5c80eff45b09eb954d6d849ef495d58`. It measures attention knockouts
without LoRA. Uniform configurations, historical results, and the skipped
two-region pilot are unchanged. No training jobs are submitted.

**Completed on 28 October 2026:** both window-3 flows contain all 200 matched
questions. Slurm jobs **48456** and **48457** completed with exit code `0:0`.
Window size changes the top-8 substantially, but the new top-16 retains **15/16**
historical layers and fails the frozen stability rule. **No new rank-8 config
was prepared, and no training was submitted.** The completed report, all 32
scores, plots, and comparison are in the isolated continuation directory:

```text
output/information_flow/llava_v1_5_7b/ScienceQA/window_sensitivity/train_correct_n200_seed0/resume_20261028/
```

Start with `comparison.json`, `importance_scores.csv`,
`importance_overlay.png`, and `selection_stability.png` there.

## Boundary audit

`InformationFlow.py:455` builds the actual layer list with:

```python
range(max(0, center - window // 2),
      min(32, center - (-window // 2)))
```

Negative integer division is floored: `-9 // 2 == -5`. The end is exclusive.
For odd widths this is a centered window clipped at the model boundaries;
the implementation does not shift boundary windows to keep a constant width.
The runner evaluates the original expression's AST rather than maintaining a
second implementation of its indexing.

| Center | Window 9 (inclusive) | Actual width | Window 3 (inclusive) | Actual width |
| --- | --- | --- | --- | --- |
| 0 | 0–4 | 5 | 0–1 | 2 |
| 1 | 0–5 | 6 | 0–2 | 3 |
| 2 | 0–6 | 7 | 1–3 | 3 |
| 3 | 0–7 | 8 | 2–4 | 3 |
| 4 | 0–8 | 9 | 3–5 | 3 |
| 5 | 1–9 | 9 | 4–6 | 3 |
| 6 | 2–10 | 9 | 5–7 | 3 |
| 31 | 27–31 | 5 | 30–31 | 2 |

At centers **4 → 5**, a nine-layer intervention drops layer **0** and adds
layer **9**; its width stays nine. In the original 989-question results, the
Image→Question mean raw relative probability change goes from approximately
**−11.9581% to −1.3824%**. The combined importance goes from approximately
**11.9581 to 1.6858**. Thus the cliff aligns with layer 0 leaving the window,
rather than the window becoming shorter at center 5.

This is not an isolated-layer ablation: layer 9 also enters, and the intervention
can have nonlinear interactions. The existing data alone cannot assign the
entire difference to layer 0. Under window 3, layer 0 instead leaves between
centers **1 → 2**. The complete windows are saved in `boundary_audit.json`.

## Frozen matched calibration subset

The initial submission used the following directory; the curated branch retains its complete `resume_20261028/` continuation:

```text
output/information_flow/llava_v1_5_7b/ScienceQA/window_sensitivity/train_correct_n200_seed0/
```

The source is the existing ScienceQA `train_correct` Image→Question and
Question→Last window-9 CSVs, plus `datasets/SQA_train_correct_ScienceQA.csv`.
Eligibility requires a correct saved prediction, one row per layer for all
32 layers in both flows, matching answers/questions/images, and an available
training image. **989** questions satisfy these requirements.

IDs are ordered by `SHA256('flow-window-sensitivity-v1|0|question_id')`, with
the question ID breaking a hash tie. The first **200** are selected. Subject
counts are **122 natural science, 73 social science, and 5 language science**.
This is a deterministic sample of the existing calibration cohort, rather
than a new test or validation set.

`subset_manifest.json` freezes IDs, sources, settings, hashes, and screening
criteria before the new measurements. `calibration_questions.csv` preserves the
original prompt column verbatim. `window9/*.csv` contains the actual saved rows
for these exact 200 questions; window 9 is not rerun in full. Input and image
hashes are saved in `input_provenance.json`.

## Measurement protocol and actual jobs

`scripts/flow_window_sensitivity.py measure` calls the unchanged original
`CustomDataset`, `create_data_loader`, `run_original`, `blockdesc2range`, and
`trace_with_attn_block_llava` functions. It uses the original model builder's
**fp16** default, rather than the PEFT evaluator's bf16 loader.

The checkpoint is `liuhaotian/llava-v1.5-7b`, with `vicuna_v1`, unchanged RGB
image processing, batch size one, temperature zero, no sampling, no top-p,
one beam, cache enabled, base generation limit 128, and knockout generation
limit one. The original baseline first-token probability remains the scored
target after attention links are blocked. Actual GPU, package versions,
attention backend, dtype, and job ID are recorded per flow.

Every current base probability must match its saved window-9 reference within
`rtol=0.001, atol=0.00001`. The first two examples per flow additionally repeat
window-9 knockouts at centers 0, 4, and 5 under that same tolerance. A parity
failure stops the job and is saved explicitly. These few repeats check numerical
compatibility and are not a replacement window-9 experiment.

Measurement jobs **48389** (Image→Question) and **48390** (Question→Last) were
actually submitted to TC2's `MGPU-TC2` partition. The user GPU quota permits one
running GPU job, so the second waits for the first. The six-hour job scripts
request one GPU, four CPUs, and 30 GB RAM. Completed questions are flushed to
CSV and can be resumed. Partial results are not accepted by the final analyzer.

The initial-run artifacts and scheduler logs remain in the original TC2 worktree. This branch retains the completed continuation and its recovery provenance, not duplicate frozen inputs or failed partial CSVs.

### Recovery on 28 October 2026

The earlier chat ended at 17:23:27 Singapore time with the recorded service
error `server_overloaded` ("Selected model is at capacity"). Its Slurm jobs
continued independently, then both exited with code `1:0`: job 48389 preserved
174 complete Image→Question examples, while job 48390 produced no measurements.
The logs do not identify their failure cause; Slurm does not report an OOM or
node failure. The original output directory, partial CSV, parity evidence,
metadata, and scheduler logs are retained unchanged.

Continuation artifacts are isolated in `train_correct_n200_seed0/resume_20261028/`.
All frozen source, prompt, and image hashes were checked before copying the
inputs and complete partial measurements there. `resume_provenance.json`
records hashes of the original artifacts and all existing rank configurations.
Measurement-only jobs **48456** (resume Image→Question at 174/200) and **48457**
(Question→Last, dependent on 48456 succeeding) used the original TC2 continuation wrapper, which remains in the active worktree. The absolute existing Python
environment avoids a separate Conda activation step, and the batch script logs
its exit code explicitly. Each allocation repeats window-9 parity checks on its
first two newly measured questions, including the resumed allocation.

The continuation's `jobs.json` and `window3/*_run.json` record
its actual progress. Its scheduler logs use `flow_window3_resume_<jobid>`.
Run the analyzer with `--output-dir` pointing to the continuation directory;
it rejects partial measurements. No training jobs are submitted.

Both continuation jobs completed successfully. Job 48456 ran from **19:52:30
to 19:56:49**, and job 48457 from **19:56:49 to 20:19:46**, all Singapore time.
Every flow now has **200 questions × 32 centers = 6,400 rows**. All **400**
current baseline probabilities match the saved window-9 values exactly. The
**18** repeated window-9 knockouts also match exactly: 12 Image→Question checks
(including the original and resumed allocations) and six Question→Last checks.
Both use fp16 and the SDPA attention backend. The 174 original questions remain
byte-for-byte as the prefix of the continuation's Image→Question CSV.

## Scoring and stability criteria

The existing definition is reused, averaging over questions before clipping:

```text
importance[layer] = max(0, -mean(Image→Question relative diff first))
                  + max(0, -mean(Question→Last relative diff first))
```

Higher scores mean greater importance. Ties select the lower layer index.
Each width produces a separate 32-element score vector. The analyzer compares
top-8 and top-16 indices, overlap and Jaccard similarity, and tie-aware Spearman
rank correlation. No p-value assumes independence between overlapping centers.

Stability uses **2,000 paired question-bootstrap resamples** (NumPy seed 0),
with the same sampled questions across both flows and both widths. The mean,
clipping, ranking, and selection are recomputed within every resample. This
preserves the dependence among measurements rather than resampling layers or
flows independently. The two fixed, disjoint 100-question halves also produce
separate selections. These checks measure sensitivity within this calibration
cohort and do not establish external generalization.

Before window-3 measurement, these practical screening rules were frozen:

- Meaningful difference: the window-3 top-16 shares **at most 13/16** layers
  with both the matched window-9 top-16 and the historical layers 0–15.
- Stable top-16: bootstrap median overlap with its full-sample selection is
  **at least 14/16**, the fifth percentile is **at least 12/16**, and each fixed
  half shares **at least 14/16** with the full-sample selection.

These are screening rules, not a statistical significance test. A new rank-8
configuration is prepared only if both difference and stability pass. If
prepared, actual adapter tensors are counted using the unchanged PEFT trainer,
requiring **2,097,152** trainable parameters and preservation of all 32 base
layers. The new file would be `configs/ranks/flow_window3_pruned16_scienceqa.json`;
its existence and validation must be confirmed from the final results, not
inferred from this planned pathname. No training is submitted.

## Matched window-9 analysis

The matched-200 top-8 is **0, 1, 2, 3, 4, 9, 10, 11**. Its top-16 is
**0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16**: it shares
15/16 historical selected layers, replacing 6 with 16. Its top-16 bootstrap
median overlap is **15/16**, fifth percentile **14/16**, and both fixed halves
share **15/16** with the full-sample selection. These are calculated from the
reused measurements; see `window9_analysis.json`.

## Completed window-3 comparison

The narrower intervention moves the Image→Question cliff to centers **1 → 2**:
within window 3, mean relative change goes from **−11.6575% to −0.8376%**.
That transition removes layer 0 and adds layer 3. It supports the explanation
that the original center-4/5 cliff follows layer 0's inclusion in the window;
it does not isolate layer 0's individual causal contribution. These are different
interventions, so raw effect sizes across widths are not treated as equivalent.

| Width | Top-8 selected indices | Top-16 selected indices |
| --- | --- | --- |
| 9 | 0, 1, 2, 3, 4, 9, 10, 11 | 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 |
| 3 | 0, 1, 2, 7, 11, 12, 13, 14 | 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18 |

The cross-window top-8 overlap is **4/8**, Jaccard **0.3333**. The top-16
overlap is **14/16**, Jaccard **0.7778**: window 3 replaces matched-window-9
layers **3 and 16** with **6 and 18**. Against the historical 0–15 selection,
window 3 retains **15/16 (93.75%)**, replacing only **3 with 18**; Jaccard is
**0.88235**. Neither selection has a tie at its cutoff.

Spearman correlation across the 32 scores is **0.686102**. The paired-question
bootstrap percentile interval is **[0.362900, 0.870255]**, with bootstrap median
**0.649658**. No independent-layer p-value is used.

| Width | Selection | Bootstrap median overlap | Fifth percentile | Fixed-half overlaps with full |
| --- | --- | --- | --- | --- |
| 9 | Top-8 | 7/8 | 5/8 | 8/8, 6/8 |
| 3 | Top-8 | 7/8 | 5/8 | 7/8, 5/8 |
| 9 | Top-16 | 15/16 | 14/16 | 15/16, 15/16 |
| 3 | Top-16 | 14/16 | 11/16 | 13/16, 13/16 |

The window-3 top-16 fails the frozen difference criterion: overlap must be
≤13/16 with both references, while actual overlaps are 14/16 and 15/16. It
also fails stability: fifth-percentile overlap is 11/16 instead of ≥12/16,
and each half has 13/16 instead of ≥14/16. Thus this ranking **does not justify
a genuinely different, stable 16-layer training experiment**. Window size
does affect the ordering and top-8 allocation; that finding alone does not
justify training another near-identical top-16 model.

No `configs/ranks/flow_window3_pruned16_scienceqa.json` is created. The
conditional 2,097,152-parameter candidate gate is not reached. All original
experimental artifacts and pre-existing rank configurations are preserved.

## Reproduction

The original preparation/submission commands, from the repository root, were:

```bash
conda activate llava
python scripts/flow_window_sensitivity.py prepare
mkdir -p slurm/logs
sbatch --parsable --export=ALL,FLOW=Image___Question scripts/tc2_flow_window_sensitivity.sh
sbatch --parsable --export=ALL,FLOW=Question___Last scripts/tc2_flow_window_sensitivity.sh
```

To analyze the completed, isolated continuation:

```bash
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR=/tmp/flow-window-sensitivity-analysis
python peft_lora/methods/window_sensitivity/analyze.py \
  --output-dir output/information_flow/llava_v1_5_7b/ScienceQA/window_sensitivity/train_correct_n200_seed0/resume_20261028
```

`--window9-only` analyzes the reused reference while measurements run. The full
analyzer requires complete matched data and parity evidence, saves overlays
and bootstrap inclusion-frequency plots, and records the actual decision in
`comparison.json`. The overlay uses within-window normalized importance and
ranks; raw signed effects are presented in separate width panels with independent
axes. **Different window widths are different interventions. Their raw knockout
magnitudes are not directly equivalent**, even with matched questions.

## Curated reproduction

The retained [measurement runner](../scripts/flow_window_sensitivity.py) reproduces the original inference protocol in a configured LLaVA environment; running measurements requires GPU allocation and is not part of this curation. The [CPU analyzer](../peft_lora/methods/window_sensitivity/analyze.py) defaults to the complete continuation directory. For an offline rerun, copy the retained continuation to a fresh destination and pass it with `--output-dir`; never analyze in place over the saved evidence. Numerical rescoring verifies the frozen CSVs and original analysis sources. External LLaVA code and image fingerprints remain recorded but are not revalidated unless `--verify-environment` is supplied. This distinction does not change any scores or parity evidence.

The curated branch excludes duplicate PDF/generated Markdown reports, transient status/log files, failed partial measurements, duplicated initial inputs and Slurm wrappers. Complete window-3 measurements, matched window-9 rows, frozen prompts/IDs, parity records, screening criteria and original job provenance are retained.
