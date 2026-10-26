# LLaVA-NeXT multimodal batch integrity fix

## Confirmed issue

The LLaVA-NeXT revision used for Nop's runs (`bce12e479bc4dfee2b9c50c88137b01ff51bd483`) defaults to `modalities=["image"]`. Multimodal preparation truncates lists using `zip(new_input_embeds, modalities)` and the analogous labels operation. The shared training collator omitted modalities, so a microbatch of four was reduced to one language-model sample, silently.

The fix supplies `modalities=["image"] * len(batch)` in the shared `peft_lora/train_lora.py` collator. It applies to both GQA and ScienceQA, and to uniform, flow-guided and inverse allocations using that collator. No frozen historical implementation is overwritten. Predictions and scores already recorded are not erased.

Runtime confirmation on the installed revision: default batch4 yielded embedding/label batch1 and two supervised answer tokens; explicit four modalities yielded batch4 and eight answer tokens. A no-update gradient probe using the same 16 examples reduced the relative gradient difference between 4x4 and 1x16 accumulation from 46.1% to 2.3% with explicit modalities. This demonstrates the batching defect, not that AdaLoRA must outperform any baseline.

## Check before deciding which runs to repeat

1. Inspect the actual training configuration and source used by each run, not just the method name. Microbatch1 is unaffected by this particular truncation. Accumulation does not recover discarded rows.
2. On the same environment/LLaVA checkout, inside a GPU allocation, run:

```bash
python scripts/check_multimodal_batch.py --train-file datasets/gqa/train_20000_llava.json --batch-size 4 --reproduce-legacy
# Or point --train-file to datasets/scienceqa/train_llava.json.
```

This imports the current shared collator, runs multimodal preparation without training, verifies row and supervised-token counts, and optionally reproduces the legacy omitted-modalities path. It prints the installed `llava_arch.py` hash. Run `python -m unittest discover -s tests -p 'test_multimodal_collate.py' -v` for the lightweight CPU collator regression.

3. If the installed version drops rows, multi-sample-microbatch runs do not represent the intended effective-batch/data exposure. Rerun affected comparisons under the corrected collator and new output paths; do not resume old checkpoints into the corrected training protocol. Matching corrected controls matters even if an old score looks better.
4. If the installed implementation already handles omitted modalities correctly, that run is not affected by this defect. Check teammate revisions before assuming a universal rerun.

## Known local effects and ongoing correction

Nop's GQA/ScienceQA AdaLoRA runs used microbatch 1/accumulation16 and are unaffected by this row truncation. His original ScienceQA Uniform seed0 used 4x4 and therefore only four language-supervised examples per update rather than sixteen (1552 instead of 6208 over 388 updates). Evaluation was one question at a time, so the 77.64% score remains a real score for that differently trained artifact, not a valid intended-batch comparison.

Corrected ScienceQA Uniform seed0 completed separately at `output/peft/uniform/scienceqa_modalities_fixed/seed0` (job 46443), with a guard checking every forward's input/output row count. Its `batch_integrity.json` confirms 6208 language rows; merged accuracy is 80.86%. This note does not classify other members' historical GQA runs without their configurations/revisions. No other reruns were launched by this fix.

BF16 merged versus unmerged evaluation differences are a separate issue. Use one consistent evaluation protocol across methods; this training fix does not change the evaluator.
