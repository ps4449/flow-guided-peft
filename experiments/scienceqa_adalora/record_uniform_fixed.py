import hashlib,json,sys,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
folder=ROOT/sys.argv[1]
assert not folder.exists(), 'New corrected run must not overwrite existing artifacts'
folder.mkdir(parents=True)
paths=['experiments/scienceqa_adalora/train_uniform_modalities_fixed.py','experiments/scienceqa_adalora/batch_guard.py','experiments/scienceqa_adalora/common.py','experiments/scienceqa_adalora/eval_scienceqa.py','experiments/scienceqa_adalora/eval_scienceqa_unmerged.py','configs/ranks/uniform.json','datasets/scienceqa/train_llava.json','datasets/scienceqa/test_questions.jsonl']
meta=dict(purpose='Uniform ScienceQA seed0 rerun with explicit modality per sample and per-forward batch-size assertion',
 corrected_batch=4,accumulation=4,optimizer_updates=388,expected_language_supervised_samples=6208,
 source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
 versions={p:importlib.metadata.version(p) for p in ['torch','transformers','peft','llava']},
 old_run_preserved='output/peft/uniform/scienceqa_train_llava/seed0',
 evaluation='Both original merged protocol and unmerged diagnostic, separate directories')
(folder/'environment.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta),flush=True)
