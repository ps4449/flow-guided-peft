import hashlib,importlib.metadata,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
method,seed,run=sys.argv[1:]
manifest=json.loads((ROOT/'experiments/scienceqa_adalora/run_manifest.json').read_text())
for name,digest in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
versions={p:importlib.metadata.version(p) for p in ['torch','transformers','peft','llava']}
expected=json.loads((ROOT/'experiments/adalora_reproduction/source_manifest.json').read_text())['training_environment']
assert versions==expected,(versions,expected)
meta=dict(method=method,seed=int(seed),shared_source_commit='a3060ab',versions=versions,
          manifest=manifest,training_questions=6218,processed_questions_per_epoch=6208,
          total_steps=388,effective_batch=16,microbatch=4 if method=='uniform' else 1,
          gradient_accumulation=4 if method=='uniform' else 16,
          dataset_download=json.loads((ROOT/'output/diagnostics/scienceqa_download.json').read_text()))
folder=ROOT/run;folder.mkdir(parents=True,exist_ok=True)
p=folder/'environment.json'
if p.exists():assert json.loads(p.read_text())==meta,'Resume environment differs'
else:p.write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta),flush=True)
