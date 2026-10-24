"""Download ScienceQA parquet shards on the login node; no model/GPU work."""
import json,os
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
root=Path(__file__).resolve().parents[2]
folder=Path.home()/'scienceqa_raw'
revision=HfApi().dataset_info('derek-thomas/ScienceQA').sha
snapshot_download('derek-thomas/ScienceQA',repo_type='dataset',revision=revision,
                  allow_patterns=['data/*.parquet'],local_dir=str(folder),max_workers=2)
files=sorted(folder.glob('data/*.parquet'))
assert len(files)==3
report=dict(status='SCIENCEQA_DOWNLOAD_COMPLETE',dataset='derek-thomas/ScienceQA',revision=revision,
            folder=str(folder),files={p.name:p.stat().st_size for p in files})
(root/'output/diagnostics/scienceqa_download.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
