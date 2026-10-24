"""CPU checks for pilot/full ablation artifacts; does not select the best condition."""
import json,math,sys
from pathlib import Path
arm,folder,stage=sys.argv[1:];p=Path(folder)
c=json.loads((p/'train_config.json').read_text());rows=list(map(json.loads,(p/'train_log.jsonl').read_text().splitlines()))
reg=0. if arm in ['B','D'] else .5;expected=512 if arm=='B' else 768
steps=20 if stage=='pilot' else 388;count=320 if stage=='pilot' else 6208
assert c['total_steps']==steps and c['seed']==0 and c['batch_size']==1 and c['grad_accum']==16
assert c['train_file']=='datasets/scienceqa/train_llava.json' and c['limit']==(320 if stage=='pilot' else None)
assert [r['step'] for r in rows]==list(range(1,steps+1))
assert all(math.isfinite(r['task_loss']) and math.isfinite(r['grad_norm']) for r in rows)
assert all(abs(r['regularization'])<1e-7 for r in rows) if not reg else all(r['regularization']>0 for r in rows)
assert c['final_rank']['retained_rank_sum']==expected
if arm!='B':assert all(r['retained_rank_sum']==768 for r in rows)
mask=json.loads((p/'adapter/rank_mask.json').read_text());assert len(mask)==64 and sum(map(sum,mask.values()))==expected
assert json.loads((p/'reload_check.json').read_text())['passed']
b=json.loads((p/'batch_integrity.json').read_text());assert b['language_rows']==b['input_rows']==count and b['supervised_tokens']==count*2
print('ABLATION_ARTIFACTS_PASSED',arm,stage,flush=True)
