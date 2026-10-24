"""Read-only CPU audit of saved ScienceQA artifacts; run from repository root."""
import hashlib,json,math,statistics
from pathlib import Path
root=Path.cwd()
def read(p):return json.loads((root/p).read_text())
def digest(p):return hashlib.sha256((root/p).read_bytes()).hexdigest()
def norm(s):return s.strip().rstrip('.').lower()
train='datasets/scienceqa/train_llava.json'
questions=read(train)
expected='8cc00fb37a7e642e05b0063e2554a5e4a0be3a3e8bb13e49dbba20045ea1c4f9'
assert digest(train)==expected
refs={q['question_id']:q['answer'] for q in map(json.loads,(root/'datasets/scienceqa/test_questions.jsonl').read_text().splitlines())}
report={'train_sha256':digest(train),'train_rows':len(questions),'all_targets_uppercase_options':all(r['conversations'][1]['value'] in list('ABCDE') for r in questions),'runs':{}}
paths=[f'output/peft/adalora/scienceqa_train_llava/seed{s}' for s in range(3)]+['output/peft/uniform/scienceqa_modalities_fixed/seed0']
for path in paths:
 p=Path(path);c=read(p/'train_config.json');env=read(p/'environment.json');log=list(map(json.loads,(root/p/'train_log.jsonl').read_text().splitlines()))
 assert c['train_file']==train and c['image_folder']=='datasets/scienceqa/images'
 assert c['epochs']==1 and c['total_steps']==388 and c['batch_size']*c['grad_accum']==16
 assert c['lr']==2e-4 and c['warmup_ratio']==.03 and c['limit'] is None and log[-1]['step']==388
 entry={'seed':c['seed'],'microbatch':c['batch_size'],'accumulation':c['grad_accum'],'updates':388,'training_minutes':log[-1]['minutes'],'adapter_config':read(p/'adapter/adapter_config.json'),'evaluations':{}}
 if 'adalora/' in path:
  assert env['manifest']['files'][train]==expected
  mask=read(p/'adapter/rank_mask.json');assert len(mask)==64 and sum(map(sum,mask.values()))==512
  reload=read(p/'reload_check.json');assert reload['passed']
  assert [x['step'] for x in log]==list(range(1,389))
  assert all(math.isfinite(x['task_loss']) and math.isfinite(x['grad_norm']) for x in log)
  entry.update(retained_rank_sum=512,reload=reload,early40_task_mean=statistics.mean(x['task_loss'] for x in log[:40]),early40_reg_mean=statistics.mean(x['regularization'] for x in log[:40]))
 else:
  entry['batch_integrity']=read(p/'batch_integrity.json');assert entry['batch_integrity']['language_rows']==6208 and entry['batch_integrity']['all_batch_sizes_match']
 for name in ['eval_scienceqa','eval_scienceqa_unmerged']:
  if not (root/p/name/'metrics.json').exists():continue
  rows=list(map(json.loads,(root/p/name/'predictions.jsonl').read_text().splitlines()))
  assert len(rows)==len({r['question_id'] for r in rows})==len(refs)==2017
  assert {r['question_id']:r['answer'] for r in rows}==refs
  correct=sum(norm(r['prediction'])==norm(r['answer']) for r in rows)
  m=read(p/name/'metrics.json');assert m['correct']==correct and m['total']==2017
  entry['evaluations'][name]={'correct':correct,'total':2017,'accuracy':round(100*correct/2017,2),'valid_option_predictions':sum(norm(r['prediction']) in list('abcde') for r in rows)}
 report['runs'][path]=entry
report['scope']='Artifact/data/protocol audit; does not establish a causal explanation for the accuracy gap.'
print(json.dumps(report,indent=2))
