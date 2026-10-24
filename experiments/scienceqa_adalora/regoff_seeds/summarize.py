"""Validate predictions and summarize exact per-seed scores; no further submissions."""
import json,statistics,sys
from pathlib import Path
seed=int(sys.argv[1]);root=Path('output/peft/adalora/scienceqa_regoff_confirm')
paths={0:Path('output/diagnostics/scienceqa_closure_seed0_20261024/B/seed0'),1:root/'seed1',2:root/'seed2'}
result={'scope':'Regularization-off confirmation seeds 1–2 requested after the exploratory seed-0 ablation; not an untouched-test confirmatory study. No configuration changes or extra runs.','runs':{},'summary':{}}
for s,p in paths.items():
 if not (p/'eval_test_unmerged/metrics.json').exists():
  assert seed!=2,('Missing completed seed',s)
  continue
 entry={}
 for split,filename,count in [('eval_val','val',2097),('eval_test','test',2017),('eval_test_unmerged','test',2017)]:
  refs={q['question_id']:q['answer'] for q in map(json.loads,Path(f'datasets/scienceqa/{filename}_questions.jsonl').read_text().splitlines())}
  rows=list(map(json.loads,(p/split/'predictions.jsonl').read_text().splitlines()));m=json.loads((p/split/'metrics.json').read_text())
  assert len(rows)==len({r['question_id'] for r in rows})==count
  assert {r['question_id']:r['answer'] for r in rows}==refs
  norm=lambda s:s.strip().rstrip('.').lower()
  correct=sum(norm(r['prediction'])==norm(r['answer']) for r in rows)
  assert correct==m['correct'] and m['total']==count
  entry[split]={'correct':correct,'total':count,'accuracy':100*correct/count}
 result['runs'][s]={'path':str(p),'metrics':entry}
if len(result['runs'])==3:
 for split in ['eval_val','eval_test','eval_test_unmerged']:
  scores=[v['metrics'][split]['accuracy'] for v in result['runs'].values()]
  result['summary'][split]={'mean_accuracy':statistics.mean(scores),'sample_std_pp':statistics.stdev(scores),'seeds':[0,1,2]}
(root/f'seed{seed}'/'COMPLETED.json').write_text(json.dumps({'seed':seed,'status':'completed'})+'\n')
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['summary'],indent=2))
