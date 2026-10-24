"""Summarize all predeclared conditions, then stop. No best-arm selection."""
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);out={}
def read(p):return json.loads(p.read_text())
base=Path('output/peft/adalora/scienceqa_train_llava/seed0')
out['A']={'reg':.5,'pruning':True,'final_rank_sum':512,'validation':read(r/'A_eval_val/metrics.json'),'test_merged':read(base/'eval_scienceqa/metrics.json'),'test_unmerged':read(base/'eval_scienceqa_unmerged/metrics.json')}
for arm in ['B','C','D']:
 p=r/arm/'seed0';out[arm]={'reg':0. if arm in ['B','D'] else .5,'pruning':arm=='B','final_rank_sum':read(p/'train_config.json')['final_rank']['retained_rank_sum'],'validation':read(p/'eval_val/metrics.json'),'test_merged':read(p/'eval_test/metrics.json'),'test_unmerged':read(p/'eval_test_unmerged/metrics.json')}
result={'arms':out,'scope':'Exploratory seed-0 mechanism audit. C/D retain rank12 (larger budget). No extra runs or model selection.','comparisons':{}}
for split in ['validation','test_merged','test_unmerged']:
 scores={a:100*v[split]['correct']/v[split]['total'] for a,v in out.items()}
 result['comparisons'][split]={'scores':scores,'remove_reg_with_pruning':scores['B']-scores['A'],'remove_reg_without_pruning':scores['D']-scores['C'],'remove_pruning_with_reg':scores['C']-scores['A'],'remove_pruning_without_reg':scores['D']-scores['B']}
(r/'final_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['comparisons'],indent=2))
