"""Artifact checks used by the login controller; stdlib only, no model loading."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def read(path):return json.loads(path.read_text())

def train(folder,method,seed,total=388,limit=None):
    c=read(folder/'train_config.json')
    assert c['total_steps']==total and c['seed']==seed and c['epochs']==1 and c['limit']==limit
    assert c['train_file']=='datasets/scienceqa/train_llava.json'
    assert c['image_folder']=='datasets/scienceqa/images'
    assert c['batch_size']*c['grad_accum']==16 and c['lr']==2e-4 and c['warmup_ratio']==.03
    assert (folder/'adapter/adapter_config.json').is_file()
    log=[json.loads(s) for s in (folder/'train_log.jsonl').read_text().splitlines()]
    assert log[-1]['step']==total
    if method=='adalora':
        assert c['trainable_params']==6292224 and c['final_rank']['retained_rank_sum']==512
        assert read(folder/'reload_check.json')['passed']
        mask=read(folder/'adapter/rank_mask.json');assert len(mask)==64 and sum(map(sum,mask.values()))==512
        assert [r['step'] for r in log]==list(range(1,total+1))
    else:assert c['trainable_params']==4194304

def evaluate(folder,limit=None):
    questions=[json.loads(s) for s in (ROOT/'datasets/scienceqa/test_questions.jsonl').read_text().splitlines()][:limit]
    rows=[json.loads(s) for s in (folder/'eval_scienceqa/predictions.jsonl').read_text().splitlines()]
    assert len(rows)==len({r['question_id'] for r in rows})==len(questions)
    assert {r['question_id']:r['answer'] for r in rows}=={q['question_id']:q['answer'] for q in questions}
    assert all(isinstance(r['prediction'],str) and r['prediction'].strip() for r in rows)
    norm=lambda s:s.strip().rstrip('.').lower()
    correct=sum(norm(r['prediction'])==norm(r['answer']) for r in rows)
    metric=read(folder/'eval_scienceqa/metrics.json');assert metric['correct']==correct and metric['total']==len(questions)
    assert metric['accuracy']==round(100*correct/len(questions),2)
    return metric

if __name__=='__main__':
    kind=sys.argv[1];folder=Path(sys.argv[2])
    if kind=='smoke':
        for method in ['uniform','adalora']:
            train(folder/method,method,0,total=20,limit=320);evaluate(folder/method,limit=20)
        result=dict(status='SCIENCEQA_SMOKE_PASSED',run=str(folder),purpose='Data, 20-update training, mask/export/reload and 20-question generation plumbing; not a full result')
        (ROOT/'output/diagnostics/scienceqa_smoke_passed.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    elif kind=='train':train(folder,sys.argv[3],int(sys.argv[4]));print('SCIENCEQA_FULL_TRAIN_VALIDATED',str(folder))
    elif kind=='eval':print(json.dumps(evaluate(folder)))
    else:raise ValueError(kind)
