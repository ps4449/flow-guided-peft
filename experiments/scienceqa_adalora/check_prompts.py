"""Check all ScienceQA prompt/answer boundaries and the 576-token image expansion."""
import json,sys
from pathlib import Path
from transformers import AutoTokenizer
from common import build_prompt,tokenize
ROOT=Path(__file__).resolve().parents[2]
tokenizer=AutoTokenizer.from_pretrained('liuhaotian/llava-v1.5-7b',use_fast=False)
rows=json.loads((ROOT/'datasets/scienceqa/train_llava.json').read_text())
lengths=[]
for r in rows:
    human=r['conversations'][0]['value'];answer=r['conversations'][1]['value']
    prefix=tokenize(build_prompt(human,None,'vicuna_v1'),tokenizer)
    full=tokenize(build_prompt(human,answer,'vicuna_v1'),tokenizer)
    assert full[:len(prefix)].equal(prefix),r['id']
    assert len(full)>len(prefix)
    lengths.append(len(full)+575)
assert max(lengths)<=2048,('Expanded sequence exceeds 2048',max(lengths))
print(json.dumps(dict(status='ALL_SCIENCEQA_TRAIN_PROMPTS_VALID',count=len(rows),maximum_expanded_tokens=max(lengths))),flush=True)
