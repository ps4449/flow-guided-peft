"""Exploratory CPU rank/flow association; no GPU or fitted mapping. Run at repo root."""
import csv,json,math,re,statistics
from collections import defaultdict
from pathlib import Path

def ranks(v):
    order=sorted(range(len(v)),key=v.__getitem__);r=[0.]*len(v);i=0
    while i<len(v):
        j=i+1
        while j<len(v) and v[order[j]]==v[order[i]]:j+=1
        for k in order[i:j]:r[k]=(i+j-1)/2
        i=j
    return r

def rho(x,y):
    x=ranks(x);y=ranks(y);a=statistics.mean(x);b=statistics.mean(y)
    den=math.sqrt(sum((v-a)**2 for v in x)*sum((v-b)**2 for v in y))
    return sum((u-a)*(v-b) for u,v in zip(x,y))/den if den else None
root=Path.cwd();result={'scope':'Exploratory descriptive association, not causality or a significance test. Window-9 points overlap; these are not isolated-layer effects. Positive rho means more rank near larger blocking-induced probability drops.','comparisons':[]}
for split in ['train_correct','train_wrong']:
    files=list((root/f'output/information_flow/llava_v1_5_7b/ScienceQA/{split}').rglob('*.csv'))
    for file in files:
        means=defaultdict(list);ids=set()
        for row in csv.DictReader(file.open()):
            means[int(row['layer'])].append(-float(row['relative diff first']));ids.add(row['question_id'])
        layers=sorted(means);assert layers==list(range(32))
        scores=[statistics.mean(means[l]) for l in layers]
        for seed in range(3):
            mask=json.loads((root/f'output/peft/adalora/scienceqa_train_llava/seed{seed}/adapter/rank_mask.json').read_text())
            values={}
            for name,m in mask.items():
                found=re.search(r'layers\.(\d+)\.self_attn\.(q_proj|v_proj)',name);assert found,name
                values[(int(found[1]),found[2])]=sum(m)
            for part in ['q_proj','v_proj','q_plus_v']:
                raw=[values[(l,part)] if part!='q_plus_v' else values[(l,'q_proj')]+values[(l,'v_proj')] for l in layers]
                # Also aggregate ranks over the same centered/clipped nine-layer window.
                windowed=[sum(raw[max(0,l-4):min(32,l+5)]) for l in layers]
                result['comparisons'].append({'file':str(file.relative_to(root)),'questions':len(ids),'seed':seed,'part':part,'spearman_center_rank':rho(scores,raw),'spearman_window_rank_sum':rho(scores,windowed)})
print(json.dumps(result,indent=2))
