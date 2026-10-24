"""No optimizer steps: compare 4x4 and 1x16 accumulation on the same initial AdaLoRA."""
import json,sys,time,random
from pathlib import Path
import numpy as np
import torch
from peft import get_peft_model
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/adalora_reproduction'))
import train_lora as engine
from adalora_runtime import snapshot_loss

def main():
    torch.manual_seed(0);np.random.seed(0);random.seed(0)
    tokenizer,base,processor=engine.load_model('liuhaotian/llava-v1.5-7b')
    base.config.use_cache=False;tokenizer.model_max_length=2048;base.config.tokenizer_model_max_length=2048
    base.requires_grad_(False);base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    spec=json.loads((ROOT/'experiments/adalora_reproduction/adalora.json').read_text())
    spec['adalora']['lora_dropout']=0.0  # deterministic gradient-equivalence probe only
    cfg,_=engine.make_peft_config(spec,388)
    model=get_peft_model(base,cfg)
    params=[p for p in model.parameters() if p.requires_grad]
    for p in params:p.data=p.data.float()
    model.train();model.get_base_model().get_vision_tower().eval()
    rows=json.loads((ROOT/'datasets/scienceqa/train_llava.json').read_text())
    order=torch.randperm(len(rows),generator=torch.Generator().manual_seed(0)).tolist()[:16]
    dataset=engine.GQATrain(rows,str(ROOT/'datasets/scienceqa/images'),tokenizer,processor,model.config,'vicuna_v1')
    samples=[dataset[i] for i in order]
    supervised=[int((s[1]!=engine.IGNORE_INDEX).sum()) for s in samples]
    captured,restore=snapshot_loss(model)
    report={};vectors={}
    for bs in [4,1]:
        model.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        start=time.perf_counter();total=task=0.
        accum=16//bs
        for i in range(0,16,bs):
            batch=engine.collate(samples[i:i+bs],tokenizer.pad_token_id)
            batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
            output=model.base_model(**batch)
            total+=float(output.loss.detach())/accum;task+=captured['task']/accum
            (output.loss/accum).backward()
            del output,batch
        torch.cuda.synchronize()
        name=f'batch{bs}_accum{accum}'
        vectors[name]=torch.cat([p.grad.detach().float().cpu().flatten() for p in params])
        norms={}
        for group in ['lora_A','lora_B','lora_E']:
            norms[group]=sum(float(p.grad.detach().float().square().sum()) for n,p in model.named_parameters() if p.requires_grad and group in n)**.5
        report[name]=dict(task_loss=task,total_loss=total,regularization=total-task,seconds=time.perf_counter()-start,
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,gradient_norms=norms)
    a=vectors['batch4_accum4'];b=vectors['batch1_accum16'];d=a-b
    report['comparison']=dict(gradient_cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0)),
                              relative_l2_difference=float(d.norm()/a.norm()),maximum_absolute_difference=float(d.abs().max()))
    report['scope']='Same initial weights and16 real training samples; dropout disabled to test accumulation equivalence. No optimizer update, rank allocation, export or modification of existing run weights. Not a convergence or final-accuracy test.'
    report['supervised_token_counts']=supervised
    out=ROOT/'output/diagnostics/scienceqa_batch_gradient_probe.json';out.write_text(json.dumps(report,indent=2)+'\n')
    restore();print(json.dumps(report),flush=True)
if __name__=='__main__':main()
