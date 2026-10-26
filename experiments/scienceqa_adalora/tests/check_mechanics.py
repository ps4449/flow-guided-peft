"""CPU regression on a tiny 32-layer q/v model; no GPU/model downloads."""
import copy,json,sys,tempfile
from pathlib import Path
import torch
from torch import nn
from types import SimpleNamespace
import importlib.util
engine_path=Path(__file__).resolve().parents[1]/"train.py"
spec_loader=importlib.util.spec_from_file_location("scienceqa_clean_train",engine_path)
e=importlib.util.module_from_spec(spec_loader);spec_loader.loader.exec_module(e)
from peft import get_peft_model,PeftModel
class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.config={'model_type':'custom'}
        self.model=nn.Module();self.model.layers=nn.ModuleList()
        for _ in range(32):
            layer=nn.Module();layer.self_attn=nn.Module();layer.self_attn.q_proj=nn.Linear(8,8,bias=False);layer.self_attn.v_proj=nn.Linear(8,8,bias=False);self.model.layers.append(layer)
    def forward(self,x):
        total=0
        for layer in self.model.layers:total=total+layer.self_attn.q_proj(x)+layer.self_attn.v_proj(x)
        return SimpleNamespace(loss=(total/64-0.1).square().mean(),logits=total)
base=Path(__file__).parent;initial=None
for arm in ['A','B','C','D']:
    reg=.5 if arm in ['A','C'] else 0.;prune=arm in ['A','B']
    spec=json.loads((e.ROOT/'experiments/adalora_reproduction/adalora.json').read_text());spec['adalora'].update(orth_reg_weight=reg,target_r=8 if prune else 12)
    torch.manual_seed(42);raw=Tiny();raw.requires_grad_(False)
    torch.manual_seed(123);cfg,_=e.make_peft_config(spec,20);m=get_peft_model(raw,cfg)
    now={n:p.detach().clone() for n,p in m.named_parameters() if 'lora_' in n}
    if initial is None:initial=now
    else:assert all(torch.equal(now[n],v) for n,v in initial.items())
    if not prune:m.peft_config['default'].rank_pattern={n:[True]*p.shape[0] for n,p in m.base_model.model.named_parameters() if 'lora_E.default' in n}
    captured,restore=e.snapshot_loss(m);opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-4)
    for step in range(1,21):
        x=torch.ones(2,8)
        out=m.base_model(x) if reg else m.get_base_model()(x)
        added=float(out.loss.detach())-captured['task']
        assert added>0 if reg else abs(added)<1e-7
        assert torch.isfinite(out.loss)
        out.loss.backward();opt.step()
        if prune:m.base_model.update_and_allocate(step)
        opt.zero_grad(set_to_none=True)
    assert e.ranks(m)['retained_rank_sum']==(512 if prune else 768)
    restore();m.eval();expected=m.get_base_model()(x).logits.detach()
    with tempfile.TemporaryDirectory() as d:
        e.export_full_shape(m,d)
        torch.manual_seed(42);loaded=PeftModel.from_pretrained(Tiny(),d).eval()
        assert torch.allclose(loaded.get_base_model()(x).logits,expected,atol=1e-6,rtol=1e-5)
    print(arm,'CPU_MECHANICS_PASSED',flush=True)
