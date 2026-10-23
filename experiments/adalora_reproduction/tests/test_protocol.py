"""Protocol checks run locally; actual PEFT round trips run inside the GPU job."""
import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from budget import validate_budget


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.modules = [(f'model.layers.{i}.self_attn.{p}', SimpleNamespace(in_features=4096, out_features=4096))
                        for i in range(32) for p in ['q_proj', 'v_proj']]
        self.model = SimpleNamespace(named_modules=lambda: self.modules)
        self.uniform = {n.removeprefix('model.'):8 for n, _ in self.modules}

    def test_uniform_and_adaptive_costs(self):
        self.assertEqual(validate_budget(self.uniform, self.model)['allocated_adapter_parameters'], 4194304)
        report = validate_budget({'adalora':dict(init_r=12,target_r=8,target_modules=['q_proj','v_proj'])},self.model)
        self.assertEqual(report['initial_adapter_parameters'],6292224)
        self.assertEqual(report['target_effective_parameters_nominal'],4194816)

    def test_wrong_targets_budget_and_dimensions_rejected(self):
        for spec in [dict(self.uniform, **{'layers.0.self_attn.q_proj':9}),
                     {'layers.0.mlp.up_proj':512},
                     {'adalora':dict(init_r=12,target_r=8,target_modules=['q_proj','k_proj'])}]:
            with self.assertRaises(ValueError): validate_budget(spec,self.model)
        self.modules[0][1].out_features=1024
        with self.assertRaises(ValueError): validate_budget(self.uniform,self.model)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('peft'),
                     'Torch/PEFT absent locally; these checks must run in the TC2 integration job')
class RuntimeTests(unittest.TestCase):
    def test_resume_and_full_shape_export(self):
        import torch
        from peft import AdaLoraConfig, PeftModel, get_peft_model
        from adalora_runtime import configure, raw_adapter_state, allocator_state, restore_allocator, export_full_shape, snapshot_loss
        torch.manual_seed(42)
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config={'model_type':'llama'}
                self.model=torch.nn.Module()
                self.model.layers=torch.nn.ModuleList()
                for _ in range(32):
                    layer=torch.nn.Module();layer.self_attn=torch.nn.Module()
                    layer.self_attn.q_proj=torch.nn.Linear(4,4,bias=False)
                    layer.self_attn.v_proj=torch.nn.Linear(4,4,bias=False)
                    self.model.layers.append(layer)
            def forward(self,x,labels=None):
                for layer in self.model.layers:
                    x=torch.tanh(layer.self_attn.q_proj(x)+layer.self_attn.v_proj(x))
                return SimpleNamespace(loss=(x-labels).square().mean() if labels is not None else None,logits=x)
        original=Toy(); weights=copy.deepcopy(original.state_dict())
        spec=dict(init_r=12,target_r=8,target_modules=['q_proj','v_proj'],lora_alpha=24,lora_dropout=.05,bias='none')
        def build():
            base=Toy();base.load_state_dict(weights)
            model=get_peft_model(base,configure(spec,20,AdaLoraConfig));model.train()
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4)
            return model,optimizer
        x=torch.randn(2,4);y=torch.randn(2,4)
        def update(model,opt,step):
            opt.zero_grad(set_to_none=True)
            captured, restore = snapshot_loss(model)
            output = model.base_model(x=x,labels=y)
            self.assertGreater(float(output.loss.detach()), captured['task'])
            output.loss.backward();opt.step()
            restore()
            model.base_model.update_and_allocate(step)
            opt.zero_grad(set_to_none=True)
        model,opt=build()
        for step in range(1,6): update(model,opt,step)
        checkpoint=dict(adapter=raw_adapter_state(model),allocator=allocator_state(model),
                        optimizer=copy.deepcopy(opt.state_dict()),rng=torch.get_rng_state().clone())
        for step in range(6,21): update(model,opt,step)
        expected=raw_adapter_state(model)
        resumed,optim=build()
        resumed.load_state_dict(checkpoint['adapter'],strict=False)
        restore_allocator(resumed,checkpoint['allocator']);optim.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['rng'])
        for step in range(6,21): update(resumed,optim,step)
        for name,value in raw_adapter_state(resumed).items():
            torch.testing.assert_close(value,expected[name],rtol=0,atol=0)
        resumed.eval()
        with torch.no_grad(): reference=resumed(x=x).logits
        with tempfile.TemporaryDirectory() as directory:
            export_full_shape(resumed,directory)
            base=Toy();base.load_state_dict(weights)
            loaded=PeftModel.from_pretrained(base,directory);loaded.eval()
            with torch.no_grad(): actual=loaded(x=x).logits
            torch.testing.assert_close(actual,reference,rtol=1e-6,atol=1e-6)


if __name__ == '__main__':
    unittest.main()
