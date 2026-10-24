"""CPU regression: reproduce old full-size rejection, verify new entry reaches loading."""
import importlib.util,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/adalora_reproduction'))
class ReachedLoader(Exception):pass

def run(path,expect_rejection):
    spec=importlib.util.spec_from_file_location('partial_batch_checked_engine',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def loader(*a,**kw):raise ReachedLoader()
    module.load_model=loader
    with tempfile.TemporaryDirectory() as folder:
        sys.argv=['train','--ranks',str(ROOT/'experiments/adalora_reproduction/adalora.json'),
                  '--train-file',str(ROOT/'datasets/scienceqa/train_llava.json'),
                  '--batch-size','1','--grad-accum','16','--out-dir',folder]
        try:module.main()
        except ReachedLoader:
            assert not expect_rejection
        except ValueError as error:
            assert expect_rejection and 'complete effective batches' in str(error)
        else:raise AssertionError('Expected a guard or loader interception')

run(ROOT/'experiments/adalora_reproduction/train_lora.py',True)
run(ROOT/'experiments/scienceqa_adalora/adalora_train_engine.py',False)
rows=json.loads((ROOT/'datasets/scienceqa/train_llava.json').read_text())
import torch,math
for seed in [0,1,2]:
    order=torch.randperm(len(rows),generator=torch.Generator().manual_seed(seed*1000)).tolist()
    uniform_steps=math.ceil(len(rows)/4)//4
    ada_steps=len(rows)//16
    assert uniform_steps==ada_steps==388
    assert order[:uniform_steps*4*4]==order[:ada_steps*16]
    assert len(set(order[:ada_steps*16]))==6208
print('SCIENCEQA_PARTIAL_BATCH_REGRESSION_PASSED: old rejection reproduced; new full-size input accepted before model loading; equal 6208 selected samples for all3 seeds',flush=True)
