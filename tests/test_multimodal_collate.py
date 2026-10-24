"""Exercise the actual shared collator without importing the model/dependencies."""
import ast
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

@unittest.skipUnless(importlib.util.find_spec('torch'),'Run with the existing TC2 torch environment')
class CollateTests(unittest.TestCase):
    def test_modality_and_label_rows_preserved(self):
        import torch
        from torch.nn.utils.rnn import pad_sequence
        source=ast.parse((ROOT/'peft_lora/train_lora.py').read_text())
        function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='collate')
        ns=dict(torch=torch,pad_sequence=pad_sequence,IGNORE_INDEX=-100)
        exec(compile(ast.Module(body=[function],type_ignores=[]),'<shared collate>','exec'),ns)
        for count in [1,4]:
            samples=[(torch.arange(i+3),torch.tensor([-100]*(i+1)+[10+i,2]),torch.zeros(3,4,4),(4,4)) for i in range(count)]
            result=ns['collate'](samples,0)
            self.assertEqual(result['modalities'],['image']*count)
            self.assertEqual(result['input_ids'].shape[0],count)
            self.assertEqual(result['labels'].shape[0],count)
            self.assertEqual(int((result['labels']!=-100).sum()),2*count)
            for i in range(count):self.assertEqual(result['labels'][i,:i+3].tolist(),samples[i][1].tolist())

if __name__=='__main__':unittest.main()
