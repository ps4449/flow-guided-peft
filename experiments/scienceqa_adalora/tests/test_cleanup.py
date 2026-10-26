"""CPU checks for current result locations and unchanged evaluation semantics."""
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / 'experiments/scienceqa_adalora'
sys.path.insert(0, str(CODE))
from summarize import summarize
from validate_run import validate_run

class PublishedResults(unittest.TestCase):
    def test_three_seed_scores_from_predictions(self):
        report = summarize(ROOT / 'output/peft/adalora/scienceqa_regoff_confirm')
        self.assertAlmostEqual(report['summary']['eval_test']['mean_accuracy'], 76.41712113700214)
        self.assertAlmostEqual(report['summary']['eval_test']['sample_std_pp'], .3301104669548545)

    def test_reference_and_ablations(self):
        for seed in (0, 1, 2):
            result = validate_run(ROOT / f'output/peft/adalora/scienceqa_train_llava/seed{seed}', historical=True)
            self.assertEqual(result['regularization'], .5)
        for name,reg in [('scienceqa_no_pruning',.5),('scienceqa_neither',0)]:
            result = validate_run(ROOT / f'output/ablations/adalora/{name}/seed0')
            self.assertEqual(result['final_rank_sum'],768)
            self.assertEqual(result['regularization'],reg)

    def test_evaluator_recomputes_without_model_import(self):
        source = ROOT / 'output/peft/adalora/scienceqa_regoff_confirm/seed0/eval_test'
        expected = json.loads((source / 'metrics.json').read_text())
        for unmerged in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp)
                (dest/'predictions.jsonl').write_bytes((source/'predictions.jsonl').read_bytes())
                cmd = [sys.executable,str(CODE/'evaluate.py'),'--questions','datasets/scienceqa/test_questions.jsonl','--out-dir',str(dest)]
                if unmerged:cmd.append('--unmerged')
                subprocess.run(cmd,cwd=ROOT,check=True,capture_output=True,text=True)
                metric=json.loads((dest/'metrics.json').read_text())
                for key in ('correct','total','accuracy','by_subject','by_category','by_grade'):
                    self.assertEqual(metric[key],expected[key])

    def test_training_operations_match_validated_engine(self):
        tree=ast.parse((CODE/'train.py').read_text())
        class NormalizeDefaults(ast.NodeTransformer):
            def visit_Call(self,node):
                if isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument' and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value in ('--batch-size','--grad-accum'):
                    for item in node.keywords:
                        if item.arg=='default':item.value=ast.Constant(4)
                return self.generic_visit(node)
        canonical='\n'.join(ast.dump(NormalizeDefaults().visit(n)) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)))
        self.assertEqual(hashlib.sha256(canonical.encode()).hexdigest(),'be743fab3e82c8fd5aff17a9cc983793a7927de64812416eeda386174924c0cb')

    def test_frozen_gqa_sources(self):
        subprocess.run([sys.executable,str(ROOT/'experiments/adalora_reproduction/verify_protocol.py')],check=True,capture_output=True)

    def test_cli_syntax(self):
        for file in CODE.rglob('*.py'):
            ast.parse(file.read_text(),filename=str(file))
        subprocess.run(['bash','-n',str(CODE/'tc2_run.sh')],check=True)

if __name__=='__main__':unittest.main()
