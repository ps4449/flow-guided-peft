"""Submission guards, quota handling and paired-seed scope; no GPU jobs."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('multi',ROOT/'peft_lora/methods/pathway_specific/multiseed.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class MultiseedTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        for field,value in [('ROOT',self.root),('MAIN',self.root/'main'),('OUT',self.root/'audit'),('MANIFEST',self.root/'audit/submission_manifest.json')]:
            p=patch.object(m,field,value);p.start();self.addCleanup(p.stop)
        self.data={'experimental_commit':'test','cases':{k:{**v,'jobs':{},'executions':{}} for k,v in m.CASES.items()}}
        m.save(m.MANIFEST,self.data)
    def invoke(self,stage='train',queued=None,history=None):
        result=type('Result',(),{'returncode':0,'stdout':'999\n','stderr':''})()
        with patch.object(m,'verify_frozen'),patch.object(m,'resources',return_value=['--gres=gpu:1']),patch.object(m.subprocess,'run',return_value=result) as call:
            accepted=m.submit(m.read(m.MANIFEST),'combined_s1',stage,queued or [],history or [])
        return accepted,call
    def test_exact_required_scope(self):
        self.assertEqual(len(m.CASES),6)
        self.assertEqual({v['seed'] for v in m.CASES.values()},{1,2})
        self.assertNotIn('early',m.ALLOCATIONS)
    def test_accept_and_do_not_duplicate(self):
        accepted,call=self.invoke();self.assertTrue(accepted);self.assertEqual(call.call_count,1)
        accepted,call=self.invoke();self.assertTrue(accepted);self.assertEqual(call.call_count,0)
    def test_evaluation_depends_on_successful_train(self):
        self.invoke();accepted,call=self.invoke('eval')
        args=call.call_args.args[0];self.assertEqual(args[args.index('--dependency')+1],'afterok:999')
    def test_quota_no_submission(self):
        accepted,call=self.invoke(queued=[{'job_id':'1','name':'other','comment':''},{'job_id':'2','name':'other','comment':''}]);self.assertFalse(accepted);call.assert_not_called()
    def test_crash_after_acceptance_adopts_job(self):
        c=m.job_comment(self.data,'combined_s1','train')
        accepted,call=self.invoke(queued=[{'job_id':'555','name':'sqa_top8_combined_s1_train','state':'RUNNING','comment':c}])
        self.assertTrue(accepted);call.assert_not_called();self.assertEqual(m.read(m.MANIFEST)['cases']['combined_s1']['jobs']['train']['job_id'],'555')
    def test_failed_previous_training_not_adopted_on_continuation(self):
        self.data['cases']['combined_s1']['previous_training_jobs']=[{'job_id':'555'}];m.save(m.MANIFEST,self.data)
        c=m.job_comment(self.data,'combined_s1','train')
        accepted,call=self.invoke(history=[{'job_id':'555','name':'sqa_top8_combined_s1_train','state':'FAILED','comment':c}])
        self.assertTrue(accepted);self.assertEqual(call.call_count,1)
    def test_external_output_blocks_duplicate(self):
        d=m.MAIN/'output/peft/flow_pruned8_scienceqa/scienceqa_train_llava/seed1';d.mkdir(parents=True);(d/'checkpoint.pt').write_text('partial')
        with self.assertRaisesRegex(ValueError,'duplicate'):self.invoke()
    def test_login_node_worker_refused(self):
        with patch.dict(os.environ,{},clear=True),self.assertRaisesRegex(RuntimeError,'Slurm allocation'):m.worker('train','combined_s1')
    def test_three_seed_mean_and_sample_sd(self):
        counts={'combined':(1605,1607,1609),'question_last':(1572,1573,1574),'middle':(1522,1523,1524)}
        for name in m.ALLOCATIONS:
            for seed in (0,1,2):
                d=m.run_dir(name,seed);(d/'eval_scienceqa').mkdir(parents=True)
                (d/'eval_scienceqa/metrics.json').write_text('{}')
                (d/'train_config.json').write_text(json.dumps({'seed':seed,'trainable_params':1048576}))
        report=self.root/'report.md'
        def correct(d):
            name=next(n for n,(stem,_) in m.ALLOCATIONS.items() if stem==d.parents[1].name)
            return counts[name][int(d.name[4:])]
        with patch.object(m,'REPORT',report),patch.object(m,'verify_predictions',side_effect=correct):value=m.summarize()
        combined=value['three_seed_summary'][0]
        self.assertAlmostEqual(combined['mean_accuracy_percent'],1607*100/2017)
        self.assertAlmostEqual(combined['sample_sd_pp'],2*100/2017)
        self.assertTrue(value['strict_seed0_order_preserved'])
    def test_incomplete_seeds_do_not_get_three_seed_statistics(self):
        for name in m.ALLOCATIONS:
            d=m.run_dir(name,0);(d/'eval_scienceqa').mkdir(parents=True)
            (d/'eval_scienceqa/metrics.json').write_text('{}')
            (d/'train_config.json').write_text(json.dumps({'seed':0,'trainable_params':1048576}))
        with patch.object(m,'REPORT',self.root/'report.md'),patch.object(m,'verify_predictions',return_value=1605):value=m.summarize()
        self.assertIsNone(value['three_seed_summary'][0]['mean_accuracy_percent'])
        self.assertIsNone(value['strict_seed0_order_preserved'])
    def test_manifest_updates_preserve_other_stage(self):
        m.change_manifest(lambda d:d['cases']['combined_s1']['jobs'].update(train={'job_id':'1'}))
        m.change_manifest(lambda d:d['cases']['combined_s1']['executions'].update(eval={'state':'running'}))
        self.assertEqual(m.read(m.MANIFEST)['cases']['combined_s1']['jobs']['train']['job_id'],'1')

if __name__=='__main__':unittest.main()
