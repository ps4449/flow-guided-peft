#!/usr/bin/env python3
"""Quota-aware ScienceQA top-8 seeds 1/2; GPU work is delegated to TC2 Slurm scripts."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import time

from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[3]
MAIN = Path('/home/msai/swarangi001/flow-guided-peft')
OUT = ROOT / 'output/pathway_top8_multiseed'
MANIFEST = OUT / 'submission_manifest.json'
RUNTIME = OUT / 'runtime_seed0'
REPORT = ROOT / 'docs/SCIENCEQA_TOP8_MULTISEED.md'
PYTHON = '/home/msai/swarangi001/.conda/envs/llava/bin/python'
REVISION = '8734991cd5c80eff45b09eb954d6d849ef495d58'
ALLOCATIONS = {
    'combined': ('flow_pruned8_scienceqa', [0,1,2,3,4,9,10,11]),
    'question_last': ('question_last_pruned8_scienceqa', list(range(9,17))),
    'middle': ('middle_pruned8_scienceqa', list(range(12,20))),
}
CASES = {f'{name}_s{seed}': {'allocation':name, 'seed':seed} for name in ALLOCATIONS for seed in (1,2)}
SOURCE_FILES = ('peft_lora/train_lora.py','peft_lora/eval_gqa.py','peft_lora/common.py','scripts/tc2_train.sh','scripts/tc2_eval.sh')
ACTIVE = {'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','REQUEUED','RESIZING'}
MATCH_KEYS = ('model_path','conv_mode','train_file','image_folder','epochs','batch_size','grad_accum','lr','warmup_ratio','save_every','max_hours','num_workers','total_steps','trainable_params')
sys.dont_write_bytecode = True


def now(): return datetime.now(ZoneInfo('Asia/Singapore')).isoformat()
def read(path): return json.loads(Path(path).read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.replace(path)


@contextmanager
def lock(name):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/f'.{name}.lock'
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|(fcntl.LOCK_NB if name=='controller' else 0));yield


def change_manifest(update):
    with lock('manifest'):
        data=read(MANIFEST);update(data);data['updated_at']=now();save(MANIFEST,data);return data


def command(args, **kwargs):
    result=subprocess.run(args,cwd=ROOT,text=True,capture_output=True,**kwargs)
    if result.returncode: raise RuntimeError(f'{shlex.join(args)}\n{result.stderr or result.stdout}')
    return result.stdout.strip()


def run_dir(allocation,seed): return ROOT/'output/peft'/ALLOCATIONS[allocation][0]/'scienceqa_train_llava'/f'seed{seed}'
def rank_path(allocation): return ROOT/'configs/ranks'/f'{ALLOCATIONS[allocation][0]}.json'
def normalize(text): return text.strip().rstrip('.').lower()


def verify_predictions(directory):
    questions=[json.loads(line) for line in (ROOT/'datasets/scienceqa/test_questions.jsonl').open()]
    gold={q['question_id']:q for q in questions}
    raw=[json.loads(line) for line in (directory/'eval_scienceqa/predictions.jsonl').open()]
    predictions={row['question_id']:row for row in raw}
    if len(questions)!=2017 or len(gold)!=2017 or len(raw)!=2017 or len(predictions)!=2017 or set(predictions)!=set(gold):
        raise ValueError(f'Prediction cohort mismatch: {directory}')
    if any(predictions[q]['answer']!=gold[q]['answer'] for q in gold):raise ValueError('Gold-label mismatch')
    correct=sum(normalize(predictions[q]['prediction'])==normalize(gold[q]['answer']) for q in gold)
    metrics=read(directory/'eval_scienceqa/metrics.json')
    if metrics['correct']!=correct or metrics['total']!=2017 or metrics['accuracy']!=round(100*correct/2017,2):raise ValueError('Saved accuracy does not reproduce')
    return correct


def verify_training(directory, allocation, seed):
    cfg=read(directory/'train_config.json');base=read(run_dir('combined',0)/'train_config.json')
    if cfg['seed']!=seed or any(cfg.get(k)!=base.get(k) for k in MATCH_KEYS) or cfg.get('flow_gates') or cfg.get('variant','lora')!='lora':raise ValueError('Training protocol mismatch')
    if Path(cfg['ranks']).resolve()!=rank_path(allocation).resolve():raise ValueError('Wrong rank configuration')
    path=directory/'adapter/adapter_model.safetensors'
    with path.open('rb') as f: header=json.loads(f.read(struct.unpack('<Q',f.read(8))[0]))
    tensors={k:v for k,v in header.items() if k!='__metadata__'}
    expected={f'.layers.{i}.self_attn.{p}.' for i in ALLOCATIONS[allocation][1] for p in ('q_proj','v_proj')}
    if len(tensors)!=32 or sum(math.prod(v['shape']) for v in tensors.values())!=1048576:raise ValueError('Saved adapter parameter count differs')
    if any(not any(part in key for part in expected) or '.lora_' not in key for key in tensors):raise ValueError('Unexpected adapter target')
    cfg_adapter=read(directory/'adapter/adapter_config.json')
    if (cfg_adapter['r'],cfg_adapter['lora_alpha'],cfg_adapter['lora_dropout'])!=(8,16,.05):raise ValueError('Adapter settings differ')
    return cfg


def prepare():
    if command(['git','branch','--show-current'])!='experiments/flow-placement-ablations':raise ValueError('Expected experimental branch')
    if str(ROOT).startswith('/tmp/'):raise ValueError('Stage the experimental worktree on shared /home storage first')
    if MANIFEST.exists():verify_frozen(read(MANIFEST));return read(MANIFEST)
    baseline=read(ROOT/'output/pathway_top8_seed0/early8_submission_manifest.json')
    for name in ALLOCATIONS:
        expected={f'layers.{i}.self_attn.{p}':8 if i in ALLOCATIONS[name][1] else 0 for i in range(32) for p in ('q_proj','v_proj')}
        if read(rank_path(name))!=expected:raise ValueError(f'Rank map differs: {name}')
        for seed in (1,2):
            directory=run_dir(name,seed);other=MAIN/directory.relative_to(ROOT)
            if directory.exists() and any(directory.iterdir()):raise ValueError(f'Unregistered output already exists: {directory}')
            if other.exists() and any(other.iterdir()):raise ValueError(f'Potential duplicate in shared dev: {other}')
    image_link=ROOT/'datasets/scienceqa/images'
    if not image_link.exists():image_link.symlink_to(MAIN/'datasets/scienceqa/images',target_is_directory=True)
    frozen=[]
    for relative in SOURCE_FILES:
        path=RUNTIME/relative;payload=subprocess.check_output(['git','show',f'{REVISION}:{relative}'],cwd=ROOT)
        if path.exists() and path.read_bytes()!=payload:raise ValueError(f'Runtime source changed: {path}')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload);frozen.append({'path':str(path),'sha256':digest(path)})
    dataset_link=RUNTIME/'datasets'
    if not dataset_link.exists():dataset_link.symlink_to(ROOT/'datasets',target_is_directory=True)
    for relative in ['datasets/scienceqa/train_llava.json','datasets/scienceqa/test_questions.jsonl',
                     'peft_lora/methods/pathway_specific/multiseed.py','scripts/launch_scienceqa_top8_multiseed.sh']:
        path=ROOT/relative;frozen.append({'path':str(path),'sha256':digest(path)})
    frozen.extend({'path':str(rank_path(name)),'sha256':digest(rank_path(name))} for name in ALLOCATIONS)
    for name in ALLOCATIONS:
        for relative in ('train_config.json','train_log.jsonl','eval_scienceqa/metrics.json','eval_scienceqa/predictions.jsonl'):
            path=run_dir(name,0)/relative;frozen.append({'path':str(path),'sha256':digest(path)})
    frozen.extend(item for item in baseline['frozen_files'] if '/LLaVA-NeXT/' in item['path'] or '/hf_cache/' in item['path'])
    data={'created_at':now(),'state':'prepared','experimental_branch':'experiments/flow-placement-ablations',
          'experimental_commit':command(['git','rev-parse','HEAD']),'training_source_commit':REVISION,
          'worktree':str(ROOT),'runtime_source':str(RUNTIME),'packages':baseline['packages'],'frozen_files':frozen,
          'cases':{key:{**case,'state':'planned','jobs':{},'executions':{}} for key,case in CASES.items()},
          'optional_early_included':False,'single_seed_caveat':'Three seeds estimate training variation descriptively; fixed-question uncertainty is different.'}
    verify_frozen(data);save(MANIFEST,data);return data


def verify_frozen(data):
    for item in data['frozen_files']:
        if digest(item['path'])!=item['sha256']:raise ValueError(f'Frozen source/input changed: {item["path"]}')
    for name,version in data['packages'].items():
        if importlib.metadata.version(name)!=version:raise ValueError(f'Package changed: {name}')


def resources(stage):
    values=[]
    for line in (RUNTIME/f'scripts/tc2_{stage}.sh').read_text().splitlines():
        if line.startswith('#SBATCH'):
            args=shlex.split(line[len('#SBATCH'):].strip())
            if args[0].split('=')[0] not in ('--job-name','--output','--error'):values.extend(args)
    return values


def slurm():
    env={**os.environ,'TZ':'Asia/Singapore'};queued=[];history=[]
    for line in command(['squeue','--noheader','--user','swarangi001','--format=%i|%100j|%T|%250k'],env=env).splitlines():
        queued.append(dict(zip(('job_id','name','state','comment'),[x.strip() for x in line.split('|')])))
    args=['sacct','--noheader','--parsable2','--allocations','--starttime','2026-10-29','--user','swarangi001',
          '--format=JobIDRaw,JobName%100,State,ExitCode,Comment%250']
    for line in command(args,env=env).splitlines():
        fields=line.rstrip('|').split('|');row=dict(zip(('job_id','name','state','exit_code','comment'),fields))
        row['state']=row['state'].split()[0].rstrip('+');history.append(row)
    return queued,history


def job_comment(data,key,stage):
    stamp=hashlib.sha256((data['experimental_commit']+str(ROOT)).encode()).hexdigest()[:12]
    return f'scienceqa_top8_seeds12:{key}:{stage}:{stamp}'


def submit(data,key,stage,queued,history):
    case=data['cases'][key]
    if case['jobs'].get(stage,{}).get('job_id'):return True
    name=f'sqa_top8_{key}_{stage}';comment=job_comment(data,key,stage)
    previous={j['job_id'] for j in case.get('previous_training_jobs',[])} if stage=='train' else set()
    matches={j['job_id']:j for j in history+queued if j['name']==name and j.get('comment')==comment and j['job_id'] not in previous}
    if len(matches)>1:raise ValueError(f'Multiple matching jobs: {matches}')
    if matches:
        job=next(iter(matches.values()));change_manifest(lambda m:m['cases'][key]['jobs'].update({stage:{**job,'adopted_existing':True}}));return True
    if any(j['name']==name and j['job_id'] not in previous for j in history+queued):raise ValueError(f'Foreign job-name collision: {name}')
    if len(queued)>=2:return False
    verify_frozen(data)
    if stage=='train':
        other=MAIN/run_dir(case['allocation'],case['seed']).relative_to(ROOT)
        if other.exists() and any(other.iterdir()):raise ValueError(f'External duplicate appeared: {other}')
    args=['sbatch','--parsable',*resources(stage),'--job-name',name,'--chdir',str(ROOT),
          '--output',str(ROOT/'slurm/logs'/f'{name}_%j.out'),'--error',str(ROOT/'slurm/logs'/f'{name}_%j.err'),
          '--comment',comment,'--open-mode=append']
    if stage=='eval':args+=['--dependency','afterok:'+case['jobs']['train']['job_id']]
    args+=['--export=ALL','--wrap','exec '+shlex.join([PYTHON,'-u',str(Path(__file__).resolve()),'--worker',stage,'--case',key])]
    change_manifest(lambda m:m['cases'][key]['jobs'].update({stage:{'state':'submission_intent','comment':comment,'command':args,'intent_at':now()}}))
    result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,env={**os.environ,'TZ':'Asia/Singapore'})
    if result.returncode:
        if any(text in result.stderr for text in ('QOSMaxSubmit','job submit limit','Job violates accounting')):
            print('Waiting for submission quota:',result.stderr.strip(),flush=True);return False
        raise RuntimeError(result.stderr)
    job_id=result.stdout.strip().split(';')[0]
    if not job_id.isdigit():raise ValueError(f'Unexpected sbatch response: {result.stdout}')
    change_manifest(lambda m:m['cases'][key]['jobs'][stage].update(job_id=job_id,state='PENDING',submitted_at=now()))
    print(f'Submitted {key} {stage}: {job_id}',flush=True);return True


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize():
    rows=[];counts={};summaries=[];contrasts=[]
    for allocation,(stem,layers) in ALLOCATIONS.items():
        complete=[]
        for seed in (0,1,2):
            directory=run_dir(allocation,seed);metric=directory/'eval_scienceqa/metrics.json'
            if not metric.exists():continue
            count=verify_predictions(directory);cfg=read(directory/'train_config.json')
            if cfg['seed']!=seed or cfg['trainable_params']!=1048576:raise ValueError('Report protocol mismatch')
            counts[(allocation,seed)]=count;complete.append(count*100/2017)
            rows.append({'allocation':allocation,'seed':seed,'layers':','.join(map(str,layers)),'correct':count,'total':2017,
                         'accuracy_percent':count*100/2017,'parameters':1048576,'metrics':str(metric),'predictions':str(directory/'eval_scienceqa/predictions.jsonl')})
        summaries.append({'allocation':allocation,'completed_seeds':len(complete),'mean_accuracy_percent':statistics.mean(complete) if len(complete)==3 else None,
                          'sample_sd_pp':statistics.stdev(complete) if len(complete)==3 else None})
    for seed in (0,1,2):
        for a,b in (('combined','question_last'),('combined','middle'),('question_last','middle')):
            if (a,seed) in counts and (b,seed) in counts:contrasts.append({'seed':seed,'a':a,'b':b,'difference_pp':100*(counts[a,seed]-counts[b,seed])/2017})
    complete=len(rows)==9;ordering={}
    for seed in (0,1,2):
        if all((name,seed) in counts for name in ALLOCATIONS):ordering[str(seed)]=counts['combined',seed]>counts['question_last',seed]>counts['middle',seed]
    value={'updated_at':now(),'state':'completed' if complete else 'in_progress','runs':rows,'three_seed_summary':summaries,
           'matched_per_seed_differences':contrasts,'strict_seed0_order_preserved':all(ordering.values()) if len(ordering)==3 else None,
           'uncertainty':'Sample SD is across training seeds; question-bootstrap intervals conditional on fixed checkpoints are different. Three seeds provide limited precision.',
           'fixed_early_seeds_1_2':'Not submitted; optional team approval pending'}
    save(OUT/'comparison.json',value);write_csv(OUT/'seed_results.csv',rows);write_csv(OUT/'three_seed_summary.csv',summaries);write_csv(OUT/'matched_seed_differences.csv',contrasts)
    lines=['# ScienceQA top-8: seeds 0–2 comparison','',f'Experiment state: **{value["state"]}**. Required runs only; fixed-early seeds 1/2 remain excluded.','',
           '| Allocation | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |','| --- | --- | --- | --- | --- |']
    for s in summaries:
        name=s['allocation'];cells=[f'{counts[name,seed]}/2017 ({100*counts[name,seed]/2017:.2f}%)' if (name,seed) in counts else 'pending' for seed in (0,1,2)]
        summary=f'{s["mean_accuracy_percent"]:.2f} ± {s["sample_sd_pp"]:.2f}%' if s['completed_seeds']==3 else 'pending'
        lines.append('| '+' | '.join([name,*cells,summary])+' |')
    lines+=['','| Seed | Combined − Question→Last pp | Combined − middle pp | Question→Last − middle pp |','| --- | --- | --- | --- |']
    for seed in (0,1,2):
        values=[f'{row["difference_pp"]:+.3f}' for row in contrasts if row['seed']==seed];lines.append('| '+' | '.join([str(seed),*(values if len(values)==3 else ['pending']*3)])+' |')
    conclusion=('The strict seed-0 ordering (combined > Question→Last > middle) is preserved in all three seeds.' if value['strict_seed0_order_preserved'] is True else
                'The strict seed-0 ordering is not preserved in every seed; inspect the per-seed counts and ties.' if complete else 'Cross-seed ordering cannot be concluded until all required runs finish.')
    lines+=['',conclusion,'','All runs use the same original checkpoint, rank-8 q_proj/v_proj, 1,048,576 trainable parameters and the exact seed-0 protocol; only seeds/output paths differ.',
            'All predictions are aligned to the same 2,017 unique IDs and exact gold labels with original normalization. Sample SD uses ddof=1 across seeds 0–2.',
            'Three seeds describe training variation with limited precision. Fixed-checkpoint question-bootstrap significance is not training-seed reliability; no statistical superiority is claimed.',
            'Fixed early seed 0 tied combined flow at 1605/2017; its optional additional seeds were not submitted.','',
            'Outputs follow `output/peft/<rank-stem>/scienceqa_train_llava/seed<k>/`; logs remain in `slurm/logs/`. Shared dev and historical artifacts are preserved.',
            '[Actual jobs/provenance](../output/pathway_top8_multiseed/submission_manifest.json) · [Seed results](../output/pathway_top8_multiseed/seed_results.csv) · [Three-seed summary](../output/pathway_top8_multiseed/three_seed_summary.csv) · [Matched differences](../output/pathway_top8_multiseed/matched_seed_differences.csv) · [Original audit](../output/pathway_top8_multiseed/preflight.json)','']
    REPORT.write_text('\n'.join(lines));return value


def controller():
    with lock('controller'):
        prepare();save(OUT/'controller.json',{'pid':os.getpid(),'node':socket.gethostname(),'started_at':now(),'mode':'nohup CPU controller; GPU work only in Slurm','log':str(ROOT/'slurm/logs/scienceqa_top8_multiseed_controller.out')})
        while True:
            data=read(MANIFEST);verify_frozen(data);queued,history=slurm();by_id={r['job_id']:r for r in history+queued}
            all_done=True
            for key in CASES:
                data=read(MANIFEST);case=data['cases'][key]
                if case['state']=='completed':continue
                all_done=False
                if not submit(data,key,'train',queued,history):break
                data=read(MANIFEST);queued,history=slurm();by_id={r['job_id']:r for r in history+queued};case=data['cases'][key]
                if case.get('rebind_eval_dependency') and case['jobs'].get('eval',{}).get('job_id'):
                    command(['scontrol','update',f'JobId={case["jobs"]["eval"]["job_id"]}',f'Dependency=afterok:{case["jobs"]["train"]["job_id"]}'])
                    change_manifest(lambda m:m['cases'][key].update(rebind_eval_dependency=False))
                train=by_id.get(case['jobs']['train']['job_id'],case['jobs']['train'])
                if train['state'] not in ACTIVE and train['state']!='COMPLETED':
                    execution=case['executions'].get('train',{}).get(train['job_id'],{})
                    if execution.get('exit_code')==3 and (run_dir(case['allocation'],case['seed'])/'checkpoint.pt').exists():
                        def resume(m):
                            c=m['cases'][key];c.setdefault('previous_training_jobs',[]).append(c['jobs'].pop('train'));c['rebind_eval_dependency']=True
                        change_manifest(resume)
                        break
                    raise RuntimeError(f'Training failed for {key}: {train}')
                if not submit(data,key,'eval',queued,history):break
                data=read(MANIFEST);case=data['cases'][key];queued,history=slurm();by_id={r['job_id']:r for r in history+queued}
                evaluation=by_id.get(case['jobs']['eval']['job_id'],case['jobs']['eval'])
                for stage in ('train','eval'):
                    job=case['jobs'][stage];state=by_id.get(job['job_id'],job)['state'];change_manifest(lambda m,s=stage,t=state:m['cases'][key]['jobs'][s].update(state=t))
                if evaluation['state']=='COMPLETED':
                    directory=run_dir(case['allocation'],case['seed']);verify_training(directory,case['allocation'],case['seed']);correct=verify_predictions(directory)
                    change_manifest(lambda m:m['cases'][key].update(state='completed',correct=correct,total=2017,completed_at=now()))
                elif evaluation['state'] not in ACTIVE:raise RuntimeError(f'Evaluation failed for {key}: {evaluation}')
                break
            summarize()
            if all_done:
                change_manifest(lambda m:m.update(state='completed',completed_at=now()));print('All six training/evaluation pairs completed.',flush=True);return
            change_manifest(lambda m:m.update(state='running'));time.sleep(30)


def worker(stage,key):
    if not os.environ.get('SLURM_JOB_ID') or not any(os.environ.get(k) for k in ('SLURM_JOB_GPUS','SLURM_GPUS_ON_NODE','CUDA_VISIBLE_DEVICES')):raise RuntimeError('GPU worker requires a Slurm allocation')
    data=read(MANIFEST);verify_frozen(data);case=data['cases'][key];directory=run_dir(case['allocation'],case['seed']);base=read(run_dir('combined',0)/'train_config.json')
    if stage=='eval':verify_training(directory,case['allocation'],case['seed'])
    import torch
    execution={'job_id':os.environ['SLURM_JOB_ID'],'state':'running','started_at':now(),'node':socket.gethostname(),'gpu':torch.cuda.get_device_name(),'packages':data['packages'],'training_source_commit':REVISION}
    def record():change_manifest(lambda m:m['cases'][key]['executions'].setdefault(stage,{}).update({execution['job_id']:execution.copy()}))
    record()
    if stage=='train':
        keys=('model_path','conv_mode','train_file','image_folder','epochs','batch_size','grad_accum','lr','warmup_ratio','save_every','max_hours','num_workers')
        extra=[v for k in keys for v in ('--'+k.replace('_','-'),str(base[k]))]+['--out-dir',str(directory)]
    else:extra=['--questions','datasets/scienceqa/test_questions.jsonl','--model-path',base['model_path'],'--conv-mode',base['conv_mode'],'--image-folder',base['image_folder'],'--max-new-tokens','16']
    if any(any(c.isspace() for c in value) for value in extra):raise ValueError('TC2 workflow needs whitespace-free argument values')
    env={**os.environ,'RANKS':str(rank_path(case['allocation'])),'SEED':str(case['seed']),'ADAPTER':str(directory/'adapter'),'EXTRA':' '.join(extra),
         'HF_HOME':str(Path.home()/'hf_cache'),'HF_HUB_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'}
    started=time.monotonic();result=subprocess.run(['bash',str(RUNTIME/f'scripts/tc2_{stage}.sh')],cwd=RUNTIME,env=env)
    execution.update(state='completed' if result.returncode==0 else 'failed',exit_code=result.returncode,ended_at=now(),wall_seconds=time.monotonic()-started)
    if result.returncode==0:
        try:
            if stage=='train':
                execution['actual_training_configuration']=verify_training(directory,case['allocation'],case['seed'])
                execution['saved_adapter_parameters_verified']=1048576
                execution['adapter_sha256']=digest(directory/'adapter/adapter_model.safetensors')
            else:execution['correct']=verify_predictions(directory);execution['total']=2017
        except Exception as error:
            execution.update(state='failed',exit_code=1,validation_error=str(error));record();raise
    record()
    if execution['exit_code']:raise SystemExit(execution['exit_code'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-only',action='store_true');p.add_argument('--controller',action='store_true');p.add_argument('--worker',choices=('train','eval'));p.add_argument('--case',choices=tuple(CASES));p.add_argument('--summarize',action='store_true');args=p.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ.setdefault('HF_HOME',str(Path.home()/'hf_cache'));os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if args.worker:
        if not args.case:p.error('--worker requires --case')
        worker(args.worker,args.case)
    elif args.prepare_only:prepare();print('Prepared; no jobs submitted.')
    elif args.summarize:summarize()
    else:
        try:controller()
        except BlockingIOError:
            print('A controller is already running; no duplicate jobs submitted.',flush=True)
        except Exception as error:
            if MANIFEST.exists():change_manifest(lambda m:m.update(state='controller_failed',last_error=str(error),failed_at=now()))
            raise

if __name__=='__main__':main()
