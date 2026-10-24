"""Login-node controller: validated smoke, uniform seed0, AdaLoRA seeds0/1/2."""
import datetime,fcntl,json,os,subprocess,time,statistics,sys
from pathlib import Path
from validate_outputs import train,evaluate
ROOT=Path(__file__).resolve().parents[2]
STATE=ROOT/'output/peft/pipeline_scienceqa.json'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(s):
    s['updated_utc']=now();tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(STATE)
def result(job):
    r=subprocess.run(['sacct','-j',job,'--noheader','--parsable2','--format=JobIDRaw,State,ExitCode'],text=True,capture_output=True,timeout=30);r.check_returncode()
    for line in r.stdout.splitlines():
        x=line.split('|')
        if x[0]==job:return x[1].split()[0].rstrip('+'),x[2]
    return 'UNKNOWN',''
def wait(job,state):
    deadline=time.monotonic()+48*3600
    while time.monotonic()<deadline:
        status,code=result(job)
        if status not in {'UNKNOWN','PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','RESIZING','REQUEUED'}:
            state['history'][-1].update(state=status,exit_code=code,finished_utc=now());save(state)
            return status,code
        time.sleep(60)
    raise RuntimeError('48-hour waiting deadline; active job left untouched')
def main():
    global STATE
    recovery="--resume-after-uniform" in sys.argv
    if recovery:STATE=ROOT/"output/peft/pipeline_scienceqa_recovery.json"
    os.chdir(ROOT);STATE.parent.mkdir(parents=True,exist_ok=True)
    lock=STATE.with_suffix('.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not STATE.exists(),'Inspect existing controller before restarting'
    smoke=sys.argv[1];assert smoke.isdigit()
    state=dict(status='WAITING_FOR_SMOKE',pid=os.getpid(),started_utc=now(),job_id=smoke,
               purpose='User authorized ScienceQA uniform seed0 and AdaLoRA seeds0/1/2; no zero-shot or flow runs',
               history=[dict(stage='prepare_and_smoke',job_id=smoke)])
    save(state)
    try:
        status,code=wait(smoke,state)
        assert (status,code)==('COMPLETED','0:0'),f'Smoke {smoke}: {status} {code}'
        smoke_result=json.loads((ROOT/'output/diagnostics/scienceqa_smoke_passed.json').read_text())
        assert smoke_result['status']=='SCIENCEQA_SMOKE_PASSED' and smoke in smoke_result['run']
        if recovery:
            uniform=ROOT/'output/peft/uniform/scienceqa_train_llava/seed0'
            train(uniform,'uniform',0);evaluate(uniform)
            print('REUSED_VALIDATED_UNIFORM_SEED0',flush=True)
        methods=[('adalora',0),('adalora',1),('adalora',2)] if recovery else [('uniform',0),('adalora',0),('adalora',1),('adalora',2)]
        for method,seed in methods:
            run=ROOT/f'output/peft/{method}/scienceqa_train_llava/seed{seed}'
            assert not run.exists(),f'Inspect pre-existing run before starting: {run}'
            for phase in ['train','eval']:
                for attempt in range(3):
                    name=f'sqa_{method}_s{seed}_{phase}'
                    state.update(status='SUBMITTING',stage=name,job_id=None);save(state)
                    r=subprocess.run(['sbatch','--parsable','--job-name='+name,
                                      'experiments/scienceqa_adalora/tc2_full.sh',method,phase,str(seed)],text=True,capture_output=True,timeout=60)
                    r.check_returncode();job=r.stdout.strip().split(';')[0];assert job.isdigit()
                    state.update(status='WAITING_FOR_JOB',job_id=job)
                    state['history'].append(dict(stage=name,job_id=job,submitted_utc=now()));save(state)
                    print('SUBMITTED',name,job,flush=True)
                    status,code=wait(job,state)
                    if phase=='train' and (status,code)==('FAILED','3:0') and attempt<2:
                        assert (run/'checkpoint.pt').is_file();continue
                    assert (status,code)==('COMPLETED','0:0'),f'{name} {job}: {status} {code}'
                    if phase=='train':train(run,method,seed)
                    else:evaluate(run)
                    print('VALIDATED',name,job,flush=True);break
        metrics=[dict(seed=seed,**evaluate(ROOT/f'output/peft/adalora/scienceqa_train_llava/seed{seed}')) for seed in [0,1,2]]
        scores=[100*m['correct']/m['total'] for m in metrics]
        report=dict(method='AdaLoRA ScienceQA',seeds=metrics,mean_accuracy=statistics.mean(scores),sample_std_accuracy=statistics.stdev(scores),std_definition='ddof1; percentage points')
        (ROOT/'output/peft/adalora/scienceqa_train_llava/three_seed_summary.json').write_text(json.dumps(report,indent=2)+'\n')
        state.update(status='COMPLETE',finished_utc=now());save(state)
    except Exception as error:
        state.update(status='STOPPED_FOR_INSPECTION',error=str(error));save(state)
        print('PIPELINE_STOPPED',repr(error),flush=True);raise
if __name__=='__main__':main()
