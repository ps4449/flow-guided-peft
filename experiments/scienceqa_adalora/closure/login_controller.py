"""Finite B/C/D pipeline on the login node. Stops after these arms or first failure."""
import json,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);state=root/'pipeline.json'
if state.exists():raise RuntimeError('Pipeline already exists; do not duplicate it')
root.mkdir(parents=True,exist_ok=True)
s={'status':'starting','scope':['B','C','D'],'jobs':[],'stop_rule':'Exactly three predeclared conditions; no follow-up training or best-arm selection.'}
def save():state.write_text(json.dumps(s,indent=2)+'\n')
save()
try:
 for arm in s['scope']:
  job=subprocess.check_output(['sbatch','--parsable','experiments/scienceqa_adalora/closure/tc2_arm.sh',arm,str(root)],text=True).strip().split(';')[0]
  assert job.isdigit(),job
  s['jobs'].append({'arm':arm,'job_id':job});s['status']='running';s['current_arm']=arm;save();print(arm,job,flush=True)
  deadline=time.monotonic()+5*3600
  while True:
   text=subprocess.check_output(['sacct','-n','-X','-j',job,'--format=State','-P'],text=True).strip()
   status=text.splitlines()[0].split('|')[0].strip() if text else 'UNKNOWN'
   if status=='COMPLETED':
    if not (root/arm/'ARM_COMPLETED.json').exists():raise RuntimeError('Missing completion artifact '+arm)
    s['jobs'][-1]['state']=status;save();break
   if status.startswith(('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE')):raise RuntimeError(f'{arm} job {job}: {status}')
   if time.monotonic()>deadline:raise RuntimeError('Controller wait deadline exceeded; inspect job before continuing')
   time.sleep(30)
 s['status']='complete';s.pop('current_arm',None);save()
 subprocess.run([sys.executable,'experiments/scienceqa_adalora/closure/summarize.py',str(root)],check=True)
 print('CLOSURE_COMPLETE_NO_MORE_RUNS',flush=True)
except Exception as error:
 s['status']='stopped';s['error']=str(error);save();raise
