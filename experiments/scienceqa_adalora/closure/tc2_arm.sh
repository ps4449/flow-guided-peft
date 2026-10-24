#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=03:00:00
#SBATCH --job-name=sqa_closure
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -eo pipefail
cd /home/msai/c250274/flow-guided-peft
arm="${1:?B C or D}";root="${2:?new pipeline output root}"
case "$arm" in B|C|D) ;; *) exit 2;; esac
module load anaconda
eval "$(conda shell.bash hook)"
conda activate llava_shared
set -u
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
code=experiments/scienceqa_adalora/closure
shared=experiments/scienceqa_adalora
run="$root/$arm"
[[ ! -e "$run" ]] || { echo 'Refusing to overwrite existing arm';exit 2; }
mkdir -p "$run"
python "$shared/validate_data.py"
python - "$run" <<'PY'
import hashlib,json,sys,importlib.metadata
from pathlib import Path
files=[p for d in ['experiments/scienceqa_adalora/closure','experiments/adalora_reproduction'] for p in Path(d).rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.json')]
files += [Path('datasets/scienceqa/train_llava.json'),Path('datasets/scienceqa/val_questions.jsonl'),Path('datasets/scienceqa/test_questions.jsonl')]
Path(sys.argv[1],'source_manifest.json').write_text(json.dumps({'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','peft','llava']},'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
PY
extra=(); [[ "$arm" == B ]] || extra+=(--no-pruning)
common=(--ranks "$code/configs/$arm.json" --seed 0 --train-file datasets/scienceqa/train_llava.json --image-folder datasets/scienceqa/images --epochs 1 --batch-size 1 --grad-accum 16 --num-workers 0 --lr 2e-4 --warmup-ratio .03 --max-length 2048 --save-every 200 --max-hours 2.2)
python -u "$code/train_engine.py" "${common[@]}" "${extra[@]}" --limit 320 --out-dir "$run/pilot"
python -u experiments/adalora_reproduction/check_shared_reload.py --run "$run/pilot"
python "$code/validate_arm.py" "$arm" "$run/pilot" pilot
python -u "$shared/eval_scienceqa.py" --adapter "$run/pilot/adapter" --questions datasets/scienceqa/val_questions.jsonl --limit 4 --out-dir "$run/pilot/eval_val4"
python - "$run/pilot/eval_val4/metrics.json" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))['total']==4
print('ABLATION_PILOT_PASSED',flush=True)
PY
# Existing A arm is reused; measure its validation split once, without retraining.
if [[ "$arm" == B ]]; then
 python -u "$shared/eval_scienceqa.py" --adapter output/peft/adalora/scienceqa_train_llava/seed0/adapter --questions datasets/scienceqa/val_questions.jsonl --out-dir "$root/A_eval_val"
fi
python -u "$code/train_engine.py" "${common[@]}" "${extra[@]}" --out-dir "$run/seed0"
python -u experiments/adalora_reproduction/check_shared_reload.py --run "$run/seed0"
python "$code/validate_arm.py" "$arm" "$run/seed0" full
python -u "$shared/eval_scienceqa.py" --adapter "$run/seed0/adapter" --questions datasets/scienceqa/val_questions.jsonl --out-dir "$run/seed0/eval_val"
python -u "$shared/eval_scienceqa.py" --adapter "$run/seed0/adapter" --questions datasets/scienceqa/test_questions.jsonl --out-dir "$run/seed0/eval_test"
python -u "$shared/eval_scienceqa_unmerged.py" --adapter "$run/seed0/adapter" --questions datasets/scienceqa/test_questions.jsonl --out-dir "$run/seed0/eval_test_unmerged"
python - "$run" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);norm=lambda s:s.strip().rstrip('.').lower()
for split,count,questions in [('eval_val',2097,'val'),('eval_test',2017,'test'),('eval_test_unmerged',2017,'test')]:
 refs={r['question_id']:r['answer'] for r in map(json.loads,Path(f'datasets/scienceqa/{questions}_questions.jsonl').read_text().splitlines())}
 rows=list(map(json.loads,(p/'seed0'/split/'predictions.jsonl').read_text().splitlines()));m=json.loads((p/'seed0'/split/'metrics.json').read_text())
 assert len(rows)==len({r['question_id'] for r in rows})==len(refs)==count
 assert {r['question_id']:r['answer'] for r in rows}==refs
 assert m['correct']==sum(norm(r['prediction'])==norm(r['answer']) for r in rows)
 assert m['total']==count
(p/'ARM_COMPLETED.json').write_text(json.dumps({'status':'completed','arm':p.name,'scope':'predeclared seed-0 diagnostic; no hyperparameter selection'})+'\n')
print('CLOSURE_ARM_COMPLETED',p,flush=True)
PY
