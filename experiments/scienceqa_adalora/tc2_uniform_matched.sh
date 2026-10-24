#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --job-name=sqa_uniform_match
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -eo pipefail
cd /home/msai/c250274/flow-guided-peft
module load anaconda
eval "$(conda shell.bash hook)"
conda activate llava_shared
set -u
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
code=experiments/scienceqa_adalora
root="output/peft/uniform/scienceqa_matched_engine/job${SLURM_JOB_ID}"
mkdir -p "$root"
python "$code/validate_data.py"
python - "$root" <<'PY'
import hashlib,json,sys
from pathlib import Path
files=['experiments/scienceqa_adalora/adalora_train_engine.py','experiments/scienceqa_adalora/train_uniform_matched.py','experiments/scienceqa_adalora/batch_guard.py','experiments/scienceqa_adalora/tc2_uniform_matched.sh','configs/ranks/uniform.json','datasets/scienceqa/train_llava.json']
Path(sys.argv[1],'source_manifest.json').write_text(json.dumps({n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in files},indent=2)+'\n')
PY
common=(--ranks configs/ranks/uniform.json --seed 0 --train-file datasets/scienceqa/train_llava.json --image-folder datasets/scienceqa/images --epochs 1 --batch-size 1 --grad-accum 16 --num-workers 0 --lr 2e-4 --warmup-ratio .03 --max-length 2048 --save-every 200 --max-hours 2.8)
# Bounded two-update pilot tests the static adapter/export/reload/generation path.
python -u "$code/train_uniform_matched.py" "${common[@]}" --limit 32 --out-dir "$root/pilot"
python -u experiments/adalora_reproduction/check_shared_reload.py --run "$root/pilot"
python -u "$code/eval_scienceqa.py" --adapter "$root/pilot/adapter" --questions datasets/scienceqa/test_questions.jsonl --limit 4 --out-dir "$root/pilot/eval_scienceqa"
python - "$root/pilot" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);assert json.loads((p/'batch_integrity.json').read_text())['language_rows']==32
assert json.loads((p/'reload_check.json').read_text())['passed']
assert json.loads((p/'eval_scienceqa/metrics.json').read_text())['total']==4
print('MATCHED_UNIFORM_PILOT_PASSED',flush=True)
PY
python -u "$code/train_uniform_matched.py" "${common[@]}" --out-dir "$root/seed0"
python -u experiments/adalora_reproduction/check_shared_reload.py --run "$root/seed0"
python "$code/validate_outputs.py" train "$root/seed0" uniform 0
python -u "$code/eval_scienceqa.py" --adapter "$root/seed0/adapter" --questions datasets/scienceqa/test_questions.jsonl --out-dir "$root/seed0/eval_scienceqa"
python "$code/validate_outputs.py" eval "$root/seed0"
python -u "$code/eval_scienceqa_unmerged.py" --adapter "$root/seed0/adapter" --questions datasets/scienceqa/test_questions.jsonl --out-dir "$root/seed0/eval_scienceqa_unmerged"
echo "MATCHED_UNIFORM_CONTROL_COMPLETED: $root"
