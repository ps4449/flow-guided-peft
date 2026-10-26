#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=03:00:00
#SBATCH --job-name=sqa_adalora
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# From repo root: mkdir -p slurm/logs; sbatch experiments/scienceqa_adalora/tc2_run.sh regularization_off 0
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
condition="${1:?regularization_off, regularization_on, no_pruning, neither}"
seed="${2:?seed 0, 1 or 2}"
case "$condition" in regularization_off|regularization_on|no_pruning|neither) ;; *) exit 2;; esac
case "$seed" in 0|1|2) ;; *) exit 2;; esac
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${ADALORA_ENV:-llava_shared}"
set -u
export HF_HOME="${HF_HOME:-$HOME/hf_cache}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
code=experiments/scienceqa_adalora
run="output/peft/adalora/scienceqa_reproduction/${condition}/seed${seed}_job${SLURM_JOB_ID}"
[[ ! -e "$run" ]] || { echo 'Refusing to overwrite an existing run'; exit 2; }
python "$code/validate_data.py"
mkdir -p "$run"
python - "$run" "$code" <<'PY'
import hashlib,importlib.metadata,json,sys
from pathlib import Path
code=Path(sys.argv[2])
files=[p for folder in [code,Path('experiments/adalora_reproduction')] for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.json')]
files += [Path(f'datasets/scienceqa/{name}') for name in ('train_llava.json','val_questions.jsonl','test_questions.jsonl')]
Path(sys.argv[1],'source_manifest.json').write_text(json.dumps({'versions':{n:importlib.metadata.version(n) for n in ('torch','transformers','peft','llava')},'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
PY
extra=(); case "$condition" in no_pruning|neither) extra+=(--no-pruning);; esac
python -u "$code/train.py" --ranks "$code/configs/$condition.json" --seed "$seed" \
 --train-file datasets/scienceqa/train_llava.json --image-folder datasets/scienceqa/images \
 --epochs 1 --batch-size 1 --grad-accum 16 --num-workers 0 --lr 2e-4 --warmup-ratio .03 \
 --max-length 2048 --save-every 200 --max-hours 2.2 --out-dir "$run" "${extra[@]}"
python experiments/adalora_reproduction/check_shared_reload.py --run "$run"
python "$code/validate_run.py" --run "$run"
python -u "$code/evaluate.py" --adapter "$run/adapter" --questions datasets/scienceqa/val_questions.jsonl --out-dir "$run/eval_val"
python -u "$code/evaluate.py" --adapter "$run/adapter" --questions datasets/scienceqa/test_questions.jsonl --out-dir "$run/eval_test"
python -u "$code/evaluate.py" --adapter "$run/adapter" --questions datasets/scienceqa/test_questions.jsonl --unmerged --out-dir "$run/eval_test_unmerged"
python "$code/validate_run.py" --run "$run"
echo "SCIENCEQA_ADALORA_REPRODUCTION_COMPLETE: $run"
