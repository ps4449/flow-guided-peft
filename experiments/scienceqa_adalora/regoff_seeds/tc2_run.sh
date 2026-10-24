#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=03:00:00
#SBATCH --job-name=sqa_regoff
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -eo pipefail
cd /home/msai/c250274/flow-guided-peft
seed="${1:?seed 1 or 2}";case "$seed" in 1|2) ;; *) exit 2;; esac
module load anaconda
eval "$(conda shell.bash hook)"
conda activate llava_shared
set -u
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
code=experiments/scienceqa_adalora
run="output/peft/adalora/scienceqa_regoff_confirm/seed${seed}"
[[ ! -e "$run" ]] || { echo 'Refusing duplicate output directory';exit 2; }
mkdir -p "$run"
python "$code/validate_data.py"
python - "$run" "$seed" <<'PY'
import hashlib,json,sys,importlib.metadata
from pathlib import Path
base=json.loads(Path('output/diagnostics/scienceqa_closure_seed0_20261024/B/source_manifest.json').read_text())['files']
files=['experiments/scienceqa_adalora/closure/train_engine.py','experiments/scienceqa_adalora/closure/configs/B.json','experiments/adalora_reproduction/adalora_runtime.py','experiments/adalora_reproduction/common.py','datasets/scienceqa/train_llava.json','datasets/scienceqa/val_questions.jsonl','datasets/scienceqa/test_questions.jsonl']
for name in files:assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==base[name],name
files += ['experiments/scienceqa_adalora/regoff_seeds/tc2_run.sh','experiments/scienceqa_adalora/regoff_seeds/validate.py','experiments/scienceqa_adalora/eval_scienceqa.py','experiments/scienceqa_adalora/eval_scienceqa_unmerged.py']
Path(sys.argv[1],'source_manifest.json').write_text(json.dumps({'seed':int(sys.argv[2]),'seed0_reference':'output/diagnostics/scienceqa_closure_seed0_20261024/B/seed0','versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','peft','llava']},'files':{n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in files}},indent=2)+'\n')
PY
# This exact configuration passed CPU, GPU pilot and full seed-0 checks already.
python -u "$code/closure/train_engine.py" --ranks "$code/closure/configs/B.json" --seed "$seed" \
 --train-file datasets/scienceqa/train_llava.json --image-folder datasets/scienceqa/images \
 --epochs 1 --batch-size 1 --grad-accum 16 --num-workers 0 --lr 2e-4 --warmup-ratio .03 \
 --max-length 2048 --save-every 200 --max-hours 2.2 --out-dir "$run"
python -u experiments/adalora_reproduction/check_shared_reload.py --run "$run"
python "$code/regoff_seeds/validate.py" B "$run" full "$seed"
python -u "$code/eval_scienceqa.py" --adapter "$run/adapter" --questions datasets/scienceqa/val_questions.jsonl --out-dir "$run/eval_val"
python -u "$code/eval_scienceqa.py" --adapter "$run/adapter" --questions datasets/scienceqa/test_questions.jsonl --out-dir "$run/eval_test"
python -u "$code/eval_scienceqa_unmerged.py" --adapter "$run/adapter" --questions datasets/scienceqa/test_questions.jsonl --out-dir "$run/eval_test_unmerged"
python "$code/regoff_seeds/summarize.py" "$seed"
echo "REGOFF_SEED_COMPLETED: $seed"
