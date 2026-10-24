#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --job-name=scienceqa_full
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -eo pipefail
cd /home/msai/c250274/flow-guided-peft
method="${1:?uniform or adalora}";stage="${2:?train or eval}";seed="${3:?0 1 or 2}"
case "$method:$seed" in uniform:0|adalora:0|adalora:1|adalora:2) ;; *) exit 2;; esac
case "$stage" in train|eval) ;; *) exit 2;; esac
module load anaconda
eval "$(conda shell.bash hook)"
conda activate llava_shared
set -u
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
code=experiments/scienceqa_adalora
run="output/peft/${method}/scienceqa_train_llava/seed${seed}"
python "$code/validate_data.py"
python - <<'CHECK'
import json
from pathlib import Path
assert json.loads(Path('output/diagnostics/scienceqa_smoke_passed.json').read_text())['status']=='SCIENCEQA_SMOKE_PASSED'
CHECK
if [[ "$stage" == train ]]; then
    python "$code/record_environment.py" "$method" "$seed" "$run"
    if [[ "$method" == uniform ]]; then
        python -u "$code/team_train_lora.py" --ranks configs/ranks/uniform.json --seed "$seed" \
            --train-file datasets/scienceqa/train_llava.json --image-folder datasets/scienceqa/images \
            --epochs 1 --batch-size 4 --grad-accum 4 --num-workers 0 --lr 2e-4 \
            --warmup-ratio .03 --save-every 200 --max-hours 5.3 --out-dir "$run"
    else
        python -u "$code/train_adalora.py" --ranks experiments/adalora_reproduction/adalora.json --seed "$seed" \
            --train-file datasets/scienceqa/train_llava.json --image-folder datasets/scienceqa/images \
            --epochs 1 --batch-size 1 --grad-accum 16 --num-workers 0 --lr 2e-4 \
            --warmup-ratio .03 --max-length 2048 --save-every 200 --max-hours 5.3 --out-dir "$run"
        python -u experiments/adalora_reproduction/check_shared_reload.py --run "$run"
    fi
    python "$code/validate_outputs.py" train "$run" "$method" "$seed"
else
    python "$code/validate_outputs.py" train "$run" "$method" "$seed"
    python -u "$code/eval_scienceqa.py" --questions datasets/scienceqa/test_questions.jsonl \
        --adapter "$run/adapter" --out-dir "$run/eval_scienceqa"
    python "$code/validate_outputs.py" eval "$run"
fi
