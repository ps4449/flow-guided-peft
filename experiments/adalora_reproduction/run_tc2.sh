#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --job-name=adalora_repro
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -eo pipefail
stage="${1:?Supply train or eval}"
seed="${2:?Supply seed 0, 1 or 2}"
case "$stage" in train|eval) ;; *) echo 'Expected train or eval' >&2;exit 2;; esac
case "$seed" in 0|1|2) ;; *) echo 'Expected seed 0, 1 or 2' >&2;exit 2;; esac
# sbatch copies this script to its spool; submit from the repository root.
root="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$root"
code=experiments/adalora_reproduction
module load anaconda
eval "$(conda shell.bash hook)"
conda activate llava_shared
set -u
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
python "$code/verify_protocol.py" --check-env
run="output/peft/adalora_reproduction/seed${seed}"
if [[ "$stage" == train ]]; then
    python "$code/prepare_data.py"
    python -u "$code/train_lora.py" --ranks "$code/adalora.json" --seed "$seed" \
        --train-file datasets/gqa/train_20000_capfirst_diagnostic.json --epochs 1 \
        --batch-size 1 --grad-accum 16 --num-workers 0 --lr 2e-4 --warmup-ratio .03 \
        --max-length 2048 --save-every 200 --max-hours 5.3 --out-dir "$run"
    python -u "$code/check_shared_reload.py" --run "$run"
else
    python - "$run" <<'CHECK'
import json,sys
from pathlib import Path
folder=Path(sys.argv[1])
assert json.loads((folder/'reload_check.json').read_text())['passed']
cfg=json.loads((folder/'train_config.json').read_text())
assert cfg['total_steps']==1250 and cfg['final_rank']['retained_rank_sum']==512
assert json.loads((folder/'train_log.jsonl').read_text().splitlines()[-1])['step']==1250
CHECK
    # Frozen inference code, then current shared evaluator only to add types.
    python -u "$code/eval_gqa.py" --adapter "$run/adapter" --out-dir "$run/eval"
    python -u peft_lora/eval_gqa.py --adapter "$run/adapter" --out-dir "$run/eval"
fi
