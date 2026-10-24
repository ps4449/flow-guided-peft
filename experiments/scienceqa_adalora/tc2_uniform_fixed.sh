#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=03:00:00
#SBATCH --job-name=sqa_uniform_fixed
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
run=output/peft/uniform/scienceqa_modalities_fixed/seed0
python "$code/validate_data.py"
python "$code/record_uniform_fixed.py" "$run"
python -u "$code/train_uniform_modalities_fixed.py" --ranks configs/ranks/uniform.json --seed 0 \
 --train-file datasets/scienceqa/train_llava.json --image-folder datasets/scienceqa/images \
 --epochs 1 --batch-size 4 --grad-accum 4 --num-workers 0 --lr 2e-4 \
 --warmup-ratio .03 --save-every 200 --max-hours 2.5 --out-dir "$run"
python "$code/validate_outputs.py" train "$run" uniform 0
python -u "$code/eval_scienceqa.py" --adapter "$run/adapter" \
 --questions datasets/scienceqa/test_questions.jsonl --out-dir "$run/eval_scienceqa"
python "$code/validate_outputs.py" eval "$run"
python -u "$code/eval_scienceqa_unmerged.py" --adapter "$run/adapter" \
 --questions datasets/scienceqa/test_questions.jsonl --out-dir "$run/eval_scienceqa_unmerged"
