#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --job-name=sqa_prepare_smoke
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
prepared="output/diagnostics/scienceqa_prepared_${SLURM_JOB_ID}"
if [[ ! -d datasets/scienceqa/images ]]; then
    python "$code/prepare_images.py" --data-dir "$HOME/scienceqa_raw" --out-dir "$prepared"
    python "$code/validate_data.py" --prepared "$prepared"
    mv "$prepared/images" datasets/scienceqa/images
fi
python "$code/validate_data.py"
python "$code/check_prompts.py"
run="output/diagnostics/scienceqa_smoke_${SLURM_JOB_ID}"
python -u "$code/team_train_lora.py" --ranks configs/ranks/uniform.json --seed 0 \
    --train-file datasets/scienceqa/train_llava.json --image-folder datasets/scienceqa/images \
    --epochs 1 --batch-size 4 --grad-accum 4 --num-workers 0 --limit 320 --out-dir "$run/uniform"
python -u "$code/eval_scienceqa.py" --questions datasets/scienceqa/test_questions.jsonl \
    --adapter "$run/uniform/adapter" --out-dir "$run/uniform/eval_scienceqa" --limit 20
python -u "$code/train_adalora.py" --ranks experiments/adalora_reproduction/adalora.json --seed 0 \
    --train-file datasets/scienceqa/train_llava.json --image-folder datasets/scienceqa/images \
    --epochs 1 --batch-size 1 --grad-accum 16 --num-workers 0 --limit 320 \
    --lr 2e-4 --warmup-ratio .03 --max-length 2048 --out-dir "$run/adalora"
python -u experiments/adalora_reproduction/check_shared_reload.py --run "$run/adalora"
python -u "$code/eval_scienceqa.py" --questions datasets/scienceqa/test_questions.jsonl \
    --adapter "$run/adalora/adapter" --out-dir "$run/adalora/eval_scienceqa" --limit 20
python "$code/validate_outputs.py" smoke "$run"
