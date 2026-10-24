#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=00:35:00
#SBATCH --job-name=sqa_unmerged_check
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -eo pipefail
cd /home/msai/c250274/flow-guided-peft
module load anaconda
eval "$(conda shell.bash hook)"
conda activate llava_shared
set -u
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
for method in uniform adalora; do
 run="output/peft/${method}/scienceqa_train_llava/seed0"
 python -u experiments/scienceqa_adalora/eval_scienceqa_unmerged.py \
   --adapter "$run/adapter" --questions datasets/scienceqa/test_questions.jsonl \
   --out-dir "$run/eval_scienceqa_unmerged"
done
