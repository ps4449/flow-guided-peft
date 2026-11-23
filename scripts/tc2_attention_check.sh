#!/bin/bash
# TC2 job for scripts/check_attention_proxy.py: image attention on correct vs wrong answers (~15 min, no training)
# QoS "normal" limits: 1 GPU, <=10 CPUs, <=30G RAM, <=6h wall, <=2 jobs at a time

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --job-name=attn_check
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Submit from the repo root, e.g.
#   sbatch scripts/tc2_attention_check.sh                                       # GQA
#   sbatch --export=ALL,DATA=SQA,NAME=ScienceQA scripts/tc2_attention_check.sh   # ScienceQA
# Quick test: add EXTRA="--limit 20" and --time=00:15:00

data=${DATA:-GQA}
name=${NAME:-All}

echo "Node: $(hostname)  Job: $SLURM_JOB_ID"
echo "Files: datasets/${data}_train_{correct,wrong}_${name}.csv  Extra: $EXTRA"
nvidia-smi

module load anaconda
eval "$(conda shell.bash hook)"
conda activate llava

export HF_HOME=$HOME/hf_cache
export HF_HUB_OFFLINE=1   # weights must be pre-downloaded on the head node

python scripts/check_attention_proxy.py \
        --correct datasets/${data}_train_correct_${name}.csv \
        --wrong datasets/${data}_train_wrong_${name}.csv \
        $EXTRA
