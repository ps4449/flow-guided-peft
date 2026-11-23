#!/bin/bash
# TC2 job for peft_lora/train_lora.py, the same for every method (the rank file sets the method)
# QoS "normal" limits: 1 GPU, <=10 CPUs, <=30G RAM, <=6h wall, <=2 jobs at a time

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --job-name=lora_train
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Submit from the repo root, e.g.
#   sbatch --export=ALL,RANKS=configs/ranks/flow_guided.json,SEED=0 scripts/tc2_train.sh
# Quick test on 100 questions:
#   sbatch --time=00:30:00 --export=ALL,RANKS=configs/ranks/flow_guided.json,EXTRA="--limit 100" scripts/tc2_train.sh
# If the job stops before the time limit (exit code 3), submit the same command again to resume.

ranks=${RANKS:?set RANKS=configs/ranks/<method>.json}
seed=${SEED:-0}

echo "Node: $(hostname)  Job: $SLURM_JOB_ID"
echo "Ranks: $ranks  Seed: $seed  Extra: $EXTRA"
nvidia-smi

module load anaconda
eval "$(conda shell.bash hook)"
conda activate llava

export HF_HOME=$HOME/hf_cache
export HF_HUB_OFFLINE=1   # weights must be pre-downloaded on the head node

python peft_lora/train_lora.py --ranks $ranks --seed $seed $EXTRA
