#!/bin/bash
# TC2 job for peft_lora/methods/layer_baselines/make_ranks.py: baseline layer choices (grad, spectrum, similarity)
# for pruned LoRA. Writes configs/ranks/<method>_pruned16_<dataset>.json and output/layer_baselines/<dataset>/scores.json
# QoS "normal" limits: 1 GPU, <=10 CPUs, <=30G RAM, <=6h wall, <=2 jobs at a time

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --job-name=layer_baselines
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Submit from the repo root:
#   sbatch scripts/tc2_layer_baselines.sh
#   sbatch --export=ALL,EXTRA="--train-file datasets/gqa/train_20000_llava.json" scripts/tc2_layer_baselines.sh

echo "Node: $(hostname)  Job: $SLURM_JOB_ID  Extra: $EXTRA"
nvidia-smi

module load anaconda
eval "$(conda shell.bash hook)"
conda activate llava

export HF_HOME=$HOME/hf_cache
export HF_HUB_OFFLINE=1   # weights must be pre-downloaded on the head node

python peft_lora/methods/layer_baselines/make_ranks.py $EXTRA
