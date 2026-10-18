#!/bin/bash
# TC2 job for peft_lora/eval_gqa.py (GQA testdev-balanced, ~12.5k questions)
# QoS "normal" limits: 1 GPU, <=10 CPUs, <=30G RAM, <=6h wall, <=2 jobs at a time

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --job-name=lora_eval
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Submit from the repo root, e.g.
#   sbatch --export=ALL,ADAPTER=output/peft/flow_guided/seed0/adapter scripts/tc2_eval.sh
#   sbatch scripts/tc2_eval.sh                                  # zero-shot, no adapter
# Quick test on 100 questions: add EXTRA="--limit 100" and --time=00:30:00
# If the job stops, submit the same command again; it skips questions already answered.

echo "Node: $(hostname)  Job: $SLURM_JOB_ID"
echo "Adapter: ${ADAPTER:-none (zero-shot)}  Extra: $EXTRA"
nvidia-smi

module load anaconda
eval "$(conda shell.bash hook)"
conda activate llava

export HF_HOME=$HOME/hf_cache
export HF_HUB_OFFLINE=1   # weights must be pre-downloaded on the head node

python peft_lora/eval_gqa.py ${ADAPTER:+--adapter $ADAPTER} $EXTRA
