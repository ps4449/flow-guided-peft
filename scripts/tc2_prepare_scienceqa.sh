#!/bin/bash
# TC2 job for scripts/prepare_scienceqa.py: writes the ScienceQA images to datasets/scienceqa/images/ (~5 min)
# QoS "normal" limits: 1 GPU, <=10 CPUs, <=30G RAM, <=6h wall, <=2 jobs at a time

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=prep_scienceqa
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Submit from the repo root after downloading the data on the head node (see SETUP.md):
#   sbatch scripts/tc2_prepare_scienceqa.sh

echo "Node: $(hostname)  Job: $SLURM_JOB_ID"

module load anaconda
eval "$(conda shell.bash hook)"
conda activate llava

python scripts/prepare_scienceqa.py --data-dir ${DATA_DIR:-$HOME/scienceqa_raw}
ls datasets/scienceqa/images | wc -l
