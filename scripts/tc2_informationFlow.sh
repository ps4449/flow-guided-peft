#!/bin/bash
# TC2 (CCDS GPU cluster) version of informationFlow.sh
# QoS "normal" limits: 1 GPU, <=10 CPUs, <=30G RAM, <=6h wall, <=2 jobs at a time

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --job-name=infoflow
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# SLURM does not create slurm/logs; it must exist before sbatch (submit from the repo root)

# Any setting below can be overridden at submit time, e.g.
#   sbatch --export=ALL,DATASET=datasets/test_ChooseAttr.csv,BLOCK="Image->Last" scripts/tc2_informationFlow.sh

current_window=${WINDOW:-9}

#"Question->Last"
#"Image->Last"
#"Image->Question"
#"Last->Last"
#"Image Central Object->Question"
#"Image Without Central Object->Question"
current_block_desc=${BLOCK:-"Question->Last"}

#models
#model_path="liuhaotian/llava-v1.6-vicuna-7b" convmode="vicuna_v1"
#model_path="lmms-lab/llama3-llava-next-8b"  convmode="llava_llama_3"
#model_path="liuhaotian/llava-v1.5-7b"   convmode="vicuna_v1"
#model_path="liuhaotian/llava-v1.5-13b"   convmode="vicuna_v1"
model_path=${MODEL:-"liuhaotian/llava-v1.5-7b"}
convmode=${CONVMODE:-"vicuna_v1"}

#dataset
dataset=${DATASET:-datasets/GQA_val_correct_question_with_choose_ChooseAttr.csv}
#dataset=datasets/GQA_val_correct_question_with_positionQuery_QueryAttr.csv
#dataset=datasets/GQA_val_correct_question_with_existThatOr_LogicalObj.csv
#dataset=datasets/GQA_val_correct_question_with_twoCommon_CompareAttr.csv
#dataset=datasets/GQA_val_correct_question_with_relChooser_ChooseRel.csv
#dataset=datasets/GQA_val_correct_question_with_categoryThatThisChoose_objThisChoose_ChooseCat.csv
#dataset=datasets/GQA_train_correct_All.csv   # training questions zero-shot gets right
#dataset=datasets/GQA_train_wrong_All.csv     # training questions zero-shot gets wrong (keeps wrong answers only)

imagefolder=$HOME/flow-guided-peft/datasets/gqa/images/

echo "Node: $(hostname)  Job: $SLURM_JOB_ID"
echo "Window: $current_window  Block: $current_block_desc"
echo "Model: $model_path  Dataset: $dataset"
nvidia-smi

module load anaconda
eval "$(conda shell.bash hook)"
conda activate llava

export HF_HOME=$HOME/hf_cache
export HF_HUB_OFFLINE=1   # weights must be pre-downloaded on the head node

python InformationFlow.py \
        --model-path $model_path \
        --image-folder $imagefolder \
        --temperature 0 \
        --conv-mode $convmode \
        --refined_dataset $dataset \
        --window $current_window \
        --block_description "$current_block_desc" \
        --num_workers 2
