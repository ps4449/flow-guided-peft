# Instructions for the NTU TC2 GPU

## Part 1: Visual Information Flow Project

### Setup the Visual Information Flow project

1. ssh <ntu_id>@10.96.189.12
2. Upload the repo from your device to the cluster (`rsync -av --exclude .git --exclude '*.pdf' ~/Desktop/flow-guided-peft/ <ntu_id>@10.96.189.12:~/flow-guided-peft/`)
3. Load conda, then create and activate environment: `module load anaconda` and `conda create -n llava python=3.10 -y && conda activate llava`
4. Inside the project folder (`cd ~/flow-guided-peft`), clone the LlaVa model: `git clone https://github.com/LLaVA-VL/LLaVA-NeXT`
5. Install LLaVA-NeXT from the previously downloaded repository and install everything it considers necessary for training: `cd LLaVA-NeXT && pip install -e ".[train]"`
6. Go back to the project folder (`cd ..`) and install the rest of the required packages: `pip install seaborn matplotlib pandas tqdm`
7. Download the pretrained model files fron HuggingFace into the server:
`export HF_HOME=~/hf_cache` `# sets the Hugging Face cache directory`
`huggingface-cli download liuhaotian/llava-v1.5-7b`
`huggingface-cli download openai/clip-vit-large-patch14-336`
8. Download the GQA image dataset (20 GB, takes a few hours). `nohup` keeps it running if you disconnect, and `-c` resumes a stopped download: 
`mkdir -p ~/flow-guided-peft/datasets/gqa && cd ~/flow-guided-peft/datasets/gqa`
`nohup wget -c https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip > wget.log 2>&1 &`
Note: Do not delete the images.zip yet, as it is contains the train and test datasets for LlaVa
Check progress with `tail -1 wget.log`. It is done when it says `'images.zip' saved [21817965542/21817965542]`
9. `datasets/gqa/needed_info_flow.txt` (in the repo) lists the 3,818 images our CSVs use, out of the 148,000 in the zip. Check it is there: `wc -l needed_info_flow.txt`
10. Extract only those images (takes a few minutes): `unzip -q images.zip $(cat needed_info_flow.txt)`
11. Check that 3,818 images were extracted: `ls images | wc -l`

### Run the Visual Information Flow experiments

Jobs must be submitted with `sbatch`. The job script is `scripts/tc2_informationFlow.sh`.

1. Go to the project folder and create the logs folder: `cd ~/flow-guided-peft && mkdir -p slurm/logs`
2. [Optional] Make a small test set with the first 20 questions: `head -n 21 datasets/GQA_val_correct_question_with_choose_ChooseAttr.csv > datasets/test_ChooseAttr.csv`
3. [Optional] Run a short test job (30 min limit) to check everything works: `sbatch --time=00:30:00 --export=ALL,DATASET=datasets/test_ChooseAttr.csv scripts/tc2_informationFlow.sh`
4. Run the full experiments on ChooseAttr, one job per flow (only 1 runs at a time; the next starts when it finishes):
`sbatch --export=ALL,BLOCK="Image->Question" scripts/tc2_informationFlow.sh` `# ~2.7 h`
`sbatch --export=ALL,BLOCK="Question->Last" scripts/tc2_informationFlow.sh` `# ~1.6 h`
`sbatch --export=ALL,BLOCK="Image->Last" scripts/tc2_informationFlow.sh` `# ~1.6 h`
5. To use another dataset, add `DATASET=` e.g. `sbatch --export=ALL,DATASET=datasets/GQA_val_correct_question_with_positionQuery_QueryAttr.csv,BLOCK="Question->Last" scripts/tc2_informationFlow.sh`
6. Results (a CSV and a PDF plot per run) are saved in `output/information_flow/llava_v1_5_7b/<dataset>/val/<flow>/`

Note: For PEFT using LoRA, we do not need to run these experiments on all the 6 datasets. A verification for the setup has already been done using the ChooseAttr dataset, and visual information flow graphs obtained. Our project focuses on enhancing this visual information flow with PEFT.

## Part 2: PEFT using LoRA

### Obtain Training and Testing Data (for LoRA)

We use 3 separate GQA splits: val (the CSVs above) to find the important layers, train to fine-tune with LoRA, and testdev to evaluate. Training uses a fixed sample of 20,000 train questions, so everyone trains on the same data.

On your device (done once, by one person, and pushed to the repo): [DONE]
1. Download the GQA questions (1.4 GB) from https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
2. Extract only the 2 files we need: `unzip questions1.2.zip train_balanced_questions.json testdev_balanced_questions.json -d ~/Downloads/gqa_questions`
3. From the project folder, build the training and testing files: `python3 scripts/prepare_gqa_splits.py --questions-dir ~/Downloads/gqa_questions`
This creates in `datasets/gqa/`:
`train_20000_llava.json` `# 20k train questions in LLaVA's training format`
`needed_train.txt` `# the images those questions use`
`testdev_questions.jsonl` `# 12,578 testdev questions and answers for evaluation`
`needed_testdev.txt` `# the images the testdev questions use`
4. Push these 4 files to the repo. They are small, so the original question JSONs (too big for the repo) are not needed after this.

On the cluster: [To by done by everyone on their GPU instance]
1. Update the project so `datasets/gqa/` has the 4 new files, then go to: `cd ~/flow-guided-peft/datasets/gqa`
2. If you deleted `images.zip`, download it again
3. Extract the train and testdev images: `unzip -q -n images.zip $(cat needed_train.txt needed_testdev.txt)`
4. Check the image count: `ls images | wc -l` (the 3,818 from before plus the new ones)
5. Delete the zip to free 20 GB once all 3 image lists are extracted: `rm images.zip`