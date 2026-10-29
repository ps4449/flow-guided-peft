#!/bin/bash
# Login-node controller only; every GPU operation runs through sbatch.
set -euo pipefail
experiment_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$experiment_root"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1
exec /home/msai/swarangi001/.conda/envs/llava/bin/python -u \
  peft_lora/methods/pathway_specific/multiseed.py "$@"
