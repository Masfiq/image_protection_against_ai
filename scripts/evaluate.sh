#!/bin/bash
#SBATCH --job-name=evaluate_edit_chains
#SBATCH --account=standard
#SBATCH --partition=peregrine-gpu
#SBATCH --qos=gpu_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=out_and_err/evaluate_%j.out
#SBATCH --error=out_and_err/evaluate_%j.err

# Falcon job: scores the results in data/result_version_2/ with evaluate.py.
# It only needs one GPU slice and takes a few minutes, so it uses the
# 30-minute debug queue (gpu_debug).
#
# Before the first submit (one time only):
#   1. Install the scoring package into projenv:
#        PYTHONNOUSERSITE=1 /s/chopin/e/proj/hyperspec/masfiq/projenv/bin/pip install lpips
#   2. Download the two scoring models (small, about 1 GB together):
#        export HF_HOME=/s/chopin/e/proj/hyperspec/masfiq/hf_cache
#        hf download facebook/dinov2-base
#        hf download openai/clip-vit-base-patch32 --exclude "*.h5" "*.msgpack"
#        hf download openai/clip-vit-base-patch32 --revision refs/pr/66 --include "*.json" "*.safetensors"
#      (the second CLIP line gets the weights in the safe format; see CLIP_REVISION in evaluate.py)
#   LPIPS also downloads a small model (AlexNet, ~230 MB) into ~/.cache/torch
#   the first time it runs.
#
# Submit from this scripts folder with:
#   sbatch evaluate.sh

# Stop at the first error.
set -e

# Ignore the packages in the home folder (~/.local); use only projenv's.
export PYTHONNOUSERSITE=1

# Same fix as in run_edit_chains.sh: the conda env ships a newer libstdc++.
export LD_LIBRARY_PATH="/s/chopin/e/proj/hyperspec/masfiq/projenv/lib:${LD_LIBRARY_PATH}"

PYTHON=/s/chopin/e/proj/hyperspec/masfiq/projenv/bin/python
export HF_HOME=/s/chopin/e/proj/hyperspec/masfiq/hf_cache
export PYTHONUNBUFFERED=1

# Run from the folder this job was submitted from (the scripts folder).
cd "$SLURM_SUBMIT_DIR"

srun $PYTHON -u evaluate.py
