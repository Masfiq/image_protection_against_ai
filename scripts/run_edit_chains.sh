#!/bin/bash
#SBATCH --job-name=photoguard_edit_chains
#SBATCH --account=standard
#SBATCH --partition=peregrine-gpu
#SBATCH --qos=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=out_and_err/run_edit_chains_%j.out
#SBATCH --error=out_and_err/run_edit_chains_%j.err

# Falcon job: protects the photos with PhotoGuard, then runs the edit chains.
# Results go to data/result_version_2/ (Editor A = InstructPix2Pix, Editor B = FLUX.1 Kontext).
#
# Both Python scripts start NUM_WORKERS worker processes (set below), spread over
# the 2 GPUs. Falcon allows at most 2 GPUs per job on peregrine-gpu.
# The number of CPUs (10) and the number of workers (2) are set separately:
#   - workers are limited by GPU memory: FLUX Kontext is very large (~24 GB on the
#     GPU at a time, ~34 GB in normal memory), so only 1 worker fits per 40 GB GPU.
#   - the extra CPUs help with loading models and reading/saving images.
# --time=8:00:00: FLUX is much slower than the old Editor B (rough guess: 3-4 hours).
# --qos=gpu_short allows up to 24 hours (gpu_medium would allow 3 days).
#
# Before the first submit (one time only):
#   1. Install the missing packages into projenv. The versions are pinned because
#      projenv has torch 2.5.1, and diffusers 0.41+ needs a newer torch:
#        /s/chopin/e/proj/hyperspec/masfiq/projenv/bin/pip install "diffusers==0.35.2" "transformers==4.56.2" "huggingface-hub>=0.34,<1.0" accelerate sentencepiece protobuf
#   2. Download the photos:
#        /s/chopin/e/proj/hyperspec/masfiq/projenv/bin/python download_dreambooth.py
#   3. Download the models (on a machine with internet, e.g. the login node):
#        export HF_HOME=/s/chopin/e/proj/hyperspec/masfiq/hf_cache
#        hf download stable-diffusion-v1-5/stable-diffusion-v1-5 --include "vae/*"
#        hf download timbrooks/instruct-pix2pix --exclude "*.ckpt"
#      FLUX.1 Kontext needs permission first:
#        a. Log in at huggingface.co and accept the license on
#           https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
#        b. Create a "Read" token at https://huggingface.co/settings/tokens
#        c. With HF_HOME exported as above, run:  hf auth login   (paste the token)
#        d. Download (~33 GB; the exclude skips a second 24 GB copy we don't need):
#             hf download black-forest-labs/FLUX.1-Kontext-dev --exclude "flux1-kontext-dev.safetensors"
#   4. Make the log folder (Slurm does not create it):
#        mkdir -p out_and_err
#
# Submit from this scripts folder with:
#   sbatch run_edit_chains.sh

# Stop at the first error.
set -e

# Number of worker processes for both Python scripts (1 per 40 GB GPU, because of FLUX).
export NUM_WORKERS=2

# Each worker process uses one CPU thread, so the workers don't fight over the CPUs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Same fix as in the biomass job scripts: the conda env ships a newer libstdc++
# than the system one.
export LD_LIBRARY_PATH="/s/chopin/e/proj/hyperspec/masfiq/projenv/lib:${LD_LIBRARY_PATH}"

# The Python environment (same one the biomass jobs use).
PYTHON=/s/chopin/e/proj/hyperspec/masfiq/projenv/bin/python

# Where the Hugging Face models are stored (shared disk, ~45 GB with FLUX).
export HF_HOME=/s/chopin/e/proj/hyperspec/masfiq/hf_cache

export PYTHONNOUSERSITE=1

# Models are downloaded automatically into HF_HOME the first time they are needed
# (this needs internet on the compute node). If the job fails because the node has
# no internet, do step 3 above and then remove the "#" in front of the next line.
# export HF_HUB_OFFLINE=1

# Show the worker processes' messages in the log straight away
# (python -u alone only covers the main process).
export PYTHONUNBUFFERED=1

# Run from the folder this job was submitted from (the scripts folder).
cd "$SLURM_SUBMIT_DIR"

nvidia-smi

srun $PYTHON -u protect_photoguard.py
srun $PYTHON -u run_edit_chains.py
