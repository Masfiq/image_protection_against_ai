# Make a protected copy of every photo using PhotoGuard (the "encoder attack").
#
# How PhotoGuard works, in short:
#   Stable Diffusion first compresses a photo into a small summary (the "latent")
#   using its image compressor (the VAE). PhotoGuard adds tiny changes to the
#   pixels so that this summary becomes meaningless (pushed towards all zeros).
#   The photo looks the same to people, but the editor "sees" an empty picture.
#
#   The changes are found step by step:
#     1. Compress the photo and measure how far the summary is from zero.
#     2. Work out which direction to nudge each pixel to make that smaller.
#     3. Nudge every pixel a tiny bit in that direction.
#     4. Make sure no pixel has moved more than EPSILON (keeps it invisible).
#     5. Repeat NUM_STEPS times.
#
# Speed-up with several worker processes:
#   The photos are split into NUM_WORKERS groups. Each worker is a separate
#   process with its own copy of the compressor, and handles one group.
#   Workers are spread evenly over the GPUs (with 10 workers and 2 GPUs,
#   5 workers share each GPU).
#
# Input:  data/dreambooth/images_512/*.png        (unprotected photos)
# Output: data/result_version_2/protected/photoguard/*.png   (protected photos, same names)
#
# Running it again is safe: photos that are already protected are skipped.
#
# Usage:
#     python protect_photoguard.py
#
# Needs at least one CUDA GPU.

import multiprocessing
import os
from datetime import datetime

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image


# Folders

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(SCRIPT_FOLDER, "..", "data")
CLEAN_FOLDER = os.path.join(DATA_FOLDER, "dreambooth", "images_512")

# Each version of the experiment keeps its results in its own folder.
#   result_version_1: Editor B = SDXL InstructPix2Pix (could not edit clean photos)
#   result_version_2: Editor B = FLUX.1 Kontext
RESULT_FOLDER = os.path.join(DATA_FOLDER, "result_version_2")
PROTECTED_FOLDER = os.path.join(RESULT_FOLDER, "protected", "photoguard")


# PhotoGuard settings (strength and number of steps are the same as in PREEMPT)

# Largest change allowed per pixel value (on a 0-255 scale this is 12).
EPSILON = 12 / 255

# How many times the noise is adjusted.
NUM_STEPS = 100

# How much the noise changes in each adjustment (1 on a 0-255 scale).
STEP_SIZE = 1 / 255

# Seed for PhotoGuard's random starting noise, so running again gives the
# exact same protected photos.
PHOTOGUARD_SEED = 0

# The model PhotoGuard studies to build its noise ("the model it is designed to fool").
# Only the image compressor (the VAE) of this model is used.
PHOTOGUARD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


# Number of worker processes.
# Set by "export NUM_WORKERS=..." in the job script. If that is missing, it uses
# the number of CPUs asked for (--cpus-per-task). Outside Slurm it is 1.
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", "1")))


# Converting between pictures and tensors

def load_image_as_tensor(path, device):
    # Read a PNG and return a tensor of shape 1 x 3 x 512 x 512 with values from 0 to 1.
    image = Image.open(path).convert("RGB")
    pixels = np.array(image).astype(np.float32) / 255.0      # 512 x 512 x 3
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)        # 3 x 512 x 512
    tensor = tensor.unsqueeze(0)                              # 1 x 3 x 512 x 512
    return tensor.to(device)


def save_tensor_as_image(tensor, path):
    # Save a 1 x 3 x 512 x 512 tensor (values 0 to 1) as a PNG.
    pixels = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # 512 x 512 x 3
    pixels = np.round(pixels * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)


# PhotoGuard

def load_compressor(device):
    # Load the image compressor (VAE) of Stable Diffusion 1.5.
    vae = AutoencoderKL.from_pretrained(PHOTOGUARD_MODEL, subfolder="vae")
    vae = vae.to(device)
    vae.eval()
    vae.requires_grad_(False)     # we change the image, never the model
    return vae


def protect(image, vae):
    # Return a protected copy of the image (same shape, values 0 to 1).

    # Start from a small random change, as the original PhotoGuard code does.
    # The fixed seed makes this random change the same every time we run.
    generator = torch.Generator(device=image.device).manual_seed(PHOTOGUARD_SEED)
    random_start = torch.empty_like(image).uniform_(-EPSILON, EPSILON, generator=generator)
    protected = (image + random_start).clamp(0, 1)

    for step in range(NUM_STEPS):
        protected.requires_grad_(True)

        # 1. Compress the photo. The VAE expects values from -1 to 1, not 0 to 1.
        summary = vae.encode(protected * 2 - 1).latent_dist.mean

        # How far the summary is from all zeros. PhotoGuard wants this small.
        distance_from_zero = summary.norm()

        # 2. Which direction to nudge each pixel to make the distance smaller.
        distance_from_zero.backward()
        direction = protected.grad.sign()

        with torch.no_grad():
            # 3. Nudge every pixel a tiny bit.
            protected = protected - STEP_SIZE * direction

            # 4. Keep every pixel within EPSILON of the original, and within 0..1.
            change = (protected - image).clamp(-EPSILON, EPSILON)
            protected = (image + change).clamp(0, 1)

    return protected.detach()


# One worker: protects its own group of photos

def protect_group(worker_number, file_names):
    # Spread the workers over the GPUs: worker 0 -> GPU 0, worker 1 -> GPU 1, ...
    gpu_number = worker_number % torch.cuda.device_count()
    device = f"cuda:{gpu_number}"

    # Each worker uses one CPU thread, so the workers don't fight over the CPUs.
    torch.set_num_threads(1)

    print(f"[worker {worker_number}] {len(file_names)} photos on {device}")
    vae = load_compressor(device)

    for number, file_name in enumerate(file_names, start=1):
        image = load_image_as_tensor(os.path.join(CLEAN_FOLDER, file_name), device)
        protected = protect(image, vae)
        save_tensor_as_image(protected, os.path.join(PROTECTED_FOLDER, file_name))

        # Sanity check: the biggest change should be about 12 (on a 0-255 scale).
        biggest_change = (protected - image).abs().max().item() * 255
        print(f"[worker {worker_number}] {number}/{len(file_names)} {file_name}"
              f"  biggest pixel change: {biggest_change:.1f}")


# Main

def main():
    print(f"=== Step 1 of 2: protect the photos with PhotoGuard  (started {datetime.now():%Y-%m-%d %H:%M}) ===")
    os.makedirs(PROTECTED_FOLDER, exist_ok=True)

    all_photos = sorted(f for f in os.listdir(CLEAN_FOLDER) if f.endswith(".png"))
    if not all_photos:
        print(f"No photos found in {CLEAN_FOLDER}. Run download_dreambooth.py first.")
        return

    # Only the photos that are not protected yet.
    to_do = [f for f in all_photos if not os.path.exists(os.path.join(PROTECTED_FOLDER, f))]
    if not to_do:
        print("All photos are already protected.")
        return

    print(f"{len(to_do)} photos to protect, {NUM_WORKERS} workers, "
          f"{torch.cuda.device_count()} GPU(s).")

    # Deal the photos out like cards: worker 0 gets photos 0, 10, 20, ...
    groups = [to_do[i::NUM_WORKERS] for i in range(NUM_WORKERS)]
    groups = [group for group in groups if group]       # skip empty groups

    # "spawn" starts each worker fresh - required when workers use the GPU.
    spawn = multiprocessing.get_context("spawn")
    workers = []
    for worker_number, group in enumerate(groups):
        worker = spawn.Process(target=protect_group, args=(worker_number, group))
        worker.start()
        workers.append(worker)

    # Wait for every worker to finish, and report any that crashed.
    failed = 0
    for worker_number, worker in enumerate(workers):
        worker.join()
        if worker.exitcode != 0:
            print(f"Worker {worker_number} failed (exit code {worker.exitcode}).")
            failed += 1

    if failed:
        raise SystemExit(f"{failed} worker(s) failed. Run again to redo the missing photos.")

    print(f"Done. Protected photos are in {PROTECTED_FOLDER}")
    print(f"=== Step 1 finished {datetime.now():%Y-%m-%d %H:%M} ===")


if __name__ == "__main__":
    main()
