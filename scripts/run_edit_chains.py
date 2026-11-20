# Run the two-step editing chains on clean and protected photos.
#
# For every photo, in both versions (clean and protected):
#
#   Step 1:  edit with A  or  edit with B         (EDIT_1_INSTRUCTION)
#   Step 2:  take each step-1 result and edit it again with A or B
#            (EDIT_2_INSTRUCTION)
#
# That gives the four sequences:  A then A,  A then B,  B then A,  B then B.
#
# Clean and protected photos get exactly the same editors, instructions and
# seeds, so any difference between them comes from the protection.
#
# Speed-up with several worker processes:
#   The photos are split into NUM_WORKERS groups. Each worker is a separate
#   process with its own copy of the editors, and runs the whole chain
#   (step 1 and step 2) for its own photos. Workers are spread evenly over
#   the GPUs (with 10 workers and 2 GPUs, 5 workers share each GPU).
#   Inside each worker, one editor is loaded at a time to save GPU memory.
#
# Output folders (inside data/result_version_2/edits/):
#   clean/seed42/step1_A/backpack_00.png
#   clean/seed42/step1_B/...
#   clean/seed42/step2_A_then_A/...
#   clean/seed42/step2_A_then_B/...
#   clean/seed42/step2_B_then_A/...
#   clean/seed42/step2_B_then_B/...
#   photoguard/seed42/...   (same structure)


import gc
import json
import multiprocessing
import os
from datetime import datetime

import torch
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
EDITS_FOLDER = os.path.join(RESULT_FOLDER, "edits")

IMAGE_SIZE = 512

# The two versions of every photo that get edited.
VERSIONS = {
    "clean": CLEAN_FOLDER,
    "photoguard": PROTECTED_FOLDER,
}


# Editors

# Model A = "seen" editor. InstructPix2Pix is built from Stable Diffusion 1.5,
#           so it uses the same image compressor PhotoGuard was built against.
# Model B = "unseen" editor:
#   "flux_kontext"          - completely different family and compressor. Very large:
#                             use only 1 worker per 40 GB GPU, and accept its license
#                             on Hugging Face first.
#   "sdxl_instruct_pix2pix" - used in result_version_1. Do not use: it could not edit
#                             even clean photos (a teapot became flamingos).
EDITOR_A = "instruct_pix2pix"
EDITOR_B = "flux_kontext"

# Where each editor is downloaded from (Hugging Face model names).
EDITOR_MODELS = {
    "instruct_pix2pix": "timbrooks/instruct-pix2pix",
    "sdxl_instruct_pix2pix": "diffusers/sdxl-instructpix2pix-768",
    "flux_kontext": "black-forest-labs/FLUX.1-Kontext-dev",
}

EDITORS = {
    "A": EDITOR_A,
    "B": EDITOR_B,
}


# The two edits in every chain
# All three editors take plain instructions, so the same text works for all.
# These are generic so they fit every DreamBooth object.

EDIT_1_INSTRUCTION = "change the background to a sunny beach"
EDIT_2_INSTRUCTION = "change the color of the main object to bright blue"


# Random seeds
# Every edit is repeated once per seed. Start with one; add more (e.g. [42, 123, 7])
# once everything works, so results are not down to luck.

SEEDS = [42]


# Number of worker processes.
# Set by "export NUM_WORKERS=..." in the job script. If that is missing, it uses
# the number of CPUs asked for (--cpus-per-task). Outside Slurm it is 1.
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", "1")))


# Loading an editor and editing one image

def load_editor(editor_name, device):
    # Download (first time only) and load an editor onto the given GPU.
    model_name = EDITOR_MODELS[editor_name]

    if editor_name == "instruct_pix2pix":
        from diffusers import StableDiffusionInstructPix2PixPipeline
        # safety_checker=None: the safety filter sometimes wrongly turns normal
        # results into black images, which would spoil the scores.
        editor = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_name, torch_dtype=torch.float16, safety_checker=None
        )
        editor = editor.to(device)

    elif editor_name == "sdxl_instruct_pix2pix":
        from diffusers import StableDiffusionXLInstructPix2PixPipeline
        editor = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        editor = editor.to(device)

    elif editor_name == "flux_kontext":
        from diffusers import FluxKontextPipeline
        editor = FluxKontextPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        # FLUX is very large. This keeps only the part currently in use on the GPU.
        gpu_number = int(device.split(":")[1])
        editor.enable_model_cpu_offload(gpu_id=gpu_number)

    else:
        raise ValueError(f"Unknown editor name: {editor_name}")

    editor.set_progress_bar_config(disable=True)
    return editor


def edit_image(editor_name, editor, image, instruction, seed):
    # Edit one PIL image with one instruction. Returns a 512 x 512 PIL image.
    # The seed fixes the editor's randomness, so clean and protected photos
    # get exactly the same treatment.
    generator = torch.Generator("cpu").manual_seed(seed)

    if editor_name == "instruct_pix2pix":
        result = editor(prompt=instruction, image=image, generator=generator).images[0]

    elif editor_name == "sdxl_instruct_pix2pix":
        result = editor(
            prompt=instruction,
            image=image,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            generator=generator,
        ).images[0]

    elif editor_name == "flux_kontext":
        # Note: FLUX Kontext enlarges the photo to about 1024 x 1024 internally.
        # Enlarging can smooth away some of the protective noise - keep this in
        # mind when reading FLUX results.
        result = editor(prompt=instruction, image=image, generator=generator).images[0]

    else:
        raise ValueError(f"Unknown editor name: {editor_name}")

    # Make every result the same size, so the next edit and the scores
    # always work on 512 x 512 images.
    if result.size != (IMAGE_SIZE, IMAGE_SIZE):
        result = result.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    return result


def free_gpu_memory():
    # Call after deleting an editor, so the next one has room on the GPU.
    gc.collect()
    torch.cuda.empty_cache()


# Small helpers

def list_image_ids():
    # Names of all photos without '.png', e.g. ['backpack_00', 'backpack_01', ...].
    file_names = sorted(f for f in os.listdir(CLEAN_FOLDER) if f.endswith(".png"))
    return [f[:-len(".png")] for f in file_names]


def list_jobs(image_ids):
    # Every (version, seed, image_id) combination, e.g. ('clean', 42, 'backpack_00').
    jobs = []
    for version in VERSIONS:
        for seed in SEEDS:
            for image_id in image_ids:
                jobs.append((version, seed, image_id))
    return jobs


def result_path(version, seed, folder_name, image_id):
    # Where a result is saved, e.g. data/edits/clean/seed42/step1_A/backpack_00.png
    folder = os.path.join(EDITS_FOLDER, version, f"seed{seed}", folder_name)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, image_id + ".png")


def save_settings():
    # Keep a record of exactly what was run, next to the results.
    os.makedirs(EDITS_FOLDER, exist_ok=True)
    settings = {
        "editor_A": EDITOR_A,
        "editor_B": EDITOR_B,
        "editor_models": EDITOR_MODELS,
        "edit_1_instruction": EDIT_1_INSTRUCTION,
        "edit_2_instruction": EDIT_2_INSTRUCTION,
        "seeds": SEEDS,
        "num_workers": NUM_WORKERS,
    }
    with open(os.path.join(EDITS_FOLDER, "settings.json"), "w") as f:
        json.dump(settings, f, indent=2)


# Step 1: edit the original photos

def run_step_1(worker_number, device, jobs):
    for letter, editor_name in EDITORS.items():
        print(f"[worker {worker_number}] step 1 with editor {letter} ({editor_name})")
        editor = load_editor(editor_name, device)

        for version, seed, image_id in jobs:
            save_to = result_path(version, seed, f"step1_{letter}", image_id)
            if os.path.exists(save_to):
                continue

            image = Image.open(os.path.join(VERSIONS[version], image_id + ".png")).convert("RGB")
            result = edit_image(editor_name, editor, image, EDIT_1_INSTRUCTION, seed)
            result.save(save_to)
            print(f"[worker {worker_number}] step1_{letter} {version} seed{seed} {image_id}")

        del editor
        free_gpu_memory()


# Step 2: edit the step-1 results again

def run_step_2(worker_number, device, jobs):
    for second_letter, editor_name in EDITORS.items():
        print(f"[worker {worker_number}] step 2 with editor {second_letter} ({editor_name})")
        editor = load_editor(editor_name, device)

        for first_letter in EDITORS:
            sequence = f"{first_letter}_then_{second_letter}"

            for version, seed, image_id in jobs:
                save_to = result_path(version, seed, f"step2_{sequence}", image_id)
                if os.path.exists(save_to):
                    continue

                step_1_file = result_path(version, seed, f"step1_{first_letter}", image_id)
                image = Image.open(step_1_file).convert("RGB")
                result = edit_image(editor_name, editor, image, EDIT_2_INSTRUCTION, seed)
                result.save(save_to)
                print(f"[worker {worker_number}] step2_{sequence} {version} seed{seed} {image_id}")

        del editor
        free_gpu_memory()


# One worker: runs the whole chain for its own group of photos

def run_worker(worker_number, image_ids):
    # Spread the workers over the GPUs: worker 0 -> GPU 0, worker 1 -> GPU 1, ...
    gpu_number = worker_number % torch.cuda.device_count()
    device = f"cuda:{gpu_number}"

    # Each worker uses one CPU thread, so the workers don't fight over the CPUs.
    torch.set_num_threads(1)

    print(f"[worker {worker_number}] {len(image_ids)} photos on {device}")
    jobs = list_jobs(image_ids)
    run_step_1(worker_number, device, jobs)
    run_step_2(worker_number, device, jobs)
    print(f"[worker {worker_number}] finished")


# Main

def main():
    print(f"=== Step 2 of 2: run the edit chains  (started {datetime.now():%Y-%m-%d %H:%M}) ===")
    image_ids = list_image_ids()
    if not image_ids:
        print(f"No photos found in {CLEAN_FOLDER}. Run download_dreambooth.py first.")
        return

    # Every clean photo must have a protected copy before we start.
    for image_id in image_ids:
        if not os.path.exists(os.path.join(PROTECTED_FOLDER, image_id + ".png")):
            print(f"Missing protected photo for {image_id}. Run protect_photoguard.py first.")
            return

    print(f"{len(image_ids)} photos x {len(VERSIONS)} versions x {len(SEEDS)} seeds, "
          f"{NUM_WORKERS} workers, {torch.cuda.device_count()} GPU(s).")
    print(f"Editor A = {EDITOR_A},  Editor B = {EDITOR_B}")
    save_settings()

    # Deal the photos out like cards: worker 0 gets photos 0, 10, 20, ...
    # Each worker keeps its photos for both steps, because step 2 needs that
    # photo's step-1 result.
    groups = [image_ids[i::NUM_WORKERS] for i in range(NUM_WORKERS)]
    groups = [group for group in groups if group]       # skip empty groups

    # "spawn" starts each worker fresh - required when workers use the GPU.
    spawn = multiprocessing.get_context("spawn")
    workers = []
    for worker_number, group in enumerate(groups):
        worker = spawn.Process(target=run_worker, args=(worker_number, group))
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
        raise SystemExit(f"{failed} worker(s) failed. Run again to redo the missing results.")

    print(f"\nDone. All results are in {EDITS_FOLDER}")
    print(f"=== Step 2 finished {datetime.now():%Y-%m-%d %H:%M} ===")


if __name__ == "__main__":
    main()
