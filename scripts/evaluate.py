# Score the results of run_edit_chains.py with numbers, for all photos.
#
# Three questions, three scores:
#
#   1. Is the protection invisible?                       -> LPIPS and PSNR
#      Compares each clean photo with its protected copy.
#      LPIPS: 0 = looks identical, higher = more visible difference.
#      PSNR:  higher = closer (above ~35 dB is hard to see).
#
#   2. Is the edited result still the same object?         -> DINOv2 similarity (as in PREEMPT)
#      Compares each edited result with the ORIGINAL clean photo.
#      1 = same object, lower = the result drifted away from the original.
#      If the protection works, the protected photos drift MORE than the clean ones.
#      The gap (clean minus protected) is PREEMPT's Δ: bigger gap = stronger protection.
#
#   3. Did the edit follow the instruction?                -> CLIP score (as in the ICIP 2025 paper)
#      Compares each edited result with a short description of what it should
#      look like, e.g. "a photo of a teapot on a sunny beach".
#      Higher = matches the description better.
#
# Why both 2 and 3: a protected edit can follow the instruction well (a nice beach)
# while losing the object (no teapot). Only looking at 3 would call that a success.
#
# Input:  data/result_version_2/protected/photoguard/   and   data/result_version_2/edits/
# Output: data/result_version_2/evaluation/
#   protection_quality.csv   one row per photo: LPIPS and PSNR
#   per_image.csv            one row per (version, chain, photo): DINOv2 similarity and CLIP score
#   summary.csv              one row per chain: averages for clean vs protected, and the gap
#   summary.png              bar chart of summary.csv
#
# This is quick (a few minutes on one GPU), so it runs in a single process.
#
# Usage:
#     python evaluate.py

import json
import os

import lpips
import matplotlib
matplotlib.use("Agg")      # save charts to a file, no screen needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor


# Folders

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(SCRIPT_FOLDER, "..", "data")
CLEAN_FOLDER = os.path.join(DATA_FOLDER, "dreambooth", "images_512")

# Which version of the experiment to score.
RESULT_FOLDER = os.path.join(DATA_FOLDER, "result_version_2")
PROTECTED_FOLDER = os.path.join(RESULT_FOLDER, "protected", "photoguard")
EDITS_FOLDER = os.path.join(RESULT_FOLDER, "edits")
EVALUATION_FOLDER = os.path.join(RESULT_FOLDER, "evaluation")

SEED = 42


# The chains, in the order they appear in the tables and the chart.
# (folder name, short label, which edit step it is)
# "no_edit" is the input photo itself, before any editing, as a starting point.
CHAINS = [
    ("no_edit", "no edit", 0),
    ("step1_A", "A", 1),
    ("step1_B", "B", 1),
    ("step2_A_then_A", "A→A", 2),
    ("step2_A_then_B", "A→B", 2),
    ("step2_B_then_A", "B→A", 2),
    ("step2_B_then_B", "B→B", 2),
]

VERSIONS = ["clean", "photoguard"]


# What each edit step should produce, used for the CLIP score.
# {object} is filled in from the photo name, e.g. "teapot_00" -> "teapot".
STEP_1_DESCRIPTION = "a photo of a {object} on a sunny beach"
STEP_2_DESCRIPTION = "a photo of a bright blue {object}"


# Scoring models
DINO_MODEL = "facebook/dinov2-base"
CLIP_MODEL = "openai/clip-vit-base-patch32"   # the model the standard CLIP score uses

# The main version of the CLIP model only has an old-style weights file
# (pytorch_model.bin), which transformers refuses to load with our torch 2.5.
# "refs/pr/66" is the same weights saved by Hugging Face in the safe format
# (model.safetensors), which loads fine.
CLIP_REVISION = "refs/pr/66"


# Colors for the chart (clean = blue, protected = orange).
CLEAN_COLOR = "#2a78d6"
PROTECTED_COLOR = "#eb6834"


# Small helpers

def list_image_ids():
    # Names of all photos without '.png', e.g. ['backpack_00', 'backpack_01', ...].
    file_names = sorted(f for f in os.listdir(CLEAN_FOLDER) if f.endswith(".png"))
    return [f[:-len(".png")] for f in file_names]


def object_name(image_id):
    # 'backpack_dog_00' -> 'backpack dog',  'dog7_00' -> 'dog'
    name = image_id.rsplit("_", 1)[0]          # drop the photo number
    name = name.rstrip("0123456789")           # drop the object number (dog7 -> dog)
    return name.replace("_", " ")


def input_photo_path(version, image_id):
    # The photo that went into the editors (clean or protected).
    if version == "clean":
        return os.path.join(CLEAN_FOLDER, image_id + ".png")
    return os.path.join(PROTECTED_FOLDER, image_id + ".png")


def result_photo_path(version, chain_folder, image_id):
    # Where an edited result is, or the input photo for "no_edit".
    if chain_folder == "no_edit":
        return input_photo_path(version, image_id)
    return os.path.join(EDITS_FOLDER, version, f"seed{SEED}", chain_folder, image_id + ".png")


def load_photo(path):
    return Image.open(path).convert("RGB")


def description_for(step, image_id):
    # The text the CLIP score compares with. No text for the unedited photo.
    if step == 1:
        return STEP_1_DESCRIPTION.format(object=object_name(image_id))
    if step == 2:
        return STEP_2_DESCRIPTION.format(object=object_name(image_id))
    return None


# Score 1: is the protection invisible?

def photo_to_lpips_tensor(photo, device):
    # LPIPS wants a 1 x 3 x H x W tensor with values from -1 to 1.
    pixels = np.array(photo).astype(np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    return (tensor * 2 - 1).to(device)


def psnr(photo_a, photo_b):
    # Peak signal-to-noise ratio in dB. Higher = the two photos are closer.
    a = np.array(photo_a).astype(np.float64)
    b = np.array(photo_b).astype(np.float64)
    mean_squared_error = np.mean((a - b) ** 2)
    return 10 * np.log10(255.0 ** 2 / mean_squared_error)


def score_protection_quality(image_ids, device):
    print("Score 1: how visible is the protection (LPIPS, PSNR)...")
    lpips_model = lpips.LPIPS(net="alex").to(device)

    rows = []
    for image_id in image_ids:
        clean = load_photo(input_photo_path("clean", image_id))
        protected = load_photo(input_photo_path("photoguard", image_id))

        with torch.no_grad():
            distance = lpips_model(photo_to_lpips_tensor(clean, device),
                                   photo_to_lpips_tensor(protected, device)).item()

        rows.append({"image_id": image_id, "lpips": distance, "psnr": psnr(clean, protected)})

    table = pd.DataFrame(rows)
    print(f"  average LPIPS {table['lpips'].mean():.3f}, average PSNR {table['psnr'].mean():.1f} dB")
    return table


# Scores 2 and 3: same object? followed the instruction?

class Scorer:
    # Holds the DINOv2 and CLIP models, so they are loaded only once.

    def __init__(self, device):
        self.device = device
        self.dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
        self.dino = AutoModel.from_pretrained(DINO_MODEL).to(device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION,
                                              use_safetensors=True).to(device).eval()

    @torch.no_grad()
    def dino_summary(self, photo):
        # DINOv2's summary of the photo, scaled to length 1.
        inputs = self.dino_processor(images=photo, return_tensors="pt").to(self.device)
        summary = self.dino(**inputs).pooler_output[0]
        return summary / summary.norm()

    def same_object_score(self, original_summary, photo):
        # Similarity between the original photo and another photo (1 = same).
        return float(original_summary @ self.dino_summary(photo))

    @torch.no_grad()
    def clip_score(self, photo, text):
        # Standard CLIP score: 2.5 x similarity between the photo and the text
        # (negative similarities count as 0).
        image_inputs = self.clip_processor(images=photo, return_tensors="pt").to(self.device)
        text_inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        image_summary = self.clip.get_image_features(**image_inputs)[0]
        text_summary = self.clip.get_text_features(**text_inputs)[0]
        similarity = (image_summary / image_summary.norm()) @ (text_summary / text_summary.norm())
        return 2.5 * max(float(similarity), 0.0)


def score_edits(image_ids, device):
    print("Scores 2 and 3: same object (DINOv2) and followed the instruction (CLIP)...")
    scorer = Scorer(device)

    rows = []
    for number, image_id in enumerate(image_ids, start=1):
        # Everything is compared with the ORIGINAL clean photo.
        original_summary = scorer.dino_summary(load_photo(input_photo_path("clean", image_id)))

        for version in VERSIONS:
            for chain_folder, label, step in CHAINS:
                photo = load_photo(result_photo_path(version, chain_folder, image_id))
                text = description_for(step, image_id)

                rows.append({
                    "version": version,
                    "chain": label,
                    "step": step,
                    "image_id": image_id,
                    "same_object_dinov2": scorer.same_object_score(original_summary, photo),
                    "clip_score": scorer.clip_score(photo, text) if text else float("nan"),
                    "description": text,
                })

        print(f"  {number}/{len(image_ids)} {image_id}")

    return pd.DataFrame(rows)


# Summary: one row per chain, clean next to protected

def make_summary(per_image):
    rows = []
    for _, label, step in CHAINS:
        clean = per_image[(per_image["chain"] == label) & (per_image["version"] == "clean")]
        protected = per_image[(per_image["chain"] == label) & (per_image["version"] == "photoguard")]

        # Line up the two versions photo by photo, to compare each photo with itself.
        clean = clean.set_index("image_id")
        protected = protected.set_index("image_id").loc[clean.index]
        gap_per_photo = clean["same_object_dinov2"] - protected["same_object_dinov2"]

        rows.append({
            "chain": label,
            "step": step,
            "same_object_clean": clean["same_object_dinov2"].mean(),
            "same_object_protected": protected["same_object_dinov2"].mean(),
            # PREEMPT's Δ: how much more the protected photos drifted. Bigger = protection worked.
            "gap": gap_per_photo.mean(),
            # Share of photos where the protected result drifted more than the clean one.
            "photos_where_protected_drifted_more_pct": 100 * (gap_per_photo > 0).mean(),
            "clip_clean": clean["clip_score"].mean(),
            "clip_protected": protected["clip_score"].mean(),
            "num_photos": len(clean),
        })

    return pd.DataFrame(rows)


# Chart

def draw_chart(summary, settings, path):
    # Two panels side by side: same object (left), followed the instruction (right).
    # Each chain has two bars: clean (blue) and protected (orange).
    edited = summary[summary["step"] > 0]       # CLIP has no value for "no edit"

    panels = [
        (summary, "same_object_clean", "same_object_protected",
         "Same object as the original? (DINOv2, higher = same)"),
        (edited, "clip_clean", "clip_protected",
         "Followed the instruction? (CLIP score, higher = better)"),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    bar_width = 0.38

    for axis, (table, clean_column, protected_column, title) in zip(axes, panels):
        positions = np.arange(len(table))
        axis.bar(positions - bar_width / 2 - 0.01, table[clean_column], bar_width,
                 color=CLEAN_COLOR, label="clean photo")
        axis.bar(positions + bar_width / 2 + 0.01, table[protected_column], bar_width,
                 color=PROTECTED_COLOR, label="protected photo (PhotoGuard)")

        axis.set_xticks(positions)
        axis.set_xticklabels(table["chain"])
        axis.set_title(title, fontsize=11)
        axis.set_ylim(0, 1.05 if clean_column.startswith("same") else None)
        axis.grid(axis="y", color="#dddddd", linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ["top", "right"]:
            axis.spines[side].set_visible(False)

    # Legend below the left panel, so it never covers a bar.
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    figure.suptitle(f"Editor A = {settings['editor_A']},  Editor B = {settings['editor_B']}  "
                    f"({int(summary['num_photos'].iloc[0])} photos, seed {SEED})", fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# Main

def main():
    os.makedirs(EVALUATION_FOLDER, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(EDITS_FOLDER, "settings.json")) as f:
        settings = json.load(f)

    image_ids = list_image_ids()
    print(f"Scoring {len(image_ids)} photos in {RESULT_FOLDER} on {device}")

    quality = score_protection_quality(image_ids, device)
    quality.to_csv(os.path.join(EVALUATION_FOLDER, "protection_quality.csv"), index=False)

    per_image = score_edits(image_ids, device)
    per_image.to_csv(os.path.join(EVALUATION_FOLDER, "per_image.csv"), index=False)

    summary = make_summary(per_image)
    summary.to_csv(os.path.join(EVALUATION_FOLDER, "summary.csv"), index=False)

    draw_chart(summary, settings, os.path.join(EVALUATION_FOLDER, "summary.png"))

    print()
    print(summary.drop(columns=["step"]).round(3).to_string(index=False))
    print()
    print(f"Done. Results are in {EVALUATION_FOLDER}")


if __name__ == "__main__":
    main()
