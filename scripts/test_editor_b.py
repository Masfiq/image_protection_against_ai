# Quick test: can Editor B (SDXL InstructPix2Pix) edit a CLEAN photo properly?
#
# In the first run, Editor B ignored the input photo even when it was not
# protected (a backpack became a cup of flowers). This script tries a few
# settings on 4 clean photos and puts the results side by side, so we can see
# which setting keeps the object.
#
# Settings tried:
#   512 size, image guidance 1.5   - what the first run used
#   768 size, image guidance 1.5   - the size this model was trained on
#   768 size, image guidance 2.5   - also tells the model to stay closer to the input photo
#
# Output: data/test_editor_b/sheet.png  (rows = photos, columns = settings)
#
# Usage:
#     python test_editor_b.py

import os

import torch
from diffusers import StableDiffusionXLInstructPix2PixPipeline
from PIL import Image, ImageDraw


SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(SCRIPT_FOLDER, "..", "data")
CLEAN_FOLDER = os.path.join(DATA_FOLDER, "dreambooth", "images_512")
OUTPUT_FOLDER = os.path.join(DATA_FOLDER, "test_editor_b")

MODEL_NAME = "diffusers/sdxl-instructpix2pix-768"
INSTRUCTION = "change the background to a sunny beach"
SEED = 42

PHOTOS = ["backpack_00", "teapot_00", "dog_00", "rc_car_00"]

# (name, image size, image guidance)
SETTINGS = [
    ("512_ig1.5", 512, 1.5),
    ("768_ig1.5", 768, 1.5),
    ("768_ig2.5", 768, 2.5),
]


def edit(editor, photo, size, image_guidance):
    # Resize the photo to the test size, edit it, and return the result.
    photo = photo.resize((size, size), Image.LANCZOS)
    generator = torch.Generator("cpu").manual_seed(SEED)
    result = editor(
        prompt=INSTRUCTION,
        image=photo,
        height=size,
        width=size,
        image_guidance_scale=image_guidance,
        generator=generator,
    ).images[0]
    return result


def make_sheet(results):
    # One row per photo: the original, then one picture per setting.
    tile = 256
    columns = ["original"] + [name for name, _, _ in SETTINGS]
    sheet = Image.new("RGB", (tile * len(columns), 20 + tile * len(PHOTOS)), "white")
    draw = ImageDraw.Draw(sheet)

    for c, name in enumerate(columns):
        draw.text((c * tile + 5, 5), name, fill="black")

    for r, photo_id in enumerate(PHOTOS):
        for c, name in enumerate(columns):
            picture = results[photo_id][name].resize((tile, tile))
            sheet.paste(picture, (c * tile, 20 + r * tile))

    return sheet


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print(f"Loading {MODEL_NAME}...")
    editor = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).to("cuda")
    editor.set_progress_bar_config(disable=True)

    results = {}
    for photo_id in PHOTOS:
        photo = Image.open(os.path.join(CLEAN_FOLDER, photo_id + ".png")).convert("RGB")
        results[photo_id] = {"original": photo}

        for name, size, image_guidance in SETTINGS:
            result = edit(editor, photo, size, image_guidance)
            result.save(os.path.join(OUTPUT_FOLDER, f"{photo_id}_{name}.png"))
            results[photo_id][name] = result
            print(f"  {photo_id}  {name}  done")

    sheet_path = os.path.join(OUTPUT_FOLDER, "sheet.png")
    make_sheet(results).save(sheet_path)
    print(f"Done. Side-by-side picture: {sheet_path}")


if __name__ == "__main__":
    main()
