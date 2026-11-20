
# Download the DreamBooth object photos and prepare them for the experiment.

# What this script does:
#   1. Asks GitHub for the list of photos in the DreamBooth dataset
#      (https://github.com/google/dreambooth, folder "dataset/").
#      The dataset has 158 photos of 30 different objects (about 5 photos per object).
#   2. Picks NUM_IMAGES photos, spread evenly across the 30 objects.
#   3. Downloads each photo (original file kept in data/dreambooth/raw/).
#   4. Crops each photo to a square (center) and resizes it to 512 x 512,
#      the same size PREEMPT uses. Saved as PNG in data/dreambooth/images_512/.
#      PNG is lossless - JPEG would blur away the tiny protective noise later.
#   5. Writes a list of the chosen photos to data/dreambooth/image_list.csv.

# Running it again is safe: photos that already exist are skipped.

# Usage:
#     python download_dreambooth.py

# Needs: Python 3 and Pillow  (pip install pillow)


import csv
import json
import os
import urllib.request

from PIL import Image


# 
# Settings - change these if needed
# 

NUM_IMAGES = 40      # how many photos to use (max 158). PREEMPT used 40.
IMAGE_SIZE = 512      # final width and height in pixels

# Where to save everything (relative to this script's folder)
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "..", "data", "dreambooth")
RAW_FOLDER = os.path.join(OUTPUT_FOLDER, "raw")
RESIZED_FOLDER = os.path.join(OUTPUT_FOLDER, "images_512")
IMAGE_LIST_FILE = os.path.join(OUTPUT_FOLDER, "image_list.csv")

# GitHub addresses for the DreamBooth repository
FILE_LIST_URL = "https://api.github.com/repos/google/dreambooth/git/trees/main?recursive=1"
RAW_FILE_URL = "https://raw.githubusercontent.com/google/dreambooth/main/"


# Step 1: get the list of photos

def get_photo_paths():
    # Return a list like ['dataset/backpack/00.jpg', 'dataset/backpack/01.jpg', ...]
    with urllib.request.urlopen(FILE_LIST_URL) as response:
        repo_files = json.load(response)

    photo_paths = []
    for item in repo_files["tree"]:
        path = item["path"]
        if path.startswith("dataset/") and path.lower().endswith((".jpg", ".jpeg", ".png")):
            photo_paths.append(path)

    photo_paths.sort()
    return photo_paths



# Step 2: pick photos spread evenly across the objects


def group_by_object(photo_paths):
    # Turn the flat list into {object name: [its photo paths]}
    photos_per_object = {}
    for path in photo_paths:
        object_name = path.split("/")[1]          # 'dataset/backpack/00.jpg' -> 'backpack'
        if object_name not in photos_per_object:
            photos_per_object[object_name] = []
        photos_per_object[object_name].append(path)
    return photos_per_object


def pick_photos(photos_per_object, how_many):
    
    # Take the 1st photo of every object, then the 2nd photo of every object,
    # and so on, until we have enough. This keeps all objects equally represented.
    
    chosen = []
    round_number = 0

    while len(chosen) < how_many:
        added_this_round = False

        for object_name in sorted(photos_per_object):
            photos = photos_per_object[object_name]
            if round_number < len(photos) and len(chosen) < how_many:
                chosen.append(photos[round_number])
                added_this_round = True

        if not added_this_round:      # every photo is already used
            break
        round_number += 1

    return chosen


 
# Steps 3 and 4: download, crop to square, resize


def make_image_id(path):
    #'dataset/backpack/00.jpg' -> 'backpack_00'
    object_name = path.split("/")[1]
    file_number = os.path.splitext(path.split("/")[2])[0]
    return object_name + "_" + file_number


def download_photo(path, save_to):
    # Download one photo from GitHub, unless we already have it
    if os.path.exists(save_to):
        return
    urllib.request.urlretrieve(RAW_FILE_URL + path, save_to)


def crop_and_resize(load_from, save_to, size):
    # Cut the largest centered square out of the photo, then resize it to size x size
    if os.path.exists(save_to):
        return

    image = Image.open(load_from).convert("RGB")
    width, height = image.size

    square_side = min(width, height)
    left = (width - square_side) // 2
    top = (height - square_side) // 2
    image = image.crop((left, top, left + square_side, top + square_side))

    image = image.resize((size, size), Image.LANCZOS)
    image.save(save_to)               # .png file name -> saved as lossless PNG



# Main


def main():
    os.makedirs(RAW_FOLDER, exist_ok=True)
    os.makedirs(RESIZED_FOLDER, exist_ok=True)

    print("Getting the list of DreamBooth photos...")
    photo_paths = get_photo_paths()
    photos_per_object = group_by_object(photo_paths)
    print(f"Found {len(photo_paths)} photos of {len(photos_per_object)} objects.")

    chosen_paths = pick_photos(photos_per_object, NUM_IMAGES)
    print(f"Using {len(chosen_paths)} photos.")

    rows_for_csv = []
    for number, path in enumerate(chosen_paths, start=1):
        image_id = make_image_id(path)
        raw_file = os.path.join(RAW_FOLDER, image_id + os.path.splitext(path)[1])
        resized_file = os.path.join(RESIZED_FOLDER, image_id + ".png")

        print(f"  [{number}/{len(chosen_paths)}] {image_id}")
        download_photo(path, raw_file)
        crop_and_resize(raw_file, resized_file, IMAGE_SIZE)

        rows_for_csv.append({
            "image_id": image_id,
            "object": path.split("/")[1],
            "image_file": os.path.relpath(resized_file, OUTPUT_FOLDER),
            "source_url": RAW_FILE_URL + path,
        })

    with open(IMAGE_LIST_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "object", "image_file", "source_url"])
        writer.writeheader()
        writer.writerows(rows_for_csv)

    print(f"Done. Photos are in {RESIZED_FOLDER}")
    print(f"List of photos: {IMAGE_LIST_FILE}")


if __name__ == "__main__":
    main()
