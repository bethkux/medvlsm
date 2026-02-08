import os
import json
import random
import numpy as np
from PIL import Image
import uuid

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_ROOT = "/mnt/proj1/eu-25-40/innovaite/VLSM-Ensemble_Execution/medvlsm/data/innovaite"
IMAGES_DIR = f"{DATA_ROOT}/images"
MASKS_DIR = f"{DATA_ROOT}/masks"
ANNS_DIR = f"{DATA_ROOT}/anns"

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# -----------------------------
# UTILS
# -----------------------------
def compute_bbox(mask_path):
    mask = np.array(Image.open(mask_path))
    ys, xs = np.where(mask > 0)

    # fallback if mask is empty
    if len(xs) == 0 or len(ys) == 0:
        h, w = mask.shape[:2]
        return [0, 0, w - 1, h - 1]

    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    ]

def build_entry(fname):
    mask_path = os.path.join(MASKS_DIR, fname)

    return {
        "bbox": compute_bbox(mask_path),
        "cat": 0,
        "segment_id": uuid.uuid4().hex[:24],
        "img_name": fname,
        "mask_name": fname,
        "sentences": [
            {"idx": 0, "sent_id": 0, "sent": ""}
        ],
        "prompts": {
            "p0": "",
            "p1": "object",
            "p2": "segmentation object",
            "p3": "target object in the image",
            "p4": "small target object",
            "p5": "one small target object",
            "p6": "one small target object in the image",
            "p7": ["object to segment"],
            "p8": ["one object to segment"],
            "p9": ["one object to segment in the image"]
        },
        "sentences_num": 1
    }

# -----------------------------
# MAIN
# -----------------------------
def main():
    os.makedirs(ANNS_DIR, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".tif", ".png", ".jpg", ".jpeg"))
    ])

    matched = [
        f for f in image_files
        if os.path.exists(os.path.join(MASKS_DIR, f))
    ]

    random.shuffle(matched)

    n = len(matched)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    splits = {
        "train": matched[:n_train],
        "val": matched[n_train:n_train + n_val],
        "test": matched[n_train + n_val:]
    }

    for split, files in splits.items():
        entries = [build_entry(f) for f in files]
        with open(f"{ANNS_DIR}/{split}.json", "w") as f:
            json.dump(entries, f, indent=2)

        print(f"Wrote {len(entries)} entries to {split}.json")

if __name__ == "__main__":
    main()

