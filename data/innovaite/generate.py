import os
import json
import random
import uuid
from pathlib import Path

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

# Prompts template (modify or add more)
PROMPTS_TEMPLATE = {
    "p0": "",
    "p1": "polyp",
    "p2": "circle polyp",
    "p3": "pink circle polyp",
    "p4": "small pink circle polyp",
    "p5": "one small pink circle polyp",
    "p6": "one small pink circle polyp, located in right of the image",
    "p7": [
        "polyp which is a projecting growth of tissue",
        "polyp which is often a bumpy flesh in rectum",
        "polyp which is a small lump in the lining of colon",
        "polyp which is a tissue growth that often resemble mushroom like stalks",
        "polyp which is an abnormal growth of tissues projecting from a mucous membrane"
    ],
    "p8": [
        "one small pink circle polyp which is a projecting growth of tissue",
        "one small pink circle polyp which is often a bumpy flesh in rectum",
        "one small pink circle polyp which is a small lump in the lining of colon",
        "one small pink circle polyp which is a tissue growth that often resemble mushroom like stalks",
        "one small pink circle polyp which is an abnormal growth of tissues projecting from a mucous membrane"
    ],
    "p9": [
        "one small pink circle polyp which is a projecting growth of tissue located in right of the image",
        "one small pink circle polyp which is often a bumpy flesh in rectum located in right of the image",
        "one small pink circle polyp which is a small lump in the lining of colon located in right of the image",
        "one small pink circle polyp which is a tissue growth that often resemble mushroom like stalks located in right of the image",
        "one small pink circle polyp which is an abnormal growth of tissues projecting from a mucous membrane located in right of the image"
    ]
}

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Scanning dataset...")

    os.makedirs(ANNS_DIR, exist_ok=True)

    # List all image files
    image_files = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".tif", ".png", ".jpg", ".jpeg"))
    ])

    # Match images with masks
    matched = []
    for img in image_files:
        mask_path = os.path.join(MASKS_DIR, img)
        if os.path.exists(mask_path):
            matched.append(img)
        else:
            print(f"Warning: mask missing for {img}")

    print(f"Found {len(matched)} matched image/mask pairs.")

    # Shuffle before splitting
    random.shuffle(matched)

    n = len(matched)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)
    n_test = n - n_train - n_val

    train_files = matched[:n_train]
    val_files = matched[n_train:n_train + n_val]
    test_files = matched[n_train + n_val:]

    print(f"Train: {len(train_files)}")
    print(f"Val:   {len(val_files)}")
    print(f"Test:  {len(test_files)}")

    def build_entries(files):
        entries = []
        for fname in files:
            entry = {
                "bbox": [0, 0, 0, 0],  # todo
                "cat": 0,
                "segment_id": str(uuid.uuid4().hex[:24]),  # random unique id
                "img_name": fname,
                "mask_name": fname,
                "sentences": [{"idx": 0, "sent_id": 0, "sent": ""}],
                "prompts": PROMPTS_TEMPLATE,
                "sentences_num": 1
            }
            entries.append(entry)
        return entries

    with open(f"{ANNS_DIR}/train.json", "w") as f:
        json.dump(build_entries(train_files), f, indent=2)

    with open(f"{ANNS_DIR}/val.json", "w") as f:
        json.dump(build_entries(val_files), f, indent=2)

    with open(f"{ANNS_DIR}/test.json", "w") as f:
        json.dump(build_entries(test_files), f, indent=2)

    print("Finished. Annotation files written to:")
    print(f"{ANNS_DIR}/train.json")
    print(f"{ANNS_DIR}/val.json")
    print(f"{ANNS_DIR}/test.json")


if __name__ == "__main__":
    main()

