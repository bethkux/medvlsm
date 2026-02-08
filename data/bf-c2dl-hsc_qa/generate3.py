import os
import json
import random
import uuid

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_ROOT = "/mnt/proj1/eu-25-40/innovaite/VLSM-Ensemble_Execution/medvlsm/data/bf-c2dl-hsc_qa"
IMAGES_DIR = f"{DATA_ROOT}/images"
MASKS_DIR = f"{DATA_ROOT}/masks"
ANNS_DIR = f"{DATA_ROOT}/anns"

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Prompt definitions for BF-C2DL-HSC (mouse stem cells)
PROMPTS_TEMPLATE = {
    "p0": "",
    "p1": "mouse stem cell",
    "p2": "hematopoietic stem cell",
    "p3": "stem cell in bright-field microscopy",
    "p4": "single mouse stem cell",
    "p5": "one stem cell",
    "p6": "one mouse hematopoietic stem cell",
    "p7": [
        "mouse stem cell",
        "hematopoietic stem cell in bright-field microscopy",
        "stem cell from BF-C2DL-HSC dataset"
    ],
    "p8": [
        "single mouse stem cell",
        "one hematopoietic stem cell",
        "one stem cell in bright-field microscopy"
    ],
    "p9": [
        "single mouse stem cell in a microscopy image",
        "one hematopoietic stem cell from BF-C2DL-HSC"
    ]
}

# -----------------------------
# MAIN
# -----------------------------
def main():
    os.makedirs(ANNS_DIR, exist_ok=True)

    # List image files
    image_files = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".tif", ".png", ".jpg", ".jpeg"))
    ])

    # Keep only images with matching masks
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
        entries = []
        for fname in files:
            entry = {
                "bbox": [0, 0, 0, 0],  # not used, kept for compatibility
                "cat": 0,
                "segment_id": uuid.uuid4().hex[:24],
                "img_name": fname,
                "mask_name": fname,
                "sentences": [
                    {"idx": 0, "sent_id": 0, "sent": ""}
                ],
                "prompts": PROMPTS_TEMPLATE,
                "sentences_num": 1
            }
            entries.append(entry)

        with open(f"{ANNS_DIR}/{split}.json", "w") as f:
            json.dump(entries, f, indent=2)

        print(f"Wrote {len(entries)} entries to {split}.json")


if __name__ == "__main__":
    main()

