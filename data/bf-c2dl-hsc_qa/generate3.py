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
# Prompts template (modify or add more)
PROMPTS_TEMPLATE = {
    "p0": "",
    "p1": "cell",
    "p2": "stem cell",
    "p3": "mouse stem cell",
    "p4": "oval mouse stem cell",
    "p5": "one oval mouse stem cell",
    "p6": "one oval mouse stem cell, located in center of the image",
    "p7": [
        "cell which is floating inside engineered hydrogel microwell",
        "cell which is round, non-adherent, with a rounded nucleus and low cytoplasm-to-nucleus ratio",
        "cell which is mouse hematopoietic stem cell",
        "cell which is stem cell that can give rise to all blood cell types",
        "cell which can be found in bone marrow"
    ],
    "p8": [
        "one oval mouse cell which is floating inside engineered hydrogel microwell",
        "one oval mouse cell which is round, non-adherent, with a rounded nucleus and low cytoplasm-to-nucleus ratio",
        "one oval mouse cell which is mouse hematopoietic stem cell",
        "one oval mouse cell which is stem cell that can give rise to all blood cell types",
        "one oval mouse cell which can be found in bone marrow"
    ],
    "p9": [
        "one oval mouse cell which is floating inside engineered hydrogel microwell located in center of the image",
        "one oval mouse cell which is round, non-adherent, with a rounded nucleus and low cytoplasm-to-nucleus ratio located in center of the image",
        "one oval mouse cell which is mouse hematopoietic stem cell located in center of the image",
        "one oval mouse cell which is stem cell that can give rise to all blood cell types located in center of the image",
        "one oval mouse cell which can be found in bone marrow located in center of the image"
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

