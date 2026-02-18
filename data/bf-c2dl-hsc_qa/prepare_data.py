import pandas as pd
import os
from PIL import Image

# -----------------------------
# CONFIGURATION
# -----------------------------
PARQUET_PATH = '../qa_crops/BF-C2DL-HSC/mixed_sz64/qa_dataset.parquet'
# DEST_ROOT = "/mnt/proj1/eu-25-40/innovaite/VLSM-Ensemble_Execution/medvlsm/data/innovaite"
DEST_ROOT = "/home/osalamon/OPENsalam/WP3/VLSM-ensemble/VLSM-Ensemble/beth/data/bf-c2dl-hsc_qa"

# Destination directories
IMAGES_DIR = os.path.join(DEST_ROOT, "images")
MASKS_DIR = os.path.join(DEST_ROOT, "masks")

def main():
    print(f"Loading parquet from: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)

    source_dir = os.path.dirname(PARQUET_PATH)

    file_ids = df['cell_id'].unique()

    success_count = 0
    missing_count = 0
    error_count = 0

    # 3. Process each file
    for cell_id in file_ids:
        # Construct filename (User said cell_id does not have .tif extension)
        filename = f"{cell_id}.tif"
        src_path = os.path.join(source_dir, filename)

        dst_img_path = os.path.join(IMAGES_DIR, filename)
        dst_mask_path = os.path.join(MASKS_DIR, filename)

        if not os.path.exists(src_path):
            # Print only first few missing files to avoid spamming
            if missing_count < 5:
                print(f"Warning: Source file not found: {src_path}")
            missing_count += 1
            continue

        try:
            with Image.open(src_path) as img:
                # --- Extract Image (Page 0) ---
                img.seek(0)
                # We copy the frame to ensure we don't carry over settings when saving
                img.copy().save(dst_img_path)

                # --- Extract Mask (Page 1) ---
                # Check if file actually has a second page
                try:
                    img.seek(1)
                    img.copy().save(dst_mask_path)
                    success_count += 1
                except EOFError:
                    print(f"Error: File {filename} has only 1 page (missing mask).")
                    error_count += 1

        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            error_count += 1

    # 4. Summary
    print("-" * 30)
    print("Processing Complete")
    print(f"Successfully processed: {success_count}")
    print(f"Missing source files:   {missing_count}")
    print(f"Errors (e.g. 1 page):   {error_count}")
    print(f"Images saved to: {IMAGES_DIR}")
    print(f"Masks saved to:  {MASKS_DIR}")

if __name__ == "__main__":
    main()
