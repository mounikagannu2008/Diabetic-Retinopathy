"""
Pre-process every image in the train/val/test splits ONCE and cache the
result to disk. Training then reads these small, already-enhanced images
directly — instead of re-running crop/CLAHE/denoise on the full-resolution
original every single epoch.

This matters a lot on CPU: preprocessing 2,500+ multi-megapixel images is
expensive, and paying that cost 20 times (once per epoch) instead of once
is the difference between a training run finishing in hours vs. barely
completing one epoch.

Usage:
    python src/cache_preprocessed.py

Run this ONCE after make_splits.py, before train.py. Safe to re-run — it
skips any image that's already cached, so if it's interrupted, rerunning
picks up where it left off.
"""

import os
import sys
import time
import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from config import cfg
from preprocessing import preprocess_image


def cache_split(split_name: str):
    split_path = os.path.join(cfg.splits_dir, f"{split_name}.csv")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file '{split_path}' not found. Run make_splits.py first.")

    df = pd.read_csv(split_path)
    cache_dir = os.path.join(cfg.splits_dir, "..", "cache")
    cache_dir = os.path.normpath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    skipped, processed, failed = 0, 0, 0
    t0 = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Caching {split_name}"):
        id_code = row["id_code"]
        src_path = row["filepath"] if "filepath" in row and pd.notna(row["filepath"]) else \
            os.path.join(cfg.images_dir, f"{id_code}{cfg.image_ext}")
        cache_path = os.path.join(cache_dir, f"{id_code}.png")

        if os.path.exists(cache_path):
            skipped += 1
            continue

        img = cv2.imread(src_path)
        if img is None:
            print(f"\nWARNING: could not read '{src_path}' — skipping this image.")
            failed += 1
            continue

        try:
            processed_img = preprocess_image(img, target_size=cfg.image_size)
        except Exception as e:
            print(f"\nWARNING: preprocessing failed for '{src_path}': {e} — skipping.")
            failed += 1
            continue

        cv2.imwrite(cache_path, processed_img)
        processed += 1

    elapsed = time.time() - t0
    print(f"\n{split_name}: {processed} newly cached, {skipped} already cached, "
          f"{failed} failed, in {elapsed:.1f}s")
    return cache_dir


def main():
    print("Caching preprocessed images for all splits. This runs the expensive "
          "crop/CLAHE/denoise pipeline ONCE per image — training will read "
          "directly from this cache afterward and skip re-processing entirely.\n")

    cache_dir = None
    for split in ["train", "val", "test"]:
        cache_dir = cache_split(split)

    print(f"\nDone. Cached images are in: {cache_dir}")
    print("You can now run train.py — it will use the cache automatically "
          "(see config.py -> use_preprocessing_cache).")


if __name__ == "__main__":
    main()
