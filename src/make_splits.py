"""
Create a stratified train/val/test split from the labeled DR dataset.

Supports TWO dataset layouts (see config.py -> dataset_format, and
DATA_SETUP.md for a full explanation with examples):

  1. "csv"              — a CSV with id/diagnosis columns + a flat folder
                           of images (standard Kaggle competition download)
  2. "folder_per_class" — images already sorted into subfolders named by
                           class (e.g. images_dir/0/, images_dir/1/, ... or
                           images_dir/No_DR/, images_dir/Mild/, ...)

Regardless of input layout, this script produces a UNIFORM output: each of
train.csv / val.csv / test.csv has columns [id_code, diagnosis, filepath],
where filepath is a full resolved path to the image file. Every downstream
script (dataset.py etc.) reads from this uniform format, so the rest of the
pipeline never needs to know which layout the raw data came in.

Run this ONCE. The test set is fixed at split time and must never be
touched during training or model selection.

Usage:
    python src/make_splits.py
"""

import os
import sys
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from config import cfg

# Common name variants -> class index, used for folder_per_class auto-detection
DEFAULT_CLASS_NAME_MAP = {
    "0": 0, "no_dr": 0, "nodr": 0, "no dr": 0, "none": 0,
    "1": 1, "mild": 1, "mild_npdr": 1, "mild npdr": 1,
    "2": 2, "moderate": 2, "moderate_npdr": 2, "moderate npdr": 2,
    "3": 3, "severe": 3, "severe_npdr": 3, "severe npdr": 3,
    "4": 4, "proliferate_dr": 4, "proliferative": 4, "proliferative_dr": 4,
    "proliferate dr": 4, "pdr": 4,
}

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


def detect_format() -> str:
    """Best-effort guess at dataset layout when cfg.dataset_format == 'auto'."""
    if os.path.exists(cfg.labels_csv):
        return "csv"

    if os.path.isdir(cfg.images_dir):
        subdirs = [d for d in os.listdir(cfg.images_dir)
                   if os.path.isdir(os.path.join(cfg.images_dir, d))]
        if len(subdirs) >= 2:
            return "folder_per_class"

    raise RuntimeError(
        "Could not auto-detect dataset format. Neither a labels CSV was "
        f"found at '{cfg.labels_csv}' nor a folder-per-class layout was "
        f"found under '{cfg.images_dir}'. Set cfg.dataset_format explicitly "
        "and check your paths — see DATA_SETUP.md."
    )


def find_id_and_label_columns(df: pd.DataFrame):
    id_candidates = ["id_code", "image", "image_id", "id", "filename"]
    label_candidates = ["diagnosis", "level", "label", "grade", "dr_grade"]

    id_col = next((c for c in id_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    if id_col is None or label_col is None:
        raise ValueError(
            f"Could not auto-detect id/label columns in {list(df.columns)}. "
            f"Edit make_splits.py -> find_id_and_label_columns() to set them "
            f"explicitly for your CSV, or see DATA_SETUP.md."
        )
    return id_col, label_col


def find_image_path(images_dir: str, id_code: str, preferred_ext: str) -> str:
    """Try the configured extension first, then fall back to scanning for
    any supported image extension — handles CSVs where the id has no
    extension and the actual files don't match cfg.image_ext exactly."""
    candidate = os.path.join(images_dir, f"{id_code}{preferred_ext}")
    if os.path.exists(candidate):
        return candidate

    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(images_dir, f"{id_code}{ext}")
        if os.path.exists(candidate):
            return candidate

    return None


def load_from_csv() -> pd.DataFrame:
    if not os.path.exists(cfg.labels_csv):
        raise FileNotFoundError(
            f"Labels CSV not found at '{cfg.labels_csv}'. "
            f"Edit src/config.py -> labels_csv, or see DATA_SETUP.md."
        )

    raw = pd.read_csv(cfg.labels_csv)
    id_col, label_col = find_id_and_label_columns(raw)

    rows = []
    missing = 0
    for _, r in raw.iterrows():
        id_code = str(r[id_col])
        path = find_image_path(cfg.images_dir, id_code, cfg.image_ext)
        if path is None:
            missing += 1
            continue
        rows.append({"id_code": id_code, "diagnosis": int(r[label_col]), "filepath": path})

    if missing > 0:
        print(f"WARNING: {missing} rows in the CSV had no matching image file "
              f"under '{cfg.images_dir}' and were skipped. If this number is "
              f"large, your images_dir or image_ext is probably misconfigured "
              f"— check DATA_SETUP.md before proceeding.")

    if len(rows) == 0:
        raise RuntimeError(
            "No CSV rows matched an actual image file. images_dir or "
            "image_ext in config.py is almost certainly wrong — see DATA_SETUP.md."
        )

    return pd.DataFrame(rows)


def load_from_folder_per_class() -> pd.DataFrame:
    if not os.path.isdir(cfg.images_dir):
        raise FileNotFoundError(
            f"images_dir '{cfg.images_dir}' not found. "
            f"Edit src/config.py, or see DATA_SETUP.md."
        )

    class_map = cfg.class_folder_map or DEFAULT_CLASS_NAME_MAP
    subdirs = [d for d in os.listdir(cfg.images_dir)
               if os.path.isdir(os.path.join(cfg.images_dir, d))]

    rows = []
    unmatched_dirs = []
    for d in subdirs:
        key = d.strip().lower()
        if key not in class_map:
            unmatched_dirs.append(d)
            continue
        label = class_map[key]

        class_dir = os.path.join(cfg.images_dir, d)
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(glob.glob(os.path.join(class_dir, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(class_dir, f"*{ext.upper()}")))

        for path in image_files:
            id_code = os.path.splitext(os.path.basename(path))[0]
            rows.append({"id_code": id_code, "diagnosis": label, "filepath": path})

    if unmatched_dirs:
        print(f"WARNING: found subfolders that didn't match any known class "
              f"name and were skipped: {unmatched_dirs}. If these ARE class "
              f"folders, set cfg.class_folder_map explicitly in config.py — "
              f"see DATA_SETUP.md.")

    if len(rows) == 0:
        raise RuntimeError(
            "No images found across any recognized class subfolder under "
            f"'{cfg.images_dir}'. See DATA_SETUP.md to check your layout."
        )

    return pd.DataFrame(rows)


def main():
    fmt = cfg.dataset_format
    if fmt == "auto":
        fmt = detect_format()
        print(f"Auto-detected dataset format: '{fmt}'")

    if fmt == "csv":
        df = load_from_csv()
    elif fmt == "folder_per_class":
        df = load_from_folder_per_class()
    else:
        raise ValueError(f"Unknown dataset_format '{fmt}' in config.py — must be "
                          f"'csv', 'folder_per_class', or 'auto'.")

    bad = df[~df["diagnosis"].isin(range(cfg.num_classes))]
    if len(bad) > 0:
        raise ValueError(
            f"Found {len(bad)} rows with diagnosis outside 0-{cfg.num_classes - 1}. "
            f"Inspect your data — this pipeline will not silently drop or relabel them."
        )

    print(f"\nLoaded {len(df)} labeled images.")
    print("Class distribution (whole dataset):")
    print(df["diagnosis"].value_counts().sort_index())

    if cfg.max_total_samples is not None and cfg.max_total_samples < len(df):
        # Stratified subsample: keep the same class proportions as the full
        # dataset, just fewer total images. This is a scope cut for hardware
        # feasibility, not a quality shortcut — proportions are preserved so
        # the subset is a legitimate (if smaller) representative sample.
        frac = cfg.max_total_samples / len(df)
        df, _ = train_test_split(
            df, train_size=frac, stratify=df["diagnosis"], random_state=cfg.random_seed
        )
        print(f"\nSubsampled to {len(df)} images (cfg.max_total_samples={cfg.max_total_samples}) "
              f"for training feasibility. Class distribution after subsampling:")
        print(df["diagnosis"].value_counts().sort_index())
        print("NOTE: results from this run reflect a reduced-size subset, not the "
              "full labeled dataset. Report this honestly — it is expected to "
              "underperform a full-dataset run, especially on rarer classes.")

    train_val_df, test_df = train_test_split(
        df, test_size=cfg.test_size, stratify=df["diagnosis"], random_state=cfg.random_seed,
    )
    relative_val_size = cfg.val_size / (1.0 - cfg.test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_size,
        stratify=train_val_df["diagnosis"], random_state=cfg.random_seed,
    )

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = os.path.join(cfg.splits_dir, f"{name}.csv")
        split_df.to_csv(out_path, index=False)
        print(f"\n{name}: {len(split_df)} images -> {out_path}")
        print(split_df["diagnosis"].value_counts().sort_index())

    print(
        "\nDone. The test.csv split must not be used for anything except the "
        "final evaluation step — no peeking during training or tuning."
    )


if __name__ == "__main__":
    main()
