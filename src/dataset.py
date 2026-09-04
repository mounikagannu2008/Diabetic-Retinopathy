"""
PyTorch Dataset for DR fundus images.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import cfg
from preprocessing import preprocess_image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int):
    """Augmentation for training. Chosen to reflect realistic fundus-image
    variation (rotation/flip = camera angle, brightness/contrast = lighting,
    slight blur = focus variation) WITHOUT distorting lesion morphology in
    ways that would misrepresent pathology (no elastic warps, no color
    inversion, no aggressive cutout over lesion-bearing regions)."""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transforms(image_size: int):
    """No augmentation for val/test — only normalization, so evaluation
    reflects the model's real behavior on unmodified (but preprocessed)
    images."""
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class DRDataset(Dataset):
    def __init__(self, df, images_dir, image_ext, transform=None,
                 apply_preprocessing=True, use_cache=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.image_ext = image_ext
        self.transform = transform
        self.apply_preprocessing = apply_preprocessing
        # Default to cfg's setting, but allow explicit override per-instance
        self.use_cache = cfg.use_preprocessing_cache if use_cache is None else use_cache
        self.cache_dir = os.path.normpath(os.path.join(cfg.splits_dir, "..", "cache"))

    def __len__(self):
        return len(self.df)

    def _load_image(self, row) -> np.ndarray:
        # Splits produced by make_splits.py always include a resolved
        # filepath column — use it directly rather than reconstructing
        # from id_code, so this works identically for both the "csv" and
        # "folder_per_class" dataset layouts.
        path = row["filepath"] if "filepath" in row and pd.notna(row["filepath"]) else \
            os.path.join(self.images_dir, f"{row['id_code']}{self.image_ext}")
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(
                f"Could not read image at '{path}'. The file may have moved "
                f"since make_splits.py was run — rerun it if your dataset "
                f"location changed."
            )
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_code = row["id_code"]
        label = int(row["diagnosis"])

        if self.use_cache:
            cache_path = os.path.join(self.cache_dir, f"{id_code}.png")
            if not os.path.exists(cache_path):
                raise FileNotFoundError(
                    f"Cached preprocessed image not found at '{cache_path}'. "
                    f"Run `python src/cache_preprocessed.py` first, or set "
                    f"cfg.use_preprocessing_cache = False to preprocess on "
                    f"the fly (much slower on CPU)."
                )
            # Already cropped, resized, CLAHE'd, denoised — no further
            # preprocessing needed, just load and hand to augmentation.
            img = cv2.imread(cache_path)
            if img is None:
                raise FileNotFoundError(f"Cached image at '{cache_path}' exists but could not be read.")
        else:
            img = self._load_image(row)
            if self.apply_preprocessing:
                img = preprocess_image(img, target_size=cfg.image_size)
            else:
                img = cv2.resize(img, (cfg.image_size, cfg.image_size))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return img, label, id_code


def compute_class_weights(df) -> torch.Tensor:
    """Inverse-frequency class weights to counter APTOS-style imbalance
    (Level 0 is heavily overrepresented). Used in the loss function, not
    for oversampling, to keep epoch composition predictable."""
    counts = df["diagnosis"].value_counts().sort_index()
    counts = counts.reindex(range(cfg.num_classes), fill_value=0)
    total = counts.sum()
    weights = total / (cfg.num_classes * counts.replace(0, 1))
    return torch.tensor(weights.values, dtype=torch.float32)
