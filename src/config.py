"""
Central configuration for the DR screening Level-1 pipeline.

EDIT THESE PATHS to point at your local copy of the dataset before running
anything. Nothing in this pipeline downloads data automatically — you must
supply it.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ------------------------------------------------------------------
    # PATHS  ---  <<< EDIT THESE FOR YOUR MACHINE >>>
    # ------------------------------------------------------------------
    # Dataset layout format. Two supported layouts — pick the one that
    # matches what you actually downloaded (see DATA_SETUP.md):
    #   "csv"              -> a CSV with id/diagnosis columns + one flat
    #                          folder of images (standard Kaggle APTOS
    #                          competition download)
    #   "folder_per_class" -> images already sorted into subfolders named
    #                          after their class (e.g. "0", "1", ... or
    #                          "No_DR", "Mild", ... under images_dir)
    #   "auto"             -> inspect images_dir at runtime and guess
    dataset_format: str = "auto"

    # Used when dataset_format is "csv" or "auto" resolves to "csv"
    labels_csv: str = "data/train.csv"

    # Used for both formats: for "csv", the flat folder of images; for
    # "folder_per_class", the parent folder containing the class subfolders
    images_dir: str = "data/train_images"
    image_ext: str = ".png"          # change to ".jpg" if your copy differs

    # Used only for "folder_per_class": maps folder name -> class index.
    # Leave as None to auto-detect (handles "0".."4" and common name
    # variants like "No_DR"/"Mild"/"Moderate"/"Severe"/"Proliferate_DR").
    # Set explicitly if your folder names don't match either pattern.
    class_folder_map: dict = field(default=None)

    # Where outputs go
    output_dir: str = "outputs"
    model_dir: str = "models"
    splits_dir: str = "data/splits"

    # ------------------------------------------------------------------
    # CLASSES
    # ------------------------------------------------------------------
    class_names: Tuple[str, ...] = (
        "No DR",            # 0
        "Mild NPDR",         # 1
        "Moderate NPDR",     # 2
        "Severe NPDR",       # 3
        "Proliferative DR",  # 4
    )
    num_classes: int = 5
    # Referable DR = Level 2, 3, 4 (per ICDR clinical convention used in the brief)
    referable_threshold: int = 2

    # ------------------------------------------------------------------
    # SPLIT
    # ------------------------------------------------------------------
    max_total_samples: int = 500  # <<< SUBSET FOR CPU FEASIBILITY >>>
                                    # Set to None to use the full labeled dataset.
                                    # 500 is a deliberate scope-cut for slow/CPU-only
                                    # hardware — stratified so class proportions match
                                    # the full dataset. Results from this subset will
                                    # be weaker than a full-dataset run and should be
                                    # reported as such, not presented as the full result.
    test_size: float = 0.15       # held-out, never touched until final eval
    val_size: float = 0.15        # used for early stopping / model selection
    random_seed: int = 42

    # ------------------------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------------------------
    image_size: int = 224         # EfficientNet-B0 native resolution — lighter than B3's 300,
                                    # matters a lot on CPU (roughly halves per-image compute)
    use_preprocessing_cache: bool = True   # read from pre-cached images (see
                                             # cache_preprocessed.py) instead of
                                             # re-running crop/CLAHE/denoise every
                                             # epoch — essential for CPU training speed
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    backbone: str = "efficientnet_b0"   # switched from B3 -> B0 for CPU feasibility:
                                          # ~5M params vs B3's ~12M, meaningfully faster
                                          # per-batch on CPU with only a modest accuracy
                                          # trade-off — the right call when there's no GPU
    pretrained: bool = True
    batch_size: int = 16
    num_epochs: int = 12            # trimmed further for the 500-image subset —
                                      # a small dataset converges (or overfits) fast,
                                      # no need for a long ceiling
    lr_head: float = 1e-3          # LR for newly-initialized classifier head
    lr_backbone: float = 1e-4      # LR for fine-tuned pretrained backbone
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    early_stopping_patience: int = 4   # trimmed further to match the shorter run
    freeze_backbone_epochs: int = 3   # warm up head before unfreezing backbone
    num_workers: int = 4
    use_class_weights: bool = True

    # ------------------------------------------------------------------
    # DEVICE
    # ------------------------------------------------------------------
    device: str = "cuda"  # falls back to cpu automatically at runtime if unavailable


cfg = Config()

for d in (cfg.output_dir, cfg.model_dir, cfg.splits_dir):
    os.makedirs(d, exist_ok=True)
