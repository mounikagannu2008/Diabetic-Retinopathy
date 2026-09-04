"""
Train the DR severity classifier.

Usage:
    python src/train.py

Requires src/make_splits.py to have been run first (produces
data/splits/{train,val,test}.csv).
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score

sys.path.insert(0, os.path.dirname(__file__))
from config import cfg
from dataset import DRDataset, get_train_transforms, get_eval_transforms, compute_class_weights
from model import DRClassifier


def get_device():
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    print("CUDA not available (or not requested) — falling back to CPU. "
          "Training will be significantly slower.")
    return torch.device("cpu")


def load_split(name: str) -> pd.DataFrame:
    path = os.path.join(cfg.splits_dir, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Split file '{path}' not found. Run `python src/make_splits.py` first."
        )
    return pd.read_csv(path)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    torch.set_grad_enabled(train)
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(imgs)
        loss = criterion(logits, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, macro_f1, macro_recall


def main():
    device = get_device()
    print(f"Using device: {device}")

    train_df = load_split("train")
    val_df = load_split("val")

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_ds = DRDataset(train_df, cfg.images_dir, cfg.image_ext,
                          transform=get_train_transforms(cfg.image_size))
    val_ds = DRDataset(val_df, cfg.images_dir, cfg.image_ext,
                        transform=get_eval_transforms(cfg.image_size))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    model = DRClassifier(backbone=cfg.backbone, num_classes=cfg.num_classes,
                          pretrained=cfg.pretrained).to(device)

    class_weights = compute_class_weights(train_df).to(device) if cfg.use_class_weights else None
    if class_weights is not None:
        print(f"Class weights: {class_weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)

    # Warm up: train only the classifier head first
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        model.get_param_groups(cfg.lr_backbone, cfg.lr_head), weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(cfg.num_epochs):
        if epoch == cfg.freeze_backbone_epochs:
            print(f"\n[Epoch {epoch}] Unfreezing backbone for full fine-tuning.")
            model.unfreeze_backbone()

        t0 = time.time()
        train_loss, train_f1, train_recall = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_f1, val_recall = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step(val_f1)
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1}/{cfg.num_epochs} ({elapsed:.1f}s) | "
              f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} val_recall={val_recall:.4f}")

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_f1": train_f1,
            "val_loss": val_loss, "val_f1": val_f1, "val_recall": val_recall,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            checkpoint_path = os.path.join(cfg.model_dir, "dr_classifier_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone": cfg.backbone,
                "num_classes": cfg.num_classes,
                "class_names": cfg.class_names,
                "image_size": cfg.image_size,
                "val_f1": val_f1,
                "epoch": epoch + 1,
            }, checkpoint_path)
            print(f"  -> New best model saved (val_f1={val_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(no improvement for {cfg.early_stopping_patience} epochs).")
                break

    history_path = os.path.join(cfg.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val_f1={best_val_f1:.4f}")
    print(f"Best model: {os.path.join(cfg.model_dir, 'dr_classifier_best.pt')}")
    print(f"Training history: {history_path}")
    print("\nThese numbers reflect the actual run on your machine — "
          "no results have been fabricated or assumed.")


if __name__ == "__main__":
    main()
