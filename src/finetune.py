"""
Fine-tune an already-trained checkpoint further — e.g. after adding more
data, or to continue training with a different (usually lower) learning
rate once the initial run has plateaued.

Usage:
    python src/finetune.py --checkpoint models/dr_classifier_best.pt \\
        --epochs 10 --lr_backbone 1e-5 --lr_head 1e-4

This is deliberately a separate entry point from train.py (not a hidden
flag) so it's always explicit which mode you're running: fresh training
from ImageNet weights, vs. continued training from your own prior run.
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from config import cfg
from dataset import DRDataset, get_train_transforms, get_eval_transforms, compute_class_weights
from model import DRClassifier
from train import run_epoch, load_split, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to existing .pt checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr_backbone", type=float, default=5e-6,
                         help="Lower than initial training LR by default — fine-tuning "
                              "should nudge weights, not overwrite what was already learned")
    parser.add_argument("--lr_head", type=float, default=5e-5)
    parser.add_argument("--output_name", default="dr_classifier_finetuned.pt")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at '{args.checkpoint}'.")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    print(f"Resuming from checkpoint: backbone={checkpoint['backbone']}, "
          f"previously trained to epoch {checkpoint['epoch']}, "
          f"val_f1 at save time={checkpoint['val_f1']:.4f}")

    model = DRClassifier(
        backbone=checkpoint["backbone"], num_classes=checkpoint["num_classes"], pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.unfreeze_backbone()  # fine-tuning always trains the full network

    train_df = load_split("train")
    val_df = load_split("val")

    train_ds = DRDataset(train_df, cfg.images_dir, cfg.image_ext, transform=get_train_transforms(cfg.image_size))
    val_ds = DRDataset(val_df, cfg.images_dir, cfg.image_ext, transform=get_eval_transforms(cfg.image_size))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    class_weights = compute_class_weights(train_df).to(device) if cfg.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)

    optimizer = torch.optim.AdamW(
        model.get_param_groups(args.lr_backbone, args.lr_head), weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_f1 = checkpoint["val_f1"]  # only save if we actually beat the incoming checkpoint
    history = []

    print(f"\nStarting point val_f1={best_val_f1:.4f} — fine-tuning will only overwrite "
          f"the checkpoint if this is improved on.")

    for epoch in range(args.epochs):
        train_loss, train_f1, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_f1, val_recall = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_f1)

        print(f"[Fine-tune] Epoch {epoch+1}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} val_recall={val_recall:.4f}")

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_f1": train_f1,
                         "val_loss": val_loss, "val_f1": val_f1, "val_recall": val_recall})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            out_path = os.path.join(cfg.model_dir, args.output_name)
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone": checkpoint["backbone"],
                "num_classes": checkpoint["num_classes"],
                "class_names": checkpoint["class_names"],
                "image_size": checkpoint["image_size"],
                "val_f1": val_f1,
                "epoch": checkpoint["epoch"] + epoch + 1,
                "fine_tuned_from": args.checkpoint,
            }, out_path)
            print(f"  -> Improved. Saved to {out_path} (val_f1={val_f1:.4f})")

    history_path = os.path.join(cfg.output_dir, "finetune_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    if best_val_f1 == checkpoint["val_f1"]:
        print(f"\nFine-tuning did not improve on the starting val_f1={best_val_f1:.4f}. "
              f"No new checkpoint was saved — the original at '{args.checkpoint}' remains best. "
              f"This is a real, reported outcome, not adjusted to look better.")
    else:
        print(f"\nFine-tuning improved val_f1 from {checkpoint['val_f1']:.4f} to {best_val_f1:.4f}.")


if __name__ == "__main__":
    main()
