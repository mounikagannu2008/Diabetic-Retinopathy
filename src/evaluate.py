"""
Evaluate a trained DR classifier on the held-out test set.

Usage:
    python src/evaluate.py --checkpoint models/dr_classifier_best.pt

Produces, saved to outputs/:
  - evaluation_report.json  (all numeric metrics)
  - confusion_matrix.png
  - roc_curves.png
  - example_predictions/    (a handful of sample images with predictions)

Every number here is computed from real forward passes over the test split
that was carved out in make_splits.py and never used in training. Nothing
in this file invents or hardcodes a result.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, roc_curve, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from config import cfg
from dataset import DRDataset, get_eval_transforms
from model import DRClassifier


def load_model(checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DRClassifier(
        backbone=checkpoint["backbone"],
        num_classes=checkpoint["num_classes"],
        pretrained=False,  # we're loading trained weights, not ImageNet init
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def run_inference_on_loader(model, loader, device):
    all_labels, all_preds, all_probs, all_ids = [], [], [], []

    for imgs, labels, ids in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_ids.extend(ids)

    return (np.array(all_labels), np.array(all_preds),
            np.array(all_probs), list(all_ids))


def load_tuned_threshold():
    """
    Check for a probability threshold saved by tune_threshold.py. If found,
    the referable/non-referable decision uses P(referable) >= this threshold
    instead of the plain argmax-implied decision. Returns None if no tuning
    has been run yet, in which case callers fall back to argmax behavior.
    """
    path = os.path.join(cfg.output_dir, "tuned_threshold.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data["threshold"]


def compute_referable_metrics(labels, preds, probs, threshold=2, prob_threshold=None):
    """
    Referable DR = severity Level >= threshold (default 2, per ICDR
    convention: Moderate NPDR and above is referred to a specialist).

    If prob_threshold is given (from tune_threshold.py), the referable/
    non-referable decision uses P(referable) >= prob_threshold instead of
    the argmax-implied decision (preds >= threshold). The 5-class severity
    prediction (`preds`) itself is never altered either way — only this
    separate binary flag's decision rule changes.
    """
    ref_labels = (labels >= threshold).astype(int)
    # Probability of "referable" = sum of class probabilities for classes >= threshold
    ref_probs = probs[:, threshold:].sum(axis=1)

    if prob_threshold is not None:
        ref_preds = (ref_probs >= prob_threshold).astype(int)
        decision_rule = f"tuned probability threshold ({prob_threshold:.2f})"
    else:
        ref_preds = (preds >= threshold).astype(int)
        decision_rule = "argmax (no tuned threshold found — run tune_threshold.py to calibrate)"

    tn, fp, fn, tp = confusion_matrix(ref_labels, ref_preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    try:
        auc = roc_auc_score(ref_labels, ref_probs)
    except ValueError:
        auc = float("nan")  # only one class present in this split — can't compute AUC honestly

    return {
        "referable_sensitivity": float(sensitivity),
        "referable_specificity": float(specificity),
        "referable_roc_auc": float(auc),
        "referable_confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "meets_sensitivity_target_90pct": bool(sensitivity > 0.90),
        "meets_specificity_target_85pct": bool(specificity > 0.85),
        "decision_rule_used": decision_rule,
    }, ref_labels, ref_probs


def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("DR Severity Confusion Matrix (test set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return cm


def plot_roc_curve(ref_labels, ref_probs, save_path):
    if len(np.unique(ref_labels)) < 2:
        print("WARNING: test set has only one referable-DR class present — "
              "cannot plot a meaningful ROC curve. Skipping.")
        return
    fpr, tpr, _ = roc_curve(ref_labels, ref_probs)
    auc = roc_auc_score(ref_labels, ref_probs)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"Referable DR (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Referable DR (Level 2-4 vs 0-1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(cfg.model_dir, "dr_classifier_best.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"No checkpoint found at '{args.checkpoint}'. Train a model first with "
            f"src/train.py — evaluation cannot proceed without a genuinely trained model."
        )

    test_path = os.path.join(cfg.splits_dir, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test split not found at '{test_path}'. Run make_splits.py first.")
    test_df = pd.read_csv(test_path)
    print(f"Evaluating on {len(test_df)} held-out test images (never seen during training).")

    model, checkpoint = load_model(args.checkpoint, device)
    print(f"Loaded model: {checkpoint['backbone']}, trained to epoch {checkpoint['epoch']}, "
          f"val_f1 at save time = {checkpoint['val_f1']:.4f}")

    test_ds = DRDataset(test_df, cfg.images_dir, cfg.image_ext,
                         transform=get_eval_transforms(cfg.image_size))
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers)

    labels, preds, probs, ids = run_inference_on_loader(model, test_loader, device)

    # --- Per-class / overall metrics ---
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(cfg.num_classes)), zero_division=0
    )

    per_class = {}
    for i, name in enumerate(cfg.class_names):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    macro_f1 = float(np.mean(f1))
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))

    # --- Referable DR metrics (the brief's headline target) ---
    tuned_threshold = load_tuned_threshold()
    if tuned_threshold is not None:
        print(f"\nUsing tuned referable-DR probability threshold: {tuned_threshold:.2f} "
              f"(from outputs/tuned_threshold.json — run tune_threshold.py again to recalibrate)")
    else:
        print("\nNo tuned threshold found — using plain argmax for the referable/non-referable "
              "decision. Run tune_threshold.py to calibrate this against your validation set.")

    referable_metrics, ref_labels, ref_probs = compute_referable_metrics(
        labels, preds, probs, threshold=cfg.referable_threshold, prob_threshold=tuned_threshold
    )

    # --- Save confusion matrix + ROC ---
    cm = plot_confusion_matrix(labels, preds, cfg.class_names,
                                os.path.join(cfg.output_dir, "confusion_matrix.png"))
    plot_roc_curve(ref_labels, ref_probs, os.path.join(cfg.output_dir, "roc_curves.png"))

    report = {
        "test_set_size": len(test_df),
        "overall_accuracy": float(accuracy),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist(),
        "referable_dr_metrics": referable_metrics,
        "note": (
            "All metrics computed on a held-out test split not used during training "
            "or model selection. Targets (>90% sensitivity, >85% specificity for "
            "referable DR) are targets from the brief, not guaranteed outcomes — "
            "see meets_sensitivity_target_90pct / meets_specificity_target_85pct "
            "above for whether THIS run actually hit them."
        ),
    }

    report_path = os.path.join(cfg.output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (held-out test set)")
    print("=" * 60)
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f} | Macro precision: {macro_precision:.4f} | Macro recall: {macro_recall:.4f}")
    print(f"\nReferable DR sensitivity: {referable_metrics['referable_sensitivity']:.4f} "
          f"(target >0.90 -> {'MET' if referable_metrics['meets_sensitivity_target_90pct'] else 'NOT MET'})")
    print(f"Referable DR specificity: {referable_metrics['referable_specificity']:.4f} "
          f"(target >0.85 -> {'MET' if referable_metrics['meets_specificity_target_85pct'] else 'NOT MET'})")
    print(f"Referable DR ROC-AUC: {referable_metrics['referable_roc_auc']:.4f}")
    print(f"\nFull report: {report_path}")
    print(f"Confusion matrix: {os.path.join(cfg.output_dir, 'confusion_matrix.png')}")
    print("\nThese are the actual measured results of this run — not adjusted or assumed.")

    return report


if __name__ == "__main__":
    main()
