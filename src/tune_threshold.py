"""
Tune the referable-DR decision threshold on the VALIDATION set, then report
the result on the (still-untouched) TEST set exactly once.

Why this exists: the model's 5-class predictions (argmax) can produce a
referable-DR sensitivity/specificity trade-off that isn't ideal for a
screening context, even when the model's underlying probabilities are
well-calibrated (see evaluation_report.json -> referable_roc_auc). Instead
of picking whichever class has the single highest probability, we pick a
probability-mass threshold on P(referable) = P(Level2) + P(Level3) + P(Level4)
directly, and choose that threshold to prioritize sensitivity — appropriate
for a screening tool, where missing disease is costlier than a false alarm.

This does NOT retrain the model or touch its weights. It only changes the
decision rule applied to outputs the model already produces. The 5-class
predictions (which DR level, for severity reporting) are unaffected — only
the separate binary referable/non-referable flag uses the tuned threshold.

Usage:
    python src/tune_threshold.py --checkpoint models/dr_classifier_best.pt
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from config import cfg
from dataset import DRDataset, get_eval_transforms
from evaluate import load_model, run_inference_on_loader


def sensitivity_specificity_at_threshold(ref_labels, ref_probs, threshold):
    ref_preds = (ref_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(ref_labels, ref_preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return sens, spec, {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def find_best_threshold(ref_labels, ref_probs, min_specificity=0.85):
    """
    Scan thresholds from 0.05 to 0.95. Among thresholds that keep
    specificity >= min_specificity (the brief's other target — we don't
    want to blow past it in the other direction), pick the one that
    maximizes sensitivity. If none satisfy the specificity floor, pick the
    threshold that maximizes sensitivity anyway and report the shortfall
    honestly rather than silently failing.
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    results = []
    for t in thresholds:
        sens, spec, cm = sensitivity_specificity_at_threshold(ref_labels, ref_probs, t)
        results.append({"threshold": float(t), "sensitivity": sens, "specificity": spec, "confusion": cm})

    valid = [r for r in results if r["specificity"] >= min_specificity]
    pool = valid if valid else results
    best = max(pool, key=lambda r: r["sensitivity"])
    return best, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(cfg.model_dir, "dr_classifier_best.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint, device)
    print(f"Loaded model: {checkpoint['backbone']}, trained to epoch {checkpoint['epoch']}")

    # --- Step 1: tune threshold on VALIDATION set only ---
    val_df = pd.read_csv(os.path.join(cfg.splits_dir, "val.csv"))
    val_ds = DRDataset(val_df, cfg.images_dir, cfg.image_ext, transform=get_eval_transforms(cfg.image_size))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    val_labels, _, val_probs, _ = run_inference_on_loader(model, val_loader, device)
    val_ref_labels = (val_labels >= cfg.referable_threshold).astype(int)
    val_ref_probs = val_probs[:, cfg.referable_threshold:].sum(axis=1)

    print(f"\nScanning thresholds on VALIDATION set ({len(val_df)} images)...")
    best, all_results = find_best_threshold(val_ref_labels, val_ref_probs, min_specificity=0.85)
    print(f"Selected threshold: {best['threshold']:.2f} "
          f"(val sensitivity={best['sensitivity']:.4f}, val specificity={best['specificity']:.4f})")

    if best["specificity"] < 0.85:
        print("WARNING: no threshold on the validation set achieved both targets "
              "simultaneously. Reporting the best sensitivity-maximizing threshold "
              "found, with its actual specificity shortfall shown above — not hidden.")

    # --- Step 2: apply that FIXED threshold to TEST set exactly once ---
    test_df = pd.read_csv(os.path.join(cfg.splits_dir, "test.csv"))
    test_ds = DRDataset(test_df, cfg.images_dir, cfg.image_ext, transform=get_eval_transforms(cfg.image_size))
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    test_labels, test_preds_argmax, test_probs, _ = run_inference_on_loader(model, test_loader, device)
    test_ref_labels = (test_labels >= cfg.referable_threshold).astype(int)
    test_ref_probs = test_probs[:, cfg.referable_threshold:].sum(axis=1)

    tuned_sens, tuned_spec, tuned_cm = sensitivity_specificity_at_threshold(
        test_ref_labels, test_ref_probs, best["threshold"]
    )

    # For comparison: what argmax (the original decision rule) gives on this same test set
    argmax_ref_preds = (test_preds_argmax >= cfg.referable_threshold).astype(int)
    argmax_tn, argmax_fp, argmax_fn, argmax_tp = confusion_matrix(
        test_ref_labels, argmax_ref_preds, labels=[0, 1]
    ).ravel()
    argmax_sens = argmax_tp / (argmax_tp + argmax_fn) if (argmax_tp + argmax_fn) > 0 else float("nan")
    argmax_spec = argmax_tn / (argmax_tn + argmax_fp) if (argmax_tn + argmax_fp) > 0 else float("nan")

    auc = roc_auc_score(test_ref_labels, test_ref_probs)

    report = {
        "tuned_threshold": float(best["threshold"]),
        "threshold_selection_note": (
            "Threshold selected by scanning validation-set performance only, "
            "then applied once to the test set. Test set was not used in threshold "
            "selection."
        ),
        "test_set_size": len(test_df),
        "original_argmax_decision_rule": {
            "referable_sensitivity": float(argmax_sens),
            "referable_specificity": float(argmax_spec),
        },
        "tuned_threshold_decision_rule": {
            "referable_sensitivity": float(tuned_sens),
            "referable_specificity": float(tuned_spec),
            "confusion_matrix": tuned_cm,
        },
        "referable_roc_auc": float(auc),
        "meets_sensitivity_target_90pct": bool(tuned_sens > 0.90),
        "meets_specificity_target_85pct": bool(tuned_spec > 0.85),
    }

    report_path = os.path.join(cfg.output_dir, "threshold_tuning_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save just the threshold value in a small, separate file that evaluate.py
    # and inference.py both check for — this is what actually wires the tuned
    # threshold into the rest of the pipeline (including the app's Evaluate
    # and Predict tabs, which call those scripts).
    tuned_threshold_path = os.path.join(cfg.output_dir, "tuned_threshold.json")
    with open(tuned_threshold_path, "w") as f:
        json.dump({
            "threshold": float(best["threshold"]),
            "tuned_on": "validation set",
            "val_sensitivity_at_tuning": float(best["sensitivity"]),
            "val_specificity_at_tuning": float(best["specificity"]),
        }, f, indent=2)
    print(f"Tuned threshold saved to: {tuned_threshold_path}")
    print("evaluate.py and inference.py will now automatically use this threshold "
          "for the referable/non-referable decision.")

    print("\n" + "=" * 60)
    print("THRESHOLD TUNING RESULTS (test set, evaluated once)")
    print("=" * 60)
    print(f"Original (argmax) decision rule: "
          f"sensitivity={argmax_sens:.4f}, specificity={argmax_spec:.4f}")
    print(f"Tuned threshold ({best['threshold']:.2f}) decision rule: "
          f"sensitivity={tuned_sens:.4f}, specificity={tuned_spec:.4f}")
    print(f"\nSensitivity target >90%: {'MET' if tuned_sens > 0.90 else 'NOT MET'}")
    print(f"Specificity target >85%: {'MET' if tuned_spec > 0.85 else 'NOT MET'}")
    print(f"\nFull report: {report_path}")
    print("\nNote: only the referable/non-referable FLAG uses this tuned threshold. "
          "The 5-class DR severity prediction (No DR / Mild / Moderate / Severe / "
          "Proliferative) is unchanged and still uses the model's direct argmax output.")


if __name__ == "__main__":
    main()
