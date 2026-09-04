"""
Single-image inference: fundus image -> DR level, probabilities, confidence,
referable/non-referable, Grad-CAM explanation.

Usage:
    python src/inference.py --image path/to/image.png --checkpoint models/dr_classifier_best.pt
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import cfg
from preprocessing import preprocess_image, assess_quality
from dataset import IMAGENET_MEAN, IMAGENET_STD
from model import DRClassifier
from gradcam import DRExplainer, tensor_to_rgb_float


def load_model(checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DRClassifier(
        backbone=checkpoint["backbone"],
        num_classes=checkpoint["num_classes"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def image_to_tensor(preprocessed_bgr: np.ndarray, device) -> torch.Tensor:
    rgb = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    normalized = (rgb - mean) / std
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    return tensor


def load_tuned_threshold():
    """Same tuned-threshold file evaluate.py checks for — kept as a small,
    independent copy here rather than a cross-import, so inference.py stays
    usable standalone without pulling in evaluate.py's dependencies."""
    path = os.path.join(cfg.output_dir, "tuned_threshold.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data["threshold"]


def predict_single_image(image_path: str, model, device, class_names,
                          referable_threshold: int, generate_explanation: bool = True):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'.")

    quality = assess_quality(img_bgr)

    preprocessed = preprocess_image(img_bgr, target_size=cfg.image_size)
    input_tensor = image_to_tensor(preprocessed, device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    referable_prob = float(probs[referable_threshold:].sum())

    tuned_threshold = load_tuned_threshold()
    if tuned_threshold is not None:
        is_referable = referable_prob >= tuned_threshold
        decision_rule = f"tuned probability threshold ({tuned_threshold:.2f})"
    else:
        is_referable = pred_class >= referable_threshold
        decision_rule = "argmax (no tuned threshold found — run tune_threshold.py to calibrate)"

    result = {
        "image_path": image_path,
        "quality_check": quality,
        "predicted_level": pred_class,
        "predicted_class_name": class_names[pred_class],
        "class_probabilities": {name: float(p) for name, p in zip(class_names, probs)},
        "confidence": confidence,
        "is_referable": bool(is_referable),
        "referable_probability": referable_prob,
        "referable_label": "Referable" if is_referable else "Non-referable",
        "referable_decision_rule": decision_rule,
    }

    heatmap, overlay = None, None
    if generate_explanation:
        explainer = DRExplainer(model, device)
        rgb_float = tensor_to_rgb_float(preprocessed)
        heatmap, overlay = explainer.explain(input_tensor, rgb_float, target_class=pred_class)

    return result, preprocessed, heatmap, overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=os.path.join(cfg.model_dir, "dr_classifier_best.pt"))
    parser.add_argument("--no-explain", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint, device)

    result, preprocessed, heatmap, overlay = predict_single_image(
        args.image, model, device, cfg.class_names, cfg.referable_threshold,
        generate_explanation=not args.no_explain,
    )

    print(json.dumps(result, indent=2))

    if not result["quality_check"]["likely_gradeable"]:
        print("\nWARNING: image failed basic quality heuristics "
              "(blur/brightness) — prediction may be unreliable. "
              "Consider recapturing this image.")

    if overlay is not None:
        out_dir = os.path.join(cfg.output_dir, "example_predictions")
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.image))[0]

        cv2.imwrite(os.path.join(out_dir, f"{base_name}_preprocessed.png"), preprocessed)
        cv2.imwrite(os.path.join(out_dir, f"{base_name}_gradcam_overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"\nSaved preprocessed image and Grad-CAM overlay to {out_dir}/")


if __name__ == "__main__":
    main()
