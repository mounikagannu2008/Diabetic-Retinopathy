"""
Grad-CAM explainability for the DR classifier.

Why Grad-CAM (recap): it's the natural fit for a CNN backbone — it uses
gradients flowing into the final convolutional layer to weight that layer's
activation maps, producing a spatial heatmap of which image regions drove
the prediction. This is directly interpretable by a clinician (highlighted
regions should correspond to lesions), cheap to compute (one extra backward
pass), and doesn't require any architecture changes or retraining.
"""

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import DRClassifier


class DRExplainer:
    def __init__(self, model: DRClassifier, device):
        self.model = model
        self.device = device
        target_layer = model.get_last_conv_layer()
        self.cam = GradCAM(model=model, target_layers=[target_layer])

    def explain(self, input_tensor: torch.Tensor, rgb_image_float: np.ndarray,
                target_class: int = None):
        """
        input_tensor: normalized model input, shape (1, C, H, W)
        rgb_image_float: the SAME image, RGB, float32, values in [0, 1],
                          shape (H, W, 3) — used only for overlay visualization
        target_class: class index to explain. If None, explains the
                       model's own top prediction (standard Grad-CAM usage).

        Returns: (heatmap [H,W] float32 in [0,1], overlay [H,W,3] uint8)
        """
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        heatmap = grayscale_cam[0]  # first (only) image in batch

        overlay = show_cam_on_image(rgb_image_float, heatmap, use_rgb=True)

        return heatmap, overlay


def tensor_to_rgb_float(img_bgr_uint8: np.ndarray) -> np.ndarray:
    """Convert a preprocessed BGR uint8 image (as produced by preprocess_image)
    into the RGB float32 [0,1] format Grad-CAM's overlay utility expects."""
    rgb = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0
