"""
DR severity classification model: EfficientNet-B3 backbone (ImageNet-pretrained),
fine-tuned for 5-class ordinal DR grading.

Why EfficientNet-B3 (recap from the architecture analysis):
  - Strong accuracy/parameter/speed trade-off among CNN backbones
  - Native Grad-CAM compatibility (final conv layer gives spatially meaningful
    activations, unlike ViT's patch-token representations)
  - Proven on this exact data regime in the published APTOS/IDRiD literature
  - Pretrained features transfer well to fundus images despite the domain
    gap from ImageNet's natural images (well established empirically for
    medical imaging transfer learning)
"""

import torch
import torch.nn as nn
import timm


class DRClassifier(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b3", num_classes: int = 5,
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        self._backbone_name = backbone

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def get_last_conv_layer(self):
        """Returns the module to hook for Grad-CAM. For timm EfficientNet
        models, `conv_head` is the final conv layer before pooling — it
        carries spatial information tied to input regions, which is what
        Grad-CAM needs."""
        if hasattr(self.backbone, "conv_head"):
            return self.backbone.conv_head
        raise AttributeError(
            f"Backbone '{self._backbone_name}' has no 'conv_head' attribute. "
            f"If you changed the backbone, update get_last_conv_layer() to "
            f"point at the correct final conv layer for Grad-CAM."
        )

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def get_param_groups(self, lr_backbone: float, lr_head: float):
        """Discriminative learning rates: smaller for the pretrained
        backbone (avoid destroying learned features), larger for the
        randomly-initialized classifier head (needs to learn fast)."""
        return [
            {"params": self.backbone.parameters(), "lr": lr_backbone},
            {"params": self.classifier.parameters(), "lr": lr_head},
        ]
