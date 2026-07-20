"""
view_encoder_backbone.py — backbone-swappable ViewEncoder for CUED-Net Table III.

Drop-in replacement for models.cued_net.ViewEncoder that accepts a `backbone`
kwarg. Preserves the EXACT downstream contract of the original:
  - self.features : the conv feature extractor (named `.features` so the
                    discriminative-LR and freeze/unfreeze logic in
                    train_single_model still finds it)
  - forward() returns the same dict (evidential output + 'features')
  - AdaptiveAvgPool2d(1) over a (B, C, H, W) feature map  -> flatten -> head

Only the backbone + the head's first Linear(feat_dim, hidden_dim) change.
Everything downstream (fusion, discordance, triple uncertainty) is dim-invariant
because it consumes `prob`/`uncertainty`, never the raw feature vector.
"""
import torch.nn as nn
import torchvision.models as tvm

from models.cued_net import EvidentialLayer

# feature-map channel dims (output of the conv stack, pre-pool)
_FEAT_DIM = {
    "densenet121":     1024,
    "resnet50":        2048,
    "efficientnet_b0": 1280,
    "convnext_tiny":   768,
}


def _build_backbone(backbone, pretrained):
    """Return (features_module, feat_dim). `features_module` outputs (B, C, H, W)."""
    if backbone == "densenet121":
        net = tvm.densenet121(
            weights=tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        features = net.features  # native, ends in (B,1024,H,W)

    elif backbone == "resnet50":
        net = tvm.resnet50(
            weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        # ResNet has no .features; conv stack = everything except avgpool + fc.
        # children()[:-2] = conv1..layer4, output (B,2048,H,W). Named `.features`
        # below so LR/freeze logic in train_single_model finds it.
        features = nn.Sequential(*list(net.children())[:-2])

    elif backbone == "efficientnet_b0":
        net = tvm.efficientnet_b0(
            weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        features = net.features  # native, ends in (B,1280,H,W)

    elif backbone == "convnext_tiny":
        net = tvm.convnext_tiny(
            weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        features = net.features  # native, ends in (B,768,H,W)

    else:
        raise ValueError(f"unknown backbone: {backbone}")

    return features, _FEAT_DIM[backbone]


class ViewEncoderBackbone(nn.Module):
    """Single view encoder with evidential output; backbone-parametrized.

    Identical to the original ViewEncoder except for the backbone + the head's
    input dim. Attribute names (`features`, `classifier`, `evidential`) and the
    forward() output dict are preserved exactly.
    """
    def __init__(self, num_classes=2, pretrained=True, hidden_dim=256,
                 backbone="densenet121"):
        super().__init__()
        self.backbone_name = backbone

        self.features, feat_dim = _build_backbone(backbone, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head — ONLY the first Linear's in-dim changes per backbone.
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.evidential = EvidentialLayer(64, num_classes)

    def forward(self, x):
        features = self.features(x)
        features = self.pool(features).flatten(1)
        hidden = self.classifier(features)
        output = self.evidential(hidden)
        output['features'] = features
        return output
