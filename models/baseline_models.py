#!/usr/bin/env python
"""
baseline_models.py — Conventional-UQ baselines for fair comparison vs CUED-Net.

Design constraints (for methodological symmetry with CUED-Net):
  - SAME backbone: two DenseNet-121 ViewEncoders (CC, MLO), ImageNet-pretrained.
  - SAME classifier trunk: 1024->256->64 with BN/ReLU/Dropout(0.4,0.3).
  - DIFFERENCE: evidential head -> plain Linear(64->2) softmax. No VDL, no
    uncertainty-weighted fusion. Views are fused by AVERAGING softmax probs
    (the simplest neutral dual-view fusion; does not borrow CUED-Net novelty).

Provides:
  - SoftmaxDualView: the shared architecture.
  - mc_dropout_predict(model, cc, mlo, T): T stochastic passes (dropout ON);
    returns mean prob + predictive entropy.
  - ensemble_predict(models, cc, mlo): mean prob over M models + predictive
    variance.

These reuse the SAME ViewEncoder backbone/classifier weights structure as
CUED-Net so the only varied factor is the UQ mechanism.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/workspace/cued_net")
from models.cued_net import ViewEncoder


class SoftmaxViewEncoder(nn.Module):
    """ViewEncoder with the evidential head replaced by a softmax linear head.
    Reuses ViewEncoder's DenseNet-121 features + classifier trunk verbatim."""
    def __init__(self, num_classes=2, pretrained=True, hidden_dim=256):
        super().__init__()
        base = ViewEncoder(num_classes=num_classes, pretrained=pretrained,
                           hidden_dim=hidden_dim)
        # reuse backbone + classifier trunk; drop the evidential layer
        self.features = base.features
        self.pool = base.pool
        self.classifier = base.classifier   # ends at 64-dim with Dropout(0.3)
        self.head = nn.Linear(64, num_classes)   # softmax logits

    def forward(self, x):
        f = self.features(x)
        f = self.pool(f).flatten(1)
        h = self.classifier(f)
        logits = self.head(h)
        return logits


class SoftmaxDualView(nn.Module):
    """Two SoftmaxViewEncoders; fuse by averaging per-view softmax probs."""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.encoder_cc = SoftmaxViewEncoder(num_classes, pretrained)
        self.encoder_mlo = SoftmaxViewEncoder(num_classes, pretrained)

    def freeze_encoders(self):
        for m in (self.encoder_cc, self.encoder_mlo):
            for p in m.features.parameters():
                p.requires_grad = False

    def unfreeze_encoders(self):
        for m in (self.encoder_cc, self.encoder_mlo):
            for p in m.features.parameters():
                p.requires_grad = True

    def forward(self, img_cc, img_mlo):
        logit_cc = self.encoder_cc(img_cc)
        logit_mlo = self.encoder_mlo(img_mlo)
        prob_cc = F.softmax(logit_cc, dim=1)
        prob_mlo = F.softmax(logit_mlo, dim=1)
        prob = 0.5 * (prob_cc + prob_mlo)        # neutral fusion
        # dict mirrors CUED-Net's interface so eval code is shared
        return {"prob": prob, "logit_cc": logit_cc, "logit_mlo": logit_mlo,
                "prob_cc": prob_cc, "prob_mlo": prob_mlo}


# --------------------------------------------------------------------------- #
# Cross-entropy loss compatible with train_single_model's call signature
#   criterion(outputs, targets, epoch, class_weights) -> {"total": loss}
# --------------------------------------------------------------------------- #
class SoftmaxDualLoss(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

    def forward(self, outputs, targets, epoch=0, class_weights=None):
        # average the per-view CE (each view supervised), matches dual-view training
        ce_cc = F.cross_entropy(outputs["logit_cc"], targets, weight=class_weights)
        ce_mlo = F.cross_entropy(outputs["logit_mlo"], targets, weight=class_weights)
        loss = 0.5 * (ce_cc + ce_mlo)
        return {"total": loss, "ce_cc": ce_cc, "ce_mlo": ce_mlo}


# --------------------------------------------------------------------------- #
# UQ inference
# --------------------------------------------------------------------------- #
def _enable_dropout(model):
    """Set ONLY dropout layers to train mode (keep BN in eval)."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model, img_cc, img_mlo, T=50):
    """T stochastic forward passes with dropout active.
    Returns dict: prob (mean), uncertainty (predictive entropy)."""
    model.eval()
    _enable_dropout(model)
    probs = []
    for _ in range(T):
        out = model(img_cc, img_mlo)
        probs.append(out["prob"].unsqueeze(0))   # (1,B,C)
    probs = torch.cat(probs, 0)                   # (T,B,C)
    mean_prob = probs.mean(0)                      # (B,C)
    # predictive entropy of the mean distribution
    ent = -(mean_prob.clamp_min(1e-8) * mean_prob.clamp_min(1e-8).log()).sum(1)
    return {"prob": mean_prob, "uncertainty": ent}


@torch.no_grad()
def ensemble_predict(models, img_cc, img_mlo):
    """Mean prob over M models; uncertainty = mean per-class variance across models."""
    probs = []
    for m in models:
        m.eval()
        out = m(img_cc, img_mlo)
        probs.append(out["prob"].unsqueeze(0))
    probs = torch.cat(probs, 0)                   # (M,B,C)
    mean_prob = probs.mean(0)
    var = probs.var(0).mean(1)                     # (B,) predictive variance
    return {"prob": mean_prob, "uncertainty": var}
