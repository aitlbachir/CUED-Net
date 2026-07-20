#!/usr/bin/env python
"""Trusted Multi-View (TMC) and single-view EDL baselines."""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/workspace/cued_net")
from models.cued_net import ViewEncoder, CUEDNetLoss


# =========================================================================== #
# Baseline 3: Single-view EDL
# =========================================================================== #
class SingleViewEDL(nn.Module):
    """One CC encoder with evidential head. No MLO, no fusion, no VDL.

    forward(img_cc, img_mlo) keeps the dual-view signature so the training
    harness is shared, but img_mlo is IGNORED (single-view by design).
    Returns a dict exposing 'prob' and 'uncertainty' (K/S vacuity), plus
    'alpha'/'cc_out' so the evidential loss can be reused unchanged.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.encoder_cc = ViewEncoder(num_classes, pretrained)

    def freeze_encoders(self):
        for p in self.encoder_cc.features.parameters():
            p.requires_grad = False

    def unfreeze_encoders(self):
        for p in self.encoder_cc.features.parameters():
            p.requires_grad = True

    def forward(self, img_cc, img_mlo=None):
        cc_out = self.encoder_cc(img_cc)          # uses CC view only
        return {
            "prob": cc_out["prob"],
            "uncertainty": cc_out["uncertainty"],  # K/S, same as CUED-Net
            "alpha": cc_out["alpha"],
            "cc_out": cc_out,
        }


class SingleViewEDLLoss(nn.Module):
    """Evidential loss on the single CC view. Reuses CUEDNetLoss._evidential_loss
    so the EDL objective is byte-for-byte identical to CUED-Net's per-view term
    (no VDL, no consistency)."""
    def __init__(self, num_classes=2, lambda_kl=0.1, annealing_epochs=10):
        super().__init__()
        # borrow the exact evidential-loss implementation
        self._core = CUEDNetLoss(num_classes=num_classes, lambda_vdl=0.0,
                                 lambda_kl=lambda_kl,
                                 annealing_epochs=annealing_epochs)
        self.num_classes = num_classes

    def forward(self, outputs, targets, epoch=0, class_weights=None):
        loss = self._core._evidential_loss(outputs["alpha"], targets, epoch,
                                           class_weights)
        return {"total": loss, "evidential": loss}


# =========================================================================== #
# Baseline 4: TMC (Han et al. 2021) — Dempster's rule of combination
# =========================================================================== #
def ds_combine(alpha1, alpha2, num_classes):
    """Dempster's rule of combination on two Dirichlet opinions (Han 2021, Eq. 11).

    Each view gives a Dirichlet alpha -> belief b_k = (alpha_k - 1)/S, vacuity
    u = K/S, with sum_k b_k + u = 1. Combine two opinions (b1,u1),(b2,u2):

        b_k = (1/(1-C)) * (b1_k b2_k + b1_k u2 + b2_k u1)
        u   = (1/(1-C)) *  u1 u2
        C   = sum_{i!=j} b1_i b2_j           (conflict)

    Then map back: S = K/u ; e_k = b_k * S ; alpha_k = e_k + 1.
    Returns the combined alpha (B,K).
    """
    # beliefs and uncertainty for each opinion
    S1 = torch.sum(alpha1, dim=1, keepdim=True)
    S2 = torch.sum(alpha2, dim=1, keepdim=True)
    E1, E2 = alpha1 - 1.0, alpha2 - 1.0
    b1, b2 = E1 / S1, E2 / S2                       # (B,K)
    u1, u2 = num_classes / S1, num_classes / S2     # (B,1)

    # conflict C = sum_{i != j} b1_i b2_j = (sum b1)(sum b2) - sum_k b1_k b2_k
    bb = torch.bmm(b1.unsqueeze(2), b2.unsqueeze(1))  # (B,K,K) outer product
    C = bb.sum(dim=(1, 2)) - torch.diagonal(bb, dim1=1, dim2=2).sum(dim=1)  # (B,)
    C = C.unsqueeze(1)                                # (B,1)

    denom = 1.0 - C
    denom = torch.clamp(denom, min=1e-8)             # guard total conflict

    b = (b1 * b2 + b1 * u2 + b2 * u1) / denom         # (B,K)
    u = (u1 * u2) / denom                             # (B,1)

    # map opinion back to Dirichlet: S = K/u, e = b*S, alpha = e + 1
    S = num_classes / torch.clamp(u, min=1e-8)        # (B,1)
    e = b * S
    alpha = e + 1.0
    return alpha


class TMCDualView(nn.Module):
    """Two evidential ViewEncoders fused by Dempster's rule (Han 2021).

    Returns combined Dirichlet prob + vacuity, plus per-view alphas for the
    TMC loss (per-view evidential loss + combined-opinion evidential loss).
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.encoder_cc = ViewEncoder(num_classes, pretrained)
        self.encoder_mlo = ViewEncoder(num_classes, pretrained)

    def freeze_encoders(self):
        for m in (self.encoder_cc, self.encoder_mlo):
            for p in m.features.parameters():
                p.requires_grad = False

    def unfreeze_encoders(self):
        for m in (self.encoder_cc, self.encoder_mlo):
            for p in m.features.parameters():
                p.requires_grad = True

    def forward(self, img_cc, img_mlo):
        cc_out = self.encoder_cc(img_cc)
        mlo_out = self.encoder_mlo(img_mlo)

        alpha_comb = ds_combine(cc_out["alpha"], mlo_out["alpha"],
                                self.num_classes)
        S = torch.sum(alpha_comb, dim=1, keepdim=True)
        prob = alpha_comb / S
        uncertainty = self.num_classes / S.squeeze(1)   # K/S of combined opinion

        return {
            "prob": prob,
            "uncertainty": uncertainty,
            "alpha": alpha_comb,
            "alpha_cc": cc_out["alpha"],
            "alpha_mlo": mlo_out["alpha"],
            "cc_out": cc_out,
            "mlo_out": mlo_out,
        }


class TMCLoss(nn.Module):
    """TMC objective (Han 2021): evidential loss on each view's opinion AND on
    the Dempster-combined opinion. Reuses CUEDNetLoss._evidential_loss for all
    three terms so the loss form matches CUED-Net's per-view term exactly."""
    def __init__(self, num_classes=2, lambda_kl=0.1, annealing_epochs=10):
        super().__init__()
        self._core = CUEDNetLoss(num_classes=num_classes, lambda_vdl=0.0,
                                 lambda_kl=lambda_kl,
                                 annealing_epochs=annealing_epochs)
        self.num_classes = num_classes

    def forward(self, outputs, targets, epoch=0, class_weights=None):
        l_cc = self._core._evidential_loss(outputs["alpha_cc"], targets, epoch,
                                           class_weights)
        l_mlo = self._core._evidential_loss(outputs["alpha_mlo"], targets, epoch,
                                            class_weights)
        l_comb = self._core._evidential_loss(outputs["alpha"], targets, epoch,
                                             class_weights)
        total = l_cc + l_mlo + l_comb
        return {"total": total, "ev_cc": l_cc, "ev_mlo": l_mlo, "ev_comb": l_comb}


# =========================================================================== #
# Inference wrappers (deterministic single forward; uncertainty from the model)
# =========================================================================== #
@torch.no_grad()
def edl_predict(model, img_cc, img_mlo=None):
    model.eval()
    out = model(img_cc, img_mlo)
    return {"prob": out["prob"], "uncertainty": out["uncertainty"]}


@torch.no_grad()
def tmc_predict(model, img_cc, img_mlo):
    model.eval()
    out = model(img_cc, img_mlo)
    return {"prob": out["prob"], "uncertainty": out["uncertainty"]}
