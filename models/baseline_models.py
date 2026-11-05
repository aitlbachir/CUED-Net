"""Comparator models: each takes the same encoder backbone as the CUED-Net it is compared with."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cued_net import FEATURE_DIMS, ViewEncoder, evidential_loss


class _EncoderControl:
    def _encoders(self):
        return (self.encoder_cc, self.encoder_mlo)

    def freeze_encoders(self):
        for enc in self._encoders():
            for p in enc.features.parameters():
                p.requires_grad = False

    def unfreeze_encoders(self):
        for enc in self._encoders():
            for p in enc.features.parameters():
                p.requires_grad = True


class SoftmaxViewEncoder(nn.Module):
    """ViewEncoder trunk with a softmax head."""

    def __init__(self, num_classes=2, pretrained=True, backbone="densenet121"):
        super().__init__()
        base = ViewEncoder(num_classes=num_classes, pretrained=pretrained, backbone=backbone)
        self.features = base.features
        self.pool = base.pool
        self.classifier = base.classifier
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        f = self.pool(self.features(x)).flatten(1)
        return self.head(self.classifier(f))


class SoftmaxDualView(nn.Module, _EncoderControl):
    """Two softmax encoders, late fusion by averaging probabilities.

    One trained model serves as the deterministic softmax control, as the
    MC-Dropout model (T stochastic passes), and as a Deep-Ensemble member.
    """

    def __init__(self, num_classes=2, pretrained=True, backbone="densenet121"):
        super().__init__()
        self.encoder_cc = SoftmaxViewEncoder(num_classes, pretrained, backbone)
        self.encoder_mlo = SoftmaxViewEncoder(num_classes, pretrained, backbone)

    def forward(self, img_cc, img_mlo):
        logit_cc, logit_mlo = self.encoder_cc(img_cc), self.encoder_mlo(img_mlo)
        p_cc, p_mlo = F.softmax(logit_cc, dim=1), F.softmax(logit_mlo, dim=1)
        return {"prob": 0.5 * (p_cc + p_mlo), "logit_cc": logit_cc, "logit_mlo": logit_mlo,
                "prob_cc": p_cc, "prob_mlo": p_mlo}


class SoftmaxDualLoss(nn.Module):
    def forward(self, outputs, targets, epoch):
        loss = 0.5 * (F.cross_entropy(outputs["logit_cc"], targets) + F.cross_entropy(outputs["logit_mlo"], targets))
        return {"total": loss}


class SingleViewEDL(nn.Module):
    """One evidential CC encoder. img_mlo is ignored."""

    def __init__(self, num_classes=2, pretrained=True, backbone="densenet121"):
        super().__init__()
        self.encoder_cc = ViewEncoder(num_classes, pretrained, backbone=backbone)

    def forward(self, img_cc, img_mlo=None):
        out = self.encoder_cc(img_cc)
        return {"prob": out["prob"], "uncertainty": out["uncertainty"], "alpha": out["alpha"]}

    def freeze_encoders(self):
        for p in self.encoder_cc.features.parameters():
            p.requires_grad = False

    def unfreeze_encoders(self):
        for p in self.encoder_cc.features.parameters():
            p.requires_grad = True


class SingleViewEDLLoss(nn.Module):
    def __init__(self, lambda_kl=0.1, annealing_epochs=10):
        super().__init__()
        self.kw = dict(lambda_kl=lambda_kl, annealing_epochs=annealing_epochs)

    def forward(self, outputs, targets, epoch):
        return {"total": evidential_loss(outputs["alpha"], targets, epoch, **self.kw)}


def ds_combine(alpha1, alpha2, num_classes):
    """Dempster's rule on two Dirichlet opinions (Han et al., 2021)."""
    S1, S2 = alpha1.sum(1, keepdim=True), alpha2.sum(1, keepdim=True)
    b1, b2 = (alpha1 - 1.0) / S1, (alpha2 - 1.0) / S2
    u1, u2 = num_classes / S1, num_classes / S2
    outer = torch.bmm(b1.unsqueeze(2), b2.unsqueeze(1))
    conflict = (outer.sum(dim=(1, 2)) - torch.diagonal(outer, dim1=1, dim2=2).sum(1)).unsqueeze(1)
    denom = torch.clamp(1.0 - conflict, min=1e-8)
    b = (b1 * b2 + b1 * u2 + b2 * u1) / denom
    u = u1 * u2 / denom
    S = num_classes / torch.clamp(u, min=1e-8)
    return b * S + 1.0


class TMCDualView(nn.Module, _EncoderControl):
    """Two evidential encoders fused by Dempster's rule."""

    def __init__(self, num_classes=2, pretrained=True, backbone="densenet121"):
        super().__init__()
        self.num_classes = num_classes
        self.encoder_cc = ViewEncoder(num_classes, pretrained, backbone=backbone)
        self.encoder_mlo = ViewEncoder(num_classes, pretrained, backbone=backbone)

    def forward(self, img_cc, img_mlo):
        cc, mlo = self.encoder_cc(img_cc), self.encoder_mlo(img_mlo)
        alpha = ds_combine(cc["alpha"], mlo["alpha"], self.num_classes)
        S = alpha.sum(1, keepdim=True)
        return {"prob": alpha / S, "uncertainty": self.num_classes / S.squeeze(1), "alpha": alpha,
                "alpha_cc": cc["alpha"], "alpha_mlo": mlo["alpha"]}


class TMCLoss(nn.Module):
    """Evidential loss on each view's opinion and on the combined opinion."""

    def __init__(self, lambda_kl=0.1, annealing_epochs=10):
        super().__init__()
        self.kw = dict(lambda_kl=lambda_kl, annealing_epochs=annealing_epochs)

    def forward(self, outputs, targets, epoch):
        total = sum(evidential_loss(outputs[k], targets, epoch, **self.kw) for k in ("alpha_cc", "alpha_mlo", "alpha"))
        return {"total": total}


class CrossViewBlock(nn.Module):
    """Cross-attention between two views at feature-map level (van Tulder et al., 2021)."""

    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(4 * dim, dim))

    def forward(self, query, key):
        key = self.norm_k(key)
        q = query + self.attn(self.norm_q(query), key, key, need_weights=False)[0]
        return q + self.ff(self.norm_ff(q))


class CrossViewTransformer(nn.Module, _EncoderControl):
    """Two encoders exchanging information through one cross-view block.

    Softmax outputs and late fusion match SoftmaxDualView, so the comparison
    isolates the effect of learned cross-view attention.
    """

    def __init__(self, num_classes=2, pretrained=True, dim=256, grid=7, backbone="densenet121"):
        super().__init__()
        self.encoder_cc = SoftmaxViewEncoder(num_classes, pretrained, backbone)
        self.encoder_mlo = SoftmaxViewEncoder(num_classes, pretrained, backbone)
        self.proj = nn.Conv2d(FEATURE_DIMS[backbone], dim, 1)
        self.pos_cc = nn.Parameter(torch.zeros(1, grid * grid, dim))
        self.pos_mlo = nn.Parameter(torch.zeros(1, grid * grid, dim))
        nn.init.trunc_normal_(self.pos_cc, std=0.02)
        nn.init.trunc_normal_(self.pos_mlo, std=0.02)
        self.block_cc = CrossViewBlock(dim)
        self.block_mlo = CrossViewBlock(dim)
        self.back = nn.Linear(dim, FEATURE_DIMS[backbone])

    def forward(self, img_cc, img_mlo):
        f_cc, f_mlo = self.encoder_cc.features(img_cc), self.encoder_mlo.features(img_mlo)
        t_cc = self.proj(f_cc).flatten(2).transpose(1, 2) + self.pos_cc
        t_mlo = self.proj(f_mlo).flatten(2).transpose(1, 2) + self.pos_mlo
        x_cc, x_mlo = self.block_cc(t_cc, t_mlo), self.block_mlo(t_mlo, t_cc)
        h_cc = self.encoder_cc.pool(f_cc).flatten(1) + self.back(x_cc.mean(1))
        h_mlo = self.encoder_mlo.pool(f_mlo).flatten(1) + self.back(x_mlo.mean(1))
        logit_cc = self.encoder_cc.head(self.encoder_cc.classifier(h_cc))
        logit_mlo = self.encoder_mlo.head(self.encoder_mlo.classifier(h_mlo))
        p_cc, p_mlo = F.softmax(logit_cc, dim=1), F.softmax(logit_mlo, dim=1)
        return {"prob": 0.5 * (p_cc + p_mlo), "logit_cc": logit_cc, "logit_mlo": logit_mlo,
                "prob_cc": p_cc, "prob_mlo": p_mlo}


def enable_dropout(model):
    """Keep dropout active and batch norm frozen."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model, img_cc, img_mlo, T=50):
    model.eval()
    enable_dropout(model)
    probs = torch.stack([model(img_cc, img_mlo)["prob"] for _ in range(T)])
    model.eval()
    mean = probs.mean(0).clamp_min(1e-8)
    return mean[:, 1], -(mean * mean.log()).sum(1)
