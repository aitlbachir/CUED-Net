"""CUED-Net: dual-encoder evidential network with view-discordance uncertainty."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


FEATURE_DIMS = {
    "densenet121": 1024,
    "resnet50": 2048,
    "efficientnet_b0": 1280,
    "convnext_tiny": 768,
}


def build_backbone(name, pretrained=True):
    """Return a convolutional stack ending in (B, C, H, W)."""
    if name == "densenet121":
        net = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        return net.features
    if name == "resnet50":
        net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        return nn.Sequential(*list(net.children())[:-2])
    if name == "efficientnet_b0":
        net = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        return net.features
    if name == "convnext_tiny":
        net = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        return net.features
    raise ValueError(f"unknown backbone: {name}")


class EvidentialLayer(nn.Module):
    """Map a hidden vector to Dirichlet parameters."""

    def __init__(self, in_features, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Dirichlet quantities stay in float32 under mixed precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            evidence = F.softplus(self.fc(x.float()))
            alpha = evidence + 1.0
            S = torch.sum(alpha, dim=1, keepdim=True)
            return {
                "evidence": evidence,
                "alpha": alpha,
                "prob": alpha / S,
                "uncertainty": self.num_classes / S.squeeze(1),
                "strength": S.squeeze(1),
            }


class ViewEncoder(nn.Module):
    """Single-view encoder with an evidential head."""

    def __init__(self, num_classes=2, pretrained=True, hidden_dim=256, backbone="densenet121"):
        super().__init__()
        self.features = build_backbone(backbone, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(FEATURE_DIMS[backbone], hidden_dim),
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
        features = self.pool(self.features(x)).flatten(1)
        output = self.evidential(self.classifier(features))
        output["features"] = features
        return output


def symmetric_kl(p, q, eps=1e-8):
    """Symmetric KL divergence between two categorical distributions."""
    p, q = p.clamp_min(eps), q.clamp_min(eps)
    return 0.5 * ((p * (p / q).log()).sum(1) + (q * (q / p).log()).sum(1))


def fuse_views(cc_out, mlo_out):
    """Weight each view by one minus its evidential uncertainty."""
    conf_cc = 1.0 - cc_out["uncertainty"]
    conf_mlo = 1.0 - mlo_out["uncertainty"]
    total = conf_cc + conf_mlo + 1e-8
    w_cc, w_mlo = conf_cc / total, conf_mlo / total
    prob = w_cc.unsqueeze(1) * cc_out["prob"] + w_mlo.unsqueeze(1) * mlo_out["prob"]
    return prob, w_cc, w_mlo


class CUEDNet(nn.Module):
    """Two independent evidential encoders with uncertainty-weighted fusion."""

    def __init__(self, num_classes=2, pretrained=True, backbone="densenet121"):
        super().__init__()
        self.num_classes = num_classes
        self.encoder_cc = ViewEncoder(num_classes, pretrained, backbone=backbone)
        self.encoder_mlo = ViewEncoder(num_classes, pretrained, backbone=backbone)

    def forward(self, img_cc, img_mlo):
        cc_out = self.encoder_cc(img_cc)
        mlo_out = self.encoder_mlo(img_mlo)
        d_sym = symmetric_kl(cc_out["prob"], mlo_out["prob"])
        prob, w_cc, w_mlo = fuse_views(cc_out, mlo_out)
        return {
            "prob": prob,
            "pred": torch.argmax(prob, dim=1),
            "cc_out": cc_out,
            "mlo_out": mlo_out,
            "uncertainty_evidential": 0.5 * (cc_out["uncertainty"] + mlo_out["uncertainty"]),
            "uncertainty_discordance": 1.0 - torch.exp(-d_sym),
            "d_sym": d_sym,
            "fusion_weights": {"cc": w_cc, "mlo": w_mlo},
        }

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


def dirichlet_kl_to_uniform(alpha):
    """KL( Dir(alpha) || Dir(1) )."""
    K = alpha.shape[1]
    a0 = alpha.sum(dim=1)
    log_gamma_k = torch.lgamma(torch.tensor(float(K), device=alpha.device))
    log_norm = torch.lgamma(a0) - torch.lgamma(alpha).sum(1) - log_gamma_k
    return log_norm + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(a0).unsqueeze(1))).sum(1)


def r1_kl_term(alpha):
    """The R1 regulariser, kept only for experiment E6.

    The log-normaliser has the opposite sign to the true KL divergence, so the
    term goes negative once the target concentrates.
    """
    K = alpha.shape[1]
    a0 = alpha.sum(dim=1)
    log_gamma_k = torch.lgamma(torch.tensor(float(K), device=alpha.device))
    log_norm = torch.lgamma(alpha).sum(1) - torch.lgamma(a0) + log_gamma_k
    return log_norm + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(a0).unsqueeze(1))).sum(1)


def kl_target(alpha, y_onehot, mode="standard"):
    """Dirichlet parameters used by the KL regulariser.

    standard: y + (1 - y) * alpha, so wrong-class evidence is penalised and the
              target reaches Dir(1) as evidence vanishes (Sensoy et al., 2018).
    r1:       y + (1 - y) * (alpha - 1) clamped at 1, the R1 variant. Kept only
              for experiment E6.
    """
    if mode == "standard":
        return y_onehot + (1.0 - y_onehot) * alpha
    if mode == "r1":
        return torch.clamp(y_onehot + (1.0 - y_onehot) * (alpha - 1.0), min=1.0)
    raise ValueError(mode)


def evidential_loss(alpha, targets, epoch, lambda_kl=0.1, annealing_epochs=10, kl_mode="standard"):
    """Expected squared error under Dir(alpha) with a KL regulariser (per-sample mean)."""
    K = alpha.shape[1]
    S = alpha.sum(dim=1, keepdim=True)
    prob = alpha / S
    y = F.one_hot(targets, K).float()
    mse = ((y - prob) ** 2).sum(1)
    var = (prob * (1.0 - prob) / (S + 1.0)).sum(1)
    anneal = min(1.0, epoch / annealing_epochs)
    target = kl_target(alpha, y, kl_mode)
    kl = r1_kl_term(target) if kl_mode == "r1" else dirichlet_kl_to_uniform(target)
    return (mse + var + lambda_kl * anneal * kl).mean()


class CUEDNetLoss(nn.Module):
    """Sum of per-view evidential losses plus an annealed consistency term.

    lambda_vdl is kept at 0 in the final model. It exists for the ablation only.
    """

    def __init__(self, lambda_kl=0.1, lambda_consistency=0.1, lambda_vdl=0.0,
                 annealing_epochs=10, kl_mode="standard"):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_consistency = lambda_consistency
        self.lambda_vdl = lambda_vdl
        self.annealing_epochs = annealing_epochs
        self.kl_mode = kl_mode

    def forward(self, outputs, targets, epoch):
        cc, mlo = outputs["cc_out"], outputs["mlo_out"]
        kw = dict(lambda_kl=self.lambda_kl, annealing_epochs=self.annealing_epochs, kl_mode=self.kl_mode)
        loss_ev = evidential_loss(cc["alpha"], targets, epoch, **kw) + evidential_loss(mlo["alpha"], targets, epoch, **kw)
        anneal = min(1.0, epoch / self.annealing_epochs)
        total = loss_ev + self.lambda_consistency * anneal * F.mse_loss(cc["prob"], mlo["prob"])
        if self.lambda_vdl > 0:
            disagree = (cc["prob"].argmax(1) != mlo["prob"].argmax(1)).float()
            vdl = (disagree * (1 - cc["uncertainty"]) * (1 - mlo["uncertainty"])).mean()
            total = total + self.lambda_vdl * anneal * vdl
        return {"total": total, "evidential": loss_ev}
