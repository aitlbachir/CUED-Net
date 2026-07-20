#!/usr/bin/env python
"""Grad-CAM uncertainty case studies (per-view saliency overlays)."""

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/cued_net")

from models.cued_net import CUEDNet
import cv_dataloaders as cv
from cv_dataloaders import get_cv_dataloaders

DECOMP_CSV = "/workspace/cued_net/selective_preds_novdl/cued_net_preds_decomposed.csv"
CKPT_ROOT = "/workspace/cued_net/cv_ablation/no_vdl"
DATA_ROOT = "/workspace/cbis-ddsm"
FOLDS_JSON = "/workspace/cued_net/cv_folds.json"
OUT = Path("/workspace/cued_net/gradcam_cases")

# component column -> (display name, selection column)
CASES = {
    "high_uevid": ("High Evidential Uncertainty", "uncertainty_evidential"),
    "high_udisc": ("High View Discordance",        "uncertainty_discordance"),
}


# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """Grad-CAM on a single ViewEncoder, hooking features.denseblock4."""
    def __init__(self, encoder):
        self.encoder = encoder
        self.target = encoder.features.denseblock4
        self.acts = None
        self.grads = None
        self.h1 = self.target.register_forward_hook(self._fwd)
        self.h2 = self.target.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o): self.acts = o.detach()
    def _bwd(self, m, gi, go): self.grads = go[0].detach()

    def remove(self):
        self.h1.remove(); self.h2.remove()

    def __call__(self, img):
        """img: (1,3,224,224). Returns (cam[224,224] in [0,1], out_dict)."""
        self.encoder.zero_grad()
        out = self.encoder(img)
        # gradient target: malignant-class evidence (softplus evidence, class 1)
        target = out["evidence"][0, 1]
        target.backward(retain_graph=False)
        # Grad-CAM: GAP of grads -> channel weights -> weighted sum of acts
        w = self.grads.mean(dim=(2, 3), keepdim=True)        # (1,C,1,1)
        cam = F.relu((w * self.acts).sum(dim=1)).squeeze(0)   # (h,w)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam[None, None], size=(224, 224),
                            mode="bilinear", align_corners=False)[0, 0]
        return cam.cpu().numpy(), out


def load_model(seed, fold, device):
    ckpt = Path(CKPT_ROOT) / f"fold_{fold}" / f"seed_{seed}" / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    model = CUEDNet(num_classes=2, pretrained=False).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def find_case_pair(patient_id, fold):
    """Locate the dataset pair (cc/mlo paths + normalized tensors) for a
    patient in a given fold's val set. Uses the eval transform (no aug)."""
    # eval-view dataset to get raw paths
    ds_eval = cv._build_full_cohort_dataset(DATA_ROOT, cv._eval_transform(224))
    # locate pair by patient_id (val pairs are patient-disjoint within a fold,
    # but a patient_id could in principle recur — match within this fold's val)
    folds_rec = json.load(open(FOLDS_JSON))
    fold_rec = folds_rec["folds"][fold]
    fpairs = folds_rec["pairs"]
    ds_pos = {cv._content_key(p): i for i, p in enumerate(ds_eval.pairs)}
    val_pos = []
    for fi in fold_rec["val_idx"]:
        k = cv._content_key(fpairs[fi])
        val_pos.append(ds_pos[k])
    # find the patient among this fold's val positions
    matches = [i for i in val_pos if ds_eval.pairs[i]["patient_id"] == patient_id]
    if not matches:
        raise ValueError(f"patient {patient_id} not in fold {fold} val set")
    pos = matches[0]
    item = ds_eval[pos]   # normalized tensors via eval transform
    pair = ds_eval.pairs[pos]
    return item, pair


def denorm(t):
    """[-1,1] tensor (3,224,224) -> [0,1] HxW grayscale for display background."""
    x = t.clone()
    x = (x + 1.0) / 2.0
    x = x.clamp(0, 1).mean(0)   # to grayscale
    return x.cpu().numpy()


def render_panel(case_key, title, pid, label, cc_img, mlo_img, cc_cam, mlo_cam,
                 pred, prob_mal, u_evid, u_disc, u_comb, action, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2))
    for ax, img, cam, vname in [(axes[0], cc_img, cc_cam, "CC"),
                                (axes[1], mlo_img, mlo_cam, "MLO")]:
        ax.imshow(img, cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.set_title(f"{vname} view", fontsize=12)
        ax.axis("off")

    truth = "Malignant" if label == 1 else "Benign"
    pred_s = "Malignant" if pred == 1 else "Benign"
    correct = "OK" if pred == label else "X"
    sup = (f"{title}  |  patient {pid}\n"
           f"Truth: {truth}    Pred: {pred_s} (p_mal={prob_mal:.2f}) {correct}    "
           f"u_evid={u_evid:.3f}  u_disc={u_disc:.3f}  u_total={u_comb:.3f}\n"
           f"Clinical action: {action}")
    fig.suptitle(sup, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(smoke):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DECOMP_CSV)
    df["patient_id"] = df["patient_id"].astype(str)

    keys = ["high_uevid"] if smoke else list(CASES.keys())
    summary = []

    for case_key in keys:
        title, sel_col = CASES[case_key]
        row = df.loc[df[sel_col].idxmax()]
        pid = str(row["patient_id"]); seed = int(row["seed"]); fold = int(row["fold"])
        print(f"\n=== {case_key}: max {sel_col}={row[sel_col]:.4f} "
              f"-> patient {pid}, seed {seed}, fold {fold} ===")

        model = load_model(seed, fold, device)
        item, pair = find_case_pair(pid, fold)
        img_cc = item["img_cc"].unsqueeze(0).to(device)
        img_mlo = item["img_mlo"].unsqueeze(0).to(device)
        label = int(item["label"])

        # full-model forward for the authoritative uncertainty values
        with torch.no_grad():
            full = model(img_cc, img_mlo)
        u_evid = float(full["uncertainty_evidential"][0])
        u_disc = float(full["uncertainty_discordance"][0])
        u_comb = float(full["uncertainty_combined"][0])
        prob_mal = float(full["prob"][0, 1])
        pred = int(full["pred"][0])

        csv_uc = float(row["uncertainty_combined"])
        if abs(u_comb - csv_uc) > 1e-3:
            print(f"  [WARN] u_combined live={u_comb:.4f} vs csv={csv_uc:.4f} "
                  f"(Δ={abs(u_comb-csv_uc):.2e}) — checkpoint/patient mismatch?")
        else:
            print(f"  [Δ-gate] u_combined live={u_comb:.4f} == csv {csv_uc:.4f} OK")

        # Grad-CAM per view (needs grad, so separate from the no_grad forward)
        cam_cc_engine = GradCAM(model.encoder_cc)
        cc_cam, _ = cam_cc_engine(img_cc)
        cam_cc_engine.remove()

        cam_mlo_engine = GradCAM(model.encoder_mlo)
        mlo_cam, _ = cam_mlo_engine(img_mlo)
        cam_mlo_engine.remove()

        cc_disp = denorm(item["img_cc"])
        mlo_disp = denorm(item["img_mlo"])

        # clinical action text per source
        if case_key == "high_uevid":
            action = ("Model lacks evidence — expert review; lesion features may be "
                      "atypical/unclear.")
        else:
            action = ("CC/MLO views disagree — recommend additional imaging "
                      "(spot compression / US) per discordant-view caution.")

        out_path = OUT / f"case_{case_key}_{pid}.png"
        render_panel(case_key, title, pid, label, cc_disp, mlo_disp,
                     cc_cam, mlo_cam, pred, prob_mal, u_evid, u_disc, u_comb,
                     action, out_path)
        print(f"  panel -> {out_path}")

        summary.append({
            "case": case_key, "title": title, "patient_id": pid,
            "seed": seed, "fold": fold, "label": label, "pred": pred,
            "prob_malignant": prob_mal, "u_evid": u_evid, "u_disc": u_disc,
            "u_combined": u_comb, "panel": str(out_path),
        })
        del model
        if device.type == "cuda": torch.cuda.empty_cache()

    json.dump(summary, open(OUT / "gradcam_summary.json", "w"), indent=2)
    print(f"\n[done] {len(summary)} case(s) -> {OUT}")
    if smoke:
        print("SMOKE OK — pipeline ran end-to-end; check the panel PNG visually, "
              "then run --full.")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--full", action="store_true")
    args = ap.parse_args()
    run(smoke=args.smoke)


if __name__ == "__main__":
    main()
