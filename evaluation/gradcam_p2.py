#!/usr/bin/env python
"""Grad-CAM against lesion masks, for every outer-fold case and seed.

Runs on the cross-validation models, so the held-out split is not read. For each
view: share of saliency energy inside the mask, the same share divided by the
mask area (1 means no concentration), and whether the peak falls inside the mask.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.p2_data import ImageCache, PairDataset, build_splits, eval_transform, load_manifest
from training.p2_config import OUTER_FOLDS, SEEDS
from training.train_p2 import build


def grad_cam(encoder, x, size=224):
    """Saliency for the predicted class evidence of one view, shape (B, H, W)."""
    store = {}
    hook = encoder.features.register_forward_hook(lambda _m, _i, o: store.__setitem__("a", o))
    out = encoder(x)
    hook.remove()
    cls = out["prob"].argmax(1, keepdim=True)
    grads = torch.autograd.grad(out["evidence"].gather(1, cls).sum(), store["a"])[0]
    weights = grads.mean((2, 3), keepdim=True)
    cam = F.relu((weights * store["a"]).sum(1, keepdim=True))
    return F.interpolate(cam, (size, size), mode="bilinear", align_corners=False)[:, 0].detach()


def mask_metrics(cam, mask):
    """cam, mask: (H, W) arrays. Returns None-free metrics, nan for an empty map."""
    area = float((mask > 0).mean())
    total = float(cam.sum())
    if total <= 0 or area == 0:
        return dict(energy=np.nan, lift=np.nan, hit=np.nan, area=area)
    energy = float(cam[mask > 0].sum() / total)
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return dict(energy=energy, lift=energy / area, hit=float(mask[peak] > 0), area=area)


def add_strata(cases, run_dir):
    """Correctness at 0.5 and evidential tercile, cut points from inner validation."""
    iv = pd.read_csv(run_dir / "inner_val.csv")
    ov = pd.read_csv(run_dir / "outer_val.csv")
    cuts = np.quantile(0.5 * (iv.u_cc + iv.u_mlo), [1 / 3, 2 / 3])
    ov["u_evid"] = 0.5 * (ov.u_cc + ov.u_mlo)
    ov["correct"] = ((ov.p >= 0.5).astype(int) == ov.label).astype(int)
    ov["tercile"] = np.digitize(ov.u_evid, cuts)
    return cases.merge(ov[["pair_key", "label", "correct", "tercile"]], on="pair_key")


def clustered_mean(df, column, B, seed):
    """Mean of per-seed means with a patient bootstrap."""
    sub = df.dropna(subset=[column])
    if sub.empty:
        return dict(value=float("nan"), ci95=[float("nan"), float("nan")], n=0)
    point = sub.groupby("seed")[column].mean().mean()
    by_patient = {p: g.groupby("seed")[column].agg(["sum", "count"]) for p, g in sub.groupby("patient_id")}
    keys = list(by_patient)
    rng = np.random.default_rng(seed)
    seeds = sorted(sub.seed.unique())
    S = np.stack([by_patient[k].reindex(seeds).fillna(0)["sum"].values for k in keys])
    C = np.stack([by_patient[k].reindex(seeds).fillna(0)["count"].values for k in keys])
    draws = []
    for _ in range(B):
        idx = rng.integers(0, len(keys), len(keys))
        with np.errstate(invalid="ignore", divide="ignore"):
            draws.append(np.nanmean(S[idx].sum(0) / C[idx].sum(0)))
    return dict(value=float(point), ci95=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], n=int(len(sub)))


def summarise(df, B=2000, seed=20261103):
    groups = {"all": df, "correct": df[df.correct == 1], "error": df[df.correct == 0]}
    groups.update({f"tercile_{t}": df[df.tercile == t] for t in (0, 1, 2)})
    groups.update({f"view_{v}": df[df.view == v] for v in ("cc", "mlo")})
    return {name: {c: clustered_mean(g, c, B, seed) for c in ("energy", "lift", "hit", "area")} for name, g in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--patches", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="cued_net")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.manifest)
    pairs, splits = manifest["pairs"], build_splits(manifest)
    report = pd.read_csv(Path(args.masks) / "masks_report.csv").set_index("file")
    cache = ImageCache(args.patches)
    cache.preload(pairs)
    rows = []
    for fold in OUTER_FOLDS:
        data = DataLoader(PairDataset(pairs, splits[fold]["outer_val"], cache, eval_transform()), batch_size=32, num_workers=2)
        for seed in SEEDS:
            run = Path(args.runs) / args.method / f"fold_{fold}" / f"seed_{seed}"
            model, _ = build(args.method, pretrained=False)
            model.load_state_dict(torch.load(run / "best_model.pt", map_location="cpu", weights_only=False)["model_state_dict"])
            model = model.to(device).eval()
            found = []
            for batch in data:
                for view, encoder, x in (("cc", model.encoder_cc, batch["img_cc"]), ("mlo", model.encoder_mlo, batch["img_mlo"])):
                    cams = grad_cam(encoder, x.to(device)).cpu().numpy()
                    for cam, idx in zip(cams, batch["index"].tolist()):
                        name = pairs[idx][view]
                        if report.status.get(name) != "aligned":
                            continue
                        mask = np.asarray(Image.open(Path(args.masks) / name))
                        found.append(dict(fold=fold, seed=seed, pair_key=pairs[idx]["key"], patient_id=pairs[idx]["patient_id"],
                                          view=view, **mask_metrics(cam, mask)))
            rows.append(add_strata(pd.DataFrame(found), run))
            print("grad-cam", fold, seed, len(found), flush=True)
    cases = pd.concat(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases.to_csv(out / "gradcam_cases.csv", index=False)
    result = dict(n_rows=len(cases), n_empty_maps=int(cases.energy.isna().sum()), summary=summarise(cases),
                  n_views_by_mask_status=report.status.value_counts().to_dict())
    json.dump(result, open(out / "gradcam_summary.json", "w"), indent=1)
    print(json.dumps(result["summary"]["all"], indent=1))


if __name__ == "__main__":
    main()
