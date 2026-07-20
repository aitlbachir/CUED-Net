#!/usr/bin/env python
"""
calibrate_cmmd.py — Calibration analysis for CUED-Net on CMMD.

WHAT IT REPORTS (per seed, for both zero-shot and fine-tuned models, on the
exact held-out test split saved in each finetune_seed*.json):
  - ECE (Expected Calibration Error, 15 bins)
  - MCE (Maximum Calibration Error)
  - Brier score
  - NLL (negative log-likelihood)
  - reliability-diagram data (bin confidences + accuracies + counts)
  - temperature scaling: T fit on the VAL split, ECE before/after on TEST

This answers the plan's calibration must-have (ECE, reliability diagrams,
temperature scaling) AND the Week-6 "ECE drift" between zero-shot and adapted.

It is pure inference on existing checkpoints — no training. Splits are read
from the saved JSONs so calibration runs on identical samples to the AUC.

USAGE (one seed):
    python calibrate_cmmd.py \
        --manifest      /workspace/cued_net/cmmd_pairs_cropped.json \
        --zs_ckpt       /workspace/outputs_cued/seed_42/best_model.pt \
        --ft_ckpt       /workspace/cued_net/finetune_out_25pct/cued_cmmd_ft_seed42.pt \
        --split_json    /workspace/cued_net/finetune_out_25pct/finetune_seed42.json \
        --models_dir    /workspace/cued_net \
        --out           /workspace/cued_net/calib_out/calib_seed42.json

Loop over seeds with a shell for-loop (see message).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


# --------------------------------------------------------------------------- #
# Calibration metrics
# --------------------------------------------------------------------------- #
def calibration_metrics(probs_mal, labels, n_bins=15):
    """probs_mal: P(malignant) in [0,1]; labels: 0/1.
    Confidence = max(p, 1-p); correct = (argmax == label)."""
    probs_mal = np.asarray(probs_mal, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    preds = (probs_mal >= 0.5).astype(np.int64)
    conf = np.maximum(probs_mal, 1.0 - probs_mal)   # confidence of the predicted class
    correct = (preds == labels).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce = 0.0, 0.0
    bin_conf, bin_acc, bin_count = [], [], []
    N = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        c = m.sum()
        if c == 0:
            bin_conf.append(None); bin_acc.append(None); bin_count.append(0)
            continue
        avg_conf = conf[m].mean()
        avg_acc = correct[m].mean()
        gap = abs(avg_conf - avg_acc)
        ece += (c / N) * gap
        mce = max(mce, gap)
        bin_conf.append(round(float(avg_conf), 4))
        bin_acc.append(round(float(avg_acc), 4))
        bin_count.append(int(c))

    # Brier and NLL use P(malignant) vs the binary label
    p = np.clip(probs_mal, 1e-7, 1 - 1e-7)
    brier = np.mean((p - labels) ** 2)
    nll = -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "brier": round(float(brier), 4),
        "nll": round(float(nll), 4),
        "reliability": {"bin_conf": bin_conf, "bin_acc": bin_acc, "bin_count": bin_count,
                        "n_bins": n_bins},
    }


# --------------------------------------------------------------------------- #
# Temperature scaling: fit T on logits to minimize val NLL, apply to test.
# We reconstruct a 2-logit from P(mal): logit = log(p/(1-p)) for class-1,
# and 0 for class-0 reference -> equivalently scale the malignant log-odds.
# --------------------------------------------------------------------------- #
def fit_temperature(val_probs_mal, T_min=0.05, T_max=20.0):
    """Fit scalar T in [T_min, T_max] minimizing val NLL on the log-odds.
    Uses a bounded grid + local refine — robust on small, flat-NLL val sets
    where gradient optimizers (LBFGS) diverge to T->0."""
    p = np.clip(np.asarray(val_probs_mal["probs"]), 1e-7, 1 - 1e-7)
    y = np.asarray(val_probs_mal["labels"], dtype=np.float64)
    logit = np.log(p / (1 - p))

    def nll_at(T):
        scaled = np.clip(logit / T, -30, 30)      # guard overflow
        q = 1.0 / (1.0 + np.exp(-scaled))
        q = np.clip(q, 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    # coarse grid (log-spaced), then fine grid around the best point
    grid = np.geomspace(T_min, T_max, 60)
    nlls = [nll_at(T) for T in grid]
    T0 = grid[int(np.argmin(nlls))]
    fine = np.linspace(max(T_min, T0 * 0.7), min(T_max, T0 * 1.4), 60)
    T = fine[int(np.argmin([nll_at(T) for T in fine]))]
    return float(T)


def apply_temperature(probs_mal, T):
    p = np.clip(np.asarray(probs_mal), 1e-7, 1 - 1e-7)
    logit = np.log(p / (1 - p))
    scaled = np.clip(logit / T, -30, 30)
    return 1.0 / (1.0 + np.exp(-scaled))   # sigmoid


# --------------------------------------------------------------------------- #
# Inference: collect P(malignant) + labels for a given subset
# --------------------------------------------------------------------------- #
def collect_predictions(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            cc = batch["img_cc"].to(device)
            mlo = batch["img_mlo"].to(device)
            out = model(cc, mlo)
            probs.append(out["prob"][:, 1].cpu().numpy())
            labels.append(batch["label"].numpy())
    return {"probs": np.concatenate(probs).tolist(),
            "labels": np.concatenate(labels).tolist()}


def subset_loader(full_ds, manifest_pairs, patient_set, batch_size, collate):
    idx = [i for i, p in enumerate(manifest_pairs) if p["patient_id"] in patient_set]
    return DataLoader(Subset(full_ds, idx), batch_size=batch_size, shuffle=False,
                      collate_fn=collate, num_workers=2), len(idx)


# --------------------------------------------------------------------------- #
def load_model(ckpt_path, CUEDNet, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"] if "model_state_dict" in ck else ck
    m = CUEDNet(num_classes=2)
    m.load_state_dict(state)
    return m.to(device)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sys.path.insert(0, args.models_dir)
    from cmmd_pair_dataset import CMMDPairDataset, cmmd_collate_fn
    from models.cued_net import CUEDNet

    manifest = json.load(open(args.manifest))
    pairs = manifest["pairs"] if isinstance(manifest, dict) and "pairs" in manifest else manifest
    splits = json.load(open(args.split_json))["patient_splits"]
    val_set, test_set = set(splits["val"]), set(splits["test"])

    full_ds = CMMDPairDataset(args.manifest)
    val_dl, n_val = subset_loader(full_ds, pairs, val_set, args.batch_size, cmmd_collate_fn)
    test_dl, n_test = subset_loader(full_ds, pairs, test_set, args.batch_size, cmmd_collate_fn)
    print(f"[calib] val pairs={n_val}  test pairs={n_test}")

    result = {"seed": args.seed, "n_val": n_val, "n_test": n_test}

    for tag, ckpt in [("zero_shot", args.zs_ckpt), ("finetuned", args.ft_ckpt)]:
        print(f"\n[calib] === {tag} ({ckpt}) ===")
        model = load_model(ckpt, CUEDNet, device)

        test_pred = collect_predictions(model, test_dl, device)
        val_pred = collect_predictions(model, val_dl, device)

        # raw calibration on test
        raw = calibration_metrics(test_pred["probs"], test_pred["labels"], args.n_bins)
        # temperature fit on val, applied to test
        T = fit_temperature(val_pred)
        ts_probs = apply_temperature(test_pred["probs"], T)
        scaled = calibration_metrics(ts_probs, test_pred["labels"], args.n_bins)

        print(f"  raw  ECE={raw['ece']:.4f} MCE={raw['mce']:.4f} "
              f"Brier={raw['brier']:.4f} NLL={raw['nll']:.4f}")
        print(f"  T={T:.3f} -> ECE={scaled['ece']:.4f} (was {raw['ece']:.4f})")

        result[tag] = {
            "raw": raw,
            "temperature": round(T, 4),
            "after_temperature_scaling": scaled,
            # keep per-sample for later pooled reliability diagram if needed
            "test_probs": [round(x, 5) for x in test_pred["probs"]],
            "test_labels": test_pred["labels"],
        }

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"))
    print(f"\n[calib] saved -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--zs_ckpt", required=True, help="zero-shot (original CBIS) checkpoint")
    ap.add_argument("--ft_ckpt", required=True, help="fine-tuned CMMD checkpoint")
    ap.add_argument("--split_json", required=True, help="finetune_seed*.json with patient_splits")
    ap.add_argument("--models_dir", default="/workspace/cued_net")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
