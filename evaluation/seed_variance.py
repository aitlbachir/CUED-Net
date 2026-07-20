#!/usr/bin/env python
"""
seed_variance.py — Single-seed vs ensemble F1 std (R3.2, R2.11).

For each model: compute per-seed pooled F1 (across folds), then the std across
the 5 seeds. This quantifies seed-induced variance. For CUED-Net we contrast the
single-model per-seed F1 std against the deep-ensemble's stability.

GPU: NOT REQUIRED.
"""
import argparse, glob, json, os
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_ORDER = ["CUED-Net", "TMC", "Deep-Ensemble(M=5)", "MC-Dropout", "Single-view-EDL"]

def pooled_f1(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    denom = 2*tp + fp + fn
    return (2*tp/denom) if denom else 0.0

def per_seed_f1(df):
    out = {}
    for seed, g in df.groupby("seed"):
        y = g["label"].to_numpy()
        pred = g["predicted"].to_numpy()
        out[int(seed)] = pooled_f1(y, pred)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="/workspace/cued_net/cv_preds")
    ap.add_argument("--out", default="/workspace/cued_net/calib_out")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(args.pred_dir, "*_preds.csv")))
    frames = {}
    for path in csv_paths:
        df = pd.read_csv(path)
        for name, g in df.groupby("model"):
            frames[name] = g.copy()

    results = {}
    print("\n── Per-seed F1 and seed-variance ──")
    for name in MODEL_ORDER:
        if name not in frames:
            print(f"  [skip] {name}")
            continue
        ps = per_seed_f1(frames[name])
        vals = np.array(list(ps.values()))
        results[name] = {
            "per_seed_f1": ps,
            "f1_mean": float(vals.mean()),
            "f1_std":  float(vals.std(ddof=1)),
            "f1_min":  float(vals.min()),
            "f1_max":  float(vals.max()),
            "f1_range": float(vals.max() - vals.min()),
        }
        print(f"  {name:22s} mean={vals.mean():.4f}  std={vals.std(ddof=1):.4f}  "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]  "
              f"per-seed={[f'{v:.3f}' for v in vals]}")

    json.dump(results, open(Path(args.out)/"seed_variance.json","w"), indent=2)
    print(f"\n[ok] -> {Path(args.out)/'seed_variance.json'}")

    # R3.2 / R2.11 contrast: single-model CUED-Net vs Deep-Ensemble stability
    if "CUED-Net" in results and "Deep-Ensemble(M=5)" in results:
        c = results["CUED-Net"]; e = results["Deep-Ensemble(M=5)"]
        print("\n── R3.2 / R2.11 contrast ──")
        print(f"  CUED-Net seed-F1 std:        {c['f1_std']:.4f} "
              f"(range {c['f1_range']:.4f})")
        print(f"  Deep-Ensemble seed-F1 std:   {e['f1_std']:.4f} "
              f"(range {e['f1_range']:.4f})")
        if e['f1_std'] > 0:
            print(f"  Note: these are per-seed pooled-F1 stds. The 0.633-0.808 figure "
                  f"in R3.2 is the raw single-model spread; this script's CUED-Net std "
                  f"reflects pooled-CV smoothing.")

if __name__ == "__main__":
    main()