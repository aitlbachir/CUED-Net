#!/usr/bin/env python
"""
temperature_scaling.py — Post-hoc temperature scaling for R2.8 (CBIS-DDSM, 5x5 CV).

Fits a single scalar T per model by minimising NLL on the pooled predictions
(binary case: logit = log(p/(1-p)), calibrated p = sigmoid(logit / T)).
Reports ECE/Brier/NLL before and after scaling for all 5 models.

GPU: NOT REQUIRED.
"""
import argparse, glob, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

MODEL_ORDER = ["CUED-Net", "TMC", "Deep-Ensemble(M=5)", "MC-Dropout", "Single-view-EDL"]
N_BINS = 15
EPS = 1e-7

def to_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def ece(y, p, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    n = len(y); total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        total += m.sum() * abs(y[m].mean() - p[m].mean())
    return total / n

def brier(y, p):
    return float(np.mean((p - y) ** 2))

def nll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def fit_temperature(y, p):
    """Fit T>0 minimising NLL of sigmoid(logit/T)."""
    z = to_logit(p)
    def obj(logT):
        T = np.exp(logT)              # ensures T>0
        pc = sigmoid(z / T)
        pc = np.clip(pc, EPS, 1 - EPS)
        return -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    res = minimize_scalar(obj, bounds=(np.log(0.05), np.log(20.0)),
                          method="bounded")
    return float(np.exp(res.x))

def load_models(csv_paths):
    frames = {}
    for path in csv_paths:
        df = pd.read_csv(path)
        for name, g in df.groupby("model"):
            frames[name] = g.copy()
    return frames

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="/workspace/cued_net/cv_preds")
    ap.add_argument("--out", default="/workspace/cued_net/calib_out")
    args = ap.parse_args()

    csv_paths = sorted(glob.glob(os.path.join(args.pred_dir, "*_preds.csv")))
    print("Loading:", *csv_paths, sep="\n  ")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    models = load_models(csv_paths)

    results = {}
    print("\n── Temperature scaling (per-model T fitted on pooled NLL) ──")
    print(f"{'Model':22s} {'T':>6s} {'ECE_raw':>9s} {'ECE_ts':>9s} "
          f"{'Brier_raw':>10s} {'Brier_ts':>10s} {'NLL_raw':>9s} {'NLL_ts':>9s}")
    for name in MODEL_ORDER:
        if name not in models:
            print(f"  [skip] {name} — not found")
            continue
        df = models[name]
        y = df["label"].to_numpy().astype(float)
        p = df["prob_malignant"].to_numpy().astype(float)

        T = fit_temperature(y, p)
        p_ts = sigmoid(to_logit(p) / T)

        r = {
            "T": T,
            "ece_raw": ece(y, p),     "ece_ts": ece(y, p_ts),
            "brier_raw": brier(y, p), "brier_ts": brier(y, p_ts),
            "nll_raw": nll(y, p),     "nll_ts": nll(y, p_ts),
        }
        results[name] = r
        print(f"{name:22s} {T:6.3f} {r['ece_raw']:9.4f} {r['ece_ts']:9.4f} "
              f"{r['brier_raw']:10.4f} {r['brier_ts']:10.4f} "
              f"{r['nll_raw']:9.4f} {r['nll_ts']:9.4f}")

    json.dump(results, open(Path(args.out) / "temperature_scaling.json", "w"), indent=2)
    print(f"\n[ok] -> {Path(args.out) / 'temperature_scaling.json'}")

    # focused CUED-Net summary line for the rebuttal
    if "CUED-Net" in results:
        c = results["CUED-Net"]
        print(f"\n── R2.8 sentence ──")
        print(f"  CUED-Net: T={c['T']:.3f}, ECE {c['ece_raw']:.4f} -> "
              f"{c['ece_ts']:.4f} after temperature scaling "
              f"({100*(c['ece_raw']-c['ece_ts'])/c['ece_raw']:.0f}% reduction)")

if __name__ == "__main__":
    main()