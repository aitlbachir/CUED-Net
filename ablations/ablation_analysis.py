#!/usr/bin/env python
"""Aggregate and interpret ablation results."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

EPS = 1e-7
N_BINS = 15

FULL = "/workspace/cued_net/cv_preds/cued_net_preds.csv"
NOVDL = "/workspace/cued_net/cv_ablation/no_vdl_preds.csv"
NOCON = "/workspace/cued_net/cv_ablation/no_consistency_preds.csv"
OUT = "/workspace/cued_net/cv_ablation/ablation_analysis.json"

def ece(y, p, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    n = len(y); tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0: continue
        tot += m.sum() * abs(y[m].mean() - p[m].mean())
    return tot / n

def brier(y, p): return float(np.mean((p - y) ** 2))
def nll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def per_seed_metrics(df):
    """Per-seed pooled metrics, then mean±std across seeds."""
    rows = []
    for seed, g in df.groupby("seed"):
        y = g["label"].to_numpy()
        p = g["prob_malignant"].to_numpy()
        pred = g["predicted"].to_numpy()
        rows.append(dict(
            f1=f1_score(y, pred),
            auc=roc_auc_score(y, p),
            ece=ece(y, p),
            brier=brier(y, p),
            nll=nll(y, p),
        ))
    d = pd.DataFrame(rows)
    return {k: (float(d[k].mean()), float(d[k].std(ddof=1))) for k in d.columns}

def load(path, name):
    df = pd.read_csv(path)
    # if the file holds multiple model names, keep the first (ablation CSVs have one)
    if df["model"].nunique() > 1:
        df = df[df["model"] == df["model"].unique()[0]]
    print(f"  {name:16s} rows={len(df)}  model={df['model'].iloc[0]}")
    return df

def main():
    print("Loading CSVs:")
    full = load(FULL, "full")
    novdl = load(NOVDL, "no_vdl")
    nocon = load(NOCON, "no_consistency")

    arms = {"Full CUED-Net": full, "w/o VDL": novdl, "w/o consistency": nocon}
    results = {}
    print(f"\n{'Arm':18s} {'F1':>14s} {'AUC':>14s} {'ECE':>14s} {'Brier':>14s} {'NLL':>14s}")
    for name, df in arms.items():
        m = per_seed_metrics(df)
        results[name] = m
        fmt = lambda k: f"{m[k][0]:.4f}\u00b1{m[k][1]:.4f}"
        print(f"{name:18s} {fmt('f1'):>14s} {fmt('auc'):>14s} {fmt('ece'):>14s} "
              f"{fmt('brier'):>14s} {fmt('nll'):>14s}")

    # ── deltas vs full ──
    print("\n── Deltas vs Full CUED-Net (positive = ablation worse for ECE/Brier/NLL) ──")
    base = results["Full CUED-Net"]
    deltas = {}
    for name in ["w/o VDL", "w/o consistency"]:
        m = results[name]
        d = {k: m[k][0] - base[k][0] for k in base}
        deltas[name] = d
        print(f"  {name:18s}  dF1={d['f1']:+.4f}  dAUC={d['auc']:+.4f}  "
              f"dECE={d['ece']:+.4f}  dBrier={d['brier']:+.4f}  dNLL={d['nll']:+.4f}")

    # ── high-discordance subset (VDL should matter most where views disagree) ──
    # Use the full model's uncertainty (combined) as the discordance proxy:
    # rank pairs by full-model uncertainty, take top 25%, compare recall there.
    print("\n── High-uncertainty subset (top 25% by full-model uncertainty) ──")
    # align positionally per (seed,fold)
    def aligned_subset_recall(df_full, df_arm, frac=0.25):
        recalls = []
        for (seed, fold), gf in df_full.groupby(["seed", "fold"]):
            ga = df_arm[(df_arm["seed"] == seed) & (df_arm["fold"] == fold)]
            if len(ga) != len(gf):  # alignment guard
                continue
            gf = gf.reset_index(drop=True); ga = ga.reset_index(drop=True)
            k = max(1, int(len(gf) * frac))
            idx = gf["uncertainty"].to_numpy().argsort()[::-1][:k]  # most-uncertain
            y = gf["label"].to_numpy()[idx]
            pred = ga["predicted"].to_numpy()[idx]
            pos = (y == 1)
            if pos.sum() > 0:
                recalls.append(((pred[pos] == 1).sum()) / pos.sum())
        return float(np.mean(recalls)) if recalls else float("nan")

    for name, df in [("Full CUED-Net", full), ("w/o VDL", novdl), ("w/o consistency", nocon)]:
        r = aligned_subset_recall(full, df)
        print(f"  {name:18s} recall@top25%-uncertain = {r:.4f}")
        results.setdefault("_subset_recall", {})[name] = r

    json.dump({"metrics": results, "deltas": deltas}, open(OUT, "w"), indent=2)
    print(f"\n[ok] -> {OUT}")

    print("\n── ablation interpretation ──")
    dvdl = deltas["w/o VDL"]
    print(f"  Removing VDL: F1 {dvdl['f1']:+.4f} (negligible), "
          f"ECE {dvdl['ece']:+.4f}, Brier {dvdl['brier']:+.4f}, NLL {dvdl['nll']:+.4f}")
    if dvdl['ece'] > 0.002 or dvdl['brier'] > 0.002:
        print("  -> VDL improves CALIBRATION more than discrimination .")
    else:
        print("  -> VDL effect on calibration is also small; the interpretation must rest on the "
              "flagging/interpretability argument, stated honestly.")

if __name__ == "__main__":
    main()
