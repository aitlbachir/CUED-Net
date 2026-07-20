#!/usr/bin/env python
"""
roc_pr_curves.py — ROC and Precision-Recall curves for all 5 models (R1.8).

Pooled 5x5 CV predictions. Reports AUROC and Average Precision (AP) per model,
draws a two-panel figure (ROC | PR) with all models overlaid.

GPU: NOT REQUIRED.
"""
import argparse, glob, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_ORDER = ["CUED-Net", "TMC", "Deep-Ensemble(M=5)", "MC-Dropout", "Single-view-EDL"]
DISPLAY = {  # clean labels for the legend
    "CUED-Net": "CUED-Net",
    "TMC": "TMC",
    "Deep-Ensemble(M=5)": "Deep-Ensemble",
    "MC-Dropout": "MC-Dropout",
    "Single-view-EDL": "Single-view-EDL",
}
COLOURS = {
    "CUED-Net":        "#1f77b4",
    "TMC":             "#ff7f0e",
    "Deep-Ensemble(M=5)": "#2ca02c",
    "MC-Dropout":      "#d62728",
    "Single-view-EDL": "#9467bd",
}

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
    Path(args.out).mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(args.pred_dir, "*_preds.csv")))
    print("Loading:", *csv_paths, sep="\n  ")
    models = load_models(csv_paths)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 5))
    results = {}

    # baseline positive rate for the PR no-skill line
    any_df = next(iter(models.values()))
    pos_rate = float(any_df["label"].mean())

    for name in MODEL_ORDER:
        if name not in models:
            print(f"  [skip] {name}")
            continue
        df = models[name]
        y = df["label"].to_numpy()
        p = df["prob_malignant"].to_numpy()

        fpr, tpr, _ = roc_curve(y, p)
        roc_auc = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y, p)
        ap_score = average_precision_score(y, p)

        results[name] = {"auroc": float(roc_auc), "ap": float(ap_score)}
        lbl = DISPLAY[name]
        ax_roc.plot(fpr, tpr, color=COLOURS[name], lw=1.8,
                    label=f"{lbl} (AUC={roc_auc:.3f})")
        ax_pr.plot(rec, prec, color=COLOURS[name], lw=1.8,
                   label=f"{lbl} (AP={ap_score:.3f})")
        print(f"  {name:22s} AUROC={roc_auc:.4f}  AP={ap_score:.4f}")

    # ROC panel
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate", fontsize=11)
    ax_roc.set_title("ROC — CBIS-DDSM (5×5 CV, pooled)", fontsize=11)
    ax_roc.legend(fontsize=8.5, loc="lower right")
    ax_roc.grid(alpha=0.3); ax_roc.set_xlim(0,1); ax_roc.set_ylim(0,1.02)

    # PR panel
    ax_pr.axhline(pos_rate, color="k", ls="--", lw=1, alpha=0.5,
                  label=f"No-skill ({pos_rate:.3f})")
    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.set_title("Precision-Recall — CBIS-DDSM (5×5 CV, pooled)", fontsize=11)
    ax_pr.legend(fontsize=8.5, loc="lower left")
    ax_pr.grid(alpha=0.3); ax_pr.set_xlim(0,1); ax_pr.set_ylim(0,1.02)

    fig.tight_layout()
    fig.savefig(Path(args.out)/"roc_pr_curves.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(Path(args.out)/"roc_pr_curves.png", dpi=300, bbox_inches="tight")
    json.dump(results, open(Path(args.out)/"roc_pr_results.json","w"), indent=2)
    print(f"\n[fig] -> {Path(args.out)/'roc_pr_curves.pdf'} / .png")
    print(f"[ok]  -> {Path(args.out)/'roc_pr_results.json'}")

if __name__ == "__main__":
    main()