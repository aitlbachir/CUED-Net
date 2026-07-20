"""Plot the main results figure."""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

WS = "/workspace/cued_net"
DIR = os.path.join(WS, "cv_preds_novdl")
OUT_PNG = os.path.join(WS, "results_figure.png")
OUT_PDF = os.path.join(WS, "results_figure.pdf")

LABEL, PRED, PROB, UNC = "label", "predicted", "prob_malignant", "uncertainty"

BASELINES = {  # display name -> csv ; prob col assumed prob_malignant, else adapt
    "CUED-Net":        "cued_net_preds.csv",
    "TMC":             "tmc_preds.csv",
    "Deep-Ensemble":   "ensemble_preds.csv",
    "MC-Dropout":      "mcdropout_preds.csv",
    "Single-view EDL": "single_view_edl_preds.csv",
}
COLORS = {
    "CUED-Net": "#1A3D6D", "TMC": "#C44E52", "Deep-Ensemble": "#55A868",
    "MC-Dropout": "#8172B3", "Single-view EDL": "#937860",
}

def load(name):
    p = os.path.join(DIR, BASELINES[name])
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    # tolerate prob col name differences
    prob = PROB if PROB in d.columns else next((c for c in d.columns
            if "prob" in c.lower() and "malig" in c.lower()), None)
    if prob is None:
        prob = next((c for c in d.columns if c.lower().startswith("prob")), None)
    return d.rename(columns={prob: PROB}) if prob and prob != PROB else d

data = {n: load(n) for n in BASELINES}
data = {n: d for n, d in data.items() if d is not None}
cued = data["CUED-Net"]
print("loaded:", list(data.keys()))

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.linewidth": 0.7, "savefig.dpi": 300})
fig, ax = plt.subplots(2, 3, figsize=(12, 7.2))

# ---- (a) ROC ----
a = ax[0, 0]
for n, d in data.items():
    fpr, tpr, _ = roc_curve(d[LABEL], d[PROB])
    a.plot(fpr, tpr, color=COLORS.get(n), lw=1.6 if n == "CUED-Net" else 1.0,
           label=f"{n} ({auc(fpr,tpr):.3f})", zorder=5 if n == "CUED-Net" else 3)
a.plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.5)
a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate")
a.set_title("(a) ROC curve"); a.legend(fontsize=6.5, loc="lower right"); a.grid(alpha=0.3)

# ---- (b) PR ----
b = ax[0, 1]
for n, d in data.items():
    prec, rec, _ = precision_recall_curve(d[LABEL], d[PROB])
    ap = average_precision_score(d[LABEL], d[PROB])
    b.plot(rec, prec, color=COLORS.get(n), lw=1.6 if n == "CUED-Net" else 1.0,
           label=f"{n} ({ap:.3f})", zorder=5 if n == "CUED-Net" else 3)
base_rate = cued[LABEL].mean()
b.axhline(base_rate, ls="--", color="k", lw=0.6, alpha=0.5)
b.set_xlabel("Recall"); b.set_ylabel("Precision")
b.set_title("(b) Precision-Recall"); b.legend(fontsize=6.5, loc="lower left"); b.grid(alpha=0.3)

# ---- (c) Selective prediction: F1 vs coverage (defer most-uncertain first) ----
from sklearn.metrics import f1_score
c = ax[0, 2]
covs = np.linspace(0.3, 1.0, 15)
for n, d in data.items():
    if UNC not in d.columns:
        continue
    f1s = []
    order = d[UNC].values.argsort()  # ascending uncertainty
    yy, pp = d[LABEL].values[order], d[PRED].values[order]
    for cov in covs:
        k = max(1, int(cov * len(yy)))
        f1s.append(f1_score(yy[:k], pp[:k], zero_division=0))
    c.plot(covs * 100, f1s, color=COLORS.get(n), lw=1.6 if n == "CUED-Net" else 1.0,
           marker="o" if n == "CUED-Net" else None, markersize=3, label=n,
           zorder=5 if n == "CUED-Net" else 3)
c.set_xlabel("Coverage (%)"); c.set_ylabel("F1 on retained cases")
c.set_title("(c) Selective prediction"); c.legend(fontsize=6.5, loc="upper right"); c.grid(alpha=0.3)
c.invert_xaxis()

# ---- (d) uncertainty correct vs incorrect (the p=0.012 story) ----
from scipy.stats import mannwhitneyu
dd = ax[1, 0]
correct = cued[cued[LABEL] == cued[PRED]][UNC].values
incorrect = cued[cued[LABEL] != cued[PRED]][UNC].values
parts = dd.violinplot([correct, incorrect], showmeans=True, showextrema=False)
for pc, col in zip(parts["bodies"], ["#55A868", "#C44E52"]):
    pc.set_facecolor(col); pc.set_alpha(0.6)
try:
    u, pval = mannwhitneyu(incorrect, correct, alternative="greater")
except Exception:
    pval = float("nan")
dd.set_xticks([1, 2]); dd.set_xticklabels(["Correct", "Incorrect"])
dd.set_ylabel("Predictive uncertainty")
dd.set_title(f"(d) Uncertainty by correctness (p={pval:.3f})"); dd.grid(alpha=0.3, axis="y")

# ---- (e) per-source uncertainty distributions (replaces retired pie) ----
e = ax[1, 1]
# look for per-source columns in CUED-Net CSV first, then any loaded baseline (the
# ensemble dump may carry uncertainty_evidential/_ensemble/_discordance columns)
def find_source_cols(frame):
    return {k: v for k, v in {
        "Evidential": next((c for c in frame.columns if "evid" in c.lower()), None),
        "Discordance": next((c for c in frame.columns if "disc" in c.lower()), None),
        "Ensemble": next((c for c in frame.columns if ("ens" in c.lower() and "uncert" in c.lower())
                          or c.lower() in ("u_ens","uncertainty_ensemble")), None),
    }.items() if v is not None}

src_frame, src_cols = cued, find_source_cols(cued)
if len(src_cols) < 2:
    for n, d in data.items():
        sc = find_source_cols(d)
        if len(sc) >= 2:
            src_frame, src_cols = d, sc
            print(f"NOTE: using per-source columns from {n} CSV for panel (e).")
            break

if len(src_cols) >= 2:
    vals = [src_frame[v].values for v in src_cols.values()]
    p = e.violinplot(vals, showmeans=True, showextrema=False)
    for pc, col in zip(p["bodies"], ["#4C72B0", "#55A868", "#C44E52"]):
        pc.set_facecolor(col); pc.set_alpha(0.6)
    e.set_xticks(range(1, len(src_cols) + 1)); e.set_xticklabels(list(src_cols.keys()))
    e.set_ylabel("Component value"); e.set_title("(e) Uncertainty components")
else:
    print("NOTE: per-source columns (u_evid/u_disc/u_ens) not found in any CSV.")
    print("      Panel (e) falls back to total-uncertainty histogram. Point me at the")
    print("      CSV with those columns to get the 3-way component panel.")
    e.hist(cued[UNC].values, bins=30, color="#4C72B0", alpha=0.7)
    e.set_xlabel("Predictive uncertainty"); e.set_ylabel("Count")
    e.set_title("(e) Uncertainty distribution")
e.grid(alpha=0.3, axis="y")

# ---- (f) confusion matrix (pooled CV) ----
f = ax[1, 2]
cm = confusion_matrix(cued[LABEL], cued[PRED])
im = f.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        f.text(j, i, str(cm[i, j]), ha="center", va="center",
               color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
f.set_xticks([0, 1]); f.set_yticks([0, 1])
f.set_xticklabels(["Benign", "Malignant"]); f.set_yticklabels(["Benign", "Malignant"])
f.set_xlabel("Predicted"); f.set_ylabel("True")
f.set_title("(f) Confusion matrix (pooled CV)")

fig.tight_layout(pad=1.0)
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print("saved:", OUT_PNG)
print("saved:", OUT_PDF)
# print the panel-(f) counts so the caption can be updated to match
tn, fp, fn, tp = cm.ravel()
print(f"confusion (pooled CV): TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"CUED-Net AUC={auc(*roc_curve(cued[LABEL],cued[PROB])[:2]):.4f} "
      f"AP={average_precision_score(cued[LABEL],cued[PROB]):.4f}")
