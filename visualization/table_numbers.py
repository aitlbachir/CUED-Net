"""Emit summary numbers for the results tables."""

import os, sys, json
import numpy as np
import pandas as pd

WS = "/workspace/cued_net"
DIR = os.path.join(WS, "cv_preds_novdl")

FILES = {
    "CUED-Net":        "cued_net_preds.csv",
    "Deep-Ensemble":   "ensemble_preds.csv",
    "MC-Dropout":      "mcdropout_preds.csv",
    "Single-view-EDL": "single_view_edl_preds.csv",
    "TMC":             "tmc_preds.csv",
}

def find_col(cols, *cands):
    low = {c.lower(): c for c in cols}
    for cand in cands:
        if cand in low:
            return low[cand]
    return None

# ---- 1. inspect the CUED-Net CSV columns first ----
path = os.path.join(DIR, FILES["CUED-Net"])
df = pd.read_csv(path)
print("=" * 64)
print("CUED-Net CSV columns:", list(df.columns))
print("shape:", df.shape)
print(df.head(3).to_string())
print("=" * 64)

label_col = find_col(df.columns, "label", "y_true", "true", "target", "gt")
pred_col  = find_col(df.columns, "pred", "predicted", "y_pred", "prediction", "pred_label")
prob_col  = find_col(df.columns, "prob", "prob_malignant", "p_mal", "prob_mal", "p_malignant",
                     "score", "prob_1", "malignant_prob", "y_prob")
print(f"detected -> label={label_col!r}  pred={pred_col!r}  prob={prob_col!r}")
if not (label_col and pred_col):
    print("\n!! Could not auto-detect label/pred columns. Tell me the real names from")
    print("   the 'columns' line above and I'll hard-code them.")
    sys.exit(0)

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

def metrics(d, lc, pc, prc=None):
    y = d[lc].values.astype(int)
    p = d[pc].values.astype(int)
    out = dict(
        n=len(y),
        acc=accuracy_score(y, p),
        prec=precision_score(y, p, zero_division=0),
        rec=recall_score(y, p, zero_division=0),
        f1=f1_score(y, p, zero_division=0),
    )
    if prc and prc in d.columns:
        try:
            out["auc"] = roc_auc_score(y, d[prc].values.astype(float))
        except Exception as e:
            out["auc"] = None
    return out

# ---- 2. CUED-Net pooled metrics + verification ----
m = metrics(df, label_col, pred_col, prob_col)
print("\nCUED-Net pooled metrics:")
print(f"  n        = {m['n']}")
print(f"  accuracy = {m['acc']:.4f}")
print(f"  precision= {m['prec']:.4f}")
print(f"  recall   = {m['rec']:.4f}")
print(f"  F1       = {m['f1']:.4f}   (expected ~0.834)")
print(f"  AUC      = {m.get('auc')}   (expected ~0.877)")
ok = abs(m["f1"] - 0.834) < 0.02 and (m.get("auc") is None or abs(m["auc"] - 0.877) < 0.02)
print("  VERIFY:", "PASS — reproduces known F1/AUC, numbers trustworthy"
      if ok else "MISMATCH — investigate before using (wrong file/columns/per-seed pooling?)")

# ---- 3. per-seed mean+/-std if a seed column exists ----
seed_col = find_col(df.columns, "seed", "run", "model", "fold_seed")
if seed_col:
    rows = []
    for s, g in df.groupby(seed_col):
        rows.append(metrics(g, label_col, pred_col, prob_col))
    arr = lambda k: np.array([r[k] for r in rows], float)
    print(f"\nPer-{seed_col} (n={len(rows)}) mean +/- std:")
    for k, lab in [("acc","accuracy"),("prec","precision"),("rec","recall"),("f1","F1")]:
        print(f"  {lab:9s}= {arr(k).mean():.4f} +/- {arr(k).std():.4f}")
else:
    print("\n(no seed column found; reporting pooled metrics only)")

# ---- 4. TMC and Single-view-EDL operating-point F1 (Table II [PEND]) ----
print("\n" + "=" * 64)
print("Table II operating-point F1 for the two [PEND] baselines:")
for name in ["TMC", "Single-view-EDL"]:
    p = os.path.join(DIR, FILES[name])
    if not os.path.exists(p):
        print(f"  {name}: FILE MISSING {p}"); continue
    d = pd.read_csv(p)
    lc = find_col(d.columns, "label","y_true","true","target","gt")
    pc = find_col(d.columns, "pred","predicted","y_pred","prediction","pred_label")
    if not (lc and pc):
        print(f"  {name}: columns = {list(d.columns)} (could not detect label/pred)")
        continue
    mm = metrics(d, lc, pc)
    print(f"  {name:16s} F1 = {mm['f1']:.4f}  (acc {mm['acc']:.4f}, rec {mm['rec']:.4f})")
print("=" * 64)
