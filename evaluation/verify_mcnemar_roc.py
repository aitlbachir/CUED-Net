#!/usr/bin/env python3
"""Cross-check McNemar and ROC computations."""

import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import binomtest

DIR = "cv_preds_novdl"
REF = f"{DIR}/cued_net_preds.csv"                       # CUED-Net, lambda_vdl = 0
BASELINES = {
    "TMC":             f"{DIR}/tmc_preds.csv",
    "Deep-Ensemble":   f"{DIR}/ensemble_preds.csv",
    "MC-Dropout":      f"{DIR}/mcdropout_preds.csv",
    "Single-view-EDL": f"{DIR}/single_view_edl_preds.csv",
}
KEYS = ["seed", "fold", "_row"]

def add_row(df):
    df = df.copy()
    df["_row"] = df.groupby(["seed", "fold"]).cumcount()
    return df

def auc_variants(df):
    """Return (AUROC pooled, AP pooled, mean-over-seeds AUROC, mean-over-folds AUROC)."""
    pooled    = roc_auc_score(df.label, df.prob_malignant)
    ap_pooled = average_precision_score(df.label, df.prob_malignant)
    seed_auc  = df.groupby("seed").apply(
        lambda g: roc_auc_score(g.label, g.prob_malignant))
    fold_auc  = df.groupby(["seed", "fold"]).apply(
        lambda g: roc_auc_score(g.label, g.prob_malignant))
    return pooled, ap_pooled, float(seed_auc.mean()), float(fold_auc.mean())

def holm(pvals: dict):
    """Holm-Bonferroni; returns {name: adjusted p} enforcing monotonicity."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, running, adj = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[k] = min(running, 1.0)
    return adj

ref = add_row(pd.read_csv(REF))
assert len(ref) == 2650, f"reference row count {len(ref)} != 2650"

print("=== CUED-Net (no_vdl) discrimination ===")
cp, cap, cms, cmf = auc_variants(ref)
print(f"AUROC  pooled={cp:.4f}  mean-over-seeds={cms:.4f}  mean-over-folds={cmf:.4f}"
      f"   AP(pooled)={cap:.4f}")

rows, mcp = [], {}
for name, path in BASELINES.items():
    bdf = add_row(pd.read_csv(path))
    m = ref.merge(bdf, on=KEYS, suffixes=("_ref", "_oth"))
    assert len(m) == len(ref), f"{name}: merge size {len(m)} != {len(ref)}"
    assert (m.patient_id_ref == m.patient_id_oth).all(), f"{name}: patient_id misalignment"
    assert (m.label_ref == m.label_oth).all(),            f"{name}: label misalignment"
    y = m.label_ref.values
    cued_ok = (m.predicted_ref.values == y)
    base_ok = (m.predicted_oth.values == y)
    b = int(np.sum(cued_ok & ~base_ok))   # CUED correct, baseline wrong
    c = int(np.sum(~cued_ok & base_ok))   # CUED wrong,   baseline correct
    p_mc = binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue if (b + c) else 1.0
    mcp[name] = p_mc
    bp, bap, bms, bmf = auc_variants(bdf)
    rows.append((name, b, c, p_mc, bp, bap, bms, bmf))

adj = holm(mcp)

print("\n=== McNemar on correctness (pooled 2650; b=CUED-correct/base-wrong, "
      "c=CUED-wrong/base-correct) ===")
print(f"{'Baseline':<16}{'b':>6}{'c':>6}{'b+c':>6}{'p_raw':>13}{'p_Holm':>13}")
for name, b, c, p_mc, *_ in rows:
    print(f"{name:<16}{b:>6}{c:>6}{b+c:>6}{p_mc:>13.3e}{adj[name]:>13.3e}")

print("\n=== AUROC variants per model (diagnose 0.877 vs 0.874) ===")
print(f"{'Model':<16}{'pooled':>10}{'mean_seed':>12}{'mean_fold':>12}{'AP_pooled':>12}")
print(f"{'CUED-Net':<16}{cp:>10.4f}{cms:>12.4f}{cmf:>12.4f}{cap:>12.4f}")
for name, b, c, p_mc, bp, bap, bms, bmf in rows:
    print(f"{name:<16}{bp:>10.4f}{bms:>12.4f}{bmf:>12.4f}{bap:>12.4f}")

# Optional cross-check with the canonical module (non-fatal if signature differs)
try:
    import stats_tests2 as st
    print("\n=== DeLong cross-check via stats_tests2 (AUC ties) ===")
    for name, path in BASELINES.items():
        bdf = add_row(pd.read_csv(path))
        m = ref.merge(bdf, on=KEYS, suffixes=("_ref", "_oth"))
        out = st.delong_test(m.label_ref.values,
                             m.prob_malignant_ref.values,
                             m.prob_malignant_oth.values)
        print(f"{name:<16} delong_test -> {out}")
except Exception as e:
    print(f"\n[DeLong cross-check skipped: {type(e).__name__}: {e}]")
