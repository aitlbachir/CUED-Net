#!/usr/bin/env python
"""DeLong test comparing ConvNeXt and DenseNet backbones."""

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/cued_net")

# import the canonical machinery — do NOT reimplement
import stats_tests2 as st

SEEDS_ABLATION = [42, 123, 456]   # the 3 seeds ConvNeXt was run on
DENSENET_CSV = "/workspace/cued_net/cv_preds_novdl/cued_net_preds.csv"   # model="CUED-Net"
CONVNEXT_CSV = "/workspace/cued_net/cv_backbone/convnext_tiny_preds.csv" # model="CUED-Net-convnext_tiny"
OUT = "/workspace/cued_net/cv_backbone/delong_convnext_vs_densenet.json"


def main():
    # load both via the LOCKED-schema loader (returns dict model_name -> df)
    dfs = st.load_models([DENSENET_CSV, CONVNEXT_CSV])
    # resolve model names robustly (don't hardcode in case of suffix drift)
    names = list(dfs.keys())
    dn_name = [n for n in names if "convnext" not in n.lower()][0]
    cx_name = [n for n in names if "convnext" in n.lower()][0]
    df_dn = dfs[dn_name].copy()
    df_cx = dfs[cx_name].copy()
    print(f"[load] DenseNet model='{dn_name}' rows={len(df_dn)}; "
          f"ConvNeXt model='{cx_name}' rows={len(df_cx)}")

    # ── FAIRNESS: subset DenseNet to the 3 ablation seeds ───────────────────
    df_dn = df_dn[df_dn["seed"].isin(SEEDS_ABLATION)].copy()
    df_cx = df_cx[df_cx["seed"].isin(SEEDS_ABLATION)].copy()
    print(f"[subset] seeds={SEEDS_ABLATION}: DenseNet rows={len(df_dn)}, "
          f"ConvNeXt rows={len(df_cx)}")

    # ── GATE 1: same (seed,fold) cells, same per-cell row counts ────────────
    dn_cells = df_dn.groupby(["seed","fold"]).size()
    cx_cells = df_cx.groupby(["seed","fold"]).size()
    if not dn_cells.index.equals(cx_cells.index):
        raise SystemExit(f"[GATE FAIL] cell sets differ:\n DN={list(dn_cells.index)}\n "
                         f"CX={list(cx_cells.index)}")
    if not (dn_cells.values == cx_cells.values).all():
        diff = {str(k): (int(dn_cells[k]), int(cx_cells[k]))
                for k in dn_cells.index if dn_cells[k] != cx_cells[k]}
        raise SystemExit(f"[GATE FAIL] per-cell row counts differ (DN,CX): {diff}")
    print(f"[gate1] {len(dn_cells)} cells, identical per-cell row counts -> OK")

    # ── align via the canonical (seed,fold,row-index) join + patient assert ─
    # align_pair(reference_df, other_df) -> merged frame with aligned probs/labels
    merged = st.align(df_dn, df_cx)
    print(f"[gate2] align_pair passed (patient_id agreement asserted), "
          f"n_paired={len(merged)}")

    # pull aligned vectors. align_pair names: prob_ref / prob_oth, label
    y = merged["label_ref"].to_numpy().astype(int)
    p_dn = merged["prob_malignant_ref"].to_numpy().astype(float)
    p_cx = merged["prob_malignant_oth"].to_numpy().astype(float)
    pid = merged["patient_id_ref"].to_numpy().astype(str)

    auc_dn = st.auc_score(y, p_dn)
    auc_cx = st.auc_score(y, p_cx)
    d_auc = auc_cx - auc_dn

    # DeLong paired test
    _auc_a, _auc_b, z, p_delong = st.delong_test(y, p_cx, p_dn)

    # BCa CIs per model (patient-clustered)
    auc_dn_b, dn_lo, dn_hi = st.bca_auc_ci(y, p_dn, pid, n_boot=1000, seed=0)
    auc_cx_b, cx_lo, cx_hi = st.bca_auc_ci(y, p_cx, pid, n_boot=1000, seed=0)

    # patient-clustered bootstrap CI on the DIFFERENCE (paired)
    rng = np.random.default_rng(0)
    uniq = np.array(sorted(set(pid)))
    pid_to_idx = {q: np.where(pid == q)[0] for q in uniq}
    diffs = []
    for _ in range(1000):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([pid_to_idx[q] for q in samp])
        try:
            diffs.append(st.auc_score(y[idx], p_cx[idx]) - st.auc_score(y[idx], p_dn[idx]))
        except Exception:
            continue
    diffs = np.array(diffs)
    d_lo, d_hi = np.percentile(diffs, [2.5, 97.5])

    print("\n" + "="*64)
    print("ConvNeXt-T vs DenseNet-121 (CUED-Net) — pooled, 3-seed, paired")
    print("="*64)
    print(f"  DenseNet-121 pooled AUC : {auc_dn:.4f}  (BCa 95% CI {dn_lo:.4f}–{dn_hi:.4f})")
    print(f"  ConvNeXt-T   pooled AUC : {auc_cx:.4f}  (BCa 95% CI {cx_lo:.4f}–{cx_hi:.4f})")
    print(f"  ΔAUC (ConvNeXt − DenseNet): {d_auc:+.4f}  "
          f"(bootstrap 95% CI {d_lo:+.4f}–{d_hi:+.4f})")
    print(f"  DeLong z={z:.3f}, p={p_delong:.4g}")
    verdict = ("SIGNIFICANT (ConvNeXt > DenseNet)" if p_delong < 0.05 and d_auc > 0
               else "SIGNIFICANT (DenseNet > ConvNeXt)" if p_delong < 0.05 and d_auc < 0
               else "NOT SIGNIFICANT (tie at alpha=0.05)")
    ci_excludes_zero = (d_lo > 0) or (d_hi < 0)
    print(f"  Verdict: {verdict}")
    print(f"  Bootstrap ΔAUC CI excludes 0: {ci_excludes_zero}")
    print("="*64)

    json.dump({
        "seeds": SEEDS_ABLATION,
        "n_paired": int(len(merged)),
        "auc_densenet": auc_dn, "auc_densenet_ci": [dn_lo, dn_hi],
        "auc_convnext": auc_cx, "auc_convnext_ci": [cx_lo, cx_hi],
        "delta_auc": d_auc, "delta_auc_boot_ci": [float(d_lo), float(d_hi)],
        "delong_z": float(z), "delong_p": float(p_delong),
        "verdict": verdict, "ci_excludes_zero": bool(ci_excludes_zero),
    }, open(OUT, "w"), indent=2)
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
