#!/usr/bin/env python
"""
uncertainty_covariance.py — Pearson r between the 3 uncertainty signals (AE.1, R2.1).

Reads the decomposed CUED-Net CSV (u_evid, u_disc, and ensemble u_ens/u_total
where available). Reports pairwise Pearson r + a correlation-matrix figure.
Near-zero off-diagonal r supports the 'complementary, not redundant' claim.

GPU: NOT REQUIRED.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomposed",
                    default="/workspace/cued_net/selective_preds/cued_net_preds_decomposed.csv")
    ap.add_argument("--ensemble",
                    default="/workspace/cued_net/selective_preds/cued_net_ensemble_preds.csv")
    ap.add_argument("--out", default="/workspace/cued_net/calib_out")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.decomposed)
    print("Decomposed CSV columns:", list(df.columns))

    # resolve column names defensively
    c_evid = find_col(df, ["u_evid", "uncertainty_evidential", "u_evidential"])
    c_disc = find_col(df, ["u_disc", "uncertainty_discordance", "u_discordance"])
    c_comb = find_col(df, ["u_combined", "u_comb", "uncertainty_combined", "uncertainty"])
    print(f"Resolved: evid={c_evid}  disc={c_disc}  comb={c_comb}")

    signals = {}
    if c_evid: signals["u_evid"] = df[c_evid].to_numpy()
    if c_disc: signals["u_disc"] = df[c_disc].to_numpy()
    if c_comb: signals["u_comb"] = df[c_comb].to_numpy()

    # try to add ensemble u_ens from the ensemble CSV (530 rows)
    try:
        dfe = pd.read_csv(args.ensemble)
        c_ens = find_col(dfe, ["u_ens", "uncertainty_ensemble", "u_ensemble"])
        c_tot = find_col(dfe, ["u_total", "uncertainty_total"])
        print("Ensemble CSV columns:", list(dfe.columns))
        # NOTE: ensemble CSV is 530 rows (per-fold CV ensemble), decomposed is 2650.
        # We compute the ensemble-level correlations separately, not mixed.
        ens_signals = {}
        if c_ens: ens_signals["u_ens"] = dfe[c_ens].to_numpy()
        c_evid_e = find_col(dfe, ["u_evid","uncertainty_evidential"])
        c_disc_e = find_col(dfe, ["u_disc","uncertainty_discordance"])
        if c_evid_e: ens_signals["u_evid"] = dfe[c_evid_e].to_numpy()
        if c_disc_e: ens_signals["u_disc"] = dfe[c_disc_e].to_numpy()
    except Exception as e:
        print("Ensemble CSV not read:", e)
        ens_signals = {}

    def corr_report(sigs, label):
        names = list(sigs.keys())
        print(f"\n── Pairwise Pearson r ({label}) ──")
        out = {}
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                a, b = names[i], names[j]
                # guard against constant arrays
                if np.std(sigs[a]) < 1e-12 or np.std(sigs[b]) < 1e-12:
                    r, p = float("nan"), float("nan")
                else:
                    r, p = pearsonr(sigs[a], sigs[b])
                out[f"{a}__{b}"] = {"r": float(r), "p": float(p)}
                print(f"  r({a}, {b}) = {r:+.4f}  (p={p:.2e})")
        return out, names

    single_corr, single_names = corr_report(signals, "single-model, 2650 rows")
    ens_corr, ens_names = ({}, [])
    if len(ens_signals) >= 2:
        ens_corr, ens_names = corr_report(ens_signals, "ensemble, 530 rows")

    json.dump({"single_model": single_corr, "ensemble": ens_corr},
              open(Path(args.out)/"uncertainty_covariance.json","w"), indent=2)
    print(f"\n[ok] -> {Path(args.out)/'uncertainty_covariance.json'}")

    # correlation-matrix heatmap (single-model signals)
    if len(single_names) >= 2:
        n = len(single_names)
        M = np.eye(n)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                key = f"{single_names[min(i,j)]}__{single_names[max(i,j)]}"
                if key in single_corr:
                    M[i, j] = single_corr[key]["r"]
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(single_names, rotation=45, ha="right")
        ax.set_yticklabels(single_names)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        fontsize=9, color="black")
        ax.set_title("Uncertainty signal correlation", fontsize=11)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(Path(args.out)/"uncertainty_covariance.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(Path(args.out)/"uncertainty_covariance.png", dpi=300, bbox_inches="tight")
        print(f"[fig] -> {Path(args.out)/'uncertainty_covariance.pdf'} / .png")

if __name__ == "__main__":
    main()