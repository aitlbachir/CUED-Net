#!/usr/bin/env python
"""Calibration metrics (ECE, Brier, NLL) on CBIS-DDSM."""

import argparse, glob, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

# ── display order & colours ─────────────────────────────────────────────────
MODEL_ORDER = ["CUED-Net", "TMC", "Deep-Ensemble(M=5)", "MC-Dropout", "Single-view-EDL"]
COLOURS     = {
    "CUED-Net":        "#1f77b4",
    "TMC":             "#ff7f0e",
    "Deep-Ensemble(M=5)":   "#2ca02c",
    "MC-Dropout":      "#d62728",
    "Single-view-EDL": "#9467bd",
}
MARKERS = {
    "CUED-Net":        "o",
    "TMC":             "s",
    "Deep-Ensemble(M=5)":   "^",
    "MC-Dropout":      "D",
    "Single-view-EDL": "v",
}

N_BINS = 15   # standard; reviewers will check this matches manuscript claim

# ── calibration metrics ──────────────────────────────────────────────────────
def reliability_bins(y_true, prob, n_bins=N_BINS):
    """Return (bin_centres, bin_accs, bin_confs, bin_counts) for the diagram."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres, accs, confs, counts = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi)
        if mask.sum() == 0:
            continue
        centres.append(0.5 * (lo + hi))
        accs.append(float(y_true[mask].mean()))
        confs.append(float(prob[mask].mean()))
        counts.append(int(mask.sum()))
    return np.array(centres), np.array(accs), np.array(confs), np.array(counts)


def ece(y_true, prob, n_bins=N_BINS):
    """Expected Calibration Error (weighted by bin size)."""
    _, accs, confs, counts = reliability_bins(y_true, prob, n_bins)
    n = len(y_true)
    return float(np.sum(counts * np.abs(accs - confs)) / n)


def mce(y_true, prob, n_bins=N_BINS):
    """Maximum Calibration Error."""
    _, accs, confs, _ = reliability_bins(y_true, prob, n_bins)
    return float(np.max(np.abs(accs - confs))) if len(accs) else 0.0


def brier(y_true, prob):
    return float(np.mean((prob - y_true) ** 2))


def nll(y_true, prob, eps=1e-7):
    p = np.clip(prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


# ── per-seed summary (feeds Table std) ──────────────────────────────────────
def per_seed_metrics(df):
    rows = []
    for seed, g in df.groupby("seed"):
        y = g["label"].to_numpy()
        p = g["prob_malignant"].to_numpy()
        rows.append({
            "seed":  int(seed),
            "ece":   ece(y, p),
            "mce":   mce(y, p),
            "brier": brier(y, p),
            "nll":   nll(y, p),
        })
    return pd.DataFrame(rows)


# ── load & align (reuse cv_preds schema) ────────────────────────────────────
def load_models(csv_paths):
    frames = {}
    for path in csv_paths:
        df = pd.read_csv(path)
        for name, g in df.groupby("model"):
            g = g.copy()
            g["patient_id"] = g["patient_id"].astype(str)
            frames[name] = g
    return frames


# ── reliability diagram ──────────────────────────────────────────────────────
def plot_reliability(models_data, out_stem):
    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    # perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect", zorder=0)

    for name in MODEL_ORDER:
        if name not in models_data:
            continue
        df = models_data[name]
        y = df["label"].to_numpy()
        p = df["prob_malignant"].to_numpy()
        _, accs, confs, counts = reliability_bins(y, p)
        ax.plot(confs, accs,
                color=COLOURS[name], marker=MARKERS[name],
                lw=1.6, ms=5, label=name, zorder=3)

    ax.set_xlabel("Mean predicted confidence", fontsize=11)
    ax.set_ylabel("Fraction of positives",     fontsize=11)
    ax.set_title("Reliability Diagram — CBIS-DDSM (5×5 CV)", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_stem + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_stem + ".png", dpi=300, bbox_inches="tight")
    print(f"[fig] {out_stem}.pdf / .png")
    plt.close(fig)


# ── LaTeX table ──────────────────────────────────────────────────────────────
def write_latex(results, ref_name, out_path):
    lines = [
        "% === TABLE: calibration | IEEEtran | Overleaf-ready ===",
        "% requires: \\usepackage{booktabs}  \\usepackage{siunitx}",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Calibration on CBIS-DDSM (5$\\times$5 CV, pooled). "
        "ECE and MCE: lower is better. Brier and NLL: lower is better. "
        "Mean $\\pm$ std over 5 seeds reported; best value per column in "
        "\\textbf{bold}.}",
        "  \\label{tab:calibration}",
        "  \\begin{tabular}{l "
        "S[table-format=1.4] S[table-format=1.4] "
        "S[table-format=1.4] S[table-format=1.4]}",
        "    \\toprule",
        "    \\textbf{Method} & {\\textbf{ECE}$\\downarrow$} "
        "& {\\textbf{MCE}$\\downarrow$} "
        "& {\\textbf{Brier}$\\downarrow$} "
        "& {\\textbf{NLL}$\\downarrow$} \\\\",
        "    \\midrule",
    ]

    # identify best (lowest) per column among all models
    col_keys = ["ece_mean", "mce_mean", "brier_mean", "nll_mean"]
    col_vals = {k: [] for k in col_keys}
    for name in MODEL_ORDER:
        if name not in results:
            continue
        for k in col_keys:
            col_vals[k].append(results[name][k])
    best = {k: min(v) for k, v in col_vals.items() if v}

    def fmt(name, key_mean, key_std):
        v = results[name][key_mean]
        s = results[name][key_std]
        cell = f"{v:.4f} \\pm {s:.4f}"
        if abs(v - best.get(key_mean, np.inf)) < 1e-8:
            cell = "\\mathbf{" + cell + "}"
        return "{" + f"${cell}$" + "}"

    for name in MODEL_ORDER:
        if name not in results:
            continue
        prefix = "\\textbf{" + name + "}" if name == ref_name else name
        row = (f"    {prefix} & "
               f"{fmt(name,'ece_mean','ece_std')} & "
               f"{fmt(name,'mce_mean','mce_std')} & "
               f"{fmt(name,'brier_mean','brier_std')} & "
               f"{fmt(name,'nll_mean','nll_std')} \\\\")
        lines.append(row)

    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    Path(out_path).write_text("\n".join(lines))
    print(f"[tex] {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir",  default=None,
                    help="dir scanned for *_preds.csv (after housekeeping, "
                         "holds exactly the 5 Table-II files)")
    ap.add_argument("--pred_csv",  nargs="+", default=None,
                    help="explicit list of prediction CSVs (alternative to --pred_dir)")
    ap.add_argument("--reference", default="CUED-Net",
                    help="model name to bold in the table")
    ap.add_argument("--n_bins",    type=int, default=N_BINS)
    ap.add_argument("--out",       default="./calib_out")
    args = ap.parse_args()

    if args.pred_csv:
        csv_paths = args.pred_csv
    elif args.pred_dir:
        csv_paths = sorted(glob.glob(os.path.join(args.pred_dir, "*_preds.csv")))
    else:
        raise SystemExit("provide --pred_dir or --pred_csv")
    if not csv_paths:
        raise SystemExit("no CSVs found — check path")
    print("Loading:", *csv_paths, sep="\n  ")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    models_data = load_models(csv_paths)

    results = {}
    print("\n── Calibration metrics (pooled) ──")
    for name in MODEL_ORDER:
        if name not in models_data:
            print(f"  [skip] {name} — not found in CSVs")
            continue
        df = models_data[name]
        y  = df["label"].to_numpy()
        p  = df["prob_malignant"].to_numpy()

        # pooled metrics
        ece_p   = ece(y, p, args.n_bins)
        mce_p   = mce(y, p, args.n_bins)
        brier_p = brier(y, p)
        nll_p   = nll(y, p)

        # per-seed mean ± std (for Table)
        seed_df = per_seed_metrics(df)
        results[name] = {
            "ece_pooled":   ece_p,
            "mce_pooled":   mce_p,
            "brier_pooled": brier_p,
            "nll_pooled":   nll_p,
            "ece_mean":     float(seed_df["ece"].mean()),
            "ece_std":      float(seed_df["ece"].std(ddof=1)),
            "mce_mean":     float(seed_df["mce"].mean()),
            "mce_std":      float(seed_df["mce"].std(ddof=1)),
            "brier_mean":   float(seed_df["brier"].mean()),
            "brier_std":    float(seed_df["brier"].std(ddof=1)),
            "nll_mean":     float(seed_df["nll"].mean()),
            "nll_std":      float(seed_df["nll"].std(ddof=1)),
            "n_samples":    int(len(y)),
            "n_seeds":      int(len(seed_df)),
        }
        print(f"  {name:22s}  ECE={ece_p:.4f}  MCE={mce_p:.4f}  "
              f"Brier={brier_p:.4f}  NLL={nll_p:.4f}")
        print(f"  {'':22s}  seed mean: ECE={results[name]['ece_mean']:.4f}±{results[name]['ece_std']:.4f}  "
              f"Brier={results[name]['brier_mean']:.4f}±{results[name]['brier_std']:.4f}")

    # outputs
    json.dump(results, open(Path(args.out)/"calibration_results.json", "w"), indent=2)
    print(f"\n[ok] -> {Path(args.out)/'calibration_results.json'}")

    write_latex(results, args.reference, Path(args.out)/"table_calibration.tex")

    plot_reliability(models_data, str(Path(args.out)/"reliability_diagram"))

    print("\n── CALIBRATION SUMMARY ──")
    ref = results.get(args.reference, {})
    for name in MODEL_ORDER:
        if name == args.reference or name not in results:
            continue
        r = results[name]
        delta_ece   = ref.get("ece_mean", 0)   - r["ece_mean"]
        delta_brier = ref.get("brier_mean", 0) - r["brier_mean"]
        delta_nll   = ref.get("nll_mean", 0)   - r["nll_mean"]
        direction   = lambda d: "better" if d < 0 else "worse"
        print(f"  CUED-Net vs {name:18s}:  "
              f"ΔECE={delta_ece:+.4f} ({direction(delta_ece)})  "
              f"ΔBrier={delta_brier:+.4f} ({direction(delta_brier)})  "
              f"ΔNLL={delta_nll:+.4f} ({direction(delta_nll)})")


if __name__ == "__main__":
    main()
