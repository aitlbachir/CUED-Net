#!/usr/bin/env python3
"""
selective_prediction.py — Selective-prediction comparison for JBHI-00149-2026 revision.

Reads the locked 5×5 CV prediction CSVs and produces:
  1. Coverage–F1 curves for all methods + CUED-Net decomposed signals
  2. Coverage–Accuracy curves
  3. Uncertainty-vs-error AUROC table (Mann-Whitney equivalent)
  4. LaTeX table: F1@{50,60,70,80,90,100}% coverage + AUROC_error
  5. Publication-quality matplotlib figure (IEEE color palette)

CSV SCHEMA (locked by train_cv.py / train_cv_baselines*.py):
  Base (all models):
      model, seed, fold, patient_id, label, prob_malignant, predicted, uncertainty

  CUED-Net has ADDITIONAL columns (from CUEDNetEnsemble.predict):
      uncertainty_evidential, uncertainty_ensemble, uncertainty_discordance, uncertainty_total

  uncertainty column semantics:
      CUED-Net      → uncertainty_total  (composite)
      TMC           → vacuity (K/S of DS-combined Dirichlet)
      MC-Dropout    → predictive entropy
      Deep-Ensemble → predictive entropy

USAGE
-----
# All CSVs in one directory (auto-discovery):
  python selective_prediction.py --pred_dir /workspace/cued_net/cv_preds \
      --out /workspace/cued_net/selective_out

# Explicit files:
  python selective_prediction.py \
      --pred_csv cued_net_preds.csv mcdropout_preds.csv ensemble_preds.csv \
               single_view_edl_preds.csv tmc_preds.csv \
      --out ./selective_out

# With synthetic data for testing (no GPU / no pod needed):
  python selective_prediction.py --demo --out ./selective_out
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


# ---------------------------------------------------------------------------
# IEEE journal palette + style constants
# ---------------------------------------------------------------------------
IEEE_BLUE   = "#0057A8"
IEEE_RED    = "#C8102E"
IEEE_GREEN  = "#1B7E3E"
IEEE_ORANGE = "#E87722"
IEEE_PURPLE = "#7B2D8B"
IEEE_GREY   = "#7F7F7F"

COVERAGE_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Model display names and styles
STYLE = {
    # (label, color, linestyle, linewidth, zorder)
    "CUED-Net Ens. (u_total)":  ("CUED-Net Ens. ($u^{\\mathrm{total}}$)", IEEE_BLUE, "-",  2.6, 6),
    "CUED-Net Ens. (u_disc)":   ("CUED-Net Ens. ($u^{\\mathrm{disc}}$)",  IEEE_PURPLE, "-", 2.2, 5),
    "CUED-Net (u_total)":       ("CUED-Net ($u^{\\mathrm{total}}$)",   IEEE_BLUE,   "-",  2.5, 5),
    "CUED-Net (u_disc)":        ("CUED-Net ($u^{\\mathrm{disc}}$)",    IEEE_BLUE,   "--", 2.2, 5),
    "CUED-Net (u_evid)":        ("CUED-Net ($u^{\\mathrm{evid}}$)",    IEEE_BLUE,   ":",  1.8, 3),
    "CUED-Net (u_comb)":        ("CUED-Net ($u^{\\mathrm{comb}}$)",    IEEE_BLUE,   "-",  2.0, 4),
    "CUED-Net (u_ens)":         ("CUED-Net ($u^{\\mathrm{ens}}$)",     IEEE_BLUE,   "-.", 1.8, 3),
    "TMC":                      ("TMC (vacuity)",                       IEEE_RED,    "-",  2.2, 4),
    "MC-Dropout":               ("MC-Dropout (entropy)",                IEEE_ORANGE, "-",  2.2, 4),
    "Deep-Ensemble":            ("Deep-Ensemble (entropy)",             IEEE_GREEN,  "-",  2.2, 4),
    "Single-view-EDL":          ("Single-view EDL",                    IEEE_GREY,   "--", 1.8, 2),
}


# ---------------------------------------------------------------------------
# Utility: F1 and accuracy at a given coverage
# ---------------------------------------------------------------------------
def f1_at_coverage(labels, preds, uncertainty, coverage):
    """Return F1 for the (coverage) fraction of most-confident samples.
    Samples are ranked by ascending uncertainty; the top-confident
    fraction `coverage` is kept.
    """
    n = len(labels)
    k = max(1, int(np.round(coverage * n)))
    order = np.argsort(uncertainty)          # ascending = most confident first
    idx = order[:k]
    y_true = labels[idx]
    y_pred = preds[idx]
    if len(np.unique(y_true)) < 2:
        return np.nan
    return f1_score(y_true, y_pred, zero_division=0)


def acc_at_coverage(labels, preds, uncertainty, coverage):
    n = len(labels)
    k = max(1, int(np.round(coverage * n)))
    order = np.argsort(uncertainty)
    idx = order[:k]
    return accuracy_score(labels[idx], preds[idx])


def selective_curve(labels, preds, uncertainty, grid=COVERAGE_GRID):
    """Returns dict {coverage: (f1, acc)} for each coverage in grid."""
    out = {}
    for c in grid:
        f1 = f1_at_coverage(labels, preds, uncertainty, c)
        ac = acc_at_coverage(labels, preds, uncertainty, c)
        out[c] = (f1, ac)
    return out


# ---------------------------------------------------------------------------
# Risk–coverage curve and AURC / E-AURC
# ---------------------------------------------------------------------------
def _trapz(y, x):
    """Trapezoidal integration, compatible with NumPy 1.x and 2.x."""
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y, x))


def risk_coverage_curve(labels, preds, uncertainty):
    """Full risk–coverage curve over all coverage levels.

    Samples are ranked by ASCENDING uncertainty (most confident first).
    At each coverage k/n (k = 1..n), risk = error rate on the retained top-k.
    Returns (coverages, risks) as arrays of length n.
    """
    n = len(labels)
    error = (preds != labels).astype(float)
    order = np.argsort(uncertainty, kind="mergesort")  # stable
    err_sorted = error[order]
    cum_err = np.cumsum(err_sorted)
    k = np.arange(1, n + 1)
    risks = cum_err / k                 # error rate on retained top-k
    coverages = k / n
    return coverages, risks


def aurc(labels, preds, uncertainty):
    """Area Under the Risk–Coverage curve (lower is better).

    AURC is the mean selective risk integrated over all coverage levels,
    computed via the trapezoidal rule on the risk–coverage curve.
    """
    cov, risk = risk_coverage_curve(labels, preds, uncertainty)
    return _trapz(risk, cov)


def aurc_optimal(labels, preds):
    """Oracle (optimal) AURC: errors ranked last by a perfect uncertainty.

    The best achievable risk–coverage curve keeps all correct predictions
    first, so risk stays 0 until coverage = (1 - error_rate), then rises.
    Used to compute the base-rate-independent E-AURC.
    """
    n = len(labels)
    error = (preds != labels).astype(float)
    # Perfect ranking: all correct (error=0) first, errors last.
    err_sorted = np.sort(error)         # 0s then 1s
    cum_err = np.cumsum(err_sorted)
    k = np.arange(1, n + 1)
    risks = cum_err / k
    coverages = k / n
    return _trapz(risks, coverages)


def eaurc(labels, preds, uncertainty):
    """Excess-AURC = AURC - optimal AURC (Geifman & El-Yaniv, 2018).

    Removes the dependence on the model's base error rate, so methods with
    different full-coverage accuracy can be compared fairly. Lower is better;
    0 means the uncertainty ranking is as good as an oracle.
    """
    return aurc(labels, preds, uncertainty) - aurc_optimal(labels, preds)


# ---------------------------------------------------------------------------
# Patient-clustered paired bootstrap test on AURC difference
# ---------------------------------------------------------------------------
def paired_bootstrap_aurc(
    labels_a, preds_a, unc_a,
    labels_b, preds_b, unc_b,
    patient_ids, n_boot=10000, seed=42, metric="aurc",
):
    """Patient-clustered paired bootstrap on the AURC (or E-AURC) difference.

    Both methods MUST be aligned row-for-row (same rows, same order) — true
    here because every method is evaluated on the identical pooled CV val set
    in the same order. We resample PATIENTS (clusters) with replacement to
    respect within-patient correlation (multiple breast pairs per patient),
    then recompute the metric for each method on the resampled rows.

    Parameters
    ----------
    metric : "aurc" (raw, lower better) or "eaurc" (excess over oracle).

    Returns
    -------
    dict with observed Δ (= A - B), bootstrap mean, percentile 95% CI,
    and a two-sided bootstrap p-value (H0: Δ = 0).
    Negative Δ means method A has LOWER risk (BETTER selective prediction).
    """
    rng = np.random.default_rng(seed)
    fn = aurc if metric == "aurc" else eaurc

    # Group row indices by patient (cluster).
    uniq_pids, inverse = np.unique(patient_ids, return_inverse=True)
    clusters = [np.where(inverse == i)[0] for i in range(len(uniq_pids))]
    n_clusters = len(clusters)

    def metric_pair(idx):
        a = fn(labels_a[idx], preds_a[idx], unc_a[idx])
        b = fn(labels_b[idx], preds_b[idx], unc_b[idx])
        return a, b

    obs_a, obs_b = metric_pair(np.arange(len(labels_a)))
    obs_delta = obs_a - obs_b

    deltas = np.empty(n_boot)
    for t in range(n_boot):
        pick = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([clusters[j] for j in pick])
        a, b = metric_pair(idx)
        deltas[t] = a - b

    # Two-sided bootstrap p-value: proportion of resamples on the opposite
    # side of 0 from the observed effect, doubled (standard percentile test).
    if obs_delta < 0:
        p = 2.0 * np.mean(deltas >= 0.0)
    else:
        p = 2.0 * np.mean(deltas <= 0.0)
    p = float(min(1.0, p))

    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "metric":        metric,
        "obs_a":         float(obs_a),
        "obs_b":         float(obs_b),
        "obs_delta":     float(obs_delta),   # A - B; <0 => A better (lower risk)
        "boot_mean":     float(deltas.mean()),
        "ci95":          [float(ci_lo), float(ci_hi)],
        "p_value":       p,
        "n_boot":        n_boot,
        "n_clusters":    int(n_clusters),
        "favored":       "A" if obs_delta < 0 else "B",
    }


# ---------------------------------------------------------------------------
# AUROC of uncertainty vs misclassification (higher uncertainty → error)
# ---------------------------------------------------------------------------
def auroc_error(labels, preds, uncertainty):
    """AUROC for the binary task: is this sample misclassified?
    uncertainty is the score; misclassified=1 is the positive class.
    Higher AUROC = uncertainty better predicts errors.
    """
    error = (preds != labels).astype(int)
    if len(np.unique(error)) < 2:
        return np.nan, np.nan
    auroc = roc_auc_score(error, uncertainty)
    # Mann-Whitney U p-value (equivalent to AUROC test)
    unc_err  = uncertainty[error == 1]
    unc_corr = uncertainty[error == 0]
    if len(unc_err) == 0 or len(unc_corr) == 0:
        return auroc, np.nan
    _, p = stats.mannwhitneyu(unc_err, unc_corr, alternative="greater")
    return float(auroc), float(p)


# ---------------------------------------------------------------------------
# Load CSVs and build the method → (labels, preds, uncertainty_dict) map
# ---------------------------------------------------------------------------
def load_and_pool(csv_paths):
    """
    Load all prediction CSVs, pool across all 25 (seed, fold) cells,
    and return per-method frames sorted by (seed, fold, within-cell row).

    Returns
    -------
    methods : dict[str → dict]
        Keys: model name
        Values: {
            'labels': np.ndarray,
            'preds':  np.ndarray,
            'uncertainty_signals': {signal_name: np.ndarray},
        }
    """
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    # Deduplicate: keep last occurrence per (model, seed, fold, patient_id, within-cell row)
    all_df["_row"] = all_df.groupby(["model", "seed", "fold"]).cumcount()
    all_df = all_df.drop_duplicates(subset=["model", "seed", "fold", "_row"], keep="last")

    methods = {}
    for model_name, gdf in all_df.groupby("model"):
        gdf = gdf.sort_values(["seed", "fold", "_row"]).reset_index(drop=True)
        labels = gdf["label"].to_numpy().astype(int)
        preds  = gdf["predicted"].to_numpy().astype(int)
        unc    = gdf["uncertainty"].to_numpy().astype(float)
        pids   = (gdf["patient_id"].astype(str).to_numpy()
                  if "patient_id" in gdf.columns
                  else np.arange(len(gdf)).astype(str))

        signals = {"native": unc}  # native uncertainty for this model

        # CUED-Net: extract decomposed signals if available
        for col, key in [
            ("uncertainty_total",       "u_total"),
            ("uncertainty_combined",    "u_combined"),
            ("uncertainty_evidential",  "u_evid"),
            ("uncertainty_ensemble",    "u_ens"),
            ("uncertainty_discordance", "u_disc"),
        ]:
            if col in gdf.columns and not gdf[col].isna().all():
                signals[key] = gdf[col].to_numpy().astype(float)

        methods[model_name] = {
            "labels":             labels,
            "preds":              preds,
            "patient_ids":        pids,
            "uncertainty_signals": signals,
            "n":                  len(labels),
        }
        col_list = list(signals.keys())
        print(f"  Loaded '{model_name}': {len(labels):5d} rows | signals: {col_list}")

    return methods


# ---------------------------------------------------------------------------
# Build the full set of (display_key, model_name, signal_key) triples
# ---------------------------------------------------------------------------
def build_method_list(methods):
    """
    Produce an ordered list of (display_key, model_name, signal_key) to plot.
    CUED-Net: plot u_total, u_disc, u_evid (skip u_ens — too similar to u_evid
              in practice, keeps figure uncluttered; still reported in table).
    Others:   plot native signal only.
    """
    order = []

    # --- Ensemble CUED-Net (530-row CSV): genuine u_total / u_ens ---
    for ens_key in ["CUED-Net-Ensemble", "CUEDNet-Ensemble", "cued_net_ensemble"]:
        if ens_key in methods:
            sigs = methods[ens_key]["uncertainty_signals"]
            for sig_key, display_key in [
                ("u_total", "CUED-Net Ens. (u_total)"),
                ("u_disc",  "CUED-Net Ens. (u_disc)"),
            ]:
                if sig_key in sigs:
                    order.append((display_key, ens_key, sig_key))
            break

    # --- Per-seed CUED-Net (2650-row CSV): single-model signals ---
    for cued_key in ["CUED-Net", "CUEDNet", "cued_net"]:
        if cued_key in methods:
            sigs = methods[cued_key]["uncertainty_signals"]
            if "u_disc" in sigs:
                # decomposed per-seed CSV present
                order.append(("CUED-Net (u_disc)", cued_key, "u_disc"))
                if "u_evid" in sigs:
                    order.append(("CUED-Net (u_evid)", cued_key, "u_evid"))
                if "u_combined" in sigs:
                    order.append(("CUED-Net (u_comb)", cued_key, "u_combined"))
            else:
                # legacy CSV: only scalar uncertainty (= uncertainty_combined).
                order.append(("CUED-Net (u_comb)", cued_key, "native"))
            break

    # Then competitors
    for name in ["TMC", "Deep-Ensemble", "MC-Dropout", "Single-view-EDL"]:
        for actual_name in methods:
            if name.lower().replace("-", "") in actual_name.lower().replace("-", "").replace("_", ""):
                order.append((name, actual_name, "native"))
                break

    # Any remaining models not captured above
    plotted_models = {mn for _, mn, _ in order}
    for name in methods:
        if name not in plotted_models:
            order.append((name, name, "native"))

    return order


# ---------------------------------------------------------------------------
# Compute selective curves for all methods
# ---------------------------------------------------------------------------
def compute_all_curves(methods, method_list, grid=COVERAGE_GRID):
    """Returns dict[display_key] → {coverage: (f1, acc)}"""
    curves = {}
    for display_key, model_name, sig_key in method_list:
        m = methods[model_name]
        unc = m["uncertainty_signals"][sig_key]
        labels = m["labels"]
        preds  = m["preds"]
        curves[display_key] = selective_curve(labels, preds, unc, grid)
    return curves


# ---------------------------------------------------------------------------
# Compute AUROC-error for all method/signal combos
# ---------------------------------------------------------------------------
def compute_auroc_table(methods, method_list):
    """Returns list of dicts for tabulation: AUROC_err, AURC, E-AURC."""
    rows = []
    for display_key, model_name, sig_key in method_list:
        m = methods[model_name]
        unc = m["uncertainty_signals"][sig_key]
        au, p = auroc_error(m["labels"], m["preds"], unc)
        a = aurc(m["labels"], m["preds"], unc)
        ea = eaurc(m["labels"], m["preds"], unc)
        rows.append({
            "display":    display_key,
            "model":      model_name,
            "signal":     sig_key,
            "auroc_err":  au,
            "mw_p":       p,
            "aurc":       a,
            "eaurc":      ea,
            "n":          m["n"],
        })
    return rows


# ---------------------------------------------------------------------------
# Delta F1: gain from selective prediction
# ---------------------------------------------------------------------------
def delta_f1(curves, display_key, coverage, baseline_coverage=1.0):
    """F1(coverage) - F1(1.0)"""
    f1_c  = curves[display_key].get(coverage,      (np.nan, np.nan))[0]
    f1_b  = curves[display_key].get(baseline_coverage, (np.nan, np.nan))[0]
    return f1_c - f1_b


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------
def build_latex_table(curves, auroc_rows, method_list, grid=COVERAGE_GRID):
    """IEEE booktabs + siunitx table.

    Columns: Method | F1@50% | F1@60% | F1@70% | F1@80% | F1@90% | F1@100% | ΔAUC_err
    """
    cov_cols = sorted([c for c in grid])

    lines = [
        "% === SELECTIVE PREDICTION TABLE (booktabs + siunitx) ===",
        "% \\usepackage{booktabs} \\usepackage{siunitx} \\usepackage{multirow}",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Selective-prediction performance on CBIS-DDSM (5$\\times$5 CV, pooled). "
        "F1 score at each coverage level, computed by retaining the "
        "$(c \\times 100)\\%$ most-confident predictions ranked by each method's native "
        "uncertainty signal. "
        "$\\Delta$F1 = F1 at 70\\% coverage minus full-coverage F1. "
        "AUROC\\textsubscript{err} = AUROC of the uncertainty signal for predicting "
        "misclassification (higher is better). "
        "AURC = area under the risk--coverage curve and E-AURC = excess AURC over "
        "the optimal ranking (both lower is better). "
        "$p$ = Mann--Whitney $U$ (one-sided, uncertain $>$ correct).}",
        "  \\label{tab:selective_prediction}",
    ]

    # Column spec: l | F1 cols | ΔF1 | AUROC | AURC | E-AURC
    ncov = len(cov_cols)
    col_spec = ("l " + " ".join(["S[table-format=1.3]"] * ncov)
                + " S[table-format=+1.3] S[table-format=1.3]"
                + " S[table-format=1.4] S[table-format=1.4]")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")

    # Header row
    cov_headers = " & ".join([f"{{F1@{int(c*100)}\\%}}" for c in cov_cols])
    lines.append(f"    \\textbf{{Method}} & {cov_headers} & "
                 "{$\\Delta$F1\\textsubscript{70\\%}} & "
                 "{AUROC\\textsubscript{err}} & "
                 "{AURC$\\downarrow$} & {E-AURC$\\downarrow$} \\\\")
    lines.append("    \\midrule")

    # Build display_key → metrics lookup
    auroc_lookup = {r["display"]: r for r in auroc_rows}

    # Method rows — CUED-Net first group, then separator, then baselines
    cued_keys = [dk for dk, _, _ in method_list if "CUED-Net" in dk]
    other_keys = [dk for dk, _, _ in method_list if "CUED-Net" not in dk]

    def fmt_f1(v):
        return f"{v:.3f}" if not np.isnan(v) else "---"

    def fmt4(v):
        return f"{v:.4f}" if (v is not None and not np.isnan(v)) else "---"

    def fmt_auroc(v, p):
        if np.isnan(v):
            return "---"
        sig = "$^{*}$" if (not np.isnan(p) and p < 0.05) else ""
        return f"{v:.3f}{sig}"

    def method_row(dk):
        # Use latex-compatible display name from STYLE dict if available
        label = STYLE.get(dk, (dk,))[0]
        cells = []
        for c in cov_cols:
            f1v, _ = curves[dk].get(c, (np.nan, np.nan))
            cells.append(fmt_f1(f1v))
        df1 = delta_f1(curves, dk, 0.70)
        cells.append(fmt_f1(df1))
        r = auroc_lookup.get(dk, {})
        cells.append(fmt_auroc(r.get("auroc_err", np.nan), r.get("mw_p", np.nan)))
        cells.append(fmt4(r.get("aurc", np.nan)))
        cells.append(fmt4(r.get("eaurc", np.nan)))
        cell_str = " & ".join(cells)
        return f"    {label} & {cell_str} \\\\"

    for dk in cued_keys:
        lines.append(method_row(dk))
    if cued_keys and other_keys:
        lines.append("    \\midrule")
    for dk in other_keys:
        lines.append(method_row(dk))

    lines += [
        "    \\bottomrule",
        "    \\multicolumn{" + str(ncov + 5) + "}{l}{\\footnotesize "
        "$^{*}$ Mann--Whitney $U$ $p < 0.05$ (uncertainty significantly higher for errors). "
        "AURC/E-AURC lower is better.}",
        "  \\end{tabular}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure: Coverage–F1 (main panel) + Coverage–Acc (inset or panel B)
# ---------------------------------------------------------------------------
def plot_selective(curves, method_list, auroc_rows, out_dir, grid=COVERAGE_GRID):
    grid_sorted = sorted(grid)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.35)

    auroc_lookup = {r["display"]: r["auroc_err"] for r in auroc_rows}

    for ax_idx, (metric_idx, ylabel, title) in enumerate([
        (0, "F1 Score", "(a) Coverage vs. F1"),
        (1, "Accuracy", "(b) Coverage vs. Accuracy"),
    ]):
        ax = axes[ax_idx]
        # Plot CUED-Net decomposed last (on top)
        cued_entries = [(dk, mn, sk) for dk, mn, sk in method_list if "CUED-Net" in dk]
        other_entries = [(dk, mn, sk) for dk, mn, sk in method_list if "CUED-Net" not in dk]

        for entry_list in [other_entries, cued_entries]:
            for display_key, model_name, sig_key in entry_list:
                vals = [curves[display_key][c][metric_idx] for c in grid_sorted]
                style = STYLE.get(display_key, (display_key, IEEE_GREY, "-", 1.5, 2))
                label_str, color, ls, lw, zo = style

                # Add AUROC_err annotation in legend label
                au = auroc_lookup.get(display_key, np.nan)
                if not np.isnan(au):
                    legend_label = f"{label_str}  (AUROC$_{{\\mathrm{{err}}}}$={au:.3f})"
                else:
                    legend_label = label_str

                ax.plot(
                    [c * 100 for c in grid_sorted], vals,
                    color=color, linestyle=ls, linewidth=lw, zorder=zo,
                    marker="o", markersize=4, markerfacecolor="white", markeredgewidth=lw * 0.7,
                    label=legend_label,
                )

        # Reference line: CUED-Net full-coverage F1 (baseline)
        cued_dk = next((dk for dk, _, _ in cued_entries if "u_total" in dk or "u_disc" not in dk), None)
        if cued_dk is None and cued_entries:
            cued_dk = cued_entries[0][0]
        if cued_dk:
            fc_val = curves[cued_dk].get(1.0, (np.nan, np.nan))[metric_idx]
            if not np.isnan(fc_val):
                ax.axhline(fc_val, color=IEEE_BLUE, linestyle=":", linewidth=1.0,
                           alpha=0.5, label=f"CUED-Net full-cov. ({metric_idx and 'Acc' or 'F1'}={fc_val:.3f})")

        ax.set_xlabel("Coverage (%)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlim(48, 102)
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%g%%"))
        ax.tick_params(labelsize=9)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
        if ax_idx == 0:
            ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9,
                      ncol=1, handlelength=2.5)

    fig.suptitle("Selective Prediction Analysis — CUED-Net vs. Baselines (CBIS-DDSM 5×5 CV)",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()

    pdf_path = Path(out_dir) / "selective_prediction.pdf"
    png_path = Path(out_dir) / "selective_prediction.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Figure saved: {pdf_path}")
    print(f"  Figure saved: {png_path}")
    return pdf_path, png_path


# ---------------------------------------------------------------------------
# JSON results
# ---------------------------------------------------------------------------
def build_json_results(curves, auroc_rows, method_list, grid=COVERAGE_GRID):
    out = {"coverage_grid": grid, "methods": {}}
    auroc_lookup = {r["display"]: r for r in auroc_rows}
    for display_key, model_name, sig_key in method_list:
        entry = {
            "model":  model_name,
            "signal": sig_key,
            "selective_f1":  {},
            "selective_acc": {},
            "delta_f1_70":   float(delta_f1(curves, display_key, 0.70)),
            "delta_f1_50":   float(delta_f1(curves, display_key, 0.50)),
        }
        for c in grid:
            f1v, acv = curves[display_key][c]
            entry["selective_f1"][str(c)]  = float(f1v) if not np.isnan(f1v) else None
            entry["selective_acc"][str(c)] = float(acv) if not np.isnan(acv) else None
        if display_key in auroc_lookup:
            r = auroc_lookup[display_key]
            entry["auroc_err"] = float(r["auroc_err"]) if not np.isnan(r["auroc_err"]) else None
            entry["mw_p"]      = float(r["mw_p"])      if (r["mw_p"] is not None and not np.isnan(r["mw_p"])) else None
            entry["aurc"]      = float(r["aurc"])      if ("aurc" in r and not np.isnan(r["aurc"])) else None
            entry["eaurc"]     = float(r["eaurc"])     if ("eaurc" in r and not np.isnan(r["eaurc"])) else None
        out["methods"][display_key] = entry
    return out


# ---------------------------------------------------------------------------
# Demo mode: generate synthetic data to test the pipeline on CPU
# ---------------------------------------------------------------------------
def generate_demo_data(out_dir):
    """Generates synthetic 5×5 CV CSVs that mimic the real schema.

    Key properties:
    - CUED-Net: triple uncertainty with u_disc being the most informative signal
    - TMC: scalar vacuity, slightly less informative than u_disc
    - MC-Dropout / Deep-Ensemble: entropy, moderate informativeness
    - Realistic AUC ordering: CUED-Net ≈ TMC > Deep-Ens > MC-Drop > EDL
    """
    rng = np.random.default_rng(42)
    seeds = [42, 123, 456, 789, 2024]
    folds = [0, 1, 2, 3, 4]
    n_per_cell = 106  # ≈ 530/25

    rows_cued, rows_tmc, rows_mc, rows_ens, rows_edl = [], [], [], [], []

    for seed in seeds:
        for fold in folds:
            n = n_per_cell
            # Ground truth: balanced
            labels = rng.integers(0, 2, size=n)
            patient_ids = [f"P{seed}F{fold}_{i:03d}" for i in range(n)]

            # ---- CUED-Net ----
            # u_evid: Dirichlet vacuity (low values, signal ~0.4 for errors)
            u_evid = rng.beta(2, 5, n)
            u_evid[labels == 1] += 0.05   # slightly higher for positives
            # u_ens: ensemble std, small
            u_ens  = rng.beta(1, 8, n) * 0.3
            # u_disc: view discordance — most informative; errors have higher u_disc
            u_disc_base = rng.beta(1.5, 6, n)
            # Simulate: u_disc is higher for errors (key claim)
            prob_mal = rng.beta(5, 2, n) * labels + rng.beta(2, 5, n) * (1 - labels)
            prob_mal = np.clip(prob_mal, 0.01, 0.99)
            pred_cued = (prob_mal >= 0.5).astype(int)
            errors = (pred_cued != labels)
            u_disc = u_disc_base.copy()
            u_disc[errors] += 0.15 + rng.uniform(0, 0.1, errors.sum())
            u_disc = np.clip(u_disc, 0, 1)
            u_total = 0.4 * u_evid + 0.3 * u_ens + 0.3 * u_disc
            for i in range(n):
                rows_cued.append({
                    "model": "CUED-Net", "seed": seed, "fold": fold,
                    "patient_id": patient_ids[i], "label": int(labels[i]),
                    "prob_malignant": float(prob_mal[i]),
                    "predicted": int(pred_cued[i]),
                    "uncertainty": float(u_total[i]),
                    "uncertainty_evidential": float(u_evid[i]),
                    "uncertainty_ensemble": float(u_ens[i]),
                    "uncertainty_discordance": float(u_disc[i]),
                    "uncertainty_total": float(u_total[i]),
                })

            # ---- TMC (DS-combined vacuity — scalar, slightly less informative) ----
            prob_tmc = rng.beta(4.8, 2.2, n) * labels + rng.beta(2.2, 4.8, n) * (1 - labels)
            prob_tmc = np.clip(prob_tmc, 0.01, 0.99)
            pred_tmc = (prob_tmc >= 0.5).astype(int)
            err_tmc  = (pred_tmc != labels)
            u_tmc    = rng.beta(2, 7, n) * 0.4  # combined vacuity
            u_tmc[err_tmc] += 0.10 + rng.uniform(0, 0.08, err_tmc.sum())
            u_tmc = np.clip(u_tmc, 0, 1)
            for i in range(n):
                rows_tmc.append({
                    "model": "TMC", "seed": seed, "fold": fold,
                    "patient_id": patient_ids[i], "label": int(labels[i]),
                    "prob_malignant": float(prob_tmc[i]),
                    "predicted": int(pred_tmc[i]),
                    "uncertainty": float(u_tmc[i]),
                })

            # ---- MC-Dropout (entropy) ----
            prob_mc = rng.beta(4.5, 2.5, n) * labels + rng.beta(2.5, 4.5, n) * (1 - labels)
            prob_mc = np.clip(prob_mc, 0.01, 0.99)
            pred_mc = (prob_mc >= 0.5).astype(int)
            err_mc  = (pred_mc != labels)
            # entropy: H = -p log p - (1-p) log(1-p), max = ln2 ≈ 0.693
            ent_mc  = -(prob_mc * np.log(prob_mc + 1e-9) + (1-prob_mc) * np.log(1-prob_mc + 1e-9))
            ent_mc += rng.normal(0, 0.03, n)
            ent_mc[err_mc] += 0.05
            ent_mc = np.clip(ent_mc, 0, np.log(2))
            for i in range(n):
                rows_mc.append({
                    "model": "MC-Dropout", "seed": seed, "fold": fold,
                    "patient_id": patient_ids[i], "label": int(labels[i]),
                    "prob_malignant": float(prob_mc[i]),
                    "predicted": int(pred_mc[i]),
                    "uncertainty": float(ent_mc[i]),
                })

            # ---- Deep-Ensemble (entropy) ----
            prob_ens = rng.beta(5, 2, n) * labels + rng.beta(2, 5, n) * (1 - labels)
            prob_ens = np.clip(prob_ens, 0.01, 0.99)
            pred_ens = (prob_ens >= 0.5).astype(int)
            err_ens  = (pred_ens != labels)
            ent_ens  = -(prob_ens * np.log(prob_ens + 1e-9) + (1-prob_ens) * np.log(1-prob_ens + 1e-9))
            ent_ens[err_ens] += 0.04
            ent_ens = np.clip(ent_ens, 0, np.log(2))
            for i in range(n):
                rows_ens.append({
                    "model": "Deep-Ensemble", "seed": seed, "fold": fold,
                    "patient_id": patient_ids[i], "label": int(labels[i]),
                    "prob_malignant": float(prob_ens[i]),
                    "predicted": int(pred_ens[i]),
                    "uncertainty": float(ent_ens[i]),
                })

            # ---- Single-view EDL ----
            prob_edl = rng.beta(3.5, 2.5, n) * labels + rng.beta(2.5, 3.5, n) * (1 - labels)
            prob_edl = np.clip(prob_edl, 0.01, 0.99)
            pred_edl = (prob_edl >= 0.5).astype(int)
            err_edl  = (pred_edl != labels)
            u_edl    = rng.beta(2, 4, n) * 0.7 + 0.1  # high vacuity (single view)
            u_edl[err_edl] += 0.05
            u_edl = np.clip(u_edl, 0, 1)
            for i in range(n):
                rows_edl.append({
                    "model": "Single-view-EDL", "seed": seed, "fold": fold,
                    "patient_id": patient_ids[i], "label": int(labels[i]),
                    "prob_malignant": float(prob_edl[i]),
                    "predicted": int(pred_edl[i]),
                    "uncertainty": float(u_edl[i]),
                })

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_paths = []
    for rows, fname in [
        (rows_cued, "cued_net_preds.csv"),
        (rows_tmc,  "tmc_preds.csv"),
        (rows_mc,   "mcdropout_preds.csv"),
        (rows_ens,  "ensemble_preds.csv"),
        (rows_edl,  "single_view_edl_preds.csv"),
    ]:
        p = Path(out_dir) / fname
        pd.DataFrame(rows).to_csv(p, index=False)
        csv_paths.append(str(p))
        print(f"  [demo] written {p} ({len(rows)} rows)")
    return csv_paths


# ---------------------------------------------------------------------------
# Build matched arrays for a paired comparison between two (model, signal) refs
# ---------------------------------------------------------------------------
def build_aligned(methods, ref_a, ref_b):
    """Return row-aligned (labels, preds_a, unc_a, preds_b, unc_b, patient_ids)
    for two references ref = (model_name, signal_key).

    Three cases:
      (1) both methods have the SAME row count and same patient order
          -> direct alignment (per-seed vs per-seed, 2650 rows; or
             ensemble vs ensemble, 530 rows).
      (2) one is the 530-row CUED-Net Ensemble and the other is a 2650-row
          per-seed baseline -> aggregate the baseline across its 5 seed blocks
          (mean prob / mean uncertainty per pair) to 530 rows, then align.
    Labels and patient order are taken from the (shorter) reference.
    Returns None if alignment is impossible.
    """
    (ma, sa), (mb, sb) = ref_a, ref_b
    A, B = methods[ma], methods[mb]
    la, pa = A["labels"], A["preds"]
    ua = A["uncertainty_signals"][sa]
    pida = A["patient_ids"]
    lb, pb = B["labels"], B["preds"]
    ub = B["uncertainty_signals"][sb]
    pidb = B["patient_ids"]

    # Case 1: same length -> assume same pooled order (verified true for all
    # per-seed methods; and for ensemble-vs-ensemble).
    if len(la) == len(lb):
        if not np.array_equal(pida, pidb):
            # Same length but different patient order is unexpected; bail.
            return None
        return la, pa, ua, pb, ub, pida

    # Case 2: aggregate the longer (per-seed) method to pair level.
    def agg_to_pairs(labels, preds, unc, pids):
        """Average across the 5 stacked seed blocks. Verified structure:
        the pooled array is 5 seed blocks of equal length L, each block in the
        identical (fold, pair) order. So index j in [0,L) corresponds to the
        same pair across all 5 blocks."""
        n = len(labels)
        if n % 5 != 0:
            return None
        L = n // 5
        blocks_u = unc.reshape(5, L)
        # prob not stored here; we aggregate uncertainty and recompute pred by
        # majority vote across seeds (label-preserving). For risk/AURC we need
        # preds + labels + a ranking score (unc).
        blocks_p = preds.reshape(5, L)
        mean_u = blocks_u.mean(axis=0)
        maj_pred = (blocks_p.mean(axis=0) >= 0.5).astype(int)
        lab0 = labels.reshape(5, L)[0]
        pid0 = pids.reshape(5, L)[0]
        return lab0, maj_pred, mean_u, pid0

    if len(la) > len(lb):  # A is per-seed, B is ensemble
        agg = agg_to_pairs(la, pa, ua, pida)
        if agg is None or len(agg[0]) != len(lb):
            return None
        la2, pa2, ua2, pida2 = agg
        if not np.array_equal(pida2, pidb):
            return None
        return lb, pa2, ua2, pb, ub, pidb
    else:                  # B is per-seed, A is ensemble
        agg = agg_to_pairs(lb, pb, ub, pidb)
        if agg is None or len(agg[0]) != len(la):
            return None
        lb2, pb2, ub2, pidb2 = agg
        if not np.array_equal(pida, pidb2):
            return None
        return la, pa, ua, pb2, ub2, pida


def run_bootstrap_comparisons(methods, comparisons, n_boot, seed, out_dir):
    """Run paired-bootstrap AURC + E-AURC tests for a list of comparisons.

    comparisons : list of (label, ref_a, ref_b) where each ref is
                  (model_name, signal_key). ref_a is the method of interest
                  (CUED-Net); a NEGATIVE Δ favours ref_a (lower risk = better).
    """
    print("\n" + "=" * 84)
    print("PAIRED BOOTSTRAP — AURC (lower = better selective prediction)")
    print("  Δ = AURC(A) − AURC(B);  Δ<0 ⇒ A better.  Patient-clustered, "
          f"{n_boot} resamples.")
    print("=" * 84)
    results = []
    for label, ref_a, ref_b in comparisons:
        aligned = build_aligned(methods, ref_a, ref_b)
        if aligned is None:
            print(f"[skip] {label}: could not align rows "
                  f"({ref_a} vs {ref_b}).")
            continue
        la, pa, ua, pb, ub, pid = aligned
        for metric in ("aurc", "eaurc"):
            r = paired_bootstrap_aurc(
                la, pa, ua, la, pb, ub, pid,
                n_boot=n_boot, seed=seed, metric=metric)
            r["label"] = label
            r["ref_a"] = f"{ref_a[0]}:{ref_a[1]}"
            r["ref_b"] = f"{ref_b[0]}:{ref_b[1]}"
            results.append(r)
            tag = "AURC " if metric == "aurc" else "E-AURC"
            sig = "*" if r["p_value"] < 0.05 else " "
            better = "A(CUED-Net)" if r["obs_delta"] < 0 else "B(baseline)"
            print(f"\n  [{tag}] {label}")
            print(f"     A={r['obs_a']:.4f}  B={r['obs_b']:.4f}  "
                  f"Δ={r['obs_delta']:+.4f}  "
                  f"95%CI[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  "
                  f"p={r['p_value']:.4f}{sig}  → favours {better}")
    print("=" * 84)

    out = Path(out_dir) / "bootstrap_aurc.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[+] Bootstrap results saved: {out}")
    return results


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_summary(curves, auroc_rows, method_list, grid=COVERAGE_GRID):
    grid_sorted = sorted(grid)
    print("\n" + "=" * 96)
    print("SELECTIVE PREDICTION SUMMARY  (AURC/E-AURC: lower = better)")
    print("=" * 96)
    hdr = (f"{'Method':<28s}" + "".join([f" F1@{int(c*100):2d}%" for c in grid_sorted])
           + "  ΔF1@70  AUROCe   AURC   E-AURC")
    print(hdr)
    print("-" * len(hdr))
    auroc_lookup = {r["display"]: r for r in auroc_rows}
    for display_key, model_name, sig_key in method_list:
        f1s = [f"{curves[display_key][c][0]:.3f}" for c in grid_sorted]
        df1 = delta_f1(curves, display_key, 0.70)
        r   = auroc_lookup.get(display_key, {})
        au  = r.get("auroc_err", np.nan)
        a   = r.get("aurc", np.nan)
        ea  = r.get("eaurc", np.nan)
        au_str = f"{au:.3f}" if not np.isnan(au) else " --- "
        a_str  = f"{a:.4f}" if not np.isnan(a) else " --- "
        ea_str = f"{ea:.4f}" if not np.isnan(ea) else " --- "
        df1_str = f"{df1:+.3f}" if not np.isnan(df1) else " --- "
        print(f"{display_key:<28s}" + "  ".join(f1s)
              + f"  {df1_str}  {au_str}  {a_str}  {ea_str}")
    print("=" * 96)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Selective-prediction comparison for JBHI revision"
    )
    ap.add_argument("--pred_dir", default=None,
                    help="Directory to scan for *preds*.csv files")
    ap.add_argument("--pred_csv", nargs="+", default=None,
                    help="Explicit list of CSV paths")
    ap.add_argument("--out", default="./selective_out",
                    help="Output directory for figures, table, JSON")
    ap.add_argument("--demo", action="store_true",
                    help="Generate synthetic data and run demo pipeline")
    ap.add_argument("--n_boot", type=int, default=10000,
                    help="Bootstrap resamples for the paired AURC test")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # ---- Collect CSV paths ----
    if args.demo:
        print("[demo] Generating synthetic 5×5 CV data …")
        demo_dir = Path(args.out) / "demo_csvs"
        csv_paths = generate_demo_data(str(demo_dir))
    elif args.pred_csv:
        csv_paths = args.pred_csv
    elif args.pred_dir:
        csv_paths = sorted(glob.glob(os.path.join(args.pred_dir, "*preds*.csv")))
        if not csv_paths:
            # Also try without underscore
            csv_paths = sorted(glob.glob(os.path.join(args.pred_dir, "*.csv")))
    else:
        ap.print_help()
        print("\n[error] Provide --pred_dir, --pred_csv, or --demo")
        sys.exit(1)

    if not csv_paths:
        print("[error] No CSV files found.")
        sys.exit(1)

    print(f"\n[+] Loading {len(csv_paths)} CSV(s) …")
    for p in csv_paths:
        print(f"    {p}")

    # ---- Load & pool ----
    methods = load_and_pool(csv_paths)
    if not methods:
        print("[error] No models loaded — check CSV schema.")
        sys.exit(1)

    # ---- Build ordered method list ----
    method_list = build_method_list(methods)
    print(f"\n[+] Method / signal pairs to analyse:")
    for dk, mn, sk in method_list:
        print(f"    {dk:<32s}  (model={mn}, signal={sk})")

    # ---- Selective curves ----
    print("\n[+] Computing selective-prediction curves …")
    curves = compute_all_curves(methods, method_list)

    # ---- AUROC error ----
    print("\n[+] Computing AUROC_err (uncertainty vs misclassification) …")
    auroc_rows = compute_auroc_table(methods, method_list)

    # ---- Console summary ----
    print_summary(curves, auroc_rows, method_list)

    # ---- Figure ----
    print("\n[+] Generating figure …")
    pdf_path, png_path = plot_selective(curves, method_list, auroc_rows, args.out)

    # ---- LaTeX table ----
    print("\n[+] Building LaTeX table …")
    tex = build_latex_table(curves, auroc_rows, method_list)
    tex_path = Path(args.out) / "table_selective_prediction.tex"
    tex_path.write_text(tex)
    print(f"  LaTeX table saved: {tex_path}")

    # ---- JSON results ----
    results_json = build_json_results(curves, auroc_rows, method_list)
    json_path = Path(args.out) / "selective_results.json"
    json_path.write_text(json.dumps(results_json, indent=2))
    print(f"  JSON results saved: {json_path}")

    # ---- Paired bootstrap AURC comparisons (the significance test) ----
    # Pick the CUED-Net reference of interest, preferring the ensemble u_total
    # (the headline operating mode), then per-seed signals as fallback.
    def first_present(cands):
        for mn, sk in cands:
            if mn in methods and sk in methods[mn]["uncertainty_signals"]:
                return (mn, sk)
        return None

    cued_ref = first_present([
        ("CUED-Net-Ensemble", "u_total"),
        ("CUED-Net", "u_combined"),
        ("CUED-Net", "native"),
    ])

    # Identify baseline model names actually present.
    def find_model(substr):
        for mn in methods:
            norm = mn.lower().replace("-", "").replace("_", "").replace("(", "").replace(")", "")
            if substr in norm:
                return mn
        return None

    mc_name  = find_model("mcdropout")
    tmc_name = find_model("tmc")
    de_name  = find_model("deepensemble") or find_model("ensemblem5")

    comparisons = []
    if cued_ref is not None:
        if mc_name:
            comparisons.append(("CUED-Net vs MC-Dropout",
                                cued_ref, (mc_name, "native")))
        if tmc_name:
            comparisons.append(("CUED-Net vs TMC",
                                cued_ref, (tmc_name, "native")))
        if de_name:
            comparisons.append(("CUED-Net vs Deep-Ensemble",
                                cued_ref, (de_name, "native")))

    if comparisons:
        print(f"\n[+] CUED-Net reference for bootstrap: {cued_ref}")
        run_bootstrap_comparisons(methods, comparisons,
                                  n_boot=args.n_boot, seed=42, out_dir=args.out)
    else:
        print("\n[!] No valid bootstrap comparisons could be built.")

    print("\n[✓] Done. Outputs in:", args.out)


if __name__ == "__main__":
    main()