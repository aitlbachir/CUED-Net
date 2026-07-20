"""Plot the learning-curve figure."""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

WS = "/workspace/cued_net"
JSON_PATH = os.path.join(WS, "cv_learning_curve", "learning_curve_results.json")
OUT_PDF = os.path.join(WS, "figure3_learning_curve.pdf")
OUT_PNG = os.path.join(WS, "figure3_learning_curve.png")

# ---- known v7 numbers, used ONLY as a labelled fallback if JSON is absent ----
V7_FALLBACK = {
    "10":  dict(mean=0.6642, std=0.0957, min=0.4118, max=0.7634),
    "25":  dict(mean=0.7430, std=0.0388, min=0.6667, max=0.7934),
    "50":  dict(mean=0.7991, std=0.0369, min=0.7255, max=0.8468),
    "75":  dict(mean=0.8103, std=0.0268, min=0.7647, max=0.8485),
    "100": dict(mean=0.8259, std=0.0116, min=0.8073, max=0.8421),
}


def _get(d, *names, default=None):
    for n in names:
        if n in d:
            return d[n]
    return default


def load_per_fraction():
    if not os.path.exists(JSON_PATH):
        print(f"WARNING: {JSON_PATH} not found — using V7 FALLBACK numbers. "
              f"Re-run on the pod to plot from the authoritative JSON.")
        return V7_FALLBACK, True
    with open(JSON_PATH) as f:
        blob = json.load(f)
    pf = _get(blob, "per_fraction", "per_frac", "fractions", default=None)
    if pf is None:
        # maybe the dict IS the per-fraction map at top level
        pf = blob
    # normalise keys to fraction-as-string ("10","25",...) regardless of "0.1"/"10%"/10
    norm = {}
    for k, v in pf.items():
        ks = str(k).replace("%", "").strip()
        try:
            fv = float(ks)
            pct = int(round(fv * 100)) if fv <= 1.0 else int(round(fv))
        except ValueError:
            continue
        norm[str(pct)] = dict(
            mean=float(_get(v, "mean", "f1_mean", "F1_mean")),
            std=float(_get(v, "std", "f1_std", "F1_std")),
            min=float(_get(v, "min", "f1_min", "F1_min")),
            max=float(_get(v, "max", "f1_max", "F1_max")),
        )
    return norm, False


pf, is_fallback = load_per_fraction()
order = sorted(pf.keys(), key=lambda s: int(s))
x = np.array([int(k) for k in order], dtype=float)
mean = np.array([pf[k]["mean"] for k in order])
std = np.array([pf[k]["std"] for k in order])
lo = np.array([pf[k]["min"] for k in order])
hi = np.array([pf[k]["max"] for k in order])
rng = hi - lo

# ---- IEEE single-column styling: serif, 3.5in wide, vector ----
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "savefig.dpi": 600,
})

fig, ax = plt.subplots(figsize=(3.5, 2.6))

# min-max band (outlier-robust spread, the metric v7 says to lead with)
ax.fill_between(x, lo, hi, color="#4C72B0", alpha=0.15,
                label="min--max range", linewidth=0)
# +/- std band (lighter, secondary)
ax.fill_between(x, mean - std, mean + std, color="#4C72B0", alpha=0.30,
                label=r"mean $\pm$ std", linewidth=0)
# mean F1 curve with markers
ax.plot(x, mean, marker="o", markersize=4, color="#1A3D6D",
        markerfacecolor="white", markeredgewidth=1.0, label="mean F1", zorder=5)

# annotate the contraction story at the endpoints
ax.annotate(f"range = {rng[0]:.3f}", xy=(x[0], hi[0]), xytext=(x[0] + 3, hi[0] + 0.01),
            fontsize=6.5, color="#444444")
ax.annotate(f"range = {rng[-1]:.3f}", xy=(x[-1], hi[-1]), xytext=(x[-1] - 30, hi[-1] + 0.025),
            fontsize=6.5, color="#444444")

ax.set_xlabel("Training-set fraction (%)")
ax.set_ylabel("Single-model F1")
ax.set_xlim(5, 105)
ax.set_ylim(0.38, 0.90)
ax.xaxis.set_major_locator(MultipleLocator(25))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.grid(True, linewidth=0.3, alpha=0.4)
ax.legend(loc="lower right", fontsize=6.5, frameon=False, handlelength=1.5)

fig.tight_layout(pad=0.3)
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
print(f"saved: {OUT_PDF}")
print(f"saved: {OUT_PNG}")
print(f"source: {'V7 FALLBACK (JSON absent)' if is_fallback else JSON_PATH}")
print("\nper-fraction summary used:")
print(f"{'frac':>5} {'mean':>7} {'std':>7} {'min':>7} {'max':>7} {'range':>7}")
for k in order:
    d = pf[k]
    print(f"{k:>5} {d['mean']:>7.4f} {d['std']:>7.4f} {d['min']:>7.4f} "
          f"{d['max']:>7.4f} {d['max']-d['min']:>7.4f}")
print(f"\nstd contraction: {std[0]:.4f} -> {std[-1]:.4f}  ({std[0]/std[-1]:.1f}x)")
print(f"range contraction: {rng[0]:.4f} -> {rng[-1]:.4f}  ({rng[0]/rng[-1]:.1f}x)")
