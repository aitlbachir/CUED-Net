"""Draw the graphical abstract (660 x 295 pixels) from the analysis outputs.

Left: the method. Right: the change in AURC when view discordance uncertainty is added to the two
within-view terms, with its 95% patient-clustered bootstrap interval, for every cohort and backbone.
Every plotted number is read from a p2_analysis.py output; none is typed here.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

W, H = 660, 295
PURPLE, DARK = "#662D91", "#222222"
ARM_COLOR = {"densenet121": "#0072B2", "convnext_tiny": "#E69F00"}
ARM_NAME = {"densenet121": "DenseNet-121", "convnext_tiny": "ConvNeXt-T"}
TERM = {"amb": "#9A9A9A", "evid": "#009E73", "vd": "#CC79A7"}
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})


def box(ax, x0, y0, x1, y1, text, face, edge=None, color="white", size=7, weight="bold", sub=None):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=4", fc=face, ec=edge or face, lw=1))
    cy = (y0 + y1) / 2
    if sub:
        ax.text((x0 + x1) / 2, cy - 5.5, text, ha="center", va="center", color=color, fontsize=size, fontweight=weight)
        ax.text((x0 + x1) / 2, cy + 6.5, sub, ha="center", va="center", color=color, fontsize=size - 1.2)
    else:
        ax.text((x0 + x1) / 2, cy, text, ha="center", va="center", color=color, fontsize=size, fontweight=weight, linespacing=1.15)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=1.1, color=PURPLE, shrinkA=0, shrinkB=0))


def method(ax, cc, mlo):
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((452, 28), 202, 260, fc="#F4EEF8", ec="#D9C9E6", lw=0.8, zorder=0))
    ax.text(W / 2, 13, "CUED-Net: an exact split of dual-view uncertainty for breast mass classification", ha="center", va="center",
            color=PURPLE, fontsize=8.6, fontweight="bold")
    rows = {"CC": 68, "MLO": 164}
    for (name, cy), image in zip(rows.items(), (cc, mlo)):
        ax.imshow(image, cmap="gray", extent=(10, 72, cy + 31, cy - 31), zorder=2)
        ax.add_patch(plt.Rectangle((10, cy - 31), 62, 62, fill=False, ec=PURPLE, lw=1.2, zorder=3))
        ax.text(41, cy + 40, f"{name} view", ha="center", va="center", fontsize=6.6, color=DARK)
        box(ax, 90, cy - 18, 158, cy + 18, "Encoder", "#4A9FDE")
        box(ax, 176, cy - 18, 262, cy + 18, "Evidential head", "#3CB371", sub="Dirichlet opinion")
        arrow(ax, (73, cy), (89, cy))
        arrow(ax, (158, cy), (175, cy))
    box(ax, 284, 92, 362, 140, "Evidence-\nweighted\nfusion", PURPLE)
    arrow(ax, (262, 68), (283, 104))
    arrow(ax, (262, 164), (283, 128))
    box(ax, 380, 96, 436, 136, "$p$ and $H(p)$", "white", edge=PURPLE, color=DARK, weight="normal", size=7.4)
    arrow(ax, (362, 116), (379, 116))
    ax.text(408, 86, "prediction and\nfused entropy", ha="center", va="bottom", fontsize=6, color=DARK, linespacing=1.1)
    arrow(ax, (408, 136), (408, 221))
    y0, y1 = 223, 259
    ax.text(12, (y0 + y1) / 2, "$H(p)$  =", ha="left", va="center", fontsize=8, color=DARK)
    for x0, x1, key, text in ((58, 168, "amb", "within-view\nambiguity"), (188, 298, "evid", "evidential\nuncertainty"), (318, 436, "vd", "view discordance\nuncertainty")):
        box(ax, x0, y0, x1, y1, text, TERM[key], size=6.8)
    for x in (178, 308):
        ax.text(x, (y0 + y1) / 2, "+", ha="center", va="center", fontsize=9, color=DARK)
    ax.text(224, 274, "exact for every case, in one forward pass, with no fitted weights", ha="center", va="center", fontsize=6.6, color=DARK, style="italic")


def forest(ax, cohorts):
    """cohorts: list of (name, analysis). One point per backbone, with its interval."""
    labels, ticks = [], []
    ax.set_facecolor("none")
    for g, (name, r) in enumerate(cohorts):
        ticks.append(g)
        labels.append(name)
        for arm, off in (("densenet121", -0.17), ("convnext_tiny", 0.17)):
            v = r["arms"][arm]["view_term"]["total_vs_within"]["aurc"]
            ax.plot(v["ci95"], [g + off] * 2, color=ARM_COLOR[arm], lw=1.4, solid_capstyle="butt")
            ax.plot([v["value"]], [g + off], "o", color=ARM_COLOR[arm], ms=3.4)
    ax.axvline(0, color=DARK, lw=0.7)
    ax.set_ylim(len(cohorts) - 0.45, -0.55)
    ax.set_xlim(-0.05, 0.035)
    ax.set_xticks([-0.04, -0.02, 0, 0.02])
    ax.set_xticklabels(["\u22120.04", "\u22120.02", "0", "0.02"], fontsize=5.7)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6.4)
    ax.tick_params(axis="both", length=2, pad=2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    # The texts are centred on the panel, which extends to the left of the axes over the row labels.
    centre = 0.145
    ax.set_xlabel("change in AURC, 95% interval", fontsize=6.4)
    ax.xaxis.set_label_coords(centre, -0.17)
    ax.text(centre, 1.19, "Adding the view term\nto within-view uncertainty", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.4,
            fontweight="bold", color=PURPLE, linespacing=1.1)
    ax.text(centre, 1.06, "below 0: referral improves", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.2, color=DARK, style="italic")
    for i, arm in enumerate(ARM_NAME):
        ax.plot([], [], "o-", color=ARM_COLOR[arm], ms=3.4, lw=1.4, label=ARM_NAME[arm])
    ax.legend(loc="upper center", bbox_to_anchor=(centre, -0.25), ncol=2, frameon=False, fontsize=6.2, handlelength=1.4, columnspacing=0.9, handletextpad=0.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--cmmd_zero_shot", required=True)
    ap.add_argument("--cmmd_adapted", required=True)
    ap.add_argument("--cases", required=True, help="case file of the case figure")
    ap.add_argument("--case", default="view_discordant_correct", help="stratum whose patches illustrate the two views")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out", required=True, help="png path; a file with _300dpi before the extension is written beside it")
    args = ap.parse_args()
    load = lambda f: json.load(open(f))
    key = load(args.cases)[args.case]["key"]
    cc, mlo = (mpimg.imread(str(Path(args.image_dir) / f"{key}_{v}.png")) for v in ("cc", "mlo"))
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    method(fig.add_axes([0, 0, 1, 1]), cc, mlo)
    forest(fig.add_axes([537 / W, 0.285, 110 / W, 0.43]), [("Cross-validation", load(args.cv)), ("Held-out split", load(args.test)),
                                                             ("CMMD, zero-shot", load(args.cmmd_zero_shot)), ("CMMD, adapted", load(args.cmmd_adapted))])
    out = Path(args.out)
    fig.savefig(out, dpi=100)
    fig.savefig(out.with_name(out.stem + "_300dpi" + out.suffix), dpi=300)
    print("written", out)


if __name__ == "__main__":
    main()