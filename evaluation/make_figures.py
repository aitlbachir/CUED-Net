"""Draw the Supplement figure of reliability and net benefit as TikZ, from reliability_p2.py and p2_analysis.py outputs.

Every coordinate comes from those files, so the figure and the calibration tables rest on the same predictions.
Plain TikZ is used, with no plotting package.
"""

import argparse
import json
from pathlib import Path

from make_tables import ARM_ROWS

W, H, PITCH = 3.2, 2.9, 4.55          # plot width, plot height and distance between panels, in cm
COLORS = {"densenet121": "armdn", "convnext_tiny": "armcx"}
MIN_COUNT = 10                         # reliability bins with fewer predictions are not drawn


class Panel:
    def __init__(self, index, title, xlabel, ylabel, xr, yr, xticks, yticks):
        self.x0, self.xr, self.yr = index * PITCH, xr, yr
        x0 = self.x0
        self.lines = [rf"\draw[black, line width=0.4pt] ({x0:.3f},0) rectangle ({x0 + W:.3f},{H:.3f});",
                      rf"\node[font=\scriptsize, above] at ({x0 + W / 2:.3f},{H:.3f}) {{{title}}};",
                      rf"\node[font=\scriptsize] at ({x0 + W / 2:.3f},-0.62) {{{xlabel}}};",
                      rf"\node[font=\scriptsize, rotate=90] at ({x0 - 0.78:.3f},{H / 2:.3f}) {{{ylabel}}};"]
        for t in xticks:
            self.lines.append(rf"\draw ({self.sx(t):.3f},0) -- ++(0,-0.06) node[below, font=\tiny, inner sep=1.5pt] {{{t:g}}};")
        for t in yticks:
            self.lines.append(rf"\draw ({x0:.3f},{self.sy(t):.3f}) -- ++(-0.06,0) node[left, font=\tiny, inner sep=1.5pt] {{{t:g}}};")
        self.body = []

    def sx(self, x):
        return self.x0 + W * (x - self.xr[0]) / (self.xr[1] - self.xr[0])

    def sy(self, y):
        return H * (y - self.yr[0]) / (self.yr[1] - self.yr[0])

    def curve(self, points, style, marks=""):
        path = " -- ".join(f"({self.sx(x):.3f},{self.sy(y):.3f})" for x, y in points)
        self.body.append(rf"\draw[{style}] {path};")
        if marks:
            self.body += [rf"\fill[{marks}] ({self.sx(x):.3f},{self.sy(y):.3f}) circle (0.035);" for x, y in points]

    def tex(self):
        clip = rf"\clip ({self.x0:.3f},0) rectangle ({self.x0 + W:.3f},{H:.3f});"
        return "\n".join(self.lines + [r"\begin{scope}", clip] + self.body + [r"\end{scope}"])


def reliability(index, title, r):
    p = Panel(index, title, "Confidence", "Accuracy", (0.5, 1.0), (0.4, 1.0), (0.5, 0.6, 0.7, 0.8, 0.9, 1.0), (0.4, 0.6, 0.8, 1.0))
    p.curve([(0.5, 0.5), (1.0, 1.0)], "gray, line width=0.4pt")
    for arm, spec in ARM_ROWS.items():
        m = r["methods"][spec[0][0]]
        for kind, dash in (("raw", "dashed, "), ("calibrated", "")):
            points = [(b["confidence"], b["accuracy"]) for b in m[kind]["bins"] if b["count"] >= MIN_COUNT]
            p.curve(points, f"{dash}{COLORS[arm]}, line width=0.7pt", COLORS[arm])
    return p.tex()


def net_benefit(index, title, r, top):
    p = Panel(index, title, "Threshold", "Net benefit", (0.05, 0.5), (0.0, top), (0.1, 0.2, 0.3, 0.4, 0.5), [round(0.1 * i, 1) for i in range(int(round(top * 10)) + 1)])
    curve = lambda key: [(row["threshold"], row["net_benefit"]) for row in r["decision_curve"][key]]
    for arm, spec in ARM_ROWS.items():
        for key, _ in spec[1:]:
            p.curve(curve(key), f"{COLORS[arm]}, line width=0.3pt, opacity=0.55")
    for arm, spec in ARM_ROWS.items():
        p.curve(curve(spec[0][0]), f"{COLORS[arm]}, line width=1.1pt")
    p.curve([(row["threshold"], row["treat_all"]) for row in r["decision_curve"][ARM_ROWS["densenet121"][0][0]]], "black, dashed, line width=0.6pt")
    return p.tex()


CASES = [("clear_correct", "Clear, correct"), ("within_view_uncertain", "Within-view uncertain"),
         ("view_discordant_correct", "View discordant, correct"), ("view_discordant_error", "View discordant, error")]


def cases(r, image_dir, placement, side, columns):
    """The case figure: CC and MLO patch of the case of each stratum, with its values from the case file."""
    cells = []
    for key, name in CASES:
        c = r[key]
        cells.append("\n".join([
            rf"\begin{{minipage}}[t]{{{0.96 / columns:.2f}\linewidth}}\centering\scriptsize",
            rf"\includegraphics[width={side}]{{{image_dir}/{c['key']}_cc.png}}\,\includegraphics[width={side}]{{{image_dir}/{c['key']}_mlo.png}}\\[2pt]",
            rf"{name} ($n={c['n_in_stratum']}$), {'malignant' if c['label'] else 'benign'}\\",
            rf"$p_{{\mathrm{{CC}}}}={c['p_cc']:.2f}$, $p_{{\mathrm{{MLO}}}}={c['p_mlo']:.2f}$, $p={c['p']:.2f}$\\",
            rf"$U^{{\mathrm{{amb}}}}={c['u_amb']:.3f}$, $U^{{\mathrm{{evid}}}}={c['u_ev']:.3f}$, $U^{{\mathrm{{vd}}}}={c['u_vd']:.3f}$",
            r"\end{minipage}"]))
    caption = ("Four development cases, one per stratum of the DenseNet-121 outer-fold predictions. Each panel shows the CC (left) and MLO (right) "
               "patches, the number $n$ of cases in the stratum, the label, the malignancy probabilities of the views and of the fusion, "
               "and the three terms in bits, as means over seeds.")
    rows = ["\n\\hfill\n".join(cells[i:i + columns]) for i in range(0, len(cells), columns)]
    return "\n".join([rf"\begin{{figure*}}[{placement}]", r"\centering", "\n\\par\\vspace{4pt}\n".join(rows), rf"\caption{{{caption}}}", r"\label{fig:cases}", r"\end{figure*}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="", help="p2_analysis.py output, cross-validation")
    ap.add_argument("--analysis_test", default="", help="p2_analysis.py output, held-out split")
    ap.add_argument("--reliability", default="", help="reliability_p2.py output, cross-validation")
    ap.add_argument("--reliability_test", default="", help="reliability_p2.py output, held-out split")
    ap.add_argument("--cases", default="", help="case file of the case figure; with it, only that figure is written")
    ap.add_argument("--image_dir", default="cases", help="folder of the case patches, as LaTeX will read it")
    ap.add_argument("--side", default="1.95cm", help="width of one case patch")
    ap.add_argument("--columns", type=int, default=4, help="cases per row of the case figure")
    ap.add_argument("--placement", default="!t")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cases:
        Path(args.out).write_text("% Written by evaluation/make_figures.py. Do not edit by hand.\n" + cases(json.load(open(args.cases)), args.image_dir, args.placement, args.side, args.columns) + "\n")
        return
    load = lambda f: json.load(open(f))
    panels = [reliability(0, "(a) Cross-validation", load(args.reliability)),
              reliability(1, "(b) Held-out split", load(args.reliability_test)),
              net_benefit(2, "(c) Cross-validation", load(args.analysis), 0.5),
              net_benefit(3, "(d) Held-out split", load(args.analysis_test), 0.4)]
    caption = ("Reliability and net benefit of CUED-Net. Blue: DenseNet-121. Orange: ConvNeXt-T. (a), (b) Accuracy against the confidence of the predicted class "
               "in 15 equal-width bins that pool the five seeds, before (dashed) and after (solid) temperature scaling. "
               f"Bins with fewer than {MIN_COUNT} predictions are not drawn. (c), (d) Net benefit of acting on the cases whose scaled probability of malignancy, "
               "averaged over seeds, reaches the threshold: CUED-Net (thick), its five comparators (thin) and acting on every case (black, dashed). "
               "The curves are retrospective and carry no interval.")
    text = "\n".join([rf"\begin{{figure*}}[{args.placement}]", r"\centering", r"\definecolor{armdn}{RGB}{0,114,178}", r"\definecolor{armcx}{RGB}{213,94,0}",
                      r"\begin{tikzpicture}", *panels, r"\end{tikzpicture}", rf"\caption{{{caption}}}", r"\label{fig:curves}", r"\end{figure*}"])
    Path(args.out).write_text("% Written by evaluation/make_figures.py. Do not edit by hand.\n" + text + "\n")


if __name__ == "__main__":
    main()