"""Reliability bins of single models, from the arrays that p2_analysis.py uses for its calibration errors.

The confidence of the predicted class is put into 15 equal-width bins on [0.5, 1], as in p2_analysis.ece.
A bin pools the seeds. The ECE written beside the bins is the mean over seeds, as in the analysis output,
so that a figure drawn from this file and the calibration tables rest on the same predictions.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.p2_analysis import Prepared, add_scores, all_metrics, read_split

BINS = 15


def bins(y, p):
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, BINS + 1)
    which = np.clip(np.digitize(conf, edges[1:-1]), 0, BINS - 1)
    rows = [dict(lower=float(edges[b]), upper=float(edges[b + 1]), count=int((which == b).sum()),
                 confidence=float(conf[which == b].mean()), accuracy=float(correct[which == b].mean()))
            for b in range(BINS) if (which == b).any()]
    return dict(bins=rows, mean_confidence=float(conf.mean()), accuracy=float(correct.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--split", default="outer_val", choices=["outer_val", "test", "cmmd", "cmmd_eval", "cmmd_mass"])
    ap.add_argument("--methods", nargs="+", default=["cued_net", "cued_convnext_k5"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = dict(split=args.split, bins=BINS, methods={})
    for name in args.methods:
        iv, ov = add_scores(read_split(args.runs, name, "inner_val")), add_scores(read_split(args.runs, name, args.split))
        if args.split != "outer_val":
            iv = iv[iv.fold == -1]
        m = Prepared(iv, ov, unc="entropy")
        assert not np.isnan(m.p).any(), name
        point = all_metrics(m, np.arange(len(m.y)))
        y = np.tile(m.y, len(m.seeds))
        result["methods"][name] = dict(n_cases=len(m.y), n_seeds=len(m.seeds), temperature=float(np.mean(m.temperatures)),
                                       ece=point["ece"], ece_cal=point["ece_cal"],
                                       raw=bins(y, m.p.ravel()), calibrated=bins(y, m.p_cal.ravel()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    for name, r in result["methods"].items():
        print(name, f"T {r['temperature']:.2f}  ECE {r['ece']:.3f} -> {r['ece_cal']:.3f}  "
                    f"confidence {r['raw']['mean_confidence']:.3f} -> {r['calibrated']['mean_confidence']:.3f}  accuracy {r['raw']['accuracy']:.3f}")


if __name__ == "__main__":
    main()