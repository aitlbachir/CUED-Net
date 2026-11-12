"""Outer-fold AUC, F1 and AURC of every single model per seed, with the arrays of p2_analysis.py.

Each value pools the five outer folds of one seed. The mean over seeds of a row is the point
estimate of the analysis output, so the file shows training variability and adds no sample size.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.p2_analysis import ALL_MODELS, CUED_MODELS, VACUITY_MODELS, Prepared, add_scores, auc, aurc, f1, read_split


def prepared(runs, split="outer_val"):
    """The single models of p2_analysis.main, each with the score of its confirmatory comparison."""
    out = {}
    for name in ALL_MODELS:
        ov = read_split(runs, name, split)
        if ov is None:
            continue
        iv, ov = add_scores(read_split(runs, name, "inner_val")), add_scores(ov)
        if name in CUED_MODELS:
            out[name] = Prepared(iv, ov, unc="u_total")
        elif name in VACUITY_MODELS:
            out[name] = Prepared(iv, ov, unc="vacuity")
        elif name.startswith("softmax"):
            out[name] = Prepared(iv, ov, unc="entropy")
            out["mc_dropout" + name.replace("softmax", "").replace("_dual", "")] = Prepared(iv, ov, prob="mc_p", unc="mc_entropy")
        else:
            out[name] = Prepared(iv, ov, unc="entropy")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True, help="json summary; a csv with one row per method and seed is written beside it")
    args = ap.parse_args()
    result, rows = {}, []
    for name, m in prepared(args.runs).items():
        r = dict(seeds=[int(s) for s in m.seeds], auc=[], f1=[], aurc=[])
        for s in range(len(m.seeds)):
            p, d = m.p[s], m.p[s] >= 0.5
            err = (d.astype(int) != m.y).astype(float)
            r["auc"].append(float(auc(m.y, p)))
            r["f1"].append(float(f1(m.y, d)))
            r["aurc"].append(float(aurc(m.u[s], err)))
            rows.append([name, r["seeds"][s], f"{r['auc'][-1]:.4f}", f"{r['f1'][-1]:.4f}", f"{r['aurc'][-1]:.4f}"])
        if name in CUED_MODELS:
            # Fusion weight of the CC view over every case and seed: how far evidence weighting departs from one half.
            w = read_split(args.runs, name, "outer_val").w_cc.values
            r["weight_cc"] = dict(p01=float(np.percentile(w, 1)), median=float(np.median(w)), p99=float(np.percentile(w, 99)))
        result[name] = r
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    with open(Path(args.out).with_suffix(".csv"), "w", newline="") as f:
        csv.writer(f).writerows([["method", "seed", "auc", "f1", "aurc"]] + rows)
    for name, r in result.items():
        if "weight_cc" in r:
            print(f"{name:22s} weight of the CC view, 1st and 99th percentiles: {r['weight_cc']['p01']:.3f} {r['weight_cc']['p99']:.3f}")
        print(f"{name:22s} AUC {min(r['auc']):.3f} to {max(r['auc']):.3f}, mean {np.mean(r['auc']):.4f}   AURC {min(r['aurc']):.3f} to {max(r['aurc']):.3f}")


if __name__ == "__main__":
    main()