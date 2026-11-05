"""Do the terms follow what radiologists recorded? Descriptive, outer folds. ANALYSIS_PLAN.md section 13."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from p2_analysis import add_scores, cluster_permutation, read_split

PAIRS = [("subtlety_gap", "u_vd"), ("assessment_gap", "u_vd"), ("subtlety_mean", "u_total"), ("subtlety_mean", "u_within")]


def case_table(runs, method, meta):
    d = add_scores(read_split(runs, method, "outer_val"))
    d["error"] = ((d.p >= 0.5).astype(int) != d.label).astype(float)
    t = d.groupby(["pair_key", "patient_id"], as_index=False)[["u_amb", "u_ev", "u_vd", "u_within", "u_total", "error"]].mean()
    t = t.merge(meta, left_on="pair_key", right_on="key", how="left", validate="1:1")
    t["subtlety_gap"] = (t.subtlety_CC - t.subtlety_MLO).abs()
    t["assessment_gap"] = (t.assessment_CC - t.assessment_MLO).abs()
    t["subtlety_mean"] = 0.5 * (t.subtlety_CC + t.subtlety_MLO)
    return t


def permutation_rho(t, x, y, perms=999, seed=7):
    s = t.dropna(subset=[x, y])
    a, b = s[x].values, s[y].values
    group = pd.factorize(s.patient_id)[0]
    rho = float(spearmanr(a, b)[0])
    rng = np.random.default_rng(seed)
    null = [abs(spearmanr(a, b[cluster_permutation(group, rng)])[0]) for _ in range(perms)]
    return dict(n=len(s), rho=rho, p=float((1 + np.sum(np.array(null) >= abs(rho))) / (perms + 1)))


def by_gap(t, column):
    g = t.dropna(subset=[column]).assign(level=lambda d: d[column].clip(upper=2).astype(int))
    return {str(k): dict(n=int(len(v)), u_vd=float(v.u_vd.mean()), error=float(v.error.mean())) for k, v in g.groupby("level")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--methods", default="cued_net,cued_convnext_k5")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    meta = pd.read_csv(args.meta)
    result = {}
    for method in args.methods.split(","):
        t = case_table(args.runs, method, meta)
        result[method] = dict(unmatched=int(t.subtlety_CC.isna().sum()),
                              correlation={f"{x}_vs_{y}": permutation_rho(t, x, y) for x, y in PAIRS},
                              by_subtlety_gap=by_gap(t, "subtlety_gap"), by_assessment_gap=by_gap(t, "assessment_gap"))
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
