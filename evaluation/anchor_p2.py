"""Do the terms follow what radiologists recorded? Descriptive. ANALYSIS_PLAN.md section 13."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.p2_analysis import add_scores, cluster_permutation, holm, read_split

VALUES = ["u_amb", "u_ev", "u_vd", "u_within", "u_total", "share_evid", "error", "s_cc", "s_mlo"]


def annotate(t):
    """Ratings of 0 are not ratings: subtlety 0 is invalid and BI-RADS 0 means an incomplete assessment."""
    for c in ("subtlety_CC", "subtlety_MLO", "assessment_CC", "assessment_MLO"):
        t[c] = t[c].where(t[c] > 0)
    both = lambda a, b: a.notna() & b.notna() & (a != b)
    t["subtlety_differs"] = both(t.subtlety_CC, t.subtlety_MLO)
    t["assessment_differs"] = both(t.assessment_CC, t.assessment_MLO)
    t["shape_differs"] = t.shape_CC.fillna("none") != t.shape_MLO.fillna("none")
    t["margins_differs"] = t.margins_CC.fillna("none") != t.margins_MLO.fillna("none")
    t["described_differently"] = t.subtlety_differs | t.assessment_differs | t.shape_differs | t.margins_differs
    t["subtlety_gap"] = (t.subtlety_CC - t.subtlety_MLO).abs()
    t["subtlety_mean"] = t[["subtlety_CC", "subtlety_MLO"]].mean(axis=1)
    return t


def case_table(runs, method, meta, split):
    d = add_scores(read_split(runs, method, split))
    d["share_evid"] = d.u_ev / (d.u_amb + d.u_ev)
    d["error"] = ((d.p >= 0.5).astype(int) != d.label).astype(float)
    t = d.groupby(["pair_key", "patient_id"], as_index=False)[VALUES].mean()
    return annotate(t.merge(meta, left_on="pair_key", right_on="key", how="left", validate="1:1"))


def rank_auc(values, flag):
    r = rankdata(values)
    n1, n0 = flag.sum(), (~flag).sum()
    return (r[flag].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def group_contrast(t, flag, y, perms=999, seed=7):
    """Probability that a flagged case has the higher value, with whole patients permuted."""
    f, v = t[flag].values, t[y].values
    group = pd.factorize(t.patient_id)[0]
    auc = float(rank_auc(v, f))
    rng = np.random.default_rng(seed)
    null = [abs(rank_auc(v[cluster_permutation(group, rng)], f) - 0.5) for _ in range(perms)]
    return dict(n_flagged=int(f.sum()), n_other=int((~f).sum()), mean_flagged=float(v[f].mean()), mean_other=float(v[~f].mean()),
                error_flagged=float(t.error.values[f].mean()), error_other=float(t.error.values[~f].mean()), auc=auc,
                p=float((1 + np.sum(np.array(null) >= abs(auc - 0.5))) / (perms + 1)))


def rank_correlation(t, x, y, perms=999, seed=7):
    s = t.dropna(subset=[x, y])
    a, b = s[x].values, s[y].values
    group = pd.factorize(s.patient_id)[0]
    rho = float(spearmanr(a, b)[0])
    rng = np.random.default_rng(seed)
    null = [abs(spearmanr(a, b[cluster_permutation(group, rng)])[0]) for _ in range(perms)]
    return dict(n=len(s), rho=rho, p=float((1 + np.sum(np.array(null) >= abs(rho))) / (perms + 1)))


def levels(t):
    g = t.dropna(subset=["subtlety_gap"]).assign(level=lambda d: d.subtlety_gap.clip(upper=2).astype(int))
    return {str(k): dict(n=int(len(v)), u_vd=float(v.u_vd.mean()), error=float(v.error.mean())) for k, v in g.groupby("level")}


def clearer_view_evidence(t, perms=9999, seed=7):
    """Exploratory. Among pairs rated differently, does the more obvious view carry more evidence?"""
    s = t[t.subtlety_differs]
    cc_clearer = (s.subtlety_CC > s.subtlety_MLO).values
    d = np.where(cc_clearer, s.s_cc - s.s_mlo, s.s_mlo - s.s_cc)
    group = pd.factorize(s.patient_id)[0]
    flips = np.random.default_rng(seed).choice([-1, 1], size=(perms, group.max() + 1))[:, group]
    null = np.abs((flips * d).mean(1))
    return dict(n=len(s), difference=float(d.mean()), share_positive=float((d > 0).mean()),
                p=float((1 + (null >= abs(d.mean())).sum()) / (perms + 1)))


def analyse(t, perms):
    tests = dict(described_differently=group_contrast(t, "described_differently", "u_vd", perms),
                 subtlety_gap=rank_correlation(t, "subtlety_gap", "u_vd", perms),
                 entropy_with_subtlety=rank_correlation(t, "subtlety_mean", "u_total", perms),
                 evidential_share_with_subtlety=rank_correlation(t, "subtlety_mean", "share_evid", perms))
    adjusted = holm({k: v["p"] for k, v in tests.items()})
    for k in tests:
        tests[k]["p_holm"] = float(adjusted[k])
    return dict(n=len(t), unmatched=int(t.subtlety_mean.isna().sum()), tests=tests, subtlety_gap_levels=levels(t),
                assessment_differs=group_contrast(t, "assessment_differs", "u_vd", perms),
                exploratory_clearer_view_evidence=clearer_view_evidence(t, perms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--methods", default="cued_net,cued_convnext_k5")
    ap.add_argument("--split", default="outer_val")
    ap.add_argument("--perms", type=int, default=9999)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    meta = pd.read_csv(args.meta)
    result = {m: analyse(case_table(args.runs, m, meta, args.split), args.perms) for m in args.methods.split(",")}
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
