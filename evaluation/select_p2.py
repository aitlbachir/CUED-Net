"""Evidence-scale rule of ANALYSIS_PLAN.md section 12. Reads inner validation of the outer folds only."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from p2_analysis import aurc, nll
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.decomposition import add_decomposition


def inner_scores(runs, method):
    rows = []
    for f in sorted(Path(runs, method).glob("fold_[0-4]/seed_*/inner_val.csv")):
        d = add_decomposition(pd.read_csv(f))
        y, p = d.label.values, d.p.values
        err = ((p >= 0.5).astype(int) != y).astype(float)
        rows.append(dict(fold=f.parts[-3], seed=f.parts[-2], aurc=aurc(d.u_total.values, err), nll=nll(y, p)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--base", default="cued_convnext")
    ap.add_argument("--candidate", default="cued_convnext_k5")
    args = ap.parse_args()
    a, b = inner_scores(args.runs, args.base), inner_scores(args.runs, args.candidate)
    assert len(a) == len(b) == 25, (len(a), len(b))
    m = a.merge(b, on=["fold", "seed"], suffixes=("_base", "_candidate"))
    result = dict(n_inner_sets=len(m), **{k: float(m[k].mean()) for k in ("aurc_base", "aurc_candidate", "nll_base", "nll_candidate")})
    result["candidate_lower_aurc_sets"] = int((m.aurc_candidate < m.aurc_base).sum())
    result["candidate_lower_nll_sets"] = int((m.nll_candidate < m.nll_base).sum())
    result["replace"] = bool(result["aurc_candidate"] < result["aurc_base"] and result["nll_candidate"] < result["nll_base"])
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
