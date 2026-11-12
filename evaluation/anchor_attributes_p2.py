"""Exploratory: does view discordance uncertainty follow a difference between the CC and MLO records in one attribute?

The planned test of anchor_p2.py flags a pair when the records differ in any of four attributes.
Here each attribute is tested alone, with the same statistic and the same patient-level permutations,
and Holm correction over the four attributes of a model. Not in the analysis plan.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.anchor_p2 import case_table, group_contrast
from evaluation.p2_analysis import holm

ATTRIBUTES = ["subtlety_differs", "assessment_differs", "shape_differs", "margins_differs"]


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
    result = {}
    for method in args.methods.split(","):
        t = case_table(args.runs, method, meta, args.split)
        tests = {a: group_contrast(t, a, "u_vd", args.perms) for a in ATTRIBUTES}
        adjusted = holm({k: v["p"] for k, v in tests.items()})
        for k in tests:
            tests[k]["p_holm"] = float(adjusted[k])
        result[method] = dict(n=len(t), tests=tests)
    Path(args.out).write_text(json.dumps(result, indent=1))
    for method, r in result.items():
        for k, v in r["tests"].items():
            print(f"{method:18s} {k:20s} flagged {v['n_flagged']:4d} of {r['n']}  AUC {v['auc']:.3f}  p {v['p']:.4f}  Holm p {v['p_holm']:.4f}")


if __name__ == "__main__":
    main()
