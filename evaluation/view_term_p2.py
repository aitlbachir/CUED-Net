"""View-term tests of one CUED-Net variant on the outer folds, with the functions of p2_analysis.py.

Used for the ConvNeXt-T model at evidence scale 1, which is not the reference model of its arm
and therefore has no view-term block in the main analysis output. On the held-out split the
analysis is exploratory: the confirmatory test is that of the reference model.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.p2_analysis import add_scores, read_split, view_term


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--split", default="outer_val", choices=["outer_val", "test"])
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    iv, ov = add_scores(read_split(args.runs, args.method, "inner_val")), add_scores(read_split(args.runs, args.method, args.split))
    if args.split != "outer_val":
        iv = iv[iv.fold == -1]
    result = dict(method=args.method, split=args.split, n_cases=int(ov.pair_key.nunique()), B=args.B, view_term=view_term(iv, ov, args.B))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    for k, v in result["view_term"].items():
        if k != "scores":
            print(k, round(v["aurc"]["value"], 4), [round(x, 4) for x in v["aurc"]["ci95"]], v["aurc"]["p"])


if __name__ == "__main__":
    main()