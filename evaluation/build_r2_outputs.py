"""Rebuild every output of the second revision from the saved prediction files.

Each step is one command. Paths are relative to --root, the working folder that holds
work/ (prediction CSVs extracted from the run archives), analysis/ and manuscript/.
With --out_dir, outputs go there instead, and later steps read earlier outputs from there,
so the rebuilt files can be compared with the ones the manuscript uses.

Stages:
  summaries   bootstrap and permutation analyses (slow: B = 10,000)
  derived     reliability bins and per-seed values (fast)
  manuscript  LaTeX macros, tables, figures and the graphical abstract (fast)

Model training, the held-out and CMMD inference, the controlled changes, Grad-CAM and the
cost measurement ran on Kaggle GPUs and are not repeated here (see README).
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
E = "evaluation/"


def steps():
    s = []
    add = lambda stage, out, script, *args: s.append((stage, out, script, list(args)))
    # Summaries of the outer folds, the held-out split and CMMD (plan sections 6 to 9).
    add("summaries", "manuscript/an_cv_final.json", E + "p2_analysis.py", "--runs", "work/cv/runs", "--split", "outer_val",
        "--multi_mass", "analysis/multi_mass.json", "--B", "10000", "--B_added", "2000")
    add("summaries", "manuscript/an_test.json", E + "p2_analysis.py", "--runs", "work/heldout", "--split", "test", "--B", "10000", "--B_added", "2000")
    for runs, split, name in [("zero_shot", "cmmd_eval", "zero_shot_eval"), ("zero_shot", "cmmd_mass", "zero_shot_mass"),
                              ("adapted", "cmmd_eval", "adapted_eval"), ("adapted", "cmmd_mass", "adapted_mass"),
                              ("zero_shot", "cmmd", "zero_shot_all")]:
        add("summaries", f"analysis/an_cmmd_{name}.json", E + "p2_analysis.py", "--runs", f"work/s6/cmmd/{runs}", "--split", split,
            "--B", "10000", "--B_added", "2000")
    add("summaries", "manuscript/e10_cv.json", E + "e10_analysis.py", "@REPO/evaluation", "work/e10/runs", "@OUT", "10000")
    # What each term measures (plan sections 12 and 13).
    add("summaries", "analysis/term_response_dev_summary.json", E + "term_response_p2.py", "--development", "--out", "work/pilot/response_dev", "--summary", "@OUT")
    add("summaries", "manuscript/term_response_test_summary.json", E + "term_response_p2.py", "--out", "work/s6/test", "--summary", "@OUT")
    add("summaries", "analysis/anchor_dev.json", E + "anchor_p2.py", "--runs", "work/cv/runs", "--meta", "analysis/lesion_meta.csv", "--split", "outer_val")
    add("summaries", "manuscript/anchor_test.json", E + "anchor_p2.py", "--runs", "work/heldout", "--meta", "analysis/lesion_meta.csv", "--split", "test")
    # Exploratory analyses, specified in writing before they were run (labelled as such in the Supplement).
    add("summaries", "analysis/anchor_attributes_dev.json", E + "anchor_attributes_p2.py", "--runs", "work/cv/runs", "--meta", "analysis/lesion_meta.csv", "--split", "outer_val")
    add("summaries", "analysis/anchor_attributes_test.json", E + "anchor_attributes_p2.py", "--runs", "work/heldout", "--meta", "analysis/lesion_meta.csv", "--split", "test")
    add("summaries", "analysis/view_term_cued_convnext_cv.json", E + "view_term_p2.py", "--runs", "work/cv/runs", "--method", "cued_convnext", "--split", "outer_val", "--B", "10000")
    add("summaries", "analysis/view_term_cued_convnext_test.json", E + "view_term_p2.py", "--runs", "work/heldout", "--method", "cued_convnext", "--split", "test", "--B", "10000")
    # Reliability bins and per-seed values.
    add("derived", "analysis/reliability_cv.json", E + "reliability_p2.py", "--runs", "work/cv/runs", "--split", "outer_val")
    add("derived", "analysis/reliability_test.json", E + "reliability_p2.py", "--runs", "work/heldout", "--split", "test")
    add("derived", "analysis/per_seed_cv.json", E + "seed_table_p2.py", "--runs", "work/cv/runs")
    # Macros: every number in the text.
    add("manuscript", "manuscript/numbers_cv.tex", E + "make_numbers.py", "--analysis", "manuscript/an_cv_final.json",
        "--anchor", "analysis/anchor_dev.json", "--response", "analysis/term_response_dev_summary.json", "--cost", "work/v4cost/cost.json",
        "--saliency", "analysis/gradcam_summary.json", "--reliability", "analysis/reliability_cv.json", "--seeds", "analysis/per_seed_cv.json",
        "--attributes", "analysis/anchor_attributes_dev.json", "--scale_view", "analysis/view_term_cued_convnext_cv.json")
    add("manuscript", "manuscript/numbers_test.tex", E + "make_numbers.py", "--analysis", "manuscript/an_test.json", "--tag", "Test",
        "--anchor", "manuscript/anchor_test.json", "--response", "manuscript/term_response_test_summary.json",
        "--reliability", "analysis/reliability_test.json", "--attributes", "analysis/anchor_attributes_test.json",
        "--reference_effect", "manuscript/an_cv_final.json", "--scale_view", "analysis/view_term_cued_convnext_test.json")
    for name, tag in [("zero_shot_eval", "CmmdZs"), ("adapted_eval", "CmmdAd"), ("zero_shot_mass", "CmmdZsMass"),
                      ("adapted_mass", "CmmdAdMass"), ("zero_shot_all", "CmmdZsAll")]:
        short = {"CmmdZs": "zs", "CmmdAd": "ad", "CmmdZsMass": "zs_mass", "CmmdAdMass": "ad_mass", "CmmdZsAll": "zs_all"}[tag]
        add("manuscript", f"manuscript/numbers_cmmd_{short}.tex", E + "make_numbers.py", "--analysis", f"analysis/an_cmmd_{name}.json", "--tag", tag)
    # Tables of the main paper.
    add("manuscript", "manuscript/tables_cv.tex", E + "make_tables.py", "--analysis", "manuscript/an_cv_final.json", "--no_view_term")
    add("manuscript", "manuscript/tables_test.tex", E + "make_tables.py", "--analysis", "manuscript/an_test.json", "--what", "held-out split",
        "--layout", "compact", "--suffix", "_test")
    add("manuscript", "manuscript/tables_cmmd.tex", E + "make_tables.py", "--analysis", "analysis/an_cmmd_zero_shot_eval.json",
        "--adapted", "analysis/an_cmmd_adapted_eval.json", "--layout", "cmmd", "--what", "evaluation patients")
    # Tables of the Supplement.
    add("manuscript", "manuscript/tables_e10.tex", E + "make_tables.py", "--analysis", "manuscript/e10_cv.json", "--layout", "e10", "--size", "scriptsize")
    for only in ("classification", "referral", "view_term"):
        add("manuscript", f"manuscript/tables_test_{only}.tex", E + "make_tables.py", "--analysis", "manuscript/an_test.json",
            "--what", "held-out split", "--suffix", "_test", "--only", only, "--size", "scriptsize")
    for layout in ("operating", "calibration"):
        add("manuscript", f"manuscript/tables_{layout}.tex", E + "make_tables.py", "--analysis", "manuscript/an_cv_final.json",
            "--what", "Cross-validation", "--second", "manuscript/an_test.json", "--layout", layout, "--reference_only", "--size", "scriptsize")
    add("manuscript", "manuscript/tables_dependence.tex", E + "make_tables.py", "--analysis", "manuscript/an_cv_final.json",
        "--what", "Cross-validation", "--second", "manuscript/an_test.json", "--layout", "dependence", "--size", "scriptsize", "--colsep", "3pt")
    add("manuscript", "manuscript/tables_cmmd_zs_transfer.tex", E + "make_tables.py", "--analysis", "analysis/an_cmmd_zero_shot_eval.json",
        "--layout", "transfer", "--what", "CMMD evaluation patients, zero-shot", "--suffix", "_cmmd_zs", "--size", "scriptsize")
    add("manuscript", "manuscript/tables_cmmd_view.tex", E + "make_tables.py", "--analysis", "analysis/an_cmmd_zero_shot_eval.json",
        "--adapted", "analysis/an_cmmd_adapted_eval.json", "--layout", "cmmd_view", "--what", "evaluation patients", "--size", "scriptsize")
    add("manuscript", "manuscript/tables_cmmd_mass.tex", E + "make_tables.py", "--analysis", "analysis/an_cmmd_zero_shot_mass.json",
        "--adapted", "analysis/an_cmmd_adapted_mass.json", "--layout", "cmmd", "--what", "mass-containing subset of the evaluation patients",
        "--suffix", "_mass", "--size", "scriptsize", "--colsep", "3pt")
    # Figures of the Supplement and the graphical abstract.
    add("manuscript", "manuscript/fig_supp_curves.tex", E + "make_figures.py", "--analysis", "manuscript/an_cv_final.json",
        "--analysis_test", "manuscript/an_test.json", "--reliability", "analysis/reliability_cv.json", "--reliability_test", "analysis/reliability_test.json")
    add("manuscript", "manuscript/fig_supp_cases.tex", E + "make_figures.py", "--cases", "analysis/case_keys.json", "--image_dir", "cases",
        "--columns", "2", "--side", "1.7cm")
    add("manuscript", "figures/CUED-Net_Graphical_Abstract_R2.png", "visualization/make_graphical_abstract.py", "--cv", "manuscript/an_cv_final.json",
        "--test", "manuscript/an_test.json", "--cmmd_zero_shot", "analysis/an_cmmd_zero_shot_eval.json", "--cmmd_adapted", "analysis/an_cmmd_adapted_eval.json",
        "--cases", "analysis/case_keys.json", "--image_dir", "manuscript/cases")
    return s


def command(step, root, out_dir):
    stage, out, script, args = step
    produced = {o for _, o, _, _ in steps()}

    def where(a):
        if a == "@REPO/evaluation":
            return str(REPO / "evaluation")
        if a == "@OUT":
            return str(out_dir / out)
        if "/" in a and not a.startswith("--"):
            # inputs made by an earlier step are read from out_dir when it holds them
            return str(out_dir / a) if a in produced and (out_dir / a).exists() else str(root / a)
        return a

    cmd = [sys.executable, str(REPO / script)] + [where(a) for a in args]
    if "@OUT" not in args:
        cmd += ["--out", str(out_dir / out)]
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="working folder with work/, analysis/ and manuscript/")
    ap.add_argument("--out_dir", default="", help="write outputs here instead of under --root")
    ap.add_argument("--stage", nargs="+", default=["summaries", "derived", "manuscript"])
    ap.add_argument("--only", nargs="*", default=[], help="output paths to rebuild, for example manuscript/numbers_cv.tex")
    ap.add_argument("--dry_run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root
    for step in steps():
        if step[0] not in args.stage or (args.only and step[1] not in args.only):
            continue
        cmd = command(step, root, out_dir)
        print(" ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
        if not args.dry_run:
            (out_dir / step[1]).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, check=True, cwd=root / "manuscript" if step[1].endswith("cases.tex") else root)


if __name__ == "__main__":
    main()