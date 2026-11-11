"""Write the main LaTeX tables from one p2_analysis.py output, so every number has one source."""

import argparse
import json
from pathlib import Path

ARM_ROWS = {
    "densenet121": [("cued_net", "CUED-Net"), ("tmc", "TMC"), ("deep_ensemble", "Deep-Ensemble"), ("mc_dropout", "MC-Dropout"),
                    ("cross_view", "Cross-view transformer"), ("sv_edl", "Single-view EDL")],
    "convnext_tiny": [("cued_convnext_k5", "CUED-Net"), ("tmc_convnext", "TMC"), ("deep_ensemble_convnext", "Deep-Ensemble"),
                      ("mc_dropout_convnext", "MC-Dropout"), ("cross_view_convnext", "Cross-view transformer"), ("sv_edl_convnext", "Single-view EDL")],
}
ARM_NAMES = {"densenet121": "DenseNet-121", "convnext_tiny": "ConvNeXt-T"}
ENTROPY = {"tmc": "tmc_entropy", "sv_edl": "sv_edl_entropy", "tmc_convnext": "tmc_convnext_entropy", "sv_edl_convnext": "sv_edl_convnext_entropy"}


def ci(v, d=3):
    return f"{v['value']:.{d}f} [{v['ci95'][0]:.{d}f}, {v['ci95'][1]:.{d}f}]"


def sg(x):
    return f"{x:+.3f}".replace("-", "$-$")


def p_text(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def table(columns, rows, caption, label, wide=True):
    env = "table*" if wide else "table"
    head = " & ".join(columns) + r" \\"
    return "\n".join([rf"\begin{{{env}}}[!t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\footnotesize",
                      rf"\begin{{tabular}}{{l{'c' * (len(columns) - 1)}}}", r"\toprule", head, r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"])


def classification(r, what, suffix=""):
    rows = []
    for arm, spec in ARM_ROWS.items():
        rows.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{ARM_NAMES[arm]}}}}} \\")
        for key, name in spec:
            if key in r["summary"]:
                s = r["summary"][key]
                rows.append(f"{name} & {ci(s['auc'])} & {ci(s['f1'])} & {ci(s['sensitivity'])} & {ci(s['specificity'])} & {ci(s['ece_cal'])} " + r"\\")
    return table(["Method", "AUC", "F1", "Sensitivity", "Specificity", "Calibrated ECE"], rows,
                 f"Classification and calibration, {what}, threshold 0.5. Brackets give 95\\% patient-clustered bootstrap intervals.", "tab:classification" + suffix)


def referral(r, what, suffix=""):
    rows = []
    for arm, spec in ARM_ROWS.items():
        rows.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{ARM_NAMES[arm]}}}}} \\")
        for key, name in spec:
            if key not in r["summary"]:
                continue
            for k, tag in ((key, "vacuity" if key in ENTROPY else "entropy"), (ENTROPY.get(key), "entropy")):
                if k is None or k not in r["summary"]:
                    continue
                s = r["summary"][k]
                label = "$H(\\mathbf{p})$" if name == "CUED-Net" else tag
                rows.append(f"{name}, {label} & {ci(s['aurc'])} & {ci(s['eaurc'])} & {ci(s['error_auroc'])} & {ci(s['errors_captured_80'])} " + r"\\")
    return table(["Method and score", "AURC", "E-AURC", "Error AUROC", "Errors referred at 80\\% coverage"], rows,
                 f"Referral, {what}. Lower AURC and E-AURC are better.", "tab:referral" + suffix)


def view_term(r, what, suffix=""):
    rows = []
    names = [("total_vs_within", "Added to both within-view terms"), ("amb_plus_view_vs_amb", "Added to $U^{\\mathrm{amb}}$ alone"),
             ("evidential_plus_view_vs_evidential", "Added to $U^{\\mathrm{evid}}$ alone")]
    for arm, a in r["arms"].items():
        rows.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{ARM_NAMES[arm]}}}}} \\")
        for key, name in names:
            v = a["view_term"][key]["aurc"]
            rows.append(f"{name} & {sg(v['value'])} [{sg(v['ci95'][0])}, {sg(v['ci95'][1])}] & {p_text(v['p'])} " + r"\\")
        f = a["prespecified_fitted_score"]["added_value"]["delta_aurc"]
        rows.append(f"Fitted score, original plan & {sg(f['value'])} [{sg(f['ci95'][0])}, {sg(f['ci95'][1])}] & {p_text(f['p'])} " + r"\\")
    return table(["Change in AURC", "Difference [95\\% CI]", "$p$"], rows,
                 f"Referral value of view discordance, {what}. The first three rows add $U^{{\\mathrm{{vd}}}}$ to the terms named. "
                 "The last row adds $1-e^{{-D_{{\\mathrm{{sym}}}}}}$ to vacuity in the fitted score of the original plan. A negative difference favors the addition.", "tab:viewterm" + suffix, wide=False)


def compact(r, what, suffix=""):
    """One column wide: AUC and AURC of every method, each under the score used in the confirmatory comparison."""
    rows = []
    for arm, spec in ARM_ROWS.items():
        if not any(key in r["summary"] for key, _ in spec):
            continue
        rows.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{ARM_NAMES[arm]}}}}} \\")
        for key, name in spec:
            if key in r["summary"]:
                s = r["summary"][key]
                rows.append(f"{name} & {ci(s['auc'])} & {ci(s['aurc'])} " + r"\\")
    return table(["Method", "AUC", "AURC"], rows,
                 f"Classification and referral, {what}. CUED-Net is ranked by $H(\\mathbf{{p}})$, TMC and single-view EDL by vacuity, "
                 "the other methods by predictive entropy. Brackets give 95\\% patient-clustered bootstrap intervals.", "tab:compact" + suffix, wide=False)


def cmmd(zero_shot, adapted, what, suffix=""):
    """One column wide: CUED-Net and TMC on CMMD, zero-shot and adapted, in separate rows as the plan requires."""
    rows = []
    for arm, spec in ARM_ROWS.items():
        rows.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{ARM_NAMES[arm]}}}}} \\")
        for key, name in spec[:2]:
            for r, setting in ((zero_shot, "zero-shot"), (adapted, "adapted")):
                s = r["summary"][key]
                rows.append(f"{name}, {setting} & {ci(s['auc'])} & {ci(s['aurc'])} " + r"\\")
    return table(["Method and setting", "AUC", "AURC"], rows,
                 f"External evaluation on CMMD, {what}. CUED-Net is ranked by $H(\\mathbf{{p}})$ and TMC by vacuity. "
                 "Brackets give 95\\% patient-clustered bootstrap intervals.", "tab:cmmd" + suffix, wide=False)


E10_ROWS = [("cued_net", "Reference model"), ("cued_resnet50", "ResNet-50"), ("cued_effnet_b0", "EfficientNet-B0"),
            ("cued_no_consistency", "No consistency term"), ("cued_vdl", "View discordance loss")]


def e10(r, suffix=""):
    """Robustness arms from the output of the e10 analysis: one row per variant of CUED-Net."""
    rows = []
    for key, name in E10_ROWS:
        s, v = r["summary"][key], r["view_term"][key]["total_vs_within"]["aurc"]
        rows.append(f"{name} & {s['auc']['value']:.3f} & {s['f1']['value']:.3f} & {s['aurc']['value']:.3f} & "
                    f"{sg(v['value'])} ({p_text(v['p'])}) " + r"\\")
    return table(["Variant", "AUC", "F1", "AURC", "$\\Delta$AURC ($p$)"], rows,
                 "Robustness arms of CUED-Net, outer folds, five seeds. The reference model uses DenseNet-121 encoders. $\\Delta$AURC is the change when $U^{\\mathrm{vd}}$ is added to the within-view terms.",
                 "tab:e10" + suffix, wide=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--what", default="five-fold cross-validation")
    ap.add_argument("--layout", default="full", choices=["full", "compact", "cmmd", "e10"],
                    help="full: the three main tables. compact: one table with AUC and AURC. cmmd: zero-shot and adapted rows. e10: robustness arms")
    ap.add_argument("--adapted", default="", help="adapted CMMD analysis, for --layout cmmd, with --analysis as the zero-shot one")
    ap.add_argument("--suffix", default="", help="appended to every table label, so that two analyses can share a document")
    ap.add_argument("--no_view_term", action="store_true", help="leave the view-term table out of --layout full, when the text gives its rows")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    r = json.load(open(args.analysis))
    if args.layout == "compact":
        tables = [compact(r, args.what, args.suffix)]
    elif args.layout == "cmmd":
        tables = [cmmd(r, json.load(open(args.adapted)), args.what, args.suffix)]
    elif args.layout == "e10":
        tables = [e10(r, args.suffix)]
    else:
        tables = [classification(r, args.what, args.suffix), referral(r, args.what, args.suffix)]
        if not args.no_view_term:
            tables.append(view_term(r, args.what, args.suffix))
    Path(args.out).write_text("\n\n".join(tables) + "\n")


if __name__ == "__main__":
    main()
