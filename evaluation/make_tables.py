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


def classification(r, what):
    rows = []
    for arm, spec in ARM_ROWS.items():
        rows.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{ARM_NAMES[arm]}}}}} \\")
        for key, name in spec:
            if key in r["summary"]:
                s = r["summary"][key]
                rows.append(f"{name} & {ci(s['auc'])} & {ci(s['f1'])} & {ci(s['sensitivity'])} & {ci(s['specificity'])} & {ci(s['ece_cal'])} " + r"\\")
    return table(["Method", "AUC", "F1", "Sensitivity", "Specificity", "Calibrated ECE"], rows,
                 f"Classification and calibration, {what}, threshold 0.5. Brackets give 95\\% patient-clustered bootstrap intervals.", "tab:classification")


def referral(r, what):
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
                 f"Referral, {what}. Lower AURC and E-AURC are better.", "tab:referral")


def view_term(r, what):
    rows = []
    names = [("total_vs_within", "Added to both within-view terms"), ("amb_plus_view_vs_amb", "Added to $U^{\\mathrm{amb}}$ alone"),
             ("evidential_plus_view_vs_evidential", "Added to $U^{\\mathrm{evid}}$ alone")]
    for arm, a in r["arms"].items():
        rows.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{ARM_NAMES[arm]}}}}} \\")
        for key, name in names:
            v = a["view_term"][key]["aurc"]
            rows.append(f"{name} & {sg(v['value'])} [{sg(v['ci95'][0])}, {sg(v['ci95'][1])}] & {p_text(v['p'])} " + r"\\")
        f = a["prespecified_fitted_score"]["added_value"]["delta_aurc"]
        rows.append(f"Fitted score, pre-specified & {sg(f['value'])} [{sg(f['ci95'][0])}, {sg(f['ci95'][1])}] & {p_text(f['p'])} " + r"\\")
    return table(["Change in AURC when $U^{\\mathrm{vd}}$ is added", "Difference [95\\% CI]", "$p$"], rows,
                 f"Referral value of view discordance uncertainty, {what}. A negative difference favors adding the term.", "tab:viewterm", wide=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--what", default="five-fold cross-validation")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    r = json.load(open(args.analysis))
    Path(args.out).write_text("\n\n".join([classification(r, args.what), referral(r, args.what), view_term(r, args.what)]) + "\n")


if __name__ == "__main__":
    main()
