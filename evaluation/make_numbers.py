"""Write the numbers quoted in the text as LaTeX macros, from one p2_analysis.py output.

Each arm gets a prefix: DN for DenseNet-121 and CX for ConvNeXt-T. A second analysis, for
example the held-out split or a CMMD summary, can use another prefix through --tag, so that
text and tables read every number from the same files. An arm writes only the comparisons
its analysis holds, so the adapted CMMD summaries (CUED-Net and TMC) need no separate code.
"""

import argparse
import json
from pathlib import Path

ARMS = {"densenet121": "DN", "convnext_tiny": "CX"}
SHORT = {"tmc": "Tmc", "deep_ensemble": "De", "mc_dropout": "Mc", "cross_view": "Cv", "sv_edl": "Sv",
         "tmc_entropy": "TmcH", "sv_edl_entropy": "SvH"}
METRICS = {"auc": "Auc", "f1": "Fone", "aurc": "Aurc", "eaurc": "Eaurc",
           "f1_inner_threshold": "FoneInner", "sens_at_90spec": "SensAtSpec", "brier_cal": "BrierCal", "nll_cal": "NllCal"}
# Operating points and calibration of the reference model, beyond those of the main tables.
EXTRA = {"ppv": "Ppv", "npv": "Npv", "fnr": "Fnr", "sens_at_90spec": "SensAtSpec", "spec_at_90spec": "SpecAtSpec",
         "f1_inner_threshold": "FoneInner", "brier": "Brier", "brier_cal": "BrierCal", "nll": "Nll", "nll_cal": "NllCal",
         "ece_mass": "EceMass", "ece_mass_cal": "EceMassCal", "mce": "Mce", "mce_cal": "MceCal"}
DEPENDENCE = {"spearman": "Rho", "distance_correlation": "Dcor", "mutual_information": "Mi"}
NET_BENEFIT = {0.10: "Ten", 0.30: "Thirty", 0.50: "Fifty"}
# The arm trained with the regulariser of the first revision, all else as the DenseNet-121 reference model.
EARLIER_LOSS = "cued_r1_kl"
# CUED-Net with ConvNeXt-T encoders at evidence scale 1, the setting that the selection rule did not keep.
SCALE_ONE = "cued_convnext"


def num(x, d=3, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$")


def ci(v, d=3, sign=False):
    lo, hi = v["ci95"]
    return f"{num(v['value'] if 'value' in v else v['difference'], d, sign)} [{num(lo, d, sign)}, {num(hi, d, sign)}]"


def pval(p, holm=False):
    name = "Holm $p" if holm else "$p"
    return f"{name}<0.001$" if p < 0.001 else f"{name}={p:.3f}$"


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def short(comparator):
    for suffix in ("_convnext_entropy", "_convnext"):
        if comparator.endswith(suffix):
            comparator = comparator[: -len(suffix)] + ("_entropy" if "entropy" in suffix else "")
    return SHORT[comparator]


def macros(r, tag):
    out = {}
    for arm, prefix in ARMS.items():
        a, p = r["arms"][arm], tag + prefix
        s = r["summary"][a["reference"]]
        for key, name in (("auc", "Auc"), ("f1", "Fone"), ("sensitivity", "Sens"), ("specificity", "Spec"), ("ap", "Ap"),
                          ("ece", "Ece"), ("ece_cal", "EceCal"), ("aurc", "Aurc"), ("eaurc", "Eaurc"), ("error_auroc", "ErrAuroc")):
            out[p + name] = ci(s[key])
        for key, name in EXTRA.items():
            out[p + name] = ci(s[key])
        out[p + "AucPt"] = num(s["auc"]["value"])
        out[p + "Acc"] = num(s["accuracy"]["value"])
        # Extremes over the comparator scores of this arm, for statements that concern every method.
        comps = list(a["comparison"].get("aurc", {}))
        if comps:
            out[p + "CmpAccMax"] = num(max(r["summary"][c]["accuracy"]["value"] for c in comps))
            err = [r["summary"][c]["error_auroc"]["value"] for c in comps]
            out[p + "CmpErrAurocMin"], out[p + "CmpErrAurocMax"] = num(min(err)), num(max(err))
        for c in (70, 80, 90):
            word = {70: "Seventy", 80: "Eighty", 90: "Ninety"}[c]
            out[p + "SensAcc" + word] = num(s[f"sens_accepted_{c}"]["value"])
            out[p + "SensRef" + word] = num(s[f"sens_referred_{c}"]["value"])
            out[p + "MalRef" + word] = pct(s[f"malignant_referred_{c}"]["value"])
            out[p + "ErrCap" + word] = pct(s[f"errors_captured_{c}"]["value"])
        t = r["temperature"][a["reference"]]
        out[p + "Temp"] = f"{t['mean']:.2f}"
        for metric, mname in METRICS.items():
            for comp, d in a["comparison"].get(metric, {}).items():
                out[p + mname + short(comp)] = f"{ci(d, sign=True)}, {pval(d['p_holm'], holm=True)}"
        for key, name in (("total_vs_within", "ViewWithin"), ("amb_plus_view_vs_amb", "ViewAmb"), ("evidential_plus_view_vs_evidential", "ViewEvid")):
            v = a["view_term"][key]["aurc"]
            out[p + name] = f"{ci(v, sign=True)}, {pval(v['p'])}"
            out[p + name + "P"] = pval(v["p"])
        f = a["prespecified_fitted_score"]["added_value"]
        out[p + "Fitted"] = f"{ci(f['delta_aurc'], sign=True)}, {pval(f['delta_aurc']['p'])}"
        out[p + "FittedAurc"] = num(f["point"]["aurc_m1"])
        dep = a["term_dependence"]
        for key, name in (("u_amb_vs_u_ev", "AmbEv"), ("u_amb_vs_u_vd", "AmbVd"), ("u_ev_vs_u_vd", "EvVd")):
            for stat, sname in DEPENDENCE.items():
                out[p + sname + name] = num(dep[key][stat]["value"], 2)
        for key, name in (("u_amb", "MeanAmb"), ("u_ev", "MeanEv"), ("u_vd", "MeanVd")):
            out[p + name] = num(a["term_means"][key])
        fus = a["fusion_ablation"]
        for rule, name in (("equal", "Equal"), ("max_evidence", "MaxEv"), ("cc", "Cc"), ("mlo", "Mlo")):
            out[p + "Fusion" + name + "Auc"] = num(fus["summary"][rule]["auc"]["value"])
            out[p + "Fusion" + name + "Aurc"] = num(fus["summary"][rule]["aurc"]["value"])
            for metric, mname in (("auc", "Auc"), ("aurc", "Aurc")):
                d = fus["against_evidence_weighted"].get(metric, {}).get(rule)
                if d:
                    out[p + "FusionVs" + name + mname] = f"{ci(d, sign=True)}, {pval(d['p_holm'], holm=True)}"
        vc = a["view_conflict"]
        out[p + "ConflictShare"], out[p + "ConflictErr"], out[p + "AgreeErr"] = pct(vc["share"]), pct(vc["error_conflict"]), pct(vc["error_agreement"])
        e = a["error_by_tercile"]
        out[p + "ErrLowLow"], out[p + "ErrLowHigh"], out[p + "ErrHighHigh"] = pct(e[0][0]), pct(e[0][2]), pct(e[2][2])
        # Net benefit of the reference model on its calibrated, seed-averaged probabilities, against acting on every case.
        curve = {round(row["threshold"], 2): row for row in r.get("decision_curve", {}).get(a["reference"], [])}
        for t, word in NET_BENEFIT.items():
            if t in curve:
                out[p + "Nb" + word], out[p + "NbAll" + word] = num(curve[t]["net_benefit"]), num(curve[t]["treat_all"])
        # The sensitivity analysis exists only when p2_analysis.py ran with --multi_mass (cross-validation).
        if "excluding_multi_mass" in r:
            x = r["excluding_multi_mass"][a["reference"]]
            out[p + "NoMultiAuc"], out[p + "NoMultiFone"], out[p + "NoMultiAurc"] = num(x["auc"]), num(x["f1"]), num(x["aurc"])
    if SCALE_ONE in r["summary"]:
        s = r["summary"][SCALE_ONE]
        for key, name in (("auc", "Auc"), ("f1", "Fone"), ("aurc", "Aurc"), ("nll", "Nll"), ("ece_cal", "EceCal")):
            out[tag + "CXScaleOne" + name] = ci(s[key])
    if EARLIER_LOSS in r["summary"]:
        s = r["summary"][EARLIER_LOSS]
        for key, name in (("auc", "Auc"), ("f1", "Fone"), ("aurc", "Aurc"), ("ece_cal", "EceCal")):
            out[tag + "EarlierLoss" + name] = ci(s[key])
    out[tag + "Cases"], out[tag + "Patients"], out[tag + "B"] = (f"{r[k]:,}".replace(",", "{,}") for k in ("n_cases", "n_patients", "B"))
    return out


REFERENCE = {"cued_net": "DN", "cued_convnext_k5": "CX"}
CONDITIONS = {"swap_opposite": "SwapOpp", "swap_same": "SwapSame", "blur_1p5": "BlurA", "blur_3": "BlurB", "blur_6": "BlurC",
              "occlude_25": "OccA", "occlude_50": "OccB"}
TERMS = {"u_amb": "Amb", "u_ev": "Ev", "u_vd": "Vd", "u_total": "Total", "share_evid": "ShareEv", "share_vd": "ShareVd"}


def anchor_macros(r, tag):
    """Radiologist annotation checks from anchor_p2.py."""
    out = {}
    for method, prefix in REFERENCE.items():
        if method not in r:
            continue
        t, p = r[method]["tests"], tag + prefix + "Ann"
        d = t["described_differently"]
        out[p + "Flagged"] = str(d["n_flagged"])
        out[p + "Diff"] = f"{num(d['auc'], 2)}, {pval(d['p_holm'], holm=True)}"
        # Short cells for narrow tables: statistic with the Holm-adjusted p value in parentheses.
        cell = lambda v, ph: f"{num(v, 2)} ({'$<$0.001' if ph < 0.001 else f'{ph:.3f}'})"
        out[p + "DiffCell"] = cell(d["auc"], d["p_holm"])
        for key, name in (("subtlety_gap", "Gap"), ("entropy_with_subtlety", "Entropy"), ("evidential_share_with_subtlety", "Share")):
            out[p + name + "Cell"] = cell(t[key]["rho"], t[key]["p_holm"])
        for key, name in (("subtlety_gap", "Gap"), ("entropy_with_subtlety", "Entropy"), ("evidential_share_with_subtlety", "Share")):
            out[p + name] = f"{num(t[key]['rho'], 2)}, {pval(t[key]['p_holm'], holm=True)}"
    return out


def cost_macros(r, tag):
    """Latency per pair and parameters from cost_p2.py, CUED-Net against the sampling-based methods."""
    out = {}
    for suffix, prefix in (("", "DN"), ("_convnext", "CX")):
        p = tag + prefix + "Cost"
        cued = r["cued_net" if not suffix else "cued_convnext"]
        out[p + "Cued"] = f"{cued['latency']['mean_ms']:.1f}"
        out[p + "CuedParams"] = f"{cued['params_m']:.1f}"
        out[p + "De"] = f"{r['deep_ensemble' + suffix]['latency']['mean_ms']:.0f}"
        out[p + "Mc"] = f"{r['mc_dropout' + suffix]['latency']['mean_ms']:,.0f}".replace(",", "{,}")
        out[p + "Tmc"] = f"{r['tmc' + suffix]['latency']['mean_ms']:.1f}"
    out[tag + "CostGpu"] = r["gpu"]
    return out


def response_macros(r, tag):
    """Controlled changes from term_response_p2.py --summary."""
    out = {}
    for method, prefix in REFERENCE.items():
        if method not in r:
            continue
        p = tag + prefix + "Resp"
        for term, name in TERMS.items():
            out[p + "Clean" + name] = num(r[method]["clean"][term])
        for cond, cname in CONDITIONS.items():
            for term, name in TERMS.items():
                v = r[method][cond][term]
                out[p + cname + name] = ci(dict(value=v["change"], ci95=v["ci95"]), sign=True)
    return out


SALIENCY = {"all": "All", "correct": "Correct", "error": "Error", "tercile_0": "TercLow", "tercile_1": "TercMid", "tercile_2": "TercHigh",
            "view_cc": "Cc", "view_mlo": "Mlo"}


def saliency_macros(r, tag):
    """Grad-CAM against lesion masks from gradcam_p2.py: energy inside the mask, its ratio to the mask area, peak inside the mask."""
    out = {tag + "SalRows": f"{r['n_rows']:,}".replace(",", "{,}")}
    for group, name in SALIENCY.items():
        s = r["summary"][group]
        for metric, mname in (("energy", "Energy"), ("lift", "Lift"), ("hit", "Hit")):
            out[tag + "Sal" + name + mname] = ci(s[metric])
        out[tag + "Sal" + name + "Area"] = num(s["area"]["value"])
    return out


def reliability_macros(r, tag):
    """Mean confidence of the predicted class before and after temperature scaling, and accuracy, from reliability_p2.py."""
    out = {}
    for method, prefix in REFERENCE.items():
        if method in r["methods"]:
            m, p = r["methods"][method], tag + prefix + "Rel"
            out[p + "Conf"], out[p + "ConfCal"], out[p + "Acc"] = num(m["raw"]["mean_confidence"]), num(m["calibrated"]["mean_confidence"]), num(m["raw"]["accuracy"])
    return out


def seed_macros(r, tag):
    """Range over seeds of the reference models, from seed_table_p2.py."""
    out = {}
    for method, prefix in REFERENCE.items():
        if method in r:
            for metric, name in (("auc", "Auc"), ("f1", "Fone"), ("aurc", "Aurc")):
                out[tag + prefix + "Seed" + name + "Min"], out[tag + prefix + "Seed" + name + "Max"] = num(min(r[method][metric])), num(max(r[method][metric]))
            w = r[method]["weight_cc"]
            out[tag + prefix + "WeightLow"], out[tag + prefix + "WeightHigh"] = num(w["p01"], 2), num(w["p99"], 2)
    return out


def scale_view_macros(r, tag):
    """View-term tests of the ConvNeXt-T model at evidence scale 1, from view_term_p2.py."""
    assert r["method"] == SCALE_ONE
    out = {}
    for key, name in (("total_vs_within", "ViewWithin"), ("amb_plus_view_vs_amb", "ViewAmb"), ("evidential_plus_view_vs_evidential", "ViewEvid")):
        v = r["view_term"][key]["aurc"]
        out[tag + "CXScaleOne" + name] = f"{ci(v, sign=True)}, {pval(v['p'])}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--tag", default="", help="letters prefixed to every macro name, e.g. Test")
    ap.add_argument("--anchor", default="", help="anchor_p2.py output")
    ap.add_argument("--response", default="", help="term_response_p2.py --summary output")
    ap.add_argument("--cost", default="", help="cost_p2.py output")
    ap.add_argument("--saliency", default="", help="gradcam_p2.py summary")
    ap.add_argument("--reliability", default="", help="reliability_p2.py output")
    ap.add_argument("--seeds", default="", help="seed_table_p2.py output")
    ap.add_argument("--scale_view", default="", help="view_term_p2.py output for the ConvNeXt-T model at evidence scale 1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.tag.isalpha() or not args.tag
    m = macros(json.load(open(args.analysis)), args.tag)
    if args.anchor:
        m.update(anchor_macros(json.load(open(args.anchor)), args.tag))
    if args.response:
        m.update(response_macros(json.load(open(args.response)), args.tag))
    if args.cost:
        m.update(cost_macros(json.load(open(args.cost)), args.tag))
    if args.saliency:
        m.update(saliency_macros(json.load(open(args.saliency)), args.tag))
    if args.reliability:
        m.update(reliability_macros(json.load(open(args.reliability)), args.tag))
    if args.seeds:
        m.update(seed_macros(json.load(open(args.seeds)), args.tag))
    if args.scale_view:
        m.update(scale_view_macros(json.load(open(args.scale_view)), args.tag))
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    Path(args.out).write_text("% Written by evaluation/make_numbers.py. Do not edit by hand.\n" + "\n".join(lines) + "\n")
    print(len(lines), "macros")


if __name__ == "__main__":
    main()
