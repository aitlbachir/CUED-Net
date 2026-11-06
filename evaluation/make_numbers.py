"""Write the numbers quoted in the text as LaTeX macros, from one p2_analysis.py output.

Each arm gets a prefix: DN for DenseNet-121 and CX for ConvNeXt-T. A second analysis, for
example the held-out split, can use another prefix through --tag, so that text and tables
read every number from the same files.
"""

import argparse
import json
from pathlib import Path

ARMS = {"densenet121": "DN", "convnext_tiny": "CX"}
SHORT = {"tmc": "Tmc", "deep_ensemble": "De", "mc_dropout": "Mc", "cross_view": "Cv", "sv_edl": "Sv",
         "tmc_entropy": "TmcH", "sv_edl_entropy": "SvH"}
METRICS = {"auc": "Auc", "f1": "Fone", "aurc": "Aurc", "eaurc": "Eaurc"}


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
        f = a["prespecified_fitted_score"]["added_value"]
        out[p + "Fitted"] = f"{ci(f['delta_aurc'], sign=True)}, {pval(f['delta_aurc']['p'])}"
        out[p + "FittedAurc"] = num(f["point"]["aurc_m1"])
        dep = a["term_dependence"]
        for key, name in (("u_amb_vs_u_ev", "RhoAmbEv"), ("u_amb_vs_u_vd", "RhoAmbVd"), ("u_ev_vs_u_vd", "RhoEvVd")):
            out[p + name] = num(dep[key]["spearman"]["value"], 2)
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
        x = r["excluding_multi_mass"][a["reference"]]
        out[p + "NoMultiAuc"], out[p + "NoMultiFone"], out[p + "NoMultiAurc"] = num(x["auc"]), num(x["f1"]), num(x["aurc"])
    out[tag + "Cases"], out[tag + "Patients"], out[tag + "B"] = str(r["n_cases"]), str(r["n_patients"]), f"{r['B']:,}".replace(",", "{,}")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--tag", default="", help="letters prefixed to every macro name, e.g. Test")
    ap.add_argument("--anchor", default="", help="anchor_p2.py output")
    ap.add_argument("--response", default="", help="term_response_p2.py --summary output")
    ap.add_argument("--cost", default="", help="cost_p2.py output")
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
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    Path(args.out).write_text("% Written by evaluation/make_numbers.py. Do not edit by hand.\n" + "\n".join(lines) + "\n")
    print(len(lines), "macros")


if __name__ == "__main__":
    main()
