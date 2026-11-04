#!/usr/bin/env python
"""Every statistic in the paper, computed from the per-case files of Protocol P2.

Unit of inference: the lesion pair, resampled by patient. A single-model metric
is computed per seed inside each bootstrap sample and then averaged, so it
estimates single-model performance and repeated seeds add no sample size.
Deep-Ensemble averages the seed models of a fold and gives one prediction per case.

Choices (temperature, thresholds, score weights) use inner validation only.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata, spearmanr
from sklearn.feature_selection import mutual_info_regression

METHOD_LABELS = {
    "cued_net": "CUED-Net", "tmc": "TMC", "softmax_dual": "Softmax dual-view", "mc_dropout": "MC-Dropout",
    "deep_ensemble": "Deep-Ensemble", "sv_edl": "Single-view EDL", "cross_view": "Cross-view transformer",
    "cued_convnext": "CUED-Net ConvNeXt-T", "cued_r1_kl": "CUED-Net R1 loss",
}
REFERENCE = "cued_net"
COMPARATORS = ["tmc", "deep_ensemble", "mc_dropout", "cross_view", "sv_edl"]
EQUIVALENCE_MARGIN = 0.03
EPS = 1e-6


# ----------------------------------------------------------------- loading
def read_split(runs, method, split):
    files = sorted(Path(runs, method).glob(f"fold_*/seed_*/{split}.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def binary_entropy(p):
    """Predictive entropy, the uncertainty score of softmax comparators."""
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def nll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_temperature(y, p):
    z = logit(p)
    res = minimize_scalar(lambda lt: nll(y, sigmoid(z / np.exp(lt))), bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def threshold_at_specificity(y, p, target=0.90):
    neg = np.sort(p[y == 0])
    return float(neg[min(len(neg) - 1, int(np.ceil(target * len(neg))))])


def threshold_max_f1(y, p):
    grid = np.unique(np.round(p, 4))
    scores = [f1(y, p >= t) for t in grid]
    return float(grid[int(np.argmax(scores))])


def add_scores(df):
    """Uncertainty components for dual-view evidential outputs."""
    if "u_cc" in df:
        df["u_evid"] = 0.5 * (df.u_cc + df.u_mlo)
        df["u_disc"] = 1.0 - np.exp(-df.d_sym)
    return df


def fit_error_model(X, e, iters=30, ridge=1e-6):
    """Logistic regression by Newton steps. X is standardised by the caller."""
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sigmoid(X @ b)
        w = p * (1 - p) + 1e-9
        H = (X * w[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (e - p) - ridge * b)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


class Prepared:
    """Per-seed outer-fold arrays with every choice taken from inner validation."""

    def __init__(self, inner, outer, prob="p", unc=None):
        self.keys = sorted(outer.pair_key.unique())
        pos = {k: i for i, k in enumerate(self.keys)}
        first = outer.drop_duplicates("pair_key").set_index("pair_key")
        self.y = first.label.reindex(self.keys).values.astype(int)
        self.group = pd.factorize(first.patient_id.reindex(self.keys))[0]
        self.seeds = sorted(outer.seed.unique())
        n, m = len(self.keys), len(self.seeds)
        blank = lambda: np.full((m, n), np.nan)
        self.p, self.p_cal, self.t90, self.t_f1 = blank(), blank(), blank(), blank()
        self.u, self.u0, self.u_evid, self.u_disc = blank(), blank(), blank(), blank()
        self.ref = {c: blank() for c in (0.9, 0.8, 0.7)}
        self.temperatures, self.weights = [], []
        for si, seed in enumerate(self.seeds):
            for fold in sorted(outer.fold.unique()):
                iv = inner[(inner.seed == seed) & (inner.fold == fold)]
                ov = outer[(outer.seed == seed) & (outer.fold == fold)]
                idx = ov.pair_key.map(pos).values
                y_i, p_i = iv.label.values, iv[prob].values
                T = fit_temperature(y_i, p_i)
                self.temperatures.append(T)
                self.p[si, idx] = ov[prob].values
                self.p_cal[si, idx] = sigmoid(logit(ov[prob].values) / T)
                self.t90[si, idx] = threshold_at_specificity(y_i, p_i)
                self.t_f1[si, idx] = threshold_max_f1(y_i, p_i)
                u_o, u_i = self._score(iv, ov, unc, y_i, p_i, si, idx)
                for c, arr in self.ref.items():
                    arr[si, idx] = u_o > np.quantile(u_i, c)
                self.u[si, idx] = u_o

    def _score(self, iv, ov, unc, y_i, p_i, si, idx):
        if "u_evid" in iv and unc is None:
            e_i = ((p_i >= 0.5).astype(int) != y_i).astype(float)
            cols = ["u_evid", "u_disc"]
            mu, sd = iv[cols].mean().values, iv[cols].std().values + 1e-9
            Zi, Zo = (iv[cols].values - mu) / sd, (ov[cols].values - mu) / sd
            b1, b0 = fit_error_model(Zi, e_i), fit_error_model(Zi[:, :1], e_i)
            self.weights.append(b1.tolist())
            self.u0[si, idx] = np.column_stack([np.ones(len(Zo)), Zo[:, :1]]) @ b0
            self.u_evid[si, idx], self.u_disc[si, idx] = ov.u_evid.values, ov.u_disc.values
            xb = lambda Z: np.column_stack([np.ones(len(Z)), Z]) @ b1
            return xb(Zo), xb(Zi)
        if unc is None and "vacuity" not in ov:
            return binary_entropy(ov["p"].values), binary_entropy(iv["p"].values)
        c = unc or "vacuity"
        return ov[c].values, iv[c].values


def as_ensemble(prep, inner, outer):
    """Deep-Ensemble: the seed models of a fold, one prediction per case."""
    e = Prepared.__new__(Prepared)
    e.keys, e.y, e.group = prep.keys, prep.y, prep.group
    e.seeds = ["ensemble"]
    pos = {k: i for i, k in enumerate(prep.keys)}
    agg = lambda d: d.groupby(["fold", "pair_key", "label"], as_index=False).agg(p=("p", "mean"), var=("p", "var"))
    iv, ov = agg(inner), agg(outer)
    n = len(prep.keys)
    blank = lambda: np.full((1, n), np.nan)
    e.p, e.p_cal, e.t90, e.t_f1, e.u = blank(), blank(), blank(), blank(), blank()
    e.u0, e.u_evid, e.u_disc = blank(), blank(), blank()
    e.ref = {c: blank() for c in (0.9, 0.8, 0.7)}
    e.temperatures, e.weights = [], []
    for fold in sorted(ov.fold.unique()):
        i, o = iv[iv.fold == fold], ov[ov.fold == fold]
        idx = o.pair_key.map(pos).values
        T = fit_temperature(i.label.values, i.p.values)
        e.temperatures.append(T)
        e.p[0, idx] = o.p.values
        e.p_cal[0, idx] = sigmoid(logit(o.p.values) / T)
        e.t90[0, idx] = threshold_at_specificity(i.label.values, i.p.values)
        e.t_f1[0, idx] = threshold_max_f1(i.label.values, i.p.values)
        ent = lambda q: -(np.clip(q, EPS, 1) * np.log(np.clip(q, EPS, 1)) + np.clip(1 - q, EPS, 1) * np.log(np.clip(1 - q, EPS, 1)))
        e.u[0, idx] = ent(o.p.values)
        for c, arr in e.ref.items():
            arr[0, idx] = ent(o.p.values) > np.quantile(ent(i.p.values), c)
    return e


# ----------------------------------------------------------------- metrics
def auc(y, s):
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return (rankdata(s)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def average_precision(y, s):
    o = np.argsort(-s, kind="stable")
    hits = y[o]
    prec = np.cumsum(hits) / np.arange(1, len(y) + 1)
    return (prec * hits).sum() / max(1, hits.sum())


def f1(y, d):
    tp = np.sum(d & (y == 1))
    return 2 * tp / max(1, np.sum(d) + np.sum(y == 1))


def aurc(unc, err):
    o = np.argsort(unc, kind="stable")
    return np.mean(np.cumsum(err[o]) / np.arange(1, len(err) + 1))


def eaurc(unc, err):
    r = err.mean()
    n = len(err)
    optimal = np.sum(np.cumsum(np.sort(err)) / np.arange(1, n + 1)) / n
    return aurc(unc, err) - optimal


def ece(y, p, bins=15, equal_mass=False):
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    if equal_mass:
        chunks = np.array_split(np.argsort(conf), bins)
    else:
        e = np.linspace(0.5, 1.0, bins + 1)
        which = np.clip(np.digitize(conf, e[1:-1]), 0, bins - 1)
        chunks = [np.where(which == b)[0] for b in range(bins)]
    return sum(len(c) / len(y) * abs(correct[c].mean() - conf[c].mean()) for c in chunks if len(c))


def mce(y, p, bins=15):
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    e = np.linspace(0.5, 1.0, bins + 1)
    which = np.clip(np.digitize(conf, e[1:-1]), 0, bins - 1)
    gaps = [abs(correct[which == b].mean() - conf[which == b].mean()) for b in range(bins) if (which == b).any()]
    return max(gaps)


def operating(y, d):
    tp, fn = np.sum(d & (y == 1)), np.sum(~d & (y == 1))
    tn, fp = np.sum(~d & (y == 0)), np.sum(d & (y == 0))
    sens = tp / max(1, tp + fn)
    return dict(sensitivity=sens, specificity=tn / max(1, tn + fp), ppv=tp / max(1, tp + fp),
                npv=tn / max(1, tn + fn), fnr=1 - sens, accuracy=(tp + tn) / len(y))


def all_metrics(m, idx):
    """Metrics for one bootstrap sample, averaged across seeds."""
    y = m.y[idx]
    rows = defaultdict(list)
    for s in range(len(m.seeds)):
        p, cal = m.p[s, idx], m.p_cal[s, idx]
        keep = ~np.isnan(p)
        yy, p, cal = y[keep], p[keep], cal[keep]
        d = p >= 0.5
        err = (d.astype(int) != yy).astype(float)
        u = m.u[s, idx][keep]
        rows["auc"].append(auc(yy, p))
        rows["ap"].append(average_precision(yy, p))
        rows["f1"].append(f1(yy, d))
        for k, v in operating(yy, d).items():
            rows[k].append(v)
        rows["sens_at_90spec"].append(operating(yy, p >= m.t90[s, idx][keep])["sensitivity"])
        rows["spec_at_90spec"].append(operating(yy, p >= m.t90[s, idx][keep])["specificity"])
        rows["f1_inner_threshold"].append(f1(yy, p >= m.t_f1[s, idx][keep]))
        rows["aurc"].append(aurc(u, err))
        rows["eaurc"].append(eaurc(u, err))
        rows["error_auroc"].append(auc(err, u) if 0 < err.sum() < len(err) else np.nan)
        rows["brier"].append(np.mean((p - yy) ** 2))
        rows["brier_cal"].append(np.mean((cal - yy) ** 2))
        rows["nll"].append(nll(yy, p))
        rows["nll_cal"].append(nll(yy, cal))
        rows["ece"].append(ece(yy, p))
        rows["ece_cal"].append(ece(yy, cal))
        rows["ece_mass"].append(ece(yy, p, equal_mass=True))
        rows["ece_mass_cal"].append(ece(yy, cal, equal_mass=True))
        rows["mce"].append(mce(yy, p))
        rows["mce_cal"].append(mce(yy, cal))
        for c, flag in m.ref.items():
            refer = flag[s, idx][keep] > 0
            acc = ~refer
            tag = int(c * 100)
            rows[f"coverage_{tag}"].append(acc.mean())
            mal = yy == 1
            rows[f"sens_accepted_{tag}"].append((d & acc & mal).sum() / max(1, (acc & mal).sum()))
            rows[f"malignant_referred_{tag}"].append((refer & mal).sum() / max(1, mal.sum()))
            rows[f"errors_captured_{tag}"].append((refer & (err > 0)).sum() / max(1, (err > 0).sum()))
            rows[f"accuracy_accepted_{tag}"].append(1 - err[acc].mean() if acc.any() else np.nan)
    return {k: float(np.nanmean(v)) for k, v in rows.items()}


def patient_samples(group, B, seed=20261103):
    rng = np.random.default_rng(seed)
    members = defaultdict(list)
    for i, g in enumerate(group):
        members[g].append(i)
    ids = sorted(members)
    members = [np.array(members[g]) for g in ids]
    for _ in range(B):
        yield np.concatenate([members[j] for j in rng.integers(0, len(ids), len(ids))])


def holm(pvals):
    names = sorted(pvals, key=pvals.get)
    adj, running = {}, 0.0
    for i, k in enumerate(names):
        running = max(running, min(1.0, (len(names) - i) * pvals[k]))
        adj[k] = running
    return adj


def bootstrap_all(models, group, B):
    names = list(models)
    full = np.arange(len(group))
    point = {n: all_metrics(models[n], full) for n in names}
    keys = list(point[names[0]])
    dist = {n: {k: np.empty(B) for k in keys} for n in names}
    for b, idx in enumerate(patient_samples(group, B)):
        for n in names:
            r = all_metrics(models[n], idx)
            for k in keys:
                dist[n][k][b] = r[k]
    return point, dist


def summarise(point, dist):
    out = {}
    for n, metrics in point.items():
        out[n] = {}
        for k, v in metrics.items():
            lo, hi = np.nanpercentile(dist[n][k], [2.5, 97.5])
            # Binned calibration errors are biased upward under resampling.
            shift = np.nanmean(dist[n][k]) - v if k.startswith(("ece", "mce")) else 0.0
            out[n][k] = dict(value=v, ci95=[float(lo - shift), float(hi - shift)])
    return out


def compare(point, dist, reference, others, metrics=("auc", "f1", "ap", "aurc", "eaurc", "sensitivity", "specificity", "f1_inner_threshold", "sens_at_90spec", "brier_cal", "nll_cal")):
    out = {}
    for metric in metrics:
        raw, rows = {}, {}
        for o in others:
            d = dist[reference][metric] - dist[o][metric]
            d = d[~np.isnan(d)]
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            raw[o] = max(p, 1 / (len(d) + 1))
            ci90 = np.percentile(d, [5, 95]).tolist()
            rows[o] = dict(difference=point[reference][metric] - point[o][metric], ci95=np.percentile(d, [2.5, 97.5]).tolist(),
                           ci90=ci90, p=raw[o])
            if metric == "auc":
                rows[o]["within_margin"] = bool(ci90[0] > -EQUIVALENCE_MARGIN and ci90[1] < EQUIVALENCE_MARGIN)
        adj = holm(raw)
        for o in others:
            rows[o]["p_holm"] = adj[o]
        out[metric] = rows
    return out


# ----------------------------------------------------------------- added value of discordance
def added_value(m, B):
    """M0 uses evidential uncertainty. M1 adds view discordance."""
    pts = np.arange(len(m.y))
    def stats(idx):
        r = defaultdict(list)
        for s in range(len(m.seeds)):
            e = ((m.p[s, idx] >= 0.5).astype(int) != m.y[idx]).astype(float)
            if 0 < e.sum() < len(e):
                r["auroc_m0"].append(auc(e, m.u0[s, idx]))
                r["auroc_m1"].append(auc(e, m.u[s, idx]))
                r["aurc_m0"].append(aurc(m.u0[s, idx], e))
                r["aurc_m1"].append(aurc(m.u[s, idx], e))
        return {k: float(np.mean(v)) for k, v in r.items()}
    base = stats(pts)
    draws = pd.DataFrame([stats(i) for i in patient_samples(m.group, B)])
    d_auc, d_aurc = draws.auroc_m1 - draws.auroc_m0, draws.aurc_m1 - draws.aurc_m0
    out = dict(point=base, delta_auroc=dict(value=base["auroc_m1"] - base["auroc_m0"], ci95=np.percentile(d_auc, [2.5, 97.5]).tolist(),
                                            p=float(min(1, 2 * min((d_auc <= 0).mean(), (d_auc >= 0).mean())))),
               delta_aurc=dict(value=base["aurc_m1"] - base["aurc_m0"], ci95=np.percentile(d_aurc, [2.5, 97.5]).tolist(),
                               p=float(min(1, 2 * min((d_aurc <= 0).mean(), (d_aurc >= 0).mean())))))
    # Error rate by tercile, seed by seed, then averaged.
    tables = []
    for s in range(len(m.seeds)):
        ev, di = m.u_evid[s], m.u_disc[s]
        e = ((m.p[s] >= 0.5).astype(int) != m.y).astype(float)
        a = np.digitize(ev, np.quantile(ev, [1 / 3, 2 / 3]))
        b = np.digitize(di, np.quantile(di, [1 / 3, 2 / 3]))
        tables.append([[float(e[(a == i) & (b == j)].mean()) for j in range(3)] for i in range(3)])
    out["error_rate_by_tercile"] = np.mean(tables, axis=0).tolist()
    out["weights_mean"] = np.mean(m.weights, axis=0).tolist()
    out["weights_std"] = np.std(m.weights, axis=0).tolist()
    return out


def distance_correlation(x, y):
    a, b = np.abs(x[:, None] - x), np.abs(y[:, None] - y)
    A = a - a.mean(0) - a.mean(1, keepdims=True) + a.mean()
    B = b - b.mean(0) - b.mean(1, keepdims=True) + b.mean()
    return float(np.sqrt(max((A * B).mean(), 0) / np.sqrt((A * A).mean() * (B * B).mean())))


def dependence(m, ens_var=None, perms=999, seed=7):
    rng = np.random.default_rng(seed)
    pairs = [("evidential", m.u_evid.mean(0), "discordance", m.u_disc.mean(0))]
    if ens_var is not None:
        pairs += [("evidential", m.u_evid.mean(0), "ensemble", ens_var), ("discordance", m.u_disc.mean(0), "ensemble", ens_var)]
    out = {}
    groups = np.unique(m.group)
    for na, x, nb, y in pairs:
        obs = dict(spearman=float(spearmanr(x, y)[0]), distance_correlation=distance_correlation(x, y),
                   mutual_information=float(mutual_info_regression(x[:, None], y, random_state=0)[0]))
        null = defaultdict(list)
        for _ in range(perms):
            perm = rng.permutation(len(y))
            null["spearman"].append(abs(spearmanr(x, y[perm])[0]))
            null["distance_correlation"].append(distance_correlation(x, y[perm]))
            null["mutual_information"].append(mutual_info_regression(x[:, None], y[perm], random_state=0)[0])
        out[f"{na}_vs_{nb}"] = {k: dict(value=v, p=float((1 + np.sum(np.array(null[k]) >= abs(v))) / (perms + 1))) for k, v in obs.items()}
    return out


def decision_curve(y, p, grid=np.arange(0.05, 0.51, 0.05)):
    rows = []
    for t in grid:
        d = p >= t
        tp, fp = np.sum(d & (y == 1)) / len(y), np.sum(d & (y == 0)) / len(y)
        rows.append(dict(threshold=float(t), net_benefit=float(tp - fp * t / (1 - t)),
                         treat_all=float(y.mean() - (1 - y.mean()) * t / (1 - t))))
    return rows


def weight_sensitivity(m, steps=21):
    """AURC over a grid of weights on standardised evidential and discordance scores."""
    out = []
    for w in np.linspace(0, 1, steps):
        vals = []
        for s in range(len(m.seeds)):
            e = ((m.p[s] >= 0.5).astype(int) != m.y).astype(float)
            z = lambda a: (a - a.mean()) / (a.std() + 1e-9)
            vals.append(aurc(w * z(m.u_evid[s]) + (1 - w) * z(m.u_disc[s]), e))
        out.append(dict(weight_evidential=float(w), aurc=float(np.mean(vals))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="outer_val", choices=["outer_val", "test", "cmmd", "cmmd_eval", "cmmd_mass"])
    ap.add_argument("--inner_fold", type=int, default=None, help="fold id that provides inner validation for --split test")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--B_added", type=int, default=2000)
    ap.add_argument("--multi_mass", default=None, help="json list of pair keys to exclude in the sensitivity analysis")
    args = ap.parse_args()

    tables = {m: (add_scores(read_split(args.runs, m, "inner_val")), add_scores(read_split(args.runs, m, args.split)))
              for m in ["cued_net", "tmc", "softmax_dual", "sv_edl", "cross_view", "cued_convnext", "cued_r1_kl"]
              if read_split(args.runs, m, args.split) is not None}
    if args.split != "outer_val":
        tables = {k: (i[i.fold == -1], o) for k, (i, o) in tables.items()}
    models = {}
    for name, (iv, ov) in tables.items():
        models[name] = Prepared(iv, ov)
        if name == "softmax_dual":
            models["mc_dropout"] = Prepared(iv, ov, prob="mc_p", unc="mc_entropy")
            models["deep_ensemble"] = as_ensemble(models[name], iv, ov)
    ref = models[REFERENCE]
    group = ref.group
    for k, v in models.items():
        assert v.keys == ref.keys or k == "sv_edl", k

    point, dist = bootstrap_all(models, group, args.B)
    others = [c for c in COMPARATORS if c in models]
    result = dict(split=args.split, n_cases=len(ref.keys), n_patients=int(group.max() + 1), B=args.B,
                  summary=summarise(point, dist), comparison=compare(point, dist, REFERENCE, others),
                  temperature={k: dict(mean=float(np.mean(v.temperatures)), std=float(np.std(v.temperatures))) for k, v in models.items()},
                  added_value=added_value(ref, args.B_added),
                  weight_sensitivity=weight_sensitivity(ref))
    ens_var = None
    if "softmax_dual" in models:
        ens_var = np.nanvar(models["softmax_dual"].p, axis=0, ddof=1)
    result["dependence"] = dependence(ref, ens_var)
    result["decision_curve"] = {k: decision_curve(models[k].y, np.nanmean(models[k].p_cal, axis=0)) for k in models}
    if args.multi_mass:
        drop = set(json.load(open(args.multi_mass)))
        keep = np.array([k not in drop for k in ref.keys])
        idx = np.where(keep)[0]
        result["excluding_multi_mass"] = {k: {x: all_metrics(v, idx)[x] for x in ("auc", "f1", "aurc")} for k, v in models.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    s = result["summary"]
    print(pd.DataFrame({k: {m: f"{v[m]['value']:.3f} [{v[m]['ci95'][0]:.3f}, {v[m]['ci95'][1]:.3f}]" for m in ("auc", "f1", "sensitivity", "specificity", "aurc", "ece", "ece_cal")} for k, v in s.items()}).T)


if __name__ == "__main__":
    main()
