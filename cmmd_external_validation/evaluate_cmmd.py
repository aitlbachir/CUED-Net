"""Evaluate CUED-Net on CMMD."""

import argparse, glob, json, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from tqdm import tqdm


def main(args):
    # -- make the CUED-Net package importable
    sys.path.insert(0, args.models_dir)
    sys.path.insert(0, str(Path(args.models_dir)))
    from models.cued_net import CUEDNet, CUEDNetEnsemble
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cmmd_pair_dataset import build_cmmd_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    loader = build_cmmd_loader(args.manifest, batch_size=args.batch_size,
                               num_workers=4,
                               path_prefix_override=args.path_prefix_override)

    # -- load the 5 CUED-Net models
    ckpts = sorted(glob.glob(args.ckpt_glob))
    assert ckpts, f"No checkpoints match {args.ckpt_glob}"
    print(f"Found {len(ckpts)} checkpoints:")
    models = []
    for cp in ckpts:
        m = CUEDNet(num_classes=2, pretrained=False).to(device)
        ck = torch.load(cp, map_location=device, weights_only=False)
        state = ck["model_state_dict"] if "model_state_dict" in ck else ck
        m.load_state_dict(state, strict=True)
        m.eval()
        models.append(m)
        va = ck.get("val_auc", ck.get("auc", "—"))
        print(f"  {cp}   (val_auc={va})")

    ensemble = CUEDNetEnsemble(models)

    # -- inference
    P, Uev, Uens, Udis, Utot, Y = [], [], [], [], [], []
    meta_all = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="inference"):
            out = ensemble.predict(batch["img_cc"], batch["img_mlo"], device)
            P.append(out["prob"][:, 1].numpy())            # P(malignant)
            Uev.append(out["uncertainty_evidential"].numpy())
            Uens.append(out["uncertainty_ensemble"].numpy())
            Udis.append(out["uncertainty_discordance"].numpy())
            Utot.append(out["uncertainty_total"].numpy())
            Y.append(batch["label"].numpy())
            meta_all.extend(batch["meta"])

    p     = np.concatenate(P)
    u_ev  = np.concatenate(Uev)
    u_ens = np.concatenate(Uens)
    u_dis = np.concatenate(Udis)
    u_tot = np.concatenate(Utot)
    y     = np.concatenate(Y).astype(int)

    # -- metrics helpers
    def clf(y_true, y_score, thr):
        pred = (y_score >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        s = lambda n, d: round(n / d, 4) if d > 0 else float("nan")
        return dict(threshold=round(float(thr), 4),
                    sensitivity=s(tp, tp+fn), specificity=s(tn, tn+fp),
                    PPV=s(tp, tp+fp), NPV=s(tn, tn+fn),
                    accuracy=s(tp+tn, tp+tn+fp+fn))

    def boot_ci(y_true, y_score, n=1000, seed=42):
        rng = np.random.default_rng(seed); a = []
        for _ in range(n):
            idx = rng.integers(0, len(y_true), len(y_true))
            if len(np.unique(y_true[idx])) < 2:
                continue
            a.append(roc_auc_score(y_true[idx], y_score[idx]))
        a = np.array(a)
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    def selective(y_true, y_score, u, cov):
        k = max(1, int(len(y_true) * cov))
        idx = np.argsort(u)[:k]
        yt, ys = y_true[idx], y_score[idx]
        auc = roc_auc_score(yt, ys) if len(np.unique(yt)) > 1 else float("nan")
        return {"coverage": cov, "n_kept": int(k),
                "auc": round(float(auc), 4), **clf(yt, ys, 0.5)}

    auc = roc_auc_score(y, p)
    ci_lo, ci_hi = boot_ci(y, p)

    # Youden-optimal threshold
    fpr, tpr, thr = roc_curve(y, p)
    youden = thr[np.argmax(tpr - fpr)]

    pred05 = (p >= 0.5).astype(int)
    correct = (pred05 == y)

    report = {
        "cohort": {"n_pairs": int(len(y)), "n_malignant": int(y.sum()),
                   "n_benign": int((1-y).sum()),
                   "malignant_ratio": round(float(y.mean()), 4)},
        "ensemble": {"n_members": len(models),
                     "checkpoints": [Path(c).parent.name for c in ckpts]},
        "discrimination": {
            "auc": round(float(auc), 4),
            "auc_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        },
        "operating_points": {
            "threshold_0.5":   clf(y, p, 0.5),
            "youden_optimal":  clf(y, p, youden),
        },
        "selective_prediction": {
            "coverage_100": {"coverage": 1.0, "auc": round(float(auc), 4),
                             **clf(y, p, 0.5)},
            "coverage_70":  selective(y, p, u_tot, 0.70),
            "coverage_50":  selective(y, p, u_tot, 0.50),
        },
        "uncertainty_decomposition": {
            "mean_evidential":  round(float(u_ev.mean()), 4),
            "mean_ensemble":    round(float(u_ens.mean()), 4),
            "mean_discordance": round(float(u_dis.mean()), 4),
            "mean_total":       round(float(u_tot.mean()), 4),
            "total_correct":    round(float(u_tot[correct].mean()), 4),
            "total_incorrect":  round(float(u_tot[~correct].mean()), 4),
        },
    }

    # subtype-stratified
    subt = np.array([m["subtype"] for m in meta_all])
    strat = {}
    for st in set(subt):
        if st is None:
            continue
        mask = subt == st
        if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
            continue
        strat[str(st)] = {"n": int(mask.sum()),
                          "auc": round(float(roc_auc_score(y[mask], p[mask])), 4),
                          **clf(y[mask], p[mask], 0.5)}
    if strat:
        report["subtype_stratified"] = strat

    print("\n" + "="*64)
    print(json.dumps(report, indent=2))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",   default="/workspace/cued_net/cmmd_pairs_full.json")
    ap.add_argument("--ckpt_glob",  default="/workspace/cued_net/outputs_cued/seed_*/best_model.pt")
    ap.add_argument("--models_dir", default="/workspace/cued_net",
                    help="Directory containing models/cued_net.py")
    ap.add_argument("--output",     default="/workspace/cued_net/cmmd_results_cuednet.json")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--path_prefix_override", default=None)
    main(ap.parse_args())
